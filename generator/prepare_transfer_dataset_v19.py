"""Transfer-learning dataset builder, V19: four 3600x3600 quadrants, multi-city."""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

BASE_DIR = Path(__file__).resolve().parent.parent

GRAPH_DIR = Path(os.environ.get("GRAPH_V19_DIR", str(BASE_DIR / "graph_data")))
if not GRAPH_DIR.is_absolute():
    GRAPH_DIR = BASE_DIR / GRAPH_DIR
OUTPUT_DIR = BASE_DIR / "graph_data"  # Output: transfer_dataset_*.pt
ANTENNA_PATH = BASE_DIR / "data_raw" / "antenas_interior_sp_final.csv"
MAX_RADIUS_KM = 30.0
QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
STRUCT_DIR = None


# Keeps the same 30 km radius as Lins; reduces edges from N x M to N x K without changing terrain.y
NODE_MATRIX_CHUNK = 2_000
TOP_K_PER_NODE    = 5
NODE_CHUNK_SIZE   = NODE_MATRIX_CHUNK


CITY_ANTENNA_NAME = {
    "lins":     "LINS",
    "campinas": "CAMPINAS",
    "sorocaba": "SOROCABA",
    "bauru":    "BAURU",
}

sys.path.insert(0, str(BASE_DIR))
from generate_realistic_coverage import combined_propagation_loss


def load_antennas(city: str = "lins"):
    """Loads and preprocesses a city's antenna records."""
    df = pd.read_csv(ANTENNA_PATH, low_memory=False)
    csv_name = CITY_ANTENNA_NAME.get(city.lower(), city.upper())
    df_lins = df[df["Municipio.NomeMunicipio"].str.upper() == csv_name].copy()
    df_lins = df_lins.rename(columns={
        "Latitude": "lat", "Longitude": "lon",
        "AlturaAntena": "altura_antena", "FreqTxMHz": "freq_mhz",
        "PotenciaTransmissorWatts": "pot_watts", "GanhoAntena": "ganho_dbi",
    })
    for col in ["lat", "lon", "altura_antena", "freq_mhz", "pot_watts", "ganho_dbi"]:
        df_lins[col] = pd.to_numeric(df_lins[col], errors="coerce").fillna(0.0)
    df_lins = df_lins[(df_lins["freq_mhz"] > 0) & (df_lins["pot_watts"] > 0)].reset_index(drop=True)
    df_lins["pot_dbm"] = 10 * np.log10(df_lins["pot_watts"] * 1000)
    df_lins.loc[df_lins["pot_dbm"] < 0, "pot_dbm"] = 30.0
    df_lins.loc[df_lins["altura_antena"] <= 0, "altura_antena"] = 30.0
    df_lins["environment"] = df_lins["freq_mhz"].apply(
        lambda f: "urban" if f > 1800 else ("suburban" if f > 800 else "rural")
    )
    return df_lins


def build_transfer_one_quadrant(q_name: str, df_lins: pd.DataFrame,
                                city: str = "lins",
                                node_chunk_size: int = NODE_MATRIX_CHUNK,
                                max_radius_km: float | None = None,
                                top_k_per_node: int = TOP_K_PER_NODE) -> HeteroData:
    """
    Builds the transfer dataset for one quadrant.

    Node-centric approach with top-K antennas per node: instead of iterating
    by antenna and storing every edge (N x M edges, prohibitive for large
    cities), nodes are processed in chunks and the distance matrix (C x M)
    is computed at once per chunk; only the top-K antennas by RSSI within
    the search radius are kept per node.

    Matches the reference dataset construction in:
        - coverage radius (MAX_RADIUS_KM = 30 km);
        - path_loss, RSSI, and coverage computation (terrain.y, same formula);
        - HeteroData layout (same keys and dtypes).

    Differs from the antenna-centric approach in edge count: N x K edges
    instead of N x M, which normalizes graph density across cities with
    different antenna counts (more edges no longer implies more coverage).
    """
    emb_path = GRAPH_DIR / f"{city}_v19_{q_name}_embeddings_512.pt"
    struct_dir = STRUCT_DIR if STRUCT_DIR is not None else GRAPH_DIR
    terrain_path = struct_dir / f"{city}_v19_{q_name}_gpu.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings não encontrados: {emb_path}")
    if not terrain_path.exists():
        raise FileNotFoundError(f"Grafo não encontrado: {terrain_path}")

    emb_data = torch.load(emb_path, map_location="cpu", weights_only=False)
    embeddings = emb_data["embeddings"]
    del emb_data
    gc.collect()

    terrain_data = torch.load(terrain_path, map_location="cpu", weights_only=False)
    pos  = terrain_data["dem"].pos
    x_dem = terrain_data["dem"].x.numpy()
    del terrain_data
    gc.collect()

    n_nodes = pos.shape[0]
    ndvi_real    = x_dem[:, 12]
    slope_approx = np.random.uniform(0.02, 0.15, n_nodes).astype(np.float32)
    pos_np       = pos.numpy()

    radius_km = max_radius_km if max_radius_km is not None else MAX_RADIUS_KM
    radius_m  = float(radius_km) * 1000.0


    ant_lat    = df_lins["lat"].values.astype(np.float64)
    ant_lon    = df_lins["lon"].values.astype(np.float64)
    ant_freq   = df_lins["freq_mhz"].values.astype(np.float32)
    ant_power  = df_lins["pot_dbm"].values.astype(np.float32)
    ant_gain   = df_lins["ganho_dbi"].values.astype(np.float32)
    ant_height = df_lins["altura_antena"].values.astype(np.float32)
    ant_env    = df_lins["environment"].values
    n_ant = len(df_lins)
    k     = min(top_k_per_node, n_ant)

    ant_feats = torch.tensor(
        df_lins[["freq_mhz", "pot_dbm", "altura_antena", "ganho_dbi", "lat", "lon"]].values,
        dtype=torch.float32,
    )

    rssi_per_node     = np.full(n_nodes, -200.0, np.float32)
    pathloss_per_node = np.full(n_nodes,  300.0, np.float32)


    all_sources: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_ea:      list[np.ndarray] = []

    C       = node_chunk_size
    n_chunks = (n_nodes + C - 1) // C

    for chunk_idx, start in enumerate(range(0, n_nodes, C)):
        end = min(start + C, n_nodes)
        Ci  = end - start

        # Distance matrix (Ci x M), approx. 34 MB for Ci=2k, M=4252
        lat_c   = pos_np[start:end, 1:2].astype(np.float64)
        lon_c   = pos_np[start:end, 0:1].astype(np.float64)
        cos_lat = np.cos(np.radians(lat_c))

        dy       = (lat_c - ant_lat[np.newaxis, :]) * 111_000.0
        dx       = (lon_c - ant_lon[np.newaxis, :]) * 111_000.0 * cos_lat
        dist_mat = np.sqrt(dx**2 + dy**2).astype(np.float32)
        del dx, dy, cos_lat, lat_c, lon_c

        within_mat = dist_mat <= radius_m

        if not within_mat.any():
            del dist_mat, within_mat
            gc.collect()
            continue


        rssi_mat = np.full((Ci, n_ant), -200.0, np.float32)
        loss_mat = np.full((Ci, n_ant),  300.0, np.float32)
        ndvi_c   = ndvi_real[start:end]
        slope_c  = slope_approx[start:end]

        for j in range(n_ant):
            node_mask = within_mat[:, j]
            if not node_mask.any():
                continue
            ni = np.where(node_mask)[0]
            losses = combined_propagation_loss(
                distance_m=dist_mat[ni, j],
                freq_mhz=float(ant_freq[j]),
                h_bs=float(ant_height[j]),
                h_ms=1.5,
                ndvi=ndvi_c[ni],
                slope=slope_c[ni],
                environment=str(ant_env[j]),
            ).astype(np.float32)
            rssi_v = float(ant_power[j]) + float(ant_gain[j]) - losses
            rssi_mat[ni, j] = rssi_v.astype(np.float32)
            loss_mat[ni, j] = losses


        best_rssi_c = rssi_mat.max(axis=1)
        better      = best_rssi_c > rssi_per_node[start:end]
        if better.any():
            best_ant_c = rssi_mat.argmax(axis=1)
            rssi_per_node[start:end]     = np.where(better, best_rssi_c,
                                                     rssi_per_node[start:end])
            pathloss_per_node[start:end] = np.where(better,
                                                     loss_mat[np.arange(Ci), best_ant_c],
                                                     pathloss_per_node[start:end])


        has_cov = within_mat.any(axis=1)
        cov_idx = np.where(has_cov)[0]

        if len(cov_idx) > 0:
            rssi_cov   = rssi_mat[cov_idx]
            dist_cov   = dist_mat[cov_idx]
            within_cov = within_mat[cov_idx]

            rssi_masked = np.where(within_cov, rssi_cov, -np.inf)
            ncov = len(cov_idx)

            top_k_col = np.argsort(-rssi_masked, axis=1)[:, :k]
            top_rssi  = rssi_masked[np.arange(ncov)[:, None], top_k_col]
            valid     = (top_rssi > -150.0) & np.isfinite(top_rssi)

            if valid.any():
                # Global node indices replicated across k columns
                node_repeat  = np.repeat(start + cov_idx, k).astype(np.int32)
                valid_flat   = valid.ravel()
                tgt_flat     = node_repeat[valid_flat]
                src_flat     = top_k_col.ravel()[valid_flat].astype(np.int32)

                dist_topk    = dist_cov[np.arange(ncov)[:, None], top_k_col]
                dist_flat    = dist_topk.ravel()[valid_flat]
                freq_flat    = ant_freq[src_flat]
                ea_flat      = np.stack([dist_flat / 30_000.0,
                                         freq_flat / 2600.0], axis=1)

                all_sources.append(src_flat)
                all_targets.append(tgt_flat)
                all_ea.append(ea_flat)

            del rssi_cov, dist_cov, within_cov, rssi_masked, top_k_col
            del top_rssi, valid, dist_topk

        del dist_mat, within_mat, rssi_mat, loss_mat
        gc.collect()

        if (chunk_idx + 1) % 500 == 0:
            n_edges_so_far = sum(len(a) for a in all_sources)
            print(f"    Chunk {chunk_idx+1}/{n_chunks} | edges acum.: {n_edges_so_far:,}",
                  flush=True)

    rssi_per_node[rssi_per_node < -150] = -110.0
    coverage_flag = (rssi_per_node > -85.0).astype(np.float32)
    y_node = np.stack([
        pathloss_per_node,
        np.zeros(n_nodes, np.float32),
        np.zeros(n_nodes, np.float32),
        rssi_per_node,
        coverage_flag,
    ], axis=1)

    sources_arr = np.concatenate(all_sources) if all_sources else np.empty(0, np.int32)
    targets_arr = np.concatenate(all_targets) if all_targets else np.empty(0, np.int32)
    ea_arr      = np.concatenate(all_ea)      if all_ea      else np.empty((0, 2), np.float32)
    del all_sources, all_targets, all_ea
    gc.collect()

    print(f"    Total edges (top-{k}/no, raio={radius_km}km): {len(sources_arr):,}", flush=True)

    ei = torch.from_numpy(np.stack([sources_arr, targets_arr], axis=0))
    ea = torch.from_numpy(ea_arr)
    del sources_arr, targets_arr, ea_arr
    gc.collect()


    # compute true Haversine distance (avoids falling back to antenna.x[:,0:2] = freq/power)
    ant_pos = torch.tensor(
        df_lins[["lon", "lat"]].values, dtype=torch.float32
    )

    data = HeteroData()
    data["terrain"].x             = embeddings
    data["terrain"].pos           = pos
    data["terrain"].features_raw  = x_dem
    data["terrain"].y             = torch.tensor(y_node, dtype=torch.float32)
    data["terrain"].quadrant      = q_name
    data["antenna"].x             = ant_feats
    data["antenna"].pos           = ant_pos
    data["antenna"].num_nodes     = n_ant
    data["antenna", "propagates_to", "terrain"].edge_index = ei
    data["antenna", "propagates_to", "terrain"].edge_attr  = ea
    return data


def run_per_quadrant(quadrants: list[str], skip_existing: bool = False,
                     city: str = "lins", node_chunk_size: int = NODE_MATRIX_CHUNK,
                     max_radius_km: float | None = None,
                     top_k_per_node: int = TOP_K_PER_NODE):
    """Gera um .pt por quadrante."""
    df_lins = load_antennas(city)
    print(f"Antenas {city.title()}: {len(df_lins)}")
    for q in quadrants:
        out_path = OUTPUT_DIR / f"transfer_dataset_{city}_v19_{q}.pt"
        if skip_existing and out_path.exists():
            size_gb = out_path.stat().st_size / 1e9
            print(f"\n[{q}] Ja existe: {out_path.name} ({size_gb:.2f} GB) - pulando.")
            continue
        t0 = time.perf_counter()
        print(f"\n[{q}] Construindo transfer_dataset_{city}_v19_{q}.pt ...")
        data = build_transfer_one_quadrant(q, df_lins, city=city,
                                           node_chunk_size=node_chunk_size,
                                           max_radius_km=max_radius_km,
                                           top_k_per_node=top_k_per_node)
        torch.save(data, out_path)
        elapsed = time.perf_counter() - t0
        size_gb = out_path.stat().st_size / 1e9
        n_edges = data["antenna", "propagates_to", "terrain"].edge_index.shape[1]
        print(f"    Salvo: {out_path.name}  {size_gb:.2f} GB  |  edges: {n_edges:,}  |  {elapsed/60:.1f} min")
        del data
        gc.collect()


def run_merge(quadrants: list[str], city: str = "lins", node_chunk_size: int = NODE_MATRIX_CHUNK,
              max_radius_km: float | None = None, top_k_per_node: int = TOP_K_PER_NODE):
    """Writes a single .pt file with all quadrants concatenated (global indices)."""
    df_lins = load_antennas(city)
    n_ant = len(df_lins)
    print(f"Antenas {city.title()}: {n_ant}")

    all_terrain_x = []
    all_pos = []
    all_features_raw = []
    all_y = []
    all_sources = []
    all_targets = []
    all_edge_attr = []
    offset = 0
    ant_x = None

    for q in quadrants:
        t0 = time.perf_counter()
        print(f"\n[{q}] Carregando e construindo...")
        data = build_transfer_one_quadrant(q, df_lins, city=city,
                                           node_chunk_size=node_chunk_size,
                                           max_radius_km=max_radius_km,
                                           top_k_per_node=top_k_per_node)
        if ant_x is None:
            ant_x = data["antenna"].x.clone()
        n_nodes = data["terrain"].x.shape[0]
        all_terrain_x.append(data["terrain"].x)
        all_pos.append(data["terrain"].pos)
        all_features_raw.append(data["terrain"].features_raw)
        all_y.append(data["terrain"].y)
        ei = data["antenna", "propagates_to", "terrain"].edge_index
        all_sources.append(ei[0])
        all_targets.append(ei[1] + offset)
        all_edge_attr.append(data["antenna", "propagates_to", "terrain"].edge_attr)
        offset += n_nodes
        print(f"    {n_nodes:,} nos  |  {ei.shape[1]:,} edges  |  {time.perf_counter()-t0:.0f}s")
        del data
        gc.collect()

    print("\nConcatenando...")
    merged = HeteroData()
    merged["terrain"].x = torch.cat(all_terrain_x, dim=0)
    merged["terrain"].pos = torch.cat(all_pos, dim=0)
    merged["terrain"].features_raw = np.concatenate(all_features_raw, axis=0)
    merged["terrain"].y = torch.cat(all_y, dim=0)
    merged["terrain"].quadrants = quadrants

    merged["antenna"].x = ant_x
    merged["antenna"].num_nodes = n_ant

    ei_merged = torch.stack([
        torch.cat(all_sources, dim=0),
        torch.cat(all_targets, dim=0),
    ], dim=0)
    ea_merged = torch.cat(all_edge_attr, dim=0)
    merged["antenna", "propagates_to", "terrain"].edge_index = ei_merged
    merged["antenna", "propagates_to", "terrain"].edge_attr = ea_merged

    out_path = OUTPUT_DIR / f"transfer_dataset_{city}_v19.pt"
    torch.save(merged, out_path)
    size_gb = out_path.stat().st_size / 1e9
    print(f"\nSalvo: {out_path}  ({size_gb:.2f} GB)")
    print(f"  Terrain: {merged['terrain'].x.shape[0]:,} nos  x  {merged['terrain'].x.shape[1]} (embeddings)")
    print(f"  Edges:   {ei_merged.shape[1]:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset de transferência RF V19 multi-cidade (4 quadrantes)"
    )
    parser.add_argument(
        "--city", type=str, default="lins",
        choices=list(CITY_ANTENNA_NAME.keys()),
        help="Cidade (default: lins)",
    )
    parser.add_argument(
        "--quadrants",
        nargs="+",
        default=QUADRANTS,
        choices=QUADRANTS,
        help=f"Quadrantes a processar (default: {' '.join(QUADRANTS)})",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Gerar um único arquivo transfer_dataset_{city}_v19.pt (todos os quadrantes concatenados)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pular quadrantes que já possuem arquivo .pt gerado",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Diretório dos grafos/embeddings V19 (default: graph_data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretório de saída dos transfer_dataset_*.pt (default: graph_data)",
    )
    parser.add_argument(
        "--struct-dir",
        type=Path,
        default=None,
        help="Diretório dos grafos *_gpu.pt (default: mesmo que --graph-dir)",
    )
    parser.add_argument(
        "--node-chunk-size",
        type=int,
        default=NODE_MATRIX_CHUNK,
        help=f"No por chunk matricial CxM (default: {NODE_MATRIX_CHUNK:,}; CxMx4 ~ 34MB)",
    )
    parser.add_argument(
        "--max-radius-km",
        type=float,
        default=None,
        help=f"Raio max. antena em km (default: {MAX_RADIUS_KM}). NAO alterar para manter "
             f"comparabilidade com Lins.",
    )
    parser.add_argument(
        "--top-k-per-node",
        type=int,
        default=TOP_K_PER_NODE,
        help=f"Top-K antenas por nó por RSSI (default: {TOP_K_PER_NODE}). "
             f"Substitui o guardamento de TODAS as arestas (NxM -> NxK), "
             f"mantendo terrain.y idêntico ao de Lins.",
    )
    args = parser.parse_args()

    city = args.city.lower()

    global GRAPH_DIR, OUTPUT_DIR, STRUCT_DIR
    if args.graph_dir is not None:
        GRAPH_DIR = Path(args.graph_dir).resolve()
    if args.output_dir is not None:
        OUTPUT_DIR = Path(args.output_dir).resolve()
    if args.struct_dir is not None:
        STRUCT_DIR = Path(args.struct_dir).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ANTENNA_PATH.exists():
        print(f"Erro: arquivo de antenas nao encontrado: {ANTENNA_PATH}")
        sys.exit(1)

    print("=" * 60)
    print(f"PREPARAR TRANSFER DATASET - {city.upper()} V19")
    print(f"  Entrada (grafos/embeddings): {GRAPH_DIR}")
    print(f"  Saida (transfer .pt):        {OUTPUT_DIR}")
    print(f"  Node chunk size (C):         {args.node_chunk_size:,} nos/chunk (C x M x 4 ~ {args.node_chunk_size*4252*4//1024//1024} MB)")
    radius_km = args.max_radius_km if args.max_radius_km is not None else MAX_RADIUS_KM
    print(f"  Raio cobertura:              {radius_km} km  (igual ao de Lins - nao alterar)")
    print(f"  Top-K antenas/no:            {args.top_k_per_node}  (NxM -> NxK arestas; terrain.y identico ao de Lins)")
    print("=" * 60)

    if args.merge:
        run_merge(args.quadrants, city=city, node_chunk_size=args.node_chunk_size,
                  max_radius_km=radius_km, top_k_per_node=args.top_k_per_node)
    else:
        run_per_quadrant(args.quadrants, skip_existing=args.skip_existing, city=city,
                         node_chunk_size=args.node_chunk_size, max_radius_km=radius_km,
                         top_k_per_node=args.top_k_per_node)

    print("\n" + "=" * 60)
    print(f"Proximo passo (exemplo um quadrante):")
    print(f"  python train_lins_physics.py --rf-data-file transfer_dataset_{city}_v19_Q1.pt")
    print(f"Próximo passo (merged 4 quadrantes):")
    print(f"  python train_lins_physics.py --rf-data-file transfer_dataset_{city}_v19.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
