"""Subprocess that enriches RF targets with vegetation and terrain-diffraction channels."""

from __future__ import annotations
import argparse
import math
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd


FREQ_MHZ_DEFAULT  = 1800.0
CHUNK_DEFAULT     = 200_000
LAMBDA_M          = 0.167       # wavelength at 1800 MHz (meters)
MAX_SHADOW_DB     = 40.0
MAX_DIFFR_DB      = 40.0
NDVI_VEG_THRESH   = 0.5         # dense-vegetation threshold
ANTENNA_BBOX_BUFFER_DEG = 0.2   # buffer in degrees around the terrain bounding box

OPERATOR_TO_ID = {
    "CLARO": 0,
    "TIM": 1,
    "VIVO": 2,
    "OUTROS": 3,
}
ID_TO_OPERATOR = {v: k for k, v in OPERATOR_TO_ID.items()}

# Feature indices in terrain.x (17 columns of the base graph):
IDX_ELEVATION = 0
IDX_SLOPE     = 1
IDX_TPI       = 4
IDX_ROUGHNESS = 6
IDX_NDVI      = 12
IDX_NDWI      = 13


# Haversine distance in meters, no pyproj dependency


def _haversine_cdist_m(
    pos_a: torch.Tensor,   # (N, 2), columns [lon, lat] in DEGREES
    pos_b: torch.Tensor,   # (M, 2), columns [lon, lat] in DEGREES
) -> torch.Tensor:         # (N, M) in meters
    """
    Haversine distance between all pairs (N, M).
    Accepts geographic coordinates (degrees) and returns meters.
    Uses float64 internally for precision over long distances.
    """
    R = 6_371_000.0  # mean Earth radius in meters

    lon_a = torch.deg2rad(pos_a[:, 0].double())
    lat_a = torch.deg2rad(pos_a[:, 1].double())
    lon_b = torch.deg2rad(pos_b[:, 0].double())
    lat_b = torch.deg2rad(pos_b[:, 1].double())


    dlon = lon_b.unsqueeze(0) - lon_a.unsqueeze(1)
    dlat = lat_b.unsqueeze(0) - lat_a.unsqueeze(1)

    a = (torch.sin(dlat / 2) ** 2
         + torch.cos(lat_a.unsqueeze(1)) * torch.cos(lat_b.unsqueeze(0))
         * torch.sin(dlon / 2) ** 2)
    c = 2 * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))
    return (R * c).float()


def _haversine_nearest_chunked(
    terrain_pos: torch.Tensor,
    antenna_pos: torch.Tensor,
    chunk: int = CHUNK_DEFAULT,
    device: torch.device | None = None,
) -> torch.Tensor:               # (N,) meters
    """Distancia ao ponto de antena mais proximo usando Haversine. CPU ou GPU."""
    dev = device or torch.device("cpu")
    N = terrain_pos.shape[0]
    dist = torch.zeros(N, dtype=torch.float32)
    if dev.type == "cuda":
        ant_gpu = antenna_pos.to(dev)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            t_chunk = terrain_pos[start:end].to(dev)
            d = _haversine_cdist_m(t_chunk, ant_gpu)
            dist[start:end] = d.min(dim=1).values.cpu()
    else:
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            d = _haversine_cdist_m(terrain_pos[start:end], antenna_pos)
            dist[start:end] = d.min(dim=1).values
    return dist


def _dist_nearest_chunked(
    terrain_pos: torch.Tensor,
    antenna_pos: torch.Tensor,
    chunk: int = CHUNK_DEFAULT,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Wrapper that auto-detects the coordinate system:
    - If terrain_pos looks like degrees (|x| <= 180, |y| <= 90): Haversine, meters.
    - Otherwise (projected coordinates, meters): Euclidean.
    """
    x_range = terrain_pos[:, 0].abs().max().item()
    is_geographic = x_range <= 181.0
    dev = device or torch.device("cpu")
    if is_geographic:
        return _haversine_nearest_chunked(terrain_pos, antenna_pos, chunk, device=dev)

    N = terrain_pos.shape[0]
    dist = torch.zeros(N, dtype=torch.float32)
    if dev.type == "cuda":
        ant_gpu = antenna_pos.float().to(dev)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            t_chunk = terrain_pos[start:end].float().to(dev)
            d = torch.cdist(t_chunk, ant_gpu)
            dist[start:end] = d.min(dim=1).values.cpu()
    else:
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            d = torch.cdist(terrain_pos[start:end].float(), antenna_pos.float())
            dist[start:end] = d.min(dim=1).values
    return dist


def _load_antenna_pos_from_csv(
    csv_path: Path,
    terrain_pos: torch.Tensor,
    buffer_deg: float = ANTENNA_BBOX_BUFFER_DEG,
    freq_col: str = "FreqTxMHz",
    target_freq_mhz: float | None = None,
) -> torch.Tensor | None:
    """
    Reads the antenna CSV, filters by the terrain bounding box (plus buffer)
    and, optionally, by frequency of interest.

    Returns a (M, 2) tensor in [lon, lat] degrees, or None on failure.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, usecols=["Latitude", "Longitude", freq_col],
                     low_memory=False)
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Terrain bounding box (with buffer)
    lon_min = float(terrain_pos[:, 0].min()) - buffer_deg
    lon_max = float(terrain_pos[:, 0].max()) + buffer_deg
    lat_min = float(terrain_pos[:, 1].min()) - buffer_deg
    lat_max = float(terrain_pos[:, 1].max()) + buffer_deg

    mask = (
        (df["Longitude"] >= lon_min) & (df["Longitude"] <= lon_max) &
        (df["Latitude"]  >= lat_min) & (df["Latitude"]  <= lat_max)
    )
    df = df[mask]

    if target_freq_mhz is not None:
        # Accept antennas within +/-200 MHz of the target frequency
        freq_mask = (df[freq_col] - target_freq_mhz).abs() <= 200.0
        df_freq = df[freq_mask]
        if len(df_freq) >= 5:
            df = df_freq

    if len(df) == 0:
        return None


    df = df.drop_duplicates(subset=["Latitude", "Longitude"])

    lons = torch.tensor(df["Longitude"].values, dtype=torch.float32)
    lats = torch.tensor(df["Latitude"].values,  dtype=torch.float32)
    return torch.stack([lons, lats], dim=1)


def _infer_city_from_rf_path(rf_data_path: Path) -> str | None:
    name = rf_data_path.name.lower()
    m = re.search(r"transfer_dataset_([a-z0-9_]+)_v\d+_q[1-4]", name)
    if m:
        return m.group(1)
    return None


def _load_city_antennas_for_metadata(csv_path: Path, city: str) -> pd.DataFrame:
    city_map = {
        "lins": "LINS",
        "campinas": "CAMPINAS",
        "sorocaba": "SOROCABA",
        "bauru": "BAURU",
    }
    city_name = city_map.get(city.lower(), city.upper())
    df = pd.read_csv(csv_path, low_memory=False)
    df_city = df[df["Municipio.NomeMunicipio"].str.upper() == city_name].copy()
    df_city = df_city.rename(columns={
        "Latitude": "lat",
        "Longitude": "lon",
        "AlturaAntena": "altura_antena",
        "FreqTxMHz": "freq_mhz",
        "PotenciaTransmissorWatts": "pot_watts",
        "GanhoAntena": "ganho_dbi",
    })
    for col in ["lat", "lon", "altura_antena", "freq_mhz", "pot_watts", "ganho_dbi"]:
        df_city[col] = pd.to_numeric(df_city[col], errors="coerce").fillna(0.0)
    df_city = df_city[(df_city["freq_mhz"] > 0) & (df_city["pot_watts"] > 0)].reset_index(drop=True)
    return df_city


def _inject_antenna_metadata(
    rf_data,
    rf_data_path: Path,
    csv_candidates: list[Path],
    log_fn,
) -> None:
    """Injeta metadados de antena alinhados com o prepare_transfer_dataset_v19.py."""
    city = _infer_city_from_rf_path(rf_data_path)
    if not city:
        log_fn("AVISO: nao foi possivel inferir cidade pelo nome do arquivo; metadados de antena nao injetados.")
        return

    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        log_fn("AVISO: CSV de antenas nao encontrado; metadados de antena nao injetados.")
        return

    try:
        df_city = _load_city_antennas_for_metadata(csv_path, city)
    except Exception as exc:
        log_fn(f"AVISO: falha ao carregar CSV para metadata ({exc}); metadados nao injetados.")
        return

    ant_x = rf_data["antenna"].x
    if isinstance(ant_x, np.ndarray):
        ant_x = torch.from_numpy(ant_x)
    M_graph = ant_x.shape[0]
    M_csv = len(df_city)
    if M_graph != M_csv:
        log_fn(f"AVISO: alinhamento de antenas falhou (grafo={M_graph}, csv={M_csv}); metadados nao injetados.")
        return

    lat_graph = ant_x[:, 4].cpu().numpy().astype(np.float32)
    lon_graph = ant_x[:, 5].cpu().numpy().astype(np.float32)
    lat_csv = df_city["lat"].values.astype(np.float32)
    lon_csv = df_city["lon"].values.astype(np.float32)
    max_lat_err = float(np.abs(lat_graph - lat_csv).max()) if M_graph > 0 else 0.0
    max_lon_err = float(np.abs(lon_graph - lon_csv).max()) if M_graph > 0 else 0.0
    if max_lat_err > 1e-3 or max_lon_err > 1e-3:
        log_fn(
            "AVISO: coordenadas antena.x divergiram do CSV "
            f"(lat={max_lat_err:.6f}, lon={max_lon_err:.6f}); metadados nao injetados."
        )
        return

    operator_col = df_city.get("Operadora", pd.Series(["OUTROS"] * M_csv)).fillna("OUTROS").astype(str).str.upper().str.strip()
    operator_ids = operator_col.map(lambda x: OPERATOR_TO_ID.get(x, OPERATOR_TO_ID["OUTROS"])).values.astype(np.int64)
    station_ids = pd.to_numeric(df_city.get("NumEstacao", pd.Series([0] * M_csv)), errors="coerce").fillna(0).astype(np.int64).values
    antenna_ids = np.arange(M_graph, dtype=np.int64)

    rf_data["antenna"].antenna_id = torch.tensor(antenna_ids, dtype=torch.long)
    rf_data["antenna"].station_id = torch.tensor(station_ids, dtype=torch.long)
    rf_data["antenna"].operator_id = torch.tensor(operator_ids, dtype=torch.long)
    rf_data["antenna"].operator_names = ID_TO_OPERATOR
    log_fn(
        "Metadados de antena injetados: "
        f"antenna_id/station_id/operator_id (M={M_graph})"
    )


def _knife_edge_loss_db(v: torch.Tensor) -> torch.Tensor:
    """
    Additional Knife-Edge diffraction loss in dB (ITU-R P.526 approximation).
    v: dimensionless Fresnel parameter.

    The original formula expresses field amplitude as a fraction of the LOS
    value; here it is converted to a path-loss term (positive dB values).
    Convention: v < 0 = obstacle below the LOS (small loss),
                v > 0 = obstacle above the LOS (loss grows).
    """
    loss = torch.zeros_like(v)

    m2 = (v >= -0.7) & (v <= 0.0)
    m3 = (v >  0.0) & (v <= 1.0)
    m4 = (v >  1.0) & (v <= 2.4)
    m5 = v > 2.4

    # Negative of 20*log10(amplitude_ratio) = positive loss in dB
    loss[m2] = -20.0 * torch.log10((0.5 - 0.62 * v[m2]).clamp(min=1e-9))
    loss[m3] = -20.0 * torch.log10((0.5 * torch.exp(-0.95 * v[m3])).clamp(min=1e-9))
    loss[m4] = -20.0 * torch.log10(
        (0.4 - torch.sqrt((0.1184 - (0.38 - 0.1 * v[m4]) ** 2).clamp(min=1e-9))).clamp(min=1e-9)
    )
    loss[m5] = -20.0 * torch.log10((0.225 / v[m5].clamp(min=1e-6)).clamp(min=1e-9))

    return loss.clamp(min=0.0)


def compute_shadow_margin(
    ndvi: torch.Tensor,
    dist_nearest_m: torch.Tensor, # (N,) meters
    freq_mhz: float = FREQ_MHZ_DEFAULT,
) -> torch.Tensor:
    """
    Shadow margin por atenuacao de vegetacao (ITU-R P.833-10).

    Modelo:
        A = d_veg * gamma_veg(f, ndvi)
        d_veg  = dist * clamp(ndvi - NDVI_VEG_THRESH, 0, 1)  [metros em vegetacao]
        gamma  = 0.2 * ndvi * (f_GHz / 1.8)^0.4              [dB/m]

    Retorna shadow_margin em dB (>= 0).
    """
    freq_ghz = freq_mhz / 1000.0
    veg_fraction = (ndvi - NDVI_VEG_THRESH).clamp(0.0, 1.0)
    d_veg = dist_nearest_m * veg_fraction                     # meters inside dense vegetation
    gamma = 0.2 * ndvi.clamp(0.0, 1.0) * (freq_ghz / 1.8) ** 0.4  # dB/m
    shadow = d_veg * gamma
    return shadow.clamp(0.0, MAX_SHADOW_DB)


def compute_diffraction_loss(
    tri_meters: torch.Tensor,     # (N,) TRI in real meters (elev_std applied)
    roughness: torch.Tensor,      # (N,) feature col 6 (normalized units)
    dist_nearest_m: torch.Tensor, # (N,) meters
    freq_mhz: float = FREQ_MHZ_DEFAULT,
) -> torch.Tensor:
    """
    Knife-Edge diffraction loss (ITU-R P.526 approximation).

    Approximation used when no full DEM profile is available:
        - obstruction proxy = TRI in real meters (Terrain Ruggedness Index),
          the mean elevation deviation from neighbors; high values indicate
          irregular terrain likely to obstruct the line of sight.
        - the obstruction is placed at the path midpoint (d1 = d2 = dist/2).
        - normalized roughness adds a smaller additional fraction.

    Returns diffraction_loss in dB (>= 0).
    """
    lambda_m = 3e8 / (freq_mhz * 1e6)

    # h_obs in meters: TRI is already in meters, roughness is still normalized -> small scale
    h_obs = (tri_meters.clamp(min=0.0) * 0.5 + roughness * 0.5).clamp(0.0, 80.0)


    d1 = (dist_nearest_m / 2.0).clamp(min=1.0)
    d2 = d1


    eps = 1e-9
    v = h_obs * torch.sqrt(
        2.0 * (d1 + d2) / (lambda_m * d1 * d2 + eps)
    )

    return _knife_edge_loss_db(v).clamp(0.0, MAX_DIFFR_DB)


def build_enriched_targets(
    rf_data_path: Path,
    struct_path: Path,
    output_path: Path,
    freq_mhz: float = FREQ_MHZ_DEFAULT,
    chunk: int = CHUNK_DEFAULT,
    device: str | torch.device = "cpu",
) -> None:
    t0 = time.perf_counter()
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
        print("[AVISO] --device cuda pedido mas CUDA nao disponivel; usando CPU.", flush=True)

    def log(msg: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Carregando RF dataset: {rf_data_path.name}")
    rf_data = torch.load(rf_data_path, map_location="cpu", weights_only=False)

    log(f"Carregando grafo base V18.5: {struct_path.name}")
    struct_data = torch.load(struct_path, map_location="cpu", weights_only=False)

    log(f"Device para calculos: {dev}")


    targets_orig = rf_data["terrain"].y
    if isinstance(targets_orig, np.ndarray):
        targets_orig = torch.from_numpy(targets_orig)
    targets_orig = targets_orig.float().clone()
    N = targets_orig.shape[0]
    log(f"Targets originais: {N:,} nos x {targets_orig.shape[1]} colunas")
    log(f"  shadow_margin (col1): mean={targets_orig[:,1].mean():.4f}  std={targets_orig[:,1].std():.4f}")
    log(f"  diffraction  (col2): mean={targets_orig[:,2].mean():.4f}  std={targets_orig[:,2].std():.4f}")


    dem_feats = struct_data["dem"].x.float()
    ndvi      = dem_feats[:, IDX_NDVI]
    tpi       = dem_feats[:, IDX_TPI]
    roughness = dem_feats[:, IDX_ROUGHNESS]

    # TRI (col 5): Terrain Ruggedness Index, elevation std in normalized units.
    # Converted to meters using elev_std from the normalization dict.
    IDX_TRI   = 5
    tri_norm  = dem_feats[:, IDX_TRI]
    norm_meta = getattr(struct_data, "normalization", {}) or {}
    elev_std  = float(norm_meta.get("elev_std", 1.0))
    elev_mean = float(norm_meta.get("elev_mean", 0.0))
    tri_m     = tri_norm * elev_std                          # TRI in real meters
    log(f"Features DEM: NDVI mean={ndvi.mean():.3f}  TPI mean={tpi.mean():.4f}  TRI_norm mean={tri_norm.mean():.4f}")
    log(f"  Normalizacao: elev_mean={elev_mean:.2f}m  elev_std={elev_std:.2f}m  TRI_m max={tri_m.max():.2f}m")


    terrain_pos = rf_data["terrain"].pos.float()   # (N, 2) [lon, lat] in degrees (EPSG:4326)


    csv_candidates = [
        rf_data_path.parent / "antenas_interior_sp_final.csv",
        rf_data_path.parent.parent / "data_raw" / "antenas_interior_sp_final.csv",
    ]
    antenna_pos = None
    antenna_source = "desconhecido"

    for csv_path in csv_candidates:
        pos_from_csv = _load_antenna_pos_from_csv(
            csv_path, terrain_pos, target_freq_mhz=freq_mhz
        )
        if pos_from_csv is not None:
            antenna_pos   = pos_from_csv
            antenna_source = f"CSV ({csv_path.name}, {antenna_pos.shape[0]} torres unicas)"
            break

    if antenna_pos is None and hasattr(rf_data["antenna"], "pos"):
        antenna_pos    = rf_data["antenna"].pos.float()
        antenna_source = "antenna.pos do dataset"

    if antenna_pos is None:


        ax = rf_data["antenna"].x
        if isinstance(ax, np.ndarray):
            ax = torch.from_numpy(ax)
        if ax.shape[1] >= 6:

            antenna_pos    = ax[:, [5, 4]].float()
            antenna_source = "antenna.x[:,5,4] lon/lat (fallback colunas 5 e 4)"
            log("AVISO: antenna.pos ausente e CSV nao encontrado.")
            log("       Usando antenna.x colunas [5=lon, 4=lat] como fallback de coordenadas.")
        else:
            antenna_pos    = ax[:, :2].float()
            antenna_source = "antenna.x[:,:2] (FALLBACK IMPRECISO — menos de 6 colunas)"
            log("AVISO: CSV nao encontrado, antenna.pos ausente, e antenna.x tem < 6 cols.")
            log("       Usando antenna.x[:,:2] — distancias INCORRETAS.")

    M = antenna_pos.shape[0]
    log(f"Posicoes de antena: {antenna_source}")
    log(f"Terrain pos: {terrain_pos.shape}  Antenna pos: {antenna_pos.shape}")


    log(f"Calculando dist_nearest ({N:,} nos, chunk={chunk:,}, device={dev})...")
    dist_nearest = _dist_nearest_chunked(terrain_pos, antenna_pos, chunk=chunk, device=dev)
    is_haversine = terrain_pos[:, 0].abs().max().item() <= 181.0
    dist_unit    = "m (Haversine)" if is_haversine else "m (Euclidiana)"
    log(f"  dist_nearest [{dist_unit}]: min={dist_nearest.min():.0f}  max={dist_nearest.max():.0f}  mean={dist_nearest.mean():.0f}")


    log("Calculando shadow_margin (ITU-R P.833-10)...")
    if dev.type == "cuda":
        shadow_margin = compute_shadow_margin(
            ndvi.to(dev), dist_nearest.to(dev), freq_mhz
        ).cpu()
    else:
        shadow_margin = compute_shadow_margin(ndvi, dist_nearest, freq_mhz)
    log(f"  shadow_margin: mean={shadow_margin.mean():.2f}dB  std={shadow_margin.std():.2f}dB  max={shadow_margin.max():.2f}dB")
    log(f"  nos com shadow > 1dB: {(shadow_margin > 1.0).sum():,} ({(shadow_margin > 1.0).float().mean()*100:.1f}%)")


    # Uses TRI in meters (denormalized) as a proxy for h_obs; far more variable than normalized TPI
    log("Calculando diffraction loss (Knife-Edge ITU-R P.526, proxy=TRI_metros)...")
    if dev.type == "cuda":
        diffr_loss = compute_diffraction_loss(
            tri_m.to(dev), roughness.to(dev), dist_nearest.to(dev), freq_mhz
        ).cpu()
    else:
        diffr_loss = compute_diffraction_loss(tri_m, roughness, dist_nearest, freq_mhz)
    log(f"  diffr_loss:    mean={diffr_loss.mean():.2f}dB  std={diffr_loss.std():.2f}dB  max={diffr_loss.max():.2f}dB")
    log(f"  nos com diffr > 1dB: {(diffr_loss > 1.0).sum():,} ({(diffr_loss > 1.0).float().mean()*100:.1f}%)")


    targets_enriched = targets_orig.clone()
    targets_enriched[:, 1] = shadow_margin
    targets_enriched[:, 2] = diffr_loss
    log(f"Targets enriquecidos:")
    for i, name in enumerate(["path_loss", "shadow_margin", "diffraction", "rssi", "coverage"]):
        col = targets_enriched[:, i]
        log(f"  col{i} {name:15s}: mean={col.mean():8.3f}  std={col.std():7.3f}  min={col.min():8.3f}  max={col.max():7.3f}")


    # Ensures the output file has the same format as the Lins one:

    # usable directly as train_lins_physics.py --rf-data-file <output>.
    rf_data["terrain"].y = targets_enriched

    rf_data["terrain"].dist_nearest_m = dist_nearest


    _inject_antenna_metadata(rf_data, rf_data_path, csv_candidates, log)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rf_data, output_path)
    size_gb  = output_path.stat().st_size / 1e9
    elapsed  = time.perf_counter() - t0
    log(f"HeteroData enriquecido salvo: {output_path}  ({size_gb:.2f} GB)  em {elapsed:.1f}s")
    log(f"  terrain.y enriquecido injetado: {targets_enriched.shape}")
    log(f"  antenna.pos presente: {hasattr(rf_data['antenna'], 'pos')}")
    log(f"  edges: {rf_data['antenna', 'propagates_to', 'terrain'].edge_index.shape[1]:,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enriquece targets RF com shadow e diffraction.")
    p.add_argument("--rf-path",   default=None, help="Caminho para transfer_dataset_*.pt")
    p.add_argument("--struct-path", default=None, help="Caminho para grafo base V18.5")
    p.add_argument("--output",    default=None, help="Caminho de saida do artefato")
    p.add_argument("--freq-mhz",  type=float, default=FREQ_MHZ_DEFAULT)
    p.add_argument("--chunk",     type=int,   default=CHUNK_DEFAULT)
    p.add_argument("--device",    default="cpu", choices=("cpu", "cuda"),
                   help="Device para calculos (cuda usa VRAM e acelera)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base = Path(__file__).parent.parent

    rf_path  = Path(args.rf_path)   if args.rf_path   else base / "graph_data/transfer_dataset_lins_q1.pt"
    struct   = Path(args.struct_path) if args.struct_path else base / "graph_data/interior_sp_v18.5_part_Lins_Q1_gpu.pt"
    out_path = Path(args.output)    if args.output    else base / "graph_data/rf_targets_enriched_lins_q1.pt"

    print("=" * 65)
    print("ENRICH RF TARGETS — shadow + diffraction")
    print("=" * 65)
    print(f"  RF dataset : {rf_path}")
    print(f"  Grafo base : {struct}")
    print(f"  Saida      : {out_path}")
    print(f"  Freq       : {args.freq_mhz} MHz")
    print("=" * 65)

    build_enriched_targets(rf_path, struct, out_path, args.freq_mhz, args.chunk, device=args.device)

    print("=" * 65)
    print("CONCLUIDO")
    print("=" * 65)
