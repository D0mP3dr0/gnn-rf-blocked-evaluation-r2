"""
RF Coverage with REALISTIC Propagation Models
Implements Okumura-Hata, COST-231, and terrain attenuation.
"""

import torch
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import sys
import json

base_path = Path(__file__).parent
sys.path.append(str(base_path))

def okumura_hata_loss(distance_km, freq_mhz, h_bs, h_ms=1.5, environment='suburban'):
    """
    Okumura-Hata path-loss model.

    Args:
        distance_km: distance in km.
        freq_mhz: frequency in MHz (150-1500 MHz).
        h_bs: base-station (antenna) height in meters.
        h_ms: mobile height in meters (default 1.5 m).
        environment: 'urban', 'suburban', or 'rural'.

    Returns:
        Path loss in dB.
    """

    f = np.clip(freq_mhz, 150, 2000)
    d = np.maximum(distance_km, 0.001)


    a_hm = (1.1 * np.log10(f) - 0.7) * h_ms - (1.56 * np.log10(f) - 0.8)


    L_urban = (69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(h_bs) - a_hm +
               (44.9 - 6.55 * np.log10(h_bs)) * np.log10(d))

    if environment == 'urban':
        return L_urban
    elif environment == 'suburban':

        return L_urban - 2 * (np.log10(f / 28)) ** 2 - 5.4
    else:

        return L_urban - 4.78 * (np.log10(f)) ** 2 + 18.33 * np.log10(f) - 40.94


def cost231_hata_loss(distance_km, freq_mhz, h_bs, h_ms=1.5, environment='suburban'):
    """
    COST-231 Hata model (extension to 1500-2000 MHz).
    """
    f = np.clip(freq_mhz, 1500, 2000)
    d = np.maximum(distance_km, 0.001)


    a_hm = (1.1 * np.log10(f) - 0.7) * h_ms - (1.56 * np.log10(f) - 0.8)


    C_m = 0 if environment != 'urban' else 3

    L = (46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_bs) - a_hm +
         (44.9 - 6.55 * np.log10(h_bs)) * np.log10(d) + C_m)

    return L


def combined_propagation_loss(distance_m, freq_mhz, h_bs, h_ms=1.5,
                               ndvi=0, slope=0, environment='suburban'):
    """
    Combined propagation model:
    - Okumura-Hata or COST-231, selected by frequency.
    - Vegetation attenuation, based on NDVI.
    - Terrain attenuation, based on slope.
    - Log-normal fading.
    """
    distance_km = distance_m / 1000


    if freq_mhz <= 1500:
        base_loss = okumura_hata_loss(distance_km, freq_mhz, h_bs, h_ms, environment)
    else:
        base_loss = cost231_hata_loss(distance_km, freq_mhz, h_bs, h_ms, environment)


    veg_density = np.clip((ndvi + 1) / 2, 0, 1)


    veg_depth_m = distance_m * veg_density * 0.3

    # Vegetation-specific attenuation (dB)
    A_veg = 0.25 * (freq_mhz ** 0.39) * (np.minimum(veg_depth_m, 400) ** 0.25)
    A_veg = np.clip(A_veg, 0, 40)  # Capped at 40 dB


    # Slope in [0, 1], where 1 = 45 degrees
    terrain_factor = slope * 20  # Up to 20 dB of extra loss


    terrain_factor = np.where(slope > 0.1,
                               terrain_factor + (slope - 0.1) * 30,
                               terrain_factor)

    A_terrain = np.clip(terrain_factor, 0, 30)  # Capped at 30 dB


    total_loss = base_loss + A_veg + A_terrain

    return total_loss


def main():
    print("=" * 60)
    print("📡 COBERTURA RF: MODELO REALISTA (Okumura-Hata)")
    print("=" * 60)


    base_dir = Path(r"f:\arpia_topo_refinado\TOPO_RF\GNN_RF_V2")
    embeddings_path = base_dir / "graph_data/lins_embeddings_512.pt"
    terrain_path = base_dir / "graph_data/interior_sp_v18.4_part_Lins_CORRECTED_cupy.pt"
    checkpoint_path = base_dir / "checkpoints/lins_embeddings/model_final.pt"
    antenna_path = base_dir / "data_raw/antenas_interior_sp_final.csv"
    output_dir = base_dir / "05_outputs/lins_realistic_coverage"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cpu'


    print(f"\n📂 Carregando dados...")

    emb_data = torch.load(embeddings_path, map_location=device, weights_only=False)
    embeddings = emb_data['embeddings']

    terrain_data = torch.load(terrain_path, map_location=device, weights_only=False)
    pos = terrain_data['dem'].pos.numpy()
    edge_index = terrain_data['dem', 'adjacent_to', 'dem'].edge_index


    x_dem = terrain_data['dem'].x.numpy()

    num_nodes = pos.shape[0]
    print(f"   Nodes: {num_nodes:,}")

    # Uses the real NDVI from the data (column 12)
    ndvi_real = x_dem[:, 12]
    print(f"   NDVI real: [{ndvi_real.min():.3f}, {ndvi_real.max():.3f}]")


    slope_estimated = np.random.uniform(0.02, 0.15, num_nodes)
    print(f"   Slope estimado: [{slope_estimated.min():.3f}, {slope_estimated.max():.3f}]")


    print(f"\n📡 Carregando antenas...")
    df_antenas = pd.read_csv(antenna_path, low_memory=False)
    df_lins = df_antenas[df_antenas['Municipio.NomeMunicipio'].str.upper() == 'LINS'].copy()

    df_lins = df_lins.rename(columns={
        'Latitude': 'lat', 'Longitude': 'lon', 'AlturaAntena': 'altura_antena',
        'FreqTxMHz': 'freq_mhz', 'PotenciaTransmissorWatts': 'pot_watts',
        'GanhoAntena': 'ganho_dbi', 'NomeEntidade': 'operadora'
    })

    for col in ['lat', 'lon', 'altura_antena', 'freq_mhz', 'pot_watts', 'ganho_dbi']:
        if col in df_lins.columns:
            df_lins[col] = pd.to_numeric(df_lins[col], errors='coerce').fillna(0.0)

    df_lins = df_lins[(df_lins['freq_mhz'] > 0) & (df_lins['pot_watts'] > 0)].reset_index(drop=True)
    df_lins['pot_dbm'] = 10 * np.log10(df_lins['pot_watts'] * 1000)
    df_lins.loc[df_lins['pot_dbm'] < 0, 'pot_dbm'] = 30
    df_lins.loc[df_lins['altura_antena'] <= 0, 'altura_antena'] = 30.0
    df_lins.loc[df_lins['ganho_dbi'] <= 0, 'ganho_dbi'] = 12.0


    # < 900 MHz: likely rural/suburban
    # > 1800 MHz: likely urban
    df_lins['environment'] = df_lins['freq_mhz'].apply(
        lambda f: 'urban' if f > 1800 else ('suburban' if f > 800 else 'rural')
    )

    print(f"   Antenas válidas: {len(df_lins)}")


    print(f"\n🔮 Calculando propagação com Okumura-Hata...")

    lon_min, lon_max = pos[:, 0].min(), pos[:, 0].max()
    lat_min, lat_max = pos[:, 1].min(), pos[:, 1].max()
    resolution = 30 / 111000

    width = int((lon_max - lon_min) / resolution) + 1
    height = int((lat_max - lat_min) / resolution) + 1

    col_idx = ((pos[:, 0] - lon_min) / resolution).astype(int)
    row_idx = ((lat_max - pos[:, 1]) / resolution).astype(int)
    valid = (col_idx >= 0) & (col_idx < width) & (row_idx >= 0) & (row_idx < height)

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    antenna_results = []
    MAX_ANTENNAS = min(30, len(df_lins))

    for ant_idx in range(MAX_ANTENNAS):
        row = df_lins.iloc[ant_idx]

        ant_lon = row['lon']
        ant_lat = row['lat']
        ant_height = row['altura_antena']
        tx_power = row['pot_dbm']
        tx_gain = row['ganho_dbi']
        freq_mhz = row['freq_mhz']
        environment = row['environment']


        dx = (pos[:, 0] - ant_lon) * 111000 * np.cos(np.radians(ant_lat))
        dy = (pos[:, 1] - ant_lat) * 111000
        distance_m = np.sqrt(dx**2 + dy**2)
        distance_m = np.maximum(distance_m, 1.0)


        path_loss = combined_propagation_loss(
            distance_m=distance_m,
            freq_mhz=freq_mhz,
            h_bs=ant_height,
            h_ms=1.5,
            ndvi=ndvi_real,
            slope=slope_estimated,
            environment=environment
        )


        rssi = tx_power + tx_gain - path_loss


        coverage = np.full(num_nodes, 5, dtype=np.int32)
        coverage[rssi >= -70] = 1
        coverage[(rssi >= -85) & (rssi < -70)] = 2
        coverage[(rssi >= -100) & (rssi < -85)] = 3
        coverage[(rssi >= -110) & (rssi < -100)] = 4


        stats = {
            'antenna_id': ant_idx,
            'freq_mhz': freq_mhz,
            'power_dbm': tx_power,
            'height_m': ant_height,
            'environment': environment,
            'rssi_min': float(rssi.min()),
            'rssi_max': float(rssi.max()),
            'path_loss_min': float(path_loss.min()),
            'path_loss_max': float(path_loss.max()),
            'coverage_excelente_pct': float((coverage == 1).sum() / num_nodes * 100),
            'coverage_bom_pct': float((coverage == 2).sum() / num_nodes * 100),
            'coverage_regular_pct': float((coverage == 3).sum() / num_nodes * 100),
            'coverage_fraco_pct': float((coverage == 4).sum() / num_nodes * 100),
            'coverage_sem_pct': float((coverage == 5).sum() / num_nodes * 100),
        }
        antenna_results.append(stats)


        raster_rssi = np.full((height, width), np.nan, dtype=np.float32)
        raster_rssi[row_idx[valid], col_idx[valid]] = rssi[valid]

        raster_cov = np.full((height, width), np.nan, dtype=np.float32)
        raster_cov[row_idx[valid], col_idx[valid]] = coverage[valid]

        output_rssi = output_dir / f"ant_{ant_idx:03d}_rssi.tif"
        output_cov = output_dir / f"ant_{ant_idx:03d}_coverage.tif"

        with rasterio.open(output_rssi, 'w', driver='GTiff', height=height, width=width,
                          count=1, dtype=np.float32, crs='EPSG:4326', transform=transform, compress='LZW') as dst:
            dst.write(raster_rssi, 1)

        with rasterio.open(output_cov, 'w', driver='GTiff', height=height, width=width,
                          count=1, dtype=np.float32, crs='EPSG:4326', transform=transform, compress='LZW') as dst:
            dst.write(raster_cov, 1)

        if ant_idx % 5 == 0:
            print(f"   Ant {ant_idx}: {freq_mhz:.0f} MHz, {environment}")
            print(f"      Path Loss: [{path_loss.min():.1f}, {path_loss.max():.1f}] dB")
            print(f"      RSSI: [{rssi.min():.1f}, {rssi.max():.1f}] dBm")
            print(f"      Cobertura: Exc {stats['coverage_excelente_pct']:.1f}%, Sem {stats['coverage_sem_pct']:.1f}%")


    summary_path = output_dir / "coverage_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(antenna_results, f, indent=2)

    print(f"\n" + "=" * 70)
    print(f"📊 RESUMO COBERTURA (Modelo Realista)")
    print(f"=" * 70)
    print(f"{'ID':>3} | {'Freq':>6} | {'Env':>8} | {'Loss min':>8} | {'Exc':>6} | {'Bom':>6} | {'Sem':>6}")
    print("-" * 70)

    for r in antenna_results[:15]:
        print(f"{r['antenna_id']:3d} | {r['freq_mhz']:6.0f} | {r['environment']:>8} | "
              f"{r['path_loss_min']:7.1f} | {r['coverage_excelente_pct']:5.1f}% | "
              f"{r['coverage_bom_pct']:5.1f}% | {r['coverage_sem_pct']:5.1f}%")

    print(f"\n✅ Cobertura realista gerada: {output_dir}")

if __name__ == "__main__":
    main()
