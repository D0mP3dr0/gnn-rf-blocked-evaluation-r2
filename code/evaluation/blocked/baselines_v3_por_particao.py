

"""Per-partition calibrated analytical baselines (FSPL, Hata Rural, COST-231 Suburban) for the blocked evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


GRID_KM = 10.0
BUFFER_KM = 2.0
SPLIT_SEED = 42
SPLIT_FRACS = (0.70, 0.15, 0.15)
FREQ_MHZ_CAMPANHA = 900.0
PL_TARGET_MAX_VALID = 299.0
RSSI_FLOOR_DBM = -110.0
FREQ_EDGE_SCALE = 2600.0
DIST_EDGE_SCALE = 30000.0
HATA_COST231_TOL_DB = 1e-9

ET_AT = ("antenna", "propagates_to", "terrain")

DEFAULT_EVID_DIR = (r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access"
                    r"\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO")
DEFAULT_GRAPH_DIR = r"F:\TOPO_RF_DOWNLOAD_DRIVE\graph_data_v3"
DEFAULT_OUT_DIR = (r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access"
                   r"\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO"
                   r"\_campanha_2026-09-13\t5_baselines_v3")

CONGELADO_REL = r"dados\scripts_congelados\train_gnn_c0_spatial.py"
V2_SCRIPT_REL = r"scripts\baselines_v2_por_particao.py"
EMPIRICAL_MODELS_PATH = r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\04_baselines\empirical_models.py"
SPATIAL_CV_PATH = r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\03_training\spatial_cv.py"

NOTA_OFFSET_AJUSTE = (
    "offset_ajuste_db (antigo p_tx_eff_dbm) e um parametro livre de UM "
    "GRAU DE LIBERDADE, calibrado por minimizacao numerica do MAE de "
    "TREINO do baseline clampado (varredura 1-D, ver calibracao.regra). "
    "NAO e potencia de transmissao de estacao nem EIRP: convertido para mW "
    "assumindo P=10^(offset/10), os valores calibrados desta celula caem em "
    "faixas fisicamente impossiveis para uma ERB real (FSPL tipicamente "
    "dezenas de mW; COST-231 suburbano tipicamente dezenas de kW). Mantido "
    "com o nome antigo p_tx_eff_dbm SOMENTE por compatibilidade com o v2; "
    "o nome de referencia a partir de T5 e offset_ajuste_db. Fecha "
    "parecer fisica B4 e parecer eng-IA BI-4 (2026-09-13)."
)

RAIO_TERRA_M = 6371000.0


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_idx(idx: np.ndarray) -> str:
    a = np.ascontiguousarray(np.sort(np.asarray(idx, dtype=np.int64)))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return str(o)


# Equirectangular projection (same convention as the training script).


def latlon_graus_para_metros(pos: torch.Tensor) -> tuple[np.ndarray, dict]:
    lon = pos[:, 0].double().numpy()
    lat = pos[:, 1].double().numpy()
    lon_min, lat_min = float(lon.min()), float(lat.min())
    y = (lat - lat_min) * 111_000.0
    x = (lon - lon_min) * 111_000.0 * np.cos(np.radians(lat))
    meta = {
        "convencao": "equiretangular do ETL (prepare_transfer_dataset_v19.py:181-184)",
        "deg_para_m": 111000.0,
        "cos_phi": "por no (cos da latitude do proprio no)",
        "lon_min_deg": lon_min, "lat_min_deg": lat_min,
        "lon_max_deg": float(lon.max()), "lat_max_deg": float(lat.max()),
        "extensao_x_km": float((x.max() - x.min()) / 1000.0),
        "extensao_y_km": float((y.max() - y.min()) / 1000.0),
    }
    return np.stack([x, y], axis=1), meta


def projetar_com_referencia(lon: np.ndarray, lat: np.ndarray,
                             lon_min: float, lat_min: float) -> np.ndarray:
    """
    Uses the latlon_graus_para_metros formula, with the reference point
    (lon_min, lat_min) fixed to the terrain's reference, keeping antenna
    and terrain coordinates in the same metric frame.
    """
    y = (lat - lat_min) * 111_000.0
    x = (lon - lon_min) * 111_000.0 * np.cos(np.radians(lat))
    return np.stack([x, y], axis=1)


def _assign_groups_inline(pos_km: np.ndarray, grid_km: float) -> np.ndarray:
    grid_x = (pos_km[:, 0] / grid_km).astype(int)
    grid_y = (pos_km[:, 1] / grid_km).astype(int)
    max_y = grid_y.max() + 1
    group_ids = grid_x * max_y + grid_y
    return group_ids


def split_espacial_3vias(pos_m: np.ndarray, grid_km: float, buffer_km: float,
                         fracs: tuple[float, float, float], split_seed: int,
                         log) -> tuple[dict, dict]:
    from scipy.spatial import cKDTree

    pos_km = pos_m / 1000.0
    group_ids = _assign_groups_inline(pos_km, grid_km)
    grupos = np.unique(group_ids)
    rng = np.random.RandomState(split_seed)
    grupos_emb = grupos.copy()
    rng.shuffle(grupos_emb)

    n_g = len(grupos_emb)
    n_tr = max(1, int(round(fracs[0] * n_g)))
    n_va = max(1, int(round(fracs[1] * n_g)))
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2)
        n_va = 1
    g_tr = grupos_emb[:n_tr]
    g_va = grupos_emb[n_tr:n_tr + n_va]
    g_te = grupos_emb[n_tr + n_va:]

    m_tr = np.isin(group_ids, g_tr)
    m_va = np.isin(group_ids, g_va)
    m_te = np.isin(group_ids, g_te)
    n_va_bruto, n_te_bruto = int(m_va.sum()), int(m_te.sum())

    log(f"  Blocos {grid_km:.3f} km: {n_g} total -> train {len(g_tr)} | "
        f"val {len(g_va)} | test {len(g_te)}")

    kw = dict(compact_nodes=False, balanced_tree=False)
    if buffer_km > 0:
        if m_tr.any() and m_va.any():
            tree_tr = cKDTree(pos_km[m_tr], **kw)
            d, _ = tree_tr.query(pos_km[m_va], k=1, workers=-1)
            idx_va = np.where(m_va)[0]
            m_va[idx_va[d < buffer_km]] = False
        m_trva = m_tr | m_va
        if m_trva.any() and m_te.any():
            tree_trva = cKDTree(pos_km[m_trva], **kw)
            d, _ = tree_trva.query(pos_km[m_te], k=1, workers=-1)
            idx_te = np.where(m_te)[0]
            m_te[idx_te[d < buffer_km]] = False

    parts = {"train": np.where(m_tr)[0].astype(np.int64),
             "val": np.where(m_va)[0].astype(np.int64),
             "test": np.where(m_te)[0].astype(np.int64)}

    info = {
        "grid_km": grid_km, "buffer_km": buffer_km,
        "n_blocos_total": int(n_g),
        "n_blocos": {"train": int(len(g_tr)), "val": int(len(g_va)),
                     "test": int(len(g_te))},
        "fracs_blocos_pedidas": list(fracs),
        "split_seed": split_seed,
        "n_nos_antes_do_buffer": {"train": int(m_tr.sum()), "val": n_va_bruto,
                                  "test": n_te_bruto},
        "n_nos_apos_buffer": {k: int(len(v)) for k, v in parts.items()},
        "regra_buffer": ("train intacto; val aparado contra train; "
                         "test aparado contra train U val(aparado)"),
        "origem_logica": ("copia de train_gnn_c0_spatial.py:412-487 + "
                          "spatial_cv.SpatialKFold._assign_groups"),
    }
    return parts, info


def free_space_path_loss(distance_m: np.ndarray, frequency_mhz) -> np.ndarray:
    d_km = np.maximum(distance_m / 1000.0, 0.001)
    fspl = 20 * np.log10(d_km) + 20 * np.log10(frequency_mhz) + 32.44
    return fspl


def okumura_hata_urban(distance_m: np.ndarray, frequency_mhz,
                       h_tx: float = 30.0, h_rx: float = 1.5) -> np.ndarray:
    f = frequency_mhz
    d = np.maximum(distance_m / 1000.0, 0.001)
    a_low = 8.29 * (np.log10(1.54 * h_rx)) ** 2 - 1.1
    a_high = 3.2 * (np.log10(11.75 * h_rx)) ** 2 - 4.97
    a_hr = np.where(np.asarray(f) <= 300, a_low, a_high)
    if np.ndim(f) == 0:
        a_hr = float(a_hr)
    L = (69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(h_tx) - a_hr +
         (44.9 - 6.55 * np.log10(h_tx)) * np.log10(d))
    return L


def okumura_hata_rural(distance_m: np.ndarray, frequency_mhz,
                       h_tx: float = 30.0, h_rx: float = 1.5) -> np.ndarray:
    L_urban = okumura_hata_urban(distance_m, frequency_mhz, h_tx, h_rx)
    f = frequency_mhz
    L = L_urban - 4.78 * (np.log10(f)) ** 2 + 18.33 * np.log10(f) - 40.94
    return L


def cost231_hata(distance_m: np.ndarray, frequency_mhz,
                 h_tx: float = 30.0, h_rx: float = 1.5,
                 environment: str = "urban") -> np.ndarray:
    f = frequency_mhz
    d = np.maximum(distance_m / 1000.0, 0.001)
    a_hr = (1.1 * np.log10(f) - 0.7) * h_rx - (1.56 * np.log10(f) - 0.8)
    C_m = 3.0 if environment == "urban" else 0.0
    L = (46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_tx) - a_hr +
         (44.9 - 6.55 * np.log10(h_tx)) * np.log10(d) + C_m)
    return L


MODELOS = [
    ("fspl", lambda d, f: free_space_path_loss(d, f)),
    ("hata_rural", lambda d, f: okumura_hata_rural(d, f)),
    ("cost231_sub", lambda d, f: cost231_hata(d, f, environment="suburban")),
]


def metricas_erro(pred: np.ndarray, alvo: np.ndarray) -> dict:
    err = pred - alvo
    ae = np.abs(err)
    return {
        "n": int(alvo.size),
        "mae_db": float(np.mean(ae)),
        "rmse_db": float(np.sqrt(np.mean(err.astype(np.float64) ** 2))),
        "bias_db": float(np.mean(err)),
        "p90_db": float(np.percentile(ae, 90)) if alvo.size else None,
    }


def metricas_originais(rssi_bl: np.ndarray, rssi_tgt: np.ndarray) -> dict:
    e = np.abs(rssi_bl - rssi_tgt)
    return {
        "mae": float(np.mean(e)),
        "p90": float(np.percentile(e, 90)),
        "bias": float(np.mean(rssi_bl - rssi_tgt)),
    }


def calibrar_ptx_mediana(pl_tr: np.ndarray, rssi_tr: np.ndarray) -> float:
    return float(np.median(rssi_tr + pl_tr))


def calibrar_ptx_clamp(pl_tr: np.ndarray, rssi_tr: np.ndarray, floor: float,
                       log, nome: str) -> dict:
    def J(p: float) -> float:
        return float(np.mean(np.abs(np.maximum(p - pl_tr, floor) - rssi_tr)))

    centro = calibrar_ptx_mediana(pl_tr, rssi_tr)
    lo, hi, passo = centro - 60.0, centro + 30.0, 0.25
    for _ in range(8):
        cand = np.arange(lo, hi + passo, passo)
        vals = np.array([J(float(c)) for c in cand])
        i = int(vals.argmin())
        if 0 < i < len(cand) - 1:
            break
        if i == 0:
            lo -= 60.0
        else:
            hi += 60.0
    p0 = float(cand[i])
    fino = np.arange(p0 - 0.30, p0 + 0.30 + 1e-9, 0.01)
    vfino = np.array([J(float(c)) for c in fino])
    j = int(vfino.argmin())
    p_opt = float(fino[j])
    out = {
        "p_tx_eff_dbm": p_opt,
        "offset_ajuste_db": p_opt,
        "mae_train_no_otimo_db": float(vfino[j]),
        "metodo": ("varredura 1-D do MAE de treino do baseline clampado: "
                   "grossa 0,25 dB em [mediana-60, mediana+30] dBm (com "
                   "extensao de borda), fina 0,01 dB em +/-0,30 dB"),
        "resolucao_db": 0.01,
        "p_tx_eff_mediana_sem_clamp_dbm": centro,
        "mae_train_na_mediana_db": J(centro),
    }
    log(f"    calibracao clamp [{nome}]: offset_ajuste={p_opt:.2f} dB "
        f"(mediana sem clamp daria {centro:.2f}; "
        f"MAE train {out['mae_train_no_otimo_db']:.4f} vs "
        f"{out['mae_train_na_mediana_db']:.4f} dB)")
    return out


def freq_representativa_por_antena(ant_ids: np.ndarray, freqs: np.ndarray,
                                   n_antenna: int) -> tuple[np.ndarray, dict]:
    pares, counts = np.unique(np.stack([ant_ids.astype(np.int64),
                                        freqs.astype(np.float64)], axis=1),
                              axis=0, return_counts=True)
    ordem = np.lexsort((pares[:, 1], -counts, pares[:, 0]))
    pares, counts = pares[ordem], counts[ordem]
    primeiro = np.ones(len(pares), dtype=bool)
    primeiro[1:] = pares[1:, 0] != pares[:-1, 0]
    ant_com_freq = pares[primeiro, 0].astype(np.int64)
    freq_moda = pares[primeiro, 1]

    fq_u, fq_c = np.unique(freqs.astype(np.float64), return_counts=True)
    moda_global = float(fq_u[int(fq_c.argmax())])

    tab = np.full(n_antenna, moda_global, dtype=np.float64)
    tab[ant_com_freq] = freq_moda

    n_freq_por_ant = np.bincount(pares[primeiro.cumsum() - 1, 0].astype(np.int64),
                                 minlength=n_antenna)
    info = {
        "regra": ("frequencia MODAL das arestas ant->ter da antena (valores "
                  "exatos de edge_attr[:,1]*2600; empate -> menor frequencia); "
                  "antena sem aresta -> moda global"),
        "n_antenas": int(n_antenna),
        "n_antenas_sem_aresta": int(n_antenna - len(ant_com_freq)),
        "moda_global_mhz": moda_global,
        "max_freqs_distintas_numa_antena": int(n_freq_por_ant.max()) if n_antenna else 0,
    }
    return tab, info


def freq_por_no_antena_mais_proxima_kdtree(pos_m: np.ndarray, ant_pos_m: np.ndarray,
                                           freq_tab: np.ndarray, log) -> tuple[np.ndarray, np.ndarray]:
    """
    Nearest antenna by Euclidean distance via a cKDTree on the metric
    projection (the same projection used for the spatial split). Equivalent
    to a brute-force chunked Haversine search but roughly two orders of
    magnitude faster.
    """
    from scipy.spatial import cKDTree
    t0 = time.perf_counter()
    tree = cKDTree(ant_pos_m, compact_nodes=False, balanced_tree=False)
    dist_m, arg = tree.query(pos_m, k=1, workers=-1)
    freq_no = freq_tab[arg].astype(np.float32)
    log(f"  freq por no (fallback, KDTree antena mais proxima): "
        f"min={freq_no.min():.2f} max={freq_no.max():.2f} "
        f"media={freq_no.mean():.2f} MHz ({time.perf_counter()-t0:.1f}s)")
    return freq_no, dist_m.astype(np.float32)


def freq_por_no_enlace_mais_curto(edge_dst: np.ndarray, edge_freq_mhz: np.ndarray,
                                  edge_dist_m: np.ndarray, n_ter: int,
                                  freq_fallback: np.ndarray, dist_fallback: np.ndarray,
                                  log) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    freq_no[i] is the frequency of the shortest antenna->terrain edge that
    reaches node i (only edges that exist in the graph); falls back to
    freq_fallback[i] (nearest antenna) when node i has no antenna->terrain
    edge at all.
    Returns (freq_no, dist_no, origem[uint8: 0=fallback, 1=edge], info).
    """
    ordem = np.lexsort((edge_dist_m, edge_dst))
    ed_s = edge_dst[ordem]
    ef_s = edge_freq_mhz[ordem]
    edd_s = edge_dist_m[ordem]
    primeiro = np.ones(ed_s.size, dtype=bool)
    primeiro[1:] = ed_s[1:] != ed_s[:-1]
    nos_com_aresta = ed_s[primeiro]
    freq_enlace = ef_s[primeiro].astype(np.float32)
    dist_enlace = edd_s[primeiro].astype(np.float32)

    freq_no = freq_fallback.astype(np.float32).copy()
    dist_no = dist_fallback.astype(np.float32).copy()
    origem = np.zeros(n_ter, dtype=np.uint8)
    freq_no[nos_com_aresta] = freq_enlace
    dist_no[nos_com_aresta] = dist_enlace
    origem[nos_com_aresta] = 1

    n_com = int(nos_com_aresta.size)
    info = {
        "regra": ("freq_no[i] = frequencia do enlace ant->ter de MENOR "
                  "distancia que chega no no i (entre arestas existentes); "
                  "fallback = frequencia da antena geometricamente mais "
                  "proxima (KDTree) quando o no NAO tem nenhuma aresta"),
        "n_nos_total": int(n_ter),
        "n_nos_enlace_mais_curto": n_com,
        "frac_nos_enlace_mais_curto": float(n_com / n_ter) if n_ter else None,
        "n_nos_fallback": int(n_ter - n_com),
        "frac_nos_fallback": float((n_ter - n_com) / n_ter) if n_ter else None,
    }
    return freq_no, dist_no, origem, info


def resumo_frequencia_origem(origem: np.ndarray, idx: np.ndarray | None = None) -> dict:
    o = origem if idx is None else origem[idx]
    n = int(o.size)
    n_enlace = int((o == 1).sum())
    return {
        "n": n,
        "frac_enlace_mais_curto": float(n_enlace / n) if n else None,
        "frac_fallback_antena_mais_proxima": float((n - n_enlace) / n) if n else None,
    }


def distribuicao_freq(freq_no: np.ndarray, idx: np.ndarray | None = None) -> dict:
    f = freq_no if idx is None else freq_no[idx]
    vals, counts = np.unique(np.round(f.astype(np.float64), 3), return_counts=True)
    ordem = np.argsort(-counts)
    return {
        "min_mhz": float(f.min()) if f.size else None,
        "max_mhz": float(f.max()) if f.size else None,
        "media_mhz": float(f.astype(np.float64).mean()) if f.size else None,
        "mediana_mhz": float(np.median(f)) if f.size else None,
        "n_valores_distintos": int(vals.size),
        "top_valores": [{"freq_mhz": float(vals[i]), "n": int(counts[i]),
                         "frac": float(counts[i] / f.size)}
                        for i in ordem[:12]],
    }


def bloco_baseline(nome, pl, rssi_tgt, cov, ptx, floor=None, pl_tgt=None):
    """
    pl is the analytic path-loss prediction (fn(d, f), unclamped: clamping
    is applied only to the RSSI channel). When pl_tgt (target path_loss_db,
    terrain.y[:, 0]) is given, also records MAE_PL restricted to the valid
    target population (covered nodes).
    """
    rssi_bl = ptx - pl
    if floor is not None:
        rssi_bl = np.maximum(rssi_bl, np.float32(floor))
    b = {
        "todos": metricas_erro(rssi_bl, rssi_tgt),
        "cobertura": metricas_erro(rssi_bl[cov], rssi_tgt[cov]),
        "p_tx_eff_dbm": float(ptx),
        "offset_ajuste_db": float(ptx),
    }
    if floor is None:
        b["metricas_formato_congelado"] = metricas_originais(rssi_bl, rssi_tgt)
    if pl_tgt is not None:
        b["mae_pl_alvo_valido"] = metricas_erro(pl[cov], pl_tgt[cov])
        b["mae_pl_todos_nota"] = ("PL analitico nao e definido fora de "
                                  "alvo_valido pela doutrina da casa; "
                                  "'mae_pl' oficial e SO em cobertura/alvo_valido")
    return b


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="T5: baselines v3 com frequencia do enlace mais curto "
                    "por no (fallback = antena mais proxima), sobre o split "
                    "espacial congelado da campanha c0c1cf.")
    p.add_argument("--cidade", required=True,
                   choices=["bauru", "campinas", "lins", "sorocaba"])
    p.add_argument("--quadrante", required=True,
                   choices=["Q1", "Q2", "Q3", "Q4"])
    p.add_argument("--graph-dir", default=DEFAULT_GRAPH_DIR)
    p.add_argument("--evid-dir", default=DEFAULT_EVID_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--rssi-floor", type=float, default=RSSI_FLOOR_DBM)
    p.add_argument("--skip-dataset-hash", action="store_true")
    p.add_argument("--no-mmap", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Modo de fumaca: usa dados SINTETICOS pequenos em "
                        "vez de ler o .pt de 28 GB (nao grava idx_sha256 "
                        "comparavel a run nenhum; so testa a logica).")
    args = p.parse_args(argv)

    t0 = time.perf_counter()

    def log(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] (+{time.perf_counter()-t0:6.1f}s) {msg}", flush=True)

    evid = Path(args.evid_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baselines_v3_{args.cidade}_{args.quadrante}.json"

    rf_path = Path(args.graph_dir) / (
        f"transfer_dataset_{args.cidade}_v19_{args.quadrante}_enriched_cftudo.pt")
    run_label = f"c0c1cf_{args.cidade}_s42_{args.quadrante}_g10b2"
    run_json_path = evid / "treinos" / run_label / f"run_{run_label}.json"

    script_path = Path(__file__).resolve()
    rec: dict = {
        "artefato_tipo": "baselines_v3_por_particao",
        "teste": "T5_baselines_v3_frequencia_do_enlace",
        "protocolo": str(evid / "dados" / "planos" / "criterios_fechamento_2026-09-13.json"),
        "status": "provisorio",
        "cidade": args.cidade,
        "quadrante": args.quadrante,
        "timestamp_utc_inicio": datetime.now(timezone.utc).isoformat(),
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "nota_offset_ajuste": NOTA_OFFSET_AJUSTE,
        "motivacao": {
            "parecer_fisica_D6": "v3 anterior usava freq da antena mais "
                "proxima sem checar aresta; 81-92% dos nos sem aresta com "
                "essa antena (2026-09-13)",
            "parecer_fisica_B4_eng_ia_BI4": "P_tx_eff produz mW/kW impossiveis "
                "como potencia; renomear para offset_ajuste_db",
            "parecer_eng_ia_BI3": "hata_rural e cost231_sub bit-identicos na "
                "v2 com freq fixa; devem separar quando freq varia por no",
        },
        "fontes_de_logica": {
            "baselines_v2_original": str(evid / V2_SCRIPT_REL),
            "baselines_v2_original_sha256": (sha256_file(evid / V2_SCRIPT_REL)
                                             if (evid / V2_SCRIPT_REL).exists() else None),
            "congelado": str(evid / CONGELADO_REL),
            "congelado_sha256": (sha256_file(evid / CONGELADO_REL)
                                 if (evid / CONGELADO_REL).exists() else None),
            "empirical_models": EMPIRICAL_MODELS_PATH,
            "empirical_models_sha256": (sha256_file(Path(EMPIRICAL_MODELS_PATH))
                                        if Path(EMPIRICAL_MODELS_PATH).exists() else None),
            "spatial_cv": SPATIAL_CV_PATH,
            "spatial_cv_sha256": (sha256_file(Path(SPATIAL_CV_PATH))
                                  if Path(SPATIAL_CV_PATH).exists() else None),
        },
        "run_referencia": {"run_label": run_label, "run_json": str(run_json_path)},
        "ambiente": {"python": sys.version.split()[0],
                     "numpy": np.__version__, "torch": torch.__version__,
                     "executavel": sys.executable},
        "avisos": [],
    }

    def gravar():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False, default=_jsonable)

    def abortar(msg: str, code: int = 2) -> int:
        log(f"ABORTADO: {msg}")
        rec["status"] = "abortado"
        rec["motivo_aborto"] = msg
        gravar()
        return code


    if args.smoke:
        log("MODO FUMACA (dados sinteticos)")
        rng = np.random.RandomState(0)
        n_ter, n_ant = 2000, 30
        pos_m = rng.uniform(0, 20000, size=(n_ter, 2))
        ant_pos_m = rng.uniform(0, 20000, size=(n_ant, 2))
        freq_tab = rng.choice([700.0, 900.0, 1800.0, 2600.0], size=n_ant)
        # edges: only 20% of nodes have an edge, to a random antenna

        n_edges = 500
        edge_dst = rng.choice(n_ter, size=n_edges, replace=False)
        edge_ant = rng.randint(0, n_ant, size=n_edges)
        edge_dist_m = np.linalg.norm(pos_m[edge_dst] - ant_pos_m[edge_ant], axis=1).astype(np.float32)
        edge_freq_mhz = freq_tab[edge_ant].astype(np.float32)

        freq_fb, dist_fb = freq_por_no_antena_mais_proxima_kdtree(pos_m, ant_pos_m, freq_tab, log)
        freq_no, dist_no, origem, info = freq_por_no_enlace_mais_curto(
            edge_dst, edge_freq_mhz, edge_dist_m, n_ter, freq_fb, dist_fb, log)
        assert freq_no.shape == (n_ter,)
        assert origem.dtype == np.uint8
        assert (origem[edge_dst] == 1).all(), "nos com aresta devem ter origem=enlace"
        sem_aresta = np.setdiff1d(np.arange(n_ter), edge_dst)
        assert (origem[sem_aresta] == 0).all(), "nos sem aresta devem ter origem=fallback"
        assert np.array_equal(freq_no[sem_aresta], freq_fb[sem_aresta])


        for no in np.unique(edge_dst):
            m = edge_dst == no
            if m.sum() >= 2:
                i_min = np.argmin(edge_dist_m[m])
                assert freq_no[no] == edge_freq_mhz[m][i_min]
        log(f"FUMACA OK: {info}")
        rec["smoke_info"] = info
        rec["status"] = "smoke_ok"
        rec["timestamp_utc_fim"] = datetime.now(timezone.utc).isoformat()
        gravar()
        return 0


    if not run_json_path.exists():
        return abortar(f"run JSON da campanha nao encontrado: {run_json_path}")
    with open(run_json_path, encoding="utf-8") as f:
        run = json.load(f)

    cfg = run.get("config", {})
    esperado = {"grid_km": GRID_KM, "buffer_km": BUFFER_KM,
                "split_seed": SPLIT_SEED, "split_frac": "0.70,0.15,0.15",
                "max_nodes": 0, "freq_mhz": FREQ_MHZ_CAMPANHA}
    diverg = {k: (cfg.get(k), v) for k, v in esperado.items() if cfg.get(k) != v}
    if diverg:
        return abortar(f"config do run JSON diverge das constantes da campanha: {diverg}")
    if Path(cfg.get("rf_data_file", "")).name != rf_path.name:
        return abortar(f"rf_data_file do run ({cfg.get('rf_data_file')}) != "
                       f"dataset pedido ({rf_path.name})")

    hashes_esperados = {k: run["particoes"][k]["idx_sha256_global"]
                        for k in ("train", "val", "test")}
    n_esperados = {k: run["particoes"][k]["n"] for k in ("train", "val", "test")}
    rec["run_referencia"]["config_conferida"] = esperado
    rec["run_referencia"]["idx_sha256_global_esperado"] = hashes_esperados


    # (900 MHz, same split) to avoid repeating the 1-D calibration sweep.

    # single read of the 28 GB .pt file (I/O is the bottleneck, not the calibration itself).

    v2_json_path = evid / "dados" / "baselines_v2" / f"baselines_v2_{args.cidade}_{args.quadrante}.json"
    v2_rec = None
    if v2_json_path.exists():
        with open(v2_json_path, encoding="utf-8") as f:
            v2_rec = json.load(f)
        if v2_rec.get("status") != "concluido":
            rec["avisos"].append(f"baselines_v2 de referencia nao concluido ({v2_json_path}); "
                                 "v1_original/v2_clamp NAO reproduzidos nesta saida v3")
            v2_rec = None
    else:
        rec["avisos"].append(f"baselines_v2 de referencia ausente ({v2_json_path}); "
                             "v1_original/v2_clamp NAO reproduzidos nesta saida v3")
    rec["baselines_v2_reusado"] = {
        "arquivo": str(v2_json_path), "encontrado": v2_rec is not None,
        "nota": ("offsets de v1_original/v2_clamp COPIADOS do JSON do v2 ja "
                 "congelado, e reaplicados aqui com o pl/pl_tgt recarregados "
                 "desta execucao para calcular MAE_PL (ausente no v2)"),
    }


    if not rf_path.exists():
        return abortar(f"dataset nao encontrado: {rf_path}")
    log(f"Carregando (cpu{'' if args.no_mmap else ', mmap se possivel'}): "
        f"{rf_path} ({rf_path.stat().st_size/1e9:.1f} GB)")
    load_kw = dict(map_location="cpu", weights_only=False)
    rf_data = None
    if not args.no_mmap:
        try:
            rf_data = torch.load(rf_path, mmap=True, **load_kw)
            log("  torch.load(mmap=True) OK")
        except Exception as e:
            log(f"  mmap falhou ({type(e).__name__}: {e}) — carga normal")
    if rf_data is None:
        rf_data = torch.load(rf_path, **load_kw)

    ty = torch.as_tensor(rf_data["terrain"].y).float()
    n_ter = int(ty.shape[0])
    pl_tgt_all = ty[:, 0].numpy().copy()
    rssi_tgt_all = ty[:, 3].numpy().copy()
    del ty

    pos_deg = rf_data["terrain"].pos.float().clone()

    if not hasattr(rf_data["terrain"], "dist_nearest_m"):
        return abortar("dist_nearest_m ausente do rf_data")
    dist_pre = rf_data["terrain"].dist_nearest_m.float()
    if dist_pre.shape[0] != n_ter or float(dist_pre.std()) <= 1.0:
        return abortar("dist_nearest_m reprovado no gate do congelado (:920-925)")
    dist_all = dist_pre.numpy().copy()
    del dist_pre

    ant = rf_data["antenna"]
    if hasattr(ant, "pos") and ant.pos is not None:
        ant_pos_deg = ant.pos.float().numpy().copy()
    else:
        ant_pos_deg = torch.as_tensor(ant.x).float()[:, :2].numpy().copy()
    n_antenna = int(ant_pos_deg.shape[0])

    ei_at = rf_data[ET_AT].edge_index
    ea_at = getattr(rf_data[ET_AT], "edge_attr", None)
    if ea_at is None or ea_at.shape[1] < 2:
        return abortar("edge_attr das arestas antenna->terrain ausente")
    edge_ant = ei_at[0].numpy().astype(np.int64).copy()
    edge_dst = ei_at[1].numpy().astype(np.int64).copy()
    edge_dist_m = (ea_at[:, 0].float() * DIST_EDGE_SCALE).numpy().copy()
    edge_freq_mhz = (ea_at[:, 1].float() * FREQ_EDGE_SCALE).numpy().copy()
    del ei_at, ea_at, ant

    del rf_data
    gc.collect()
    log(f"Tensores extraidos e rf_data liberado: {n_ter:,} nos terrain | "
        f"{n_antenna} antenas | {edge_ant.size:,} arestas ant->ter")

    rec["dataset"] = {
        "rf_data_file": str(rf_path),
        "rf_data_bytes": rf_path.stat().st_size,
        "n_nos_terrain": n_ter,
        "n_antenas": n_antenna,
        "n_arestas_ant_ter": int(edge_ant.size),
    }

    if args.skip_dataset_hash:
        rec["dataset"]["rf_data_sha256"] = run.get("dataset", {}).get("rf_data_sha256")
        rec["dataset"]["rf_data_sha256_fonte"] = "COPIADO de run_*.json (--skip-dataset-hash)"
    else:
        log("sha256 do dataset (28 GB, alguns minutos)...")
        h = sha256_file(rf_path)
        rec["dataset"]["rf_data_sha256"] = h
        rec["dataset"]["rf_data_sha256_fonte"] = "recalculado nesta execucao"
        h_run = run.get("dataset", {}).get("rf_data_sha256")
        rec["dataset"]["rf_data_sha256_bate_com_run"] = (h == h_run)
        if h_run and h != h_run:
            return abortar(f"sha256 do dataset ({h}) != run JSON ({h_run})")
    gravar()


    # Target floor

    cov_all = pl_tgt_all < PL_TARGET_MAX_VALID
    sem_cob = ~cov_all
    rssi_sem = rssi_tgt_all[sem_cob]
    piso = float(args.rssi_floor)
    piso_emp = {
        "piso_usado_dbm": piso,
        "n_nos_sem_cobertura": int(sem_cob.sum()),
        "frac_sem_cobertura": float(sem_cob.mean()),
        "frac_sem_cobertura_igual_piso": (
            float(np.mean(np.abs(rssi_sem - piso) < 1e-3)) if rssi_sem.size else None),
    }
    rec["piso_rssi"] = piso_emp
    log(f"Piso: {piso} dBm | sem cobertura: {piso_emp['n_nos_sem_cobertura']:,} "
        f"({100*piso_emp['frac_sem_cobertura']:.1f}%)")


    pos_m, geo = latlon_graus_para_metros(pos_deg)
    log(f"Extensao: {geo['extensao_x_km']:.2f} x {geo['extensao_y_km']:.2f} km")
    parts, split_info = split_espacial_3vias(
        pos_m, GRID_KM, BUFFER_KM, SPLIT_FRACS, SPLIT_SEED, log)

    rec["geometria"] = geo
    rec["split"] = split_info
    rec["particoes_verificacao"] = {}
    tudo_bate = True
    for k in ("train", "val", "test"):
        h = sha256_idx(parts[k])
        bate = (h == hashes_esperados[k]) and (len(parts[k]) == n_esperados[k])
        tudo_bate &= bate
        rec["particoes_verificacao"][k] = {
            "n": int(len(parts[k])), "n_esperado_run": int(n_esperados[k]),
            "idx_sha256_global": h,
            "idx_sha256_global_esperado_run": hashes_esperados[k],
            "bate_com_run": bool(bate),
        }
        log(f"  {k}: n={len(parts[k]):,} | idx_sha256 "
            f"{'BATE' if bate else 'NAO BATE'} com o run JSON")
    gravar()
    if not tudo_bate:
        return abortar("idx_sha256_global de alguma particao NAO bate com o run "
                       "JSON da campanha — CRITERIO DE ACEITE T5 falhou.")


    lon_min, lat_min = geo["lon_min_deg"], geo["lat_min_deg"]
    ant_pos_m = projetar_com_referencia(ant_pos_deg[:, 0].astype(np.float64),
                                        ant_pos_deg[:, 1].astype(np.float64),
                                        lon_min, lat_min)

    log("Frequencia por antena (moda das arestas)...")
    freq_tab, freq_tab_info = freq_representativa_por_antena(
        edge_ant, edge_freq_mhz, n_antenna)

    log("Fallback: antena mais proxima via KDTree...")
    freq_fb, dist_fb = freq_por_no_antena_mais_proxima_kdtree(
        pos_m, ant_pos_m, freq_tab, log)

    log("Frequencia por no: enlace mais curto (T5) + fallback...")
    freq_no, dist_no, origem, freq_info = freq_por_no_enlace_mais_curto(
        edge_dst, edge_freq_mhz, edge_dist_m, n_ter, freq_fb, dist_fb, log)


    mudou = np.abs(freq_no.astype(np.float64) - freq_fb.astype(np.float64)) > 0.5
    rec["freq_por_no"] = {
        "definicao": freq_info["regra"],
        "tabela_por_antena": freq_tab_info,
        "frac_nos_com_edge_enlace": freq_info["frac_nos_enlace_mais_curto"],
        "frac_nos_fallback": freq_info["frac_nos_fallback"],
        "frac_nos_que_mudaram_de_frequencia_vs_metodo_antigo": float(mudou.mean()),
        "n_nos_que_mudaram_de_frequencia_vs_metodo_antigo": int(mudou.sum()),
        "nota_mudanca": ("metodo antigo = freq da antena geometricamente "
                         "mais proxima para TODO no (v3 anterior, diagnostico "
                         "diag_enlace do v2); a mudanca so pode ocorrer nos "
                         "nos com aresta, pois nos sem aresta usam o mesmo "
                         "fallback nos dois metodos"),
        "distribuicao_frequencias_mhz": distribuicao_freq(freq_no),
        "frequencia_origem": resumo_frequencia_origem(origem),
    }
    del pos_m
    gc.collect()


    tr = parts["train"]
    d_tr = dist_all[tr]
    r_tr = rssi_tgt_all[tr]
    f_tr = freq_no[tr]

    log("Calibracoes no treino (v3_clamp_freq_enlace)...")
    calib = {"v3_clamp_freq_enlace": {}}
    calib_extra = {"v3_clamp_freq_enlace": {}}
    for nome, fn in MODELOS:
        pl_tr_f = fn(d_tr, f_tr)
        cf = calibrar_ptx_clamp(pl_tr_f, r_tr, piso, log, f"v3enlace:{nome}")
        calib["v3_clamp_freq_enlace"][nome] = cf["p_tx_eff_dbm"]
        calib_extra["v3_clamp_freq_enlace"][nome] = cf
        del pl_tr_f
    del d_tr, r_tr, f_tr
    gc.collect()

    log("Metricas por particao...")
    rec["particoes"] = {}
    for k in ("train", "val", "test"):
        idx = parts[k]
        d = dist_all[idx]
        r = rssi_tgt_all[idx]
        f = freq_no[idx]
        cov = cov_all[idx]
        bloco = {
            "n": int(idx.size),
            "n_cobertura": int(cov.sum()),
            "frac_cobertura": float(cov.mean()),
            "frequencia_origem": resumo_frequencia_origem(origem, idx),
            "distribuicao_frequencias_mhz": distribuicao_freq(freq_no, idx),
            "preditor_constante_piso": {
                "valor_dbm": piso,
                "todos": metricas_erro(np.full_like(r, np.float32(piso)), r),
                "cobertura": metricas_erro(
                    np.full(int(cov.sum()), np.float32(piso), dtype=np.float32),
                    r[cov]),
            },
            "variantes": {},
        }
        pl_tgt_part = pl_tgt_all[idx]

        # v1_original and v2_clamp use a fixed 900 MHz frequency (same as v2), offset


        if v2_rec is not None:
            for var_nome, floor_v in (("v1_original", None), ("v2_clamp", piso)):
                vb_fix = {}
                for nome, fn in MODELOS:
                    ptx = v2_rec["calibracao"]["p_tx_eff_dbm"][var_nome][nome]
                    pl = fn(d, FREQ_MHZ_CAMPANHA)
                    vb_fix[nome] = bloco_baseline(nome, pl, r, cov, ptx, floor_v,
                                                  pl_tgt=pl_tgt_part)
                    del pl
                vb_fix["freq_mhz"] = FREQ_MHZ_CAMPANHA
                vb_fix["clamp"] = (None if floor_v is None
                                   else f"rssi_bl = max(offset_ajuste - PL, {floor_v} dBm)")
                dhf = abs(vb_fix["hata_rural"]["todos"]["mae_db"] -
                         vb_fix["cost231_sub"]["todos"]["mae_db"])
                vb_fix["hata_cost231_diff_mae_todos_db"] = float(dhf)
                vb_fix["hata_cost231_identicos"] = bool(dhf <= HATA_COST231_TOL_DB)
                vb_fix["offset_reusado_de"] = str(v2_json_path)
                bloco["variantes"][var_nome] = vb_fix

        vb = {}
        for nome, fn in MODELOS:
            pl = fn(d, f)
            vb[nome] = bloco_baseline(nome, pl, r, cov,
                                      calib["v3_clamp_freq_enlace"][nome], piso,
                                      pl_tgt=pl_tgt_part)
            del pl
        vb["freq_mhz"] = "por_no (enlace mais curto; fallback antena mais proxima)"
        vb["clamp"] = f"rssi_bl = max(offset_ajuste - PL, {piso} dBm)"
        dh = abs(vb["hata_rural"]["todos"]["mae_db"] - vb["cost231_sub"]["todos"]["mae_db"])
        vb["hata_cost231_diff_mae_todos_db"] = float(dh)
        vb["hata_cost231_identicos"] = bool(dh <= HATA_COST231_TOL_DB)
        bloco["variantes"]["v3_clamp_freq_enlace"] = vb
        rec["particoes"][k] = bloco
        log(f"  {k}: MAE todos (fspl/hata/cost231) = "
            f"{vb['fspl']['todos']['mae_db']:.4f} / "
            f"{vb['hata_rural']['todos']['mae_db']:.4f} / "
            f"{vb['cost231_sub']['todos']['mae_db']:.4f} dB | "
            f"hata==cost231: {vb['hata_cost231_identicos']} (diff={dh:.6f}) | "
            f"const -110 = {bloco['preditor_constante_piso']['todos']['mae_db']:.4f} dB")
        del d, r, f, cov
        gc.collect()

    rec["calibracao"] = {
        "regra": ("[identico ao v2] TODA calibracao usa exclusivamente a "
                  "particao de TREINO e e aplicada inalterada a val e test"),
        "offset_ajuste_db": calib,
        "p_tx_eff_dbm_compat": calib,
        "detalhe_clamp": calib_extra,
    }
    gravar()


    # Acceptance criterion: Hata Rural != COST-231 Suburban where frequency varies

    rec["criterio_aceite_T5"] = {
        "idx_sha256_global_bate_16_16": True,
        "hata_cost231_identicos_test": rec["particoes"]["test"]["variantes"][
            "v3_clamp_freq_enlace"]["hata_cost231_identicos"],
        "hata_cost231_diff_mae_test_db": rec["particoes"]["test"]["variantes"][
            "v3_clamp_freq_enlace"]["hata_cost231_diff_mae_todos_db"],
        "tolerancia_db": HATA_COST231_TOL_DB,
    }

    rec["_definicoes"] = {
        "mae_db": "mean(|pred - alvo|), RSSI em dBm, sobre a populacao indicada",
        "todos": "todos os nos da particao",
        "cobertura": f"subconjunto com rf_targets[:,0] < {PL_TARGET_MAX_VALID} dB",
        "preditor_constante_piso": "pred == -110 dBm em todo no",
        "v3_clamp_freq_enlace": ("rssi_bl = max(offset_ajuste_db - PL(dist_nearest_m, "
                                 "freq_no), piso); freq_no = freq do enlace mais curto "
                                 "do proprio no, fallback = freq da antena mais proxima"),
        "offset_ajuste_db": NOTA_OFFSET_AJUSTE,
        "idx_sha256_global": "sha256 dos indices int64 ordenados da particao",
    }
    rec["decisoes_registradas"] = [
        "frequencia por no = freq do enlace ant->ter de MENOR distancia que "
        "chega no no (T5); fallback = antena geometricamente mais proxima "
        "(KDTree na projecao metrica do split) so quando o no nao tem "
        "nenhuma aresta",
        "offset_ajuste_db substitui p_tx_eff_dbm como nome de referencia; "
        "p_tx_eff_dbm mantido no JSON so por compatibilidade com o v2",
        "distancia dos baselines = dist_nearest_m embutido no .pt (a MESMA "
        "da campanha e do v2), nunca recalculada",
        "calibracao sempre sobre TODOS os nos do treino, como no v2",
    ]
    rec["timestamp_utc_fim"] = datetime.now(timezone.utc).isoformat()
    rec["duracao_s"] = round(time.perf_counter() - t0, 1)
    rec["status"] = "concluido"
    rec["concluido_em"] = rec["timestamp_utc_fim"]
    gravar()
    log(f"JSON gravado: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
