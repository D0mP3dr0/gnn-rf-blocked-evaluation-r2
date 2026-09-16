

"""Frozen graph-free control training script (capacity-matched MLP, no message passing)."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      ReduceLROnPlateau)


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# Constants inherited from the original training script.

COL_ELEV, COL_SLOPE, COL_ROUGH, COL_NDVI, COL_NDWI = 0, 1, 6, 12, 13

RSSI_THRESHOLD_DBM = -100.0
RSSI_THRESHOLD_CONF_DBM = -85.0


PL_TARGET_MAX_VALID = 299.0


ORIGEM_PATH = (r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access"
               r"\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO"
               r"\dados\scripts_congelados\train_gnn_c0_spatial.py")
ORIGEM_SHA256 = "6f955629cde164f2843f454e1ebf6977292fd80647ef48ecd0df31e464c42445"


N_PARAMS_ALVO_GNN = 1_812_515
MLP_LARGURAS = (712, 712, 720, 688, 256)

DEFAULT_BASE_DIR = r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2"
DEFAULT_EVID_DIR = (r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access"
                    r"\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO\treinos")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("E1/B4: controle MLP SEM grafo na regua da campanha c0c1cf "
                     "(split espacial bloqueado, mesma loss, mesma selecao)."))

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=24576)
    p.add_argument("--k-antenna", type=int, default=20,
                   help="[M7] SEM EFEITO no MLP (fan-out do NeighborLoader); mantido por paridade de CLI.")
    p.add_argument("--k-terrain", type=int, default=8,
                   help="[M7] SEM EFEITO no MLP; mantido por paridade de CLI.")
    p.add_argument("--hidden-dim", type=int, default=256,
                   help="[M7] SEM EFEITO no MLP: larguras fixas pela paridade [M1] "
                        f"{MLP_LARGURAS}; mantido por paridade de CLI.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--version", type=int, default=21)
    p.add_argument("--run-label", type=str, default="mlpcf_smoke")
    p.add_argument("--variance-weight", type=float, default=0.02)
    p.add_argument("--ndvi-weight", type=float, default=0.03)
    p.add_argument("--dist-grad-weight", type=float, default=0.05)
    p.add_argument("--max-norm", type=float, default=0.5)
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["plateau", "cosine", "cosine_simple"])
    p.add_argument("--scheduler-patience", type=int, default=2)
    p.add_argument("--cosine-t0", type=int, default=10)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--transfer-from", type=str, default="")
    p.add_argument("--checkpoint-interval", type=int, default=0)
    p.add_argument("--p-tx-dbm", type=float, default=43.0)
    p.add_argument("--graph-file", type=str, default="bauru_v19_Q1_gpu.pt",
                   help="[M2] carregado SO se features_raw faltar no rf_data "
                        "(features por no); arestas ter->ter NUNCA sao lidas.")
    p.add_argument("--graph-dir", type=str, default="")
    p.add_argument("--rf-data-file", type=str,
                   default="transfer_dataset_bauru_v19_Q1_enriched.pt")
    p.add_argument("--num-workers", type=int, default=0,
                   help="[M7] SEM EFEITO no MLP (nao ha NeighborLoader).")
    p.add_argument("--prefetch", type=int, default=2,
                   help="[M7] SEM EFEITO no MLP.")
    p.add_argument("--seed", type=int, default=42)


    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR,
                   help="[C0-3] Raiz do GNN_RF_V2 usada SO PARA LEITURA (modulos + graph_data).")
    p.add_argument("--evid-dir", type=str, default=DEFAULT_EVID_DIR,
                   help="[C0-3] Raiz de saida (EVIDENCIA_RESUBMISSAO/treinos).")
    p.add_argument("--freq-mhz", type=float, default=900.0,
                   help="[C0-4] Frequencia (MHz) do termo FSPL da LOSS e dos baselines analiticos.")
    p.add_argument("--diag-freq-mhz", type=float, default=1800.0,
                   help="[C0-4] Frequencia (MHz) do FSPL de RFDiagnosticMetrics (physics_fspl_compliance).")


    p.add_argument("--grid-km", type=float, default=5.0,
                   help="[C1-5] Lado do bloco espacial (km). Default do spatial_cv.py.")
    p.add_argument("--buffer-km", type=float, default=2.0,
                   help="[C1-5] Buffer entre particoes (km). Default do spatial_cv.py.")
    p.add_argument("--split-frac", type=str, default="0.70,0.15,0.15",
                   help="[C1-5] Fracao de BLOCOS train,val,test.")
    p.add_argument("--split-seed", type=int, default=42,
                   help="[C1-5] Seed do embaralhamento de blocos (independente de --seed).")
    p.add_argument("--test-every", type=int, default=1,
                   help="[D4] Avaliar teste a cada N epocas (0 = so no final). Nunca usado em selecao.")
    p.add_argument("--eval-batch-size", type=int, default=0,
                   help="Batch de inferencia (0 = igual a --batch-size).")


    p.add_argument("--max-nodes", type=int, default=0,
                   help="Subamostra espacialmente CONTIGUA de ate N nos (0 = todos).")
    p.add_argument("--window-anchor", type=str, default="cobertura",
                   choices=["cobertura", "mediana"],
                   help="Centro da janela de --max-nodes: 'cobertura' = mediana dos nos "
                        "com alvo de PL valido (evita janela sem antena); 'mediana' = "
                        "mediana de todos os nos (comportamento ingenuo).")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 2 epocas, --max-nodes 200000 se nao dado, workers=0, verificacoes extras.")
    p.add_argument("--selftest", action="store_true",
                   help="Roda so os testes unitarios do termo de distancia (C0-1) e sai.")
    p.add_argument("--hash", action="store_true",
                   help="Calcular sha256 do dataset (lento: ~28 GB).")
    p.add_argument("--mmap", action="store_true",
                   help="torch.load(mmap=True) — util no smoke para nao materializar 28 GB.")
    p.add_argument("--no-baselines", action="store_true")
    return p.parse_args(argv)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_idx(idx: np.ndarray) -> str:
    """Hash canonico de um conjunto de indices: int64, ordenado, little-endian."""
    a = np.ascontiguousarray(np.sort(np.asarray(idx, dtype=np.int64)))
    return hashlib.sha256(a.tobytes()).hexdigest()


def log_vram() -> str:
    if torch.cuda.is_available():
        return (f"VRAM {torch.cuda.memory_allocated()/1024**3:.2f}GB/"
                f"{torch.cuda.memory_reserved()/1024**3:.2f}GB")
    return "CPU"


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


def latlon_graus_para_metros(pos: torch.Tensor) -> tuple[np.ndarray, dict]:
    """
    pos: (N,2) float, col0 = lon (graus), col1 = lat (graus).
    Retorna (N,2) float64 em METROS, origem no canto (lon_min, lat_min).

        dy = (lat - lat_min) * 111000
        dx = (lon - lon_min) * 111000 * cos(lat_do_no)
    """
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


def janela_contigua(pos_m: np.ndarray, max_nodes: int,
                    ancora: np.ndarray | None = None) -> np.ndarray:
    """
    Spatially contiguous subsample: a square window.

    `ancora` (bool, N) sets the center. Without an anchor, the center is
    the median of all nodes, which can fall in an area with no
    antenna->terrain edges when antenna coverage is sparse and
    concentrated (e.g. a city quadrant with coverage over roughly 16% of
    nodes): the run would still execute but would not exercise the
    antenna message-passing path or the path-loss metric (target all
    sentinel). Anchoring on the median of covered nodes avoids this
    degenerate window.
    """
    n = pos_m.shape[0]
    if max_nodes <= 0 or n <= max_nodes:
        return np.arange(n, dtype=np.int64)
    x, y = pos_m[:, 0], pos_m[:, 1]
    if ancora is not None and bool(ancora.any()):
        cx, cy = float(np.median(x[ancora])), float(np.median(y[ancora]))
    else:
        cx, cy = float(np.median(x)), float(np.median(y))
    lo, hi = 0.0, float(max(x.max() - x.min(), y.max() - y.min()))
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        cnt = int(((np.abs(x - cx) <= mid) & (np.abs(y - cy) <= mid)).sum())
        if cnt > max_nodes:
            hi = mid
        else:
            lo = mid
    sel = np.where((np.abs(x - cx) <= lo) & (np.abs(y - cy) <= lo))[0]
    return sel.astype(np.int64)


# Three-way split by blocks, with a three-way buffer.

def split_espacial_3vias(pos_m: np.ndarray, grid_km: float, buffer_km: float,
                         fracs: tuple[float, float, float], split_seed: int,
                         SpatialKFold, log) -> tuple[dict, dict]:
    """
    Usa SpatialKFold._assign_groups (03_training/spatial_cv.py:94-103) para
    atribuir blocos; divide BLOCOS em 3 grupos; apara val contra train e test
    contra (train U val aparado).
    """
    from scipy.spatial import cKDTree

    pos_km = pos_m / 1000.0
    skf = SpatialKFold(n_splits=3, buffer_km=buffer_km,
                       grid_size_km=grid_km, random_state=split_seed)
    group_ids = skf._assign_groups(pos_km)
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
    log(f"  Nos por bloco (antes do buffer): train {int(m_tr.sum()):,} | "
        f"val {n_va_bruto:,} | test {n_te_bruto:,}")

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
        "n_blocos": {"train": int(len(g_tr)), "val": int(len(g_va)), "test": int(len(g_te))},
        "fracs_blocos_pedidas": list(fracs),
        "split_seed": split_seed,
        "n_nos_antes_do_buffer": {"train": int(m_tr.sum()), "val": n_va_bruto, "test": n_te_bruto},
        "n_nos_apos_buffer": {k: int(len(v)) for k, v in parts.items()},
        "retencao_apos_buffer": {
            "val": (len(parts["val"]) / n_va_bruto) if n_va_bruto else None,
            "test": (len(parts["test"]) / n_te_bruto) if n_te_bruto else None},
        "n_descartados_pelo_buffer": int(n_va_bruto - len(parts["val"])
                                         + n_te_bruto - len(parts["test"])),
        "regra_buffer": ("train intacto; val aparado contra train; "
                         "test aparado contra train U val(aparado)"),
    }
    return parts, info


def verificar_split(parts: dict, pos_m: np.ndarray, buffer_km: float,
                    validate_spatial_cv, log) -> dict:
    """
    Exact check (cKDTree over all points), plus reuse of
    validate_spatial_cv on a subsample.
    """
    from scipy.spatial import cKDTree
    pos_km = pos_m / 1000.0
    out = {"buffer_km_exigido": buffer_km, "pares": {}, "intersecoes": {}, "ok": True}

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ia, ib = parts[a], parts[b]
        inter = int(np.intersect1d(ia, ib, assume_unique=True).size)
        out["intersecoes"][f"{a}|{b}"] = inter
        if inter:
            out["ok"] = False
        if len(ia) and len(ib):
            tree = cKDTree(pos_km[ia], compact_nodes=False, balanced_tree=False)
            d, _ = tree.query(pos_km[ib], k=1, workers=-1)
            dmin = float(d.min())
        else:
            dmin = float("nan")
        out["pares"][f"{a}|{b}"] = {"dist_min_km": dmin,
                                    "respeita_buffer": bool(dmin >= buffer_km)}
        if not (dmin >= buffer_km):
            out["ok"] = False
        log(f"  [SPLIT] {a} x {b}: intersecao={inter} | dist_min={dmin:.3f} km "
            f"({'OK' if dmin >= buffer_km else 'VIOLADO'})")


    rs = np.random.RandomState(0)
    sub_tr = parts["train"][rs.choice(len(parts["train"]),
                                      min(20000, len(parts["train"])), replace=False)]
    out["validate_spatial_cv_casa"] = {}
    for b in ("val", "test"):
        if len(parts[b]) == 0:
            out["validate_spatial_cv_casa"][b] = None
            continue
        sub_b = parts[b][rs.choice(len(parts[b]), min(20000, len(parts[b])), replace=False)]
        ok = bool(validate_spatial_cv(sub_tr, sub_b, pos_m, buffer_km=buffer_km, verbose=False))
        out["validate_spatial_cv_casa"][b] = ok
        out["ok"] = out["ok"] and ok
        log(f"  [SPLIT] validate_spatial_cv(train~{len(sub_tr)}, {b}~{len(sub_b)}) = {ok}")
    out["nota_validate_spatial_cv"] = ("executada sobre subamostra de <=20k por conjunto "
                                       "(cdist e O(n*m)); a verificacao exaustiva e a "
                                       "cKDTree acima")
    return out


ET_AT = ("antenna", "propagates_to", "terrain")
ET_TA = ("terrain", "in_range_of", "antenna")
ET_TT = ("terrain", "connects_to", "terrain")


def contar_arestas_ant_ter(ei_at_dst_global: torch.Tensor, idx_global: np.ndarray,
                           n_ter_total: int) -> int:
    """
    Number of antenna->terrain edges whose terrain node (global endpoint)
    belongs to the partition. Matches n_arestas_ant_ter of the induced
    subgraph (all antennas are kept; only the terrain endpoint is
    filtered).
    """
    mask = torch.zeros(n_ter_total, dtype=torch.bool)
    mask[torch.from_numpy(np.asarray(idx_global, dtype=np.int64))] = True
    total = 0
    for i in range(0, ei_at_dst_global.numel(), 50_000_000):
        total += int(mask[ei_at_dst_global[i:i + 50_000_000]].sum())
    return total


class MLPRFModel(nn.Module):
    """
    Graph-free control: without edges or message passing.

    encoder: 18 -> 712 -> 712 -> 720 -> 688 -> 256, each layer Linear +
             BatchNorm1d + LeakyReLU(0.2) (+ Dropout 0.1, except the
             last), an extended version of the standard terrain encoder
             block.
    decoder: PhysicsConstrainedDecoder(in=256, hidden=256, out=5,
             dropout=0.1), the same class used by GNNRFModel, with
             identical clamps.

    Trainable parameter count: 1,812,515 (exact parity with the GNN
    model).
    """

    def __init__(self, PhysicsConstrainedDecoder, terrain_dim: int,
                 larguras=MLP_LARGURAS, dropout: float = 0.1):
        super().__init__()
        dims = [terrain_dim] + list(larguras)
        camadas = []
        for i, (m, n) in enumerate(zip(dims, dims[1:])):
            camadas += [nn.Linear(m, n), nn.BatchNorm1d(n), nn.LeakyReLU(0.2)]
            if i < len(dims) - 2:
                camadas.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*camadas)
        self.decoder = PhysicsConstrainedDecoder(
            in_channels=dims[-1], hidden_channels=256,
            out_channels=5, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict:
        emb = self.encoder(x)
        preds = self.decoder(emb)
        # Same output interface as GNNRFModel (ensures comparable predictions).
        return {"terrain_embeddings": emb, "predictions": preds}


class _DiagCache:
    """
    Factory for RFDiagnosticMetrics restricted to a node subset, so that
    N == n_mask inside diag.compute() and the linspace-based resampling
    never triggers.
    """

    def __init__(self, RFDiagnosticMetrics, pos, feats, dist, antenna_pos, freq_mhz):
        self._cls = RFDiagnosticMetrics
        self._pos, self._feats, self._dist = pos, feats, dist
        self._ant = antenna_pos
        self._freq = freq_mhz
        self._cache = {}

    def get(self, key: str, sub: torch.Tensor | None):
        if key in self._cache:
            return self._cache[key]
        if sub is None:
            obj = self._cls(terrain_pos=self._pos, antenna_pos=self._ant,
                            terrain_features=self._feats,
                            frequency_mhz=self._freq, dist_nearest_m=self._dist)
        else:
            obj = self._cls(terrain_pos=self._pos[sub], antenna_pos=self._ant,
                            terrain_features=self._feats[sub],
                            frequency_mhz=self._freq, dist_nearest_m=self._dist[sub])
        self._cache[key] = obj
        return obj


def _scatter_por_semente(seed_ids: torch.Tensor, preds: torch.Tensor,
                         tgts: torch.Tensor, n_nodes: int):
    """
    Scatters per-seed predictions into a node-ordered array.
    Returns (preds_ord, tgts_ord, visto) with preds_ord[i] holding the
    prediction for node i.
    """
    preds_ord = torch.zeros((n_nodes, preds.shape[1]), dtype=torch.float32)
    tgts_ord = torch.zeros((n_nodes, tgts.shape[1]), dtype=torch.float32)
    visto = torch.zeros(n_nodes, dtype=torch.bool)
    preds_ord[seed_ids] = preds.float()
    tgts_ord[seed_ids] = tgts.float()
    visto[seed_ids] = True
    return preds_ord, tgts_ord, visto


def metricas_particao(preds_ord, tgts_ord, visto, diag_cache, chave, pl_valid_full):
    """
    MAE/RMSE of RSSI (all seen nodes) and of PL restricted to targets
    < 299 dB, plus the full RFDiagnosticMetrics row aligned node-by-node.
    """
    n_nodes = visto.numel()
    cobertura = float(visto.float().mean())
    if cobertura >= 1.0:
        sub = None
        key = f"{chave}:full"
        p, t = preds_ord, tgts_ord
    else:
        sub = torch.where(visto)[0]
        key = f"{chave}:{int(sub.numel())}:{int(sub[0])}:{int(sub[-1])}"
        p, t = preds_ord[sub], tgts_ord[sub]

    err_rssi = (p[:, 3] - t[:, 3])
    pl_valid = pl_valid_full if sub is None else pl_valid_full[sub]
    err_pl = (p[:, 0] - t[:, 0])[pl_valid]

    m = {
        "n_nos": int(n_nodes),
        "n_nos_avaliados": int(p.shape[0]),
        "cobertura_epoca": cobertura,
        "mae_rssi_db": float(err_rssi.abs().mean()),
        "rmse_rssi_db": float(torch.sqrt((err_rssi ** 2).mean())),
        "bias_rssi_db": float(err_rssi.mean()),
        "p90_rssi_db": float(np.percentile(err_rssi.abs().numpy(), 90)),
        "n_pl_alvo_valido": int(pl_valid.sum()),
        "definicao_pl_valido": f"rf_targets[:,0] < {PL_TARGET_MAX_VALID} dB",
    }
    if err_pl.numel() > 0:
        m["mae_pl_db"] = float(err_pl.abs().mean())
        m["rmse_pl_db"] = float(torch.sqrt((err_pl ** 2).mean()))
        m["bias_pl_db"] = float(err_pl.mean())
        m["p90_pl_db"] = float(np.percentile(err_pl.abs().numpy(), 90))
    else:
        m.update({"mae_pl_db": None, "rmse_pl_db": None,
                  "bias_pl_db": None, "p90_pl_db": None})

    diag = diag_cache.get(key, sub)
    m["diag"] = {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                     else float(v)) for k, v in diag.compute(p, t).items()}
    m["physics_distance_gradient"] = m["diag"].get("physics_distance_gradient")
    return m


def baselines_por_particao(base_dir: Path, dist_m: np.ndarray, rssi_tgt: np.ndarray,
                           p_tx_eff: float | None, freq_mhz: float, log):
    sys.path.insert(0, str(base_dir / "04_baselines"))
    from empirical_models import (cost231_hata, free_space_path_loss,
                                  okumura_hata_rural)
    out, calib = {}, {}
    for name, fn, kw in [("fspl", free_space_path_loss, {}),
                         ("hata_rural", okumura_hata_rural, {}),
                         ("cost231_sub", cost231_hata, {"environment": "suburban"})]:
        pl = fn(dist_m, freq_mhz, **kw)
        if p_tx_eff is None:
            ptx = float(np.median(rssi_tgt + pl))
        else:
            ptx = float(p_tx_eff[name])
        calib[name] = ptx
        rssi_bl = ptx - pl
        e = np.abs(rssi_bl - rssi_tgt)
        out[f"baseline_{name}_mae"] = float(np.mean(e))
        out[f"baseline_{name}_p90"] = float(np.percentile(e, 90))
        out[f"baseline_{name}_bias"] = float(np.mean(rssi_bl - rssi_tgt))
    out["p_tx_eff_dbm"] = calib
    out["freq_mhz"] = freq_mhz
    return out, calib


def selftest_distancia(base_dir: Path) -> bool:
    sys.path.append(str(base_dir / "02_models"))
    from physics_loss import CurriculumRFLoss

    print("=" * 72)
    print("SELFTEST C0-1 — termo de distancia do FSPL")
    print("=" * 72)
    ok = True

    # Scale: the ETL normalizes by 30000.
    d_real = np.array([300.0, 1200.0, 9000.0, 27000.0], dtype=np.float32)
    ea0 = d_real / 30_000.0
    d_antigo = ea0 * 15_000.0
    d_novo = ea0 * 30_000.0
    t1 = np.allclose(d_novo, d_real) and np.allclose(d_antigo, d_real / 2)
    vies_db = 20 * np.log10(0.5)
    print(f"T1 escala   : formula antiga devolve d/2 -> vies FSPL "
          f"{vies_db:.2f} dB | {'PASS' if t1 else 'FAIL'}")
    ok &= t1


    n_seed, n_edges = 4, 9
    torch.manual_seed(0)
    ea = torch.rand(n_edges, 2)
    edge_dst = torch.tensor([3, 1, 0, 2, 1, 3, 0, 2, 1])
    dist_por_no = torch.tensor([10.0, 20.0, 30.0, 40.0])
    fatia_arestas = ea[:n_seed, 0]
    t2 = not torch.allclose(fatia_arestas, dist_por_no / dist_por_no.max())

    agreg = torch.full((n_seed,), float("inf"))
    for e in range(n_edges):
        agreg[edge_dst[e]] = torch.minimum(agreg[edge_dst[e]], ea[e, 0])
    t2 = t2 and not torch.allclose(agreg, fatia_arestas)
    print(f"T2 indexacao: ea[:n_seed,0] != distancia por no  | {'PASS' if t2 else 'FAIL'}")
    ok &= t2


    loss_fn = CurriculumRFLoss(frequency_mhz=900.0, distance_gradient_weight=0.0,
                               variance_weight=0.0, shadowing_ndvi_weight=0.0)
    loss_fn.set_phase(5)
    n = 256
    preds = torch.zeros(n, 5)
    preds[:, 0] = 60.0
    tgts = torch.zeros(n, 5)
    vals = []
    for d in [100.0, 1000.0, 10000.0, 30000.0]:
        dd = torch.full((n,), d)
        _, ld = loss_fn(preds, tgts, distances=dd)
        vals.append(float(ld["constraint_fspl"]))
    t3 = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    print(f"T3 monotonia: constraint_fspl por d(100/1k/10k/30k m) = "
          f"{[round(v,3) for v in vals]} | {'PASS' if t3 else 'FAIL'}")
    ok &= t3


    t4 = (max(vals) - min(vals)) > 1.0
    print(f"T4 termo vivo: amplitude {max(vals)-min(vals):.3f} dB | "
          f"{'PASS' if t4 else 'FAIL'}")
    ok &= t4

    print("=" * 72)
    print("SELFTEST: " + ("PASS" if ok else "FAIL"))
    print("=" * 72)
    return bool(ok)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] (+{time.perf_counter()-t_start:6.1f}s) {msg}", flush=True)

    base_dir = Path(args.base_dir)
    if args.selftest:
        return 0 if selftest_distancia(base_dir) else 1

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        if args.max_nodes <= 0:
            args.max_nodes = 200_000
        args.num_workers = 0
        args.batch_size = min(args.batch_size, 8192)

    eval_bs = args.eval_batch_size or args.batch_size
    fracs = tuple(float(v) for v in args.split_frac.split(","))
    assert len(fracs) == 3, "--split-frac precisa de 3 valores"


    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 72)
    print(f"E1/B4 — controle MLP SEM grafo, regua c0c1cf [{args.run_label}]")
    print("=" * 72)
    log(f"base_dir (LEITURA)  : {base_dir}")
    out_dir = Path(args.evid_dir) / args.run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"run_{args.run_label}.json"
    csv_path = out_dir / "training_log.csv"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log(f"saida (ESCRITA)     : {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log(f"Device: cuda | {torch.cuda.get_device_name(0)} | "
            f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    else:
        log("Device: cpu (GPU indisponivel)")

    sys.path.append(str(base_dir / "02_models"))
    sys.path.append(str(base_dir / "03_training"))


    from rf_decoder import PhysicsConstrainedDecoder
    from physics_loss import CurriculumRFLoss
    from rf_diagnostic_metrics import RFDiagnosticMetrics
    from spatial_cv import SpatialKFold, validate_spatial_cv

    rec: dict = {
        "artefato_tipo": "run_treino_c0c1_mlp",
        "controle": ("E1/B4 do PARECER_ENG_IA_C1V2_TREINO16.md: MLP sem estrutura "
                     "de grafo, mesma regua da campanha c0c1cf"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "origem_script": ORIGEM_PATH,
        "origem_script_sha256": ORIGEM_SHA256,
        "roadmap_itens": ["C0", "C1", "E1/B4 (controle MLP do parecer eng-IA)"],
        "pontos_revisor": "R1-1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "smoke": bool(args.smoke),
        "status": "provisorio",
        "config": vars(args).copy(),
        "ambiente": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda_disponivel": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "cudnn_deterministic": True, "cudnn_benchmark": False,
            "executavel": sys.executable,
        },
    }

    def gravar():
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False, default=_jsonable)

    gravar()


    graph_dir = Path(args.graph_dir).resolve() if args.graph_dir else (base_dir / "graph_data")
    rf_path = graph_dir / args.rf_data_file
    struct_path = graph_dir / args.graph_file
    log(f"Carregando RF targets: {rf_path}  ({rf_path.stat().st_size/1e9:.1f} GB)")
    load_kw = dict(map_location="cpu", weights_only=False)
    if args.mmap:
        try:
            rf_data = torch.load(rf_path, mmap=True, **load_kw)
            log("  torch.load(mmap=True) OK")
        except Exception as e:
            log(f"  mmap falhou ({type(e).__name__}: {e}) — carga normal")
            rf_data = torch.load(rf_path, **load_kw)
    else:
        rf_data = torch.load(rf_path, **load_kw)

    rec["dataset"] = {
        "rf_data_file": str(rf_path), "rf_data_bytes": rf_path.stat().st_size,
        "graph_file": str(struct_path),
        "graph_bytes": struct_path.stat().st_size if struct_path.exists() else None,
    }
    if args.hash:
        log("Calculando sha256 do dataset (lento)...")
        rec["dataset"]["rf_data_sha256"] = sha256_file(rf_path)
    gravar()

    ty = rf_data["terrain"].y
    n_ter_total = int(ty.shape[0])
    log(f"Nos terrain no dataset: {n_ter_total:,}")

    dist_pre = None
    if hasattr(rf_data["terrain"], "dist_nearest_m"):
        c = rf_data["terrain"].dist_nearest_m.float()
        if c.shape[0] == n_ter_total and float(c.std()) > 1.0:
            dist_pre = c
            log(f"  dist_nearest_m embutido: mean={float(c.mean()):.0f}m "
                f"std={float(c.std()):.0f}m")


    if getattr(rf_data["terrain"], "features_raw", None) is not None:
        feats = rf_data["terrain"].features_raw
        feats = torch.from_numpy(feats) if isinstance(feats, np.ndarray) else feats
        log(f"Features embutidas no rf_data: {tuple(feats.shape)}")
    else:
        log(f"Carregando estrutura DEM para features: {struct_path.name}")
        sd = torch.load(struct_path, **load_kw)
        feats = torch.as_tensor(sd["dem"].x)
        del sd
        gc.collect()
    feats = feats.float()

    pos_deg = rf_data["terrain"].pos.float()
    tgt_all = torch.as_tensor(rf_data["terrain"].y).float()
    ant_x = torch.as_tensor(rf_data["antenna"].x).float()
    n_antenna = int(ant_x.shape[0])
    ant_pos = (rf_data["antenna"].pos.float()
               if hasattr(rf_data["antenna"], "pos") else ant_x[:, :2])

    ei_at_full = rf_data[ET_AT].edge_index.long()
    ea_at_full = (rf_data[ET_AT].edge_attr.float()
                  if hasattr(rf_data[ET_AT], "edge_attr") else None)


    if ea_at_full is not None and ea_at_full.shape[1] >= 2:
        fq = (ea_at_full[:, 1] * 2600.0)
        uq, cnt = torch.unique(torch.round(fq), return_counts=True)
        rec.setdefault("dataset", {})["freq_antenas_mhz"] = {
            "fonte": "edge_attr[:,1] * 2600 (prepare_transfer_dataset_v19.py:256-257)",
            "min": float(fq.min()), "max": float(fq.max()), "media": float(fq.mean()),
            "valores_mais_comuns": [[float(u), int(c)] for u, c in
                                    zip(uq[cnt.argsort(descending=True)][:6],
                                        cnt.sort(descending=True).values[:6])],
        }


    # Avoids loading the ~15 GB gpu.pt file just for this.


    ei_at_dst_global = ei_at_full[1].clone()
    del ei_at_full, ea_at_full
    log("  [M2] ter->ter ignorado; gpu.pt NAO carregado para arestas")

    del rf_data
    gc.collect()
    log(f"Dados por no: {n_ter_total:,} terrain | {n_antenna} antenas | "
        f"{ei_at_dst_global.numel():,} arestas ant->ter (so p/ guarda [M3], "
        f"fora do modelo)")


    pos_m_full, geo = latlon_graus_para_metros(pos_deg)
    log(f"Extensao: {geo['extensao_x_km']:.2f} x {geo['extensao_y_km']:.2f} km "
        f"(lon {geo['lon_min_deg']:.4f}..{geo['lon_max_deg']:.4f}, "
        f"lat {geo['lat_min_deg']:.4f}..{geo['lat_max_deg']:.4f})")


    ancora = None
    if args.max_nodes > 0 and args.window_anchor == "cobertura":
        ancora = (tgt_all[:, 0] < PL_TARGET_MAX_VALID).numpy()
        log(f"[JANELA] ancora=cobertura: {int(ancora.sum()):,} de {n_ter_total:,} nos "
            f"com alvo de PL valido ({100*ancora.mean():.1f}%)")
    g_idx = janela_contigua(pos_m_full, args.max_nodes, ancora)
    subamostrado = int(g_idx.size) != n_ter_total
    if subamostrado:
        log(f"[JANELA] subamostra contigua: {g_idx.size:,} de {n_ter_total:,} nos")
    g_idx_t = torch.from_numpy(g_idx)


    # Crop/relabel step; the guard uses the global g_idx[loc].
    pos_win = pos_deg[g_idx_t]
    tgt_win = tgt_all[g_idx_t]
    x_base = feats[g_idx_t]                                  # 17 columns (excludes the distance feature)
    del feats, tgt_all, pos_deg
    gc.collect()

    n_ter = int(x_base.shape[0])
    pos_m = pos_m_full[g_idx]
    del pos_m_full
    dist_all = (dist_pre[g_idx_t] if dist_pre is not None else None)
    if dist_all is None:
        log("dist_nearest_m ausente — calculando Haversine (RFDiagnosticMetrics)")
        tmp = RFDiagnosticMetrics(terrain_pos=pos_win, antenna_pos=ant_pos,
                                  terrain_features=x_base,
                                  frequency_mhz=args.diag_freq_mhz)
        dist_all = tmp.dist_nearest.clone()
        del tmp
    log(f"dist_to_antenna: min={float(dist_all.min()):.0f}m "
        f"max={float(dist_all.max()):.0f}m mean={float(dist_all.mean()):.0f}m")


    grid_km, buffer_km, ajustado = args.grid_km, args.buffer_km, None
    ext = min((pos_m[:, 0].max() - pos_m[:, 0].min()) / 1000.0,
              (pos_m[:, 1].max() - pos_m[:, 1].min()) / 1000.0)
    if args.smoke and ext / max(grid_km, 1e-9) < 6.0:
        novo = float(ext / 6.0)
        ajustado = {"grid_km_original": grid_km, "buffer_km_original": buffer_km,
                    "grid_km_usado": novo, "buffer_km_usado": novo * (buffer_km / grid_km),
                    "motivo": ("janela do smoke tem menos de 6 blocos por lado; "
                               "grid e buffer reduzidos na mesma proporcao 5:2. "
                               "GEOMETRIA DE SMOKE — NAO E A DE PRODUCAO")}
        buffer_km = novo * (buffer_km / grid_km)
        grid_km = novo
        log(f"[SMOKE] grid ajustado {args.grid_km} -> {grid_km:.3f} km | "
            f"buffer {args.buffer_km} -> {buffer_km:.3f} km (nao e geometria de producao)")

    log("Split espacial por blocos (logica de spatial_cv.SpatialKFold)...")
    parts_local, split_info = split_espacial_3vias(
        pos_m, grid_km, buffer_km, fracs, args.split_seed, SpatialKFold, log)
    for k, v in parts_local.items():
        if len(v) == 0:
            raise RuntimeError(f"Particao '{k}' ficou VAZIA apos o buffer — "
                               f"reduza --buffer-km ou aumente --grid-km.")

    verif = verificar_split(parts_local, pos_m, buffer_km, validate_spatial_cv, log)

    rec["geometria"] = dict(geo)
    rec["geometria"].update({"grid_km_usado": grid_km, "buffer_km_usado": buffer_km,
                             "grid_km_ajustado_por_smoke": ajustado,
                             "n_nos_no_grafo_de_trabalho": n_ter,
                             "subamostrado": subamostrado,
                             "n_nos_dataset_completo": n_ter_total})
    rec["split"] = split_info
    rec["split"]["verificacao"] = verif
    rec["particoes"] = {}
    pl_valid = {}
    for k, loc in parts_local.items():
        glob = g_idx[loc]
        pv = tgt_win[torch.from_numpy(loc), 0] < PL_TARGET_MAX_VALID
        pl_valid[k] = pv
        bb = pos_m[loc]
        rec["particoes"][k] = {
            "n": int(loc.size),
            "frac_dos_nos": float(loc.size / n_ter),
            "idx_sha256_global": sha256_idx(glob),
            "idx_sha256_local_janela": sha256_idx(loc),
            "n_pl_alvo_valido": int(pv.sum()),
            "frac_pl_alvo_valido": float(pv.float().mean()),
            "bbox_km": [float(bb[:, 0].min() / 1000), float(bb[:, 1].min() / 1000),
                        float(bb[:, 0].max() / 1000), float(bb[:, 1].max() / 1000)],
            "nota_indices": ("idx_sha256_global = sha256 dos indices int64 ORDENADOS "
                             "no dataset original; idx_sha256_local_janela idem, "
                             "relativos a subamostra contigua"),
        }
        log(f"  {k}: {loc.size:,} nos ({100*loc.size/n_ter:.1f}%) | "
            f"PL alvo valido: {int(pv.sum()):,} ({100*float(pv.float().mean()):.1f}%)")
    gravar()

    if not verif["ok"]:
        raise RuntimeError("Verificacao do split espacial FALHOU — ver split.verificacao")


    # Feature col 17 (distance) normalized using train-only statistics (avoids leakage).

    tr_loc = torch.from_numpy(parts_local["train"])
    # distance (meters) reindexed to the local node numbering of the training subgraph:


    dist_train_local = dist_all[tr_loc].contiguous()
    d_mean_tr = float(dist_all[tr_loc].mean())
    d_std_tr = max(float(dist_all[tr_loc].std()), 1.0)
    d_mean_all = float(dist_all.mean())
    d_std_all = max(float(dist_all.std()), 1.0)
    x_full = torch.cat([x_base, ((dist_all - d_mean_tr) / d_std_tr).unsqueeze(1)], dim=1)
    rec["normalizacao"] = {
        "feature_col17": "dist_to_nearest_antenna",
        "dist_mean_train": d_mean_tr, "dist_std_train": d_std_tr,
        "dist_mean_todos": d_mean_all, "dist_std_todos": d_std_all,
        "decisao": ("[D1] media/desvio do TREINO (original train_lins_physics.py:569-571 "
                    "usava todos os nos = estatistica de teste na padronizacao)"),
    }
    log(f"[D1] col17 normalizada com estatistica de treino "
        f"(mean={d_mean_tr:.0f}m std={d_std_tr:.0f}m; todos: {d_mean_all:.0f}/{d_std_all:.0f})")


    log("Contando arestas ant->ter por particao (guarda [M3], fora do modelo)...")
    x_parts, tgt_parts, sub_info = {}, {}, {}
    for k in ("train", "val", "test"):
        loc_t = torch.from_numpy(parts_local[k])
        x_parts[k] = x_full[loc_t]
        tgt_parts[k] = tgt_win[loc_t]
        n_at = contar_arestas_ant_ter(ei_at_dst_global, g_idx[parts_local[k]],
                                      n_ter_total)
        sub_info[k] = {"n_terrain": int(loc_t.numel()), "n_antenna": int(n_antenna),
                       "n_arestas_ant_ter": n_at,
                       "n_arestas_ant_ter_base": int(ei_at_dst_global.numel())}
        log(f"  [{k}] {int(loc_t.numel()):,} nos terrain | ant->ter na particao "
            f"{n_at:,} (contagem p/ guarda; NENHUMA aresta alimenta o MLP)")
    # Guard: a partition with no antenna-to-terrain edge
    # and zero valid PL targets would only measure the -110 dBm sentinel.
    degen = [k for k in ("train", "val", "test")
             if sub_info[k]["n_arestas_ant_ter"] == 0]
    rec["guarda_particao_degenerada"] = {
        "particoes_sem_aresta_antena": degen,
        "n_pl_alvo_valido": {k: rec["particoes"][k]["n_pl_alvo_valido"]
                             for k in ("train", "val", "test")},
        "significado": ("particao com 0 arestas ant->ter e 0 alvo de PL valido nao "
                        "testa o caminho da antena; MAE de RSSI ali e regressao "
                        "sobre o sentinela constante de -110 dBm"),
        "nota_mlp": ("[M3] contagem por pertencimento do endpoint terrain global; "
                     "identica a do subgrafo induzido do congelado"),
    }
    if degen:
        msg = (f"PARTICAO DEGENERADA: {degen} sem nenhuma aresta antena->terreno. "
               f"Use --window-anchor cobertura ou aumente --max-nodes.")
        log(f"  ERRO: {msg}")
        gravar()
        raise RuntimeError(msg)

    rec["estrutura_sem_grafo"] = sub_info
    rec["estrutura_sem_grafo"]["declaracao"] = (
        "[M2] Controle E1/B4: o modelo NAO recebe nenhuma aresta (nem ant->ter "
        "nem ter->ter) e nao ha passagem de mensagem. Cada predicao usa apenas "
        "as 18 features do proprio no. As contagens de arestas acima existem so "
        "para a guarda de particao degenerada e para auditoria.")
    del ei_at_dst_global
    gc.collect()
    gravar()


    # Same seed semantics:
    # every node in the partition is sampled as a seed exactly once per epoch.

    def lotes(k: str, bs: int, shuffle: bool):
        n = x_parts[k].shape[0]
        ordem = torch.randperm(n) if shuffle else torch.arange(n)

        # Train mode requires batch > 1; every node remains a seed.
        cortes = list(range(0, n, bs))
        for j, i in enumerate(cortes):
            fim = i + bs
            if j == len(cortes) - 2 and n - cortes[-1] == 1:
                fim = n
            elif j == len(cortes) - 1 and n - i == 1 and len(cortes) > 1:
                return
            yield ordem[i:fim]

    def n_lotes(k: str, bs: int) -> int:
        return (x_parts[k].shape[0] + bs - 1) // bs

    n_batches = n_lotes("train", args.batch_size)
    log(f"Lotes [M4]: train {n_batches} batches (bs={args.batch_size}) | "
        f"val {n_lotes('val', eval_bs)} | test {n_lotes('test', eval_bs)} "
        f"(bs={eval_bs}) — sem NeighborLoader, sem vizinhanca")


    diag_caches = {}
    for k in ("train", "val", "test"):
        loc = torch.from_numpy(parts_local[k])
        diag_caches[k] = _DiagCache(RFDiagnosticMetrics,
                                    pos=pos_win[loc],
                                    feats=x_base[loc],
                                    dist=dist_all[loc],
                                    antenna_pos=ant_pos,
                                    freq_mhz=args.diag_freq_mhz)


    if not args.no_baselines:
        try:
            log(f"Baselines analiticos (freq={args.freq_mhz} MHz, calibracao SO no treino)...")
            bl = {}
            d_tr = dist_all[tr_loc].numpy()
            r_tr = tgt_parts["train"][:, 3].numpy()
            bl["train"], calib = baselines_por_particao(
                base_dir, d_tr, r_tr, None, args.freq_mhz, log)
            for k in ("val", "test"):
                loc = torch.from_numpy(parts_local[k])
                bl[k], _ = baselines_por_particao(
                    base_dir, dist_all[loc].numpy(),
                    tgt_parts[k][:, 3].numpy(),
                    calib, args.freq_mhz, log)
            bl["decisao"] = ("[D2] P_tx_eff calibrado SO com alvos de treino e aplicado "
                             "inalterado a val/test (original calibrava no proprio "
                             "conjunto avaliado: train_lins_physics.py:186-193)")
            rec["baselines_analiticos"] = bl
            for k in ("train", "val", "test"):
                log(f"  {k}: FSPL {bl[k]['baseline_fspl_mae']:.2f} | "
                    f"Hata {bl[k]['baseline_hata_rural_mae']:.2f} | "
                    f"COST231 {bl[k]['baseline_cost231_sub_mae']:.2f} dB")
        except Exception as e:
            log(f"  AVISO: baselines falharam ({type(e).__name__}: {e})")
            rec["baselines_analiticos"] = {"erro": f"{type(e).__name__}: {e}"}
    gravar()


    if int(x_full.shape[1]) != 18:
        raise RuntimeError(
            f"terrain_dim={int(x_full.shape[1])} != 18: dataset fora da regua da "
            f"campanha c0c1cf; a paridade de parametros [M1] nao se aplica.")
    model = MLPRFModel(PhysicsConstrainedDecoder, terrain_dim=x_full.shape[1],
                       larguras=MLP_LARGURAS, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Modelo: {n_params:,} parametros ({n_params_tr:,} treinaveis) | {log_vram()}")
    log(f"[M1] paridade de capacidade: MLP {n_params_tr:,} vs GNN "
        f"{N_PARAMS_ALVO_GNN:,} (delta {n_params_tr - N_PARAMS_ALVO_GNN:+d})")
    if n_params_tr != N_PARAMS_ALVO_GNN:
        raise RuntimeError(
            f"PARIDADE VIOLADA: n_params_treinaveis={n_params_tr:,} != "
            f"{N_PARAMS_ALVO_GNN:,} (GNN c0c1cf). Larguras {MLP_LARGURAS} "
            f"foram calculadas para paridade EXATA — investigar antes de treinar.")
    rec["modelo"] = {"classe": "MLPRFModel", "decoder": "PhysicsConstrainedDecoder",
                     "larguras_encoder": list(MLP_LARGURAS),
                     "output_dim": 5, "dropout": 0.1,
                     "terrain_dim": int(x_full.shape[1]),
                     "antenna_dim_nao_usado": int(ant_x.shape[1]),
                     "sem_arestas": True, "sem_passagem_de_mensagem": True,
                     "n_params": int(n_params), "n_params_treinaveis": int(n_params_tr),
                     "n_params_alvo_gnn": int(N_PARAMS_ALVO_GNN),
                     "paridade_delta": int(n_params_tr - N_PARAMS_ALVO_GNN)}

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_t0,
                                                T_mult=2, eta_min=1e-6)
    elif args.scheduler == "cosine_simple":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.cosine_t0, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                      patience=args.scheduler_patience, min_lr=1e-6)


    loss_fn = CurriculumRFLoss(
        frequency_mhz=args.freq_mhz,
        distance_gradient_weight=args.dist_grad_weight,
        distance_gradient_n_pairs=512,
        variance_weight=args.variance_weight,
        shadowing_ndvi_weight=args.ndvi_weight,
        shadowing_ndvi_min_corr=0.15,
    ).to(device)
    rec["loss"] = {
        "classe": "CurriculumRFLoss",
        "frequency_mhz_loss_fspl": args.freq_mhz,
        "frequency_mhz_diagnostico": args.diag_freq_mhz,
        "nota_frequencia": ("[C0-4] original nao passava frequency_mhz -> default 900 MHz "
                            "(physics_loss.py:36) na loss, enquanto RFDiagnosticMetrics "
                            "auditava com 1800 MHz (rf_diagnostic_metrics.py:38)"),
        "distance_gradient_weight": args.dist_grad_weight,
        "variance_weight": args.variance_weight,
        "shadowing_ndvi_weight": args.ndvi_weight,
        "fonte_de_distances": ("[C0-1] dist_to_nearest_antenna em metros, alinhada no-a-no "
                               "(antes: edge_attr[:n_seed,0]*15000)"),
    }

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.transfer_from and Path(args.transfer_from).exists():
        ck = torch.load(args.transfer_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        log(f"[TRANSFER] pesos carregados de {args.transfer_from}")
        rec["transfer_from"] = args.transfer_from


    def rodar_avaliacao(k: str) -> dict:


        model.eval()
        n_p = x_parts[k].shape[0]
        P, T, I = [], [], []
        with torch.no_grad():
            for idx in lotes(k, eval_bs, shuffle=False):
                xb = x_parts[k][idx].to(device)
                ctx = torch.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
                with ctx:
                    out = model(xb)
                P.append(out["predictions"].float().detach().cpu())
                T.append(tgt_parts[k][idx].float())
                I.append(idx)
        po, to, vi = _scatter_por_semente(torch.cat(I), torch.cat(P), torch.cat(T), n_p)
        return metricas_particao(po, to, vi, diag_caches[k], k, pl_valid[k])

    rec["epocas"] = []
    best_val, best_epoch, sem_melhora = float("inf"), -1, 0
    csv_fields = ["epoch", "train_loss", "lr", "elapsed_s",
                  "train_mae_rssi", "train_rmse_rssi", "train_mae_pl",
                  "val_mae_rssi", "val_rmse_rssi", "val_mae_pl",
                  "test_mae_rssi", "test_rmse_rssi", "test_mae_pl",
                  "train_dist_grad", "val_dist_grad", "test_dist_grad"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore").writeheader()

    log(f"Iniciando treino: {args.epochs} epocas x {n_batches} batches")
    print("-" * 72, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        total_loss, nb = 0.0, 0
        P, T, I = [], [], []
        comp_dist_gradiente_ok = True

        for bi, seed_ids in enumerate(lotes("train", args.batch_size, shuffle=True), 1):


            xb = x_parts["train"][seed_ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                out = model(xb)
                preds_s = out["predictions"]
                targets_s = tgt_parts["train"][seed_ids].to(device)


                dist_b = dist_train_local[seed_ids].to(device)

                ndvi_b = (xb[:, COL_NDVI]
                          if xb.shape[1] > COL_NDVI else None)

                loss, ldict = loss_fn(preds_s, targets_s,
                                      distances=dist_b,
                                      dist_to_ant=dist_b,
                                      ndvi=ndvi_b)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss nao-finita na epoca {epoch}, batch {bi}")
            scaler.scale(loss).backward()
            if args.max_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            nb += 1
            P.append(preds_s.float().detach().cpu())
            T.append(targets_s.float().detach().cpu())
            I.append(seed_ids)
            if bi % max(1, n_batches // 5) == 0 or bi == n_batches:
                log(f"  Epoch {epoch}/{args.epochs} | batch {bi}/{n_batches} | "
                    f"loss={total_loss/nb:.3f} | {log_vram()}")

        avg_loss = total_loss / max(nb, 1)
        if args.scheduler in ("cosine", "cosine_simple"):
            scheduler.step()
        else:
            scheduler.step(avg_loss)

        n_tr_nodes = x_parts["train"].shape[0]
        po, to, vi = _scatter_por_semente(torch.cat(I), torch.cat(P), torch.cat(T), n_tr_nodes)
        m_train = metricas_particao(po, to, vi, diag_caches["train"], "train", pl_valid["train"])
        del P, T, I, po, to, vi
        gc.collect()

        m_val = rodar_avaliacao("val")
        m_test = (rodar_avaliacao("test")
                  if (args.test_every > 0 and epoch % args.test_every == 0) else None)

        elapsed = time.perf_counter() - t_ep
        lr_now = optimizer.param_groups[0]["lr"]
        linha = {"epoch": epoch, "train_loss": round(avg_loss, 4),
                 "lr": lr_now, "elapsed_s": round(elapsed, 1),
                 "loss_componentes": {k: float(v) for k, v in ldict.items()
                                      if torch.is_tensor(v) and v.numel() == 1},
                 "train": m_train, "val": m_val, "test": m_test,
                 "test_usado_na_selecao": False}
        if device == "cuda":
            linha["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        rec["epocas"].append(linha)

        log(f"Epoch {epoch:02d} | loss={avg_loss:.3f} | "
            f"MAE_RSSI train={m_train['mae_rssi_db']:.3f} val={m_val['mae_rssi_db']:.3f}"
            + (f" test={m_test['mae_rssi_db']:.3f}" if m_test else "") + " dB")
        log(f"          MAE_PL(alvo<299) train={m_train['mae_pl_db']} "
            f"val={m_val['mae_pl_db']}" + (f" test={m_test['mae_pl_db']}" if m_test else ""))
        log(f"          dist_gradient train={m_train['physics_distance_gradient']} "
            f"val={m_val['physics_distance_gradient']} | cobertura da epoca "
            f"train={m_train['cobertura_epoca']:.3f} | {elapsed:.0f}s")

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore").writerow({
                "epoch": epoch, "train_loss": round(avg_loss, 4), "lr": lr_now,
                "elapsed_s": round(elapsed, 1),
                "train_mae_rssi": m_train["mae_rssi_db"], "train_rmse_rssi": m_train["rmse_rssi_db"],
                "train_mae_pl": m_train["mae_pl_db"],
                "val_mae_rssi": m_val["mae_rssi_db"], "val_rmse_rssi": m_val["rmse_rssi_db"],
                "val_mae_pl": m_val["mae_pl_db"],
                "test_mae_rssi": (m_test or {}).get("mae_rssi_db"),
                "test_rmse_rssi": (m_test or {}).get("rmse_rssi_db"),
                "test_mae_pl": (m_test or {}).get("mae_pl_db"),
                "train_dist_grad": m_train["physics_distance_gradient"],
                "val_dist_grad": m_val["physics_distance_gradient"],
                "test_dist_grad": (m_test or {}).get("physics_distance_gradient")})

        # Checkpoint selection uses validation only (explicit guard against test leakage).
        criterio = _selecionar_melhor_epoca(m_val)
        if criterio < best_val:
            best_val, best_epoch, sem_melhora = criterio, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_mae_rssi": criterio, "args": vars(args),
                        "selecao": "val mae_rssi_db (test nunca usado)"},
                       ckpt_dir / "checkpoint_best.pt")
            log(f"  CKPT best (val MAE_RSSI={criterio:.4f} dB)")
        else:
            sem_melhora += 1

        gravar()
        if args.early_stopping_patience and sem_melhora >= args.early_stopping_patience:
            log(f"  EARLY STOP na epoca {epoch}")
            break
        print("-" * 72, flush=True)


    ck = ckpt_dir / "checkpoint_best.pt"
    m_test_final = None
    if ck.exists():
        st = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(st["model_state_dict"])
        log(f"[FINAL] recarregado checkpoint best (epoca {st['epoch']}) — avaliando test")
        m_test_final = rodar_avaliacao("test")
        m_val_final = rodar_avaliacao("val")
        log(f"[FINAL] test  MAE_RSSI={m_test_final['mae_rssi_db']:.4f} dB | "
            f"RMSE={m_test_final['rmse_rssi_db']:.4f} | MAE_PL={m_test_final['mae_pl_db']}")
    else:
        m_val_final = None

    rec["selecao"] = {
        "criterio": "menor mae_rssi_db em VAL",
        "melhor_epoca": best_epoch,
        "melhor_val_mae_rssi_db": (None if best_val == float("inf") else best_val),
        "test_usado_na_selecao": False,
        "checkpoint": str(ck),
        "test_no_melhor_ckpt": m_test_final,
        "val_no_melhor_ckpt": m_val_final,
        "nota": ("[D4] o teste por epoca em `epocas[].test` e diagnostico; o numero "
                 "reportavel e `selecao.test_no_melhor_ckpt`, medido uma unica vez "
                 "sobre o checkpoint escolhido por val."),
    }
    rec["custo"] = {
        "tempo_total_s": round(time.perf_counter() - t_start, 1),
        "n_epocas_rodadas": len(rec["epocas"]),
        "vram_peak_mb": (round(torch.cuda.max_memory_allocated() / 1e6, 1)
                         if device == "cuda" else None),
    }
    rec["_fontes"] = {
        "n_nos_dataset": "rf_data['terrain'].y.shape[0] (torch.load do rf-data-file)",
        "particoes.*.n": "len(np.where(mask)[0]) apos buffer — split_espacial_3vias()",
        "particoes.*.idx_sha256_global": "sha256(int64 ordenados) — sha256_idx()",
        "split.verificacao.pares.*.dist_min_km": "scipy.spatial.cKDTree.query(k=1) exato",
        "epocas.*.<part>.mae_rssi_db": "metricas_particao() sobre array ordenado por no",
        "epocas.*.<part>.mae_pl_db": f"idem, restrito a rf_targets[:,0] < {PL_TARGET_MAX_VALID}",
        "epocas.*.<part>.physics_distance_gradient": ("RFDiagnosticMetrics.compute() com "
                                                      "N == n_mask (sem reamostragem)"),
        "selecao.test_no_melhor_ckpt": "rodar_avaliacao('test') apos recarregar checkpoint_best.pt",
        "modelo.n_params_treinaveis": "sum(p.numel() for p in model.parameters() if p.requires_grad)",
        "modelo.paridade_delta": "n_params_treinaveis - 1.812.515 (GNN c0c1cf); 0 por construcao [M1]",
        "dataset.freq_antenas_mhz": "edge_attr[:,1] * 2600 do proprio .pt",
        "estrutura_sem_grafo.*.n_arestas_ant_ter": ("[M3] contagem por pertencimento do "
                                                    "endpoint terrain global a particao; "
                                                    "nenhuma aresta alimenta o modelo"),
    }
    gravar()
    log(f"JSON gravado: {json_path}")
    log(f"CSV  gravado: {csv_path}")
    print("=" * 72)
    return 0


def _selecionar_melhor_epoca(m_val: dict) -> float:
    """
    Explicit guard: checkpoint selection reads only validation metrics.
    Passing a dict that contains a test key raises AssertionError.
    """
    assert "test" not in m_val, "selecao nunca pode ver o conjunto de teste"
    return float(m_val["mae_rssi_db"])


if __name__ == "__main__":
    sys.exit(main())
