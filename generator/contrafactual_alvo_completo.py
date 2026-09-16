"""Recomputes the reference field with all three known defects corrected at once, for the decisive self-consistency test."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

EVID = Path(__file__).resolve().parents[1]
OUT_DIR = EVID / "dados" / "alvo"
GD = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\graph_data")
V3 = Path(r"F:\TOPO_RF_DOWNLOAD_DRIVE\graph_data_v3")
PROJ = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2")
sys.path.insert(0, str(PROJ / "data_raw"))
sys.path.insert(0, str(PROJ))

SENTINELA = 300.0
SEED = 42
FREQ_MHZ = 1800.0
IDX_SLOPE, IDX_TPI_COL5, IDX_TRI_COL6, IDX_ROUGH_COL7 = 1, 5, 6, 7


def a_terrain(slope: np.ndarray) -> np.ndarray:
    """Copia fiel de generate_realistic_coverage.py:107-116."""
    tf = slope * 20.0
    tf = np.where(slope > 0.1, tf + (slope - 0.1) * 30.0, tf)
    return np.clip(tf, 0, 30)


def resumo(a: np.ndarray) -> dict:
    return {"min": float(a.min()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "media": float(a.mean()), "std": float(a.std())}


def carimbo(c: str, q: str):


    for d in (V3, GD, Path(r"D:\ARPIA_RF\GRAPH_V19")):
        p = d / f"{c}_v19_{q}_gpu.pt"
        if p.exists():
            sd = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            nm = getattr(sd, "normalization", None)
            out = ((float(nm["elev_std"]), float(nm["elev_mean"]))
                   if nm and "elev_std" in nm and "elev_mean" in nm else (None, None))
            del sd
            return out
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="lins")
    ap.add_argument("--quad", default="Q1")
    ap.add_argument("--src", default="",
                    help="caminho do tile a corrigir; se omitido, procura nos "
                         "diretorios conhecidos")
    ap.add_argument("--dst", default="",
                    help="caminho de saida; se omitido, acrescenta _cftudo ao nome")
    args = ap.parse_args()
    c, q = args.city, args.quad

    import enrich_rf_targets as E


    if args.src:
        src = Path(args.src)
    else:
        candidatos = [GD / f"transfer_dataset_{c}_v19_{q}_enriched_v2.pt",
                      V3 / f"transfer_dataset_{c}_v19_{q}_enriched.pt",
                      GD / f"transfer_dataset_{c}_v19_{q}_enriched.pt"]
        src = next((p for p in candidatos if p.exists()), candidatos[0])
    dst = (Path(args.dst) if args.dst
           else src.with_name(src.stem + "_cftudo" + src.suffix))
    if not src.exists():
        print(f"ABORTA: {src} ausente")
        return 1
    es, em = carimbo(c, q)
    if es is None:
        print("ABORTA: carimbo elev_std/elev_mean ausente")
        return 1

    print(f"lendo {src.name} | elev_std={es}", flush=True)
    d = torch.load(src, map_location="cpu", weights_only=False)
    t = d["terrain"]
    fr = t.features_raw
    x = (fr.numpy() if torch.is_tensor(fr) else np.asarray(fr)).astype(np.float32)
    y = (t.y.numpy() if torch.is_tensor(t.y) else np.asarray(t.y)).astype(np.float32).copy()
    dist = t.dist_nearest_m.float()
    n = x.shape[0]
    rec = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "tile": f"{c}_{q}", "origem": str(src), "saida": str(dst),
           "elev_std": es, "correcoes": {}}


    np.random.seed(SEED)
    slope_sorteado = np.random.uniform(0.02, 0.15, n).astype(np.float32)
    slope_real = np.clip(x[:, IDX_SLOPE].astype(np.float64), 0, None)
    at_old, at_new = a_terrain(slope_sorteado.astype(np.float64)), a_terrain(slope_real)
    delta = at_new - at_old
    com_cob = y[:, 0] != SENTINELA
    y[com_cob, 0] = (y[com_cob, 0].astype(np.float64) + delta[com_cob]).astype(np.float32)
    if y.shape[1] > 3:
        y[com_cob, 3] = (y[com_cob, 3].astype(np.float64) - delta[com_cob]).astype(np.float32)
    rec["correcoes"]["1_A_terrain"] = {
        "sorteado": resumo(at_old), "com_slope_real": resumo(at_new),
        "delta_abs_medio_db": float(np.abs(delta[com_cob]).mean()),
        "delta_abs_max_db": float(np.abs(delta[com_cob]).max()),
        "n_nos_com_cobertura": int(com_cob.sum())}
    print(f"  1) A_terrain: {at_old.mean():.4f} -> {at_new.mean():.4f} dB "
          f"(delta medio {np.abs(delta[com_cob]).mean():.4f})", flush=True)

    # defects 2 and 3: correct column and correct unit in the diffraction term
    col2_antes = y[:, 2].copy()
    tri_errado_m = torch.from_numpy((x[:, IDX_TPI_COL5].astype(np.float32) * es))
    rough_norm = torch.from_numpy(x[:, IDX_TRI_COL6].astype(np.float32))
    col2_reproduzida = E.compute_diffraction_loss(tri_errado_m, rough_norm, dist, FREQ_MHZ).numpy()

    tri_certo_m = torch.from_numpy((x[:, IDX_TRI_COL6].astype(np.float32) * es))
    rough_m = torch.from_numpy((x[:, IDX_ROUGH_COL7].astype(np.float32) * es))
    col2_corrigida = E.compute_diffraction_loss(tri_certo_m, rough_m, dist, FREQ_MHZ).numpy()

    rec["correcoes"]["2e3_difracao"] = {
        "col2_publicada": resumo(col2_antes.astype(np.float64)),
        "col2_reproduzida_com_o_codigo_atual": resumo(col2_reproduzida.astype(np.float64)),
        "reproducao_bate_com_a_publicada": bool(np.array_equal(col2_reproduzida, col2_antes)),
        "col2_corrigida": resumo(col2_corrigida.astype(np.float64)),
        "delta_abs_medio_db": float(np.abs(col2_corrigida - col2_antes).mean()),
        "delta_abs_max_db": float(np.abs(col2_corrigida - col2_antes).max()),
        "frac_no_piso_antes": float(np.mean(col2_antes <= 6.0206004)),
        "frac_no_piso_depois": float(np.mean(col2_corrigida <= 6.0206004))}
    y[:, 2] = col2_corrigida.astype(np.float32)
    print(f"  2+3) difracao: p50 {np.percentile(col2_antes,50):.4f} -> "
          f"{np.percentile(col2_corrigida,50):.4f} dB | no piso "
          f"{np.mean(col2_antes<=6.0206004):.2%} -> "
          f"{np.mean(col2_corrigida<=6.0206004):.2%}", flush=True)
    print(f"       (reproducao do codigo atual bate com a publicada: "
          f"{rec['correcoes']['2e3_difracao']['reproducao_bate_com_a_publicada']})",
          flush=True)

    t.y = torch.from_numpy(y)
    print(f"\ngravando {dst.name} ...", flush=True)
    torch.save(d, dst)
    rec["saida_bytes"] = dst.stat().st_size
    rec["sha256_gerador"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dest = OUT_DIR / f"contrafactual_alvo_completo_{c}_{q}.json"
    dest.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"gravado: {dst} ({dst.stat().st_size/1e9:.2f} GB)")
    print(f"registro: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
