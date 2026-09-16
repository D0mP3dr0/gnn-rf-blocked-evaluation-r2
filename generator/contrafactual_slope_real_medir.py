"""Corrector step 1: checks that the real DEM slope fits the scale the target generator expects."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

EVID = Path(__file__).resolve().parents[1]
OUT_DIR = EVID / "dados" / "alvo"
GD = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\graph_data")

TILE = ("lins", "Q1")
NOMES = {0: "elevacao_norm", 1: "slope", 2: "aspect", 3: "curvatura",
         4: "TPI(col4)", 5: "col5(rotulada TRI, e TPI)", 6: "TRI(col6)",
         7: "roughness(col7)", 12: "NDVI"}


def a_terrain(slope: np.ndarray) -> np.ndarray:
    """Reproduz generate_realistic_coverage.py:107-116, sem alterar."""
    tf = slope * 20.0
    tf = np.where(slope > 0.1, tf + (slope - 0.1) * 30.0, tf)
    return np.clip(tf, 0, 30)


def resumo(a: np.ndarray) -> dict:
    return {"min": float(a.min()), "p05": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "media": float(a.mean()), "std": float(a.std())}


def main() -> int:
    c, q = TILE
    p = GD / f"transfer_dataset_{c}_v19_{q}_enriched_v2.pt"
    if not p.exists():
        print(f"ABORTA: {p} ausente")
        return 1
    d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    t = d["terrain"]
    fr = t.features_raw
    x = (fr.numpy() if torch.is_tensor(fr) else np.asarray(fr)).astype(np.float32)
    y = (t.y.numpy() if torch.is_tensor(t.y) else np.asarray(t.y)).astype(np.float32)
    del d

    res = {"tile": f"{c}_{q}", "arquivo": p.name, "n_nos": int(x.shape[0]),
           "colunas": {}, "hipoteses_de_escala": {}, "peso_do_A_terrain": {}}

    print(f"=== colunas de {c}_{q} ===")
    for ci, nome in NOMES.items():
        if ci >= x.shape[1]:
            continue
        col = x[:, ci].astype(np.float64)
        r = resumo(col)
        r["frac_negativa"] = float((col < 0).mean())
        r["frac_zero"] = float((col == 0).mean())
        res["colunas"][f"{ci}_{nome}"] = r
        print(f"  col{ci:<2} {nome:<28} min={r['min']:9.4f} p50={r['p50']:9.4f} "
              f"max={r['max']:9.4f} neg={r['frac_negativa']:.3f}")

    # the random draw, reproduced with the rebuild seed
    rng = np.random.RandomState(42)
    sort = rng.uniform(0.02, 0.15, x.shape[0]).astype(np.float32)
    at_sort = a_terrain(sort.astype(np.float64))
    res["A_terrain_do_sorteio"] = resumo(at_sort)
    print(f"\n=== A_terrain hoje (sorteio uniforme 0,02-0,15) ===")
    print(f"  {res['A_terrain_do_sorteio']}")


    s1 = x[:, 1].astype(np.float64)
    hip = {
        "col1_como_esta": s1,
        "col1_dividida_por_45_graus": s1 / 45.0,
        "col1_como_graus_para_tangente": np.tan(np.deg2rad(np.clip(s1, 0, 89.9))),
        "col1_normalizada_pelo_max": s1 / max(s1.max(), 1e-9),
    }
    print("\n=== A_terrain com a declividade REAL, por hipotese de escala ===")
    for nome, sv in hip.items():
        at = a_terrain(np.clip(sv, 0, None))
        res["hipoteses_de_escala"][nome] = {
            "slope": resumo(sv), "A_terrain": resumo(at),
            "frac_slope_acima_0_1": float((sv > 0.1).mean()),
            "frac_A_terrain_saturado_30": float((at >= 30.0 - 1e-9).mean()),
        }
        print(f"  {nome:<32} slope p50={np.percentile(sv,50):8.4f} | "
              f"A_terrain p50={np.percentile(at,50):7.3f} med={at.mean():7.3f} "
              f"sat30={(at >= 30-1e-9).mean():.3%}")


    com_cob = y[:, 0] != 300.0
    res["peso_do_A_terrain"] = {
        "n_nos_com_cobertura": int(com_cob.sum()),
        "path_loss_com_cobertura": resumo(y[com_cob, 0].astype(np.float64)),
        "A_terrain_sorteio_media_db": float(at_sort.mean()),
        "razao_media_A_terrain_sobre_path_loss": float(
            at_sort.mean() / max(y[com_cob, 0].mean(), 1e-9)),
    }
    print(f"\n=== peso ===")
    print(f"  nos com cobertura: {int(com_cob.sum()):,}")
    print(f"  path_loss (com cobertura) p50={np.percentile(y[com_cob,0],50):.2f} dB")
    print(f"  A_terrain sorteado, media={at_sort.mean():.3f} dB "
          f"({res['peso_do_A_terrain']['razao_media_A_terrain_sobre_path_loss']:.2%} "
          f"da perda media)")

    dest = OUT_DIR / "contrafactual_slope_real_medida.json"
    dest.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ngravado: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
