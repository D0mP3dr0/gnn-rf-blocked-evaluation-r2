"""Passo 2 do contrafactual: refaz o alvo de lins_Q1 com a DECLIVIDADE REAL.

PERGUNTA: o termo de terreno do alvo e sorteado
(`prepare_transfer_dataset_v19.py:141`, np.random.uniform(0.02, 0.15)) em vez
de calculado do relevo — enquanto a vegetacao, na linha 140, entra com o dado
medido. Corrigir isso muda o resultado do treino?

POR QUE DA PARA FAZER ISSO SEM RECOMPUTAR A PROPAGACAO INTEIRA:
`generate_realistic_coverage.py:120` compoe a perda como
    total_loss = base_loss + A_veg + A_terrain
e A_terrain (linhas 107-116) depende SO do no, nao da antena. Logo:
  - ele e o mesmo para todas as antenas daquele no;
  - nao altera qual antena vence o `rssi_mat.max(axis=1)`;
  - entra aditivamente, entao o alvo contrafactual sai por diferenca exata,
    sem refazer distancias, sem tocar nas arestas e sem GPU.

    y_novo[:,0] = y[:,0] - A_terrain(sorteio) + A_terrain(real)      # path loss
    y_novo[:,3] = y[:,3] + A_terrain(sorteio) - A_terrain(real)      # RSSI

O sorteio e reproduzivel: `np.random.uniform` da linha 141 e a UNICA chamada
aleatoria do arquivo, e `e3_alvos_dataset_v2_multi.py:423` faz
np.random.seed(42) imediatamente antes de chamar build_transfer_one_quadrant.

ESCALA: a col1 (slope) ja esta na escala que o modelo espera — o comentario do
gerador diz "slope em [0,1] onde 1 = 45 graus" e a col1 de lins_Q1 vive em
[0,0000; 0,4791], sem negativos (contrafactual_slope_real_medida.json). Nada a
converter.

NAO TOCA no tile original. Escreve um arquivo NOVO, com sufixo _cfslope.
Os nos SENTINELA (path loss == 300,0) ficam intocados: sao marcador de "fora
do alcance", nao perda medida, e somar/subtrair dB neles seria inventar dado.
"""
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
SENTINELA = 300.0
SEED = 42
IDX_SLOPE = 1


def a_terrain(slope: np.ndarray) -> np.ndarray:
    """Copia fiel de generate_realistic_coverage.py:107-116."""
    tf = slope * 20.0
    tf = np.where(slope > 0.1, tf + (slope - 0.1) * 30.0, tf)
    return np.clip(tf, 0, 30)


def resumo(a: np.ndarray) -> dict:
    return {"min": float(a.min()), "p05": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "media": float(a.mean()), "std": float(a.std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="lins")
    ap.add_argument("--quad", default="Q1")
    args = ap.parse_args()
    c, q = args.city, args.quad

    src = GD / f"transfer_dataset_{c}_v19_{q}_enriched_v2.pt"
    dst = GD / f"transfer_dataset_{c}_v19_{q}_enriched_v2_cfslope.pt"
    if not src.exists():
        print(f"ABORTA: {src} ausente")
        return 1

    print(f"lendo {src.name}", flush=True)
    d = torch.load(src, map_location="cpu", weights_only=False)
    t = d["terrain"]
    fr = t.features_raw
    x = (fr.numpy() if torch.is_tensor(fr) else np.asarray(fr)).astype(np.float32)
    y = (t.y.numpy() if torch.is_tensor(t.y) else np.asarray(t.y)).astype(np.float32).copy()
    n = x.shape[0]

    # sorteio reproduzido exatamente como o rebuild o fez
    np.random.seed(SEED)
    slope_sorteado = np.random.uniform(0.02, 0.15, n).astype(np.float32)
    slope_real = np.clip(x[:, IDX_SLOPE].astype(np.float64), 0, None)

    at_old = a_terrain(slope_sorteado.astype(np.float64))
    at_new = a_terrain(slope_real)
    delta = at_new - at_old                      # dB a somar na perda

    com_cob = y[:, 0] != SENTINELA
    n_cob = int(com_cob.sum())

    y_novo = y.copy()
    y_novo[com_cob, 0] = (y[com_cob, 0].astype(np.float64) + delta[com_cob]).astype(np.float32)
    if y.shape[1] > 3:
        y_novo[com_cob, 3] = (y[com_cob, 3].astype(np.float64) - delta[com_cob]).astype(np.float32)

    rec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tile": f"{c}_{q}", "origem": str(src), "saida": str(dst),
        "o_que_muda": ("col0 (path loss) e col3 (RSSI) nos nos COM cobertura; "
                       "features, arestas, pos e nos sentinela intocados"),
        "seed_do_sorteio_reproduzido": SEED,
        "n_nos": int(n), "n_nos_com_cobertura": n_cob,
        "A_terrain_sorteio": resumo(at_old),
        "A_terrain_slope_real": resumo(at_new),
        "delta_db_nos_com_cobertura": resumo(delta[com_cob]),
        "path_loss_antes": resumo(y[com_cob, 0].astype(np.float64)),
        "path_loss_depois": resumo(y_novo[com_cob, 0].astype(np.float64)),
        "frac_nos_com_cobertura_alterados": float(np.mean(np.abs(delta[com_cob]) > 1e-6)),
        "delta_abs_medio_db": float(np.abs(delta[com_cob]).mean()),
        "delta_abs_max_db": float(np.abs(delta[com_cob]).max()),
    }

    print(f"\nnos com cobertura: {n_cob:,} de {n:,}")
    print(f"A_terrain sorteio : media {rec['A_terrain_sorteio']['media']:.4f} dB")
    print(f"A_terrain real    : media {rec['A_terrain_slope_real']['media']:.4f} dB")
    print(f"delta |dB| medio  : {rec['delta_abs_medio_db']:.4f} | max "
          f"{rec['delta_abs_max_db']:.4f}")
    print(f"path_loss p50 antes {rec['path_loss_antes']['p50']:.4f} -> "
          f"depois {rec['path_loss_depois']['p50']:.4f}")

    t.y = torch.from_numpy(y_novo)
    print(f"\ngravando {dst.name} ...", flush=True)
    torch.save(d, dst)
    rec["saida_bytes"] = dst.stat().st_size
    rec["sha256_gerador"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    dest = OUT_DIR / f"contrafactual_slope_real_{c}_{q}.json"
    dest.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"gravado: {dst}  ({dst.stat().st_size/1e9:.2f} GB)")
    print(f"registro: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
