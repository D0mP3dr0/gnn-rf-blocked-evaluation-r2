"""BA_por_celula.py -- per-cell means (5 seeds) of the node-wise adaptation (arm ba1), GNN-RF (arm gnn) and MLP (arm mlp)
on the blocked test split, read ONLY from BA_metricas_por_corrida.json (per-run metrics, custody-checked, BA_custodia.json ok).
Purpose: two new columns of Table 6 (tab:blocked_three_pop) and the fourth series of Fig. 3 (fig:multiseed), decision of the
owner on 23/10/2026. Writes BA_por_celula.json. Cross-checks: the 16-cell means of ba1 against BA_resumo.json (media_BA1 of the
primary contrasts) and the gnn per-cell means against the values printed in Table 6 of the R4 (t15 seed-level evaluation).
Run from results/ba1/ with the interpreter of requirements.txt
"""
import json, hashlib, datetime
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent
SRC = AQUI / "BA_metricas_por_corrida.json"
RES = AQUI / "BA_resumo.json"
T15 = AQUI.parent / "blocked" / "t15_tres_populacoes.json"   # optional cross-check against the seed-level evaluation behind Table 6 (skipped if absent)
CITIES = ["bauru", "campinas", "lins", "sorocaba"]
CELLS = [f"{c}_Q{q}" for c in CITIES for q in range(1, 5)]
POPS = {"ALL": "mae_rssi_ALL", "VALID": "mae_rssi_VALID"}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

m = json.load(open(SRC, encoding="utf-8"))
assert m["completo"] and m["n_runs_por_braco"] == {"gnn": 80, "ba1": 80, "mlp": 80}, m["n_runs_por_braco"]
runs = {}
for r in m["runs"]:
    runs.setdefault((r["braco"], f"{r['cidade']}_{r['Q']}"), []).append(r)
out = {"fonte": {"arquivo": SRC.name, "sha256": sha(SRC)}, "gerado_em": datetime.datetime.now().isoformat(timespec="seconds"),
       "definicao": "mean and sample s.d. (ddof=1) over the five seeds 42-46 of each cell; populations ALL (all test nodes) and VALID (valid-target nodes); dB",
       "por_celula": {}, "media_16_celulas": {}, "contagens": {}}
for arm in ("ba1", "gnn", "mlp"):
    for c in CELLS:
        rs = runs[(arm, c)]
        assert sorted(r["seed"] for r in rs) == [42, 43, 44, 45, 46], (arm, c)
        d = out["por_celula"].setdefault(c, {})
        for pop, k in POPS.items():
            v = np.array([r[k] for r in rs], float)
            d[f"{arm}_{pop}"] = {"media": float(v.mean()), "dp": float(v.std(ddof=1)), "n_seeds": 5}
    for pop in POPS:
        out["media_16_celulas"][f"{arm}_{pop}"] = float(np.mean([out["por_celula"][c][f"{arm}_{pop}"]["media"] for c in CELLS]))
for pop in POPS:
    out["contagens"][f"ba1_menor_que_gnn_{pop}_celulas"] = int(sum(out["por_celula"][c][f"ba1_{pop}"]["media"] < out["por_celula"][c][f"gnn_{pop}"]["media"] for c in CELLS))
    out["contagens"][f"ba1_menor_que_mlp_{pop}_celulas"] = int(sum(out["por_celula"][c][f"ba1_{pop}"]["media"] < out["por_celula"][c][f"mlp_{pop}"]["media"] for c in CELLS))

# cross-check 1: 16-cell means of ba1 and gnn against the primary contrasts of BA_resumo.json
res = json.load(open(RES, encoding="utf-8"))
chk = {}
for row in res["tabela"]:
    if row["contraste"] == "BA1_menos_C0" and row["metrica"] in ("mae_rssi_ALL", "mae_rssi_VALID") and row["populacao_leitura"] == "todas as Q":
        pop = row["metrica"].split("_")[-1]
        chk[pop] = {"media_BA1_resumo": row["media_BA1"], "media_C0_resumo": row["media_comparador"],
                    "media_ba1_aqui": out["media_16_celulas"][f"ba1_{pop}"], "media_gnn_aqui": out["media_16_celulas"][f"gnn_{pop}"]}
        assert abs(row["media_BA1"] - out["media_16_celulas"][f"ba1_{pop}"]) < 1e-9, (pop, chk[pop])
        assert abs(row["media_comparador"] - out["media_16_celulas"][f"gnn_{pop}"]) < 1e-9, (pop, chk[pop])
assert set(chk) == {"ALL", "VALID"}, chk
out["conferencia_BA_resumo"] = chk
# cross-check 2 (informational, optional): gnn per-cell means against the seed-level evaluation that feeds the GNN-RF columns of Table 6.
# The GNN-RF columns of Table 6 keep those values; the counts above compare ba1 and gnn under the SAME evaluator (this per-run JSON).
if T15.exists():
    t15 = json.load(open(T15, encoding="utf-8"))["por_celula"]
    dmax = {"ALL": 0.0, "VALID": 0.0}
    for c in CELLS:
        dmax["ALL"] = max(dmax["ALL"], abs(t15[c]["mae_rssi_total_db"]["media"] - out["por_celula"][c]["gnn_ALL"]["media"]))
        dmax["VALID"] = max(dmax["VALID"], abs(t15[c]["mae_rssi_valido_db"]["media"] - out["por_celula"][c]["gnn_VALID"]["media"]))
    out["conferencia_t15_gnn"] = {"t15": T15.name, "sha256": sha(T15), "max_abs_diff_db": dmax, "uso": "informativo"}
else:
    out["conferencia_t15_gnn"] = {"t15": "not in this package", "uso": "informativo"}
out["script_sha256"] = sha(__file__)
json.dump(out, open(AQUI / "BA_por_celula.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("medias 16 celulas:", {k: round(v, 4) for k, v in out["media_16_celulas"].items()})
print("contagens:", out["contagens"]); print("t15 gnn:", out["conferencia_t15_gnn"])
for c in CELLS:
    d = out["por_celula"][c]; print(f"{c:12s} ba1 ALL {d['ba1_ALL']['media']:.3f}±{d['ba1_ALL']['dp']:.3f}  VALID {d['ba1_VALID']['media']:.2f}±{d['ba1_VALID']['dp']:.2f} | gnn ALL {d['gnn_ALL']['media']:.3f} VALID {d['gnn_VALID']['media']:.2f}")
