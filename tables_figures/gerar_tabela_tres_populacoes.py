
"""Main three-population (all/covered/sentinel) table per cell under the blocked partition."""
import json
from pathlib import Path
import numpy as np

INV = Path(__file__).resolve().parent
EVID = INV.parent
OUT = INV / "novas"; OUT.mkdir(exist_ok=True)
T15 = json.load(open(EVID / "_campanha_2026-09-13/t15_tres_populacoes/t15_tres_populacoes.json", encoding="utf-8"))
PLANO_V2 = EVID / "dados/planos/plano_T15_tres_populacoes_2026-09-13_v2.json"
TOL = 2.5e-4 if PLANO_V2.exists() else 1.5e-4   # chain-of-custody tolerance: plan v2 (2.5e-4 dB, from the T1 measurement) if stamped
for _r in T15["registros"]:
    if "erro" not in _r:
        _r["custodia_ok"] = bool(_r["idx_sha256_global_test_ok"] and _r["diff_rssi_total_vs_run_db"] <= TOL)
MLP = json.load(open(EVID / "dados/treinos_c1/agregado_mlp_vs_gnn_c1v2.json", encoding="utf-8"))["celulas"]
NOME = {"v1_original": "raw", "v2_clamp": "clamped", "v3_clamp_freq_enlace": "clamped, per-link $f$"}
MODELO = {"fspl": "FSPL", "hata_rural": "Hata rural", "cost231_sub": "COST-231 sub."}

def ms(vals):
    v = np.array(vals, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan")), len(v)

rows, rastro = [], {}
ordem = [(c, q) for c in ("bauru", "campinas", "lins", "sorocaba") for q in (1, 2, 3, 4)]
for c, q in ordem:
    regs = [r for r in T15["registros"] if "erro" not in r and r["cidade"] == c and r["quadrante"] == q]
    ok = [r for r in regs if r["custodia_ok"]]
    excl = [r["seed"] for r in regs if not r["custodia_ok"]]
    frac = ok[0]["frac_valido"] if ok else regs[0]["frac_valido"]
    tot = ms([r["mae_rssi_total_db"] for r in ok]); val = ms([r["mae_rssi_valido_db"] for r in ok]); pl = ms([r["mae_pl_valido_db"] for r in ok])
    mlp = MLP.get(f"{c}_Q{q}", {}).get("mlp_mae_rssi_db", {}).get("media")
    b = json.load(open(EVID / f"_campanha_2026-09-13/t5_baselines_v3/baselines_v3_{c}_Q{q}.json", encoding="utf-8"))["particoes"]["test"]
    cands = [(b["variantes"][v][m]["todos"]["mae_db"], b["variantes"][v][m]["cobertura"]["mae_db"], v, m)
             for v in b["variantes"] for m in ("fspl", "hata_rural", "cost231_sub")]
    const_t, const_c = b["preditor_constante_piso"]["todos"]["mae_db"], b["preditor_constante_piso"]["cobertura"]["mae_db"]
    best_all = min(cands + [(const_t, const_c, "const", "const")], key=lambda x: x[0])
    best_cov = min(cands, key=lambda x: x[1])
    def rot(v, m):
        return "constant" if v == "const" else f"{MODELO[m]} ({NOME[v]})"
    nseed = f"{tot[2]}" if tot[2] == 5 else f"{tot[2]}$^{{\\dagger}}$"
    rows.append(f"{c.capitalize()} & Q{q} & {frac:.3f} & {tot[0]:.3f} $\\pm$ {tot[1]:.3f} & {val[0]:.2f} $\\pm$ {val[1]:.2f} & {pl[0]:.2f} $\\pm$ {pl[1]:.2f} & "
                + (f"{mlp:.3f}" if mlp is not None else "--") +
                f" & {best_all[0]:.3f} & {best_cov[1]:.2f} & {const_t:.3f} & {nseed} \\\\")
    rastro[f"{c}_Q{q}"] = {"gnn": {"artefato": "_campanha_2026-09-13/t15_tres_populacoes/t15_tres_populacoes.json", "corridas_usadas": [r["seed"] for r in ok],
                                   "corridas_excluidas_custodia": excl, "rssi_todos": tot[0], "rssi_valido": val[0], "pl_valido": pl[0]},
                           "empirico_melhor_todos": {"artefato": f"_campanha_2026-09-13/t5_baselines_v3/baselines_v3_{c}_Q{q}.json", "variante": best_all[2], "modelo": best_all[3], "mae_todos": best_all[0]},
                           "empirico_melhor_cobertura": {"variante": best_cov[2], "modelo": best_cov[3], "mae_cobertura": best_cov[1]},
                           "constante": {"todos": const_t, "cobertura": const_c}, "frac_valido": frac}

tex = r"""\begin{table*}[t]
\caption{Test error on the spatially blocked split (10~km blocks, 2~km exclusion buffer), all 16 city--quadrant cells, three populations of test nodes: all nodes (sentinel included), nodes with a valid path-loss target (``covered''), and the covered subset for path loss. GNN-RF entries are mean $\pm$ s.d. over training seeds; MLP is the capacity-matched control without message passing (all nodes; Lins and Bauru). ``Best empirical'' is the lowest-error calibrated empirical model on the same nodes, chosen separately for the all-node and covered populations; ``Constant'' is the $-110$~dBm sentinel predictor. A seed enters the table when its recomputed test error reproduces the training record within $2.5\times10^{-4}$~dB, the measured inference non-determinism of the pipeline; all 80 runs meet this criterion.}
\label{tab:blocked_three_pop}
\centering
\small
\begin{tabular}{ll c ccc c cc c c}
\toprule
 & & & \multicolumn{3}{c}{GNN-RF (dB)} & MLP & \multicolumn{2}{c}{Best empirical (dB)} & Constant & \\
City & Q & Covered frac. & RSSI, all & RSSI, covered & PL, covered & RSSI, all & RSSI, all & RSSI, covered & RSSI, all & seeds \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % "\n".join(rows)
(OUT / "tab_blocked_16_tres_pop.tex").write_text(tex, encoding="utf-8")
json.dump({"status": "T15 contra-auditado (ca-t15-tres-populacoes.json); tolerancia de custodia do plano v2", "tolerancia_db": TOL, "celulas": rastro}, open(OUT / "tres_pop_rastro.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n".join(rows))
