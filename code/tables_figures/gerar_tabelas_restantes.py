
"""Remaining manuscript tables/figures, read only from independently verified artifacts."""
import json, glob, statistics, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INV = Path(__file__).resolve().parent
EVID = INV.parent
OUT = INV / "novas"; OUT.mkdir(exist_ok=True)
R2 = EVID.parents[1] / "REVISAO_R2" / "novas"


C = json.load(open(EVID / "dados/treinos_c1/agregado_controles_c1v2.json", encoding="utf-8"))["sem_transferencia"]
cl = C["estatistica_unidade_cluster"]; tost = C["tost_equivalencia"]
pares = C["pares_24_declarados"]
rows = []
for cidade in ("bauru", "campinas", "lins", "sorocaba"):
    for seed in (42, 43):
        for q in (2, 3, 4):
            p = pares[f"{cidade}_s{seed}_Q{q}"]
            rows.append(f"{cidade.capitalize()} & {seed} & Q{q} & {p['sem_cadeia_db']:.3f} & {p['encadeado_db']:.3f} & {p['delta_encadeado_menos_semcadeia_db']:+.3f} \\\\")
tex9 = r"""\begin{table}[t]
\caption{Transfer chain versus training from scratch on the same blocked split. Each pair compares the quadrant model obtained by fine-tuning from the previous quadrant (chained: 20 epochs, learning rate $2\times10^{-4}$) with a model trained from scratch on the same data and split (35 epochs, $10^{-3}$); test RSSI MAE in dB, all test nodes. Unit of analysis: (city, seed) cluster, $n=8$. Mean difference (chained minus scratch) $%.3f$~dB, 95\%% CI [$%.3f$; $%.3f$], paired $t$ $p=%.2f$; two one-sided tests at a $\pm0.1$~dB margin exclude a chaining effect of that size ($p=%.4f$). Because the two regimes differ in epochs and learning rate, the comparison establishes the absence of a systematic advantage of chaining, not a clean estimate of its effect.}
\label{tab:transfer_control}
\centering
\small
\begin{tabular}{llcccc}
\toprule
City & Seed & Q & Scratch (dB) & Chained (dB) & $\Delta$ (dB) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (cl["media_db"], cl["ic95_db"][0], cl["ic95_db"][1], cl["p_bicaudal"], tost["p_tost"], "\n".join(rows))
(OUT / "tab_controles_24.tex").write_text(tex9, encoding="utf-8")


A = json.load(open(EVID / "_campanha_2026-09-13/t14_mc_dropout/agregado_T14.json", encoding="utf-8"))
P = A["populacoes"]["test_valido_rssi"]["metodos"]
MET = [("A_mc_percentil", "MC-dropout (percentile)"), ("B_mc_gaussiano", "MC-dropout (Gaussian)"),
       ("C_conformal_split", "Split-conformal on validation blocks"), ("D_residual_deterministico", "Deterministic residual (validation-calibrated)")]
NIV = ["50", "80", "90", "95", "99"]
rows = []
for k, rot in MET:
    picp = " & ".join(f"{P[k][n]['picp_media']:.3f} [{P[k][n]['picp_min']:.2f}; {P[k][n]['picp_max']:.2f}]" for n in NIV)
    larg = " & ".join(f"{P[k][n]['largura_media_db']:.1f}" for n in NIV)
    rows.append(f"{rot} & PICP & {picp} & {P[k]['C2_celulas_ok_80_90_95']}/16 \\\\")
    rows.append(f" & width (dB) & {larg} & \\\\")
tex14 = r"""\begin{table*}[t]
\caption{Prediction-interval coverage on the spatially blocked test split, covered nodes, RSSI channel, 16 cells (seed-42 checkpoints, 50 stochastic passes). PICP is the mean over cells with the [min; max] across cells; width is the mean interval width. The last column counts cells within $\pm0.05$ of nominal simultaneously at 80, 90 and 95\,\%; the pre-registered criterion for describing a method as calibrated required at least 12 of 16, which no method meets. Calibration for the conformal and residual methods uses the validation blocks of the same cell, which are spatially separated from the test blocks.}
\label{tab:uncertainty_blocked}
\centering
\small
\begin{tabular}{llccccc c}
\toprule
Method & & \multicolumn{5}{c}{Nominal level} & Cells within 0.05 \\
 & & 50\,\% & 80\,\% & 90\,\% & 95\,\% & 99\,\% & (80/90/95) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""".replace("%s", "\n".join(rows))
(OUT / "tab_incerteza_T14.tex").write_text(tex14, encoding="utf-8")

fig, ax = plt.subplots(figsize=(5.2, 4.2))
x = [0.5, 0.8, 0.9, 0.95, 0.99]
cores = {"A_mc_percentil": "#1f5f8b", "C_conformal_split": "#2f6b3a", "D_residual_deterministico": "#c98a3a"}
for k, rot in MET:
    if k == "B_mc_gaussiano":
        continue
    per = [np.array(list(P[k][n]["por_celula_picp"].values())) for n in NIV]
    m = [float(np.mean(v)) for v in per]; lo = [float(np.percentile(v, 25)) for v in per]; hi = [float(np.percentile(v, 75)) for v in per]
    ax.plot(x, m, "o-", color=cores[k], label=rot)
    ax.fill_between(x, lo, hi, color=cores[k], alpha=.15)
ax.plot([0.4, 1], [0.4, 1], "k--", lw=.8, label="nominal")
ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage (PICP), covered test nodes")
ax.set_title("Mean over 16 cells; band = interquartile range across cells", fontsize=8)
ax.set_xlim(0.45, 1.0); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_calibracao_T14.png", dpi=200); fig.savefig(OUT / "fig_calibracao_T14.pdf")


b = json.load(open(EVID / "dados/baselines_v2/baselines_v2_bauru_Q1.json", encoding="utf-8"))["dataset"]
census = json.load(open(EVID / "dados/censo_grafo/fase0_graph_census.json", encoding="utf-8"))
tt = None
try:
    tt = census["quadrantes"]["Q1"]["edge_types"]["('terrain','connects_to','terrain') [de gpu.pt dem-adjacent_to-dem]"]["num_edges"]
except Exception:
    for k, v in census["quadrantes"]["Q1"]["edge_types"].items():
        if "terrain" in k and "connects_to" in k or "adjacent" in k:
            tt = v["num_edges"]
vr, tp = [], []
for f in glob.glob(str(EVID / "dados/treinos_c1/run_c0c1cf_*_g10b2.json")):
    r = json.load(open(f, encoding="utf-8")); vr.append(r["custo"]["vram_peak_mb"]); tp.append(r["custo"]["tempo_total_s"])
r42 = json.load(open(EVID / "dados/treinos_c1/run_c0c1cf_bauru_s42_Q1_g10b2.json", encoding="utf-8"))
cfg, mod = r42["config"], r42["modelo"]
texc = r"""\begin{table}[t]
\caption{Graph census and training cost per quadrant (Bauru Q1 shown; the other 15 quadrants have the same node counts and edge counts within 1\,\%%). Costs are over the 80 blocked runs on one NVIDIA RTX 5070 Ti (16~GB).}
\label{tab:graph_census}
\centering
\small
\begin{tabular}{lr}
\toprule
Terrain nodes & \num{%d} \\
Antenna nodes & \num{%d} \\
Antenna$\to$terrain edges (and reverse) & \num{%d} \\
Terrain$\to$terrain edges (8-neighbour) & \num{%d} \\
Serialised graph file & %.1f~GB \\
Sampled neighbours per node (antenna / terrain) & %d / %d \\
Training batch (seed nodes) & \num{%d} \\
Epochs (head quadrant / transferred) & 35 / 20 \\
Trainable parameters & \num{%d} \\
Peak training VRAM (min--max, median) & %.0f--%.0f~MB, %.0f \\
Wall-clock per run (min--max, median) & %.0f--%.0f~s, %.0f \\
\bottomrule
\end{tabular}
\end{table}
""" % (b["n_nos_terrain"], b["n_antenas"], b["n_arestas_ant_ter"], tt, b["rf_data_bytes"] / 1e9, cfg["k_antenna"], cfg["k_terrain"],
       cfg["batch_size"], mod["n_params_treinaveis"], min(vr), max(vr), statistics.median(vr), min(tp), max(tp), statistics.median(tp))
(OUT / "tab_censo_grafo.tex").write_text(texc, encoding="utf-8")

rastro = {"tab_controles_24": "dados/treinos_c1/agregado_controles_c1v2.json (CA ca_recomputo.json)",
          "tab_incerteza_T14": "_campanha_2026-09-13/t14_mc_dropout/agregado_T14.json (CA ca-t14-mc-dropout.json)",
          "tab_censo_grafo": "dados/baselines_v2/baselines_v2_bauru_Q1.json :: dataset; dados/censo_grafo/fase0_graph_census.json (ter-ter); 80 run_c0c1cf JSONs :: custo, modelo, config"}
json.dump(rastro, open(OUT / "rastro_tabelas_restantes.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
R2.mkdir(parents=True, exist_ok=True)
for f in OUT.iterdir():
    shutil.copy(f, R2 / f.name)
print("ok:", sorted(p.name for p in OUT.iterdir()))
