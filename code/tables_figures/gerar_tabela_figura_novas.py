
"""New manuscript table and figure, read only from independently verified artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INV = Path(__file__).resolve().parent
EVID = INV.parent
OUT = INV / "novas"; OUT.mkdir(exist_ok=True)

T8 = json.load(open(EVID / "_campanha_2026-09-13/passada_unica/agregado_passada_unica.json", encoding="utf-8"))["T8_embaralhamento_por_relacao"]
MLP = json.load(open(EVID / "dados/treinos_c1/agregado_mlp_vs_gnn_c1v2.json", encoding="utf-8"))
CA = json.load(open(EVID / "dados/contra_auditoria/rodada2_2026-09-13/ca_recomputo.json", encoding="utf-8"))["mlp_vs_gnn"]
T5 = json.load(open(EVID / "_campanha_2026-09-13/t5_baselines_v3/tabela_baselines_3variantes.json", encoding="utf-8"))["celulas"]
CLAIM = json.load(open(EVID / "dados/consolidacao/claim_c1v2_estatistica.json", encoding="utf-8"))["claim"]
rastro = {}


bracos = [("ant_ter_embaralhado", "Antenna$\\to$terrain edges shuffled (degree preserved)"),
          ("ter_ter_embaralhado", "Terrain$\\to$terrain edges shuffled (degree preserved)"),
          ("ambas_embaralhadas", "Both relations shuffled"),
          ("vazio", "Empty graph (no message passing)")]
linhas = []
for k, rot in bracos:
    b = T8[k]
    linhas.append(f"{rot} & {b['delta_medio_db']:+.3f} & [{b['ic95_db'][0]:+.3f}; {b['ic95_db'][1]:+.3f}] & {b['p_t']:.4f} & {b['p_wilcoxon']:.4f} \\\\")
    rastro[f"T8.{k}"] = {"artefato": "_campanha_2026-09-13/passada_unica/agregado_passada_unica.json",
                         "campo": f"T8_embaralhamento_por_relacao.{k}.{{delta_medio_db,ic95_db,p_t,p_wilcoxon}}",
                         "contra_auditoria": "ca-passada-unica-e-gate.json (reproduziu no dígito)"}
c8 = CA["unidade_celula_n8"]; tost = CA["tost_celula_margem_0.1"]
linha_mlp = (f"GNN-RF minus capacity-matched MLP (1{{,}}812{{,}}515 params) & {c8['media_db']:+.3f} & "
             f"[{c8['ic95_db'][0]:+.3f}; {c8['ic95_db'][1]:+.3f}] & {c8['p_bicaudal']:.4f} & {c8['wilcoxon_p']:.4f} \\\\")
rastro["MLP_vs_GNN"] = {"artefato": "dados/contra_auditoria/rodada2_2026-09-13/ca_recomputo.json",
                        "campo": "mlp_vs_gnn.unidade_celula_n8.{media_db,ic95_db,p_bicaudal,wilcoxon_p}; tost_celula_margem_0.1.p_tost",
                        "ressalva": "sem 'empate'; equivalência a 0,1 dB não estabelecida (p_TOST=%.4f)" % tost["p_tost"]}
tex = r"""\begin{table}[t]
\caption{Structural ablation on the spatially blocked test split (Lins and Bauru, Q1--Q4, seed~42 checkpoints; $n=8$ cells, three shuffling seeds per arm). Entries are the change in test RSSI MAE relative to the trained model with the real graph, paired by cell; positive means worse. The last row is a different contrast: the paired difference GNN-RF minus a multilayer perceptron with exactly the same number of trainable parameters, loss, split and selection rule and no message passing, averaged over five training seeds per cell (positive means the MLP is more accurate); equivalence within $\pm0.1$~dB is not established (TOST $p=%.3f$).}
\label{tab:structural_ablation}
\centering
\begin{tabular}{lccc c}
\toprule
Arm & $\Delta$MAE (dB) & 95\%% CI & $p$ (paired $t$) & $p$ (Wilcoxon) \\
\midrule
%s
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (tost["p_tost"], "\n".join(linhas), linha_mlp)
(OUT / "tab_ablacao_estrutural.tex").write_text(tex, encoding="utf-8")

# Table + figure: 16 blocked cells
ordem = [(c, q) for c in ("bauru", "campinas", "lins", "sorocaba") for q in (1, 2, 3, 4)]
NOME = {"v1_original": "raw", "v2_clamp": "clamped", "v3_clamp_freq_enlace": "clamped, per-link $f$"}
MODELO = {"fspl": "FSPL", "hata_rural": "Hata rural", "cost231_sub": "COST-231 sub."}
rows = []; fig_rows = []
for c, q in ordem:
    cel = T5[f"{c}_Q{q}"]
    gnn = cel["gnn_test"]["mae_rssi_db_media_seeds"]
    gnn_sd = float(np.std(list(cel["gnn_test"]["mae_rssi_db_por_seed"].values()), ddof=1))
    best = cel["melhor_competidor"]
    const = cel["competidores_test_todos"]["preditor_constante_-110dBm"]
    const_v = const["mae_rssi_todos_db"]
    nome_best = best["nome"]
    if nome_best.startswith("preditor_constante"):
        rot = "constant $-110$~dBm"
    else:
        var, mdl = nome_best.split(".")
        rot = f"{MODELO[mdl]} ({NOME[var]})"
    mlp = MLP["celulas"].get(f"{c}_Q{q}", {}).get("mlp_mae_rssi_db", {}).get("media")
    perde = "$\\ast$" if cel["gnn_perde_do_melhor_competidor"] else ""
    rows.append(f"{c.capitalize()} & Q{q} & {gnn:.3f} $\\pm$ {gnn_sd:.3f} & " + (f"{mlp:.3f}" if mlp is not None else "--") +
                f" & {best['mae_rssi_todos_db']:.3f}{perde} & {rot} & {const_v:.3f} \\\\")
    fig_rows.append((f"{c[:3].capitalize()} Q{q}", gnn, gnn_sd, mlp, best["mae_rssi_todos_db"], const_v))
    rastro[f"blocked.{c}_Q{q}"] = {"artefato": "_campanha_2026-09-13/t5_baselines_v3/tabela_baselines_3variantes.json",
                                   "campo": f"celulas.{c}_Q{q}.{{gnn_test.mae_rssi_db_media_seeds, melhor_competidor, competidores_test_todos.preditor_constante_-110dBm}}",
                                   "contra_auditoria": "ca-t5-baselines-v3.json (CONFIRMA, diff 0.0)"}
uc, ucity = CLAIM["unidade_celula"], CLAIM["unidade_cidade"]
rastro["blocked.media"] = {"artefato": "dados/consolidacao/claim_c1v2_estatistica.json", "campo": "claim.unidade_celula.{media_db,ic95_db}; claim.unidade_cidade.ic95_db"}
tex2 = r"""\begin{table*}[t]
\caption{Test RSSI MAE on the spatially blocked split (10~km blocks, 2~km exclusion buffer, 70/15/15 by block, split seed~42), all 16 city--quadrant cells. GNN-RF: mean $\pm$ s.d. over five training seeds. MLP: capacity-matched control without message passing (Lins and Bauru only). Best empirical: the lowest-error variant among FSPL, Okumura--Hata rural and COST-231 suburban, calibrated on training blocks only, evaluated on exactly the same test nodes. Constant: the $-110$~dBm sentinel predictor on the same nodes. Cell mean over the 16 cells: %.3f~dB, 95\%% CI [%.3f; %.3f] (unit = cell); city-level CI [%.3f; %.3f] ($n=4$). $\ast$: the only cell in which the best empirical model beats GNN-RF (by 0.009~dB).}
\label{tab:blocked_results}
\centering
\begin{tabular}{llcccl c}
\toprule
City & Q & GNN-RF (dB) & MLP (dB) & Best empirical (dB) & Best empirical model & Constant (dB) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (uc["media_db"], uc["ic95_db"][0], uc["ic95_db"][1], ucity["ic95_db"][0], ucity["ic95_db"][1], "\n".join(rows))
(OUT / "tab_blocked_16.tex").write_text(tex2, encoding="utf-8")

fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(len(fig_rows)); w = 0.2
g = [r[1] for r in fig_rows]; gs = [r[2] for r in fig_rows]
m = [r[3] if r[3] is not None else np.nan for r in fig_rows]
b = [r[4] for r in fig_rows]; k = [r[5] for r in fig_rows]
ax.bar(x - 1.5*w, g, w, yerr=gs, capsize=2, label="GNN-RF (5 seeds)", color="#1f5f8b")
ax.bar(x - 0.5*w, m, w, label="MLP, capacity-matched", color="#7fb3dc")
ax.bar(x + 0.5*w, b, w, label="Best empirical (calibrated)", color="#c98a3a")
ax.bar(x + 1.5*w, k, w, label="Constant $-110$ dBm", color="#9a9a9a")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in fig_rows], rotation=45, ha="right")
ax.set_ylabel("Test RSSI MAE (dB), blocked split"); ax.set_ylim(0, max(k) * 1.15)
ax.legend(ncol=4, fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=.3)
for i, r in enumerate(fig_rows):
    ax.text(x[i] - 1.5*w, r[1] + r[2] + 0.03, f"{r[1]:.2f}", ha="center", fontsize=6.5)
fig.tight_layout()
fig.savefig(OUT / "fig_blocked_16.png", dpi=200); fig.savefig(OUT / "fig_blocked_16.pdf")


fig2, ax2 = plt.subplots(figsize=(7.5, 3.8))
cells = list(T8["ambas_embaralhadas"]["delta_por_celula"].keys())
x2 = np.arange(len(cells)); w2 = 0.2
for j, (kk, rot, col) in enumerate([("ant_ter_embaralhado", "antenna→terrain shuffled", "#c98a3a"),
                                    ("ter_ter_embaralhado", "terrain→terrain shuffled", "#2f6b3a"),
                                    ("ambas_embaralhadas", "both shuffled", "#1f5f8b"),
                                    ("vazio", "empty graph", "#9a9a9a")]):
    ax2.bar(x2 + (j - 1.5) * w2, [T8[kk]["delta_por_celula"][c] for c in cells], w2, label=rot, color=col)
ax2.axhline(0, color="k", lw=.8)
ax2.set_xticks(x2); ax2.set_xticklabels([c.replace("_", " ") for c in cells], rotation=45, ha="right")
ax2.set_ylabel("ΔMAE RSSI vs. real graph (dB)"); ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=.3)
fig2.tight_layout(); fig2.savefig(OUT / "fig_ablacao_estrutural.png", dpi=200); fig2.savefig(OUT / "fig_ablacao_estrutural.pdf")

json.dump(rastro, open(OUT / "numeros_desta_prosa.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("ok:", sorted(p.name for p in OUT.iterdir()))
