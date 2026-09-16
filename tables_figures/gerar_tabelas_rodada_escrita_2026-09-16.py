
"""R3 writing-round LaTeX tables, read only from independently verified artifacts."""
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

INV = Path(__file__).resolve().parent
EVID = INV.parent
R3 = EVID.parents[1] / "REVISAO_R3"
NOVAS = R3 / "novas"
MANIFEST = EVID / "manifest.jsonl"
RASTRO = INV / "rastro_tabelas_rodada_escrita_2026-09-16.json"
TOL_REPRO = 2.5e-4


class Aborto(SystemExit):
    pass


def exige(cond, msg):
    if not cond:
        raise Aborto(f"ABORTA: {msg}")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


FONTES = {
    "t11_agregado": ("dados/treinos_c1/agregado_t11_mlp_vs_gnn_4cidades_2026-09-15.json",
                     "1c771575783bd0aeb9edb2c733afe187b3615bb84317616d7de8b25fb52b7f69", "fonte T1/T2"),
    "t11_ca": ("dados/contra_auditoria/rodada2_2026-09-13/ca-t11-4cidades.json",
               "747e22fbab1c6da2c2b640afbf2da2a6880ebac5a2339478cb88e46bfa0e0e1a", "CA T1/T2"),
    "transf": ("_campanha_2026-09-15/transferencia_bloqueada/transferencia_bloqueada_seed42.json",
               "70c26a21a6247091c43b1d8cad48d72f12d0929059edc41ea20db3ab2b19beea", "fonte T3"),
    "transf_ca": ("dados/contra_auditoria/rodada2_2026-09-13/ca-transferencia-bloqueada.json",
                  "6212aaa958084322c34cc363043a6e942945df34493ae7b48f850dd106e8c66a", "CA T3"),
    "nn": ("dados/contraste_1nn_vizinho_valido/contraste_1nn_vizinho_valido.json",
           "623d29a6ec57a50a9a39b46203b3ebd07988ad9cb26f94b0ded67e09db064ebd", "fonte T4"),
    "nn_ca": ("dados/contra_auditoria/rodada2_2026-09-13/ca-contraste-1nn-16celulas.json",
              "c1cc7cf24210f686b4a5175285c7503bdb075a02d28e65405a98add3a67c0cf7", "CA T4"),


    "linhagem": ("dados/consolidacao/linhagem_mae_e_custo_2026-09-15.json",
                 "c78ea098e15f426259ddb1193bb1eff3fc4e3c1bc60692711748f0ae74f288b4", "fonte T5"),
    "params": ("dados/consolidacao/params_checkpoints_2026-09-15.json",
               "59685cc57e9b4b0ee3f636c47a9155eb33b3e7264b2bd02a0da05d2f5152c5d1", "fonte T5/arquitetura (conferencia)"),
    "adm_ca": ("dados/contra_auditoria/rodada2_2026-09-13/ca-admissibilidade-params.json",
               "d6f3c111b7c5f38bc2a1bb1e16ceb6df81704b01acf3841451d382be60e41dbd", "CA T5/T6"),
    "fase0_q2": ("dados/fase0/fase0_metrics_bauru_s42_Q2.json",
                 "64a5fe01b89c06abf1d23be8ccb32647b10f53c64dc9be9ea1d3398430ebfcee", "fonte T6"),

    # Values cross-checked against the numbers quoted in ca-admissibilidade-params.json.
    "adm_bloq": ("dados/consolidacao/admissibilidade_bloqueada_2026-09-15.json",
                 "faa08924b046901a746d5d404459f56934af6666b94988f3351225b18f6d7b88", "fonte T6 v3"),
    "run_gnn_q1": ("treinos/c0c1cf_bauru_s42_Q1_g10b2/run_c0c1cf_bauru_s42_Q1_g10b2.json",
                   "552890b4dd972c6e7e4bb2193d03a7483b70f6d6bcedb2434c0401cbd9c5f7a2", "fonte T5 v3"),
    "run_gnn_q4": ("treinos/c0c1cf_bauru_s42_Q4_g10b2/run_c0c1cf_bauru_s42_Q4_g10b2.json",
                   "adddf2211d473a592fc18722c12eb4ec980b1d5da7512f18611ede57d7fec885", "fonte T5 v3"),
    "run_mlp_q1": ("treinos/mlpcf_bauru_s42_Q1_g10b2/run_mlpcf_bauru_s42_Q1_g10b2.json",
                   "417e2280782703a1d9c8473b2ee4c1a64dc24eb0fdd65af26494f597b690613d", "fonte T5 v3 (sha completo conferido contra agregado T11 :: fontes_sha256)"),
    "run_mlp_q4": ("treinos/mlpcf_bauru_s42_Q4_g10b2/run_mlpcf_bauru_s42_Q4_g10b2.json",
                   "c249a91b5a5436f9fc45fa3443da9b0662473874276445960651d57daab2ee5a", "fonte T5 v3 (sha completo conferido contra agregado T11 :: fontes_sha256)"),
    "tpl_blocked": (str(NOVAS / "tab_blocked_16_tres_pop.tex"),
                    "abf0aaf7553e298b6ee25cf4595fb670c2e914deb62b6eee2c90cd4167531ba6", "modelo T1"),
    "tpl_ablacao": (str(NOVAS / "tab_ablacao_estrutural.tex"),
                    "896d9918dde166f8661220e4ce9501e8ed2ea44ac4a827d9b3c941a6fa8afed7", "modelo T2"),
}

BLOCOS_MAIN = {
    "tab:physics_compliance": "144a291c8139c6b888ed9da420c71f4d12ef0fdd4c145e33f4999f447da1a9bc",
    "tab:baseline_arch": "2bc2dbce67fb4ace3edfe2c3281af521b254d47bbb5ac2927f9910eefa7f3631",
    "tab:cost": "c748b8245e722de3477df0bcf07a8b64a1cbfb521ddcf2d248da3893c9a7ae8c",
    "tab:ood_crosscity": "07e60b2d8d0b29d199de648ef15196675d6f70c51a78ac3e4bc37dd3bfee77dc",
}
SAIDAS = ["tab_blocked_16_tres_pop_v2.tex", "tab_ablacao_estrutural_v2.tex", "tab_ood_bloqueada.tex",
          "tab_contraste_1nn.tex", "tab_custo_v2.tex", "tab_arquitetura_v2.tex", "tab_admissibilidade_v2.tex",
          "tab_custo_v3.tex", "tab_admissibilidade_v3.tex"]

rastro = {"gerador": str(Path(__file__).resolve()), "gerado_em_utc": None, "fontes": {}, "blocos_main_tex": {},
          "guardas": [], "saidas": {}}


def guarda(nome, ok, detalhe):
    exige(ok, f"{nome}: {detalhe}")
    rastro["guardas"].append({"guarda": nome, "passou": True, "detalhe": detalhe})


def caminho(rel):
    p = Path(rel)
    return p if p.is_absolute() else EVID / rel


def carrega_json(chave):
    return json.loads(caminho(FONTES[chave][0]).read_text(encoding="utf-8"))


manifest = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
for k, (rel, esperado, papel) in FONTES.items():
    p = caminho(rel)
    exige(p.exists(), f"fonte ausente {p}")
    h = sha(p)
    guarda(f"G0_sha_{k}", h == esperado, f"{rel} sha256={h[:12]} (esperado {esperado[:12]})")
    rastro["fontes"][k] = {"caminho": str(p), "sha256": h, "papel": papel}


nn_ca = carrega_json("nn_ca")
guarda("G0_ca1nn_cita_fonte", any(FONTES["nn"][1] in a for a in nn_ca["alvo"]),
       "ca-contraste-1nn-16celulas.json::alvo contem o sha256 de contraste_1nn_vizinho_valido.json")

for k in ("t11_agregado", "t11_ca", "transf", "transf_ca", "nn", "nn_ca", "params", "adm_ca", "fase0_q2"):
    rel = FONTES[k][0].replace("\\", "/")
    regs = [m for m in manifest if m["artefato"].replace("\\", "/").endswith(rel)]
    exige(regs, f"{rel} sem entrada no manifest")
    guarda(f"G0_manifest_{k}", regs[-1]["sha256"] == FONTES[k][1], f"ultima entrada do manifest para {rel} tem o mesmo sha")
regs_lin = [m for m in manifest if m["artefato"].replace("\\", "/").endswith(FONTES["linhagem"][0])]
rastro["excecao_manifest_linhagem"] = {
    "sha_no_manifest": regs_lin[-1]["sha256"], "sha_atual": FONTES["linhagem"][1],
    "explicacao": "arquivo regravado pela reexecucao da contra-auditoria (timestamp_utc interno 02:11:12, registro 02:08:27; script com o mesmo sha). Valores usados conferidos contra params_checkpoints (sha igual ao manifest) e contra os numeros citados em ca-admissibilidade-params.json."}


main_txt = (R3 / "main.tex").read_text(encoding="utf-8")


def bloco_tabela(label):
    i = main_txt.find("\\label{" + label + "}")
    exige(i >= 0 and main_txt.count("\\label{" + label + "}") == 1, f"label {label} ausente ou repetido em main.tex")
    a = main_txt.rfind("\\begin{table", 0, i)
    b = main_txt.find("\\end{table}", i)
    exige(a >= 0 and b > 0, f"bloco de {label} nao delimitado")
    return main_txt[a:b + len("\\end{table}")].replace("\r\n", "\n")


BLOCO = {}
for lab, esperado in BLOCOS_MAIN.items():
    txt = bloco_tabela(lab)
    h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    if esperado != "PREENCHER":
        guarda(f"G0_bloco_{lab}", h == esperado, f"bloco {lab} de main.tex sha256={h[:12]}")
    else:
        exige(False, f"sha do bloco {lab} nao fixado: {h}")
    BLOCO[lab] = txt
    rastro["blocos_main_tex"][lab] = {"sha256_bloco_lf": h}


def fnum(x, d):
    """Formats a signed number in the R3 style: '+x' for positive values, '$-x$' for negative values."""
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        return f"$-{s[1:]}$"
    return "+" + s


def fint(x):
    """Rounds to an integer using round-half-up (matches the published tab:cost convention: 1510.5 -> 1511)."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def fpct(x):
    s = f"{x:.1f}"
    return (f"$-{s[1:]}$" if s.startswith("-") else "+" + s) + "\\%"


CID = ["bauru", "campinas", "lins", "sorocaba"]
Q = ["Q1", "Q2", "Q3", "Q4"]
saida_txt = {}


A = carrega_json("t11_agregado")
guarda("T1_metrica", A["metrica"]["campo"] == "selecao.test_no_melhor_ckpt.mae_rssi_db"
       and "TODOS os nos" in A["metrica"]["definicao"], "metrica do agregado = RSSI MAE, todos os nos de teste (mesma da coluna)")
cel = A["celulas"]
exige(len(cel) == 16, "agregado T11 sem 16 celulas")
tpl = FONTES["tpl_blocked"][0]
t1 = Path(tpl).read_text(encoding="utf-8").replace("\r\n", "\n")
linhas = t1.split("\n")
novas_linhas = []
n_rows = 0
for ln in linhas:
    m = re.match(r"^(Bauru|Campinas|Lins|Sorocaba) & (Q[1-4]) & (.*) \\\\$", ln)
    if not m:
        novas_linhas.append(ln)
        continue
    n_rows += 1
    cid, q = m.group(1).lower(), m.group(2)
    campos = m.group(3).split(" & ")
    exige(len(campos) == 9, f"linha {cid} {q} com {len(campos)} campos")
    c = cel[f"{cid}_{q}"]
    seeds = c["por_semente"]
    guarda(f"T1_seeds_{cid}_{q}", sorted(seeds) == ["s42", "s43", "s44", "s45", "s46"], "cinco sementes 42-46")
    mlp = float(np.mean([seeds[s]["mlp"] for s in sorted(seeds)]))
    gnn = float(np.mean([seeds[s]["gnn"] for s in sorted(seeds)]))
    exige(abs(mlp - c["mlp_mae_rssi_db_media"]) < 1e-12 and abs(gnn - c["gnn_mae_rssi_db_media"]) < 1e-12,
          f"{cid} {q}: media das sementes nao reproduz o agregado")
    sd_g = float(np.std([seeds[s]["gnn"] for s in sorted(seeds)], ddof=1))
    # The printed GNN column comes from the T15 re-inference (not the training log):
    # same cell and seeds, difference bounded to half a print unit plus inference non-determinism.
    pm, ps = (float(x) for x in campos[1].split(" $\\pm$ "))
    tol_imp = 0.0005 + TOL_REPRO
    guarda(f"T1_gnn_impresso_{cid}_{q}", abs(pm - gnn) <= tol_imp and abs(ps - sd_g) <= tol_imp,
           f"coluna GNN impressa '{campos[1]}' vs registro de treino {gnn:.5f}/{sd_g:.5f} (|d| <= {tol_imp}): mesma celula e sementes")
    if cid in ("lins", "bauru"):
        guarda(f"T1_mlp_impresso_{cid}_{q}", campos[4] == f"{mlp:.3f}",
               f"MLP ja impresso '{campos[4]}' reproduzido ({mlp:.6f})")
    else:
        exige(campos[4] == "--", f"{cid} {q}: coluna MLP do modelo nao e '--'")
        campos[4] = f"{mlp:.3f}"
    novas_linhas.append(f"{m.group(1)} & {q} & " + " & ".join(campos) + " \\\\")
exige(n_rows == 16, f"modelo T1 com {n_rows} linhas de celula")
d16 = [cel[f"{c}_{q}"]["delta_gnn_menos_mlp_db"] for c in CID for q in Q]
guarda("T1_delta_celula_reproduz_unidade", abs(np.mean(d16) - A["unidades"]["celula_n16"]["media_delta_db"]) < 1e-12,
       "media dos 16 deltas de celula == unidades.celula_n16.media_delta_db")
t1n = "\n".join(novas_linhas)
velho = "MLP is the capacity-matched control without message passing (all nodes; Lins and Bauru)."
novo = "MLP is the capacity-matched control without message passing (all nodes; mean over training seeds)."
exige(t1n.count(velho) == 1, "trecho da legenda T1 nao encontrado exatamente uma vez")
t1n = t1n.replace(velho, novo)
saida_txt["tab_blocked_16_tres_pop_v2.tex"] = t1n


U = A["unidades"]
guarda("T2_sinal_agregado", A["metrica"]["sinal"].startswith("delta = MAE(GNN) - MAE(MLP)"), "agregado usa GNN menos MLP")
t2 = Path(FONTES["tpl_ablacao"][0]).read_text(encoding="utf-8").replace("\r\n", "\n")
guarda("T2_sinal_tabela", "the paired difference GNN-RF minus a multilayer perceptron" in t2
       and "(positive means the MLP is more accurate)" in t2, "tabela usa GNN-RF menos MLP (positivo = MLP melhor)")
linha_velha = "GNN-RF minus capacity-matched MLP (1{,}812{,}515 params) & +0.011 & [$-0.126$; +0.148] & 0.8579 & 0.7422 \\\\"
exige(t2.count(linha_velha) == 1, "ultima linha do modelo T2 nao encontrada")
rec = A["conferencia_de_continuidade"]["recomputado"]
guarda("T2_n8_impresso_reproduz", fnum(rec["delta_db"], 3) == "+0.011" and fnum(rec["ic95_db"][0], 3) == "$-0.126$"
       and fnum(rec["ic95_db"][1], 3) == "+0.148" and f"{rec['p_bicaudal_t']:.4f}" == "0.8579",
       "linha n=8 ja impressa (+0.011; [-0.126; +0.148]; p=0.8579) reproduzida pelo agregado; sinal consistente")


def unidade_recalc(chave, n):
    u = U[chave]
    v = np.array(list(u["valores_db"].values()), dtype=float)
    exige(len(v) == n == u["n"], f"{chave}: n diverge")
    t = stats.ttest_1samp(v, 0.0)
    ep = v.std(ddof=1) / math.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    ic = (v.mean() - tc * ep, v.mean() + tc * ep)
    w = stats.wilcoxon(v).pvalue
    ok = (abs(v.mean() - u["media_delta_db"]) < 1e-12 and abs(t.pvalue - u["p_bicaudal_t"]) < 1e-9
          and abs(ic[0] - u["ic95_db"][0]) < 1e-9 and abs(ic[1] - u["ic95_db"][1]) < 1e-9
          and abs(w - u["wilcoxon_p_bicaudal"]) < 1e-9)
    guarda(f"T2_recalculo_{chave}", ok, f"media/IC95/p t/p Wilcoxon recalculados com scipy a partir de valores_db (n={n})")
    return u


rows2 = ["\\multicolumn{5}{l}{GNN-RF minus capacity-matched MLP (1{,}812{,}515 params), four cities} \\\\"]
for chave, n, rot in (("cidade_n4", 4, "city"), ("celula_n16", 16, "cell"), ("cadeia_n20", 20, "seed chain")):
    u = unidade_recalc(chave, n)
    rows2.append(f"\\quad unit: {rot} ($n={n}$) & {fnum(u['media_delta_db'], 3)} & "
                 f"[{fnum(u['ic95_db'][0], 3)}; {fnum(u['ic95_db'][1], 3)}] & "
                 f"{u['p_bicaudal_t']:.4f} & {u['wilcoxon_p_bicaudal']:.4f} \\\\")
guarda("T2_piso_n4", U["cidade_n4"]["p_minimo_nao_parametrico_possivel"] == 0.125 and abs(2 / 2 ** 4 - 0.125) < 1e-15,
       "piso nao parametrico bicaudal com n=4 = 2/2^4 = 0.125")
t2n = t2.replace(linha_velha, "\n".join(rows2))
cap_velha = ("Structural ablation on the spatially blocked test split (Lins and Bauru, Q1--Q4, seed~42 checkpoints; $n=8$ cells, "
             "three shuffling seeds per arm). Entries are the change in test RSSI MAE relative to the trained model with the real "
             "graph, paired by cell; positive means worse. The last row is a different contrast: the paired difference GNN-RF minus "
             "a multilayer perceptron with exactly the same number of trainable parameters, loss, split and selection rule and no "
             "message passing, averaged over five training seeds per cell (positive means the MLP is more accurate); equivalence "
             "within $\\pm0.1$~dB is not established (TOST $p=0.083$).")
cap_nova = ("Structural ablation on the spatially blocked test split. The four edge-ablation rows use Lins and Bauru (Q1--Q4, "
            "seed~42 checkpoints; $n=8$ cells, three shuffling seeds per arm); entries are the change in test RSSI MAE relative "
            "to the trained model with the real graph, paired by cell; positive means worse. The last three rows are a different "
            "contrast: the paired difference GNN-RF minus a multilayer perceptron with exactly the same number of trainable "
            "parameters, loss, split and selection rule and no message passing, in all four cities with five training seeds per "
            "cell, tested in the three pre-registered units (city; city--quadrant cell; seed chain, the mean of the four quadrants "
            "of one seed); negative means GNN-RF is more accurate. Cells and chains are not independent, since Q2--Q4 inherit the "
            "Q1 checkpoint of the same seed. With $n=4$ the smallest attainable two-sided non-parametric $p$ is 0.125.")
exige(t2n.count(cap_velha) == 1, "legenda T2 do modelo nao encontrada literalmente")
t2n = t2n.replace(cap_velha, cap_nova)
saida_txt["tab_ablacao_estrutural_v2.tex"] = t2n


T = carrega_json("transf")
reg = T["registros"]
exige(len(reg) == 12, "transferencia sem 12 registros")
labels = {r["checkpoint_origem_run_label"] for r in reg}
guarda("T3_checkpoints", labels == {f"c0c1cf_lins_s42_Q{i}_g10b2" for i in range(1, 5)}
       and len({r["checkpoint_origem_sha256"] for r in reg}) == 4
       and all(r["checkpoint_origem_run_label"] == f"c0c1cf_lins_s42_Q{r['quadrante']}_g10b2" for r in reg),
       "so os 4 checkpoints c0c1cf_lins_s42, cada um aplicado ao mesmo quadrante do alvo")
guarda("T3_particao", all(r["idx_sha256_global_test_ok"] is True
                          and r["idx_sha256_global_test_recomputado"] == r["idx_sha256_global_test_ref"] for r in reg),
       "hash da particao de teste bloqueada recomputada == run da celula alvo (12/12)")
guarda("T3_ablacao", all(r["colunas_zeradas_no_topo"] == list(range(8)) and r["n_terrain_cols"] == 18 for r in reg),
       "no_topo zera colunas 0-7 de 18 em todos os registros")
POPS = [("mae_rssi_total_db", 3), ("mae_rssi_valido_db", 2), ("mae_pl_valido_db", 2)]
res3 = {}
for cid in ("lins", "campinas", "sorocaba"):
    rr = sorted([r for r in reg if r["cidade_alvo"] == cid], key=lambda r: r["quadrante"])
    exige([r["quadrante"] for r in rr] == [1, 2, 3, 4], f"{cid}: quadrantes incompletos")
    ag = T["agregados_por_cidade"][cid]
    res3[cid] = {}
    for pop, _ in POPS:
        mf = float(np.mean([r["populacoes"]["full"][pop] for r in rr]))
        mn = float(np.mean([r["populacoes"]["no_topo"][pop] for r in rr]))
        de = ag["delta_e_percentual"][pop]
        pct = 100.0 * (mn - mf) / mf
        guarda(f"T3_media_{cid}_{pop}", abs(mf - de["mae_full_dB"]) < 1e-12 and abs(mn - de["mae_no_topo_dB"]) < 1e-12
               and abs(pct - de["pct_den_full"]) < 1e-9 and de["denominador_correto_para_o_verbo_raises_lowers"] == "full",
               f"{cid} {pop}: media nao ponderada Q1-Q4 e percentual (denominador = full) recalculados")
        res3[cid][pop] = (mf, mn, mn - mf, pct)
# cross-check: Lins full-quadrant MAE equals the seed-42 test MAE in the T11 aggregate (an independent artifact).
for r in reg:
    if r["cidade_alvo"] == "lins":
        q = f"Q{r['quadrante']}"
        g42 = cel[f"lins_{q}"]["por_semente"]["s42"]["gnn"]
        guarda(f"T3_cruzada_lins_{q}", abs(r["populacoes"]["full"]["mae_rssi_total_db"] - g42) < TOL_REPRO,
               f"Lins {q} full {r['populacoes']['full']['mae_rssi_total_db']:.6f} vs registro de treino s42 {g42:.6f} (tol {TOL_REPRO})")

guarda("T3_ca_percentuais", [round(res3[c]["mae_rssi_total_db"][3]) for c in ("lins", "campinas", "sorocaba")] == [591, 144, 142],
       "percentuais +591/+144/+142 citados em ca-transferencia-bloqueada reproduzidos")
rows3 = []
NOME = {"lins": "Lins (in-domain)", "campinas": "Campinas (OOD)", "sorocaba": "Sorocaba (OOD)"}
for j, cid in enumerate(("lins", "campinas", "sorocaba")):
    if j:
        rows3.append("\\midrule")
    full = " & ".join(f"{res3[cid][p][0]:.{d}f} & -- & --" for p, d in POPS)
    nt = " & ".join(f"{res3[cid][p][1]:.{d}f} & {fnum(res3[cid][p][2], d)} & {fpct(res3[cid][p][3])}" for p, d in POPS)
    rows3.append(f"{NOME[cid]} & full & {full} \\\\")
    rows3.append(f"{NOME[cid]} & no\\_topo & {nt} \\\\")
t3 = r"""\begin{table*}[t]
\caption{Cross-city transfer under the spatially blocked protocol. Lins checkpoints (seed~42, blocked campaign, one per quadrant) are applied without fine-tuning to the blocked test partition of the same quadrant in Lins (in-domain), Campinas and Sorocaba. The no\_topo arm zeroes the eight topographic input features of the terrain nodes at inference. Entries are the unweighted mean MAE over Q1--Q4 for three test populations: all nodes (sentinel included), covered nodes for RSSI, and covered nodes for path loss (PL). $\Delta$ is no\_topo minus full; the change is $\Delta$ divided by the full-arm MAE.}
\label{tab:ood_blocked}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ll ccc ccc ccc}
\toprule
 & & \multicolumn{3}{c}{RSSI, all nodes} & \multicolumn{3}{c}{RSSI, covered} & \multicolumn{3}{c}{PL, covered} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
Target city & Arm & MAE (dB) & $\Delta$ (dB) & Change & MAE (dB) & $\Delta$ (dB) & Change & MAE (dB) & $\Delta$ (dB) & Change \\
\midrule
""" + "\n".join(rows3) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
saida_txt["tab_ood_bloqueada.tex"] = t3


N = carrega_json("nn")
cs = N["celulas"]
exige(len(cs) == 16 and N["abortado"] is None, "1-NN: 16 celulas sem aborto")
guarda("T4_definicao", N["parametros"]["vizinho_candidato"] == "treino com PL < 299 dB"
       and all(c["mae_1nn_aleatorio"]["restricao_vizinho_valido"] is True and c["mae_1nn_bloqueado"]["restricao_vizinho_valido"] is True for c in cs),
       "definicao com vizinho valido (candidato = treino com PL < 299 dB) nas duas particoes")
guarda("T4_presente", all(c["leitura_criterio"] == "presente" for c in cs) and N["agregado_geral"]["veredito_conjunto"] == "PRESENTE",
       "16/16 celulas 'presente'")
guarda("T4_particao", all(all(c["particao_bloqueada_conferencia"][k]["confere"] is True for k in ("train", "val", "test")) for c in cs),
       "particao bloqueada reproduz idx_sha256_global do run c0c1cf nas 16 celulas")
guarda("T4_sentinela", all(c["mae_1nn_aleatorio"]["n_pred_sentinela"] == 0 and c["mae_1nn_bloqueado"]["n_pred_sentinela"] == 0 for c in cs),
       "nenhuma predicao copia sentinela")
por = {}
for c in cs:
    ra, rb = c["mae_1nn_aleatorio"]["mae_db"], c["mae_1nn_bloqueado"]["mae_db"]
    exige(abs(rb / ra - c["razao_R"]) < 1e-9, f"{c['cidade']} {c['quadrante']}: razao nao reproduz")
    exige(c["mae_1nn_aleatorio"]["n_test_valido_amostrado"] == c["mae_1nn_bloqueado"]["n_test_valido_amostrado"] >= 1000,
          f"{c['cidade']} {c['quadrante']}: n amostrado")
    por[(c["cidade"], c["quadrante"])] = (ra, rb, c["razao_R"])
exige(sorted(por) == sorted((a, b) for a in CID for b in Q), "1-NN: celulas faltando")


def resumo(v):
    return float(np.median(v)), float(min(v)), float(max(v))


def confere_resumo(nome, v, ag):
    m, lo, hi = resumo(v)
    return abs(m - ag["mediana"]) < 1e-12 and abs(lo - ag["min"]) < 1e-12 and abs(hi - ag["max"]) < 1e-12 and ag["n"] == len(v)


for grupo, cids in [(c, [c]) for c in CID] + [("geral", CID)]:
    ag = N["agregado_geral"] if grupo == "geral" else N["agregado_por_cidade"][grupo]
    for i, campo in enumerate(("mae_aleatorio_db", "mae_bloqueado_db", "razao_R")):
        v = [por[(c, q)][i] for c in cids for q in Q]
        guarda(f"T4_resumo_{grupo}_{campo}", confere_resumo(grupo, v, ag[campo]), f"mediana/min/max recalculados ({grupo}, {campo})")
g = N["agregado_geral"]
guarda("T4_numeros_da_ca", (f"{g['mae_bloqueado_db']['min']:.2f}", f"{g['mae_bloqueado_db']['max']:.2f}", f"{g['mae_bloqueado_db']['mediana']:.2f}",
                            f"{g['mae_aleatorio_db']['min']:.2f}", f"{g['mae_aleatorio_db']['max']:.2f}", f"{g['mae_aleatorio_db']['mediana']:.2f}",
                            f"{g['razao_R']['min']:.2f}", f"{g['razao_R']['max']:.2f}", f"{g['razao_R']['mediana']:.2f}")
       == ("2.01", "7.03", "4.14", "0.16", "1.08", "0.41", "3.13", "40.27", "10.45"),
       "faixas e medianas citadas em ca-contraste-1nn-16celulas reproduzidas")
rows4 = []
for c in CID:
    for q in Q:
        ra, rb, R = por[(c, q)]
        rows4.append(f"{c.capitalize()} & {q} & {ra:.2f} & {rb:.2f} & {R:.1f} \\\\")
rows4.append("\\midrule")
for grupo, cids in [(c, [c]) for c in CID] + [("geral", CID)]:
    vs = [[por[(c, q)][i] for c in cids for q in Q] for i in range(3)]
    rs = [resumo(v) for v in vs]
    nome = "All cells" if grupo == "geral" else grupo.capitalize()
    if grupo == "geral":
        rows4.append("\\midrule")
    rows4.append(f"{nome} & median & {rs[0][0]:.2f} & {rs[1][0]:.2f} & {rs[2][0]:.1f} \\\\")
    rows4.append(f" & range & {rs[0][1]:.2f}--{rs[0][2]:.2f} & {rs[1][1]:.2f}--{rs[1][2]:.2f} & {rs[2][1]:.1f}--{rs[2][2]:.1f} \\\\")
t4 = r"""\begin{table}[t]
\caption{Nearest-neighbour (1-NN) path-loss predictor under a random and a spatially blocked node partition, 16 city--quadrant cells (seed~42). The prediction for a test node is the path loss of the nearest training node with a valid path loss; MAE is computed on up to \num{20000} sampled test nodes with a valid path loss per cell. The random partition has the same training and test sizes as the blocked partition. Ratio is blocked over random MAE in each cell; summary rows give the median and range of each column. Cells share a seed and adjacent tiles, so medians and ranges are shown instead of confidence intervals.}
\label{tab:nn_contrast}
\centering
\small
\begin{tabular}{llccc}
\toprule
 & & \multicolumn{2}{c}{1-NN MAE (dB)} & \\
\cmidrule(lr){3-4}
City & Q & Random & Blocked & Ratio \\
\midrule
""" + "\n".join(rows4) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
saida_txt["tab_contraste_1nn.tex"] = t4


L = carrega_json("linhagem")
PC = carrega_json("params")
ADM = carrega_json("adm_ca")
it = {i["achado_id"]: i for i in ADM["itens"]}
guarda("T5_ca_item1_item2", it["item1_linhagem_mae_sem_tt_msg"]["refutado"] is False
       and it["item2_tempos_alternativos_version_20_1"]["refutado"] is False, "CA: itens 1 e 2 nao refutados")
MODS = [("radiounet", "RadioUNet"), ("rem_gnn", "REM-GNN"), ("radiogat", "RadioGAT"), ("rme_gan", "RME-GAN")]
ob = L["opcao_B_manter_parametros_trocar_tempos"]
motivo2 = it["item2_tempos_alternativos_version_20_1"]["motivo"]
bloco_cost = BLOCO["tab:cost"]
PUB = {}
for m in re.finditer(r"^(RadioUNet|REM-GNN|RadioGAT|RME-GAN|GNN-RF)\s*& \\num\{(\d+)\}\s*& ([\d.]+) & (\d+) & (\d+) \\\\$", bloco_cost, re.M):
    PUB[m.group(1)] = (int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5)))
exige(len(PUB) == 5, f"tab:cost com {len(PUB)} linhas reconhecidas")
ck = PC["checkpoints"]


def ck_de(modelo, conjunto, q):
    r = [c for c in ck if c["modelo"] == modelo and c["conjunto_de_treinos"] == conjunto and c["quadrante"] == q]
    exige(len(r) == 1, f"params_checkpoints: {modelo}/{conjunto}/{q} tem {len(r)} entradas")
    return r[0]


rows5 = []
for chave, nome in MODS:
    o = ob[chave]
    params_pub = PUB[nome][0]
    guarda(f"T5_params_{chave}", o["params"] == params_pub, f"parametro publicado {params_pub} mantido == opcao B")
    guarda(f"T5_linhagem_{chave}", o["linhagem"] == "SEM tt_msg (version_20.1)", "opcao B = version_20.1, sem tt_msg")
    for q, campo in (("Q1", "Q1_s"), ("Q4", "Q4_s")):
        c = ck_de(chave, "version_20.1", q)
        comp = c["treinaveis_por_componente"]
        n_modelo = comp["generator_state_dict"] if chave == "rme_gan" else comp["model_state_dict"]
        guarda(f"T5_ckpt_{chave}_{q}", c["tem_tt_msg"] is False and n_modelo == params_pub
               and c["tempo_observado_s"] == o[campo] == c["computational_cost_irmao"]["conteudo"]["total_training_s"],
               f"{chave} {q}: checkpoint version_20.1 sem tt_msg, {n_modelo} params, tempo {o[campo]} s no computational_cost irmao")
        exige(f"{o[campo]}" in motivo2, f"tempo {o[campo]} nao citado na CA item 2")
    size = f"{params_pub * 4 / 2 ** 20:.2f}"
    guarda(f"T5_size_{chave}", size == PUB[nome][1] == f"{o['size_mb']:.2f}", f"tamanho {size} MB = 4 bytes x params")

    oa = L["opcao_A_manter_tempos_trocar_parametros"][chave]
    guarda(f"T5_tempo_antigo_{chave}", (fint(oa['Q1_s']), fint(oa['Q4_s'])) == (str(PUB[nome][2]), str(PUB[nome][3]))
           and "version_20.3" in oa["linhagem"], "tempo hoje publicado = version_20.3 (com tt_msg)")
    rows5.append(f"{nome:<9} & \\num{{{params_pub}}}{' ' * (7 - len(str(params_pub)))}& {PUB[nome][1]} & {fint(o['Q1_s'])} & {fint(o['Q4_s'])} \\\\")

# The 0.349 dB Bauru Q2 figure, alongside the baselines, comes from the v20 multiseed lineage.

LG = L["linhagem_dos_mae_publicados"]["gnn"]
guarda("T5_gnn_conjunto_do_erro", LG["mae_rssi_db_publicado"] == 0.349 and f"{LG['mae_rssi_db_no_artefato']:.3f}" == "0.349"
       and "GNN_RF_V2/logs/version_20" in LG["conjunto_de_treinos"] and "multiseed_bauru_s42_Q2" in LG["checkpoint"]
       and LG["tem_tt_msg"] is False, "erro 0.349 dB publicado sai do checkpoint v20_multiseed Q2 (sem tt_msg)")
gq = {}
buf = set()
for q in ("Q1", "Q4"):
    c = ck_de("gnn_rf", "v20_multiseed", q)
    cc = c["computational_cost_irmao"]["conteudo"]
    guarda(f"T5_gnn_{q}", c["treinaveis"] == PUB["GNN-RF"][0] == 1812515 == cc["n_params"] and c["tem_tt_msg"] is False
           and c["total_state_dict"] - c["elementos_em_buffers_bn"] == c["treinaveis"]
           and "multiseed_bauru_s42_" + q in c["caminho"]
           and c["tempo_observado_s"] == cc["total_training_s"],
           f"GNN-RF {q}: v20_multiseed, {c['treinaveis']} treinaveis ({c['total_state_dict']} no state_dict menos {c['elementos_em_buffers_bn']} de buffers BN), "
           f"{c['tempo_observado_s']} s no computational_cost irmao (sha {c['computational_cost_irmao']['sha256'][:12]})")
    buf.add(c["elementos_em_buffers_bn"])
    gq[q] = c["tempo_observado_s"]

    guarda(f"T5_gnn_size_unidade_{q}", f"{cc['model_size_mb']:.2f}" == f"{1812515 * 4 / 1e6:.2f}",
           "model_size_mb 7.25 do artefato = 4 bytes x params / 1e6 (unidade decimal); tabela mantem 4 bytes / 2^20")
guarda("T5_gnn_Q2_avaliado", LG["params_no_state_dict"]["model_state_dict"] - buf.pop() == 1812515 and not buf
       and LG["n_params_declarado_pelo_fase0"] == 1812515,
       "checkpoint avaliado (Q2): 1814311 no state_dict menos 1796 de buffers BN = 1812515")
rastro["nota_tempos_gnn_rf"] = ("os tempos 5269.6/3034.2 s nao sao citados literalmente pela CA (item 2 cobre os baselines); "
                                "vem de params_checkpoints (sha = manifest, reproduzido pela CA item 1) e do computational_cost irmao")
guarda("T5_gnn_publicado_era_c0c1cf", (fint(ck_de("gnn_rf", "c0c1cf_bloqueada", "Q1")["tempo_observado_s"]),
                                        fint(ck_de("gnn_rf", "c0c1cf_bloqueada", "Q4")["tempo_observado_s"])) == (str(PUB["GNN-RF"][2]), str(PUB["GNN-RF"][3])),
       "tempos hoje publicados (3987/2323) sao da campanha bloqueada c0c1cf: trocados")
guarda("T5_gnn_size", f"{1812515 * 4 / 2 ** 20:.2f}" == PUB["GNN-RF"][1], "tamanho GNN-RF 6.91 MB (4 bytes / 2^20)")
rows5.append(f"GNN-RF    & \\num{{{PUB['GNN-RF'][0]}}} & {PUB['GNN-RF'][1]} & {fint(gq['Q1'])} & {fint(gq['Q4'])} \\\\")
guarda("T5_forma_das_linhas", len(rows5) == 5 and all(
    re.fullmatch(r"(RadioUNet|REM-GNN|RadioGAT|RME-GAN|GNN-RF) *& \\num\{\d+\} *& \d+\.\d\d & \d+ & \d+ \\\\", r) for r in rows5),
    "as 5 linhas da tab:cost tem a forma 'Modelo & \\num{N} & x.xx & s & s \\\\' (sem escape quebrado)")
rastro["hardware_registrado"] = {
    "baselines_version_20.1": "computational_cost.json so registra device='cuda'; modelo de GPU nao registrado",
    "gnn_rf_v20_multiseed": "computational_cost.json sem campo de dispositivo; modelo de GPU nao registrado",
    "nota": "o 'NVIDIA GeForce RTX 5070 Ti' dos run JSON c0c1cf descreve a campanha bloqueada, nao estes conjuntos"}
t5 = r"""\begin{table}[!t]
\centering
\caption{Computational cost per model (Bauru, seed~42). All rows come from the checkpoints of the earlier campaign evaluated in the Bauru~Q2 comparison: parameters and training times are read from the Q1 and Q4 checkpoints of the same training set. The RME-GAN count is that of the generator only. Q1 trains from scratch; Q4 transfers. Size assumes 4 bytes per parameter.}
\label{tab:cost}
\begin{tabular}{lrrrr}
\toprule
Model & Params & Size (MB) & Q1 (s) & Q4 (s) \\
\midrule
""" + "\n".join(rows5) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
saida_txt["tab_custo_v2.tex"] = t5


ba = BLOCO["tab:baseline_arch"]
frase_velha = "Parameters counted on the trained Bauru checkpoints; the other cities add a terrain-to-terrain message-passing step."
exige(ba.count(frase_velha) == 1, "frase da legenda de tab:baseline_arch nao encontrada")
tt_bauru = {c["modelo"] for c in ck if c["cidade"] == "bauru" and c["tem_tt_msg"] is True}
guarda("ARQ_tt_msg_e_bauru", tt_bauru == {"radiounet", "rem_gnn", "radiogat", "rme_gan"},
       "as contagens com passo terreno-terreno (397186/73554/375938/73206) sao de checkpoints de BAURU (version_20.2/20.3), nao de outras cidades")
for chave, nome in MODS:
    mm = re.search(r"^" + re.escape(nome) + r"\s*& \\num\{(\d+)\}\s*& \\SI\{([\d.]+)\}\{\\mega\\byte\}", ba, re.M)
    exige(mm is not None, f"linha {nome} de tab:baseline_arch")
    guarda(f"ARQ_{chave}", int(mm.group(1)) == ob[chave]["params"] and mm.group(2) == f"{ob[chave]['size_mb']:.2f}",
           f"{nome}: params/tamanho da arquitetura == checkpoints version_20.1 (sem tt_msg)")
frase_nova = ("Parameters counted on the Bauru seed-42 checkpoints whose errors are reported (training set without the "
              "terrain-to-terrain step); the RME-GAN count is that of the generator only.")
saida_txt["tab_arquitetura_v2.tex"] = ba.replace(frase_velha, frase_nova) + "\n"


F0 = carrega_json("fase0_q2")
it3 = it["item3_admissibilidade_contagens_e_divergencia_6_vs_4"]
guarda("T6_ca_item3", it3["refutado"] is False, "CA item 3 nao refutado (contagens 2/0/0/0/5 e 6 para GNN-RF)")
mods6 = [("gnn", "GNN-RF"), ("radiounet", "RadioUNet"), ("radiogat", "RadioGAT"), ("rem_gnn", "REM-GNN"), ("rme_gan", "RME-GAN")]
Ns = {F0["modelos"][k]["todos_os_nos"]["n_com_piso"] for k, _ in mods6}
guarda("T6_N", Ns == {1225847}, "n_com_piso = 1225847 para todos os modelos")
Nn = 1225847


def contagem(frac):
    x = Nn * (1.0 - frac)
    k = round(x)
    exige(abs(x - k) < 1e-3, f"fracao {frac} nao corresponde a contagem inteira sobre {Nn} (x={x})")
    return int(k)


cont = {k: contagem(F0["modelos"][k]["todos_os_nos"]["fspl_floor_fraction"]) for k, _ in mods6}
alvos = {F0["modelos"][k]["todos_os_nos"]["alvo_fspl_floor_fraction"] for k, _ in mods6}
exige(len(alvos) == 1, "fracao do campo de referencia difere entre modelos")
cont["ref"] = contagem(alvos.pop())
guarda("T6_contagens_ca", (cont["radiounet"], cont["radiogat"], cont["rem_gnn"], cont["rme_gan"], cont["ref"], cont["gnn"]) == (2, 0, 0, 0, 5, 6),
       "contagens 2/0/0/0/5 e GNN-RF 6 reproduzidas (CA item 3)")
bpc = BLOCO["tab:physics_compliance"]
pub6 = dict(re.findall(r"^(GNN-RF|RadioUNet|RadioGAT|REM-GNN|RME-GAN|Reference field) & ([\d.]+) \\\\$", bpc, re.M))
exige(len(pub6) == 6, "tab:physics_compliance com linhas nao reconhecidas")
guarda("T6_publicado_literal", pub6["RadioUNet"] == "0.9999984" and pub6["Reference field"] == "0.9999959"
       and pub6["GNN-RF"] == "0.9999967" and all(pub6[n] == "1.000" for n in ("RadioGAT", "REM-GNN", "RME-GAN")),
       "fracoes hoje impressas em tab:physics_compliance sao as esperadas")

pub_cont = {n: round(Nn * (1 - float(v))) for n, v in pub6.items()}
guarda("T6_publicado_vs_artefato", pub_cont["RadioUNet"] == cont["radiounet"] and pub_cont["Reference field"] == cont["ref"]
       and pub_cont["RadioGAT"] == pub_cont["REM-GNN"] == pub_cont["RME-GAN"] == 0 and pub_cont["GNN-RF"] == 4 and cont["gnn"] == 6,
       "fracoes publicadas dao 2/5/0/0/0 (batem) e 4 para GNN-RF (artefato arquivado da 6: divergencia documentada pela CA)")
guarda("T6_ordem", [n for n in re.findall(r"^(GNN-RF|RadioUNet|RadioGAT|REM-GNN|RME-GAN|Reference field) &", bpc, re.M)]
       == ["GNN-RF", "RadioUNet", "RadioGAT", "REM-GNN", "RME-GAN", "Reference field"], "ordem das linhas preservada (sem ordenacao)")
t6 = r"""\begin{table}[!t]
\centering
\caption{Free-space-floor admissibility (Bauru~Q2, seed-42 checkpoints of the earlier campaign; the \num{1225847} nodes with an explicit antenna link). Number of node predictions below the free-space floor, recomputed per link at its carrier frequency; the last row is the reference field itself. For GNN-RF the archived evaluation gives %d, and an earlier evaluation that could not be traced to an archived artifact gave fewer, so the larger value is reported as a bound.}
\label{tab:physics_compliance}
\begin{tabular}{lc}
\toprule
Model / field & Nodes below floor (of \num{1225847}) \\
\midrule
GNN-RF & at most %d \\
RadioUNet & %d \\
RadioGAT & %d \\
REM-GNN & %d \\
RME-GAN & %d \\
Reference field & %d \\
\bottomrule
\end{tabular}
\end{table}
""" % (cont["gnn"], cont["gnn"], cont["radiounet"], cont["radiogat"], cont["rem_gnn"], cont["rme_gan"], cont["ref"])
saida_txt["tab_admissibilidade_v2.tex"] = t6


rastro["declaracao_do_dono_hardware"] = ("o dono declara que o hardware e o mesmo em todas as corridas da campanha bloqueada "
                                         "(registrado como declaracao, 2026-09-16)")

# cell and seed match the currently published GNN-RF row (3987/2323 s = c0c1cf_bauru_s42_Q1/Q4).
guarda("T5v3_celula_publicada", (fint(ck_de("gnn_rf", "c0c1cf_bloqueada", "Q1")["tempo_observado_s"]),
                                 fint(ck_de("gnn_rf", "c0c1cf_bloqueada", "Q4")["tempo_observado_s"]))
       == (str(PUB["GNN-RF"][2]), str(PUB["GNN-RF"][3]))
       and all(ck_de("gnn_rf", "c0c1cf_bloqueada", q)["cidade"] == "bauru" and ck_de("gnn_rf", "c0c1cf_bloqueada", q)["seed"] == 42
               for q in ("Q1", "Q4")),
       "linha GNN-RF publicada (3987/2323 s) = c0c1cf Bauru semente 42 Q1/Q4; mesma celula mantida")
fsa = A["fontes_sha256"]
RUNS = {}
for arm, chave in (("c0c1cf", "run_gnn"), ("mlpcf", "run_mlp")):
    for q in ("Q1", "Q4"):
        k = f"{chave}_{q.lower()}"
        rel = FONTES[k][0]
        guarda(f"T5v3_sha_cruzado_{k}", fsa[rel] == FONTES[k][1], f"{rel}: sha fixado == agregado T11 :: fontes_sha256 (pareamento contra-auditado)")
        r = carrega_json(k)
        exige(r["run_label"] == f"{arm}_bauru_s42_{q}_g10b2" and r["seed"] == 42, f"{k}: run_label/seed")
        RUNS[(arm, q)] = r
g1 = ck_de("gnn_rf", "c0c1cf_bloqueada", "Q1")
guarda("T5v3_sha_run_gnn", g1["run_json"]["sha256"] == FONTES["run_gnn_q1"][1]
       and ck_de("gnn_rf", "c0c1cf_bloqueada", "Q4")["run_json"]["sha256"] == FONTES["run_gnn_q4"][1],
       "run JSON c0c1cf Q1/Q4 = os lidos por params_checkpoints (contagem no checkpoint)")
gpus = set()
lin3 = []
for arm, nome, classe in (("c0c1cf", "GNN-RF", "GNNRFModel"), ("mlpcf", "MLP", "MLPRFModel")):
    rq = {q: RUNS[(arm, q)] for q in ("Q1", "Q4")}
    for q, r in rq.items():
        mo, cu = r["modelo"], r["custo"]
        guarda(f"T5v3_params_{arm}_{q}", mo["classe"] == classe and mo["n_params"] == mo["n_params_treinaveis"] == 1812515,
               f"{arm} {q}: {classe}, {mo['n_params_treinaveis']} parametros treinaveis")
        guarda(f"T5v3_custo_{arm}_{q}", isinstance(cu["tempo_total_s"], float) and isinstance(cu["vram_peak_mb"], float)
               and cu["n_epocas_rodadas"] == (35 if q == "Q1" else 20),
               f"{arm} {q}: custo.tempo_total_s={cu['tempo_total_s']}, vram_peak_mb={cu['vram_peak_mb']}, epocas={cu['n_epocas_rodadas']}")
        gpus.add(r["ambiente"]["gpu"])
    if arm == "c0c1cf":
        guarda("T5v3_gnn_ckpt", g1["treinaveis"] == 1812515 and g1["tempo_do_run_json_s"] == rq["Q1"]["custo"]["tempo_total_s"],
               "GNN-RF: 1812515 tambem contados no state_dict do checkpoint (params_checkpoints)")
    else:
        guarda("T5v3_mlp_alvo", all(rq[q]["modelo"]["n_params_alvo_gnn"] == 1812515 for q in rq), "MLP construido com alvo de 1812515 parametros")
    lin3.append(f"{nome} & \\num{{1812515}} & {1812515 * 4 / 2 ** 20:.2f} & {fint(rq['Q1']['custo']['tempo_total_s'])} & "
                f"{fint(rq['Q4']['custo']['tempo_total_s'])} & {fint(rq['Q1']['custo']['vram_peak_mb'])} & {fint(rq['Q4']['custo']['vram_peak_mb'])} \\\\")
guarda("T5v3_gpu_unica", len(gpus) == 1, f"as 4 corridas registram a mesma GPU: {sorted(gpus)}")
GPU = gpus.pop()

gp_all = set()
n_all = 0
for rel, h in fsa.items():
    if rel.startswith("treinos/") and ("/run_c0c1cf_" in rel or "/run_mlpcf_" in rel):
        p = EVID / rel
        exige(sha(p) == h, f"{rel}: sha diverge do agregado T11")
        gp_all.add(json.loads(p.read_text(encoding="utf-8"))["ambiente"]["gpu"])
        n_all += 1
guarda("T5v3_gpu_todas", n_all == 160 and gp_all == {GPU}, f"{n_all} run JSON (80 c0c1cf + 80 mlpcf) registram a mesma GPU {GPU}")
for r in lin3:
    exige(re.fullmatch(r"(GNN-RF|MLP) & \\num\{1812515\} & 6\.91 & \d+ & \d+ & \d+ & \d+ \\\\", r), f"forma da linha: {r}")
t5v3 = r"""\begin{table}[!t]
\centering
\caption{Computational cost in the spatially blocked campaign (Bauru, seed~42): GNN-RF and the capacity-matched control without message passing (MLP). Both have the same number of trainable parameters. Q1 trains from scratch (35 epochs); Q4 transfers (20 epochs). All runs of the blocked campaign used the same hardware (%s). Size assumes 4 bytes per parameter ($2^{20}$ bytes per MB); peak VRAM is the maximum memory allocated by PyTorch during training ($10^{6}$ bytes per MB).}
\label{tab:cost}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}
\toprule
 & & & \multicolumn{2}{c}{Time (s)} & \multicolumn{2}{c}{Peak VRAM (MB)} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Model & Params & Size (MB) & Q1 & Q4 & Q1 & Q4 \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (GPU, "\n".join(lin3))
saida_txt["tab_custo_v3.tex"] = t5v3


AB = carrega_json("adm_bloq")
it4 = it["item4_medida_bloqueada_media_0_99996_69_de_80"]
guarda("T6v3_ca_item4", it4["refutado"] is False and "2536" in it4["motivo"] and "65930590" in it4["motivo"]
       and "69/80" in it4["motivo"], "CA item 4 nao refutado e cita 69/80, 2536 e 65930590")
guarda("T6v3_definicao", "dist_nearest_m" in AB["definicao"] and "1800 MHz" in AB["definicao"] and "TESTE bloqueada" in AB["definicao"]
       and AB["campo_medido"] == "selecao.test_no_melhor_ckpt.diag.physics_fspl_compliance",
       "definicao: pl_pred >= FSPL(distancia ao transmissor mais proximo, 1800 MHz), nos de teste da particao bloqueada")
cor = AB["corridas"]
exige(len(cor) == AB["n_corridas"] == 80, "admissibilidade: 80 corridas")
guarda("T6v3_corridas", sorted((c["cidade"], c["quadrante"], c["seed"]) for c in cor)
       == sorted((c, q, s) for c in CID for q in Q for s in (42, 43, 44, 45, 46))
       and all(c["run_label"] == f"c0c1cf_{c['cidade']}_s{c['seed']}_{c['quadrante']}_g10b2" for c in cor),
       "80 corridas c0c1cf = 4 cidades x 4 quadrantes x 5 sementes, sem repeticao")
fontes_ab = {Path(f["caminho"]).name: f["sha256"] for f in AB["fontes"]}
guarda("T6v3_run_json_sha", all(fontes_ab[f"run_{c['run_label']}.json"] == c["run_json_sha256"]
                                == fsa[f"treinos/{c['run_label']}/run_{c['run_label']}.json"] for c in cor),
       "sha de cada run JSON lido pela medida == sha do agregado T11 (80/80)")
por_cid = {c: {"limpas": 0, "abaixo": 0, "n": 0, "runs": 0} for c in CID}
for c in cor:
    f, n = c["physics_fspl_compliance_test"], c["n_nos_test"]
    exige(isinstance(f, float) and 0.0 <= f <= 1.0 and isinstance(n, int) and n > 0, f"{c['run_label']}: fracao/n invalidos")
    x = (1.0 - f) * n
    k = round(x)

    exige(abs(x - k) <= n * 2 ** -24, f"{c['run_label']}: (1-f)*n = {x} nao e inteiro dentro da resolucao float32")
    d = por_cid[c["cidade"]]
    d["runs"] += 1
    d["n"] += n
    d["abaixo"] += k
    d["limpas"] += int(f == 1.0)
tot = {k: sum(por_cid[c][k] for c in CID) for k in ("limpas", "abaixo", "n", "runs")}
guarda("T6v3_totais", tot == {"limpas": 69, "abaixo": 2536, "n": 65930590, "runs": 80}
       and AB["agregados"]["n_corridas_exatamente_1.0"] == 69 and AB["n_nos_test_somados"] == 65930590,
       "totais 69/80 corridas sem no abaixo do piso, 2536 de 65930590 nos (== artefato e CA)")
guarda("T6v3_media_ca", f"{AB['agregados']['global_media_por_corrida']:.9f}" == "0.999961510", "media por corrida 0.999961510 (CA)")
for c in CID:
    guarda(f"T6v3_cidade_{c}", por_cid[c]["runs"] == 20 == AB["por_cidade"][c]["n_corridas"]
           and (por_cid[c]["limpas"] == 20) == (AB["por_cidade"][c]["min"] == 1.0),
           f"{c}: 20 corridas; {por_cid[c]['limpas']} sem no abaixo do piso, {por_cid[c]['abaixo']} nos abaixo de {por_cid[c]['n']}")
rows6 = [f"{c.capitalize()} & {por_cid[c]['limpas']} of {por_cid[c]['runs']} & \\num{{{por_cid[c]['abaixo']}}} & \\num{{{por_cid[c]['n']}}} \\\\" for c in CID]
rows6.append("\\midrule")
rows6.append(f"Total & {tot['limpas']} of {tot['runs']} & \\num{{{tot['abaixo']}}} & \\num{{{tot['n']}}} \\\\")
t6v3 = r"""\begin{table}[!t]
\centering
\caption{Free-space-floor admissibility of GNN-RF under the spatially blocked protocol (80 runs: four cities, Q1--Q4, five seeds; checkpoint of lowest validation RSSI MAE). A test node is below the floor when its predicted path loss is lower than the free-space loss at the distance to the nearest transmitter and 1800~MHz; the population is the test partition of the blocked split. Node counts are recovered from the recorded per-run fraction and number of test nodes.}
\label{tab:physics_compliance}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
City & Runs with no node & Nodes below & Test nodes \\
 & below the floor & the floor & evaluated \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(rows6)
saida_txt["tab_admissibilidade_v3.tex"] = t6v3


exige(sorted(saida_txt) == sorted(SAIDAS), "conjunto de saidas diverge do declarado")
existentes_protegidas = {p.name for p in NOVAS.glob("*.tex")} - set(SAIDAS)
for nome in SAIDAS:
    exige(nome not in existentes_protegidas, f"{nome} colidiria com tabela existente")
    alvo = NOVAS / nome
    if alvo.exists():  # only overwrites its own output, as registered in the manifest by this generator.
        exige(any(m["artefato"].replace("\\", "/").endswith("REVISAO_R3/novas/" + nome)
                  and m.get("origem", "").startswith("gerar_tabelas_rodada_escrita_2026-09-16.py") for m in manifest),
              f"{nome} ja existe e nao e saida registrada deste gerador")
for nome in SAIDAS:
    txt = saida_txt[nome].replace("\r\n", "\n")
    (NOVAS / nome).write_bytes(txt.replace("\n", "\r\n").encode("utf-8"))
    rastro["saidas"][nome] = {"caminho": str(NOVAS / nome), "sha256": sha(NOVAS / nome)}
rastro["gerado_em_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
rastro["ambiente"] = {"python": sys.version.split()[0], "executavel": sys.executable, "numpy": np.__version__,
                      "scipy": __import__("scipy").__version__, "cuda_usada": False}
RASTRO.write_text(json.dumps(rastro, indent=1, ensure_ascii=False), encoding="utf-8")


reg_novos = []
agora = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
gerador_sha = sha(Path(__file__).resolve())
ja = {(m["artefato"].replace("\\", "/"), m["sha256"]) for m in manifest}
candidatos = [(str(Path(__file__).resolve()), gerador_sha, "script", "gerador das tabelas da rodada de escrita"),
              (str(RASTRO), sha(RASTRO), "rastro", "hashes das fontes, blocos de main.tex e guardas que passaram")]
md = INV / "gerar_tabelas_rodada_escrita_2026-09-16.md"
if md.exists():
    candidatos.append((str(md), sha(md), "nota", "escolhas de forma (at most 6, linha GNN-RF de custo, legendas)"))
for nome in SAIDAS:
    candidatos.append((rastro["saidas"][nome]["caminho"], rastro["saidas"][nome]["sha256"], "tabela_latex", nome))
for art, h, tipo, nota in candidatos:
    if (art.replace("\\", "/"), h) in ja:
        continue
    reg_novos.append({"artefato": art, "sha256": h, "origem": f"gerar_tabelas_rodada_escrita_2026-09-16.py (sha {gerador_sha[:12]})",
                      "categoria": "tabelas_rodada_escrita", "tipo": tipo, "status": "gerado_de_contra_auditado",
                      "citabilidade": "numeros copiados de artefatos contra-auditados (ver rastro); tabela ainda nao revisada pelo forum editorial",
                      "fontes_sha256": {k: v["sha256"] for k, v in rastro["fontes"].items()},
                      "nota": nota, "autor": "agente forum-eng-dados", "timestamp_utc": agora})

lin_rel = FONTES["linhagem"][0]
if regs_lin[-1]["sha256"] != FONTES["linhagem"][1]:
    reg_novos.append({
        "artefato": lin_rel, "sha256": FONTES["linhagem"][1],
        "origem": "reexecucao de scripts/linhagem_mae_e_custo_2026-09-15.py (sha 1bb14359...) pela contra-auditoria ca-admissibilidade-params",
        "categoria": "consolidacao", "tipo": "resultado", "status": "provisorio",
        "citabilidade": "valores usados nas tabelas conferidos contra ca-admissibilidade-params.json e params_checkpoints_2026-09-15.json",
        "nota": ("hash atual apos regravacao; substitui a entrada anterior (sha " + regs_lin[-1]["sha256"][:12] + "). A contra-auditoria "
                 "reexecutou o script as 02:11:12 UTC (registro anterior 02:08:27) e regravou o arquivo com o mesmo script; os valores "
                 "que ela cita (MAE, parametros, oito tempos da opcao B) e os usados por gerar_tabelas_rodada_escrita_2026-09-16.py "
                 "conferem. Os bytes da versao anterior nao foram preservados, entao a igualdade campo a campo com ela nao e verificavel."),
        "autor": "agente forum-eng-dados", "timestamp_utc": agora})
regs_ab = [m for m in manifest if m["artefato"].replace("\\", "/").endswith(FONTES["adm_bloq"][0])]
exige(regs_ab, "admissibilidade_bloqueada sem entrada no manifest")
if regs_ab[-1]["sha256"] != FONTES["adm_bloq"][1]:
    reg_novos.append({
        "artefato": FONTES["adm_bloq"][0], "sha256": FONTES["adm_bloq"][1],
        "origem": "reexecucao de scripts/medir_admissibilidade_e_params_2026-09-15.py --item 1 (sha e11c31aa...) pela contra-auditoria ca-admissibilidade-params",
        "categoria": "consolidacao", "tipo": "resultado", "status": "provisorio",
        "citabilidade": "valores usados em tab_admissibilidade_v3 conferidos contra ca-admissibilidade-params.json (item 4)",
        "nota": ("hash atual apos regravacao; substitui a entrada anterior (sha " + regs_ab[-1]["sha256"][:12] + "). A contra-auditoria "
                 "reexecutou o item 1 (timestamp interno 02:12:07 UTC; registro anterior 02:00:48) e regravou o arquivo com o mesmo script; "
                 "media 0.999961510, 69/80 corridas em 1.0, 2536 de 65930590 nos conferem. Bytes da versao anterior nao preservados."),
        "autor": "agente forum-eng-dados", "timestamp_utc": agora})
if reg_novos:
    bak =MANIFEST.with_name(f"manifest.jsonl.pre_tabelas_escrita_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(MANIFEST, bak)
    with open(MANIFEST, "a", encoding="utf-8") as fh:
        for r in reg_novos:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"ok: {len(SAIDAS)} tabelas, {len(rastro['guardas'])} guardas, {len(reg_novos)} entradas novas no manifest")
