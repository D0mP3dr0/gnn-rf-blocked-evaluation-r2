
"""Paired test of GNN-RF against calibrated analytical models across the 16 blocked cells."""
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access"
            r"\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO")
PREREG = BASE / "dados" / "planos" / "preregistro_teste_pareado_empiricos_2026-09-15.json"
PREREG_SHA = "ca54a9dc8b8d9f74d4bce1499ea05f28e6d0e9897f157dd68845e5a2f6046b3a"
OUT = BASE / "dados" / "consolidacao" / "teste_pareado_empiricos_c1v2.json"
TEX = BASE.parents[1] / "REVISAO_R3" / "novas" / "tab_blocked_16_tres_pop.tex"

CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QUADS = [1, 2, 3, 4]
MODELOS = ["fspl", "hata_rural", "cost231_sub"]
VARIANTES = ["v1_original", "v2_clamp", "v3_clamp_freq_enlace"]
TOL_CUSTODIA_DB = 2.5e-4
CHAVE_CONST = "preditor_constante_piso"
BOOT_N = 10000
BOOT_SEED = 42


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def morre(msg: str):
    print("ABORTA:", msg, file=sys.stderr)
    raise SystemExit(2)


if sha256(PREREG) != PREREG_SHA:
    morre(f"pre-registro alterado: sha256 {sha256(PREREG)} != {PREREG_SHA} carimbado")
prereg = json.loads(PREREG.read_text(encoding="utf-8"))
congeladas = prereg["fontes_congeladas"]

fontes_conferidas = {}
for nome, meta in congeladas.items():
    p = Path(meta["caminho"])
    if not p.exists():
        morre(f"fonte ausente: {nome} -> {p}")
    h = sha256(p)
    if h != meta["sha256"]:
        morre(f"fonte mudou desde o pre-registro: {nome}\n  congelado {meta['sha256']}\n  agora     {h}")
    fontes_conferidas[nome] = {"caminho": str(p), "sha256": h}

T15 = json.loads(Path(congeladas["gnn_por_seed"]["caminho"]).read_text(encoding="utf-8"))
CA = json.loads(Path(congeladas["contra_auditoria_baselines"]["caminho"]).read_text(encoding="utf-8"))
BL = {}
for c in CIDADES:
    for q in QUADS:
        BL[f"{c}_Q{q}"] = json.loads(
            Path(congeladas[f"baselines_v3_{c}_Q{q}"]["caminho"]).read_text(encoding="utf-8"))

if CA["veredito"] != "CONFIRMA":
    morre(f"contra-auditoria dos empiricos nao confirma: veredito={CA['veredito']}")


def gnn_por_celula():
    fora, out = [], {}
    for c in CIDADES:
        for q in QUADS:
            regs = [r for r in T15["registros"]
                    if "erro" not in r and r["cidade"] == c and r["quadrante"] == q]
            if not regs:
                morre(f"sem registros T15 para {c}_Q{q}")
            ok, excl = [], []
            for r in regs:
                custodia = bool(r["idx_sha256_global_test_ok"]) and \
                    float(r["diff_rssi_total_vs_run_db"]) <= TOL_CUSTODIA_DB
                (ok if custodia else excl).append(r)
            if not ok:
                morre(f"celula {c}_Q{q} sem nenhuma semente com custodia ok")
            if excl:
                fora.append({"celula": f"{c}_Q{q}", "seeds_excluidas": [r["seed"] for r in excl]})
            tot = np.array([r["mae_rssi_total_db"] for r in ok], dtype=float)
            cov = np.array([r["mae_rssi_valido_db"] for r in ok], dtype=float)
            out[f"{c}_Q{q}"] = {
                "seeds_usadas": [r["seed"] for r in ok],
                "n_seeds": len(ok),
                "n_test": ok[0]["n_test"],
                "frac_valido": ok[0]["frac_valido"],
                "mae_rssi_todos_media_db": float(tot.mean()),
                "mae_rssi_todos_dp_db": float(tot.std(ddof=1)),
                "mae_rssi_todos_pior_semente_db": float(tot.max()),
                "mae_rssi_cobertos_media_db": float(cov.mean()),
            }
    return out, fora


GNN, SEEDS_FORA = gnn_por_celula()


def competidores(celula: str):
    """Devolve (analiticos_todos, analiticos_cobertos, const_todos, const_cobertos)."""
    t = BL[celula]["particoes"]["test"]
    va = t["variantes"]
    ana_t, ana_c = {}, {}
    for v in VARIANTES:
        if v not in va:
            morre(f"{celula}: variante {v} ausente (presentes: {sorted(va)})")
        for m in MODELOS:
            ana_t[f"{v}.{m}"] = float(va[v][m]["todos"]["mae_db"])
            ana_c[f"{v}.{m}"] = float(va[v][m]["cobertura"]["mae_db"])
    if len(ana_t) != 9:
        morre(f"{celula}: esperados 9 competidores analiticos, obtidos {len(ana_t)}")
    const = t[CHAVE_CONST]
    if float(const["valor_dbm"]) != -110.0:
        morre(f"{celula}: preditor constante nao esta no piso de -110 dBm ({const['valor_dbm']})")
    return ana_t, ana_c, float(const["todos"]["mae_db"]), float(const["cobertura"]["mae_db"])


PARES = {}
for c in CIDADES:
    for q in QUADS:
        k = f"{c}_Q{q}"
        ana_t, ana_c, const_t, const_c = competidores(k)
        n_ana = min(ana_t, key=ana_t.get)
        tudo_t = dict(ana_t, **{"preditor_constante_-110dBm": const_t})
        n_pub = min(tudo_t, key=tudo_t.get)                    # published column (includes the constant)
        n_cov = min(ana_c, key=ana_c.get)
        g = GNN[k]
        PARES[k] = {
            "n_test": g["n_test"],
            "frac_valido": g["frac_valido"],
            "seeds_usadas": g["seeds_usadas"],
            "gnn_mae_rssi_todos_media_db": g["mae_rssi_todos_media_db"],
            "gnn_mae_rssi_todos_dp_db": g["mae_rssi_todos_dp_db"],
            "gnn_mae_rssi_todos_pior_semente_db": g["mae_rssi_todos_pior_semente_db"],
            "gnn_mae_rssi_cobertos_media_db": g["mae_rssi_cobertos_media_db"],
            "competidores_todos_db": tudo_t,
            "competidores_cobertos_db": ana_c,
            "primaria": {"competidor": n_ana, "mae_db": ana_t[n_ana],
                         "d_db": ana_t[n_ana] - g["mae_rssi_todos_media_db"]},
            "secundaria_A_publicada": {"competidor": n_pub, "mae_db": tudo_t[n_pub],
                                       "d_db": tudo_t[n_pub] - g["mae_rssi_todos_media_db"]},
            "secundaria_C_cobertos": {"competidor": n_cov, "mae_db": ana_c[n_cov],
                                      "d_db": ana_c[n_cov] - g["mae_rssi_cobertos_media_db"]},
            "constante_todos_db": const_t,
        }
        for m in MODELOS:
            melhor_v = min(VARIANTES, key=lambda v: ana_t[f"{v}.{m}"])
            PARES[k][f"familia_{m}"] = {"variante": melhor_v, "mae_db": ana_t[f"{melhor_v}.{m}"],
                                        "d_db": ana_t[f"{melhor_v}.{m}"] - g["mae_rssi_todos_media_db"]}


def confere_tabela():
    txt = TEX.read_text(encoding="utf-8")
    linhas = [l for l in txt.splitlines() if re.match(r"^(Bauru|Campinas|Lins|Sorocaba) & Q\d", l)]
    if len(linhas) != 16:
        morre(f"tabela publicada com {len(linhas)} linhas de celula, esperadas 16")
    checagens = []
    for l in linhas:
        col = [x.strip() for x in l.rstrip("\\ ").split("&")]
        k = f"{col[0].lower()}_Q{col[1][1:]}"
        pub_frac, pub_gnn = float(col[2]), float(col[3].split("$\\pm$")[0])
        pub_cov_gnn = float(col[4].split("$\\pm$")[0])
        pub_best_all, pub_best_cov, pub_const = float(col[7]), float(col[8]), float(col[9])
        p = PARES[k]
        itens = [
            ("frac_valido", pub_frac, p["frac_valido"], 5.1e-4),
            ("gnn_rssi_todos", pub_gnn, p["gnn_mae_rssi_todos_media_db"], 5.1e-4),
            ("gnn_rssi_cobertos", pub_cov_gnn, p["gnn_mae_rssi_cobertos_media_db"], 5.1e-3),
            ("best_empirico_todos", pub_best_all, p["secundaria_A_publicada"]["mae_db"], 5.1e-4),
            ("best_empirico_cobertos", pub_best_cov, p["secundaria_C_cobertos"]["mae_db"], 5.1e-3),
            ("constante_todos", pub_const, p["constante_todos_db"], 5.1e-4),
        ]
        for nome, pub, meu, tol in itens:
            dif = abs(pub - meu)
            checagens.append({"celula": k, "campo": nome, "publicado": pub, "recalculado": meu,
                              "diff": dif, "bate": dif <= tol})
    ruins = [c for c in checagens if not c["bate"]]
    if ruins:
        morre("o pareamento NAO reproduz a tabela publicada: " + json.dumps(ruins[:5], ensure_ascii=False))
    pior = max(checagens, key=lambda c: c["diff"])
    return {"n_checagens": len(checagens), "todas_batem": True,
            "diff_maxima": pior["diff"],
            "checagem_da_diff_maxima": pior,
            "nota_sobre_a_tolerancia": ("campos impressos com 3 casas usam tolerancia 5,1e-4 e os de 2 casas "
                                        "5,1e-3: a diferenca residual e arredondamento de impressao, nao divergencia de valor"),
            "checagens": checagens,
            "regra": "cada numero impresso na tab:blocked_three_pop e reconferido contra o artefato; divergencia ABORTA"}


CONF_TABELA = confere_tabela()


def bateria(d, rotulo, chaves):
    d = np.asarray(d, dtype=float)
    n = len(d)
    media, dp = float(d.mean()), float(d.std(ddof=1))
    ep = dp / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    tt = stats.ttest_1samp(d, 0.0)
    zeros = int((d == 0).sum())
    w = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="exact" if zeros == 0 else "auto")
    pos = int((d > 0).sum())
    sinal = stats.binomtest(pos, n, 0.5, alternative="two-sided")
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(BOOT_N)])
    r = stats.rankdata(np.abs(d))
    w_mais = float(r[d > 0].sum())
    w_menos = float(r[d < 0].sum())
    return {
        "rotulo": rotulo,
        "n": n,
        "celulas": chaves,
        "d_db": [float(x) for x in d],
        "media_d_db": media,
        "dp_d_db": dp,
        "erro_padrao_db": float(ep),
        "ic95_t_db": [float(media - tcrit * ep), float(media + tcrit * ep)],
        "mediana_d_db": float(np.median(d)),
        "t_pareado": {"estatistica": float(tt.statistic), "gl": n - 1, "p": float(tt.pvalue),
                      "nota": "ttest_1samp sobre as diferencas = ttest_rel sobre os dois vetores pareados"},
        "wilcoxon": {"estatistica": float(w.statistic), "p": float(w.pvalue),
                     "zeros": zeros, "w_mais": w_mais, "w_menos": w_menos,
                     "r_biserial_pareado": float((w_mais - w_menos) / (w_mais + w_menos))},
        "teste_do_sinal": {"celulas_com_d_positivo": pos, "n": n, "p": float(sinal.pvalue)},
        "bootstrap_percentil": {"reamostragens": BOOT_N, "semente": BOOT_SEED,
                                "ic95_db": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]},
        "cohen_dz": float(media / dp),
        "celulas_em_que_a_gnn_perde": [k for k, x in zip(chaves, d) if x <= 0],
    }


CH =[f"{c}_Q{q}" for c in CIDADES for q in QUADS]

primaria = bateria([PARES[k]["primaria"]["d_db"] for k in CH],
                   "primaria: melhor analitico calibrado por celula (9 candidatos, sem o preditor constante), populacao todos os nos", CH)
sec_A = bateria([PARES[k]["secundaria_A_publicada"]["d_db"] for k in CH],
                "secundaria A: coluna publicada 'Best empirical / RSSI, all' (9 analiticos + preditor constante)", CH)
sec_C = bateria([PARES[k]["secundaria_C_cobertos"]["d_db"] for k in CH],
                "secundaria C: populacao de nos cobertos (rf_targets[:,0] < 299 dB), melhor analitico na mesma populacao", CH)
rob_pior = bateria([PARES[k]["primaria"]["mae_db"] - PARES[k]["gnn_mae_rssi_todos_pior_semente_db"] for k in CH],
                   "robustez: primaria com a PIOR semente da GNN por celula, nao a media", CH)


log_raz = bateria([float(np.log10(PARES[k]["primaria"]["mae_db"] / PARES[k]["gnn_mae_rssi_todos_media_db"])) for k in CH],
                  "complemento: log10(MAE_baseline / MAE_GNN) da primaria (adimensional, nao dB)", CH)

# unit of analysis = city (n=4)
d_cidade = [float(np.mean([PARES[f"{c}_Q{q}"]["primaria"]["d_db"] for q in QUADS])) for c in CIDADES]
por_cidade = bateria(d_cidade, "secundaria: unidade = cidade (media dos 4 quadrantes), n=4", CIDADES)


fam = {}
for m in MODELOS:
    fam[m] = bateria([PARES[k][f"familia_{m}"]["d_db"] for k in CH],
                     f"secundaria B: familia {m} (melhor variante por celula), populacao todos os nos", CH)
ordem = sorted(MODELOS, key=lambda m: fam[m]["t_pareado"]["p"])
for i, m in enumerate(ordem):
    fam[m]["holm"] = {"posicao": i + 1, "p_bruto": fam[m]["t_pareado"]["p"],
                      "p_ajustado": min(1.0, fam[m]["t_pareado"]["p"] * (len(MODELOS) - i)),
                      "metodo": "Holm-Bonferroni sobre os p do t pareado das 3 familias"}

# reading criterion
crit = prereg["criterio_de_leitura_fixado_antes_do_resultado"]
c_media_pos = primaria["media_d_db"] > 0
c_tres_p = all(primaria[k2]["p"] < 0.05 for k2 in ("t_pareado", "wilcoxon", "teste_do_sinal"))
c_ic = primaria["ic95_t_db"][0] > 0 and primaria["bootstrap_percentil"]["ic95_db"][0] > 0
n_perde = len(primaria["celulas_em_que_a_gnn_perde"])
if not c_media_pos or not c_tres_p:
    veredito = "NAO_SUPORTA"
elif c_ic and n_perde <= 1:
    veredito = "SUPORTA"
else:
    veredito = "SUPORTA_COM_RESSALVA"


cruzada = []
for k in CH:
    r = CA["recomputo_por_celula"][k]
    comp = r["competidores_recomputados"]
    ana = {a: b for a, b in comp.items() if "constante" not in a}
    melhor_ca = min(ana.values())
    cruzada.append({
        "celula": k,
        "diff_melhor_analitico_db": abs(melhor_ca - PARES[k]["primaria"]["mae_db"]),
        "diff_gnn_media_db": abs(float(r["gnn_mae_rssi_media_seeds_recomputado"]) - PARES[k]["gnn_mae_rssi_todos_media_db"]),
    })

doc = {
    "artefato_tipo": "teste_estatistico_pareado",
    "id": "T16_teste_pareado_gnn_x_empiricos",
    "status": "provisorio (pendente de contra-auditoria por quem nao escreveu o script)",
    "regra_de_citabilidade": (
        "Nenhum numero deste JSON e citavel no manuscrito antes de contra-auditoria: reexecucao "
        "independente e leitura por dentro do script por quem nao o escreveu. Ate la, provisorio."
    ),
    "pre_registro": {"caminho": str(PREREG), "sha256": PREREG_SHA,
                     "criterio_de_leitura": crit},
    "script": {"caminho": str(Path(__file__).resolve()),
               "sha256": sha256(Path(__file__).resolve()),
               "nota": "sha256 do proprio script no momento da execucao"},
    "ambiente": {"python": sys.version.split()[0], "numpy": np.__version__,
                 "scipy": __import__("scipy").__version__, "executavel": sys.executable,
                 "dispositivo": "CPU (nenhum modelo ou dataset .pt carregado; so JSON)"},
    "linhagem": {
        "campanha": "C1 v2, particao espacialmente bloqueada (blocos 10 km, buffer 2 km), 16 celulas cidade x quadrante, 5 sementes por celula",
        "gnn": "t15_tres_populacoes.json (recomputo com custodia de idx_sha256 de teste)",
        "empiricos": "t5_baselines_v3 (NAO dados/baselines_v2), contra-auditado por ca-t5-baselines-v3.json, veredito CONFIRMA",
        "aviso": "numeros desta saida NAO sao comparaveis com statistical_report_gold.json (linhagem v20 gold, n=4 quadrantes, outra populacao e sem clamp)",
    },
    "definicoes": {
        "unidade": "celula cidade x quadrante (n=16, caixas delimitadoras disjuntas)",
        "d_db": "MAE do baseline menos MAE da GNN, na mesma celula e nos mesmos nos de teste; d>0 = GNN melhor",
        "populacao_primaria": "todos os nos de teste, sentinela -110 dBm incluida",
        "gnn_por_celula": "media das sementes com custodia ok (idx_sha256_global_test_ok e diff_rssi_total_vs_run_db <= 2.5e-4 dB)",
        "melhor_analitico": "min sobre 3 modelos x 3 variantes calibradas; escolha conservadora que favorece o baseline",
    },
    "custodia": {"tolerancia_db": TOL_CUSTODIA_DB,
                 "celulas_com_semente_excluida": SEEDS_FORA,
                 "n_sementes_por_celula": {k: GNN[k]["n_seeds"] for k in CH}},
    "conferencia_contra_tabela_publicada": CONF_TABELA,
    "checagem_cruzada_contra_auditoria": {
        "fonte": "ca-t5-baselines-v3.json (recomputo independente dos empiricos e da media da GNN)",
        "diff_maxima_melhor_analitico_db": max(x["diff_melhor_analitico_db"] for x in cruzada),
        "diff_maxima_gnn_media_db": max(x["diff_gnn_media_db"] for x in cruzada),
        "por_celula": cruzada,
    },
    "pares_por_celula": PARES,
    "resultado_primario": primaria,
    "veredito_pelo_criterio_pre_registrado": {
        "veredito": veredito,
        "media_d_positiva": bool(c_media_pos),
        "tres_testes_p_menor_0_05": bool(c_tres_p),
        "ambos_ic95_acima_de_zero": bool(c_ic),
        "n_celulas_em_que_a_gnn_perde": n_perde,
    },
    "secundarias": {"A_coluna_publicada": sec_A, "B_por_familia": fam,
                    "C_nos_cobertos": sec_C, "unidade_cidade_n4": por_cidade},
    "robustez": {"pior_semente": rob_pior, "log_razao": log_raz},
    "reducao_relativa_primaria": {
        "por_celula": {k: 1.0 - PARES[k]["gnn_mae_rssi_todos_media_db"] / PARES[k]["primaria"]["mae_db"] for k in CH},
        "mediana": float(np.median([1.0 - PARES[k]["gnn_mae_rssi_todos_media_db"] / PARES[k]["primaria"]["mae_db"] for k in CH])),
        "min": float(np.min([1.0 - PARES[k]["gnn_mae_rssi_todos_media_db"] / PARES[k]["primaria"]["mae_db"] for k in CH])),
        "max": float(np.max([1.0 - PARES[k]["gnn_mae_rssi_todos_media_db"] / PARES[k]["primaria"]["mae_db"] for k in CH])),
    },
    "limitacoes_declaradas": [
        "as 16 celulas nao sao estritamente independentes: quadrantes da mesma cidade sao tiles adjacentes (bordas coladas) e do mesmo contexto geografico; por isso a secundaria com n=4 cidades",
        "o lado da GNN e media de 5 sementes; a incerteza entre sementes nao entra no teste pareado (entra na robustez com a pior semente)",
        "o melhor empirico e escolhido POR CELULA, o que e um envelope otimista para o baseline e portanto conservador contra o claim",
        "pre-registro nao cego: as margens por celula ja estavam visiveis em ca-t5-baselines-v3.json antes de este desenho ser escrito (declarado no pre-registro)",
    ],
    "fontes": fontes_conferidas,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=1, ensure_ascii=False)

print("saida:", OUT)
print("veredito:", veredito)
print("primaria: media d = %.4f dB  IC95 t [%.4f, %.4f]  boot [%.4f, %.4f]" % (
    primaria["media_d_db"], *primaria["ic95_t_db"], *primaria["bootstrap_percentil"]["ic95_db"]))
print("p: t=%.3e  wilcoxon=%.3e  sinal=%.3e  dz=%.3f  perde em %s" % (
    primaria["t_pareado"]["p"], primaria["wilcoxon"]["p"], primaria["teste_do_sinal"]["p"],
    primaria["cohen_dz"], primaria["celulas_em_que_a_gnn_perde"]))
print("conferencia com a tabela publicada: %d numeros, diff max %.2e" % (
    CONF_TABELA["n_checagens"], CONF_TABELA["diff_maxima"]))
