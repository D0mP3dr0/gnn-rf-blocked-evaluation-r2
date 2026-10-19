"""D_tost_aninhado.py — fase D minima da R4 IEEE Access 47637 (pedido R2-8c do revisor 2; folha F-R2.8.08 / F-R2.8.09).

Aprovacao: ROADMAP_R4_2026-10-14.md secao 4 "Fase D" + "aceito" do dono em 19/10/2026 (TOST com cidades aninhadas).
Le SO os run JSON da campanha C1 v2 (cadeia c0c1cf_* e treino do zero c0c1cfsc_*, seeds 42/43, Q2-Q4, 4 cidades) e o
agregado publicado na R3 (agregado_controles_c1v2.json); grava SO D_tost_aninhado.json ao lado deste script.
CPU, nenhum treino, nenhuma inferencia.

Passo 1 (reproducao, antes de qualquer analise nova): refaz, com as MESMAS formulas de
scripts/agregar_controles_c1v2.py (statistics.mean/stdev + scipy.stats.t), o TOST publicado na Tab. 9 da R3
(unidade = cluster (cidade, seed), n = 8, 24 pares) e grava o diff contra o agregado e contra o valor publicado.

Passo 2 (analise aninhada, sinal Delta = cadeia - zero, MAE RSSI de teste, populacao ALL):
  (a) unidade = cidade (n = 4): media, IC95 t (gl = 3), t pareado bilateral, TOST a +-0,1 dB (p = max dos dois unilaterais);
      t/IC reaproveitados de BA_inferencia.t_cidades e conferidos por formula propria.
  (b) bootstrap hierarquico B = 20000, numpy default_rng(20261018), IC95 percentil:
      b1  funcao REAPROVEITADA BA_inferencia.bootstrap (cidade -> cadeia (cidade, seed); os Q dentro da cadeia, medias),
          indices IC_ (B,4) e IS_ (B,4,2) gerados primeiro, na ordem do BA;
      b2  pedido: cidade -> (cidade, seed) -> quadrante (os 3 Q reamostrados com reposicao dentro da cadeia), indices IQ_
          gerados DEPOIS de IC_ e IS_, do mesmo rng; estatistica = media de todas as Delta sorteadas (pesos iguais);
      b3  redacao da folha F-R2.8.09: cidade -> celula (cidade, Q) -> seed, indices proprios do mesmo rng depois de b2.
  (c) MixedLM REML: Delta ~ 1 + (1|cidade) + (1|cidade:seed) (statsmodels; groups = cidade, vc = cadeia); convergencia,
      avisos e componente que zera declarados; statsmodels nao oferece Satterthwaite: IC Wald (z) declarado como tal, e um IC
      t com gl = 3 (n de cidades - 1) como sensibilidade rotulada. Secundario: BA_inferencia.misto (cidade + celula + cadeia).
  (d) VALID: so se o insumo existir (run JSON dos c0c1cfsc nao traz MAE RSSI VALID; ver campo "valid").
  (e) MDE80 na unidade cidade (n = 4) pela variancia observada das 4 medias de cidade: MDE do t pareado (convencao do
      projeto, _pareceres_2026-10-19/critico-verificador_mde_margens.md), poder exato do TOST a +-0,1 dB sob Delta verdadeiro
      = 0, menor margem com poder >= 80 % e menor n de cidades com poder >= 80 % na margem 0,1 dB.
Destinos pre-fixados (regra do protocolo da fase C, texto da tarefa): IC inteiro dentro de +-0,1 -> equivalencia na unidade
cidade; IC cruza +-0,1 -> "no systematic difference detectable at the city unit"; IC inteiro fora -> diferenca.
Uso: /trabalho/ambientes/s33_amb_virtual/.venv/bin/python -I D_tost_aninhado.py [--saida D_tost_aninhado.json]
Numeros PROVISORIOS ate contra-auditoria."""
import sys
sys.dont_write_bytecode = True  # nada de __pycache__ nas pastas somente-leitura de onde se importa
import argparse, importlib.util, json, math, warnings
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy import stats as st
from scipy import integrate, optimize

AQUI = Path(__file__).resolve().parent
REV = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14")
E = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO")
TREINOS = E / "treinos"
AGREG = E / "dados/treinos_c1/agregado_controles_c1v2.json"
AGREG_SCRIPT = E / "scripts/agregar_controles_c1v2.py"
BA_PY = REV / "fase_BA1/avaliacao/metricas/BA_inferencia.py"
COMMON_PY = REV / "fase_B/common.py"
FOLHA = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/R4_RESSUBMISSAO_2026-10/FOLHA_DE_FATOS_R4.md")
TAB_R3 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/REVISAO_R3/novas/tab_controles_24.tex")

# importa BA_inferencia (que por sua vez importa fase_B/common.py) sem escrever bytecode
_spec = importlib.util.spec_from_file_location("BA_inferencia", BA_PY)
BA = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(BA)
CIDADES, carimbo, dump = BA.CIDADES, BA.carimbo, BA.dump      # ["bauru", "campinas", "lins", "sorocaba"]

SEEDS_D = [42, 43]
QS_D = ["Q2", "Q3", "Q4"]
MARGEM = 0.1
ALFA = 0.05
B_BOOT = BA.B_BOOT          # 20000
SEED_BOOT = BA.SEED_BOOT    # 20261018
PODER_ALVO = 0.80
PUBLICADO_R3 = {"media_db": -0.014, "ic95_db": [-0.072, 0.045], "p_bicaudal": 0.60, "p_tost": 0.0051, "n": 8, "n_pares": 24}


def rotulo(c, s, q, braco):
    return f"{braco}_{c}_s{s}_{q}_g10b2"


def run_path(lbl):
    return TREINOS / lbl / f"run_{lbl}.json"


def le(lbl):
    d = json.load(open(run_path(lbl), encoding="utf-8"))
    if d["custo"].get("tempo_total_s") is None:
        sys.exit(f"ERRO: {lbl} incompleto")
    return d


def confere_par(da, db, r):  # mesmo pareamento endurecido do agregador original
    if da["dataset"]["rf_data_sha256"] != db["dataset"]["rf_data_sha256"]:
        sys.exit(f"ERRO: dataset diverge em {r}")
    for part in ("train", "val", "test"):
        if da["particoes"][part]["idx_sha256_global"] != db["particoes"][part]["idx_sha256_global"]:
            sys.exit(f"ERRO: split ({part}) diverge em {r}")


def destino(lo, hi, m=MARGEM):
    if lo >= -m and hi <= m:
        return "equivalencia na unidade cidade (IC inteiro dentro de +-0,1 dB)"
    if hi < -m or lo > m:
        return "diferenca (IC inteiro fora de +-0,1 dB)"
    return "no systematic difference detectable at the city unit (IC cruza +-0,1 dB)"


def bloco_ic(lo, hi):
    return {"ic95": [float(lo), float(hi)], "destino": destino(lo, hi), "ic_exclui_zero": bool(lo > 0 or hi < 0)}


# ---------------- passo 1: formulas identicas a agregar_controles_c1v2.py ----------------
def estat_um_grupo_orig(vals):
    n = len(vals); m = mean(vals); sd = stdev(vals); se = sd / math.sqrt(n); gl = n - 1
    tc = float(st.t.ppf(1 - ALFA / 2, gl))
    return {"n": n, "media_db": m, "dp_db": sd, "ep_db": se, "ic95_db": [m - tc * se, m + tc * se],
            "t": m / se, "p_bicaudal": 2 * float(st.t.sf(abs(m / se), gl))}


def tost_orig(vals, margem):
    n = len(vals); m, sd = mean(vals), stdev(vals); se = sd / math.sqrt(n); gl = n - 1
    t_low = (m + margem) / se; p_low = float(st.t.sf(t_low, gl))
    t_up = (m - margem) / se; p_up = 1.0 - float(st.t.sf(t_up, gl))
    return {"margem_db": margem, "n": n, "t_inferior": t_low, "p_inferior": p_low, "t_superior": t_up,
            "p_superior": p_up, "p_tost": max(p_low, p_up)}


# ---------------- (a) unidade cidade ----------------
def tost_unidade(vals, margem=MARGEM):
    v = np.asarray(vals, float); n = v.size; m = float(v.mean()); sd = float(v.std(ddof=1)); se = sd / math.sqrt(n); gl = n - 1
    t_low = (m + margem) / se; p_low = float(st.t.sf(t_low, gl))       # H0a: mu <= -margem
    t_up = (m - margem) / se; p_up = float(st.t.cdf(t_up, gl))         # H0b: mu >= +margem
    tc90 = float(st.t.ppf(1 - ALFA, gl))
    return {"margem_db": margem, "n": n, "gl": gl, "media_db": m, "dp_db": sd, "ep_db": se,
            "t_inferior": t_low, "p_inferior": p_low, "t_superior": t_up, "p_superior": p_up,
            "p_tost": max(p_low, p_up), "rejeita_ao_nivel_0_05": bool(max(p_low, p_up) < ALFA),
            "ic90_t_equivalente_ao_tost": [m - tc90 * se, m + tc90 * se]}


# ---------------- (b) bootstraps ----------------
def boot_tres_niveis(A, IC_, IS_, IQ_):
    """A: (4 cidades, 2 seeds, 3 Q). cidade -> cadeia -> Q, cada nivel com reposicao; estatistica = media das Delta sorteadas."""
    B = IC_.shape[0]
    cid = IC_[:, :, None, None]; sd_ = IS_[:, :, :, None]
    vals = A[cid, sd_, IQ_]                                   # (B, 4, 2, 3)
    stt = vals.reshape(B, -1).mean(axis=1)
    return {"estimativa": float(A.mean()), "ic95": [float(np.percentile(stt, 2.5)), float(np.percentile(stt, 97.5))],
            "B_validas": int(np.isfinite(stt).sum()), "ep_bootstrap": float(stt.std(ddof=1))}


def boot_cidade_celula_seed(A, IC_, ICel_, ISd_):
    """cidade -> celula (Q dentro da cidade) -> seed dentro da celula."""
    B = IC_.shape[0]
    cid = IC_[:, :, None, None]; q = ICel_[:, :, :, None]   # ISd_: (B, 4, 3, 2) = 2 seeds sorteadas por celula sorteada
    vals = A[cid, ISd_, q]                                    # (B, 4, 3, 2)
    stt = vals.reshape(B, -1).mean(axis=1)
    return {"estimativa": float(A.mean()), "ic95": [float(np.percentile(stt, 2.5)), float(np.percentile(stt, 97.5))],
            "B_validas": int(np.isfinite(stt).sum()), "ep_bootstrap": float(stt.std(ddof=1))}


# ---------------- (c) modelo misto ----------------
def misto_cidade_cadeia(D):
    import pandas as pd
    import statsmodels.formula.api as smf
    df = pd.DataFrame([{"delta": v, "cidade": c, "cadeia": f"{c}_s{s}", "Q": q} for (c, s, q), v in sorted(D.items())])
    out = {"formula": "delta ~ 1 + (1|cidade) + (1|cidade:seed)",
           "implementacao": "statsmodels MixedLM(groups=cidade, re_formula='1', vc_formula={'cadeia': '0 + C(cadeia)'}), REML",
           "n_obs": int(len(df)), "n_cidades": int(df.cidade.nunique()), "n_cadeias": int(df.cadeia.nunique()),
           "satterthwaite": "indisponivel no statsmodels; IC Wald (z) e IC t com gl = 3 so como sensibilidade", "ajustes": {}}
    for metodo in ["default", "lbfgs", "powell", "nm"]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                md = smf.mixedlm("delta ~ 1", df, groups="cidade", re_formula="1", vc_formula={"cadeia": "0 + C(cadeia)"})
                f = md.fit(reml=True) if metodo == "default" else md.fit(reml=True, method=[metodo])
                b0 = float(f.params["Intercept"]); ep = float(f.bse["Intercept"])
                ci = f.conf_int().loc["Intercept"].tolist()
                tq3 = float(st.t.ppf(0.975, 3))
                vc = float(f.cov_re.iloc[0, 0]); vk = float(f.vcomp[0]); vr = float(f.scale)
                tot = vc + vk + vr
                zeram = [k for k, v in {"cidade": vc, "cidade:seed": vk}.items() if v <= 1e-6 * tot]
                o = {"ajustado": True, "convergiu": bool(f.converged), "intercepto_db": b0, "ep_db": ep,
                     "ic95_wald_z": [float(ci[0]), float(ci[1])], "destino_wald": destino(ci[0], ci[1]),
                     "ic95_t_gl3_sensibilidade": [b0 - tq3 * ep, b0 + tq3 * ep], "destino_t_gl3": destino(b0 - tq3 * ep, b0 + tq3 * ep),
                     "p_wald_z": float(f.pvalues["Intercept"]),
                     "var_cidade": vc, "var_cidade_seed": vk, "var_residual_Q": vr,
                     "componentes_que_zeram": zeram, "criterio_zera": "variancia <= 1e-6 x variancia total",
                     "llf_reml": float(f.llf)}
            except Exception as e:
                o = {"ajustado": False, "motivo": f"{type(e).__name__}: {e}"}
        o["avisos"] = sorted({f"{x.category.__name__}: {str(x.message)[:200]}" for x in w})
        out["ajustes"][metodo] = o
    out["primario"] = "default"
    return out


# ---------------- (e) poder / MDE ----------------
def poder_tost(sigma, n, margem, alfa=ALFA):
    """Poder exato do TOST (t de uma amostra) sob Delta verdadeiro = 0: P(|m| < margem - t_{1-a,n-1} s / sqrt(n)),
    m ~ N(0, sigma^2/n) independente de s, (n-1)s^2/sigma^2 ~ chi2_{n-1}."""
    gl = n - 1; tc = float(st.t.ppf(1 - alfa, gl)); se0 = sigma / math.sqrt(n)

    def integrando(u):  # u = (n-1) s^2 / sigma^2
        s = sigma * math.sqrt(u / gl); h = margem - tc * s / math.sqrt(n)
        return (2 * st.norm.cdf(h / se0) - 1) * st.chi2.pdf(u, gl) if h > 0 else 0.0
    umax = gl * ((margem * math.sqrt(n)) / (tc * sigma)) ** 2   # acima disso h <= 0
    val, _ = integrate.quad(integrando, 0, umax, limit=200)
    return float(val)


def poder_tost_sim(sigma, n, margem, nsim, rng, alfa=ALFA):
    gl = n - 1; tc = float(st.t.ppf(1 - alfa, gl))
    x = rng.normal(0.0, sigma, (nsim, n)); m = x.mean(1); se = x.std(1, ddof=1) / math.sqrt(n)
    return float(np.mean(np.abs(m) < margem - tc * se))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(AQUI / "D_tost_aninhado.json"))
    a = ap.parse_args()

    # ---- insumos: 48 run JSON (24 pares) ----
    insumos = {"agregado_controles_c1v2.json": AGREG, "agregar_controles_c1v2.py": AGREG_SCRIPT,
               "BA_inferencia.py": BA_PY, "fase_B_common.py": COMMON_PY, "FOLHA_DE_FATOS_R4.md": FOLHA, "tab_controles_24.tex_R3": TAB_R3}
    D, pares = {}, []
    for c in CIDADES:
        for s in SEEDS_D:
            for q in QS_D:
                lsc, lch = rotulo(c, s, q, "c0c1cfsc"), rotulo(c, s, q, "c0c1cf")
                sc, ch = le(lsc), le(lch)
                confere_par(sc, ch, f"{c}_s{s}_{q}")
                insumos[lsc] = run_path(lsc); insumos[lch] = run_path(lch)
                msc = sc["selecao"]["test_no_melhor_ckpt"]["mae_rssi_db"]; mch = ch["selecao"]["test_no_melhor_ckpt"]["mae_rssi_db"]
                D[(c, s, q)] = mch - msc
                pares.append({"cidade": c, "seed": s, "Q": q, "zero_db": msc, "cadeia_db": mch, "delta_db": mch - msc,
                              "n_nos_teste": sc["selecao"]["test_no_melhor_ckpt"]["n_nos"],
                              "chave_sc": lsc, "chave_ch": lch})
    assert len(D) == 24

    # ---- passo 1: reproducao do numero publicado ----
    ag = json.load(open(AGREG, encoding="utf-8"))["sem_transferencia"]
    diff_pares = max(abs(D[(v["cidade"], v["seed"], v["quadrante"])] - v["delta_encadeado_menos_semcadeia_db"])
                     for v in ag["pares_24_declarados"].values())
    assert len(ag["pares_24_declarados"]) == 24
    # mesma ordem de clusters do original (sorted por (cidade, seed)), mesma ordem de pares (SC: lins, bauru, campinas, sorocaba)
    ordem_orig = [(c, s, q) for c in ("lins", "bauru", "campinas", "sorocaba") for s in SEEDS_D for q in QS_D]
    brutos = {}
    for (c, s, q) in ordem_orig:
        brutos.setdefault((c, s), []).append(D[(c, s, q)])
    clusters = {f"{c}_s{s}": mean(ds) for (c, s), ds in sorted(brutos.items())}
    med_cl = list(clusters.values())
    e8 = estat_um_grupo_orig(med_cl); t8 = tost_orig(med_cl, MARGEM)
    ref_e = ag["estatistica_unidade_cluster"]; ref_t = ag["tost_equivalencia"]
    campos = {"media_db": (e8["media_db"], ref_e["media_db"]), "dp_db": (e8["dp_db"], ref_e["dp_db"]),
              "ic95_lo": (e8["ic95_db"][0], ref_e["ic95_db"][0]), "ic95_hi": (e8["ic95_db"][1], ref_e["ic95_db"][1]),
              "p_bicaudal": (e8["p_bicaudal"], ref_e["p_bicaudal"]), "p_inferior": (t8["p_inferior"], ref_t["p_inferior"]),
              "p_superior": (t8["p_superior"], ref_t["p_superior"]), "p_tost": (t8["p_tost"], ref_t["p_tost"])}
    diff_ag = {k: abs(x - y) for k, (x, y) in campos.items()}
    diff_cl = max(abs(clusters[k] - v["delta_medio_db"]) for k, v in ref_e["clusters"].items())
    arred = {"media_db": round(e8["media_db"], 3), "ic95_db": [round(e8["ic95_db"][0], 3), round(e8["ic95_db"][1], 3)],
             "p_bicaudal": round(e8["p_bicaudal"], 2), "p_tost": round(t8["p_tost"], 4), "n": e8["n"], "n_pares": 24}
    diff_pub = {"media_db": arred["media_db"] - PUBLICADO_R3["media_db"],
                "ic95_lo": arred["ic95_db"][0] - PUBLICADO_R3["ic95_db"][0], "ic95_hi": arred["ic95_db"][1] - PUBLICADO_R3["ic95_db"][1],
                "p_bicaudal": arred["p_bicaudal"] - PUBLICADO_R3["p_bicaudal"], "p_tost": arred["p_tost"] - PUBLICADO_R3["p_tost"],
                "n": arred["n"] - PUBLICADO_R3["n"], "n_pares": 0}
    # dados (pares, clusters, media, dp, IC) devem bater bit a bit; p-valores tolerancia 1e-15 (ULP do scipy.stats.t de outra
    # versao: o agregado foi gerado no Windows em 13/09); o valor exato de cada diff fica gravado
    reproduz = bool(diff_pares == 0 and diff_cl == 0
                    and all(diff_ag[k] == 0 for k in ("media_db", "dp_db", "ic95_lo", "ic95_hi"))
                    and all(v <= 1e-15 for v in diff_ag.values())
                    and all(abs(v) < 1e-12 for v in diff_pub.values()))
    if not reproduz:
        sys.exit(f"ERRO: reproducao falhou: pares {diff_pares} clusters {diff_cl} agregado {diff_ag} publicado {diff_pub}")

    # ---- (a) unidade cidade ----
    a_ba = BA.t_cidades(D)                                     # funcao reaproveitada (t, IC95 t, gl = 3)
    med_c = np.array([np.mean([D[(c, s, q)] for s in SEEDS_D for q in QS_D]) for c in CIDADES])
    a_tost = tost_unidade(med_c)
    tq3 = float(st.t.ppf(0.975, 3)); ic_a = [a_tost["media_db"] - tq3 * a_tost["ep_db"], a_tost["media_db"] + tq3 * a_tost["ep_db"]]
    conf_a = max(abs(ic_a[0] - a_ba["ic95_t"][0]), abs(ic_a[1] - a_ba["ic95_t"][1]), abs(a_tost["media_db"] - a_ba["media"]))
    assert conf_a < 1e-12, conf_a
    a_out = {"unidade": "cidade (media das 6 Delta: 2 seeds x Q2-Q4)", "n": 4, "gl": 3,
             "medias_cidade_db": {c: float(v) for c, v in zip(CIDADES, med_c)},
             "media_db": a_ba["media"], "dp_db": a_tost["dp_db"], "ep_db": a_tost["ep_db"],
             "ic95_t": a_ba["ic95_t"], "t": a_ba["t"], "p_bilateral": a_ba["p_bilateral"],
             "destino_ic95_t": destino(*a_ba["ic95_t"]), "ic95_exclui_zero": bool(a_ba["ic95_t"][0] > 0 or a_ba["ic95_t"][1] < 0),
             "tost": a_tost, "conferencia_formula_propria_vs_BA_t_cidades_max_abs": conf_a,
             "funcao_reaproveitada": "BA_inferencia.t_cidades"}

    # ---- (b) bootstraps ----
    rng = np.random.default_rng(SEED_BOOT)
    IC_ = rng.integers(0, len(CIDADES), (B_BOOT, len(CIDADES)))
    IS_ = rng.integers(0, len(SEEDS_D), (B_BOOT, len(CIDADES), len(SEEDS_D)))
    IQ_ = rng.integers(0, len(QS_D), (B_BOOT, len(CIDADES), len(SEEDS_D), len(QS_D)))
    IC3 = rng.integers(0, len(CIDADES), (B_BOOT, len(CIDADES)))
    ICel = rng.integers(0, len(QS_D), (B_BOOT, len(CIDADES), len(QS_D)))
    ISd = rng.integers(0, len(SEEDS_D), (B_BOOT, len(CIDADES), len(QS_D), len(SEEDS_D)))
    CM5 = BA.matriz_cadeias(D)                                 # funcao reaproveitada; colunas = SEEDS 42..46 do BA
    assert list(BA.SEEDS[:2]) == SEEDS_D and np.isnan(CM5[:, 2:]).all() and np.isfinite(CM5[:, :2]).all()
    CM = CM5[:, :2]
    b1 = BA.bootstrap(CM, IC_, IS_)
    A = np.array([[[D[(c, s, q)] for q in QS_D] for s in SEEDS_D] for c in CIDADES])
    b2 = boot_tres_niveis(A, IC_, IS_, IQ_)
    b3 = boot_cidade_celula_seed(A, IC3, ICel, ISd)
    b_out = {"B": B_BOOT, "rng": f"numpy default_rng({SEED_BOOT})",
             "ordem_indices": "IC_ (B,4), IS_ (B,4,2) [b1 e b2]; IQ_ (B,4,2,3) [b2]; IC3 (B,4), ICel (B,4,3), ISd (B,4,3,2) [b3]",
             "IC": "percentil 2,5-97,5",
             "nivel_superior_multiconjuntos_de_cidade": math.comb(4 + 4 - 1, 4),
             "nivel_cadeia_multiconjuntos_por_cidade": math.comb(2 + 2 - 1, 2),
             "b1_cidade_cadeia_funcao_BA": {**b1, **bloco_ic(*b1["ic95"]), "funcao_reaproveitada": "BA_inferencia.bootstrap + matriz_cadeias",
                                           "niveis": "cidade -> (cidade, seed); Q2-Q4 medidos dentro da cadeia"},
             "b2_cidade_cadeia_quadrante": {**b2, **bloco_ic(*b2["ic95"]), "niveis": "cidade -> (cidade, seed) -> Q (com reposicao)"},
             "b3_cidade_celula_seed_folha": {**b3, **bloco_ic(*b3["ic95"]), "niveis": "cidade -> celula (cidade, Q) -> seed (redacao F-R2.8.09)"}}

    # ---- (c) modelo misto ----
    c_out = misto_cidade_cadeia(D)
    c_sec = BA.misto(D)                                        # cidade + celula + cadeia (funcao reaproveitada)
    c_sec["formula"] = "delta ~ 1 + (1|cidade) + vc celula(cidade,Q) + vc cadeia(cidade,seed) [BA_inferencia.misto]"
    if c_sec.get("ajustado"):
        c_sec.update(bloco_ic(*c_sec["ic95_wald"]))
    c_out["secundario_BA_misto_com_celula"] = c_sec

    # ---- (d) VALID ----
    tem_valid = all("mae_rssi_valido_db" in le(p["chave_sc"])["selecao"]["test_no_melhor_ckpt"] for p in pares)
    d_out = {"disponivel": bool(tem_valid),
             "motivo": ("os run JSON c0c1cfsc_* trazem so mae_rssi_db (todos os nos de teste) e mae_pl_db (alvo valido, canal PL); "
                        "o MAE RSSI VALID existe so para a cadeia c0c1cf_* (t15_tres_populacoes.json); nao ha predicao por no dos "
                        "24 checkpoints do zero em fase_B/fase_C/fase_BA1, logo o Delta VALID exige inferencia nova (fora do escopo: "
                        "CPU, nenhum treino/inferencia)") if not tem_valid else "insumo presente"}

    # ---- (e) MDE80 / poder na unidade cidade ----
    sig = float(med_c.std(ddof=1)); n = 4
    mde_t = (float(st.t.ppf(0.975, n - 1)) + float(st.t.ppf(PODER_ALVO, n - 1))) * sig / math.sqrt(n)
    mde_z = (float(st.norm.ppf(0.975)) + float(st.norm.ppf(PODER_ALVO))) * sig / math.sqrt(n)
    p01 = poder_tost(sig, n, MARGEM)
    p01_sim = poder_tost_sim(sig, n, MARGEM, 200000, np.random.default_rng(SEED_BOOT))
    marg80 = float(optimize.brentq(lambda m: poder_tost(sig, n, m) - PODER_ALVO, 1e-6, 50 * sig))
    n80 = next(k for k in range(3, 1000) if poder_tost(sig, k, MARGEM) >= PODER_ALVO)
    e_out = {"sigma_observado_medias_cidade_db": sig, "n": n, "gl": n - 1, "alfa": ALFA, "poder_alvo": PODER_ALVO,
             "mde80_t_pareado_bilateral_db": mde_t, "mde80_z_db": mde_z,
             "formula_mde": "(t_0,975;3 + t_0,80;3) * sigma / sqrt(4) (convencao de critico-verificador_mde_margens.md, com n = 4)",
             "poder_tost_margem_0_1_sob_delta_0_exato": p01, "poder_tost_margem_0_1_sob_delta_0_simulado_200k": p01_sim,
             "menor_margem_com_poder_80_tost_db": marg80, "menor_n_cidades_com_poder_80_tost_margem_0_1": n80,
             "nota": "poder exato por integracao sobre a distribuicao de s (qui-quadrado), media normal; simulacao so confere"}

    # ---- poder do TOST publicado (n = 8 clusters), mesma conta, so para referencia ----
    ref8 = {"sigma_clusters_db": e8["dp_db"], "poder_tost_margem_0_1_exato_n8": poder_tost(e8["dp_db"], 8, MARGEM),
            "poder_publicado_por_simulacao_no_agregado": ref_t.get("poder_sob_delta_zero")}

    out = {"etapa": "D_tost_aninhado", "versao": "fase D minima (R2-8c)",
           "aprovacao": "ROADMAP_R4_2026-10-14.md secao 4 Fase D; 'aceito' do dono 19/10/2026 (TOST com cidades aninhadas)",
           "carimbo": carimbo(Path(__file__).resolve(), {k: str(v) for k, v in insumos.items()}),
           "sinal": "Delta = MAE RSSI(cadeia c0c1cf) - MAE RSSI(do zero c0c1cfsc), mesma celula/seed/split; Delta < 0 => cadeia com MAE menor",
           "populacao": "ALL (todos os nos de teste; selecao.test_no_melhor_ckpt.mae_rssi_db)",
           "nota_regime": ag.get("nota_regime"),
           "margem_db": MARGEM, "alfa": ALFA,
           "regra_destino": {"dentro": "equivalencia na unidade cidade", "cruza": "no systematic difference detectable at the city unit",
                             "fora": "diferenca", "proibido_no_cruza": ["parity", "equivalent"]},
           "passo1_reproducao": {"reproduz": reproduz, "unidade": "cluster (cidade, seed)", "n": e8["n"], "n_pares": 24,
                                 "recalculado": {**e8, "tost": t8}, "clusters_recalculados": clusters,
                                 "diff_abs_vs_agregado_json": diff_ag, "diff_abs_clusters_vs_agregado": diff_cl,
                                 "diff_abs_pares_run_json_vs_agregado": diff_pares,
                                 "arredondado_como_publicado": arred, "publicado_R3_tab9": PUBLICADO_R3, "diff_vs_publicado": diff_pub,
                                 "formulas": "identicas a agregar_controles_c1v2.py (statistics.mean/stdev, scipy.stats.t)",
                                 "criterio": "dados/media/dp/IC: diff 0 exato; p-valores: diff <= 1e-15 (ULP entre versoes do scipy)",
                                 "scipy_versao_aqui": __import__("scipy").__version__},
           "pares_24": pares,
           "a_unidade_cidade": a_out, "b_bootstrap_hierarquico": b_out, "c_modelo_misto": c_out, "d_valid": d_out,
           "e_mde_poder_unidade_cidade": e_out, "referencia_poder_n8": ref8}
    dump(out, a.saida)
    print(f"OK {a.saida}")
    print(f"reproducao n=8: media {e8['media_db']:+.6f} IC [{e8['ic95_db'][0]:+.6f}; {e8['ic95_db'][1]:+.6f}] p {e8['p_bicaudal']:.4f} "
          f"TOST {t8['p_tost']:.6f}  diff max agregado {max(diff_ag.values()):.3g} pares {diff_pares:.3g}")
    print(f"(a) cidade n=4: {a_out['media_db']:+.6f} IC95t [{a_out['ic95_t'][0]:+.6f}; {a_out['ic95_t'][1]:+.6f}] p {a_out['p_bilateral']:.4f} "
          f"TOST p {a_tost['p_tost']:.4f} -> {a_out['destino_ic95_t']}")
    for k in ("b1_cidade_cadeia_funcao_BA", "b2_cidade_cadeia_quadrante", "b3_cidade_celula_seed_folha"):
        b = b_out[k]; print(f"(b) {k}: {b['estimativa']:+.6f} [{b['ic95'][0]:+.6f}; {b['ic95'][1]:+.6f}] -> {b['destino']}")
    p = c_out["ajustes"]["default"]
    if p.get("ajustado"):
        print(f"(c) misto: {p['intercepto_db']:+.6f} ep {p['ep_db']:.6f} Wald [{p['ic95_wald_z'][0]:+.6f}; {p['ic95_wald_z'][1]:+.6f}] "
              f"conv {p['convergiu']} zeram {p['componentes_que_zeram']} avisos {len(p['avisos'])}")
    print(f"(d) VALID disponivel: {d_out['disponivel']}")
    print(f"(e) sigma {sig:.6f} MDE80 t {mde_t:.6f} poder TOST 0,1 {p01:.4f} (sim {p01_sim:.4f}) margem80 {marg80:.6f} n80 {n80}")


if __name__ == "__main__":
    main()
