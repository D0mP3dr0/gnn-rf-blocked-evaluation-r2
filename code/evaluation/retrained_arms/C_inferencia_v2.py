"""C_inferencia_v2.py — COPIA de ../metricas/C_inferencia.py (v1, sha256 9b3633e2...) com ACRESCIMOS MINIMOS, por emenda decidida
pelo dono em 18/10/2026 (ata AGENTES/cientista_dados/ATAS/ata_2026-10-18_ieee-access-47637-faseC.md, Bloco 4, "Decisao do dono":
"a medida em dB dos nos SENT entra na resposta ao R2-5b, rotulada como definida depois do protocolo"). Nada da v1 muda: mesmos
B, seed, indices de reamostragem (gerados ANTES e na mesma ordem), mesma ordem das metricas da v1; os campos novos vem DEPOIS.
Diff completo contra a v1 em C_inferencia_v2.diff. Acrescimos (V2-f..V2-j):
  V2-f  secundarias SENT (SENT_DB), mesmo bootstrap cidade -> cadeia B = 20000: vies_rssi_SENT, mae_rssi_SENT,
        mae_rssi_SENT_parte_abaixo_110, mae_rssi_SENT_parte_acima_110 e as contribuicoes ao MAE RSSI ALL
        (contrib_mae_rssi_ALL_VALID, _SENT, _SENT_abaixo_110, _SENT_acima_110, resto_mae_rssi_ALL).
  V2-g  vies_rssi_SENT no braco c2nomono recebe o criterio +-0,1 dB da Eq. 16 (destino_vies), ROTULADO
        "post hoc (decisao do dono 18/10)" — nao e pre-registrado; no c1noreg fica descritivo, como os vieses da v1.
        frac_pred_rssi_lt_sent_SENT continua descritiva; "tabela_SENT" a apresenta sempre ao lado dos valores em dB.
  V2-h  bloco "decomposicao_delta_mae_rssi_ALL": Delta ALL = Delta contrib VALID + Delta contrib SENT + Delta resto (identidade
        exata por corrida; resto ~ 0 por construcao), com IC de cada parte e a fracao do Delta ALL vinda de SENT.
  V2-i  assert n_pares == 80 em PRIMARIO, VIES_EQ16 e SENT_DB (salvo --permitir-incompleto).
  V2-j  entrada/saida padrao: C_metricas_por_corrida_v2.json -> C_resumo_v2.json; carimbo liga a v1 (insumo C_inferencia_v1.py).

--- cabecalho da v1 (inalterado) ---
C_inferencia.py — fase C (R4 IEEE Access 47637): endpoint e regras de destino PRE-REGISTRADOS no protocolo
_revisao_2026-10-14/PROTOCOLO_faseC_C1_C2_2026-10-14.md, secao "Endpoint, unidade e criterios fixados antes de rodar"
(aprovado pelo dono 14/10/2026; "va" da avaliacao 18/10/2026). Le SO C_metricas_por_corrida.json; grava SO C_resumo.json ao lado.

Endpoint primario: Delta_braco = metrica(braco) - metrica(C0), pareado por (cidade, seed, Q); populacoes ALL e VALID; MAE RSSI.
Tambem: MAE PL VALID (com e sem alvo > 200 dB), vies de PL/RSSI VALID (e estrato < 1 km), cada taxa de violacao, |Delta| B3,
canais 1 e 2, fracao de RSSI previsto abaixo do piso do sentinela (-110 dBm).
Inferencia primaria: bootstrap hierarquico cidade -> cadeia (cadeia = braco x cidade x seed, os 4 Q dentro), B = 20000,
IC95 percentil, rng numpy default_rng(20261018); estatistica = media das medias de cadeia (Q disponiveis) das cadeias sorteadas
(= media das 80 Delta quando completo). Os MESMOS indices de reamostragem para todas as metricas e bracos.
Confirmacao: (i) MixedLM (statsmodels) Delta ~ 1 com efeitos aleatorios de cidade (grupo), celula e cadeia (componentes de
variancia aninhados na cidade), REML; convergencia e avisos registrados; (ii) t sobre as 4 medias de cidade (descritivo;
cidades parcialmente sobrepostas). Sensibilidade declarada (nao substitui o pareado): Delta contra a media das 5 seeds do C0 na
celula.
Destinos (exatamente como no protocolo):
  MAE RSSI, delta = 0,045 dB (ALL) e 0,35 dB (VALID):
    IC inteiro em [-delta, +delta] -> "nao muda de forma resolvivel"; IC inteiro < -delta -> braco sem regularizador melhor,
    "vai ao dono (inverte)"; IC inteiro > +delta -> "regularizadores ajudam (reforca)"; IC cruzando +-delta -> "inconclusivo".
  Vies da Eq. 16 (C2 vs C0, erro com sinal em VALID), margem 0,1 dB: IC inteiro fora de [-0,1, +0,1] -> "vies atribuivel ao
    termo"; IC inteiro dentro -> "sem vies atribuivel acima de 0,1 dB"; cruzando -> "inconclusivo".
  Demais metricas: descritivas (sem margem pre-registrada); registra-se apenas se o IC exclui 0.
Uso: python C_inferencia.py [--metricas C_metricas_por_corrida.json] [--saida C_resumo.json] [--permitir-incompleto]
Numeros PROVISORIOS ate contra-auditoria."""
import sys, json, argparse, warnings
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent
FASE_B = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_B")
sys.path.insert(0, str(FASE_B)); sys.dont_write_bytecode = True
from common import CIDADES, QS, SEEDS, carimbo, dump  # noqa: E402

B_BOOT = 20000
SEED_BOOT = 20261018
DELTA_MAE = {"ALL": 0.045, "VALID": 0.35}
DELTA_VIES = 0.1
BRACOS = ["c1noreg", "c2nomono"]

PRIMARIO = ["mae_rssi_ALL", "mae_rssi_VALID"]
VIES_EQ16 = ["vies_rssi_VALID", "vies_pl_VALID"]
SECUNDARIOS = ["mae_pl_VALID", "mae_pl_VALID_ate200", "vies_rssi_ALL",
               "vies_pl_VALID_lt1km", "vies_rssi_VALID_lt1km", "mae_pl_VALID_lt1km", "mae_rssi_VALID_lt1km",
               "frac_pred_rssi_lt_sent_ALL", "frac_pred_rssi_lt_sent_VALID", "frac_pred_rssi_lt_sent_SENT",
               "frac_viol_srv_VALID", "frac_viol_atual_VALID", "frac_viol_atual_ALL",
               "pred_mono_cod_viol_frac", "pred_mono_fis_viol_frac", "pred_mono_fis_viol_frac_crit_codigo", "pred_mono_fis_viol_frac_estrito",
               "abs_delta_mediana", "abs_delta_p95", "abs_delta_mae", "delta_vies", "concord_ALL", "kappa_ALL", "concord_VALID", "kappa_VALID",
               "mae_c1_ALL", "vies_c1_ALL", "r2_c1_ALL", "mae_c1_VALID", "vies_c1_VALID", "r2_c1_VALID",
               "mae_c2_ALL", "vies_c2_ALL", "r2_c2_ALL", "mae_c2_VALID", "vies_c2_VALID", "r2_c2_VALID"]
# V2-f: secundarias SENT em dB (definidas depois do protocolo; decisao do dono 18/10/2026)
SENT_DB = ["vies_rssi_SENT", "mae_rssi_SENT", "mae_rssi_SENT_parte_abaixo_110", "mae_rssi_SENT_parte_acima_110",
           "contrib_mae_rssi_ALL_VALID", "contrib_mae_rssi_ALL_SENT", "contrib_mae_rssi_ALL_SENT_abaixo_110",
           "contrib_mae_rssi_ALL_SENT_acima_110", "resto_mae_rssi_ALL"]
ROTULO_POST_HOC = "post hoc (decisão do dono 18/10)"


def destino_mae(lo, hi, d):
    if lo >= -d and hi <= d: return "IC inteiro dentro de +-delta: os regularizadores nao mudam o erro de forma resolvivel"
    if hi < -d: return "IC inteiro abaixo de -delta: braco sem regularizador melhor (regularizadores prejudicam) -> vai ao dono (inverte)"
    if lo > d: return "IC inteiro acima de +delta: regularizadores ajudam (reforca)"
    return "IC cruza +-delta: inconclusivo"


def destino_vies(lo, hi, d=DELTA_VIES):
    if lo > d or hi < -d: return "IC inteiro fora de +-0,1 dB: vies atribuivel ao termo (Eq. 16)"
    if lo >= -d and hi <= d: return "IC inteiro dentro de +-0,1 dB: sem vies atribuivel ao termo acima de 0,1 dB"
    return "IC cruza +-0,1 dB: inconclusivo"


def matriz_cadeias(D):
    """D: {(c, s, q): delta}. Devolve CM (4 cidades x 5 seeds) com a media das Delta de cada cadeia (Q disponiveis), NaN se vazia."""
    CM = np.full((len(CIDADES), len(SEEDS)), np.nan)
    for ic, c in enumerate(CIDADES):
        for js, s in enumerate(SEEDS):
            v = [D[(c, s, q)] for q in QS if (c, s, q) in D]
            if v: CM[ic, js] = float(np.mean(v))
    return CM


def bootstrap(CM, IC_, IS_):
    vals = CM[IC_[:, :, None], IS_]                       # (B, 4, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        st = np.nanmean(vals.reshape(len(IC_), -1), axis=1)
    st = st[np.isfinite(st)]
    return {"estimativa": float(np.nanmean(CM)), "ic95": [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))],
            "B_validas": int(st.size), "ep_bootstrap": float(st.std(ddof=1))}


def misto(D):
    import pandas as pd
    import statsmodels.formula.api as smf
    rows = [{"delta": v, "cidade": c, "celula": f"{c}_{q}", "cadeia": f"{c}_s{s}"} for (c, s, q), v in D.items()]
    df = pd.DataFrame(rows)
    if df["delta"].std() == 0 or len(df) < 8:
        return {"ajustado": False, "motivo": "Delta constante ou n < 8"}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            md = smf.mixedlm("delta ~ 1", df, groups="cidade", re_formula="1",
                             vc_formula={"celula": "0 + C(celula)", "cadeia": "0 + C(cadeia)"})
            f = md.fit(reml=True)
            ci = f.conf_int().loc["Intercept"].tolist()
            o = {"ajustado": True, "convergiu": bool(f.converged), "intercepto": float(f.params["Intercept"]), "ep": float(f.bse["Intercept"]),
                 "ic95_wald": [float(ci[0]), float(ci[1])], "var_cidade": float(f.cov_re.iloc[0, 0]),
                 "var_componentes": {k: float(v) for k, v in zip(md.exog_vc.names, f.vcomp)}, "var_residual": float(f.scale), "n_obs": int(len(df))}
        except Exception as e:  # registrar, nao esconder
            o = {"ajustado": False, "motivo": f"{type(e).__name__}: {e}"}
    o["avisos"] = sorted({f"{x.category.__name__}: {str(x.message)[:160]}" for x in w})
    return o


def t_cidades(D):
    from scipy import stats
    m = [np.mean([v for (c, s, q), v in D.items() if c == cc]) for cc in CIDADES if any(k[0] == cc for k in D)]
    m = np.asarray(m, float)
    if m.size < 2 or m.std(ddof=1) == 0:
        return {"medias_cidade": m.tolist(), "n_cidades": int(m.size), "t": None}
    ep = m.std(ddof=1) / np.sqrt(m.size); tq = stats.t.ppf(0.975, m.size - 1)
    tt = stats.ttest_1samp(m, 0.0)
    return {"medias_cidade": dict(zip([c for c in CIDADES if any(k[0] == c for k in D)], m.tolist())), "n_cidades": int(m.size), "media": float(m.mean()),
            "ic95_t": [float(m.mean() - tq * ep), float(m.mean() + tq * ep)], "t": float(tt.statistic), "p_bilateral": float(tt.pvalue), "gl": int(m.size - 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metricas", default=str(AQUI / "C_metricas_por_corrida_v2.json"))  # V2-j
    ap.add_argument("--saida", default=str(AQUI / "C_resumo_v2.json"))  # V2-j
    ap.add_argument("--permitir-incompleto", action="store_true")
    ap.add_argument("--sem-misto", action="store_true", help="pula o MixedLM (teste rapido)")
    a = ap.parse_args()
    M = json.load(open(a.metricas))
    R = {(r["braco"], r["cidade"], r["seed"], r["Q"]): r for r in M["runs"]}
    cont = {b: sum(1 for k in R if k[0] == b) for b in ["gnn"] + BRACOS}
    if not a.permitir_incompleto:
        assert M.get("completo") and all(v == 80 for v in cont.values()), f"metricas incompletas: {cont}"
    rng = np.random.default_rng(SEED_BOOT)
    IC_ = rng.integers(0, len(CIDADES), (B_BOOT, len(CIDADES)))
    IS_ = rng.integers(0, len(SEEDS), (B_BOOT, len(CIDADES), len(SEEDS)))
    out = {"etapa": "C_inferencia", "versao": "v2 (emenda do dono 18/10/2026)", "protocolo": "_revisao_2026-10-14/PROTOCOLO_faseC_C1_C2_2026-10-14.md#Endpoint", "n_runs": cont,
           "metricas_completas": bool(M.get("completo")),
           "parametros": {"B": B_BOOT, "rng_bootstrap": f"numpy default_rng({SEED_BOOT}); indices comuns a todas as metricas",
                          "delta_MAE_RSSI": DELTA_MAE, "delta_vies_eq16": DELTA_VIES, "IC": "percentil 2,5-97,5",
                          "sinal": "Delta = braco - C0; Delta < 0 => braco (sem regularizador) com metrica menor"},
           "bracos": {}}
    todas = PRIMARIO + VIES_EQ16 + SECUNDARIOS + SENT_DB  # V2-f: novas no fim
    for b in BRACOS:
        ob = {}
        for k in todas:
            D, Ds = {}, {}; a0, a1 = [], []
            for c in CIDADES:
                for q in QS:
                    c0cel = [R[("gnn", c, s, q)][k] for s in SEEDS if ("gnn", c, s, q) in R and R[("gnn", c, s, q)].get(k) is not None]
                    mc0 = float(np.mean(c0cel)) if c0cel else None
                    for s in SEEDS:
                        rb, r0 = R.get((b, c, s, q)), R.get(("gnn", c, s, q))
                        if rb is None or r0 is None or rb.get(k) is None or r0.get(k) is None: continue
                        D[(c, s, q)] = rb[k] - r0[k]; Ds[(c, s, q)] = rb[k] - mc0; a1.append(rb[k]); a0.append(r0[k])
            if not a.permitir_incompleto and k in PRIMARIO + VIES_EQ16 + SENT_DB:  # V2-i
                assert len(D) == 80, f"V2-i n_pares {len(D)} != 80 em {b}/{k}"
            if not D:
                ob[k] = {"n_pares": 0, "motivo": "sem pares disponiveis"}; continue
            CM = matriz_cadeias(D); bs = bootstrap(CM, IC_, IS_); bss = bootstrap(matriz_cadeias(Ds), IC_, IS_)
            e = {"n_pares": len(D), "n_celulas": len({(c, q) for c, s, q in D}), "n_cadeias": int(np.isfinite(CM).sum()),
                 "media_braco_runs_pareados": float(np.mean(a1)), "media_C0_runs_pareados": float(np.mean(a0)),
                 "delta": bs["estimativa"], "ic95": bs["ic95"], "ep_bootstrap": bs["ep_bootstrap"], "B_validas": bs["B_validas"],
                 "ic_exclui_zero": bool(bs["ic95"][0] > 0 or bs["ic95"][1] < 0),
                 "delta_por_cidade": {c: float(np.nanmean(CM[i])) if np.isfinite(CM[i]).any() else None for i, c in enumerate(CIDADES)},
                 "delta_por_celula": {f"{c}_{q}": float(np.mean([D[(c, s, q)] for s in SEEDS if (c, s, q) in D])) for c in CIDADES for q in QS
                                      if any((c, s, q) in D for s in SEEDS)},
                 "t_4_medias_cidade": t_cidades(D),
                 "sensibilidade_vs_media_5_seeds_C0": {"delta": bss["estimativa"], "ic95": bss["ic95"]}}
            if not a.sem_misto and (k in PRIMARIO or k in VIES_EQ16 or k in ("mae_pl_VALID", "mae_pl_VALID_ate200", "frac_viol_srv_VALID",
                                                                              "pred_mono_cod_viol_frac", "pred_mono_fis_viol_frac")):
                e["modelo_misto"] = misto(D)
            if k in PRIMARIO:
                pop = k.split("_")[-1]; d = DELTA_MAE[pop]
                e["papel"] = "endpoint primario"; e["delta_margem"] = d
                e["destino"] = destino_mae(*bs["ic95"], d)
                e["destino_sensibilidade"] = destino_mae(*bss["ic95"], d)
            elif k in VIES_EQ16 and b == "c2nomono":
                e["papel"] = "vies da Eq. 16 (C2 vs C0)"; e["delta_margem"] = DELTA_VIES
                e["destino"] = destino_vies(*bs["ic95"]); e["destino_sensibilidade"] = destino_vies(*bss["ic95"])
            elif k == "vies_rssi_SENT" and b == "c2nomono":  # V2-g
                e["papel"] = "vies da Eq. 16 em SENT (C2 vs C0), secundaria"; e["delta_margem"] = DELTA_VIES
                e["rotulo_criterio"] = ROTULO_POST_HOC
                e["destino"] = destino_vies(*bs["ic95"]) + " [" + ROTULO_POST_HOC + "]"
                e["destino_sensibilidade"] = destino_vies(*bss["ic95"]) + " [" + ROTULO_POST_HOC + "]"
            elif k in SENT_DB:  # V2-f
                e["papel"] = "secundaria SENT em dB, definida depois do protocolo (" + ROTULO_POST_HOC + "), descritiva"; e["destino"] = None
            else:
                e["papel"] = "descritivo (sem margem pre-registrada)"; e["destino"] = None
            ob[k] = e
        out["bracos"][b] = ob
    # tabela curta
    tab = []
    for b in BRACOS:
        for k in PRIMARIO + VIES_EQ16 + ["mae_pl_VALID", "mae_pl_VALID_ate200", "frac_pred_rssi_lt_sent_VALID", "frac_viol_srv_VALID",
                                        "pred_mono_cod_viol_frac", "pred_mono_fis_viol_frac", "abs_delta_mae"]:
            e = out["bracos"][b].get(k, {})
            if e.get("n_pares"):
                tab.append({"braco": b, "metrica": k, "delta": e["delta"], "ic95": e["ic95"], "n_pares": e["n_pares"], "destino": e["destino"],
                            "sensibilidade_delta": e["sensibilidade_vs_media_5_seeds_C0"]["delta"], "sensibilidade_ic95": e["sensibilidade_vs_media_5_seeds_C0"]["ic95"],
                            "destino_sensibilidade": e.get("destino_sensibilidade")})
    out["tabela"] = tab
    # V2-g: tabela SENT — a fracao (descritiva) sempre ao lado dos valores em dB
    tsent = []
    for b in BRACOS:
        for k in ["frac_pred_rssi_lt_sent_SENT", "vies_rssi_SENT", "mae_rssi_SENT", "mae_rssi_SENT_parte_abaixo_110", "mae_rssi_SENT_parte_acima_110"]:
            e = out["bracos"][b][k]
            tsent.append({"braco": b, "metrica": k, "unidade": "fracao" if k.startswith("frac") else "dB",
                          "media_C0": e["media_C0_runs_pareados"], "media_braco": e["media_braco_runs_pareados"], "delta": e["delta"], "ic95": e["ic95"],
                          "n_pares": e["n_pares"], "ic_exclui_zero": e["ic_exclui_zero"], "papel": e["papel"], "destino": e["destino"],
                          "rotulo": e.get("rotulo_criterio", ROTULO_POST_HOC if k in SENT_DB else "descritiva (v1)")})
    out["tabela_SENT"] = tsent
    # V2-h: decomposicao exata do Delta MAE RSSI ALL
    dec = {}
    for b in BRACOS:
        ob = out["bracos"][b]
        res = [abs((R[(b, c, s, q)]["mae_rssi_ALL"] - R[("gnn", c, s, q)]["mae_rssi_ALL"])
                   - (R[(b, c, s, q)]["contrib_mae_rssi_ALL_VALID"] - R[("gnn", c, s, q)]["contrib_mae_rssi_ALL_VALID"])
                   - (R[(b, c, s, q)]["contrib_mae_rssi_ALL_SENT"] - R[("gnn", c, s, q)]["contrib_mae_rssi_ALL_SENT"]))
               for c in CIDADES for q in QS for s in SEEDS if (b, c, s, q) in R and ("gnn", c, s, q) in R]
        tot = ob["mae_rssi_ALL"]["delta"]
        part = lambda k: {"delta": ob[k]["delta"], "ic95": ob[k]["ic95"]}
        dec[b] = {"delta_mae_rssi_ALL": tot, "ic95_ALL": ob["mae_rssi_ALL"]["ic95"],
                  "contrib_VALID": part("contrib_mae_rssi_ALL_VALID"), "contrib_SENT": part("contrib_mae_rssi_ALL_SENT"),
                  "contrib_SENT_abaixo_110": part("contrib_mae_rssi_ALL_SENT_abaixo_110"), "contrib_SENT_acima_110": part("contrib_mae_rssi_ALL_SENT_acima_110"),
                  "resto": part("resto_mae_rssi_ALL"),
                  "residuo_identidade_na_media": float(tot - ob["contrib_mae_rssi_ALL_VALID"]["delta"] - ob["contrib_mae_rssi_ALL_SENT"]["delta"]
                                                       - ob["resto_mae_rssi_ALL"]["delta"]),
                  "residuo_identidade_max_abs_por_corrida": float(max(res)) if res else None, "n_pares": len(res),
                  "fracao_do_delta_ALL_vinda_de_SENT": float(ob["contrib_mae_rssi_ALL_SENT"]["delta"] / tot) if tot else None,
                  "rotulo": ROTULO_POST_HOC}
    out["decomposicao_delta_mae_rssi_ALL"] = dec
    out["carimbo"] = carimbo(Path(__file__), {"C_metricas_por_corrida_v2.json": a.metricas, "common.py": FASE_B / "common.py",
                                              "C_inferencia_v1.py": AQUI.parent / "metricas" / "C_inferencia.py"})  # V2-j
    dump(out, a.saida)
    for t in tab:
        print("%-9s %-30s D=%+.4f IC=[%+.4f, %+.4f] n=%d | %s" % (t["braco"], t["metrica"], t["delta"], t["ic95"][0], t["ic95"][1], t["n_pares"], t["destino"]))
    for t in tsent:  # V2-g
        print("SENT %-9s %-32s C0=%+.4f br=%+.4f D=%+.4f IC=[%+.4f, %+.4f] n=%d | %s | %s" % (t["braco"], t["metrica"], t["media_C0"], t["media_braco"], t["delta"],
              t["ic95"][0], t["ic95"][1], t["n_pares"], t["rotulo"], t["destino"]))
    for b, d in dec.items():  # V2-h
        print("DECOMP %-9s D_ALL=%+.4f = VALID %+.4f + SENT %+.4f (abaixo %+.4f, acima %+.4f) + resto %+.2e | frac SENT %.4f | res max %.1e" % (
              b, d["delta_mae_rssi_ALL"], d["contrib_VALID"]["delta"], d["contrib_SENT"]["delta"], d["contrib_SENT_abaixo_110"]["delta"],
              d["contrib_SENT_acima_110"]["delta"], d["resto"]["delta"], d["fracao_do_delta_ALL_vinda_de_SENT"], d["residuo_identidade_max_abs_por_corrida"]))
    print("gravado", a.saida)


if __name__ == "__main__":
    main()
