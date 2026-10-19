"""BA_inferencia.py — avaliacao OFICIAL do BA1 (R2-3): endpoint e destinos do PROTOCOLO_faseC_baselines_RASCUNHO_2026-10-14.md,
secoes 7 e 8, com o escopo da "Decisao do dono (15/10/2026)" (so BA1; BF e BA2-BA4 caem). COPIA de
fase_C/avaliacao/metricas_v2/C_inferencia_v2.py com ADAPTACAO MINIMA (diff completo em BA_inferencia.diff). Mantidos sem mudanca:
B = 20000, rng numpy default_rng(20261018) com os indices de reamostragem gerados ANTES e comuns a todas as metricas e contrastes,
matriz_cadeias (cadeia = cidade x seed, os 4 Q juntos), bootstrap (IC95 percentil), misto (MixedLM REML), t_cidades (df = 3),
sensibilidade contra a media das 5 seeds do comparador, decomposicao exata do Delta MAE RSSI ALL (V2-h). Adaptacoes (BA-g..BA-n):
  BA-g  contrastes Delta = BA1 - comparador, pareado por (cidade, seed, Q): BA1_menos_C0 (gnn = c0c1cf; endpoint primario),
        BA1_menos_MLP (mlpcf, mesma seed; descritivo) e BA1_menos_empirico (melhor empirico da celula, B1_empiricos.json, repetido
        nas 5 seeds; descritivo). Sinal: Delta > 0 = BA1 com MAE maior (baseline pior que o comparador).
  BA-h  destino_mae com o texto da secao 8 (IC acima de +delta: o GNN-RF supera o BA1, reforca; IC dentro de +-delta: equivalencia,
        delimita; IC abaixo de -delta: o baseline vence, vai ao dono antes de qualquer redacao; IC cruzando: inconclusivo). Regra
        numerica identica a da v2. Destino so no BA1_menos_C0 x {ALL, VALID}; o resto e descritivo (destino None).
  BA-i  metricas: primario mae_rssi_ALL e mae_rssi_VALID; secundarias mae_rssi_COV85 (so celulas n >= 1000, tabela por celula com n),
        mae_pl_VALID, mae_pl_VALID_ate200, mae_rssi_VALID_ate200 e mae_rssi_SENT (uma linha); vieses da Eq. 16, metricas B2-B4 e
        monotonicidade da v2 NAO entram (fora da secao 7 do protocolo de baselines; continuam por corrida no JSON de metricas).
  BA-j  leitura so Q1 (sem cadeia, descritiva): mesmos Delta e IC restritos a Q = Q1 (cada "cadeia" do bootstrap tem 1 quadrante).
  BA-k  declaracao do nivel superior: multiconjuntos de cidade = comb(4 + 4 - 1, 4) = 35.
  BA-l  parametros nominais/efetivos e tempos de treino e de inferencia por braco (custo_por_corrida do JSON de metricas), com a
        conferencia dos nominais pedidos (BA1 369.827; GNN-RF 1.812.515).
  BA-m  assert n_pares == 80 no primario, em mae_rssi_SENT e na decomposicao; COV85 = 5 x celulas elegiveis.
  BA-n  entrada/saida: BA_metricas_por_corrida.json -> BA_resumo.json; carimbo liga a origem (insumo C_inferencia_v2.py).

--- cabecalho da v2 (inalterado) ---
C_inferencia_v2.py — COPIA de ../metricas/C_inferencia.py (v1, sha256 9b3633e2...) com ACRESCIMOS MINIMOS, por emenda decidida
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
import sys, json, argparse, warnings, math  # BA-k: math.comb
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent
FASE_B = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_B")
sys.path.insert(0, str(FASE_B)); sys.dont_write_bytecode = True
from common import CIDADES, QS, SEEDS, carimbo, dump  # noqa: E402
FASE_C_V2 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_C/avaliacao/metricas_v2")  # BA-n: origem, so leitura

B_BOOT = 20000
SEED_BOOT = 20261018
DELTA_MAE = {"ALL": 0.045, "VALID": 0.35}
CONTRASTES = {"BA1_menos_C0": "gnn", "BA1_menos_MLP": "mlp", "BA1_menos_empirico": "empirico"}  # BA-g
ROTULO_COMP = {"gnn": "GNN-RF (C0 = c0c1cf)", "mlp": "MLP sem grafo (mlpcf)", "empirico": "melhor empirico da celula (B1_empiricos.json)"}

PRIMARIO = ["mae_rssi_ALL", "mae_rssi_VALID"]
SECUNDARIOS = ["mae_rssi_COV85", "mae_pl_VALID", "mae_pl_VALID_ate200", "mae_rssi_VALID_ate200", "mae_rssi_SENT"]  # BA-i
DECOMP = ["contrib_mae_rssi_ALL_VALID", "contrib_mae_rssi_ALL_SENT", "contrib_mae_rssi_ALL_SENT_abaixo_110",
          "contrib_mae_rssi_ALL_SENT_acima_110", "resto_mae_rssi_ALL"]  # V2-h
NOMINAIS = {"ba1": 369827, "gnn": 1812515}  # BA-l: valores pedidos, conferidos contra o run JSON


def destino_mae(lo, hi, d):  # BA-h: texto da secao 8 do protocolo de baselines; regra numerica da v2
    if lo >= -d and hi <= d: return "IC inteiro dentro de +-delta: equivalencia (delimita)"
    if hi < -d: return "IC inteiro abaixo de -delta: o baseline vence o GNN-RF -> vai ao dono antes de qualquer redacao (secao 8, iii)"
    if lo > d: return "IC inteiro acima de +delta: o GNN-RF supera o BA1 (reforca)"
    return "IC cruza +-delta: inconclusivo"


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


def valor(R, E, comp, c, s, q, k):  # BA-g/BA-i: valor do braco/comparador; COV85 so com n >= 1000
    if comp == "empirico":
        return (E.get(f"{c}_{q}") or {}).get(k)
    r = R.get((comp, c, s, q))
    if r is None: return None
    if k == "mae_rssi_COV85" and not r["COV85_n_ge_1000"]: return None
    return r.get(k)


def contraste(R, E, comp, k, IC_, IS_, qs=QS, com_misto=False):
    """Bloco da v2 (laco por metrica) fatorado: Delta = BA1 - comp pareado por (c, s, q), bootstrap, t, sensibilidade, misto."""
    D, Ds = {}, {}; a0, a1 = [], []
    for c in CIDADES:
        for q in qs:
            cc = [valor(R, E, comp, c, s, q, k) for s in SEEDS]; cc = [x for x in cc if x is not None]
            mc0 = float(np.mean(cc)) if cc else None
            for s in SEEDS:
                vb, v0 = valor(R, E, "ba1", c, s, q, k), valor(R, E, comp, c, s, q, k)
                if vb is None or v0 is None: continue
                D[(c, s, q)] = vb - v0; Ds[(c, s, q)] = vb - mc0; a1.append(vb); a0.append(v0)
    if not D:
        return {"n_pares": 0, "motivo": "sem pares disponiveis"}, D
    CM = matriz_cadeias(D); bs = bootstrap(CM, IC_, IS_); bss = bootstrap(matriz_cadeias(Ds), IC_, IS_)
    e = {"n_pares": len(D), "n_celulas": len({(c, q) for c, s, q in D}), "n_cadeias": int(np.isfinite(CM).sum()),
         "media_BA1_runs_pareados": float(np.mean(a1)), "media_comparador_runs_pareados": float(np.mean(a0)),
         "delta": bs["estimativa"], "ic95": bs["ic95"], "ep_bootstrap": bs["ep_bootstrap"], "B_validas": bs["B_validas"],
         "ic_exclui_zero": bool(bs["ic95"][0] > 0 or bs["ic95"][1] < 0),
         "delta_por_cidade": {c: float(np.nanmean(CM[i])) if np.isfinite(CM[i]).any() else None for i, c in enumerate(CIDADES)},
         "delta_por_celula": {f"{c}_{q}": float(np.mean([D[(c, s, q)] for s in SEEDS if (c, s, q) in D])) for c in CIDADES for q in qs
                              if any((c, s, q) in D for s in SEEDS)},
         "t_4_medias_cidade": t_cidades(D),
         "sensibilidade_vs_media_5_seeds_comparador": {"delta": bss["estimativa"], "ic95": bss["ic95"]}}
    if com_misto:
        e["modelo_misto"] = misto(D)
    return e, D


def resumo_custo(M):  # BA-l
    o = {}
    for b in ("ba1", "gnn", "mlp"):
        L = [x for x in M["custo_por_corrida"] if x["braco"] == b]

        def st(k):
            v = np.asarray([x[k] for x in L if x.get(k) is not None], float)
            if not v.size: return {"n": 0}
            o_ = {"n": int(v.size), "mediana": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "soma": float(v.sum())}
            if k.startswith(("tempo", "dt_")): o_["soma_h"] = float(v.sum() / 3600.0)
            return o_
        o[b] = {"n_corridas": len(L), "n_params_nominal_valores": sorted({x["n_params"] for x in L if x["n_params"] is not None}),
                "n_params_treinaveis_valores": sorted({x["n_params_treinaveis"] for x in L if x["n_params_treinaveis"] is not None}),
                "n_params_com_gradiente_1o_lote_valores": sorted({x["n_params_com_gradiente_1o_lote"] for x in L if x["n_params_com_gradiente_1o_lote"] is not None}),
                "tempo_treino_s": st("tempo_treino_s"), "vram_peak_mb": st("vram_peak_mb"), "n_epocas_rodadas_valores": sorted({x["n_epocas_rodadas"] for x in L}),
                "tempo_inferencia_parede_s": st("tempo_inferencia_parede_s"), "tempo_inferencia_forward_s": st("tempo_inferencia_forward_s"),
                "plataformas_treino": sorted({str(x["plataforma_treino"]) for x in L}), "gpus_treino": sorted({str(x["gpu_treino"]) for x in L}),
                "plataformas_inferencia": sorted({str(x["plataforma_inferencia"]) for x in L}),
                "fontes": {"n_params": "run JSON modelo.n_params / modelo.n_params_treinaveis", "efetivos": "run JSON censo_gradiente_1o_lote.n_params_com_gradiente",
                           "tempo_treino_s": "run JSON custo.tempo_total_s", "tempo_inferencia": "sidecar .json custo.tempo_parede_total_s / tempo_forward_s"}}
        if b == "ba1":
            o[b]["dt_lote_inferencia_s"] = st("dt_lote_inferencia_s"); o[b]["dt_lote_logs"] = sorted({str(x.get("dt_lote_log")) for x in L})
        if b in NOMINAIS:
            o[b]["conferencia_nominal"] = {"pedido": NOMINAIS[b], "lido": o[b]["n_params_treinaveis_valores"],
                                           "ok": o[b]["n_params_treinaveis_valores"] == [NOMINAIS[b]] and o[b]["n_params_nominal_valores"] == [NOMINAIS[b]]}
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metricas", default=str(AQUI / "BA_metricas_por_corrida.json"))  # BA-n
    ap.add_argument("--saida", default=str(AQUI / "BA_resumo.json"))  # BA-n
    ap.add_argument("--permitir-incompleto", action="store_true")
    ap.add_argument("--sem-misto", action="store_true", help="pula o MixedLM (teste rapido)")
    a = ap.parse_args()
    M = json.load(open(a.metricas))
    R = {(r["braco"], r["cidade"], r["seed"], r["Q"]): r for r in M["runs"]}
    E = M["empirico_por_celula"]
    cont = {b: sum(1 for k in R if k[0] == b) for b in ["gnn", "ba1", "mlp"]}
    if not a.permitir_incompleto:
        assert M.get("completo") and all(v == 80 for v in cont.values()) and len(E) == 16, f"metricas incompletas: {cont}, empirico {len(E)}"
    rng = np.random.default_rng(SEED_BOOT)
    IC_ = rng.integers(0, len(CIDADES), (B_BOOT, len(CIDADES)))
    IS_ = rng.integers(0, len(SEEDS), (B_BOOT, len(CIDADES), len(SEEDS)))
    n_multi = math.comb(len(CIDADES) + len(CIDADES) - 1, len(CIDADES))  # BA-k
    out = {"etapa": "BA_inferencia", "versao": "BA1 oficial (copia adaptada de C_inferencia_v2.py)",
           "protocolo": "_revisao_2026-10-14/PROTOCOLO_faseC_baselines_RASCUNHO_2026-10-14.md#7-8 + Decisao do dono (15/10/2026)", "n_runs": cont,
           "metricas_completas": bool(M.get("completo")),
           "parametros": {"B": B_BOOT, "rng_bootstrap": f"numpy default_rng({SEED_BOOT}); indices comuns a todas as metricas e contrastes",
                          "unidade": "bootstrap hierarquico cidade -> cadeia (cadeia = cidade x seed, os 4 Q juntos); estatistica = media das medias de cadeia",
                          "nivel_superior_multiconjuntos_de_cidade": n_multi,
                          "declaracao_nivel_superior": f"o nivel superior (4 cidades reamostradas com reposicao) tem so {n_multi} multiconjuntos de cidade distintos",
                          "delta_MAE_RSSI": DELTA_MAE, "IC": "percentil 2,5-97,5",
                          "confirmacao": "t sobre as 4 medias de cidade (df = 3, descritivo: cidades parcialmente sobrepostas); MixedLM REML no primario",
                          "sinal": "Delta = BA1 - comparador; Delta > 0 => BA1 com MAE maior (baseline pior)"},
           "comparadores": ROTULO_COMP, "contrastes": {}}
    for nome, comp in CONTRASTES.items():
        ob = {}
        for k in PRIMARIO + SECUNDARIOS + ([] if comp == "empirico" else DECOMP):
            e, D = contraste(R, E, comp, k, IC_, IS_, com_misto=(k in PRIMARIO and not a.sem_misto))
            if not a.permitir_incompleto and (k in PRIMARIO or k == "mae_rssi_SENT" or k in DECOMP):  # BA-m
                assert len(D) == 80, f"BA-m n_pares {len(D)} != 80 em {nome}/{k}"
            if k == "mae_rssi_COV85" and e.get("n_pares"):  # BA-i: tabela por celula com n
                el = {kk: v["n_COV85"] for kk, v in M["celulas"].items()}
                e["celulas_elegiveis_n_ge_1000"] = sorted(kk for kk, v in el.items() if v >= 1000)
                e["n_COV85_por_celula"] = el
                e["tabela_por_celula"] = [{"celula": kk, "n_COV85": el[kk], "delta_media_5_seeds": e["delta_por_celula"].get(kk)} for kk in el]
                if not a.permitir_incompleto:
                    assert e["n_pares"] == 5 * len(e["celulas_elegiveis_n_ge_1000"]), "BA-m COV85 n_pares"
            if k in PRIMARIO and comp == "gnn" and e.get("n_pares"):
                pop = k.split("_")[-1]; d = DELTA_MAE[pop]
                e["papel"] = "endpoint primario (secao 7)"; e["delta_margem"] = d
                e["destino"] = destino_mae(*e["ic95"], d)
                e["destino_sensibilidade"] = destino_mae(*e["sensibilidade_vs_media_5_seeds_comparador"]["ic95"], d)
            elif k in DECOMP:
                e["papel"] = "parcela da decomposicao do Delta MAE RSSI ALL (V2-h), descritiva"; e["destino"] = None
            elif e.get("n_pares"):
                e["papel"] = "secundaria descritiva (sem margem)" if comp == "gnn" else "descritiva"; e["destino"] = None
            ob[k] = e
        ob["so_Q1"] = {}  # BA-j
        for k in PRIMARIO:
            e, D = contraste(R, E, comp, k, IC_, IS_, qs=["Q1"])
            if not a.permitir_incompleto:
                assert len(D) == 20, f"BA-j n_pares Q1 {len(D)} != 20 em {nome}/{k}"
            e["papel"] = "leitura so Q1 (sem cadeia), descritiva"; e["destino"] = None
            ob["so_Q1"][k] = e
        out["contrastes"][nome] = ob
    # tabela curta
    tab = []
    for nome in CONTRASTES:
        for k in PRIMARIO + SECUNDARIOS:
            e = out["contrastes"][nome].get(k, {})
            if e.get("n_pares"):
                tab.append({"contraste": nome, "metrica": k, "populacao_leitura": "todas as Q", "delta": e["delta"], "ic95": e["ic95"], "n_pares": e["n_pares"],
                            "media_BA1": e["media_BA1_runs_pareados"], "media_comparador": e["media_comparador_runs_pareados"],
                            "destino": e.get("destino"), "sensibilidade_delta": e["sensibilidade_vs_media_5_seeds_comparador"]["delta"],
                            "sensibilidade_ic95": e["sensibilidade_vs_media_5_seeds_comparador"]["ic95"], "destino_sensibilidade": e.get("destino_sensibilidade")})
        for k in PRIMARIO:
            e = out["contrastes"][nome]["so_Q1"][k]
            if e.get("n_pares"):
                tab.append({"contraste": nome, "metrica": k, "populacao_leitura": "so Q1", "delta": e["delta"], "ic95": e["ic95"], "n_pares": e["n_pares"],
                            "media_BA1": e["media_BA1_runs_pareados"], "media_comparador": e["media_comparador_runs_pareados"], "destino": None})
    out["tabela"] = tab
    out["endpoint_primario"] = {k: {kk: out["contrastes"]["BA1_menos_C0"][k][kk] for kk in ("delta", "ic95", "delta_margem", "destino", "n_pares",
                                                                                          "media_BA1_runs_pareados", "media_comparador_runs_pareados", "t_4_medias_cidade")}
                                for k in PRIMARIO if out["contrastes"]["BA1_menos_C0"][k].get("n_pares")}
    # V2-h: decomposicao exata do Delta MAE RSSI ALL (comparadores com predicao por no)
    dec = {}
    for nome, comp in CONTRASTES.items():
        if comp == "empirico": continue
        ob = out["contrastes"][nome]
        res = [abs((R[("ba1", c, s, q)]["mae_rssi_ALL"] - R[(comp, c, s, q)]["mae_rssi_ALL"])
                   - (R[("ba1", c, s, q)]["contrib_mae_rssi_ALL_VALID"] - R[(comp, c, s, q)]["contrib_mae_rssi_ALL_VALID"])
                   - (R[("ba1", c, s, q)]["contrib_mae_rssi_ALL_SENT"] - R[(comp, c, s, q)]["contrib_mae_rssi_ALL_SENT"]))
               for c in CIDADES for q in QS for s in SEEDS if ("ba1", c, s, q) in R and (comp, c, s, q) in R]
        tot = ob["mae_rssi_ALL"]["delta"]
        part = lambda k: {"delta": ob[k]["delta"], "ic95": ob[k]["ic95"]}
        dec[nome] = {"delta_mae_rssi_ALL": tot, "ic95_ALL": ob["mae_rssi_ALL"]["ic95"],
                     "contrib_VALID": part("contrib_mae_rssi_ALL_VALID"), "contrib_SENT": part("contrib_mae_rssi_ALL_SENT"),
                     "contrib_SENT_abaixo_110": part("contrib_mae_rssi_ALL_SENT_abaixo_110"), "contrib_SENT_acima_110": part("contrib_mae_rssi_ALL_SENT_acima_110"),
                     "resto": part("resto_mae_rssi_ALL"),
                     "residuo_identidade_na_media": float(tot - ob["contrib_mae_rssi_ALL_VALID"]["delta"] - ob["contrib_mae_rssi_ALL_SENT"]["delta"]
                                                          - ob["resto_mae_rssi_ALL"]["delta"]),
                     "residuo_identidade_max_abs_por_corrida": float(max(res)) if res else None, "n_pares": len(res),
                     "fracao_do_delta_ALL_vinda_de_SENT": float(ob["contrib_mae_rssi_ALL_SENT"]["delta"] / tot) if tot else None,
                     "definicao": "contrib_pop = (n_pop / n_ALL) * MAE RSSI pop; ALL = VALID u SENT (V2-b)"}
    out["decomposicao_delta_mae_rssi_ALL"] = dec
    out["parametros_e_tempos"] = resumo_custo(M)  # BA-l
    out["carimbo"] = carimbo(Path(__file__), {"BA_metricas_por_corrida.json": a.metricas, "common.py": FASE_B / "common.py",
                                              "C_inferencia_v2.py": FASE_C_V2 / "C_inferencia_v2.py"})  # BA-n
    dump(out, a.saida)
    for t in tab:
        print("%-19s %-22s %-11s D=%+.4f IC=[%+.4f, %+.4f] n=%d | %s" % (t["contraste"], t["metrica"], t["populacao_leitura"], t["delta"], t["ic95"][0], t["ic95"][1],
                                                                       t["n_pares"], t["destino"]))
    for b, d in dec.items():  # V2-h
        print("DECOMP %-14s D_ALL=%+.4f = VALID %+.4f + SENT %+.4f (abaixo %+.4f, acima %+.4f) + resto %+.2e | frac SENT %.4f | res max %.1e" % (
              b, d["delta_mae_rssi_ALL"], d["contrib_VALID"]["delta"], d["contrib_SENT"]["delta"], d["contrib_SENT_abaixo_110"]["delta"],
              d["contrib_SENT_acima_110"]["delta"], d["resto"]["delta"], d["fracao_do_delta_ALL_vinda_de_SENT"], d["residuo_identidade_max_abs_por_corrida"]))
    for b, v in out["parametros_e_tempos"].items():
        print("CUSTO %-4s params %s efetivos %s | treino mediana %.1f s soma %.2f h | inferencia mediana %.1f s | conf %s" % (
              b, v["n_params_treinaveis_valores"], v["n_params_com_gradiente_1o_lote_valores"], v["tempo_treino_s"]["mediana"], v["tempo_treino_s"]["soma_h"],
              v["tempo_inferencia_parede_s"]["mediana"], v.get("conferencia_nominal", {}).get("ok")))
    print("gravado", a.saida)


if __name__ == "__main__":
    main()
