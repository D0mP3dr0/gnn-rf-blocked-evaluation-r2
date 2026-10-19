"""BA_fisica.py — admissibilidade fisica das saidas e vies assinado de RSSI por faixa, para BA1 (ba1rus), GNN-RF (C0 = c0c1cf) e
MLP (mlpcf), 80 corridas cada (4 cidades x 4 Q x seeds 42-46). R4 IEEE Access 47637, baseline BA1. Protocolo:
_revisao_2026-10-14/PROTOCOLO_faseC_baselines_RASCUNHO_2026-10-14.md + "Decisao do dono (15/10/2026)". Arquivo NOVO no padrao dos
BA_*.py; nao altera BA_metricas.py / BA_inferencia.py nem seus JSON. Escreve SO em fase_BA1/avaliacao/metricas/ (BA_fisica.json).

Reuso LITERAL (import, nao copia) das definicoes da fase C v2 (fase_C/avaliacao/metricas_v2/C_metricas_v2.py, so leitura):
  pares_celula(k, y, ic) ...... pares fixos por celula (rng default_rng([20261018, ic]), ic na ordem CIDADES x QS, M_FIS = 1e7)
  taxas_mono(rssi, P) ......... monotonicidade; forma fisica = pares VALID do mesmo servidor (slot_ant[:,0]), razao d_longe/d_perto
                                >= 2 (slot_d[:,0]), viol = RSSI_longe - RSSI_perto - 1 > 0  (campo mono_fis_viol_frac)
  metricas_run(z, y, k, ctx, P) campos frac_viol_srv_VALID / n_viol_srv_VALID / deficit_srv_* (piso FSPL do servidor, float32,
                                viol = PL_pred < piso_srv em VALID) e pred_mono_fis_viol_frac (taxas_mono sobre pred[:,3])
  ctx ......................... montado com as MESMAS linhas de C_metricas_v2.py:main (ctx_celula abaixo, copia literal);
                                piso_srv = fspl_db(slot_d[:,0] float32, slot_f[:,0] float32) (fase_B/common.py:fspl_db).
Bootstrap (a3): matriz_cadeias / bootstrap / t_cidades importados de BA_inferencia.py (B = 20000, rng default_rng(20261018),
IC_ e IS_ gerados na mesma ordem, cadeia = cidade x seed com os Q juntos). So descritivo; sem destino.

Definicoes novas desta etapa (declaradas; mesma fonte de geometria do ctx):
  d_srv = slot_d[:,0] do cache da fase B (a mesma coluna do piso_srv e do near1km de C_metricas_v2.py), em float64;
  id do servidor = slot_ant[:,0] (o mesmo de pares_celula).
  (a1-alvo) alvo abaixo do piso: (y[:,0] < piso_srv) & VALID, fracao sobre n_VALID (mesma comparacao float32 de metricas_run);
            n_viol_em_no_alvo_viola = violacoes da predicao em nos onde o alvo tambem viola.
  (a2-alvo) taxas_mono(y[:,3] float64, P) (= alvo_mono_fis_viol_frac de C_metricas_v2.py:main).
  (B) erro e3 = pred[:,3] - alvo[:,3] (float64) em VALID; por banda: n, vies = mean(e3), MAE = mean(|e3|),
      frac_pred_abaixo_alvo = mean(pred RSSI < alvo RSSI). Bandas de RSSI alvo (dBm): >= -70; [-85, -70); [-100, -85); < -100.
      Bandas de d_srv (m): < 250; [250, 500); [500, 1000); [1000, 2000); [2000, 4000); [4000, 8000); >= 8000.
      Agregado: media das corridas com n > 0 na banda (n_runs declarado), dp entre as corridas (ddof 1), dp entre seeds
      (dp ddof 1 das 5 medias por seed; cada uma = media das corridas daquela seed com n > 0), e o agregado ponderado por no
      (soma e3 / soma n) ao lado.
  (B-incl) inclinacao MQO (dB/decada) de PL contra log10(d_srv) nos nos VALID com d_srv < 2000 m, por corrida e no alvo, so em
      celulas com >= 50 nos (mesma medida de _pareceres_2026-10-19/_contra_fisico/fm_BA1_perto.py: np.linalg.lstsq com [X, 1]).
Conferencias gravadas: C0 a1 (media 80) == C_resumo_v2.json bracos.*.frac_viol_srv_VALID.media_C0_runs_pareados; C0 a2 (media 40) ==
pred_mono_fis_viol_frac.media_C0_runs_pareados; por corrida, os campos a1/a2 == BA_metricas_por_corrida.json (240 runs, igualdade
exata); bandas < 250 / 250-500 / 500-1000 m (n e soma e3) == fm_BA1_fisica.json; inclinacoes == fm_BA1_perto.json (provisorios,
so consistencia). Custodia: sha do npz == saida_npz_sha256 do sidecar; idx_global/target do C0 == cache; ba1/mlp == C0.
Uso: python -I -B BA_fisica.py [--celulas c_Q,...] [--saida X] [--permitir-incompleto]   (CPU)
Numeros PROVISORIOS ate contra-auditoria."""
import sys, json, argparse
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True  # nada de __pycache__ em fase_C (so leitura)
AQUI = Path(__file__).resolve().parent
REV = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14")
FASE_B = REV / "fase_B"
FASE_C_V2 = REV / "fase_C" / "avaliacao" / "metricas_v2"
FM = REV / "_pareceres_2026-10-19" / "_contra_fisico"
for p in (FASE_B, FASE_C_V2, AQUI):
    sys.path.insert(0, str(p))
from common import CIDADES, QS, SEEDS, cache, fspl_db, pops, carimbo, sha256_file, dump  # noqa: E402
import C_metricas_v2 as CM2  # noqa: E402  (definicoes da fase C v2, import literal)
from BA_inferencia import matriz_cadeias, bootstrap, t_cidades, B_BOOT, SEED_BOOT  # noqa: E402
from BA_metricas import npz_braco, BRACOS_C  # noqa: E402  (mesmos caminhos de npz dos BA_*)

BRACOS = ["gnn", "ba1", "mlp"]
ROTULO = {"gnn": "GNN-RF (C0 = c0c1cf)", "ba1": "BA1 (ba1rus, RadioUNet-style in-house adaptation)", "mlp": "MLP sem grafo (mlpcf)"}
CONTRASTES = {"BA1_menos_C0": "ba1", "MLP_menos_C0": "mlp"}
BANDAS_RSSI = [("ge_-70", -70.0, np.inf), ("-85_a_-70", -85.0, -70.0), ("-100_a_-85", -100.0, -85.0), ("lt_-100", -np.inf, -100.0)]
BANDAS_D = [("lt_250m", 0.0, 250.0), ("250_500m", 250.0, 500.0), ("500_1000m", 500.0, 1000.0), ("1_2km", 1000.0, 2000.0),
            ("2_4km", 2000.0, 4000.0), ("4_8km", 4000.0, 8000.0), ("ge_8km", 8000.0, np.inf)]
LIM_PERTO = 2000.0
N_MIN_INCL = 50
REF_C0_A1 = 1.3946e-4   # pedido (F-AD40 da folha); conferido contra o valor exato do C_resumo_v2.json
REF_C0_A2 = 0.0446


def ctx_celula(k, y):
    """Copia literal das linhas de C_metricas_v2.py:main que montam ctx (nenhuma mudanca)."""
    Pc = pops(y); v = Pc["VALID"]
    d0 = k["slot_d"][:, 0].astype(np.float32); f0 = k["slot_f"][:, 0].astype(np.float32)
    assert np.isfinite(d0[v]).all(), "VALID sem servidor"
    ax = k["ant_x"].astype(np.float64); a0 = k["slot_ant"][:, 0]
    ctx = {"piso_srv": fspl_db(d0, f0), "piso_atual": fspl_db(k["dist_near_test"], 1800.0),
           "PG": np.where(a0 >= 0, ax[np.maximum(a0, 0), 1] + ax[np.maximum(a0, 0), 3], np.nan),
           "lo200": v & (y[:, 0] <= CM2.TETO_PL), "near1km": v & (k["slot_d"][:, 0].astype(np.float64) < CM2.LIM_1KM)}
    return ctx, Pc


def bandas(e3, pr, tr, key, defs, v):
    o = {}
    for nm, lo, hi in defs:
        m = v & (key >= lo) & (key < hi); n = int(m.sum())
        o[nm] = {"n": n, "vies_rssi": float(e3[m].mean()) if n else None, "mae_rssi": float(np.abs(e3[m]).mean()) if n else None,
                 "frac_pred_abaixo_alvo": float((pr[m] < tr[m]).mean()) if n else None, "soma_e3": float(e3[m].sum())}
    return o


def inclinacao(X, yv):
    A = np.vstack([X, np.ones_like(X)]).T
    return float(np.linalg.lstsq(A, yv, rcond=None)[0][0])


def st_runs(L, campo):
    """L: lista de dicts com seed e campo. Media/dp das corridas com valor nao nulo e dp entre as 5 medias por seed."""
    x = [(r["seed"], r[campo]) for r in L if r.get(campo) is not None]
    if not x:
        return {"n_runs": 0, "media": None}
    a = np.asarray([vv for _, vv in x], float)
    ms = [float(np.mean([vv for s, vv in x if s == ss])) for ss in SEEDS if any(s == ss for s, _ in x)]
    return {"n_runs": int(a.size), "media": float(a.mean()), "dp_runs": float(a.std(ddof=1)) if a.size > 1 else None,
            "min": float(a.min()), "max": float(a.max()), "media_por_seed": dict(zip([str(s) for s in SEEDS if any(t == s for t, _ in x)], ms)),
            "dp_entre_seeds": float(np.std(ms, ddof=1)) if len(ms) > 1 else None}


def contraste(R, braco, campo, IC_, IS_):
    D = {}
    for c in CIDADES:
        for q in QS:
            for s in SEEDS:
                rb, r0 = R.get((braco, c, s, q)), R.get(("gnn", c, s, q))
                if rb is None or r0 is None or rb.get(campo) is None or r0.get(campo) is None: continue
                D[(c, s, q)] = rb[campo] - r0[campo]
    if not D:
        return {"n_pares": 0}
    CMx = matriz_cadeias(D); bs = bootstrap(CMx, IC_, IS_)
    return {"n_pares": len(D), "n_celulas": len({(c, q) for c, s, q in D}), "n_cadeias": int(np.isfinite(CMx).sum()),
            "media_braco_runs_pareados": float(np.mean([R[(braco, c, s, q)][campo] for c, s, q in D])),
            "media_C0_runs_pareados": float(np.mean([R[("gnn", c, s, q)][campo] for c, s, q in D])),
            "delta": bs["estimativa"], "ic95": bs["ic95"], "ep_bootstrap": bs["ep_bootstrap"], "B_validas": bs["B_validas"],
            "ic_exclui_zero": bool(bs["ic95"][0] > 0 or bs["ic95"][1] < 0),
            "delta_por_cidade": {c: float(np.nanmean(CMx[i])) if np.isfinite(CMx[i]).any() else None for i, c in enumerate(CIDADES)},
            "delta_por_celula": {f"{c}_{q}": float(np.mean([D[(c, s, q)] for s in SEEDS if (c, s, q) in D])) for c in CIDADES for q in QS
                                 if any((c, s, q) in D for s in SEEDS)},
            "t_4_medias_cidade": t_cidades(D), "papel": "descritivo (sem margem nem destino)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celulas", default="", help="subconjunto c_Q separado por virgula (teste)")
    ap.add_argument("--saida", default=str(AQUI / "BA_fisica.json"))
    ap.add_argument("--permitir-incompleto", action="store_true")
    a = ap.parse_args()
    sel = set(a.celulas.split(",")) if a.celulas else None
    runs, celulas, npz_sha, sidecar_sha = [], {}, {}, {}
    for ic, (c, q) in enumerate([(c, q) for c in CIDADES for q in QS]):
        if sel and f"{c}_{q}" not in sel: continue
        k = cache(c, q); idx = k["idx"]; y = k["y_test"]; yy = y.astype(np.float64)
        ctx, Pc = ctx_celula(k, y); v = Pc["VALID"]
        assert np.all(y[~v, 3] == CM2.RSSI_SENT), "RSSI alvo do SENT != -110"
        P = CM2.pares_celula(k, y, ic)
        dsrv = k["slot_d"][:, 0].astype(np.float64)                     # mesma coluna do piso_srv / near1km
        alvo_viol = (y[:, 0] < ctx["piso_srv"]) & v                       # mesma comparacao float32 de metricas_run
        perto = v & (dsrv < LIM_PERTO); n_perto = int(perto.sum())
        cel = {"n": int(len(y)), "n_VALID": int(v.sum()), "n_servidores_VALID": P["n_servidores_VALID"], "n_pares_fis": P["n_fis"],
               "alvo_n_viol_srv_VALID": int(alvo_viol.sum()), "alvo_frac_viol_srv_VALID": float(alvo_viol.sum() / v.sum()),
               "alvo_folga_min_PL_menos_piso_srv_VALID_dB": float((y[v, 0] - ctx["piso_srv"][v]).min()),
               "n_VALID_d_srv_lt_2km": n_perto,
               "n_VALID_bandas_rssi": {nm: int((v & (yy[:, 3] >= lo) & (yy[:, 3] < hi)).sum()) for nm, lo, hi in BANDAS_RSSI},
               "n_VALID_bandas_d_srv": {nm: int((v & (dsrv >= lo) & (dsrv < hi)).sum()) for nm, lo, hi in BANDAS_D}}
        cel.update({f"alvo_{kk}": vv for kk, vv in CM2.taxas_mono(yy[:, 3], P).items()})
        if n_perto >= N_MIN_INCL:
            X = np.log10(dsrv[perto]); cel["alvo_inclinacao_PL_dB_dec_lt2km"] = inclinacao(X, yy[perto, 0])
        else:
            X = None; cel["alvo_inclinacao_PL_dB_dec_lt2km"] = None
        celulas[f"{c}_{q}"] = cel
        for s in SEEDS:
            ref = None
            for b in BRACOS:
                f = npz_braco(b, c, q, s)
                if not f.is_file():
                    if a.permitir_incompleto and b != "gnn": continue
                    raise FileNotFoundError(f)
                z = np.load(f); npz_sha[f.name] = sha256_file(f)
                sj = f.with_suffix(".json"); jj = json.load(open(sj)); sidecar_sha[sj.name] = sha256_file(sj)
                assert jj["saida_npz_sha256"] == npz_sha[f.name], f"sha npz != sidecar {sj}"
                if b == "gnn":
                    assert np.array_equal(z["idx_global"], idx) and np.array_equal(z["target"], y), f"C0 != cache {f}"
                    ref = z
                else:
                    assert np.array_equal(z["idx_global"], ref["idx_global"]) and np.array_equal(z["target"], ref["target"]), f"CUSTODIA {f}"
                    assert np.array_equal(z["sentinela"], ref["sentinela"]), f"CUSTODIA sentinela {f}"
                    assert jj.get("run_label") == f"{BRACOS_C[b]}_{c}_s{s}_{q}_g10b2", f"run_label {sj}"
                mr = CM2.metricas_run(z, y, k, ctx, P)                     # definicoes literais da fase C v2
                p = z["pred"].astype(np.float64); e3 = p[:, 3] - yy[:, 3]
                viol = (z["pred"][:, 0] < ctx["piso_srv"]) & v
                assert int(viol.sum()) == mr["n_viol_srv_VALID"]
                r = {"braco": b, "cidade": c, "Q": q, "seed": s,
                     "n_VALID": mr["n_VALID"], "n_viol_srv_VALID": mr["n_viol_srv_VALID"], "frac_viol_srv_VALID": mr["frac_viol_srv_VALID"],
                     "deficit_srv_mediana": mr["deficit_srv_mediana"], "deficit_srv_p95": mr["deficit_srv_p95"], "deficit_srv_max": mr["deficit_srv_max"],
                     "n_viol_srv_em_no_alvo_viola": int((viol & alvo_viol).sum()),
                     "pred_mono_fis_viol_frac": mr["pred_mono_fis_viol_frac"], "pred_mono_fis_viol_frac_estrito": mr["pred_mono_fis_viol_frac_estrito"],
                     "n_pares_fis": P["n_fis"],
                     "bandas_rssi_alvo": bandas(e3, p[:, 3], yy[:, 3], yy[:, 3], BANDAS_RSSI, v),
                     "bandas_d_srv": bandas(e3, p[:, 3], yy[:, 3], dsrv, BANDAS_D, v),
                     "inclinacao_PL_pred_dB_dec_lt2km": inclinacao(X, p[perto, 0]) if X is not None else None}
                runs.append(r)
        print(c, q, "| alvo piso %.3e mono_fis %s | incl alvo %s | runs %d" % (
            cel["alvo_frac_viol_srv_VALID"], cel["alvo_mono_fis_viol_frac"], cel["alvo_inclinacao_PL_dB_dec_lt2km"],
            sum(1 for r in runs if r["cidade"] == c and r["Q"] == q)), flush=True)
    cont = {b: sum(1 for r in runs if r["braco"] == b) for b in BRACOS}
    completo = all(v_ == 80 for v_ in cont.values()) and len(celulas) == 16
    if not a.permitir_incompleto:
        assert completo, f"incompleto: {cont}, celulas {len(celulas)}"
    R = {(r["braco"], r["cidade"], r["seed"], r["Q"]): r for r in runs}

    # ---------------- agregados (A)
    agg = {"a1_frac_viol_srv_VALID": {}, "a2_mono_fis_viol_frac": {}}
    for b in BRACOS:
        L = [r for r in runs if r["braco"] == b]
        x = st_runs(L, "frac_viol_srv_VALID")
        x.update(n_viol_total=int(sum(r["n_viol_srv_VALID"] for r in L)), n_runs_com_viol=int(sum(r["n_viol_srv_VALID"] > 0 for r in L)),
                 celulas_com_viol=sorted({f"{r['cidade']}_{r['Q']}" for r in L if r["n_viol_srv_VALID"] > 0}),
                 n_viol_em_no_alvo_viola_total=int(sum(r["n_viol_srv_em_no_alvo_viola"] for r in L)),
                 deficit_srv_max=max([r["deficit_srv_max"] for r in L if r["deficit_srv_max"] is not None], default=None))
        agg["a1_frac_viol_srv_VALID"][b] = x
        y2 = st_runs(L, "pred_mono_fis_viol_frac")
        y2["celulas"] = sorted({f"{r['cidade']}_{r['Q']}" for r in L if r["pred_mono_fis_viol_frac"] is not None})
        y2["n_celulas"] = len(y2["celulas"])
        y2["media_por_celula"] = {cc: float(np.mean([r["pred_mono_fis_viol_frac"] for r in L if f"{r['cidade']}_{r['Q']}" == cc])) for cc in y2["celulas"]}
        agg["a2_mono_fis_viol_frac"][b] = y2
    cv = list(celulas.values())
    agg["a1_frac_viol_srv_VALID"]["alvo"] = {
        "media_16_celulas": float(np.mean([x["alvo_frac_viol_srv_VALID"] for x in cv])),
        "nota": "o alvo e o mesmo nas 5 seeds; a media das 80 corridas e igual a media das 16 celulas",
        "n_viol_total_16_celulas": int(sum(x["alvo_n_viol_srv_VALID"] for x in cv)),
        "celulas_com_viol": sorted(kk for kk, x in celulas.items() if x["alvo_n_viol_srv_VALID"] > 0),
        "por_celula": {kk: x["alvo_frac_viol_srv_VALID"] for kk, x in celulas.items()},
        "folga_min_PL_menos_piso_dB": float(min(x["alvo_folga_min_PL_menos_piso_srv_VALID_dB"] for x in cv))}
    am = {kk: x["alvo_mono_fis_viol_frac"] for kk, x in celulas.items() if x["alvo_mono_fis_viol_frac"] is not None}
    agg["a2_mono_fis_viol_frac"]["alvo"] = {"n_celulas": len(am), "celulas": sorted(am), "media_celulas": float(np.mean(list(am.values()))) if am else None,
                                            "max_celulas": float(max(am.values())) if am else None, "por_celula": am}
    # ---------------- (a3) delta pareado e bootstrap (mesmos IC_/IS_ do BA_inferencia.py)
    rng = np.random.default_rng(SEED_BOOT)
    IC_ = rng.integers(0, len(CIDADES), (B_BOOT, len(CIDADES)))
    IS_ = rng.integers(0, len(SEEDS), (B_BOOT, len(CIDADES), len(SEEDS)))
    a3 = {}
    for nome, b in CONTRASTES.items():
        a3[nome] = {campo: contraste(R, b, campo, IC_, IS_) for campo in ("frac_viol_srv_VALID", "pred_mono_fis_viol_frac")}
        if not a.permitir_incompleto:
            assert a3[nome]["frac_viol_srv_VALID"]["n_pares"] == 80 and a3[nome]["pred_mono_fis_viol_frac"]["n_pares"] == 40, f"a3 n_pares {nome}"
    agg["a3_delta_pareado"] = a3
    # ---------------- (B) vies por banda
    for chave, defs in (("bandas_rssi_alvo", BANDAS_RSSI), ("bandas_d_srv", BANDAS_D)):
        ob = {}
        for b in BRACOS:
            L = [r for r in runs if r["braco"] == b]; ob[b] = {}
            for nm, lo, hi in defs:
                LL = [{"seed": r["seed"], **r[chave][nm]} for r in L if r[chave][nm]["n"] > 0]
                nt = int(sum(x["n"] for x in LL))
                ob[b][nm] = {"limites": [lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None],
                             "vies_rssi": st_runs(LL, "vies_rssi"), "mae_rssi": st_runs(LL, "mae_rssi"),
                             "frac_pred_abaixo_alvo": st_runs(LL, "frac_pred_abaixo_alvo"),
                             "n_nos_total": nt, "n_nos_por_corrida_mediana": float(np.median([x["n"] for x in LL])) if LL else None,
                             "celulas": sorted({f"{r['cidade']}_{r['Q']}" for r in L if r[chave][nm]["n"] > 0}),
                             "vies_rssi_ponderado_por_no": float(sum(x["soma_e3"] for x in LL) / nt) if nt else None}
        agg[f"vies_{chave}"] = ob
    # ---------------- inclinacao < 2 km
    inc = {}
    for kk, x in celulas.items():
        if x["alvo_inclinacao_PL_dB_dec_lt2km"] is None: continue
        c, q = kk.split("_")
        e = {"n_VALID_lt2km": x["n_VALID_d_srv_lt_2km"], "alvo": x["alvo_inclinacao_PL_dB_dec_lt2km"]}
        for b in BRACOS:
            vv = [R[(b, c, s, q)]["inclinacao_PL_pred_dB_dec_lt2km"] for s in SEEDS if (b, c, s, q) in R]
            e[b] = {"media_5_seeds": float(np.mean(vv)), "dp_seeds": float(np.std(vv, ddof=1)) if len(vv) > 1 else None, "por_seed": vv}
        inc[kk] = e
    agg["inclinacao_PL_log10d_lt2km"] = {"celulas": inc, "media_celulas": {
        **{"alvo": float(np.mean([e["alvo"] for e in inc.values()])) if inc else None},
        **{b: float(np.mean([e[b]["media_5_seeds"] for e in inc.values()])) if inc else None for b in BRACOS}}}

    # ---------------- conferencias
    conf = {}
    if completo:
        Cr = json.load(open(FASE_C_V2 / "C_resumo_v2.json"))
        ref1 = {br: Cr["bracos"][br]["frac_viol_srv_VALID"]["media_C0_runs_pareados"] for br in Cr["bracos"]}
        ref2 = {br: Cr["bracos"][br]["pred_mono_fis_viol_frac"]["media_C0_runs_pareados"] for br in Cr["bracos"]}
        m1 = agg["a1_frac_viol_srv_VALID"]["gnn"]["media"]; m2 = agg["a2_mono_fis_viol_frac"]["gnn"]["media"]
        conf["a1_C0_vs_C_resumo_v2"] = {"calculado": m1, "C_resumo_v2": ref1, "abs_dif_max": float(max(abs(m1 - v_) for v_ in ref1.values())),
                                        "pedido_F_AD40": REF_C0_A1, "arredonda_ao_pedido": round(m1, 8) == round(REF_C0_A1, 8) or abs(m1 - REF_C0_A1) < 5e-9,
                                        "passa": all(abs(m1 - v_) <= 1e-15 for v_ in ref1.values()) and abs(m1 - REF_C0_A1) < 5e-9}
        conf["a2_C0_vs_C_resumo_v2"] = {"calculado": m2, "C_resumo_v2": ref2, "abs_dif_max": float(max(abs(m2 - v_) for v_ in ref2.values())),
                                        "n_runs": agg["a2_mono_fis_viol_frac"]["gnn"]["n_runs"], "pedido": REF_C0_A2,
                                        "passa": all(abs(m2 - v_) <= 1e-15 for v_ in ref2.values()) and abs(m2 - REF_C0_A2) < 5e-5
                                        and agg["a2_mono_fis_viol_frac"]["gnn"]["n_runs"] == 40}
        M = json.load(open(AQUI / "BA_metricas_por_corrida.json"))
        RM = {(r["braco"], r["cidade"], r["seed"], r["Q"]): r for r in M["runs"]}
        dif = [kk for kk, r in R.items() for campo in ("n_viol_srv_VALID", "frac_viol_srv_VALID", "deficit_srv_max", "pred_mono_fis_viol_frac")
               if RM[kk][campo] != r[campo]]
        conf["por_corrida_vs_BA_metricas_por_corrida"] = {"n_runs_comparadas": len(R), "campos": ["n_viol_srv_VALID", "frac_viol_srv_VALID",
                                                          "deficit_srv_max", "pred_mono_fis_viol_frac"], "n_divergencias": len(dif),
                                                          "divergencias": [list(x) for x in dif[:20]], "passa": not dif}
        F = json.load(open(FM / "fm_BA1_fisica.json"))
        FR = {(r["braco"], r["cidade"], r["seed"], r["Q"]): r for r in F["runs"]}
        dn, de = 0, 0.0
        for kk, r in R.items():
            for i, nm in enumerate(("lt_250m", "250_500m", "500_1000m")):
                fb = FR[kk]["bandas_dist"][i]; mb = r["bandas_d_srv"][nm]
                dn += int(fb["n"] != mb["n"]); de = max(de, abs(fb["soma_e3"] - mb["soma_e3"]))
        conf["bandas_d_srv_vs_fm_BA1_fisica"] = {"bandas": ["lt_250m", "250_500m", "500_1000m"], "n_runs": len(R), "n_divergencias_n": dn,
                                                 "max_abs_dif_soma_e3_dB": de, "passa": dn == 0 and de < 1e-6,
                                                 "nota": "fm_BA1_fisica.json e provisorio (contra-fisico); conferencia so de consistencia de fonte"}
        Pf = json.load(open(FM / "fm_BA1_perto.json"))
        PR = {(r["braco"], r["cidade"], r["seed"], r["Q"]): r.get("inclinacao_PL_pred_dB_dec") for r in Pf["runs"]}
        di = [abs(PR[kk] - r["inclinacao_PL_pred_dB_dec_lt2km"]) for kk, r in R.items() if r["inclinacao_PL_pred_dB_dec_lt2km"] is not None]
        da = [abs(Pf["celulas"][kk]["inclinacao_PL_alvo_dB_dec"] - e["alvo"]) for kk, e in inc.items()]
        conf["inclinacao_vs_fm_BA1_perto"] = {"n_runs": len(di), "n_celulas": len(da), "max_abs_dif_pred": float(max(di)) if di else None,
                                              "max_abs_dif_alvo": float(max(da)) if da else None,
                                              "passa": bool(di) and max(di) < 1e-9 and max(da) < 1e-9,
                                              "nota": "fm_BA1_perto.json e provisorio; conferencia so de consistencia da medida"}
        conf["passa_tudo"] = all(v_["passa"] for v_ in conf.values() if isinstance(v_, dict))

    ins = {"common.py": FASE_B / "common.py", "C_metricas_v2.py": FASE_C_V2 / "C_metricas_v2.py", "C_resumo_v2.json": FASE_C_V2 / "C_resumo_v2.json",
           "BA_inferencia.py": AQUI / "BA_inferencia.py", "BA_metricas.py": AQUI / "BA_metricas.py",
           "BA_metricas_por_corrida.json": AQUI / "BA_metricas_por_corrida.json",
           "fm_BA1_fisica.json": FM / "fm_BA1_fisica.json", "fm_BA1_perto.json": FM / "fm_BA1_perto.json", "fm_BA1_perto.py": FM / "fm_BA1_perto.py"}
    for c in CIDADES:
        for q in QS:
            ins[f"_cache/{c}_{q}.npz"] = FASE_B / "_cache" / f"{c}_{q}.npz"
    out = {"etapa": "BA_fisica", "versao": "1 (19/10/2026)",
           "protocolo": "_revisao_2026-10-14/PROTOCOLO_faseC_baselines_RASCUNHO_2026-10-14.md + Decisao do dono (15/10/2026)",
           "bracos": {b: ROTULO[b] for b in BRACOS}, "n_runs_por_braco": cont, "completo": completo,
           "definicoes": {
               "VALID": "alvo PL < 299 dB (fase_B/common.py:pops)",
               "a1": "frac_viol_srv_VALID de C_metricas_v2.metricas_run: (pred PL float32 < piso_srv) & VALID, sobre n_VALID; "
                     "piso_srv = fspl_db(slot_d[:,0], slot_f[:,0]) (common.py; d_km com piso de 1 m; 32,44)",
               "a1_alvo": "(alvo PL < piso_srv) & VALID, sobre n_VALID (mesma comparacao)",
               "a2": "pred_mono_fis_viol_frac de C_metricas_v2.taxas_mono sobre os pares de C_metricas_v2.pares_celula (mesmo servidor, "
                     "ambos VALID, razao >= 2; viol = RSSI_longe - RSSI_perto - 1 > 0); so celulas com pares elegiveis",
               "a2_alvo": "taxas_mono(alvo RSSI float64, mesmos pares)",
               "a3": "Delta = braco - C0 pareado por (cidade, seed, Q); bootstrap cidade -> cadeia, B = 20000, default_rng(20261018), IC95 percentil "
                     "(funcoes de BA_inferencia.py); descritivo",
               "d_srv": "slot_d[:,0] do cache da fase B (mesma coluna do piso_srv); servidor = slot_ant[:,0]",
               "vies": "mean(pred RSSI - alvo RSSI) em VALID na banda; MAE = mean(|.|); frac_pred_abaixo_alvo = mean(pred RSSI < alvo RSSI)",
               "agregado_bandas": "media das corridas com n > 0 na banda; dp_runs ddof 1; dp_entre_seeds = dp ddof 1 das medias por seed; "
                                  "vies_rssi_ponderado_por_no = soma e3 / soma n",
               "inclinacao": "MQO np.linalg.lstsq de PL contra [log10(d_srv), 1] em VALID & d_srv < 2000 m; celulas com >= 50 nos"},
           "constantes": {"bandas_rssi_alvo_dBm": [[nm, lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None] for nm, lo, hi in BANDAS_RSSI],
                          "bandas_d_srv_m": [[nm, lo, hi if np.isfinite(hi) else None] for nm, lo, hi in BANDAS_D],
                          "intervalo": "[lo, hi)", "LIM_PERTO_m": LIM_PERTO, "N_MIN_INCL": N_MIN_INCL, "M_FIS": CM2.M_FIS, "SEED_PARES": CM2.SEED_PARES,
                          "B": B_BOOT, "SEED_BOOT": SEED_BOOT},
           "agregado": agg, "conferencia": conf, "celulas": celulas, "runs": runs,
           "carimbo": carimbo(Path(__file__), ins) | {"npz_sha256": npz_sha, "sidecar_sha256": sidecar_sha}}
    dump(out, a.saida)
    for b in BRACOS:
        print("A1 %-4s media80 %.6e | A2 %-4s media40 %.6f" % (b, agg["a1_frac_viol_srv_VALID"][b]["media"], b, agg["a2_mono_fis_viol_frac"][b]["media"]))
    print("A1 alvo %.6e | A2 alvo %s" % (agg["a1_frac_viol_srv_VALID"]["alvo"]["media_16_celulas"], agg["a2_mono_fis_viol_frac"]["alvo"]["media_celulas"]))
    for nome in CONTRASTES:
        for campo in ("frac_viol_srv_VALID", "pred_mono_fis_viol_frac"):
            e = a3[nome][campo]
            if e.get("n_pares"):
                print("A3 %-13s %-24s D=%+.6e IC=[%+.6e, %+.6e] n=%d" % (nome, campo, e["delta"], e["ic95"][0], e["ic95"][1], e["n_pares"]))
    for chave in ("bandas_rssi_alvo", "bandas_d_srv"):
        for nm in agg[f"vies_{chave}"]["gnn"]:
            print("VIES %-17s %-11s " % (chave, nm) + " | ".join(
                "%s %s" % (b, ("%+.3f (n_runs %d)" % (agg[f"vies_{chave}"][b][nm]["vies_rssi"]["media"], agg[f"vies_{chave}"][b][nm]["vies_rssi"]["n_runs"]))
                           if agg[f"vies_{chave}"][b][nm]["vies_rssi"]["n_runs"] else "-") for b in BRACOS))
    print("CONFERENCIA", {k_: (v_["passa"] if isinstance(v_, dict) else v_) for k_, v_ in conf.items()})
    print("gravado", a.saida)


if __name__ == "__main__":
    main()
