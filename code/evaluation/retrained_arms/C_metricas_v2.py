"""C_metricas_v2.py — COPIA de ../metricas/C_metricas.py (v1, sha256 9977e1b7...) com ACRESCIMOS MINIMOS, por emenda decidida
pelo dono em 18/10/2026 (ata AGENTES/cientista_dados/ATAS/ata_2026-10-18_ieee-access-47637-faseC.md, Bloco 4, "Decisao do dono").
Nenhuma metrica da v1 muda de definicao; o diff completo contra a v1 esta em C_metricas_v2.diff. Acrescimos (V2-a..V2-e):
  V2-a  por corrida, populacao SENT (fase_B/common.py:pops; SENT = alvo PL >= 299 dB, RSSI alvo = -110 dBm):
          vies_rssi_SENT = mean(pred RSSI - alvo RSSI) em SENT;
          mae_rssi_SENT  ja existia na v1 (laco B1 sobre pops) e e mantido como esta;
          mae_rssi_SENT_parte_abaixo_110 = sum_{SENT, pred < -110} |e| / n_SENT ;
          mae_rssi_SENT_parte_acima_110  = sum_{SENT, pred >= -110} |e| / n_SENT  (as duas somam mae_rssi_SENT; pred = -110 da e = 0);
          frac_pred_rssi_lt_sent_SENT ja existia na v1 e e mantida.
  V2-b  decomposicao exata do MAE RSSI ALL por corrida (ALL = VALID u SENT, disjuntos; pesos iguais nos bracos, mesmo alvo):
          contrib_mae_rssi_ALL_VALID = (n_VALID/n_ALL) * mae_rssi_VALID ; contrib_mae_rssi_ALL_SENT = (n_SENT/n_ALL) * mae_rssi_SENT ;
          contrib_mae_rssi_ALL_SENT_abaixo_110 / _acima_110 = (n_SENT/n_ALL) * parte ; resto_mae_rssi_ALL = ALL - VALID - SENT.
  V2-c  sidecar .json AUSENTE aborta (v1 gravava None), para os bracos E para o C0; sha256 do npz lido == saida_npz_sha256 do
          sidecar (assert), nos 240 npz; checkpoint_sha256 do sidecar do braco presente (assert). Sidecars do C0 vao em
          "sidecars_C0" (chave nova; "sidecars_bracos" fica como na v1).
  V2-d  saida padrao: metricas_v2/C_metricas_por_corrida_v2.json.
  V2-e  carimbo: insumo extra "C_metricas_v1.py" (sha da v1) para ligar a copia a origem.

--- cabecalho da v1 (inalterado) ---
C_metricas.py — fase C (R4 IEEE Access 47637): metricas por corrida de C0 (gnn = c0c1cf), C1 (c1noreg) e C2 (c2nomono)
no MESMO harness da fase B. Protocolo: _revisao_2026-10-14/PROTOCOLO_faseC_C1_C2_2026-10-14.md, secoes "Avaliacao" e
"Endpoint, unidade e criterios fixados antes de rodar" (aprovado pelo dono 14/10/2026; "va" da avaliacao 18/10/2026).

So LEITURA de fase_B (importada por sys.path; nunca editada), dos npz do C0 (predictions/c0)
e dos npz dos bracos (fase_C/e3_predicoes). Escreve SO em fase_C/avaliacao/metricas/.

Definicoes (copiadas, com a origem):
  populacoes ALL/VALID/COV85/SENT ............ fase_B/common.py:pops (VALID = alvo PL < 299 dB; COV85 = VALID & RSSI alvo >= -85)
  MAE RSSI / MAE PL por populacao ............ fase_B/B1_modelos.py:21-27 (float64; |pred - alvo| medio)
  MAE PL VALID sem alvo > 200 dB ............. fase_AB_v2/reagregacoes/R5_pl_teto200.py:11,18 (VALID & y0 <= 200)
  estrato VALID a < 1 km do servidor ......... fase_AB_v2/reagregacoes/R6_sens_1km.py:47-49 (d0 = slot_d[:,0] < 1000 m)
  vies (erro com sinal) = mean(pred - alvo) .. fm_scripts/fm_S1.py:85,103 (vies_e0_VALID / vies_e3_VALID)
  piso FSPL do servidor (B2) ................. fase_B/B2_fspl_servidor.py:15-31 (float32; viol = PL_pred < piso_srv em VALID)
  |Delta| entre saidas (B3) .................. fase_B/B3_consistencia_cobertura.py:20-35 (Delta = RSSI_pred - (P+G - PL_pred), VALID)
  canais 1 e 2 (B4) .......................... fase_B/B4_canais_aux.py:13-22 (MAE, vies, R2 em ALL e VALID)
  RSSI previsto abaixo do piso do sentinela .. alvo SENT tem RSSI = -110 dBm (cache y_test[:,3] em SENT); frac(pred RSSI < -110)
  monotonicidade, forma do codigo ............ fm_scripts/fm_S1.py:73-77: pares aleatorios de TODOS os nos, distancia a antena
                                               mais proxima (dist_near_test), dn_i != dn_j, i = mais perto; viol = RSSI_longe -
                                               RSSI_perto + 1 > 0 (queda minima de 1 dB; physics_loss.py:258 relu(rssi_j-rssi_i+1));
                                               M = min(400000, 4*n_VALID) (fm_S1.py:44).
  monotonicidade, forma fisica ............... forum-eng-ia_debate.py:35-38 (os dois nos VALID, mesmo servidor slot 0, srv >= 0,
                                               razao d_longe/max(d_perto,1e-6) >= 2) com o criterio C2b de fm_S1.py:49-52
                                               (viol = RSSI_longe - RSSI_perto - 1 > 0). Auxiliar: mesmos pares com o criterio do
                                               codigo (dr + 1 > 0) e estrito (dr > 0).
Amostragem dos pares: UMA VEZ por celula, a partir do cache (alvo/geometria), com rng fixo por celula
(numpy default_rng([20261018, indice_da_celula])); os MESMOS pares valem para alvo, C0, C1, C2 e todas as seeds.
Forma fisica: sorteio CONDICIONADO a "mesmo servidor e ambos VALID" (grupo g com prob. n_g^2 / sum n^2, depois i, j uniformes
no grupo) -> uniforme sobre os pares ordenados elegiveis, o mesmo estimando do sorteio sobre todos os nos do debate, mas com
~1/frac_mesmo_servidor mais pares uteis (desvio de eficiencia declarado no relatorio; nao muda a definicao).

Custodia: para cada (celula, seed), idx_global, target e sentinela do npz do braco == os do C0; e os do C0 == cache da fase B
(idx, y_test), como em B2_fspl_servidor.py:22-23. Sidecar .json do braco: run_label do braco e rng_seed == seed.

Uso:  python C_metricas.py [--bracos gnn,c1noreg,c2nomono] [--permitir-ausentes]
Saida: fase_C/avaliacao/metricas/C_metricas_por_corrida.json (carimbo common.carimbo + sha de cada npz lido).
Numeros PROVISORIOS ate contra-auditoria."""
import sys, json, argparse, datetime
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent
FASE_B = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_B")
sys.path.insert(0, str(FASE_B))
sys.dont_write_bytecode = True
from common import (CIDADES, QS, SEEDS, PL_MAX_VALID, LIM_COV, npz_path, cache, fspl_db, pops, carimbo, sha256_file,  # noqa: E402
                    dump)

FASE_C = AQUI.parent.parent
PRED_C = FASE_C / "e3_predicoes"
BRACOS_C = {"gnn": "c0c1cf", "c1noreg": "c1noreg", "c2nomono": "c2nomono"}
TETO_PL = 200.0          # R5_pl_teto200.py:8
LIM_1KM = 1000.0         # R6_sens_1km.py:44
RSSI_SENT = -110.0       # alvo RSSI dos nos SENT no cache (conferido por assert abaixo)
M_FIS = 10_000_000       # pares mesmo-servidor sorteados por celula (forma fisica)
SEED_PARES = 20261018


def npz_braco(braco, c, q, s):
    return npz_path(braco, c, q, s) if braco == "gnn" else PRED_C / f"{braco}_{c}_{q}_s{s}.npz"


def kappa(a, b):  # B3_consistencia_cobertura.py:11-16
    a = np.asarray(a, bool); b = np.asarray(b, bool); n = a.size
    if n == 0: return None
    po = float((a == b).mean()); pa = a.mean(); pb = b.mean()
    pe = float(pa * pb + (1 - pa) * (1 - pb))
    return None if pe >= 1.0 else (po - pe) / (1 - pe)


def pares_celula(k, y, ic):
    """Pares fixos por celula (alvo/geometria). Devolve dict com (perto, longe) das duas formas."""
    rng = np.random.default_rng([SEED_PARES, ic])
    v = y[:, 0] < PL_MAX_VALID; nv = int(v.sum()); N = len(y)
    # --- forma do codigo (fm_S1.py:73-77)
    M = min(400000, 4 * nv)
    dn = k["dist_near_test"].astype(np.float64)
    i = rng.integers(0, N, M); j = rng.integers(0, N, M); m = dn[i] != dn[j]; i, j = i[m], j[m]
    sw = dn[i] > dn[j]; cp = np.where(sw, j, i); cl = np.where(sw, i, j)
    # --- forma fisica (forum-eng-ia_debate.py:35-38 + fm_S1.py:46-52), sorteio condicionado ao mesmo servidor
    srv = k["slot_ant"][:, 0]; ds = k["slot_d"][:, 0].astype(np.float64)
    iv = np.where(v & (srv >= 0))[0]
    g, inv, cnt = np.unique(srv[iv], return_inverse=True, return_counts=True)
    ordem = np.argsort(inv, kind="stable"); ini = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pg = cnt.astype(np.float64) ** 2; pg /= pg.sum()
    gs = rng.choice(len(g), M_FIS, p=pg)
    a = iv[ordem[ini[gs] + rng.integers(0, cnt[gs])]]; b = iv[ordem[ini[gs] + rng.integers(0, cnt[gs])]]
    assert (srv[a] == srv[b]).all() and v[a].all() and v[b].all()
    r = np.maximum(ds[a], ds[b]) / np.maximum(np.minimum(ds[a], ds[b]), 1e-6)
    m2 = (r >= 2) & (ds[a] != ds[b]); a, b = a[m2], b[m2]
    sw = ds[a] > ds[b]; fp = np.where(sw, b, a); fl = np.where(sw, a, b)
    frac_mesmo_srv_ambos_valid = float((cnt.astype(np.float64) ** 2).sum() / float(N) ** 2)
    return {"cod": (cp, cl), "fis": (fp, fl), "n_cod": int(len(cp)), "n_fis": int(len(fp)), "M_cod": int(M), "M_fis": M_FIS,
            "frac_par_elegivel_fis_entre_mesmo_srv": float(m2.mean()),
            "frac_par_elegivel_fis_sobre_todos_os_pares_estimada": float(m2.mean()) * frac_mesmo_srv_ambos_valid,
            "n_servidores_VALID": int(len(g))}


def taxas_mono(rssi, P):
    cp, cl = P["cod"]; fp, fl = P["fis"]
    dro = rssi[cl] - rssi[cp]; dr = rssi[fl] - rssi[fp]
    o = {"mono_cod_viol_frac": float((dro + 1.0 > 0).mean()) if len(cp) else None}
    if len(fp):
        o.update(mono_fis_viol_frac=float((dr - 1.0 > 0).mean()), mono_fis_viol_frac_crit_codigo=float((dr + 1.0 > 0).mean()),
                 mono_fis_viol_frac_estrito=float((dr > 0).mean()))
    else:
        o.update(mono_fis_viol_frac=None, mono_fis_viol_frac_crit_codigo=None, mono_fis_viol_frac_estrito=None)
    return o


def metricas_run(z, y, k, ctx, P):
    t = z["target"].astype(np.float64); p = z["pred"].astype(np.float64)
    Pp = pops(z["target"]); v = Pp["VALID"]
    r = {}
    # B1 (B1_modelos.py:21-27)
    for pn, m in Pp.items():
        n = int(m.sum()); r[f"n_{pn}"] = n
        if n == 0:
            r[f"mae_rssi_{pn}"] = r[f"mae_pl_{pn}"] = None; continue
        r[f"mae_rssi_{pn}"] = float(np.abs(p[m, 3] - t[m, 3]).mean())
        r[f"mae_pl_{pn}"] = float(np.abs(p[m, 0] - t[m, 0]).mean())
    r["COV85_n_ge_1000"] = r["n_COV85"] >= 1000
    # R5: PL VALID sem alvo > 200 dB
    lo = ctx["lo200"]; r["n_VALID_ate200"] = int(lo.sum())
    r["mae_pl_VALID_ate200"] = float(np.abs(p[lo, 0] - t[lo, 0]).mean()) if lo.any() else None
    # vies (fm_S1.py:85,103) em VALID e VALID < 1 km (R6)
    e0 = p[:, 0] - t[:, 0]; e3 = p[:, 3] - t[:, 3]
    r["vies_pl_VALID"] = float(e0[v].mean()); r["vies_rssi_VALID"] = float(e3[v].mean()); r["vies_rssi_ALL"] = float(e3.mean())
    nr = ctx["near1km"]; r["n_VALID_lt1km"] = int(nr.sum())
    for nm, e in (("pl", e0), ("rssi", e3)):
        r[f"vies_{nm}_VALID_lt1km"] = float(e[nr].mean()) if nr.any() else None
        r[f"mae_{nm}_VALID_lt1km"] = float(np.abs(e[nr]).mean()) if nr.any() else None
    # RSSI previsto abaixo do piso do sentinela (-110 dBm)
    for pn in ("ALL", "VALID", "SENT"):
        m = Pp[pn]; r[f"frac_pred_rssi_lt_sent_{pn}"] = float((p[m, 3] < RSSI_SENT).mean()) if m.any() else None
    # V2-a: SENT em dB (vies com sinal e as duas partes do MAE SENT, abaixo/acima de -110 dBm)
    S = Pp["SENT"]; nS = int(S.sum()); nA = int(len(t))
    if nS:
        aS = np.abs(e3[S]); ab = p[S, 3] < RSSI_SENT
        r["vies_rssi_SENT"] = float(e3[S].mean())
        r["mae_rssi_SENT_parte_abaixo_110"] = float(aS[ab].sum() / nS)
        r["mae_rssi_SENT_parte_acima_110"] = float(aS[~ab].sum() / nS)
    else:
        r["vies_rssi_SENT"] = r["mae_rssi_SENT_parte_abaixo_110"] = r["mae_rssi_SENT_parte_acima_110"] = None
    # V2-b: decomposicao exata do MAE RSSI ALL (pesos n_pop / n_ALL)
    wS = nS / nA; wV = r["n_VALID"] / nA
    assert r["n_VALID"] + nS == nA
    r["peso_SENT_em_ALL"] = float(wS)
    r["contrib_mae_rssi_ALL_VALID"] = float(wV * r["mae_rssi_VALID"]) if r["n_VALID"] else 0.0
    r["contrib_mae_rssi_ALL_SENT"] = float(wS * r["mae_rssi_SENT"]) if nS else 0.0
    r["contrib_mae_rssi_ALL_SENT_abaixo_110"] = float(wS * r["mae_rssi_SENT_parte_abaixo_110"]) if nS else 0.0
    r["contrib_mae_rssi_ALL_SENT_acima_110"] = float(wS * r["mae_rssi_SENT_parte_acima_110"]) if nS else 0.0
    r["resto_mae_rssi_ALL"] = float(r["mae_rssi_ALL"] - r["contrib_mae_rssi_ALL_VALID"] - r["contrib_mae_rssi_ALL_SENT"])
    # B2 (B2_fspl_servidor.py:24-31), float32 como no original
    plp = z["pred"][:, 0]
    viol = (plp < ctx["piso_srv"]) & v; dfc = (ctx["piso_srv"] - plp)[viol]
    r.update(n_viol_srv_VALID=int(viol.sum()), frac_viol_srv_VALID=float(viol.sum() / v.sum()),
             deficit_srv_mediana=float(np.median(dfc)) if dfc.size else None, deficit_srv_p95=float(np.percentile(dfc, 95)) if dfc.size else None,
             deficit_srv_max=float(dfc.max()) if dfc.size else None,
             frac_viol_atual_VALID=float(((plp < ctx["piso_atual"]) & v).sum() / v.sum()), frac_viol_atual_ALL=float((plp < ctx["piso_atual"]).mean()))
    # B3 (B3_consistencia_cobertura.py:28-35)
    dl = (p[:, 3] - (ctx["PG"] - p[:, 0]))[v]; ad = np.abs(dl)
    r.update(abs_delta_mediana=float(np.median(ad)), abs_delta_p95=float(np.percentile(ad, 95)), abs_delta_mae=float(ad.mean()), delta_vies=float(dl.mean()))
    prob = p[:, 4] >= 0.5; rs = p[:, 3] >= LIM_COV
    for pn in ("ALL", "VALID"):
        m = Pp[pn]; r[f"concord_{pn}"] = float((prob[m] == rs[m]).mean()); r[f"kappa_{pn}"] = kappa(prob[m], rs[m])
    # B4 (B4_canais_aux.py:15-21)
    for pn in ("ALL", "VALID"):
        m = Pp[pn]
        for ch in (1, 2):
            e = p[m, ch] - t[m, ch]; sst = float(((t[m, ch] - t[m, ch].mean()) ** 2).sum())
            r[f"mae_c{ch}_{pn}"] = float(np.abs(e).mean()); r[f"vies_c{ch}_{pn}"] = float(e.mean())
            r[f"r2_c{ch}_{pn}"] = float(1 - (e ** 2).sum() / sst) if sst > 0 else None
    # monotonicidade na PREDICAO (mesmos pares do alvo)
    r.update({f"pred_{kk}": vv for kk, vv in taxas_mono(p[:, 3], P).items()})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bracos", default="gnn,c1noreg,c2nomono")
    ap.add_argument("--permitir-ausentes", action="store_true", help="npz de braco ausente vira entrada em 'ausentes' (teste parcial)")
    ap.add_argument("--celulas", default="", help="subconjunto c_Q separado por virgula (teste)")
    ap.add_argument("--saida", default=str(AQUI / "C_metricas_por_corrida_v2.json"))  # V2-d
    a = ap.parse_args()
    bracos = a.bracos.split(","); assert bracos[0] == "gnn" and set(bracos) <= set(BRACOS_C)
    sel = set(a.celulas.split(",")) if a.celulas else None
    runs, celulas, ausentes, npz_sha, sidecars = [], {}, [], {}, {}
    sidecars_c0 = {}  # V2-c
    for ic, (c, q) in enumerate([(c, q) for c in CIDADES for q in QS]):
        if sel and f"{c}_{q}" not in sel: continue
        k = cache(c, q); idx = k["idx"]; y = k["y_test"]; yy = y.astype(np.float64)
        Pc = pops(y); v = Pc["VALID"]
        assert np.all(y[~v, 3] == RSSI_SENT), "RSSI alvo do SENT != -110"
        d0 = k["slot_d"][:, 0].astype(np.float32); f0 = k["slot_f"][:, 0].astype(np.float32)
        assert np.isfinite(d0[v]).all(), "VALID sem servidor"
        ax = k["ant_x"].astype(np.float64); a0 = k["slot_ant"][:, 0]
        ctx = {"piso_srv": fspl_db(d0, f0), "piso_atual": fspl_db(k["dist_near_test"], 1800.0),
               "PG": np.where(a0 >= 0, ax[np.maximum(a0, 0), 1] + ax[np.maximum(a0, 0), 3], np.nan),
               "lo200": v & (y[:, 0] <= TETO_PL), "near1km": v & (k["slot_d"][:, 0].astype(np.float64) < LIM_1KM)}
        P = pares_celula(k, y, ic)
        cel = {"n": int(len(y)), "n_VALID": int(v.sum()), "n_COV85": int(Pc["COV85"].sum()), "n_VALID_ate200": int(ctx["lo200"].sum()),
               "n_VALID_lt1km": int(ctx["near1km"].sum()), "n_pares_cod": P["n_cod"], "M_cod": P["M_cod"], "n_pares_fis": P["n_fis"],
               "M_fis_mesmo_servidor": P["M_fis"], "frac_par_elegivel_fis_entre_mesmo_srv": P["frac_par_elegivel_fis_entre_mesmo_srv"],
               "frac_par_elegivel_fis_sobre_todos_os_pares_estimada": P["frac_par_elegivel_fis_sobre_todos_os_pares_estimada"],
               "n_servidores_VALID": P["n_servidores_VALID"],
               "alvo_frac_rssi_lt_sent_VALID": float((y[v, 3] < RSSI_SENT).mean())}
        cel.update({f"alvo_{kk}": vv for kk, vv in taxas_mono(yy[:, 3], P).items()})
        celulas[f"{c}_{q}"] = cel
        for s in SEEDS:
            ref = None
            for b in bracos:
                f = npz_braco(b, c, q, s)
                if not f.is_file():
                    if b != "gnn" and a.permitir_ausentes:
                        ausentes.append(f.name); continue
                    raise FileNotFoundError(f)
                z = np.load(f); npz_sha[f.name] = sha256_file(f)
                # V2-c: sidecar obrigatorio e sha do npz == saida_npz_sha256 gravado na inferencia (240 npz)
                sj = f.with_suffix(".json")
                if not sj.is_file():
                    raise FileNotFoundError(f"V2-c sidecar ausente (aborta): {sj}")
                jj = json.load(open(sj))
                assert jj["saida_npz_sha256"] == npz_sha[f.name], f"V2-c sha npz {npz_sha[f.name]} != sidecar {jj['saida_npz_sha256']} em {sj}"
                if b == "gnn":
                    assert np.array_equal(z["idx_global"], idx) and np.array_equal(z["target"], y), f"C0 != cache {f}"
                    ref = z
                    sidecars_c0[f.name] = {"run_label": jj.get("run_label"), "saida_npz_sha256": jj["saida_npz_sha256"], "sha256_json": sha256_file(sj)}
                else:
                    assert np.array_equal(z["idx_global"], ref["idx_global"]), f"CUSTODIA idx_global {f}"
                    assert np.array_equal(z["target"], ref["target"]), f"CUSTODIA target {f}"
                    assert np.array_equal(z["sentinela"], ref["sentinela"]), f"CUSTODIA sentinela {f}"
                    if True:  # v1: "if sj.is_file():" (agora obrigatorio, V2-c)
                        rl = jj.get("run_label", "")
                        rng_seed = (jj.get("config_efetiva") or {}).get("rng_seed", (jj.get("patch_faseC") or {}).get("rng_seed"))
                        assert rl == f"{BRACOS_C[b]}_{c}_s{s}_{q}_g10b2", f"run_label {rl} em {sj}"
                        assert rng_seed == s, f"rng_seed {rng_seed} != {s} em {sj}"
                        assert (jj.get("_fontes") or {}).get("checkpoint_sha256"), f"V2-c checkpoint_sha256 ausente em {sj}"
                        sidecars[f.name] = {"run_label": rl, "rng_seed": rng_seed, "checkpoint_sha256": (jj.get("_fontes") or {}).get("checkpoint_sha256"),
                                            "status": jj.get("status"), "sha256_json": sha256_file(sj)}
                r = {"braco": b, "cidade": c, "Q": q, "seed": s}
                r.update(metricas_run(z, y, k, ctx, P)); runs.append(r)
        print(c, q, "pares cod/fis", P["n_cod"], P["n_fis"], "| alvo mono cod %.3f fis %s" % (cel["alvo_mono_cod_viol_frac"], cel["alvo_mono_fis_viol_frac"]),
              "| runs", sum(1 for r in runs if r["cidade"] == c and r["Q"] == q), flush=True)
    cont = {b: sum(1 for r in runs if r["braco"] == b) for b in bracos}
    ins = {"common.py": FASE_B / "common.py", "B0_custodia.json": FASE_B / "B0_custodia.json"}
    ins["C_metricas_v1.py"] = AQUI.parent / "metricas" / "C_metricas.py"  # V2-e
    for c in CIDADES:
        for q in QS:
            ins[f"_cache/{c}_{q}.npz"] = FASE_B / "_cache" / f"{c}_{q}.npz"
    out = {"etapa": "C_metricas", "versao": "v2 (emenda do dono 18/10/2026)", "protocolo": "_revisao_2026-10-14/PROTOCOLO_faseC_C1_C2_2026-10-14.md#Avaliacao",
           "bracos": {b: BRACOS_C[b] for b in bracos}, "n_runs_por_braco": cont, "completo": all(cont[b] == 80 for b in bracos) and not ausentes,
           "ausentes": ausentes, "constantes": {"PL_MAX_VALID": PL_MAX_VALID, "LIM_COV": LIM_COV, "TETO_PL": TETO_PL, "LIM_1KM_m": LIM_1KM,
                                                 "RSSI_SENT": RSSI_SENT, "M_FIS": M_FIS, "SEED_PARES": SEED_PARES,
                                                 "rng_pares": "numpy default_rng([20261018, indice_celula]) com indice na ordem CIDADES x QS"},
           "celulas": celulas, "runs": runs, "sidecars_bracos": sidecars, "sidecars_C0": sidecars_c0,
           "carimbo": carimbo(Path(__file__), ins) | {"npz_sha256": npz_sha}}
    dump(out, a.saida)
    print("gravado", a.saida, cont, "ausentes", len(ausentes))


if __name__ == "__main__":
    main()
