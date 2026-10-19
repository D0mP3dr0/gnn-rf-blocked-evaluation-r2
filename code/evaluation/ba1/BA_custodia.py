"""BA_custodia.py — custodia das 80 corridas do BA1 (PROTOCOLO_faseC_baselines_RASCUNHO_2026-10-14.md, secao 7: "MAE RSSI do npz
contra selecao.test_no_melhor_ckpt do run JSON, com tolerancia <= 2,5e-4 dB"). COPIA de fase_C/avaliacao/metricas_v2/C_custodia_v2.py
com ADAPTACAO MINIMA (diff em BA_custodia.diff). Escreve SO BA_custodia.json nesta pasta. Adaptacoes (BA-o..BA-r):
  BA-o  testemunha recomputada DO NPZ em float64 (como fase_B/B0_custodia.py), nao lida do sidecar: MAE RSSI ALL e MAE PL VALID
        (alvo PL < 299) contra run JSON selecao.test_no_melhor_ckpt.mae_rssi_db / mae_pl_db; criterio do protocolo so no RSSI,
        |Delta| <= 2,5e-4 dB (passa/falha por corrida); PL relatado como distribuicao, sem tolerancia nova (como a nota da v2).
        Tambem: n do npz == n_nos e n_VALID == n_pl_alvo_valido do run JSON; sha256_idx(idx_global) == particoes.test.idx_sha256_global;
        e a testemunha gravada pelo sidecar (float32 da inferencia) ao lado.
  BA-p  custodia por hash do BA1: sha256 atual do npz == saida_npz_sha256 do sidecar == carimbo.npz_sha256 de
        BA_metricas_por_corrida.json; sha256 atual de checkpoint_best.pt e do run JSON == _fontes do sidecar.
  BA-q  contexto: C0 (gnn) e MLP (mlp) na B0 (fase_B/B0_custodia.json), as mesmas verificacoes de hash da v2 aplicadas aos dois.
  BA-r  sem nota .md (fora das saidas pedidas); sai com codigo 4 se alguma corrida do BA1 falhar (RSSI > 2,5e-4, n, idx ou hash),
        DEPOIS de gravar o JSON.

--- cabecalho da v2 (inalterado) ---
C_custodia_v2.py — nota de custodia da fase C (demanda da ata 18/10/2026, Bloco 4: "distribuicao da testemunha PL lado a lado
com B0, custodia por hash, sem tolerancia nova"). Escreve SO em metricas_v2/: C_custodia_v2.json e NOTA_CUSTODIA_faseC.md
(a nota e gerada daqui; todo numero dela e copia do JSON).

Testemunha: |Delta MAE| entre a metrica recomputada da predicao e a gravada no run JSON (selecao.test_no_melhor_ckpt), RSSI em
ALL e PL em VALID — as mesmas duas definicoes nos dois lados:
  fase C (160 corridas c1noreg + c2nomono): campo testemunha_contra_run_json do sidecar .json da inferencia
            (e3_inferencia_faseC.py:1018-1034; recomputado = populacoes.todos.mae_rssi_db / cobertos.mae_pl_db);
  C0 na B0 (80 corridas gnn): fase_B/B0_custodia.json, linhas[braco=gnn] (B0_custodia.py: MAE do npz em float64 vs run JSON).
  Sinal = recomputado - run JSON. Estatisticas: mediana, p90, maximo, fracao > 1e-3 dB (o criterio de 1e-3 ja gravado pela
  inferencia, mae_*_criterio_1e-3_ok — nenhuma tolerancia nova), fracao positiva/negativa. Contagens acima de 2,5e-4 (B0) e de
  3,2e-3 (emenda B0) sao contexto, com as tolerancias que ja existem.
Custodia por hash (fase C, 160): sha256 atual do npz == saida_npz_sha256 do sidecar == carimbo.npz_sha256 da v2; sha256 atual do
  checkpoint_best.pt e do run JSON de origem == _fontes.checkpoint_sha256 / _fontes.run_json_sha256 do sidecar.
  C0 (80): npz_sha256 e run_json_sha256 da B0 == sha atual; npz == sidecar do C0.
Uso: python C_custodia_v2.py   (depois de C_metricas_v2.py)"""
import sys, json
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent
FASE_B = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_B")
sys.path.insert(0, str(FASE_B)); sys.dont_write_bytecode = True
from common import CIDADES, QS, SEEDS, PL_MAX_VALID, sha256_file, sha256_idx, npz_path, carimbo, dump  # noqa: E402

PRED_C = AQUI.parent.parent / "e3_predicoes"  # BA1 (fase_BA1/e3_predicoes)
TREINOS = AQUI.parent.parent / "treinos"  # BA-o
FASE_C_V2 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/_revisao_2026-10-14/fase_C/avaliacao/metricas_v2")  # origem, so leitura
LIM_E3 = 1e-3          # criterio ja gravado pela inferencia (mae_*_criterio_1e-3_ok)
CTX = {"B0_tol": 2.5e-4, "B0_emenda_tol_PL": 3.2e-3}  # tolerancias existentes, so contexto
TOL_PROT = 2.5e-4      # BA-o: tolerancia do protocolo (secao 7) no MAE RSSI


def dist(x):
    x = np.asarray(x, float); a = np.abs(x)
    o = {"n": int(x.size), "mediana_abs": float(np.median(a)), "p90_abs": float(np.percentile(a, 90)), "max_abs": float(a.max()),
         "frac_abs_gt_1e-3": float((a > LIM_E3).mean()), "n_abs_gt_1e-3": int((a > LIM_E3).sum()),
         "frac_sinal_pos": float((x > 0).mean()), "frac_sinal_neg": float((x < 0).mean()), "media_com_sinal": float(x.mean())}
    o.update({f"n_abs_gt_{k}": int((a > v).sum()) for k, v in CTX.items()})
    return o


def main():
    m2 = json.load(open(AQUI / "BA_metricas_por_corrida.json")); car = m2["carimbo"]["npz_sha256"]  # BA-p
    b0 = json.load(open(FASE_B / "B0_custodia.json"))
    C = {"rssi": [], "pl": []}; Cs = {"rssi": [], "pl": []}
    falhas, linhas = [], []
    for c in CIDADES:
        for q in QS:
            for s in SEEDS:
                rl = f"ba1rus_{c}_s{s}_{q}_g10b2"
                f = PRED_C / f"ba1rus_{c}_{q}_s{s}.npz"; j = json.load(open(f.with_suffix(".json")))
                rjf = TREINOS / rl / f"run_{rl}.json"; rj = json.load(open(rjf)); sel = rj["selecao"]["test_no_melhor_ckpt"]
                z = np.load(f); t = z["target"].astype(np.float64); p = z["pred"].astype(np.float64); v = z["target"][:, 0] < PL_MAX_VALID
                m_r = float(np.abs(p[:, 3] - t[:, 3]).mean()); m_p = float(np.abs(p[v, 0] - t[v, 0]).mean())  # BA-o: float64 do npz
                dr = m_r - sel["mae_rssi_db"]; dp = m_p - sel["mae_pl_db"]
                C["rssi"].append(dr); C["pl"].append(dp)
                w = j["testemunha_contra_run_json"]  # float32 da inferencia, ao lado
                Cs["rssi"].append(w["mae_rssi_db_recomputado"] - w["mae_rssi_db_run_json"]); Cs["pl"].append(w["mae_pl_db_recomputado"] - w["mae_pl_db_run_json"])
                n_ok = len(t) == sel["n_nos"] == sel["n_nos_avaliados"] and int(v.sum()) == sel["n_pl_alvo_valido"]
                idx_ok = sha256_idx(z["idx_global"]) == rj["particoes"]["test"]["idx_sha256_global"]
                assert j["run_label"] == rl and rj["run_label"] == rl, f"run_label {j['run_label']} / {rj['run_label']} != {rl}"
                h_npz = sha256_file(f); h_ck = sha256_file(j["checkpoint_origem"]); h_rj = sha256_file(rjf)
                hash_ok = (h_npz == j["saida_npz_sha256"] == car[f.name]) and h_ck == j["_fontes"]["checkpoint_sha256"] and h_rj == j["_fontes"]["run_json_sha256"] \
                    and Path(j["run_json_origem"]).resolve() == rjf.resolve()
                rssi_ok = abs(dr) <= TOL_PROT
                ok = rssi_ok and n_ok and idx_ok and hash_ok
                if not ok: falhas.append(rl)
                linhas.append({"run_label": rl, "cidade": c, "Q": q, "seed": s, "n": int(len(t)), "n_VALID": int(v.sum()),
                               "mae_rssi_ALL_npz_f64": m_r, "mae_rssi_ALL_run_json": sel["mae_rssi_db"], "delta_mae_rssi_ALL": dr, "rssi_tol_2.5e-4_ok": rssi_ok,
                               "mae_pl_VALID_npz_f64": m_p, "mae_pl_VALID_run_json": sel["mae_pl_db"], "delta_mae_pl_VALID": dp,
                               "testemunha_sidecar_diff_abs_rssi": w["diff_abs_mae_rssi_db"], "testemunha_sidecar_diff_abs_pl": w["diff_abs_mae_pl_db"],
                               "n_ok": n_ok, "idx_sha256_ok": idx_ok, "hash_ok": hash_ok, "passa": ok,
                               "npz": str(f), "npz_sha256": h_npz, "checkpoint": j["checkpoint_origem"], "checkpoint_sha256": h_ck,
                               "run_json": str(rjf), "run_json_sha256": h_rj})
    # BA-q: C0 e MLP na B0 (contexto), mesmas verificacoes de hash da v2
    ctx = {}
    for b in ("gnn", "mlp"):
        g = [l for l in b0["linhas"] if l["braco"] == b]; assert len(g) == 80
        g_r = [l["mae_rssi_npz"] - l["mae_rssi_run"] for l in g]; g_p = [l["mae_pl_valid_npz"] - l["mae_pl_valid_run"] for l in g]
        hf = []
        for l in g:
            f = npz_path(b, l["cidade"], l["Q"], l["seed"]); sj = json.load(open(f.with_suffix(".json")))
            ok = sha256_file(f) == l["npz_sha256"] == sj["saida_npz_sha256"] == car[f.name] and sha256_file(l["run_json"]) == l["run_json_sha256"]
            if not ok: hf.append(f.name)
        ctx[b] = {"rssi": dist(g_r), "pl": dist(g_p), "n_rssi_gt_2.5e-4": int(sum(abs(x) > TOL_PROT for x in g_r)), "hash_n": len(g), "hash_falhas": hf}
    out = {"etapa": "BA_custodia", "protocolo": "PROTOCOLO_faseC_baselines_RASCUNHO_2026-10-14.md#7 (Custodia)",
           "definicao": "Delta = MAE recomputado do npz (float64) - MAE do run JSON (selecao.test_no_melhor_ckpt); RSSI em ALL, PL em VALID (alvo PL < 299)",
           "tolerancia_rssi_db": TOL_PROT, "n_corridas": len(linhas), "n_passa": sum(l["passa"] for l in linhas), "falhas": falhas,
           "ok": not falhas and len(linhas) == 80,
           "BA1_80": {"rssi": dist(C["rssi"]), "pl": dist(C["pl"]), "n_rssi_gt_2.5e-4": int(sum(abs(x) > TOL_PROT for x in C["rssi"]))},
           "BA1_80_testemunha_sidecar_float32": {"rssi": dist(Cs["rssi"]), "pl": dist(Cs["pl"])},
           "contexto_B0": ctx, "linhas": linhas}
    out["carimbo"] = carimbo(Path(__file__), {"BA_metricas_por_corrida.json": AQUI / "BA_metricas_por_corrida.json", "B0_custodia.json": FASE_B / "B0_custodia.json",
                                              "C_custodia_v2.py": FASE_C_V2 / "C_custodia_v2.py"})
    dump(out, AQUI / "BA_custodia.json")
    print("BA1 custodia: %d/%d passa | RSSI |D| max %.3e mediana %.3e | PL |D| max %.3e | falhas %s" % (
          out["n_passa"], out["n_corridas"], out["BA1_80"]["rssi"]["max_abs"], out["BA1_80"]["rssi"]["mediana_abs"], out["BA1_80"]["pl"]["max_abs"], falhas))
    for b, v in ctx.items():
        print("contexto B0 %s: RSSI |D| max %.3e (n > 2,5e-4: %d) | PL |D| max %.3e | hash falhas %d" % (b, v["rssi"]["max_abs"], v["n_rssi_gt_2.5e-4"], v["pl"]["max_abs"], len(v["hash_falhas"])))
    print("gravado", AQUI / "BA_custodia.json")
    if not out["ok"]:
        sys.exit(4)  # BA-r


if __name__ == "__main__":
    main()
