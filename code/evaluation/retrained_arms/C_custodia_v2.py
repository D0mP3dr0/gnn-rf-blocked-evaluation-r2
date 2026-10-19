"""C_custodia_v2.py — nota de custodia da fase C (demanda da ata 18/10/2026, Bloco 4: "distribuicao da testemunha PL lado a lado
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
from common import CIDADES, QS, SEEDS, sha256_file, npz_path, carimbo, dump  # noqa: E402

PRED_C = AQUI.parent.parent / "e3_predicoes"
LIM_E3 = 1e-3          # criterio ja gravado pela inferencia (mae_*_criterio_1e-3_ok)
CTX = {"B0_tol": 2.5e-4, "B0_emenda_tol_PL": 3.2e-3}  # tolerancias existentes, so contexto


def dist(x):
    x = np.asarray(x, float); a = np.abs(x)
    o = {"n": int(x.size), "mediana_abs": float(np.median(a)), "p90_abs": float(np.percentile(a, 90)), "max_abs": float(a.max()),
         "frac_abs_gt_1e-3": float((a > LIM_E3).mean()), "n_abs_gt_1e-3": int((a > LIM_E3).sum()),
         "frac_sinal_pos": float((x > 0).mean()), "frac_sinal_neg": float((x < 0).mean()), "media_com_sinal": float(x.mean())}
    o.update({f"n_abs_gt_{k}": int((a > v).sum()) for k, v in CTX.items()})
    return o


def main():
    m2 = json.load(open(AQUI / "C_metricas_por_corrida_v2.json")); car = m2["carimbo"]["npz_sha256"]
    b0 = json.load(open(FASE_B / "B0_custodia.json"))
    C = {"c1noreg": {"rssi": [], "pl": []}, "c2nomono": {"rssi": [], "pl": []}}
    hash_falhas, n_hash, pl_nok, rssi_nok, linhas = [], 0, 0, 0, []
    for b in C:
        for c in CIDADES:
            for q in QS:
                for s in SEEDS:
                    f = PRED_C / f"{b}_{c}_{q}_s{s}.npz"; j = json.load(open(f.with_suffix(".json")))
                    w = j["testemunha_contra_run_json"]
                    dr = w["mae_rssi_db_recomputado"] - w["mae_rssi_db_run_json"]; dp = w["mae_pl_db_recomputado"] - w["mae_pl_db_run_json"]
                    assert abs(abs(dr) - w["diff_abs_mae_rssi_db"]) < 1e-12 and abs(abs(dp) - w["diff_abs_mae_pl_db"]) < 1e-12
                    C[b]["rssi"].append(dr); C[b]["pl"].append(dp)
                    pl_nok += not w["mae_pl_db_criterio_1e-3_ok"]; rssi_nok += not w["mae_rssi_db_criterio_1e-3_ok"]
                    h_npz = sha256_file(f); h_ck = sha256_file(j["checkpoint_origem"]); h_rj = sha256_file(j["run_json_origem"])
                    ok = (h_npz == j["saida_npz_sha256"] == car[f.name]) and h_ck == j["_fontes"]["checkpoint_sha256"] and h_rj == j["_fontes"]["run_json_sha256"]
                    n_hash += 1
                    if not ok: hash_falhas.append(f.name)
                    linhas.append({"braco": b, "cidade": c, "Q": q, "seed": s, "delta_mae_rssi_ALL": dr, "delta_mae_pl_VALID": dp,
                                   "npz_sha256": h_npz, "checkpoint_sha256": h_ck, "run_json_sha256": h_rj, "hash_ok": ok})
    # C0 na B0
    g = [l for l in b0["linhas"] if l["braco"] == "gnn"]; assert len(g) == 80
    g_r = [l["mae_rssi_npz"] - l["mae_rssi_run"] for l in g]; g_p = [l["mae_pl_valid_npz"] - l["mae_pl_valid_run"] for l in g]
    c0_hash_falhas = []
    for l in g:
        f = npz_path("gnn", l["cidade"], l["Q"], l["seed"]); sj = json.load(open(f.with_suffix(".json")))
        ok = sha256_file(f) == l["npz_sha256"] == sj["saida_npz_sha256"] == car[f.name] and sha256_file(l["run_json"]) == l["run_json_sha256"]
        if not ok: c0_hash_falhas.append(f.name)
    todos_r = C["c1noreg"]["rssi"] + C["c2nomono"]["rssi"]; todos_p = C["c1noreg"]["pl"] + C["c2nomono"]["pl"]
    out = {"etapa": "C_custodia_v2", "demanda": "ata 2026-10-18 faseC, Bloco 4: nota de custodia da fase C (sem tolerancia nova)",
           "definicao": "Delta = recomputado - run JSON (selecao.test_no_melhor_ckpt); RSSI em ALL, PL em VALID",
           "faseC_160": {"rssi": dist(todos_r), "pl": dist(todos_p)},
           "faseC_por_braco": {b: {"rssi": dist(v["rssi"]), "pl": dist(v["pl"])} for b, v in C.items()},
           "C0_B0_80": {"rssi": dist(g_r), "pl": dist(g_p)},
           "criterio_1e-3_gravado_pela_inferencia": {"n_pl_ok_False": pl_nok, "n_rssi_ok_False": rssi_nok},
           "custodia_hash": {"faseC_n": n_hash, "faseC_falhas": hash_falhas, "C0_n": len(g), "C0_falhas": c0_hash_falhas,
                             "ok": not hash_falhas and not c0_hash_falhas and n_hash == 160},
           "linhas_faseC": linhas}
    out["carimbo"] = carimbo(Path(__file__), {"C_metricas_por_corrida_v2.json": AQUI / "C_metricas_por_corrida_v2.json", "B0_custodia.json": FASE_B / "B0_custodia.json"})
    dump(out, AQUI / "C_custodia_v2.json")
    sha_json = sha256_file(AQUI / "C_custodia_v2.json")

    def lin(nome, d):
        return (f"| {nome} | {d['n']} | {d['mediana_abs']:.3e} | {d['p90_abs']:.3e} | {d['max_abs']:.3e} | {d['frac_abs_gt_1e-3']:.4f} ({d['n_abs_gt_1e-3']}) "
                f"| {d['frac_sinal_pos']:.3f} / {d['frac_sinal_neg']:.3f} | {d['media_com_sinal']:+.3e} | {d['n_abs_gt_B0_tol']} | {d['n_abs_gt_B0_emenda_tol_PL']} |")
    cab = ("| conjunto | n | mediana abs (dB) | p90 abs (dB) | máx abs (dB) | fração > 1e-3 (n) | sinal + / − | média com sinal (dB) "
           "| n > 2,5e-4 | n > 3,2e-3 |\n|---|---|---|---|---|---|---|---|---|---|")
    F = out["faseC_160"]; B0 = out["C0_B0_80"]; P = out["faseC_por_braco"]; H = out["custodia_hash"]
    md = f"""# NOTA_CUSTODIA_faseC — testemunha da inferência contra o run JSON (fase C) lado a lado com o C0 na B0

Gerada por `C_custodia_v2.py` (sha256 {out['carimbo']['script_sha256']}) a partir de `C_custodia_v2.json` (sha256 {sha_json}).
Demanda: ata `AGENTES/cientista_dados/ATAS/ata_2026-10-18_ieee-access-47637-faseC.md`, Bloco 4. Números PROVISÓRIOS até reexecução
independente. Nenhuma tolerância nova: 1e-3 dB é o critério que a inferência já grava (`mae_*_criterio_1e-3_ok`); 2,5e-4 (B0) e
3,2e-3 (emenda B0 de 14/10) aparecem só como contagem de contexto, com as tolerâncias que já existem.

Definição: Δ = MAE recomputado da predição − MAE do run JSON (`selecao.test_no_melhor_ckpt`); RSSI em ALL, PL em VALID. Fase C:
campo `testemunha_contra_run_json` dos 160 sidecars. C0: `fase_B/B0_custodia.json`, 80 linhas `gnn` (MAE do npz em float64).

## |ΔMAE PL| (VALID)
{cab}
{lin("fase C, 160", F['pl'])}
{lin("fase C, c1noreg", P['c1noreg']['pl'])}
{lin("fase C, c2nomono", P['c2nomono']['pl'])}
{lin("C0 na B0, 80", B0['pl'])}

## |ΔMAE RSSI| (ALL)
{cab}
{lin("fase C, 160", F['rssi'])}
{lin("fase C, c1noreg", P['c1noreg']['rssi'])}
{lin("fase C, c2nomono", P['c2nomono']['rssi'])}
{lin("C0 na B0, 80", B0['rssi'])}

Critério de 1e-3 gravado pela inferência: `pl_ok=False` em {out['criterio_1e-3_gravado_pela_inferencia']['n_pl_ok_False']}/160,
`rssi_ok=False` em {out['criterio_1e-3_gravado_pela_inferencia']['n_rssi_ok_False']}/160.

## Custódia por hash
- Fase C ({H['faseC_n']} corridas): sha256 atual do npz == `saida_npz_sha256` do sidecar == `carimbo.npz_sha256` de
  `C_metricas_por_corrida_v2.json`; sha256 atual de `checkpoint_best.pt` e do run JSON de origem == `_fontes.checkpoint_sha256` e
  `_fontes.run_json_sha256` do sidecar. Falhas: {len(H['faseC_falhas'])} {H['faseC_falhas'] if H['faseC_falhas'] else ''}.
- C0 ({H['C0_n']} corridas): sha256 atual do npz == `B0_custodia.json:npz_sha256` == sidecar == carimbo v2; run JSON == `run_json_sha256`
  da B0. Falhas: {len(H['C0_falhas'])} {H['C0_falhas'] if H['C0_falhas'] else ''}.
- Custódia declarada por hash: {'PASSA' if H['ok'] else 'FALHA'}. A identidade dos artefatos é provada pelo sha; a diferença da
  testemunha é relatada como distribuição e não é convertida em tolerância.
"""
    (AQUI / "NOTA_CUSTODIA_faseC.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
