
"""Sensitivity of the capacity-matched control comparison (GNN-RF vs. graph-free) to the corrected Sorocaba Q3 target."""
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access"
            r"\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO")
AGREGADOR = BASE / "scripts" / "agregar_t11_mlp_vs_gnn_4cidades_2026-09-15.py"
AGREGADO = (BASE / "dados" / "treinos_c1"
            / "agregado_t11_mlp_vs_gnn_4cidades_2026-09-15.json")
R2 = BASE / "dados" / "contra_auditoria" / "rodada2_2026-09-13"
T1_GNN = R2 / "t1-sorocaba-imputacao.json"
T1_MLP = R2 / "t1-mlp-sorocaba-imputacao.json"
T1_MLP_BACKUP = R2 / "t1-mlp-sorocaba-imputacao.ORIGINAL_backup.json"
MANIFEST = BASE / "manifest.jsonl"
OUT = (BASE / "dados" / "treinos_c1"
       / "sensibilidade_t11_sorocaba_q3_alvo_corrigido_2026-09-16.json")

CIDADES = ["lins", "bauru", "campinas", "sorocaba"]
SEEDS = [42, 43, 44, 45, 46]
QUADS = ["Q1", "Q2", "Q3", "Q4"]
CEL = "sorocaba_Q3"
UNIDADES = ("cidade_n4", "celula_n16", "cadeia_n20")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return str(Path(p).relative_to(BASE)).replace("\\", "/")


def manifest_ultimo_hash():
    ult = {}
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for i, linha in enumerate(f, 1):
            try:
                d = json.loads(linha)
            except json.JSONDecodeError:
                continue
            ult[d["artefato"]] = d["sha256"]
    return ult


def carrega_agregador():
    spec = importlib.util.spec_from_file_location("agt11", AGREGADOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unidades(mod, delta):
    """
    Maps delta[(c, s, q)] to the three aggregation units, using the same
    population definitions as the primary aggregator.
    """
    nomes_cel = [f"{c}_{q}" for c in CIDADES for q in QUADS]
    d_cel = {f"{c}_{q}": float(np.mean(
        [delta[(c, s, q)] for s in SEEDS])) for c in CIDADES for q in QUADS}
    cad = {f"{c}_s{s}": float(np.mean([delta[(c, s, q)] for q in QUADS]))
           for c in CIDADES for s in SEEDS}
    cid = {c: float(np.mean([cad[f"{c}_s{s}"] for s in SEEDS]))
           for c in CIDADES}
    v_cel = [d_cel[k] for k in nomes_cel]
    v_cad = [cad[f"{c}_s{s}"] for c in CIDADES for s in SEEDS]
    v_cid = [cid[c] for c in CIDADES]
    m = mod.MARGEM_TOST_DB
    return {
        "cidade_n4": {**mod.estat(v_cid, "cidade (n=4)"), "valores_db": cid,
                      "tost_margem_0_1": mod.tost(v_cid, m)},
        "celula_n16": {**mod.estat(v_cel, "celula (n=16)"),
                       "valores_db": d_cel,
                       "tost_margem_0_1": mod.tost(v_cel, m)},
        "cadeia_n20": {**mod.estat(v_cad, "cadeia (n=20)"),
                       "valores_db": cad,
                       "tost_margem_0_1": mod.tost(v_cad, m)},
    }


def resumo(u):
    return {"n": u["n"], "media_delta_db": u["media_delta_db"],
            "ic95_db": u["ic95_db"], "p_bicaudal_t": u["p_bicaudal_t"],
            "wilcoxon_p_bicaudal": u["wilcoxon_p_bicaudal"],
            "teste_do_sinal_p_bicaudal": u["teste_do_sinal_p_bicaudal"],
            "sinal_gnn_melhor": u["sinal_n_delta_negativo_gnn_melhor"],
            "sinal_mlp_melhor": u["sinal_n_delta_positivo_mlp_melhor"],
            "p_minimo_nao_parametrico_possivel":
                u["p_minimo_nao_parametrico_possivel"],
            "tost_p_margem_0_1": u["tost_margem_0_1"]["p_tost"]}


def main():
    fontes = {}
    man = manifest_ultimo_hash()
    for p in (AGREGADOR, AGREGADO, T1_GNN):
        h = sha(p)
        fontes[rel(p)] = h
        if man.get(rel(p)) != h:
            sys.exit(f"ERRO: {rel(p)} sha256 {h} != manifest {man.get(rel(p))}")

    h_mlp = sha(T1_MLP)
    fontes[rel(T1_MLP)] = h_mlp
    nota_mlp_manifest = {"sha256_atual": h_mlp,
                         "sha256_no_manifest": man[rel(T1_MLP)],
                         "iguais": h_mlp == man[rel(T1_MLP)]}

    mod = carrega_agregador()
    ag = json.loads(AGREGADO.read_text(encoding="utf-8"))
    g = json.loads(T1_GNN.read_text(encoding="utf-8"))
    m = json.loads(T1_MLP.read_text(encoding="utf-8"))

    if T1_MLP_BACKUP.exists():
        hb = sha(T1_MLP_BACKUP)
        fontes[rel(T1_MLP_BACKUP)] = hb
        mb = json.loads(T1_MLP_BACKUP.read_text(encoding="utf-8"))
        campos = ("mae_rssi_db_reproduzido_alvo_original",
                  "mae_rssi_db_registrado_no_run",
                  "mae_rssi_db_alvo_corrigido",
                  "delta_mae_rssi_db_corrigido_menos_original",
                  "n_afetados_test", "epoca_do_ckpt", "checkpoint")
        for s in SEEDS:
            k = f"mlpcf_sorocaba_s{s}_Q3_g10b2"
            for c in campos:
                if m["corridas"][k][c] != mb["corridas"][k][c]:
                    sys.exit(f"ERRO: {k}.{c} difere entre atual e backup")
        nota_mlp_manifest["backup_sha256"] = hb
        nota_mlp_manifest["backup_igual_ao_manifest"] = (
            hb == man[rel(T1_MLP)])
        nota_mlp_manifest["campos_de_mae_identicos_atual_x_backup"] = True


    delta_prim = {}
    for c in CIDADES:
        for q in QUADS:
            ps = ag["celulas"][f"{c}_{q}"]["por_semente"]
            for s in SEEDS:
                e = ps[f"s{s}"]
                delta_prim[(c, s, q)] = e["gnn"] - e["mlp"]
    u_prim = unidades(mod, delta_prim)
    for k in UNIDADES:
        a, b = u_prim[k], ag["unidades"][k]
        for campo in ("media_delta_db", "p_bicaudal_t"):
            if abs(a[campo] - b[campo]) > 1e-12:
                sys.exit(f"ERRO: recomputo do primario diverge em {k}.{campo}")
        if max(abs(x - y) for x, y in zip(a["ic95_db"], b["ic95_db"])) > 1e-12:
            sys.exit(f"ERRO: recomputo do primario diverge em {k}.ic95_db")


    eg = g["etapa2"]["Q3"]
    em = m["corridas"]
    n_af_g, n_af_m = set(), set()
    subst = {}
    delta_lit = dict(delta_prim)
    delta_mix = dict(delta_prim)
    ps = ag["celulas"][CEL]["por_semente"]
    for s in SEEDS:
        rg = eg[f"run_c0c1cf_sorocaba_s{s}_Q3_g10b2.json"]
        rm = em[f"mlpcf_sorocaba_s{s}_Q3_g10b2"]
        if rg["run_label"] != f"c0c1cf_sorocaba_s{s}_Q3_g10b2":
            sys.exit(f"ERRO: run_label inesperado {rg['run_label']}")
        if rm["seed"] != s:
            sys.exit(f"ERRO: semente inesperada no braco MLP s{s}")
        if abs(rg["mae_rssi_db_registrado_no_run"] - ps[f"s{s}"]["gnn"]) > 1e-9:
            sys.exit(f"ERRO: GNN s{s} registrado != agregado")
        if abs(rm["mae_rssi_db_registrado_no_run"] - ps[f"s{s}"]["mlp"]) > 1e-9:
            sys.exit(f"ERRO: MLP s{s} registrado != agregado")
        if rm["guarda_reproducao_passou"] is not True:
            sys.exit(f"ERRO: guarda de reproducao MLP falhou em s{s}")
        if abs(rg["delta_reproducao_rssi_db"]) >= 1e-3:
            sys.exit(f"ERRO: reproducao GNN s{s} fora de 1e-3 dB")
        n_af_g.add(rg["n_afetados_test"])
        n_af_m.add(rm["n_afetados_test"])
        g_corr = rg["mae_rssi_db_alvo_corrigido"]
        m_corr = rm["mae_rssi_db_alvo_corrigido"]
        g_mix = (rg["mae_rssi_db_registrado_no_run"]
                 + (g_corr - rg["mae_rssi_db_reproduzido_alvo_original"]))
        m_mix = (rm["mae_rssi_db_registrado_no_run"]
                 + (m_corr - rm["mae_rssi_db_reproduzido_alvo_original"]))
        delta_lit[("sorocaba", s, "Q3")] = g_corr - m_corr
        delta_mix[("sorocaba", s, "Q3")] = g_mix - m_mix
        subst[f"s{s}"] = {
            "gnn_mae_primario": ps[f"s{s}"]["gnn"],
            "gnn_mae_alvo_corrigido": g_corr,
            "mlp_mae_primario": ps[f"s{s}"]["mlp"],
            "mlp_mae_alvo_corrigido": m_corr,
            "delta_primario": ps[f"s{s}"]["delta"],
            "delta_alvo_corrigido": g_corr - m_corr,
            "delta_variante_registrado_mais_delta": g_mix - m_mix,
            "gnn_erro_reproducao_db": rg["delta_reproducao_rssi_db"],
            "mlp_erro_reproducao_db": rm["delta_reproducao_rssi_db"],
        }
    if len(n_af_g | n_af_m) != 1:
        sys.exit(f"ERRO: n_afetados_test difere entre bracos {n_af_g} {n_af_m}")

    u_lit = unidades(mod, delta_lit)
    u_mix = unidades(mod, delta_mix)
    cel_prim = ag["celulas"][CEL]["delta_gnn_menos_mlp_db"]
    cel_lit = u_lit["celula_n16"]["valores_db"][CEL]

    lado_a_lado = {}
    for k in UNIDADES:
        lado_a_lado[k] = {
            "primario_pre_registrado": resumo(ag["unidades"][k]),
            "sensibilidade_sorocaba_Q3_alvo_corrigido": resumo(u_lit[k]),
            "variante_registrado_mais_delta": resumo(u_mix[k]),
            "diferenca_media_sensibilidade_menos_primario_db":
                u_lit[k]["media_delta_db"] - ag["unidades"][k]["media_delta_db"],
        }
    contagem = {
        "gnn_melhor": sum(1 for v in u_lit["celula_n16"]["valores_db"].values()
                          if v < 0),
        "mlp_melhor": sum(1 for v in u_lit["celula_n16"]["valores_db"].values()
                          if v > 0)}

    saida = {
        "artefato_tipo": "sensibilidade_t11_sorocaba_q3_alvo_corrigido",
        "data": "2026-09-16",
        "gravado_utc": datetime.now(timezone.utc).isoformat(),
        "status": "provisorio (pendente de contra-auditoria por quem nao "
                  "escreveu este script)",
        "natureza": "ANALISE DE SENSIBILIDADE. NAO substitui o resultado "
                    "primario pre-registrado do T11 "
                    "(agregado_t11_mlp_vs_gnn_4cidades_2026-09-15.json :: "
                    "unidades), que continua sendo o resultado do criterio. "
                    "Feita depois do primario; nao ha criterio pre-registrado "
                    "para ela.",
        "o_que_muda": "somente as 5 sementes (42-46) de sorocaba_Q3: MAE de "
                      "teste de cada braco trocado pelo MAE com alvo de RSSI "
                      "corrigido (regra prepare_transfer_dataset_v19.py:274 "
                      "aplicada em memoria), medido por inferencia nos "
                      "checkpoint_best.pt das corridas c0c1cf e mlpcf",
        "o_que_nao_muda": "as outras 15 celulas; os pesos (nenhum retreino: "
                          "os dois modelos foram TREINADOS contra o alvo "
                          "defeituoso; so a avaliacao mudou)",
        "fontes_campos": {
            "gnn": "t1-sorocaba-imputacao.json :: etapa2.Q3."
                   "run_c0c1cf_sorocaba_s{42..46}_Q3_g10b2.json."
                   "mae_rssi_db_alvo_corrigido (c0c1cfsc s42/s43 IGNORADAS)",
            "mlp": "t1-mlp-sorocaba-imputacao.json :: corridas."
                   "mlpcf_sorocaba_s{42..46}_Q3_g10b2.mae_rssi_db_alvo_corrigido",
            "demais": "agregado_t11_mlp_vs_gnn_4cidades_2026-09-15.json :: "
                      "celulas.*.por_semente",
        },
        "estimador": "funcoes estat() e tost() importadas de "
                     + rel(AGREGADOR) + " (mesmo codigo do primario)",
        "controle_recomputo_primario": "agregado.unidades reproduzido a partir "
                                       "de agregado.celulas com |dif| <= 1e-12",
        "celula_sorocaba_Q3": {
            "delta_primario_db": cel_prim,
            "delta_alvo_corrigido_db": cel_lit,
            "mudanca_db": cel_lit - cel_prim,
            "por_semente": subst,
            "n_afetados_test": n_af_g.pop(),
        },
        "lado_a_lado": lado_a_lado,
        "contagem_celulas_sensibilidade": contagem,
        "unidades_sensibilidade_completas": u_lit,
        "nota_manifest_braco_mlp": nota_mlp_manifest,
        "limites": [
            "sensibilidade so da AVALIACAO: o efeito do alvo errado sobre os "
            "pesos (treino) nao e medido aqui; so retreino (T1-bis) mede",
            "sorocaba Q2 e Q4 tem nos afetados no TREINO (T1-A4); o delta de "
            "teste deles e zero por construcao e nao foi mexido",
            "herda todos os limites do primario (celulas nao independentes, "
            "buffer 2 km < alcance de autocorrelacao)",
        ],
        "script": str(Path(__file__)),
        "script_sha256": sha(Path(__file__)),
        "fontes_sha256": fontes,
        "ambiente": {"python": sys.version.split()[0],
                     "numpy": np.__version__,
                     "executavel": sys.executable,
                     "dispositivo": "CPU, so leitura de JSON"},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    print("OK", OUT)
    print(f"sorocaba_Q3: delta primario {cel_prim:+.4f} -> corrigido "
          f"{cel_lit:+.4f} dB")
    for k in UNIDADES:
        a, b, c = (lado_a_lado[k]["primario_pre_registrado"],
                   lado_a_lado[k]["sensibilidade_sorocaba_Q3_alvo_corrigido"],
                   lado_a_lado[k]["variante_registrado_mais_delta"])
        for nome, u in (("prim", a), ("sens", b), ("mix ", c)):
            print(f"{k} {nome}: d={u['media_delta_db']:+.4f} IC95 "
                  f"[{u['ic95_db'][0]:+.4f}; {u['ic95_db'][1]:+.4f}] "
                  f"p_t={u['p_bicaudal_t']:.4f} wil={u['wilcoxon_p_bicaudal']} "
                  f"sinal={u['teste_do_sinal_p_bicaudal']} "
                  f"({u['sinal_gnn_melhor']}G/{u['sinal_mlp_melhor']}M) "
                  f"tost={u['tost_p_margem_0_1']:.4f}")
    print("celulas", contagem, "| manifest MLP:", nota_mlp_manifest)


if __name__ == "__main__":
    main()
