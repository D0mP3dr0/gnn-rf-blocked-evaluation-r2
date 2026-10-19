
"""
Paired comparison of the GNN model (c0c1cf) against a graph-free MLP
control (mlpcf) across four cities under a spatially blocked
train/val/test split.

Aggregation units:
    cell  (n=16): city x quadrant, five seeds (replicates).
    chain (n=20): city x seed, averaged over the four quadrants.
    city  (n=4):  city, averaged over chains.

Metric and population: MAE of RSSI in dB over all nodes of the blocked
test partition (the -110 dBm sentinel included), at the checkpoint
selected by validation loss. Only this population is used here (not
valid-target PL<299 or covered nodes), to keep the estimator
comparable across runs.

Sign convention: delta = MAE(GNN) - MAE(MLP); a negative delta means
the GNN has lower error.

Pairing: by cell and seed. A pair is discarded if the input-data hash
or any split index hash (train/val/test) differs between the GNN and
MLP runs, or if either run is missing a recorded wall-clock cost.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access"
            r"\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO")
TREINOS = BASE / "treinos"
BASEL = BASE / "dados" / "baselines_v2"
PLANO = BASE / "dados" / "planos" / "criterios_fechamento_2026-09-13.json"
OUT = (BASE / "dados" / "treinos_c1"
       / "agregado_t11_mlp_vs_gnn_4cidades_2026-09-15.json")

CIDADES = ["lins", "bauru", "campinas", "sorocaba"]
SEEDS = [42, 43, 44, 45, 46]
QUADS = ["Q1", "Q2", "Q3", "Q4"]
MARGEM_TOST_DB = 0.1   # pre-registered margin
ALFA = 0.05


def sha256_arquivo(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def le_run(prefixo, c, s, q, fontes):
    label = f"{prefixo}_{c}_s{s}_{q}_g10b2"
    p = TREINOS / label / f"run_{label}.json"
    if not p.exists():
        sys.exit(f"ERRO: corrida ausente: {p}")
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    if d["custo"]["tempo_total_s"] is None:
        sys.exit(f"ERRO: {label} incompleto (custo.tempo_total_s nulo)")
    fontes[str(p.relative_to(BASE)).replace("\\", "/")] = sha256_arquivo(p)
    return label, d


def identidade(d):
    return (d["dataset"]["rf_data_sha256"],
            tuple(d["particoes"][k]["idx_sha256_global"]
                  for k in ("train", "val", "test")))


def estat(vals, rotulo):
    """
    Paired one-sample test on the deltas: two-sided t-test with 95% CI, plus
    the non-parametric Wilcoxon signed-rank test and exact sign test, each
    reported together with its own minimum achievable p-value for the given
    sample size.
    """
    v = np.asarray(vals, dtype=float)
    n = int(v.size)
    m = float(v.mean())
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(n)
    gl = n - 1
    t, p = stats.ttest_1samp(v, 0.0)
    tc = float(stats.t.ppf(1 - ALFA / 2, gl))
    try:
        w = stats.wilcoxon(v)
        wp = float(w.pvalue)
    except Exception as e:
        wp = None
        print(f"[aviso] wilcoxon indisponivel em {rotulo}: {e}")
    n_pos = int((v > 0).sum())
    n_neg = int((v < 0).sum())
    n_zero = int((v == 0).sum())
    if n_pos + n_neg > 0:
        sinal_p = float(stats.binomtest(n_pos, n_pos + n_neg, 0.5,
                                        alternative="two-sided").pvalue)
    else:
        sinal_p = None
    return {
        "unidade": rotulo,
        "n": n,
        "media_delta_db": m,
        "dp_db": sd,
        "ep_db": se,
        "gl": gl,
        "t": float(t),
        "p_bicaudal_t": float(p),
        "ic95_db": [m - tc * se, m + tc * se],
        "t_critico": tc,
        "wilcoxon_p_bicaudal": wp,
        "sinal_n_delta_positivo_mlp_melhor": n_pos,
        "sinal_n_delta_negativo_gnn_melhor": n_neg,
        "sinal_n_delta_zero": n_zero,
        "teste_do_sinal_p_bicaudal": sinal_p,
        "p_minimo_nao_parametrico_possivel": 2.0 / (2 ** n),
        "nota_piso": ("piso exato de um teste bicaudal de sinais/postos com "
                      f"n={n}: nenhum p nao parametrico pode ficar abaixo de "
                      f"{2.0 / (2 ** n):.6g}, mesmo com todas as unidades "
                      "apontando para o mesmo lado"),
    }


def tost(vals, margem):
    v = np.asarray(vals, dtype=float)
    n = int(v.size)
    m = float(v.mean())
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(n)
    gl = n - 1
    p_low = float(stats.ttest_1samp(v, -margem, alternative="greater").pvalue)
    p_up = float(stats.ttest_1samp(v, margem, alternative="less").pvalue)
    return {"margem_db": margem, "n": n, "p_inferior": p_low,
            "p_superior": p_up, "p_tost": max(p_low, p_up),
            "equivalencia_estabelecida_alfa_0_05": max(p_low, p_up) < ALFA}


def main():
    fontes = {}
    plano = json.loads(PLANO.read_text(encoding="utf-8"))
    criterio_t11 = plano["testes"]["T11_mlp_em_campinas_e_sorocaba"]
    fontes[str(PLANO.relative_to(BASE)).replace("\\", "/")] = sha256_arquivo(PLANO)

    celulas = {}
    problemas = []
    delta_por_cel_seed = {}
    for c in CIDADES:
        for q in QUADS:
            pb = BASEL / f"baselines_v2_{c}_{q}.json"
            bl = json.loads(pb.read_text(encoding="utf-8"))
            fontes[str(pb.relative_to(BASE)).replace("\\", "/")] = sha256_arquivo(pb)
            mlp, gnn, det = [], [], {}
            for s in SEEDS:
                lm, dm = le_run("mlpcf", c, s, q, fontes)
                lg, dg = le_run("c0c1cf", c, s, q, fontes)
                if identidade(dm) != identidade(dg):
                    problemas.append(
                        {"celula": f"{c}_{q}", "seed": s,
                         "mlp": {"rf_data_sha256": dm["dataset"]["rf_data_sha256"],
                                 "idx": [dm["particoes"][k]["idx_sha256_global"]
                                         for k in ("train", "val", "test")]},
                         "gnn": {"rf_data_sha256": dg["dataset"]["rf_data_sha256"],
                                 "idx": [dg["particoes"][k]["idx_sha256_global"]
                                         for k in ("train", "val", "test")]}})
                    sys.exit(f"ERRO: pareamento invalido em {c}_{q} s{s}; "
                             f"nada agregado (ver {lm} x {lg})")
                a = dm["selecao"]["test_no_melhor_ckpt"]["mae_rssi_db"]
                b = dg["selecao"]["test_no_melhor_ckpt"]["mae_rssi_db"]
                mlp.append(a)
                gnn.append(b)
                det[f"s{s}"] = {"mlp": a, "gnn": b, "delta": b - a,
                                "n_test": dm["particoes"]["test"]["n"]}
                delta_por_cel_seed[(c, s, q)] = b - a
            celulas[f"{c}_{q}"] = {
                "cidade": c, "quadrante": q,
                "frac_cobertura_test": bl["particoes"]["test"]["frac_cobertura"],
                "n_test": det["s42"]["n_test"],
                "mlp_mae_rssi_db_media": float(np.mean(mlp)),
                "gnn_mae_rssi_db_media": float(np.mean(gnn)),
                "mlp_dp": float(np.std(mlp, ddof=1)),
                "gnn_dp": float(np.std(gnn, ddof=1)),
                "delta_gnn_menos_mlp_db": float(np.mean(gnn) - np.mean(mlp)),
                "por_semente": det,
            }

    # the three pre-registered units of analysis
    nomes_cel = [f"{c}_{q}" for c in CIDADES for q in QUADS]
    d_celula = [celulas[k]["delta_gnn_menos_mlp_db"] for k in nomes_cel]

    cadeias = {}
    for c in CIDADES:
        for s in SEEDS:
            cadeias[f"{c}_s{s}"] = float(np.mean(
                [delta_por_cel_seed[(c, s, q)] for q in QUADS]))
    d_cadeia = [cadeias[f"{c}_s{s}"] for c in CIDADES for s in SEEDS]

    # unidade_cidade = 'city, averaged over chains' (frozen definition)
    cidades = {c: float(np.mean([cadeias[f"{c}_s{s}"] for s in SEEDS]))
               for c in CIDADES}
    d_cidade = [cidades[c] for c in CIDADES]

    cidade_por_celula = {c: float(np.mean(
        [celulas[f"{c}_{q}"]["delta_gnn_menos_mlp_db"] for q in QUADS]))
        for c in CIDADES}
    max_dif_cidade = max(abs(cidades[c] - cidade_por_celula[c])
                         for c in CIDADES)

    unidades = {
        "cidade_n4": {**estat(d_cidade, "cidade (n=4)"),
                      "valores_db": cidades,
                      "tost_margem_0_1": tost(d_cidade, MARGEM_TOST_DB)},
        "celula_n16": {**estat(d_celula, "celula (n=16)"),
                       "valores_db": {k: celulas[k]["delta_gnn_menos_mlp_db"]
                                      for k in nomes_cel},
                       "tost_margem_0_1": tost(d_celula, MARGEM_TOST_DB)},
        "cadeia_n20": {**estat(d_cadeia, "cadeia (n=20)"),
                       "valores_db": cadeias,
                       "tost_margem_0_1": tost(d_cadeia, MARGEM_TOST_DB)},
    }

    # continuity with the published number (Lins+Bauru, n=8)


    sub = [f"{c}_{q}" for c in ("lins", "bauru") for q in QUADS]
    d8 = [celulas[k]["delta_gnn_menos_mlp_db"] for k in sub]
    e8 = estat(d8, "celula lins+bauru (n=8) -- replica do publicado")
    continuidade = {
        "o_que_e": "recomputo do mesmo estimador sobre as 8 celulas de lins e "
                   "bauru, para conferir contra main.tex:546 e "
                   "ca_recomputo.json :: mlp_vs_gnn.unidade_celula_n8",
        "publicado_main_tex_546": {"delta_db": 0.011, "ic95_db": [-0.126, 0.148],
                                   "n": 8, "p": 0.86, "tost_p": 0.083},
        "recomputado": {"delta_db": e8["media_delta_db"],
                        "ic95_db": e8["ic95_db"], "n": e8["n"],
                        "p_bicaudal_t": e8["p_bicaudal_t"],
                        "tost_p": tost(d8, MARGEM_TOST_DB)["p_tost"]},
        "diferenca_absoluta": {
            "delta_db": abs(e8["media_delta_db"] - 0.011),
            "p": abs(e8["p_bicaudal_t"] - 0.86),
            "tost_p": abs(tost(d8, MARGEM_TOST_DB)["p_tost"] - 0.083)},
        "nota": "o publicado esta impresso com 2-3 casas; diferenca residual "
                "abaixo de 5e-4 e arredondamento de impressao",
    }

    # literal reading of the criterion
    melhor_gnn_cel = sum(1 for v in d_celula if v < 0)
    pior_gnn_cel = sum(1 for v in d_celula if v > 0)
    leitura = {
        "frase_do_criterio": criterio_t11["criterio_aceite"],
        "o_que_o_criterio_fixa": [
            "reportar o teste pareado em TRES unidades: cidade n=4, celula "
            "n=16, cadeia n=20 (feito acima)",
            "declarar, junto do numero de cidade, que com n=4 o p minimo nao "
            "parametrico e 0.125",
            "a frase do artigo e escrita para o resultado que sair: o criterio "
            "nao define limiar de aprovacao/reprovacao",
        ],
        "o_que_o_criterio_NAO_fixa": [
            "nenhum limiar de p, nenhuma margem de equivalencia e nenhuma "
            "direcao esperada; o TOST a 0,1 dB entra aqui apenas por "
            "continuidade com a frase ja publicada (main.tex:546), que o "
            "reporta, e esta rotulado como tal",
        ],
        "p_minimo_nao_parametrico_n4": 0.125,
        "declaracao_obrigatoria_do_criterio": (
            "com n=4 cidades, o menor p bicaudal alcancavel por teste nao "
            "parametrico e 0,125; nenhum resultado por cidade pode ser "
            "declarado significativo a 0,05 por essa via, independentemente "
            "dos dados"),
        "contagem_por_celula": {
            "gnn_melhor_delta_negativo": melhor_gnn_cel,
            "mlp_melhor_delta_positivo": pior_gnn_cel,
            "empate_exato": 16 - melhor_gnn_cel - pior_gnn_cel},
    }


    # RSSI < -150 dBm not imputed. The effect on the GNN arm was measured by inference
    # (+0.0787 to +0.0943 dB); not measured on the MLP arm.


    # Re-evaluates the three-unit conclusion under this worst case.

    pt1 = (BASE / "dados" / "contra_auditoria" / "rodada2_2026-09-13"
           / "t1-sorocaba-imputacao.json")
    t1 = json.loads(pt1.read_text(encoding="utf-8"))
    fontes[str(pt1.relative_to(BASE)).replace("\\", "/")] = sha256_arquivo(pt1)
    dmae_gnn_max = t1["resumo"]["dmae_rssi_db_max_abs_por_quadrante"]["Q3"]
    cota_mlp = t1["cota_superior_db"]["Q3"]["cota_superior_dmae_rssi_db"]
    desloc = dmae_gnn_max + cota_mlp
    d_cel_pior = [v + (desloc if k == "sorocaba_Q3" else 0.0)
                  for k, v in zip(nomes_cel, d_celula)]
    cad_pior = {}
    for c in CIDADES:
        for s in SEEDS:
            extra = desloc / len(QUADS) if c == "sorocaba" else 0.0
            cad_pior[f"{c}_s{s}"] = cadeias[f"{c}_s{s}"] + extra
    d_cad_pior = [cad_pior[f"{c}_s{s}"] for c in CIDADES for s in SEEDS]
    d_cid_pior = [float(np.mean([cad_pior[f"{c}_s{s}"] for s in SEEDS]))
                  for c in CIDADES]
    sens_t1 = {
        "o_que_e": "pior caso do defeito T1 (aberto, bloqueante) sobre ESTE "
                   "resultado; nao substitui o primario",
        "fonte": {"arquivo": str(pt1),
                  "sha256": fontes[str(pt1.relative_to(BASE)).replace("\\", "/")],
                  "campos": ["resumo.dmae_rssi_db_max_abs_por_quadrante.Q3",
                             "cota_superior_db.Q3.cota_superior_dmae_rssi_db"]},
        "celula_afetada": "sorocaba_Q3 (unica celula com nos afetados no "
                          "teste: 2232; Q1, Q2 e Q4 tem 0)",
        "dmae_rssi_gnn_medido_db": dmae_gnn_max,
        "cota_superior_dmae_rssi_mlp_db": cota_mlp,
        "deslocamento_pior_caso_no_delta_da_celula_db": desloc,
        "unidades_sob_pior_caso": {
            "cidade_n4": estat(d_cid_pior, "cidade (n=4) pior caso T1"),
            "celula_n16": estat(d_cel_pior, "celula (n=16) pior caso T1"),
            "cadeia_n20": estat(d_cad_pior, "cadeia (n=20) pior caso T1"),
        },
        "nota": "o desenho balanceado faz o pior caso deslocar a media das "
                "TRES unidades pelo mesmo valor (desloc/16 dB)",
    }

    sha_self = sha256_arquivo(Path(__file__))
    saida = {
        "artefato_tipo": "agregado_t11_mlp_vs_gnn_4cidades",
        "data": "2026-09-15",
        "status": "provisorio (pendente de contra-auditoria por quem nao "
                  "escreveu este script)",
        "agregador": {
            "proprio": True,
            "declaracao": "AGREGADOR PROPRIO desta analise; NAO e o agregador "
                          "da campanha. agregar_c1_multiseed_v2.py grava "
                          "dados/treinos_c1/agregado_c1v2_<cidade>_g10b2.json, "
                          "que e o agregado do modelo COM grafo, e o "
                          "sobrescreveria; este script grava em nome distinto "
                          "e nao importa nem executa aquele.",
            "saida": str(OUT),
            "nomes_que_este_script_NAO_toca": [
                "dados/treinos_c1/agregado_c1v2_<cidade>_g10b2.json",
                "dados/treinos_c1/agregado_mlp_vs_gnn_c1v2.json",
                "dados/consolidacao/claim_c1v2_estatistica.json"],
        },
        "pre_registro": {
            "caminho": str(PLANO),
            "sha256": fontes[str(PLANO.relative_to(BASE)).replace("\\", "/")],
            "entrada": "testes.T11_mlp_em_campinas_e_sorocaba",
            "acao": criterio_t11["acao"],
            "criterio_aceite": criterio_t11["criterio_aceite"],
            "definicoes_de_unidade_usadas":
                {k: plano["definicoes_congeladas"][k]
                 for k in ("unidade_celula", "unidade_cadeia", "unidade_cidade",
                           "mae_rssi_total")},
        },
        "metrica": {
            "campo": "selecao.test_no_melhor_ckpt.mae_rssi_db",
            "definicao": plano["definicoes_congeladas"]["mae_rssi_total"],
            "populacao": "todos os nos da particao de teste do split "
                         "espacialmente bloqueado, sentinela -110 dBm incluida",
            "por_que_esta": "e a metrica e a populacao da comparacao ja "
                            "publicada para lins e bauru (main.tex:546; "
                            "agregado_mlp_vs_gnn_c1v2.json; ca_recomputo.json "
                            ":: mlp_vs_gnn), verificadas em "
                            "conferencia_de_continuidade",
            "sinal": "delta = MAE(GNN) - MAE(MLP); negativo = GNN melhor",
        },
        "protocolo_de_pareamento": {
            "regra": "por celula e semente; rf_data_sha256 e os tres "
                     "idx_sha256_global (train/val/test) identicos entre "
                     "mlpcf e c0c1cf, senao o script aborta sem gravar",
            "problemas_encontrados": problemas,
            "aviso_colisao_idx": "idx_sha256_global e o hash dos indices no "
                                 "dataset da propria celula; celulas de mesma "
                                 "geometria colidem (Q1 com Q2, Q3 com Q4 em "
                                 "campinas e sorocaba). O hash de indices "
                                 "SOZINHO nao identifica a celula; por isso o "
                                 "pareamento exige tambem rf_data_sha256.",
        },
        "criterio_pre_registrado_aplicado": leitura,
        "unidades": unidades,
        "conferencia_de_continuidade": continuidade,
        "sensibilidade_declarada_T1_sorocaba_Q3": sens_t1,
        "celulas": celulas,
        "controle_de_definicao_de_cidade": {
            "media_das_cadeias_db": cidades,
            "media_das_celulas_db": cidade_por_celula,
            "max_diferenca_absoluta_db": max_dif_cidade,
            "nota": "as duas leituras coincidem porque o desenho e balanceado "
                    "(5 sementes x 4 quadrantes em toda cidade)",
        },
        "limites_declarados": [
            "as 16 celulas nao sao independentes: Q2-Q4 herdam o checkpoint "
            "do Q1 da mesma semente (cadeia de transferencia) e os tiles sao "
            "adjacentes; a unidade cadeia (n=20) e a unidade cidade (n=4) "
            "existem exatamente para tornar isso visivel",
            "o buffer de 2 km e menor que o alcance de autocorrelacao medido "
            "(20-30 km, T7 do mesmo pre-registro): o delta pareado e robusto a "
            "isso (os dois bracos veem o mesmo split), mas o nivel absoluto de "
            "MAE nao e",
            "nenhum numero aqui e citavel antes de contra-auditoria",
        ],
        "ambiente": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "executavel": sys.executable,
            "dispositivo": "CPU (so leitura de JSON; nenhum .pt carregado)",
        },
        "script": str(Path(__file__)),
        "script_sha256": sha_self,
        "fontes_sha256": fontes,
        "n_fontes": len(fontes),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    print(f"OK {OUT}")
    for chave in ("cidade_n4", "celula_n16", "cadeia_n20"):
        u = unidades[chave]
        print(f"{chave}: n={u['n']} delta={u['media_delta_db']:+.4f} dB "
              f"IC95 [{u['ic95_db'][0]:+.4f}; {u['ic95_db'][1]:+.4f}] "
              f"p_t={u['p_bicaudal_t']:.4f} wilcoxon={u['wilcoxon_p_bicaudal']} "
              f"sinal={u['teste_do_sinal_p_bicaudal']} "
              f"piso_np={u['p_minimo_nao_parametrico_possivel']:.6g} "
              f"tost={u['tost_margem_0_1']['p_tost']:.4f}")
    print("celulas GNN melhor / MLP melhor:",
          leitura["contagem_por_celula"])
    print("continuidade n=8:", continuidade["recomputado"],
          "diff:", continuidade["diferenca_absoluta"])
    print("pior caso T1 (desloc %.4f dB em sorocaba_Q3):" % desloc)
    for chave in ("cidade_n4", "celula_n16", "cadeia_n20"):
        u = sens_t1["unidades_sob_pior_caso"][chave]
        print(f"  {chave}: delta={u['media_delta_db']:+.4f} dB "
              f"IC95 [{u['ic95_db'][0]:+.4f}; {u['ic95_db'][1]:+.4f}] "
              f"p_t={u['p_bicaudal_t']:.4f}")
    print("fontes hasheadas:", len(fontes))


if __name__ == "__main__":
    main()
