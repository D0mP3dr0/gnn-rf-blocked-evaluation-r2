
"""Statistical-claim aggregation (bootstrap CI, TOST) for the capacity-matched control comparison."""
import json
import hashlib
import math
import sys
from pathlib import Path
from statistics import mean, stdev

BASE = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access"
            r"\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO")
TREINOS = BASE / "treinos"
OUT_DIR = BASE / "dados" / "consolidacao"
OUT = OUT_DIR / "claim_c1v2_estatistica.json"

CIDADES = ["lins", "bauru", "campinas", "sorocaba"]
SEEDS = [42, 43, 44, 45, 46]
QUADS = ["Q1", "Q2", "Q3", "Q4"]


T975 = {3: 3.182446, 15: 2.131450}


def carrega_runs():
    runs = {}
    faltando = []
    for c in CIDADES:
        for s in SEEDS:
            for q in QUADS:
                label = f"c0c1cf_{c}_s{s}_{q}_g10b2"
                p = TREINOS / label / f"run_{label}.json"
                if not p.exists():
                    faltando.append(label)
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    runs[(c, s, q)] = json.load(f)
    if faltando:
        sys.exit(f"ERRO: runs ausentes: {faltando}")
    return runs


def ic(valores, t):
    m = mean(valores)
    sd = stdev(valores)
    ep = sd / math.sqrt(len(valores))
    return m, sd, [m - t * ep, m + t * ep]


def icc_oneway(grupos):
    """ICC(1) por ANOVA one-way; grupos: dict chave -> lista de replicatas."""
    k = len(grupos)
    ns = [len(v) for v in grupos.values()]
    n_h = mean(ns)
    gm = mean([x for v in grupos.values() for x in v])
    ss_between = sum(len(v) * (mean(v) - gm) ** 2 for v in grupos.values())
    ss_within = sum((x - mean(v)) ** 2 for v in grupos.values() for x in v)
    df_b = k - 1
    df_w = sum(ns) - k
    ms_b = ss_between / df_b
    ms_w = ss_within / df_w
    icc = (ms_b - ms_w) / (ms_b + (n_h - 1) * ms_w)
    return {"icc1": icc, "f": ms_b / ms_w, "ms_between": ms_b, "ms_within": ms_w,
            "dp_entre_grupos": math.sqrt(max(0.0, (ms_b - ms_w) / n_h)),
            "dp_dentro": math.sqrt(ms_w)}


def main():
    runs = carrega_runs()


    mae = {k: r["selecao"]["test_no_melhor_ckpt"]["mae_rssi_db"]
           for k, r in runs.items()}

    celulas = {}
    for c in CIDADES:
        for q in QUADS:
            celulas[f"{c}_{q}"] = [mae[(c, s, q)] for s in SEEDS]
    medias_celula = {k: mean(v) for k, v in celulas.items()}

    m16, sd16, ic16 = ic(list(medias_celula.values()), T975[15])
    medias_cidade = {c: mean([medias_celula[f"{c}_{q}"] for q in QUADS])
                     for c in CIDADES}
    m4, sd4, ic4 = ic(list(medias_cidade.values()), T975[3])
    dec = icc_oneway(celulas)

    # sensitivity to the selection criterion


    # only answers 'what if selection had been by PL instead?'
    sens = {}
    melhora = 0
    tot_of, tot_alt = [], []
    for k, r in runs.items():
        label = "%s_s%s_%s" % k
        epocas = r["epocas"]
        of = min(epocas, key=lambda e: e["val"]["mae_rssi_db"])
        alt = min(epocas, key=lambda e: e["val"]["mae_pl_db"])
        pl_of = of["test"]["mae_pl_db"]
        pl_alt = alt["test"]["mae_pl_db"]
        rssi_of = of["test"]["mae_rssi_db"]
        rssi_alt = alt["test"]["mae_rssi_db"]
        sens[label] = {
            "epoca_oficial": of["epoch"], "epoca_alt_pl": alt["epoch"],
            "test_mae_pl_oficial": pl_of, "test_mae_pl_alt": pl_alt,
            "test_mae_rssi_oficial": rssi_of, "test_mae_rssi_alt": rssi_alt,
        }
        tot_of.append(pl_of)
        tot_alt.append(pl_alt)
        if pl_alt < pl_of:
            melhora += 1

    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = {
        "artefato_tipo": "claim_estatistica_c1v2",
        "script": str(Path(__file__)),
        "script_sha256": sha,
        "fonte": "selecao.test_no_melhor_ckpt.mae_rssi_db dos 80 run_c0c1cf_*.json",
        "status": "provisorio (pendente de contra-auditoria por quem nao escreveu este script)",
        "n_runs": len(runs),
        "claim": {
            "definicao": "MAE de RSSI (dB) no teste espacialmente bloqueado, "
                         "checkpoint escolhido por val, teste tocado 1x",
            "unidade_celula": {"n": 16, "media_db": m16, "dp_db": sd16,
                               "ic95_db": ic16, "t": T975[15]},
            "unidade_cidade": {"n": 4, "media_db": m4, "dp_db": sd4,
                               "ic95_db": ic4, "t": T975[3]},
            "medias_por_celula_db": medias_celula,
            "medias_por_cidade_db": medias_cidade,
            "decomposicao_celula_x_seed": dec,
            "nota": "80 runs nao sao independentes (ICC alto + cadeia de "
                    "transferencia Q1->Q4); a unidade de inferencia e a "
                    "celula (ou cidade), nunca o run",
        },
        "sensibilidade_criterio_selecao": {
            "criterio_oficial": "min val.mae_rssi_db",
            "criterio_alternativo": "min val.mae_pl_db",
            "runs_em_que_alt_melhora_pl": melhora,
            "mae_pl_medio_oficial_db": mean(tot_of),
            "mae_pl_medio_alt_db": mean(tot_alt),
            "por_run": sens,
            "nota": "teste por epoca e diagnostico; a selecao oficial nunca "
                    "viu o teste. Esta tabela responde 'e se o criterio "
                    "fosse outro' sem alterar nenhum resultado publicado",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"OK {OUT}")
    print(f"celula n=16: {m16:.4f} dB IC95 [{ic16[0]:.4f}; {ic16[1]:.4f}]")
    print(f"cidade n=4 : {m4:.4f} dB IC95 [{ic4[0]:.4f}; {ic4[1]:.4f}]")
    print(f"ICC(1) = {dec['icc1']:.4f}  F = {dec['f']:.2f}")
    print(f"E3: alt melhora PL em {melhora}/80; medio {mean(tot_of):.3f} -> {mean(tot_alt):.3f} dB")


if __name__ == "__main__":
    main()
