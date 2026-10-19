"""FSPL-floor admissibility under the spatially blocked partition, plus trainable-parameter recount per model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_EVID = Path(__file__).resolve().parents[1]
TREINOS = BASE_EVID / "treinos"
TREINOS_C1 = BASE_EVID / "dados" / "treinos_c1"
SAIDA = BASE_EVID / "dados" / "consolidacao"
PREREG = BASE_EVID / "dados" / "planos" / "preregistro_admissibilidade_e_params_2026-09-15.json"
V2 = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2")
BASE_COLAB = V2 / "baseline_colab_" / "logs"

CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
SEEDS = [42, 43, 44, 45, 46]
QUADS = ["Q1", "Q2", "Q3", "Q4"]
DIAG_FREQ_ESPERADA = 1800.0

FALHAS: list[str] = []


def morre(msg: str) -> None:
    """Guarda que aborta: nenhuma ausencia vira default silencioso."""
    raise SystemExit(f"[ABORTA] {msg}")


def sha256(p: Path, buf: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def carimbo(p: Path) -> dict:
    st = p.stat()
    return {
        "caminho": str(p),
        "bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(p),
    }


def exigir(d: dict, caminho: list[str], onde: str):
    """Indexacao com erro ruidoso; substitui d.get(k, default)."""
    cur = d
    for k in caminho:
        if not isinstance(cur, dict) or k not in cur:
            morre(f"chave ausente {'.'.join(caminho)} em {onde}")
        cur = cur[k]
    return cur


def media(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def ic95_t(xs: list[float]) -> dict:
    """IC 95% por t de Student; n pequeno, unidade = celula."""
    n = len(xs)
    if n < 2:
        return {"n": n, "media": media(xs) if xs else None, "ic95": None,
                "nota": "n<2, IC nao definido"}
    m = media(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))

    TAB = {3: 3.182, 4: 2.776, 15: 2.131, 19: 2.093, 79: 1.990}
    gl = n - 1
    if gl not in TAB:
        morre(f"t_0.975 nao tabelado para gl={gl}; nao inventar valor")
    h = TAB[gl] * s / math.sqrt(n)
    return {"n": n, "media": m, "desvio_amostral": s, "gl": gl,
            "t_0975": TAB[gl], "ic95": [m - h, m + h]}


def item1() -> dict:
    esperados = [(c, s, q) for c in CIDADES for s in SEEDS for q in QUADS]
    corridas = []
    fontes = []
    ausentes = []

    for cidade, seed, q in esperados:
        rot = f"c0c1cf_{cidade}_s{seed}_{q}_g10b2"
        p = TREINOS / rot / f"run_{rot}.json"
        if not p.exists():
            ausentes.append(str(p))
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        onde = str(p)


        freq = exigir(d, ["config", "diag_freq_mhz"], onde)
        if float(freq) != DIAG_FREQ_ESPERADA:
            morre(f"diag_freq_mhz={freq} != {DIAG_FREQ_ESPERADA} em {onde}; "
                  "piso de FSPL de frequencias diferentes nao agrega")
        if exigir(d, ["selecao", "test_usado_na_selecao"], onde) is not False:
            morre(f"test_usado_na_selecao != False em {onde}")
        if exigir(d, ["config", "buffer_km"], onde) != 2.0:
            morre(f"buffer_km != 2.0 em {onde}")

        compl = exigir(d, ["selecao", "test_no_melhor_ckpt", "diag",
                           "physics_fspl_compliance"], onde)
        if compl is None or not isinstance(compl, (int, float)) or math.isnan(compl):
            morre(f"physics_fspl_compliance nao numerico em {onde}")
        if not (0.0 <= float(compl) <= 1.0):
            morre(f"physics_fspl_compliance={compl} fora de [0,1] em {onde}")

        n_aval = exigir(d, ["selecao", "test_no_melhor_ckpt", "n_nos_avaliados"], onde)
        n_test = exigir(d, ["particoes", "test", "n"], onde)
        if int(n_aval) != int(n_test):
            morre(f"n_nos_avaliados={n_aval} != particoes.test.n={n_test} em {onde}")

        compl_val = exigir(d, ["selecao", "val_no_melhor_ckpt", "diag",
                               "physics_fspl_compliance"], onde)

        corridas.append({
            "run_label": exigir(d, ["run_label"], onde),
            "cidade": cidade, "seed": seed, "quadrante": q,
            "celula": f"{cidade}_{q}",
            "physics_fspl_compliance_test": float(compl),
            "physics_fspl_compliance_val": float(compl_val),
            "n_nos_test": int(n_test),
            "idx_sha256_test": exigir(d, ["particoes", "test", "idx_sha256_global"], onde),
            "melhor_epoca": exigir(d, ["selecao", "melhor_epoca"], onde),
            "mae_rssi_db_test": exigir(d, ["selecao", "test_no_melhor_ckpt", "mae_rssi_db"], onde),
            "status_run": exigir(d, ["status"], onde),
            "run_json_sha256": sha256(p),
        })
        fontes.append(carimbo(p))

    if ausentes:
        morre(f"{len(ausentes)} corridas esperadas ausentes: {ausentes[:5]}")
    if len(corridas) != 80:
        morre(f"N={len(corridas)} != 80 corridas esperadas")


    confronto_copia = {"comparadas": 0, "divergentes": [], "ausentes_na_copia": []}
    for r in corridas:
        pc = TREINOS_C1 / f"run_{r['run_label']}.json"
        if not pc.exists():
            confronto_copia["ausentes_na_copia"].append(str(pc))
            continue
        confronto_copia["comparadas"] += 1
        dc = json.loads(pc.read_text(encoding="utf-8"))
        vc = exigir(dc, ["selecao", "test_no_melhor_ckpt", "diag",
                         "physics_fspl_compliance"], str(pc))
        if float(vc) != r["physics_fspl_compliance_test"]:
            confronto_copia["divergentes"].append(
                {"run": r["run_label"], "treinos": r["physics_fspl_compliance_test"],
                 "dados_treinos_c1": float(vc)})


    vals = [r["physics_fspl_compliance_test"] for r in corridas]
    por_celula = {}
    for cidade in CIDADES:
        for q in QUADS:
            cel = f"{cidade}_{q}"
            xs = [r["physics_fspl_compliance_test"] for r in corridas if r["celula"] == cel]
            if len(xs) != 5:
                morre(f"celula {cel} com {len(xs)} sementes, esperado 5")
            por_celula[cel] = {"n_sementes": 5, "media": media(xs),
                               "min": min(xs), "max": max(xs)}
    por_semente = {}
    for s in SEEDS:
        xs = [r["physics_fspl_compliance_test"] for r in corridas if r["seed"] == s]
        if len(xs) != 16:
            morre(f"semente {s} com {len(xs)} celulas, esperado 16")
        por_semente[str(s)] = {"n_celulas": 16, "media": media(xs),
                               "min": min(xs), "max": max(xs)}
    por_cidade = {}
    for cidade in CIDADES:
        xs = [r["physics_fspl_compliance_test"] for r in corridas if r["cidade"] == cidade]
        por_cidade[cidade] = {"n_corridas": len(xs), "media": media(xs),
                              "min": min(xs), "max": max(xs)}

    medias_celula = [por_celula[c]["media"] for c in por_celula]
    soma_n = sum(r["n_nos_test"] for r in corridas)
    ponderado = sum(r["physics_fspl_compliance_test"] * r["n_nos_test"]
                    for r in corridas) / soma_n

    n_exatamente_1 = sum(1 for v in vals if v == 1.0)
    pior = min(corridas, key=lambda r: r["physics_fspl_compliance_test"])

    alarme_constante = None
    if n_exatamente_1 == 80:
        alarme_constante = ("TODAS as 80 corridas valem exatamente 1.0: testemunha que "
                            "nao varia e alarme de guarda desarmada, nao confirmacao. "
                            "Verificar se has_dist cobre a particao de teste.")

    return {
        "artefato_tipo": "medida_admissibilidade_bloqueada",
        "status": "provisorio",
        "citavel": False,
        "nota_citabilidade": "numero nao citavel antes de contra-auditoria por quem nao escreveu o script",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "campo_medido": "selecao.test_no_melhor_ckpt.diag.physics_fspl_compliance",
        "definicao": ("rf_diagnostic_metrics.RFDiagnosticMetrics.compute() "
                      "(GNN_RF_V2/03_training/rf_diagnostic_metrics.py:277-280): "
                      "frac(pl_pred >= FSPL(dist_nearest_m, 1800 MHz)) sobre os nos com "
                      "dist_nearest_m valida da particao de TESTE bloqueada, avaliada uma "
                      "unica vez no checkpoint de menor MAE de RSSI em VAL"),
        "protocolo": "c0c1cf (espacialmente bloqueado, grid 10 km, buffer 2 km), 4 cidades x 5 sementes x 4 quadrantes",
        "n_corridas": len(corridas),
        "n_nos_test_somados": soma_n,
        "agregados": {
            "global_media_por_corrida": media(vals),
            "global_ponderado_por_nos": ponderado,
            "ic95_unidade_celula": ic95_t(medias_celula),
            "min_global": min(vals), "max_global": max(vals),
            "n_corridas_exatamente_1.0": n_exatamente_1,
            "pior_corrida": {"run_label": pior["run_label"],
                             "valor": pior["physics_fspl_compliance_test"]},
        },
        "por_celula": por_celula,
        "por_semente": por_semente,
        "por_cidade": por_cidade,
        "corridas": corridas,
        "alarme_testemunha_constante": alarme_constante,
        "confronto_copia_dados_treinos_c1": confronto_copia,
        "nao_equivalencia_com_o_publicado": {
            "publicado": "REVISAO_R3/main.tex:491-508 (tab:physics_compliance)",
            "estimador_publicado": ("definicao SECUNDARIA de fase0_eval.py: piso por ENLACE, "
                                    "min sobre enlaces incidentes de 20log10(d_km)+20log10(f_MHz)+32.44, "
                                    "f = frequencia de cada enlace"),
            "estimador_aqui": "definicao do PROJETO: piso unico por no via dist_nearest_m a 1800 MHz",
            "populacao_publicada": "1.225.847 nos de Bauru Q2 com enlace explicito, modelo treinado sobre todos os nos (in-sample)",
            "populacao_aqui": "nos da particao de teste bloqueada, fora do treino",
            "linhagem_publicada": "checkpoints seed-42 da campanha v20, campo de referencia antes da correcao",
            "linhagem_aqui": "checkpoints c0c1cf, campo de referencia corrigido",
            "conclusao": "os dois numeros NAO sao intercambiaveis; trocar um pelo outro exige reescrever legenda, populacao e texto",
        },
        "fontes": fontes,
        "preregistro": carimbo(PREREG),
        "script": carimbo(Path(__file__).resolve()),
        "ambiente": {"python": sys.version.split()[0], "executavel": sys.executable,
                     "cuda_usada": False},
    }


def item1_campo_referencia() -> dict:
    """Checks whether the per-node target of the blocked partition exists in a recorded artifact."""
    candidatos = {
        "f10_erro_espacial": sorted((BASE_EVID / "_campanha_2026-09-13" / "f10_erro_espacial").glob("*.npz")),
        "t14_mc_dropout": sorted((BASE_EVID / "_campanha_2026-09-13" / "t14_mc_dropout").glob("*.npz")),
        "dados_spatial_error": sorted((BASE_EVID / "dados" / "spatial_error").glob("*")),
    }
    inventario = {}
    tem_pl_alvo_por_no = False
    tem_dist_por_no = False
    try:
        import numpy as np
    except Exception as e:
        morre(f"numpy indisponivel: {e}")
    for grupo, arquivos in candidatos.items():
        inventario[grupo] = []
        for a in arquivos[:4]:
            if a.suffix == ".npz":
                z = np.load(a)
                chaves = {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}
                inventario[grupo].append({"arquivo": str(a), "chaves": chaves})
                if any(k in z.files for k in ("pl_alvo", "y_pl", "path_loss_alvo")):
                    tem_pl_alvo_por_no = True
                if any(k in z.files for k in ("dist_nearest_m", "dist_m", "dist_nearest")):
                    tem_dist_por_no = True
            else:
                inventario[grupo].append({"arquivo": str(a), "chaves": "nao .npz"})
    possivel = tem_pl_alvo_por_no and tem_dist_por_no
    return {
        "recomponivel": possivel,
        "necessario": ["path loss ALVO por no da particao de teste bloqueada",
                       "dist_nearest_m por no (ou o piso ja calculado) da mesma particao"],
        "encontrado": {"pl_alvo_por_no": tem_pl_alvo_por_no,
                       "dist_nearest_por_no": tem_dist_por_no},
        "inventario_lido": inventario,
        "veredito": ("PARA: a linha do campo de referencia sob protocolo bloqueado NAO e "
                     "recomponivel a partir de artefato ja gravado. Os .npz das campanhas "
                     "gravam erro de RSSI, posicao e mascara de validade, nao o path loss "
                     "alvo nem a distancia por no. Obte-la exigiria abrir "
                     "transfer_dataset_{cidade}_v19_{Q}_enriched_cftudo.pt (~28 GB por tile) "
                     "e reaplicar o split bloqueado - custo de I/O incompativel com a "
                     "restricao desta rodada e fora do escopo 'de graca'.")
        if not possivel else "prosseguir",
        "valor_publicado_da_linha": {
            "valor": 0.9999959211875544,
            "artefato": str(BASE_EVID / "dados" / "fase0" / "fase0_metrics_bauru_s42_Q2.json"),
            "campo": "modelos.*.todos_os_nos.alvo_fspl_floor_fraction",
            "linhagem": "campanha v20, Bauru Q2 inteiro, piso por enlace, campo antes da correcao",
        },
    }


BUFFER_SUFIXOS = ("running_mean", "running_var", "num_batches_tracked")
CHAVES_SD = ("model_state_dict", "state_dict", "model")


def _conta_um(sd: dict, p: Path) -> dict:
    total = 0
    buffers = 0
    chaves_buffer = []
    for k, v in sd.items():
        if not hasattr(v, "numel"):
            morre(f"valor nao tensor em {p}, chave {k}")
        n = int(v.numel())
        total += n
        if k.endswith(BUFFER_SUFIXOS):
            buffers += n
            chaves_buffer.append({"chave": k, "numel": n})
    return {"n_tensores": len(sd), "total_state_dict": total,
            "elementos_em_buffers_bn": buffers, "treinaveis": total - buffers,
            "chaves_buffer_excluidas": chaves_buffer}


def conta_state_dict(p: Path) -> dict:
    """
    Counts parameters per component. A GAN checkpoint has a generator and a
    discriminator; both receive gradients during training, so both are
    reported separately, never summed silently or chosen arbitrarily.
    """
    import torch
    obj = torch.load(p, map_location="cpu", weights_only=False)

    def eh_sd(v) -> bool:
        return isinstance(v, dict) and len(v) > 0 and all(hasattr(x, "numel") for x in v.values())

    componentes: dict[str, dict] = {}
    if isinstance(obj, dict):
        for k in CHAVES_SD:
            if k in obj and eh_sd(obj[k]):
                componentes[k] = obj[k]
                break
        if not componentes:
            for k, v in obj.items():
                if k.endswith("_state_dict") and not k.startswith("optimizer") \
                        and "optimizer" not in k and eh_sd(v):
                    componentes[k] = v
        if not componentes and eh_sd(obj):
            componentes["<proprio dict de tensores>"] = obj
    if not componentes:
        morre(f"state_dict nao localizado em {p} (chaves: "
              f"{list(obj.keys())[:10] if isinstance(obj, dict) else type(obj)})")

    por_componente = {k: _conta_um(v, p) for k, v in componentes.items()}


    chaves_tt = [f"{c}:{k}" for c, sd in componentes.items() for k in sd
                 if k.startswith("tt_msg")]
    total = sum(c["total_state_dict"] for c in por_componente.values())
    buffers = sum(c["elementos_em_buffers_bn"] for c in por_componente.values())
    chaves_buffer = [b for c in por_componente.values() for b in c["chaves_buffer_excluidas"]]
    n_tensores = sum(c["n_tensores"] for c in por_componente.values())
    origem_chave = "+".join(sorted(por_componente))

    meta = {}
    if isinstance(obj, dict):
        for k in ("epoch", "timestamp"):
            if k in obj:
                meta[k] = str(obj[k])
        for k in ("run_cfg", "args", "config"):
            if k in obj and isinstance(obj[k], dict):
                meta[k] = {kk: obj[k][kk] for kk in
                           ("run_label", "version", "seed", "epochs", "lr",
                            "hidden_dim", "topk_links", "k_antenna", "k_terrain")
                           if kk in obj[k]}
        if "metrics" in obj and isinstance(obj["metrics"], dict):
            meta["metrics"] = {k: float(v) for k, v in obj["metrics"].items()
                               if isinstance(v, (int, float))}
    del obj, componentes
    return {
        "chave_do_state_dict": origem_chave,
        "tem_tt_msg": bool(chaves_tt),
        "chaves_tt_msg": chaves_tt,
        "n_componentes": len(por_componente),
        "por_componente": {k: {kk: vv for kk, vv in c.items()
                               if kk != "chaves_buffer_excluidas"}
                           for k, c in por_componente.items()},
        "n_tensores_no_state_dict": n_tensores,
        "total_state_dict": total,
        "elementos_em_buffers_bn": buffers,
        "treinaveis": total - buffers,
        "treinaveis_por_componente": {k: c["treinaveis"] for k, c in por_componente.items()},
        "chaves_buffer_excluidas": chaves_buffer,
        "regra_exclusao": f"chave terminada em {BUFFER_SUFIXOS}",
        "metadados_do_checkpoint": meta,
    }


def custo_irmao(ck: Path) -> dict | None:
    """Reads computational_cost.json from the same run (sibling of the checkpoints directory)."""
    cand = ck.parent.parent / "computational_cost.json"
    if not cand.exists():
        return None
    d = json.loads(cand.read_text(encoding="utf-8"))
    return {"arquivo": str(cand), "sha256": sha256(cand), "conteudo": d}


def item2() -> dict:
    conjuntos = {
        "version_20": BASE_COLAB / "version_20",
        "version_20.1": BASE_COLAB / "version_20.1" / "logs" / "version_20",
        "version_20.2": BASE_COLAB / "version_20.2",
        "version_20.3": BASE_COLAB / "version_20.3" / "version_20",
    }
    modelos = ["radiounet", "rem_gnn", "radiogat", "rme_gan"]
    registros = []
    ausencias = []

    for conj, raiz in conjuntos.items():
        for m in modelos:
            for q in QUADS:
                ck = raiz / f"{m}_bauru_s42_{q}" / "checkpoints" / "checkpoint_best.pt"
                if not ck.exists():
                    ausencias.append({"conjunto": conj, "modelo": m, "quadrante": q,
                                      "caminho": str(ck)})
                    continue
                print(f"  [ck] {conj} {m} {q}", flush=True)
                r = {"modelo": m, "conjunto_de_treinos": conj, "quadrante": q,
                     "cidade": "bauru", "seed": 42}
                r.update(carimbo(ck))
                r.update(conta_state_dict(ck))
                r["computational_cost_irmao"] = custo_irmao(ck)
                registros.append(r)


    gnn_alvos = [
        ("v20_multiseed", V2 / "logs" / "version_20" / "multiseed_bauru_s42_Q1" / "checkpoints" / "checkpoint_best.pt", "Q1"),
        ("v20_multiseed", V2 / "logs" / "version_20" / "multiseed_bauru_s42_Q4" / "checkpoints" / "checkpoint_best.pt", "Q4"),
        ("c0c1cf_bloqueada", TREINOS / "c0c1cf_bauru_s42_Q1_g10b2" / "checkpoints" / "checkpoint_best.pt", "Q1"),
        ("c0c1cf_bloqueada", TREINOS / "c0c1cf_bauru_s42_Q4_g10b2" / "checkpoints" / "checkpoint_best.pt", "Q4"),
    ]
    for conj, ck, q in gnn_alvos:
        if not ck.exists():
            ausencias.append({"conjunto": conj, "modelo": "gnn_rf", "quadrante": q,
                              "caminho": str(ck)})
            continue
        print(f"  [ck] {conj} gnn_rf {q}", flush=True)
        r = {"modelo": "gnn_rf", "conjunto_de_treinos": conj, "quadrante": q,
             "cidade": "bauru", "seed": 42}
        r.update(carimbo(ck))
        r.update(conta_state_dict(ck))
        r["computational_cost_irmao"] = custo_irmao(ck)

        if conj == "c0c1cf_bloqueada":
            rj = ck.parent.parent / f"run_c0c1cf_bauru_s42_{q}_g10b2.json"
            if rj.exists():
                d = json.loads(rj.read_text(encoding="utf-8"))
                r["tempo_do_run_json_s"] = exigir(d, ["custo", "tempo_total_s"], str(rj))
                r["n_params_declarado_no_run_json"] = exigir(
                    d, ["modelo", "n_params_treinaveis"], str(rj))
                r["run_json"] = carimbo(rj)
        registros.append(r)


    PUBLICADO = {"radiounet": 265602, "rem_gnn": 72450, "radiogat": 301826,
                 "rme_gan": 72194, "gnn_rf": 1812515}
    TEMPO_PUBLICADO = {"radiounet": {"Q1": 444, "Q4": 740},
                       "rem_gnn": {"Q1": 383, "Q4": 1080},
                       "radiogat": {"Q1": 2347, "Q4": 1511},
                       "rme_gan": {"Q1": 879, "Q4": 762},
                       "gnn_rf": {"Q1": 3987, "Q4": 2323}}

    confronto = {}
    for m in PUBLICADO:
        regs = [r for r in registros if r["modelo"] == m]

        def bate(r, alvo=PUBLICADO[m]):
            """The published parameter count may correspond to the total or to a single component (GAN case)."""
            if r["treinaveis"] == alvo:
                return "todos_os_componentes"
            for k, v in r["treinaveis_por_componente"].items():
                if v == alvo:
                    return k
            return None

        for r in regs:
            r["componente_que_bate_com_o_publicado"] = bate(r)
        casam_params = [r for r in regs if r["componente_que_bate_com_o_publicado"]]
        casam_tempo = []
        for r in regs:
            if r["quadrante"] not in TEMPO_PUBLICADO[m]:

                r["tempo_publicado_s"] = None
                r["tempo_casa_com_publicado"] = None
                r["nota_tempo"] = "quadrante sem tempo publicado em tab:cost"
                continue
            t_pub = TEMPO_PUBLICADO[m][r["quadrante"]]
            t_obs = None
            if r.get("computational_cost_irmao"):
                t_obs = r["computational_cost_irmao"]["conteudo"].get("total_training_s")
            if t_obs is None and "tempo_do_run_json_s" in r:
                t_obs = r["tempo_do_run_json_s"]
            r["tempo_observado_s"] = t_obs
            r["tempo_publicado_s"] = t_pub
            r["tempo_casa_com_publicado"] = (
                t_obs is not None and abs(float(t_obs) - float(t_pub)) <= 0.5)
            if r["tempo_casa_com_publicado"]:
                casam_tempo.append(r)
        conj_params = sorted({r["conjunto_de_treinos"] for r in casam_params})
        conj_tempo = sorted({r["conjunto_de_treinos"] for r in casam_tempo})
        tam_pub_mb = round(4 * PUBLICADO[m] / (1024 ** 2), 2)
        confronto[m] = {
            "params_publicado": PUBLICADO[m],
            "contagens_observadas": sorted({r["treinaveis"] for r in regs}),
            "contagens_por_componente_observadas": sorted(
                {v for r in regs for v in r["treinaveis_por_componente"].values()}),
            "conjuntos_cujo_param_bate_com_o_publicado": conj_params,
            "checkpoints_cujo_param_bate": [
                {"conjunto": r["conjunto_de_treinos"], "quadrante": r["quadrante"],
                 "caminho": r["caminho"], "sha256": r["sha256"],
                 "treinaveis": r["treinaveis"],
                 "componente_que_bate": r["componente_que_bate_com_o_publicado"],
                 "treinaveis_por_componente": r["treinaveis_por_componente"]}
                for r in casam_params],
            "conjuntos_cujo_tempo_bate_com_o_publicado": conj_tempo,
            "checkpoints_cujo_tempo_bate": [
                {"conjunto": r["conjunto_de_treinos"], "quadrante": r["quadrante"],
                 "tempo_observado_s": r["tempo_observado_s"],
                 "tempo_publicado_s": r["tempo_publicado_s"],
                 "treinaveis_nesse_conjunto": r["treinaveis"],
                 "caminho": r["caminho"], "sha256": r["sha256"]} for r in casam_tempo],
            "divergencia_de_conjunto": (
                bool(conj_params) and bool(conj_tempo) and
                not set(conj_params) & set(conj_tempo)),
            "tamanho_mb_publicado_recalculado": tam_pub_mb,
        }


    linhagens = {}
    for r in registros:
        chave = (r["modelo"], r["conjunto_de_treinos"])
        alvo = linhagens.setdefault(
            f"{r['modelo']}|{r['conjunto_de_treinos']}",
            {"quadrantes": [], "treinaveis": set(), "tem_tt_msg": set(),
             "timestamps": set()})
        alvo["quadrantes"].append(r["quadrante"])
        alvo["treinaveis"].add(r["treinaveis"])
        alvo["tem_tt_msg"].add(r["tem_tt_msg"])
        ts = r["metadados_do_checkpoint"].get("timestamp")
        if ts:
            alvo["timestamps"].add(ts[:10])
    for k, v in linhagens.items():
        v["quadrantes"] = sorted(v["quadrantes"])
        v["treinaveis"] = sorted(v["treinaveis"])
        v["tem_tt_msg"] = sorted(str(x) for x in v["tem_tt_msg"])
        v["datas_de_treino"] = sorted(v.pop("timestamps"))

    return {
        "artefato_tipo": "recontagem_parametros_state_dict",
        "status": "provisorio",
        "citavel": False,
        "nota_citabilidade": "numero nao citavel antes de contra-auditoria por quem nao escreveu o script",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "definicao": ("torch.load(map_location='cpu', weights_only=False); soma de numel() "
                      "sobre o state_dict; treinaveis = total menos elementos de buffer de "
                      f"BatchNorm (chaves terminadas em {BUFFER_SUFIXOS}). Nenhum modelo foi "
                      "instanciado: a contagem descreve o checkpoint, nao a classe de hoje"),
        "n_checkpoints_lidos": len(registros),
        "ausencias_declaradas": ausencias,
        "checkpoints": registros,
        "linhagens_de_arquitetura": linhagens,
        "nota_tt_msg": ("tt_msg.* = passo de mensagem terrain->terrain acrescentado ao "
                        "codigo DEPOIS do primeiro treino (fase0_eval.load_colab_state, "
                        "dados/scripts_congelados/fase0_eval.py:168-182, carrega os "
                        "checkpoints pre-refactor com strict=False). Presenca ou ausencia "
                        "dessas chaves separa as duas linhagens de checkpoint e explica "
                        "toda a diferenca de contagem dentro do mesmo modelo"),
        "confronto_com_o_publicado": confronto,
        "publicado": {"fonte": "REVISAO_R3/main.tex:513-528 (tab:baseline_arch) e main.tex:644-660 (tab:cost)",
                      "params": PUBLICADO, "tempos_s": TEMPO_PUBLICADO},
        "preregistro": carimbo(PREREG),
        "script": carimbo(Path(__file__).resolve()),
        "ambiente": {"python": sys.version.split()[0], "executavel": sys.executable,
                     "cuda_usada": False},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", choices=["1", "2", "ambos"], default="ambos")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    SAIDA.mkdir(parents=True, exist_ok=True)
    if not PREREG.exists():
        morre(f"pre-registro ausente: {PREREG}")

    if args.item in ("1", "ambos"):
        print("[ITEM 1] admissibilidade sob particao bloqueada", flush=True)
        r1 = item1()
        r1["linha_campo_de_referencia"] = item1_campo_referencia()
        out1 = SAIDA / "admissibilidade_bloqueada_2026-09-15.json"
        out1.write_text(json.dumps(r1, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  gravado {out1}")
        a = r1["agregados"]
        print(f"  media por corrida = {a['global_media_por_corrida']:.9f}")
        print(f"  ponderado por nos = {a['global_ponderado_por_nos']:.9f}")
        print(f"  min = {a['min_global']:.9f} ({a['pior_corrida']['run_label']}), "
              f"exatamente 1.0 em {a['n_corridas_exatamente_1.0']}/80")

    if args.item in ("2", "ambos"):
        print("[ITEM 2] recontagem de parametros nos state_dict", flush=True)
        r2 = item2()
        out2 = SAIDA / "params_checkpoints_2026-09-15.json"
        out2.write_text(json.dumps(r2, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  gravado {out2}")
        for m, c in r2["confronto_com_o_publicado"].items():
            print(f"  {m:10s} publicado={c['params_publicado']:>8d} "
                  f"observados={c['contagens_observadas']} "
                  f"param_de={c['conjuntos_cujo_param_bate_com_o_publicado']} "
                  f"tempo_de={c['conjuntos_cujo_tempo_bate_com_o_publicado']} "
                  f"DIVERGENCIA_DE_CONJUNTO={c['divergencia_de_conjunto']}")


if __name__ == "__main__":
    main()
