
"""Integrity check, run by run, of the graph-free control campaign for Campinas and Sorocaba."""
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access"
            r"\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO")
TREINOS = BASE / "treinos"
OUT = BASE / "dados" / "integridade" / \
    "integridade_mlpcf_campinas_sorocaba_2026-09-15.json"

CIDADES_NOVAS = ["campinas", "sorocaba"]
SEEDS = [42, 43, 44, 45, 46]
QUADS = ["Q1", "Q2", "Q3", "Q4"]

SHA_SCRIPT_CONGELADO = ("4b75093f52e8b7fdec45c7f2b8c5bcd68c38093c420697927"
                        "b49fa99a474b31c")
SCRIPT_CONGELADO_EM_DISCO = BASE / "dados" / \
    "scripts_congelados" / "train_mlp_c0_spatial.py"
PARAMS_ALVO = 1812515
HIPER = {  # (epochs, lr) per quadrant
    "Q1": (35, 1e-3), "Q2": (20, 2e-4), "Q3": (20, 2e-4), "Q4": (20, 2e-4)}


def sha256_arquivo(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def caminho_run(label):
    return TREINOS / label / f"run_{label}.json"


def carrega(label):
    p = caminho_run(label)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def chk(itens, chave, observado, esperado, ok=None):
    """Registra um item. ok=None => igualdade estrita observado == esperado."""
    if ok is None:
        ok = (observado == esperado)
    itens[chave] = {"observado": observado, "esperado": esperado,
                    "ok": bool(ok)}
    return bool(ok)


def audita(cidade, seed, quad, sha_disco_bate):
    label = f"mlpcf_{cidade}_s{seed}_{quad}_g10b2"
    par = f"c0c1cf_{cidade}_s{seed}_{quad}_g10b2"
    reg = {"run": label, "par_gnn": par, "itens": {}, "falhas": [],
           "erros_de_campo": []}
    it = reg["itens"]

    d = carrega(label)
    if d is None:
        reg["falhas"].append("I1:JSON_AUSENTE")
        it["I1_json_existe"] = {"observado": False, "esperado": True,
                                "ok": False}
        return reg
    g = carrega(par)
    if g is None:
        reg["falhas"].append("I2:PAR_GNN_AUSENTE")

    try:

        t = d["custo"]["tempo_total_s"]
        if not chk(it, "I1_tempo_total_s", t, "nao nulo e > 0",
                   ok=(t is not None and t > 0)):
            reg["falhas"].append("I1:tempo_total_s")
        ne = d["custo"]["n_epocas_rodadas"]
        ep_cfg = d["config"]["epochs"]
        if not chk(it, "I1_n_epocas_rodadas_igual_config", ne, ep_cfg):
            reg["falhas"].append("I1:n_epocas_rodadas")
        it["I1_status_declarado"] = {"observado": d["status"],
                                     "esperado": "provisorio",
                                     "ok": d["status"] == "provisorio"}
        it["I1_timestamp_utc"] = {"observado": d["timestamp_utc"],
                                  "esperado": "(informativo)", "ok": True}


        if g is not None:
            for parte in ("test", "train", "val"):
                hm = d["particoes"][parte]["idx_sha256_global"]
                hg = g["particoes"][parte]["idx_sha256_global"]
                if not chk(it, f"I2_idx_sha256_global_{parte}", hm, hg):
                    reg["falhas"].append(f"I2:idx_{parte}")
            nm = d["particoes"]["test"]["n"]
            ng = g["particoes"]["test"]["n"]
            if not chk(it, "I2_n_test_igual_gnn", nm, ng):
                reg["falhas"].append("I2:n_test")


            sm = d["dataset"]["rf_data_sha256"]
            sg = g["dataset"]["rf_data_sha256"]
            if not chk(it, "I3_rf_data_sha256", sm, sg):
                reg["falhas"].append("I3:rf_data_sha256")
            if not chk(it, "I3_rf_data_file", d["dataset"]["rf_data_file"],
                       g["dataset"]["rf_data_file"]):
                reg["falhas"].append("I3:rf_data_file")
            if not chk(it, "I3_rf_data_bytes", d["dataset"]["rf_data_bytes"],
                       g["dataset"]["rf_data_bytes"]):
                reg["falhas"].append("I3:rf_data_bytes")


            it["I3_graph_file_sem_sha256"] = {
                "observado": {"graph_file": d["dataset"]["graph_file"],
                              "graph_bytes": d["dataset"]["graph_bytes"],
                              "tem_sha256": "graph_sha256" in d["dataset"]},
                "esperado": "(observacao: gerador nao carimba sha do gpu.pt)",
                "ok": True}


        for campo in ("n_params_treinaveis", "n_params", "n_params_alvo_gnn"):
            if not chk(it, f"I4_{campo}", d["modelo"][campo], PARAMS_ALVO):
                reg["falhas"].append(f"I4:{campo}")
        if not chk(it, "I4_paridade_delta", d["modelo"]["paridade_delta"], 0):
            reg["falhas"].append("I4:paridade_delta")


        if not chk(it, "I5_script_sha256", d["script_sha256"],
                   SHA_SCRIPT_CONGELADO):
            reg["falhas"].append("I5:script_sha256")
        if not chk(it, "I5_script_path", d["script"],
                   str(SCRIPT_CONGELADO_EM_DISCO)):
            reg["falhas"].append("I5:script_path_nao_e_o_congelado")
        it["I5_sha_do_arquivo_em_disco_bate"] = {
            "observado": sha_disco_bate, "esperado": True,
            "ok": bool(sha_disco_bate)}
        if not sha_disco_bate:
            reg["falhas"].append("I5:sha_disco")


        ep_esp, lr_esp = HIPER[quad]
        c = d["config"]
        if not chk(it, "I6_epochs", c["epochs"], ep_esp):
            reg["falhas"].append("I6:epochs")
        if not chk(it, "I6_lr", c["lr"], lr_esp,
                   ok=abs(c["lr"] - lr_esp) <= 1e-12):
            reg["falhas"].append("I6:lr")
        if not chk(it, "I6_seed", c["seed"], seed):
            reg["falhas"].append("I6:seed")
        if not chk(it, "I6_seed_topo", d["seed"], seed):
            reg["falhas"].append("I6:seed_topo")
        if not chk(it, "I6_split_seed", c["split_seed"], 42):
            reg["falhas"].append("I6:split_seed")
        if not chk(it, "I6_grid_km", c["grid_km"], 10.0):
            reg["falhas"].append("I6:grid_km")
        if not chk(it, "I6_buffer_km", c["buffer_km"], 2.0):
            reg["falhas"].append("I6:buffer_km")
        if not chk(it, "I6_split_frac", c["split_frac"], "0.70,0.15,0.15"):
            reg["falhas"].append("I6:split_frac")
        if not chk(it, "I6_freq_mhz_perda",
                   d["loss"]["frequency_mhz_loss_fspl"], 900.0):
            reg["falhas"].append("I6:freq_perda")
        if not chk(it, "I6_freq_mhz_diagnostico",
                   d["loss"]["frequency_mhz_diagnostico"], 1800.0):
            reg["falhas"].append("I6:freq_diag")
        if not chk(it, "I6_grid_km_usado_na_geometria",
                   d["geometria"]["grid_km_usado"], 10.0):
            reg["falhas"].append("I6:grid_km_usado")
        if not chk(it, "I6_buffer_km_usado_na_geometria",
                   d["geometria"]["buffer_km_usado"], 2.0):
            reg["falhas"].append("I6:buffer_km_usado")
        if not chk(it, "I6_fracs_blocos_pedidas",
                   d["split"]["fracs_blocos_pedidas"], [0.7, 0.15, 0.15]):
            reg["falhas"].append("I6:fracs_blocos")

        # checkpoint selection by validation, test scored once
        sel = d["selecao"]
        if not chk(it, "I7_criterio", sel["criterio"],
                   "menor mae_rssi_db em VAL"):
            reg["falhas"].append("I7:criterio")
        if not chk(it, "I7_test_usado_na_selecao",
                   sel["test_usado_na_selecao"], False):
            reg["falhas"].append("I7:test_usado_na_selecao")
        epocas = d["epocas"]
        marcas = sorted({e["test_usado_na_selecao"] for e in epocas})
        if not chk(it, "I7_test_usado_na_selecao_por_epoca", marcas, [False]):
            reg["falhas"].append("I7:test_usado_por_epoca")

        vals = [e["val"]["mae_rssi_db"] for e in epocas]
        i_min = min(range(len(vals)), key=lambda i: vals[i])
        ep_argmin = epocas[i_min]["epoch"]
        if not chk(it, "I7_melhor_epoca_e_argmin_de_val",
                   {"melhor_epoca_declarada": sel["melhor_epoca"],
                    "argmin_val_recomputado": ep_argmin},
                   "iguais", ok=(sel["melhor_epoca"] == ep_argmin)):
            reg["falhas"].append("I7:melhor_epoca_nao_e_argmin_val")
        if not chk(it, "I7_melhor_val_mae_rssi_db_bate_com_a_epoca",
                   {"declarado": sel["melhor_val_mae_rssi_db"],
                    "epocas[argmin].val": vals[i_min]}, "iguais",
                   ok=(abs(sel["melhor_val_mae_rssi_db"] - vals[i_min])
                       <= 1e-12)):
            reg["falhas"].append("I7:melhor_val_valor")
        if not chk(it, "I7_n_epocas_registradas", len(epocas), ep_cfg):
            reg["falhas"].append("I7:n_epocas_registradas")
        tm = sel["test_no_melhor_ckpt"]
        if not chk(it, "I7_test_n_nos_igual_particao", tm["n_nos"],
                   d["particoes"]["test"]["n"]):
            reg["falhas"].append("I7:test_n_nos")
        if not chk(it, "I7_test_n_nos_avaliados", tm["n_nos_avaliados"],
                   d["particoes"]["test"]["n"]):
            reg["falhas"].append("I7:test_n_nos_avaliados")
        if not chk(it, "I7_test_cobertura_epoca", tm["cobertura_epoca"], 1.0):
            reg["falhas"].append("I7:cobertura_epoca")
        # diagnostic only (not a criterion): re-evaluation of the checkpoint against the test set


        it["I7_diag_test_reavaliado_menos_test_da_epoca_db"] = {
            "observado": tm["mae_rssi_db"] - epocas[i_min]["test"]["mae_rssi_db"],
            "esperado": "(diagnostico, sem limiar pre-registrado)", "ok": True}
        it["I7_mae_rssi_db_reportavel"] = {
            "observado": tm["mae_rssi_db"], "esperado": "(valor)", "ok": True}


        tf = c["transfer_from"]
        if quad == "Q1":
            if not chk(it, "X1_transfer_from", tf, ""):
                reg["falhas"].append("X1:Q1_com_transferencia")
        else:
            anterior = QUADS[QUADS.index(quad) - 1]
            esperado = str(TREINOS / f"mlpcf_{cidade}_s{seed}_{anterior}_g10b2"
                           / "checkpoints" / "checkpoint_best.pt")
            if not chk(it, "X1_transfer_from", tf, esperado):
                reg["falhas"].append("X1:transfer_from")
            if not chk(it, "X1_checkpoint_de_origem_existe",
                       Path(tf).exists() if tf else False, True):
                reg["falhas"].append("X1:origem_inexistente")


        v = d["split"]["verificacao"]
        if not chk(it, "X2_split_verificacao_ok", v["ok"], True):
            reg["falhas"].append("X2:verificacao_ok")
        inter = {k: v["intersecoes"][k] for k in v["intersecoes"]}
        if not chk(it, "X2_intersecoes", inter,
                   {k: 0 for k in inter}):
            reg["falhas"].append("X2:intersecoes")
        dmin = {k: v["pares"][k]["dist_min_km"] for k in v["pares"]}
        if not chk(it, "X2_dist_min_km_por_par", dmin, ">= 2.0",
                   ok=all(x >= 2.0 for x in dmin.values())):
            reg["falhas"].append("X2:dist_min")


        if not chk(it, "X3_classe", d["modelo"]["classe"], "MLPRFModel"):
            reg["falhas"].append("X3:classe")
        if not chk(it, "X3_sem_arestas", d["modelo"]["sem_arestas"], True):
            reg["falhas"].append("X3:sem_arestas")
        if not chk(it, "X3_sem_passagem_de_mensagem",
                   d["modelo"]["sem_passagem_de_mensagem"], True):
            reg["falhas"].append("X3:sem_msg")


        ck = TREINOS / label / "checkpoints" / "checkpoint_best.pt"
        if not chk(it, "X4_checkpoint_best_existe", ck.exists(), True):
            reg["falhas"].append("X4:checkpoint")

    except KeyError as e:
        reg["erros_de_campo"].append(f"CAMPO_AUSENTE:{e}")
        reg["falhas"].append(f"CAMPO_AUSENTE:{e}")
    reg["ok"] = (len(reg["falhas"]) == 0)
    return reg


def main():
    if not SCRIPT_CONGELADO_EM_DISCO.exists():
        sys.exit(f"ERRO: script congelado ausente: {SCRIPT_CONGELADO_EM_DISCO}")
    sha_disco = sha256_arquivo(SCRIPT_CONGELADO_EM_DISCO)
    sha_disco_bate = (sha_disco == SHA_SCRIPT_CONGELADO)

    corridas = []
    for cidade in CIDADES_NOVAS:
        for seed in SEEDS:
            for quad in QUADS:
                corridas.append(audita(cidade, seed, quad, sha_disco_bate))


    por_hash = {}
    for cidade in CIDADES_NOVAS:
        for seed in SEEDS:
            for quad in QUADS:
                label = f"mlpcf_{cidade}_s{seed}_{quad}_g10b2"
                d = carrega(label)
                if d is None:
                    continue
                h = d["particoes"]["test"]["idx_sha256_global"]
                por_hash.setdefault(h, set()).add(f"{cidade}_{quad}")
    colisoes = {h: sorted(v) for h, v in por_hash.items() if len(v) > 1}


    dec_p = (BASE / "dados" / "planos"
             / "decisao_dono_T11_levanta_portao_T13_2026-09-15.json")
    decisao = json.loads(dec_p.read_text(encoding="utf-8"))
    fila_p = Path(decisao["execucao"]["fila"])
    i8 = {
        "decisao_do_dono": str(dec_p),
        "decisao_sha256": sha256_arquivo(dec_p),
        "fila_declarada": str(fila_p),
        "fila_sha256_declarado": decisao["execucao"]["fila_sha256"],
        "fila_sha256_medido": sha256_arquivo(fila_p) if fila_p.exists() else None,
        "fila_bate": (fila_p.exists()
                      and sha256_arquivo(fila_p)
                      == decisao["execucao"]["fila_sha256"]),
        "script_treino_sha256_declarado":
            decisao["execucao"]["script_treino_sha256"],
        "script_treino_bate": (decisao["execucao"]["script_treino_sha256"]
                               == SHA_SCRIPT_CONGELADO),
        "parametros_declarados": decisao["execucao"]["parametros"],
        "ressalva_registrada_na_decisao": decisao["ressalva_registrada"],
    }

    shas_script = sorted({c["itens"]["I5_script_sha256"]["observado"]
                          for c in corridas
                          if "I5_script_sha256" in c["itens"]})

    falhas = {c["run"]: c["falhas"] for c in corridas if c["falhas"]}
    resumo = {
        "n_corridas_auditadas": len(corridas),
        "n_ok": sum(1 for c in corridas if c["ok"]),
        "n_com_falha": len(falhas),
        "falhas_por_corrida": falhas,
        "itens_falhos_distintos": sorted({f.split(":")[0] + ":" + f.split(":")[1]
                                          for v in falhas.values() for f in v}),
        "script_sha256_distintos_nas_40": shas_script,
        "script_sha256_unico_e_congelado": (shas_script
                                            == [SHA_SCRIPT_CONGELADO]),
        "sha256_recomputado_do_arquivo_congelado_em_disco": sha_disco,
        "X5_colisao_idx_test_entre_celulas": colisoes,
        "I8_fila_e_decisao_do_dono": i8,
    }

    saida = {
        "artefato_tipo": "integridade_corridas_mlpcf_campinas_sorocaba",
        "data": "2026-09-15",
        "autor": "forum-eng-dados (agente), sob ordem da sessao principal",
        "status": "provisorio (pendente de contra-auditoria por quem nao "
                  "escreveu este script)",
        "escopo": "as 40 corridas novas mlpcf_{campinas,sorocaba}_s{42..46}_"
                  "{Q1..Q4}_g10b2; o par c0c1cf da mesma celula/semente entra "
                  "so como referencia de pareamento",
        "fonte_primaria": "treinos/<label>/run_<label>.json (originais; as "
                          "copias em dados/treinos_c1 NAO foram lidas)",
        "referencias_do_criterio": {
            "fila_congelada": "scripts/fila_mlp_c1v2.ps1:88-126 "
                              "(-Cidades parametriza a cidade)",
            "script_congelado": str(SCRIPT_CONGELADO_EM_DISCO),
            "paridade": "modelo.n_params_treinaveis == 1812515",
        },
        "script": str(Path(__file__)),
        "script_sha256": sha256_arquivo(Path(__file__)),
        "resumo": resumo,
        "corridas": corridas,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    print(f"OK {OUT}")
    print(f"auditadas {resumo['n_corridas_auditadas']}, "
          f"ok {resumo['n_ok']}, com falha {resumo['n_com_falha']}")
    for run, fs in falhas.items():
        print(f"  FALHA {run}: {fs}")
    print("script_sha256 distintos:", shas_script)
    print("colisoes idx test entre celulas:", colisoes)


if __name__ == "__main__":
    main()
