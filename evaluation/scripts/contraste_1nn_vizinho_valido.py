"""Nearest-valid-neighbor path-loss MAE contrast under random vs. spatially blocked partitions, all 16 city-quadrant cells."""
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

EVID = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO")
GNN_RF_V2 = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2")
DATA_DIR = Path(r"F:\TOPO_RF_DOWNLOAD_DRIVE\graph_data_v3")
OUT_DIR = EVID / "dados" / "contraste_1nn_vizinho_valido"
PREREG = EVID / "dados" / "planos" / "preregistro_contraste_1nn_vizinho_valido_2026-09-16.json"
PREREG_SHA = "1b5ea2d845232bde040f96e5bbc77fc04f1ccf78cd6eadd78cdd789c161de03b"
PREREG_ANT = EVID / "dados" / "planos" / "preregistro_contraste_1nn_16celulas_2026-09-16.json"
PREREG_ANT_SHA = "77ed66be9f6a59c49cb6bb71ad9a15f31f0ab417ae1c785fa3eb61d5e61aef9d"
SCRIPT_ANT = EVID / "dados" / "contraste_1nn_16celulas" / "contraste_1nn_16celulas.py"
SCRIPT_ANT_SHA = "0c7c65b406974c15b1356ce0ba0a43c7db034aaa6cc7b16e1c514e200ef719cc"
JSON_ANT = EVID / "dados" / "contraste_1nn_16celulas" / "contraste_1nn_16celulas.json"
JSON_ANT_SHA = "6bf5c567312dd9e2a26bf73693915c85ec1d764a58b475f925bc49bb2e7b1924"
CONGELADO = EVID / "dados" / "scripts_congelados" / "train_gnn_c0_spatial.py"
CONGELADO_SHA = "6f955629cde164f2843f454e1ebf6977292fd80647ef48ecd0df31e464c42445"

SEED = 42
GRID_KM = 10.0
BUFFER_KM = 2.0
FRACS = (0.70, 0.15, 0.15)
PL_VALID_MAX = 299.0
N_AMOSTRA_TEST = 20000
LIMITE_CUSTO_S = 600.0
TOL_REPRO_DB = 1e-9

# pre-registered criterion (copied verbatim from the earlier pre-registration)
R_MIN = 2.0
DELTA_MIN_DB = 1.0
N_MIN = 1000
CIDADE_MIN = 3
CONJ_PRESENTE_MIN = 13
CONJ_AUSENTE_MAX = 8

GUARDAS = {
    ("lins", "Q1"): {"aleatorio": 0.3013, "bloqueado": 3.0792},
    ("bauru", "Q1"): {"aleatorio": 0.4685, "bloqueado": 15.6038},
}

CIDADES = ["lins", "bauru", "campinas", "sorocaba"]
QUADS = ["Q1", "Q2", "Q3", "Q4"]


class Aborto(RuntimeError):
    pass


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha256_bytes_arquivo(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def exige(cond, msg):
    if not cond:
        raise Aborto(msg)


def carrega_congelado():
    exige(sha256_bytes_arquivo(CONGELADO) == CONGELADO_SHA, f"script congelado divergente: {CONGELADO}")
    sys.path.append(str(GNN_RF_V2 / "03_training"))
    from spatial_cv import SpatialKFold
    spec = importlib.util.spec_from_file_location("train_gnn_c0_spatial_congelado", str(CONGELADO))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, SpatialKFold


def hashes_datasets():
    out = {}
    for linha in (DATA_DIR / "HASHES_SHA256.txt").read_text(encoding="utf-8").splitlines():
        partes = linha.split()
        if len(partes) == 2:
            out[partes[1]] = partes[0]
    return out


def mae_1nn(pos_km, pl, idx_train, idx_test, restrito, seed=SEED, n_amostra=N_AMOSTRA_TEST):
    """
    Same 1-NN MAE computation used elsewhere in the evaluation scripts; the
    only difference is that, when restricted, neighbor candidates are
    limited to idx_train entries with PL < 299 dB. Also returns the
    prediction.
    """
    n_train_total = int(len(idx_train))
    if restrito:
        idx_train = idx_train[pl[idx_train] < PL_VALID_MAX]
    exige(len(idx_train) > 0, "nenhum no de treino candidato")
    pos_tr = pos_km[idx_train]
    pl_tr = pl[idx_train]
    pos_te_full = pos_km[idx_test]
    pl_te_full = pl[idx_test]
    valid = pl_te_full < PL_VALID_MAX
    idx_valid = np.where(valid)[0]
    n_validos_total = int(len(idx_valid))
    if n_validos_total == 0:
        return {"n_test_total": int(len(idx_test)), "n_test_valido_total": 0,
                "n_test_valido_amostrado": 0, "mae_db": None}, None
    if len(idx_valid) > n_amostra:
        rng = np.random.RandomState(seed)
        idx_valid = rng.choice(idx_valid, size=n_amostra, replace=False)
    pos_te = pos_te_full[idx_valid]
    pl_te = pl_te_full[idx_valid]
    tree = cKDTree(pos_tr, compact_nodes=False, balanced_tree=False)
    d, nn = tree.query(pos_te, k=1, workers=-1)
    pred = pl_tr[nn]
    err = np.abs(pred - pl_te)
    mae = float(np.mean(err))
    sent = pred >= PL_VALID_MAX
    return {
        "restricao_vizinho_valido": bool(restrito),
        "n_train_total": n_train_total,
        "n_train_candidatos": int(len(idx_train)),
        "n_test_total": int(len(idx_test)),
        "n_test_valido_total": n_validos_total,
        "n_test_valido_amostrado": int(len(idx_valid)),
        "mae_db": mae,
        "dist_vizinho_km_media": float(d.mean()),
        "dist_vizinho_km_mediana": float(np.median(d)),
        "frac_pred_sentinela": float(sent.mean()),
        "n_pred_sentinela": int(sent.sum()),
        "diag_mae_db_so_vizinho_valido": (float(err[~sent].mean()) if (~sent).any() else None),
        "_idx_amostra_sha256": hashlib.sha256(np.sort(np.asarray(idx_valid, dtype=np.int64)).tobytes()).hexdigest(),
    }, pred


def confere_anterior(cidade, quad, particao, r_off, ant):
    """Guarda: restricao desligada tem de reproduzir o JSON anterior."""
    a = ant[(cidade, quad)][particao]
    for k in ("mae_db", "n_test_valido_amostrado", "n_test_total", "n_test_valido_total"):
        exige(k in a, f"chave {k} ausente no JSON anterior ({cidade} {quad} {particao})")
    exige(abs(r_off["mae_db"] - a["mae_db"]) <= TOL_REPRO_DB,
          f"GUARDA {cidade} {quad} {particao}: MAE desligado {r_off['mae_db']!r} != anterior {a['mae_db']!r}")
    exige(r_off["frac_pred_sentinela"] == a["diag_frac_pred_sentinela"],
          f"GUARDA {cidade} {quad} {particao}: frac sentinela {r_off['frac_pred_sentinela']} != anterior {a['diag_frac_pred_sentinela']}")
    for k in ("n_test_valido_amostrado", "n_test_total", "n_test_valido_total"):
        exige(r_off[k] == a[k], f"GUARDA {cidade} {quad} {particao}: {k} {r_off[k]} != anterior {a[k]}")
    return {"mae_db_anterior": a["mae_db"], "mae_db_desligado": r_off["mae_db"],
            "abs_dif": abs(r_off["mae_db"] - a["mae_db"]), "confere": True}


def processa(mod, SpatialKFold, cidade, quad, hashes_ds, ant):
    t_cel = time.time()
    run_path = EVID / "dados" / "treinos_c1" / f"run_c0c1cf_{cidade}_s42_{quad}_g10b2.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    nome = f"transfer_dataset_{cidade}_v19_{quad}_enriched_cftudo.pt"
    rf_file = DATA_DIR / nome


    exige(Path(run["dataset"]["rf_data_file"]).name == nome, f"{cidade} {quad}: rf_data_file da corrida {run['dataset']['rf_data_file']} != {nome}")
    exige(run["dataset"]["rf_data_bytes"] == rf_file.stat().st_size, f"{cidade} {quad}: tamanho do dataset diverge da corrida")
    exige(run["dataset"]["rf_data_sha256"] == hashes_ds[nome], f"{cidade} {quad}: rf_data_sha256 da corrida != HASHES_SHA256.txt")
    exige(run["config"]["max_nodes"] == 0, f"{cidade} {quad}: corrida usou janela (max_nodes={run['config']['max_nodes']})")
    exige(run["split_seed"] == SEED and run["seed"] == SEED, f"{cidade} {quad}: seed/split_seed da corrida != 42")
    exige(run["split"]["grid_km"] == GRID_KM and run["split"]["buffer_km"] == BUFFER_KM, f"{cidade} {quad}: grade/buffer divergente")
    exige(list(run["split"]["fracs_blocos_pedidas"]) == list(FRACS), f"{cidade} {quad}: fracs divergentes")
    exige(run["script_sha256"] == CONGELADO_SHA, f"{cidade} {quad}: corrida gerada por outro script ({run['script_sha256']})")

    log(f"=== {cidade} {quad}: carregando {rf_file} (mmap, READ-ONLY) ===")
    t0 = time.time()
    rf = torch.load(str(rf_file), mmap=True, map_location="cpu", weights_only=False)
    pos_deg = rf["terrain"].pos.float()
    tgt = torch.as_tensor(rf["terrain"].y).float()
    n_total = int(pos_deg.shape[0])
    pl = tgt[:, 0].numpy().copy()
    exige(int(tgt.shape[0]) == n_total, f"{cidade} {quad}: y e pos com N diferentes")
    t_carga = time.time() - t0
    log(f"  n_total={n_total:,} em {t_carga:.1f}s")
    exige(n_total == ant[(cidade, quad)]["n_total"], f"{cidade} {quad}: n_total diverge do JSON anterior")

    pos_m, geo = mod.latlon_graus_para_metros(pos_deg)
    del pos_deg, tgt, rf
    gc.collect()
    exige(abs(geo["lon_min_deg"] - run["geometria"]["lon_min_deg"]) == 0.0 and
          abs(geo["lat_min_deg"] - run["geometria"]["lat_min_deg"]) == 0.0,
          f"{cidade} {quad}: origem da projecao diverge da corrida")
    pos_km = pos_m / 1000.0

    t0 = time.time()
    parts, info = mod.split_espacial_3vias(pos_m, GRID_KM, BUFFER_KM, FRACS, SEED, SpatialKFold, log)
    t_split = time.time() - t0
    conf = {}
    for k in ("train", "val", "test"):
        h = mod.sha256_idx(parts[k])
        g = run["particoes"][k]["idx_sha256_global"]
        conf[k] = {"n": int(len(parts[k])), "idx_sha256_global": h, "confere": h == g}
        exige(h == g, f"{cidade} {quad}: idx_sha256_global de {k} diverge da corrida c0c1cf ({h} != {g})")
        exige(len(parts[k]) == run["particoes"][k]["n"], f"{cidade} {quad}: n de {k} diverge")
    exige(info["n_nos_apos_buffer"] == run["split"]["n_nos_apos_buffer"], f"{cidade} {quad}: n_nos_apos_buffer diverge")
    exige(sum(len(parts[k]) for k in parts) <= n_total, "particoes maiores que N")
    exige(run["particoes"]["test"]["n"] + run["particoes"]["train"]["n"] <= n_total, "n_train+n_test > N")
    log(f"  particao bloqueada reproduzida (idx_sha256_global train/val/test conferem) em {t_split:.1f}s")

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_total)
    n_tr, n_te = len(parts["train"]), len(parts["test"])
    idx_tr_r = perm[:n_tr]
    idx_te_r = perm[n_tr:n_tr + n_te]

    t0 = time.time()
    res = {}
    for particao, itr, ite in (("bloqueado", parts["train"], parts["test"]),
                               ("aleatorio", idx_tr_r, idx_te_r)):
        r_off, p_off = mae_1nn(pos_km, pl, itr, ite, restrito=False)
        r_on, p_on = mae_1nn(pos_km, pl, itr, ite, restrito=True)

        g = confere_anterior(cidade, quad, particao, r_off, ant)

        exige(r_on["_idx_amostra_sha256"] == r_off["_idx_amostra_sha256"],
              f"{cidade} {quad} {particao}: amostra de teste mudou com a restricao")

        exige(r_on["n_pred_sentinela"] == 0 and r_on["frac_pred_sentinela"] == 0.0,
              f"{cidade} {quad} {particao}: {r_on['n_pred_sentinela']} predicoes sentinela com a restricao ligada")
        mudou = p_on != p_off
        sent_off = p_off >= PL_VALID_MAX
        r_on["diag_frac_pred_mudou_vs_desligado"] = float(mudou.mean())
        r_on["diag_n_pred_mudou_sem_ser_sentinela_antes"] = int((mudou & ~sent_off).sum())
        r_on["diag_n_sentinela_antes_nao_mudou"] = int((sent_off & ~mudou).sum())
        exige(r_on["diag_n_sentinela_antes_nao_mudou"] == 0, f"{cidade} {quad} {particao}: sentinela anterior nao substituido")
        r_off.pop("_idx_amostra_sha256")
        res[particao] = {"restrito": r_on, "desligado_guarda": r_off, "guarda": g}
        log(f"  1-NN {particao}: restrito={r_on['mae_db']:.4f} desligado={r_off['mae_db']:.4f} "
            f"(sentinela antes={r_off['frac_pred_sentinela']:.4f}, mudou={r_on['diag_frac_pred_mudou_vs_desligado']:.4f})")
    t_1nn = time.time() - t0

    guarda_fixa = None
    if (cidade, quad) in GUARDAS:
        esp = GUARDAS[(cidade, quad)]
        obt = {k: res[k]["desligado_guarda"]["mae_db"] for k in ("aleatorio", "bloqueado")}
        for k in ("aleatorio", "bloqueado"):
            exige(round(obt[k], 4) == esp[k], f"GUARDA {cidade} {quad} {k}: {obt[k]:.6f} != {esp[k]}")
        guarda_fixa = {"esperado_round4": esp, "obtido_desligado": obt, "passou": True}
        log(f"  GUARDA pre-registrada (restricao desligada) OK: {obt}")

    r_bloq = res["bloqueado"]["restrito"]
    r_rand = res["aleatorio"]["restrito"]
    for r in (r_bloq, r_rand):
        r.pop("_idx_amostra_sha256")

    # per-cell criterion (same as above)
    mb, ma = r_bloq["mae_db"], r_rand["mae_db"]
    R = (mb / ma) if (mb is not None and ma is not None and ma > 0) else None
    delta = (mb - ma) if (mb is not None and ma is not None) else None
    n_min_ok = (r_bloq["n_test_valido_amostrado"] >= N_MIN and r_rand["n_test_valido_amostrado"] >= N_MIN)
    if not n_min_ok or R is None:
        leitura = "inconclusiva"
    elif R >= R_MIN and delta >= DELTA_MIN_DB:
        leitura = "presente"
    elif delta <= 0:
        leitura = "ausente"
    else:
        leitura = "abaixo_do_limiar"

    t_total = time.time() - t_cel
    del pos_m, pos_km, pl, parts, perm, idx_tr_r, idx_te_r
    gc.collect()
    return {
        "cidade": cidade, "quadrante": quad, "seed": SEED, "split_seed": SEED,
        "rf_data_file": str(rf_file), "rf_data_sha256_HASHES_txt": hashes_ds[nome],
        "run_c0c1cf": str(run_path), "run_c0c1cf_sha256": sha256_bytes_arquivo(run_path),
        "n_total": n_total,
        "particao_bloqueada_conferencia": conf,
        "mae_1nn_aleatorio": r_rand,
        "mae_1nn_bloqueado": r_bloq,
        "razao_R": R, "delta_db": delta,
        "leitura_criterio": leitura,
        "guarda_restricao_desligada": {"aleatorio": res["aleatorio"]["guarda"],
                                       "bloqueado": res["bloqueado"]["guarda"],
                                       "guarda_preregistrada_round4": guarda_fixa},
        "diag_definicao_desligada": {"aleatorio": res["aleatorio"]["desligado_guarda"],
                                     "bloqueado": res["bloqueado"]["desligado_guarda"]},
        "custo_s": {"carga": t_carga, "particao": t_split, "1nn": t_1nn, "total_celula": t_total},
    }


def resumo(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {"n": len(v), "mediana": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}


def agrega(celulas):
    def bloco(cs):
        return {
            "n_celulas": len(cs),
            "n_presente": sum(c["leitura_criterio"] == "presente" for c in cs),
            "mae_aleatorio_db": resumo([c["mae_1nn_aleatorio"]["mae_db"] for c in cs]),
            "mae_bloqueado_db": resumo([c["mae_1nn_bloqueado"]["mae_db"] for c in cs]),
            "razao_R": resumo([c["razao_R"] for c in cs]),
            "delta_db": resumo([c["delta_db"] for c in cs]),
            "frac_pred_sentinela_max": max(max(c["mae_1nn_aleatorio"]["frac_pred_sentinela"],
                                               c["mae_1nn_bloqueado"]["frac_pred_sentinela"]) for c in cs),
        }
    por_cidade = {}
    for cid in CIDADES:
        cs = [c for c in celulas if c["cidade"] == cid]
        if cs:
            b = bloco(cs)
            b["leitura"] = ("presente" if b["n_presente"] >= CIDADE_MIN else "nao_presente") if len(cs) == 4 else "incompleta"
            por_cidade[cid] = b
    geral = bloco(celulas)
    n_pres = geral["n_presente"]
    med_R = geral["razao_R"]["mediana"] if geral["razao_R"] else None
    if len(celulas) < 16:
        veredito = "INCOMPLETO"
    elif n_pres >= CONJ_PRESENTE_MIN and med_R is not None and med_R >= R_MIN:
        veredito = "PRESENTE"
    elif n_pres <= CONJ_AUSENTE_MAX:
        veredito = "AUSENTE"
    else:
        veredito = "HETEROGENEO"
    geral["veredito_conjunto"] = veredito
    return por_cidade, geral


def carrega_anterior():
    exige(sha256_bytes_arquivo(JSON_ANT) == JSON_ANT_SHA, "JSON anterior divergente do manifest")
    d = json.loads(JSON_ANT.read_text(encoding="utf-8"))
    exige(d["abortado"] is None, "JSON anterior abortado")
    exige(d["script_sha256"] == SCRIPT_ANT_SHA, "JSON anterior gerado por outro script")
    exige(d["parametros"]["seed"] == SEED and d["parametros"]["n_amostra_test"] == N_AMOSTRA_TEST
          and d["parametros"]["pl_valid_max_db"] == PL_VALID_MAX, "parametros do JSON anterior divergem")
    ant = {}
    for c in d["celulas"]:
        chave = (c["cidade"], c["quadrante"])
        exige(chave not in ant, f"celula duplicada no JSON anterior: {chave}")
        ant[chave] = {"n_total": c["n_total"], "aleatorio": c["mae_1nn_aleatorio"], "bloqueado": c["mae_1nn_bloqueado"]}
    exige(len(ant) == 16, f"JSON anterior tem {len(ant)} celulas, esperado 16")
    return ant


def main():
    t_ini = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "contraste_1nn_vizinho_valido.json"
    exige(sha256_bytes_arquivo(PREREG) == PREREG_SHA, "pre-registro alterado desde o registro no manifest")
    exige(sha256_bytes_arquivo(PREREG_ANT) == PREREG_ANT_SHA, "pre-registro anterior alterado")
    exige(sha256_bytes_arquivo(SCRIPT_ANT) == SCRIPT_ANT_SHA, "script anterior alterado")
    ant = carrega_anterior()
    mod, SpatialKFold = carrega_congelado()
    hashes_ds = hashes_datasets()
    log(f"python={sys.executable} | torch={torch.__version__} | cpu_count={os.cpu_count()} | CPU apenas")

    ordem = [("lins", "Q1"), ("bauru", "Q1")] + [(c, q) for c in CIDADES for q in QUADS
                                                  if (c, q) not in (("lins", "Q1"), ("bauru", "Q1"))]
    out = {
        "artefato_tipo": "medicao_contraste_1nn_vizinho_valido",
        "status": "provisorio",
        "citabilidade": "NAO citavel antes de contra-auditoria independente",
        "cegueira": "NAO cego: releitura exploratoria do JSON anterior ja indicava o sentido (ver pre-registro)",
        "gerado_em_utc": None,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_bytes_arquivo(Path(__file__).resolve()),
        "preregistro": str(PREREG), "preregistro_sha256": PREREG_SHA,
        "medicao_anterior": {"preregistro": str(PREREG_ANT), "preregistro_sha256": PREREG_ANT_SHA,
                             "script": str(SCRIPT_ANT), "script_sha256": SCRIPT_ANT_SHA,
                             "json": str(JSON_ANT), "json_sha256": JSON_ANT_SHA},
        "script_particao": str(CONGELADO), "script_particao_sha256": CONGELADO_SHA,
        "ambiente": {"python": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__, "dispositivo": "cpu"},
        "parametros": {"seed": SEED, "grid_km": GRID_KM, "buffer_km": BUFFER_KM, "fracs": FRACS,
                       "pl_valid_max_db": PL_VALID_MAX, "n_amostra_test": N_AMOSTRA_TEST,
                       "vizinho_candidato": "treino com PL < 299 dB", "tol_reproducao_db": TOL_REPRO_DB},
        "criterio": {"R_min": R_MIN, "delta_min_db": DELTA_MIN_DB, "n_min_amostrado": N_MIN,
                     "cidade_min_presentes_de_4": CIDADE_MIN,
                     "conjunto_presente_min_de_16": CONJ_PRESENTE_MIN,
                     "conjunto_ausente_max_de_16": CONJ_AUSENTE_MAX},
        "celulas": [], "abortado": None, "parada_por_custo": False,
    }

    def grava():
        out["gerado_em_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if out["celulas"]:
            pc, g = agrega(out["celulas"])
            out["agregado_por_cidade"] = pc
            out["agregado_geral"] = g
        out["custo_total_s"] = time.time() - t_ini
        dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    estourou = False
    try:
        for i, (cid, q) in enumerate(ordem, 1):
            r = processa(mod, SpatialKFold, cid, q, hashes_ds, ant)
            out["celulas"].append(r)
            log(f"  [{i}/16] {cid} {q}: aleat={r['mae_1nn_aleatorio']['mae_db']} bloq={r['mae_1nn_bloqueado']['mae_db']} "
                f"R={r['razao_R']} -> {r['leitura_criterio']} | {r['custo_s']['total_celula']:.1f}s")
            grava()
            if r["custo_s"]["total_celula"] > LIMITE_CUSTO_S:
                estourou = True
            if estourou and i >= 4:
                out["parada_por_custo"] = True
                log("PARADA POR CUSTO: celula > 600 s; parando apos a quarta celula")
                break
    except Aborto as e:
        out["abortado"] = str(e)
        log(f"ABORTO: {e}")
        grava()
        return 2
    grava()
    log(f"Gravado: {dest} | veredito={out.get('agregado_geral', {}).get('veredito_conjunto')} | total {time.time()-t_ini:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
