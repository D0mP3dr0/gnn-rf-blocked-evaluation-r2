
"""
Cross-city transfer evaluation under a spatially blocked partition,
inference only (without training; only model.eval() / torch.no_grad()).

Only checkpoints from the c0c1cf_lins_s42_Q<n>_g10b2 family
(blocked-split campaign) supply the transfer source.

For each target cell (Campinas/Sorocaba, Q1-Q4) and its Lins reference
cell:
    1. The target-cell dataset is loaded, the blocked spatial
       partition is recomputed, and idx_sha256_global is checked
       against the recorded run JSON
       (particoes.test.idx_sha256_global) for that cell; a hash
       mismatch aborts only that cell.
    2. A GNNRFModel is built with the target cell's architecture
       (identical across the c0c1cf_g10b2 family: terrain_dim=18,
       hidden_dim=256, num_layers=4, heads=4, output_dim=5,
       dropout=0.1) and loaded with the source checkpoint's
       state_dict (Lins, same quadrant index, seed 42).
    3. Inference is run on two branches over the same blocked test
       partition: "full" (terrain.x unchanged) and "no_topo"
       (columns 0-7 of terrain.x zeroed).
    4. Metrics are computed for three populations (all nodes;
       covered nodes; path-loss on covered nodes) for both
       branches.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
EVID = AQUI.parents[1]
TREINOS_DIR = EVID / "treinos"
PU_PATH = EVID / "_campanha_2026-09-13" / "passada_unica" / "passada_unica.py"
TRAIN_GNN_SCRIPT = EVID / "dados" / "scripts_congelados" / "train_gnn_c0_spatial.py"
PREREG = EVID / "dados" / "planos" / "preregistro_transferencia_bloqueada_2026-09-15.json"
OLD_RESULT = EVID / "dados" / "evidencia_projeto" / "ood_crosscity_s42.json"

OUT_JSON = AQUI / "transferencia_bloqueada_seed42.json"
LOG_PATH = AQUI / "transferencia_bloqueada_seed42.log"

SEED = 42
NO_TOPO_COLS = 8
EXPECTED_TERRAIN_COLS = 18


CELULAS_ALVO = [("campinas", q) for q in (1, 2, 3, 4)] + \
               [("sorocaba", q) for q in (1, 2, 3, 4)]
CELULAS_REFERENCIA = [("lins", q) for q in (1, 2, 3, 4)]

PROIBIDO_NO_CAMINHO = ("version_20", "multiseed_")


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def checkpoint_origem_path(quadrante: int) -> Path:
    run_label = f"c0c1cf_lins_s{SEED}_Q{quadrante}_g10b2"
    p = TREINOS_DIR / run_label / "checkpoints" / "checkpoint_best.pt"
    caminho_str = str(p)
    for proibido in PROIBIDO_NO_CAMINHO:
        if proibido in caminho_str:
            raise RuntimeError(f"GUARDA: checkpoint proibido pela ordem do dono: {p}")
    if "c0c1cf_" not in caminho_str:
        raise RuntimeError(f"GUARDA: checkpoint fora da campanha bloqueada c0c1cf_*: {p}")
    if not p.exists():
        raise FileNotFoundError(f"checkpoint de origem ausente: {p}")
    return p, run_label


def clone_hetero_com_x(graph, x_novo, ET_AT, ET_TA, ET_TT):
    from torch_geometric.data import HeteroData
    g2 = HeteroData()
    g2["terrain"].x = x_novo
    g2["terrain"].pos = graph["terrain"].pos
    g2["terrain"].rf_targets = graph["terrain"].rf_targets
    g2["antenna"].x = graph["antenna"].x
    g2["antenna"].num_nodes = graph["antenna"].num_nodes
    g2[ET_AT].edge_index = graph[ET_AT].edge_index
    if hasattr(graph[ET_AT], "edge_attr"):
        g2[ET_AT].edge_attr = graph[ET_AT].edge_attr
    g2[ET_TA].edge_index = graph[ET_TA].edge_index
    g2[ET_TT].edge_index = graph[ET_TT].edge_index
    g2[ET_TT].edge_attr = graph[ET_TT].edge_attr
    return g2


def ambiente_hash() -> dict:
    freeze = ""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=120)
        freeze = r.stdout
    except Exception as e:
        freeze = f"ERRO ao capturar pip freeze: {e}"
    h = hashlib.sha256(freeze.encode("utf-8", errors="replace")).hexdigest()
    return {
        "python": sys.version.split()[0],
        "executavel": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_disponivel": torch.cuda.is_available(),
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pip_freeze_sha256": h,
    }


def avaliar_celula(pu, mod, cidade_alvo: str, q: int, device: str, use_amp: bool) -> dict:
    ckpt_origem, run_label_origem = checkpoint_origem_path(q)
    run_label_alvo = f"c0c1cf_{cidade_alvo}_s{SEED}_Q{q}_g10b2"
    run_json_alvo = TREINOS_DIR / run_label_alvo / f"run_{run_label_alvo}.json"
    if not run_json_alvo.exists():
        raise FileNotFoundError(f"run JSON da celula alvo ausente: {run_json_alvo}")

    t0 = time.perf_counter()
    prep = pu.preparar_celula(mod, run_label_alvo)
    t_prep = time.perf_counter() - t0
    log(f"[{cidade_alvo}_Q{q}] preparar_celula em {t_prep:.1f}s "
        f"idx_sha256_global_ok={prep['hash_ok']}")

    if not prep["hash_ok"]:
        raise RuntimeError(
            f"CUSTODIA FALHOU em {cidade_alvo}_Q{q}: idx_sha256_global recomputado "
            f"({prep['sha_test_recomputado']}) != run JSON ({prep['sha_test_ref']}). "
            f"Celula abortada, nenhuma metrica calculada.")

    ref_alvo = prep["ref"]
    cfg = prep["cfg"]
    test_graph = prep["graphs"]["test"]
    ET_AT, ET_TA, ET_TT = mod.ET_AT, mod.ET_TA, mod.ET_TT

    n_cols = int(test_graph["terrain"].x.shape[1])
    if n_cols != EXPECTED_TERRAIN_COLS:
        raise RuntimeError(
            f"GUARDA: terrain.x tem {n_cols} colunas em {cidade_alvo}_Q{q}, "
            f"esperava {EXPECTED_TERRAIN_COLS}; ablacao no_topo abortada para nao "
            f"zerar indices errados.")

    y_test = test_graph["terrain"].rf_targets
    valid_mask = (y_test[:, 0] < mod.PL_TARGET_MAX_VALID).numpy()
    sentinel_mask = ~valid_mask

    model = pu.montar_modelo(mod, None, ref_alvo, device, "gnn")
    ck = torch.load(str(ckpt_origem), map_location=device, weights_only=False)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model.load_state_dict(state)
    del ck

    resultados = {}
    for arm, ablate in (("full", False), ("no_topo", True)):
        ta = time.perf_counter()
        if ablate:
            x2 = test_graph["terrain"].x.clone()
            x2[:, :NO_TOPO_COLS] = 0.0
            g = clone_hetero_com_x(test_graph, x2, ET_AT, ET_TA, ET_TT)
        else:
            g = test_graph
        po, to, vi = pu.avaliar_gnn(mod, model, g, cfg, device, use_amp, seed_loader=0)
        if not bool(vi.all()):
            raise RuntimeError(
                f"nem todos os nos de teste foram vistos pelo NeighborLoader em "
                f"{cidade_alvo}_Q{q}/{arm} ({int(vi.sum())}/{vi.numel()})")
        m, err_rssi, err_pl = pu.node_metrics(po, to, valid_mask, sentinel_mask)
        m["elapsed_s"] = time.perf_counter() - ta
        resultados[arm] = m
        log(f"[{cidade_alvo}_Q{q}/{arm}] mae_total={m['mae_rssi_total_db']:.4f} "
            f"mae_valido={m['mae_rssi_valido_db']:.4f} "
            f"mae_pl_valido={m['mae_pl_valido_db']:.4f} "
            f"frac_valido={m['frac_valido']:.4f} tempo={m['elapsed_s']:.0f}s")
        if ablate:
            del g

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "cidade_alvo": cidade_alvo,
        "quadrante": q,
        "checkpoint_origem_run_label": run_label_origem,
        "checkpoint_origem_caminho": str(ckpt_origem),
        "checkpoint_origem_sha256": sha256_file(ckpt_origem),
        "run_json_alvo": str(run_json_alvo),
        "idx_sha256_global_test_recomputado": prep["sha_test_recomputado"],
        "idx_sha256_global_test_ref": prep["sha_test_ref"],
        "idx_sha256_global_test_ok": prep["hash_ok"],
        "rf_data_file": str(Path(cfg["graph_dir"]) / cfg["rf_data_file"]),
        "rf_data_sha256": ref_alvo["dataset"].get("rf_data_sha256"),
        "graph_file": str(Path(cfg["graph_dir"]) / cfg["graph_file"]),
        "n_terrain_cols": n_cols,
        "colunas_zeradas_no_topo": list(range(NO_TOPO_COLS)),
        "tempo_preparar_celula_s": t_prep,
        "populacoes": resultados,
    }


def agregados_por_cidade(registros: list[dict]) -> dict:
    """Media Q1-Q4 por cidade/braco/populacao + delta e os dois percentuais."""
    pops = ["mae_rssi_total_db", "mae_rssi_valido_db", "mae_pl_valido_db"]
    por_cidade: dict[str, list[dict]] = {}
    for r in registros:
        if "erro" in r:
            continue
        por_cidade.setdefault(r["cidade_alvo"], []).append(r)

    agg = {}
    for cidade, regs in por_cidade.items():
        n_por_quadrante = {arm: [r["populacoes"][arm]["n"] for r in regs] for arm in ("full", "no_topo")}
        medias_iguais = all(len(set(v)) == 1 for v in n_por_quadrante.values())
        bloco = {"n_quadrantes": len(regs), "n_por_quadrante_todos_iguais": medias_iguais,
                 "por_braco": {}, "delta_e_percentual": {}}
        for arm in ("full", "no_topo"):
            bloco["por_braco"][arm] = {}
            for pop in pops:
                vals = [r["populacoes"][arm][pop] for r in regs if r["populacoes"][arm].get(pop) is not None]
                if vals:
                    bloco["por_braco"][arm][pop] = {
                        "media_q1_q4": float(np.mean(vals)),
                        "dp_q1_q4": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                        "por_quadrante": [r["populacoes"][arm][pop] for r in regs],
                    }
                else:
                    bloco["por_braco"][arm][pop] = None
        for pop in pops:
            full = bloco["por_braco"]["full"][pop]
            notopo = bloco["por_braco"]["no_topo"][pop]
            if full is None or notopo is None:
                bloco["delta_e_percentual"][pop] = None
                continue
            mf, mn = full["media_q1_q4"], notopo["media_q1_q4"]
            delta = mn - mf
            bloco["delta_e_percentual"][pop] = {
                "mae_full_dB": mf, "mae_no_topo_dB": mn, "delta_dB": delta,
                "pct_den_full": (100.0 * delta / mf) if mf != 0 else None,
                "pct_den_no_topo": (100.0 * delta / mn) if mn != 0 else None,
                "denominador_correto_para_o_verbo_raises_lowers": "full",
            }
        agg[cidade] = bloco
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--celulas", type=str, default="",
                     help="lista 'cidade:Q' separada por virgula; default = 8 alvo + 4 referencia")
    ap.add_argument("--so-alvo", action="store_true",
                     help="processa so as 8 celulas alvo (campinas/sorocaba), sem a referencia lins")
    args = ap.parse_args()

    if not PREREG.exists():
        raise SystemExit(f"ABORTA: pre-registro ausente -> {PREREG}")
    if not torch.cuda.is_available():
        raise SystemExit("ABORTA: CUDA indisponivel neste interpretador; "
                          "verifique se esta usando F:\\S33_dslm\\s33_amb_virtual\\.venv\\Scripts\\python.exe")

    if args.celulas:
        pares = []
        for spec in args.celulas.split(","):
            c, qq = spec.split(":")
            pares.append((c, int(qq)))
    else:
        pares = list(CELULAS_ALVO)
        if not args.so_alvo:
            pares += list(CELULAS_REFERENCIA)

    pu = import_module(PU_PATH, "pu_harness")
    mod = import_module(TRAIN_GNN_SCRIPT, "train_gnn_c0_spatial_frozen")
    device = "cuda"
    use_amp = True

    out = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {
        "artefato_tipo": "transferencia_bloqueada_ood_crosscity",
        "status": "provisorio",
        "citavel": False,
        "seed": SEED,
        "pre_registro": str(PREREG),
        "pre_registro_sha256": sha256_file(PREREG),
        "script_sha256": sha256_file(Path(__file__)),
        "ambiente": ambiente_hash(),
        "ordem_do_dono": "somente checkpoints c0c1cf_lins_s42_Q<n>_g10b2; nenhum version_20*/multiseed_*; nenhum treino novo",
        "resultado_antigo_a_substituir": {
            "caminho": str(OLD_RESULT),
            "sha256": sha256_file(OLD_RESULT) if OLD_RESULT.exists() else None,
        },
        "registros": [],
        "custo_medido": [],
    }

    feitos = {(r["cidade_alvo"], r["quadrante"]) for r in out["registros"] if "erro" not in r}
    orcamento_min = 90.0
    t_inicio_orcamento = time.perf_counter()
    parou_por_orcamento = False

    for cidade, q in pares:
        if (cidade, q) in feitos:
            log(f"[{cidade}_Q{q}] ja processada — pulando (reuso de artefato)")
            continue
        gasto_min = (time.perf_counter() - t_inicio_orcamento) / 60.0
        if gasto_min > orcamento_min:
            log(f"ORCAMENTO ESTOURADO ({gasto_min:.1f} min > {orcamento_min} min) "
                f"antes de {cidade}_Q{q} — parando, resultado parcial e valido")
            parou_por_orcamento = True
            break
        t_cel0 = time.perf_counter()
        try:
            rec = avaliar_celula(pu, mod, cidade, q, device, use_amp)
            out["registros"].append(rec)
        except Exception:
            erro = traceback.format_exc()
            log(f"[{cidade}_Q{q}] ERRO:\n{erro}")
            out["registros"].append({"cidade_alvo": cidade, "quadrante": q, "erro": erro})
        t_cel = time.perf_counter() - t_cel0
        out["custo_medido"].append({"cidade_alvo": cidade, "quadrante": q, "segundos": t_cel})
        log(f"[{cidade}_Q{q}] celula completa em {t_cel:.1f}s")
        OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    out["agregados_por_cidade"] = agregados_por_cidade(out["registros"])
    out["parou_por_orcamento"] = parou_por_orcamento
    out["orcamento_gpu_min"] = orcamento_min
    out["tempo_total_medido_s"] = sum(c["segundos"] for c in out["custo_medido"])
    out["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"Gravado: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
