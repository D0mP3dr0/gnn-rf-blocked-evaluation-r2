
"""MC Dropout uncertainty on the spatially blocked partition, current lineage, 16 seed-42 checkpoints."""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, math, sys, time, traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

AQUI = Path(__file__).resolve().parent
EVID = AQUI.parents[1]
PU_PATH = EVID / "_campanha_2026-09-13/passada_unica/passada_unica.py"
PRE_REG = EVID / "dados/planos/criterios_T14_mc_dropout_2026-09-13_v2.json"
LOG = AQUI / "t14.log"

NIVEIS = [0.50, 0.80, 0.90, 0.95, 0.99]
Z = {0.50: 0.674489750196, 0.80: 1.281551565545, 0.90: 1.644853626951,
     0.95: 1.959963984540, 0.99: 2.575829303549}
CANAIS = {"rssi": 3, "pl": 0}


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def passada(mod, model, graph, cfg, device, use_amp, seed_loader, estocastica):
    """Runs one forward pass over all nodes of the induced graph; returns predictions [N, 5] ordered by node."""
    ET_AT, ET_TT, ET_TA = mod.ET_AT, mod.ET_TT, mod.ET_TA
    nn_kw = {ET_AT: [cfg["k_antenna"]], ET_TT: [cfg["k_terrain"]], ET_TA: [cfg["k_antenna"]]}
    eval_bs = cfg["eval_batch_size"] or cfg["batch_size"]
    torch.manual_seed(seed_loader)
    loader = NeighborLoader(data=graph, num_neighbors=nn_kw, input_nodes=("terrain", None),
                            batch_size=eval_bs, shuffle=False, num_workers=0)
    model.eval()
    if estocastica:


        # an earlier version also put BatchNorm in batch-statistics mode, which shifted predictions by about 13 dB; this version activates dropout layers only.
        from torch_geometric.nn import GATv2Conv
        for m in model.modules():
            if isinstance(m, (torch.nn.Dropout, GATv2Conv)):
                m.train()
    n_p = graph["terrain"].x.shape[0]
    P, T, I = [], [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            bs = b["terrain"].batch_size
            ctx = torch.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                out = model(b)
            P.append(out["predictions"][:bs].float().detach().cpu())
            T.append(b["terrain"].rf_targets[:bs].float().detach().cpu())
            I.append(b["terrain"].n_id[:bs].cpu())
    po, to, vi = mod._scatter_por_semente(torch.cat(I), torch.cat(P), torch.cat(T), n_p)
    model.eval()
    return po.numpy(), to.numpy(), vi.numpy()


def crps_ensemble(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes CRPS per node from samples [m, N] and targets y [N], using the closed-form expression over ordered samples."""
    m = samples.shape[0]
    xs = np.sort(samples, axis=0)
    i = np.arange(1, m + 1, dtype=np.float64)[:, None]
    term1 = np.mean(np.abs(samples - y[None, :]), axis=0)
    term2 = np.sum((2.0 * i - m - 1.0) * xs, axis=0) / (m * m)
    return term1 - term2


def interval_score(lo, hi, y, alpha):
    w = hi - lo
    return w + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)


def picp_largura_score(lo, hi, y, alpha):
    dentro = (y >= lo) & (y <= hi)
    return {"picp": float(dentro.mean()), "largura_media_db": float((hi - lo).mean()),
            "interval_score_medio": float(interval_score(lo, hi, y, alpha).mean())}


def quantil_conformal(scores: np.ndarray, nivel: float) -> float:
    n = scores.shape[0]
    k = math.ceil((n + 1) * nivel)
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def avaliar_metodos(S_test, mu_t, sd_t, y_t, yhat_eval_t, mu_v, sd_v, y_v, yhat_eval_v):
    """Evaluates calibration methods given test samples S_test [m, N] and mu/sd/y/yhat on the test set [N]; *_v holds the matching validation-set arrays used for calibration."""
    res = {"A_mc_percentil": {}, "B_mc_gaussiano": {}, "C_conformal_split": {},
           "D_residual_deterministico": {}}
    sd_floor_t = np.maximum(sd_t, 1e-3)
    sd_floor_v = np.maximum(sd_v, 1e-3)
    esc_v_C = np.abs(y_v - mu_v) / sd_floor_v
    esc_v_D = np.abs(y_v - yhat_eval_v)
    for nv in NIVEIS:
        a = 1.0 - nv
        key = f"{int(round(nv*100))}"
        lo = np.percentile(S_test, 100 * a / 2, axis=0)
        hi = np.percentile(S_test, 100 * (1 - a / 2), axis=0)
        res["A_mc_percentil"][key] = picp_largura_score(lo, hi, y_t, a)
        z = Z[nv]
        res["B_mc_gaussiano"][key] = picp_largura_score(mu_t - z * sd_t, mu_t + z * sd_t, y_t, a)
        qC = quantil_conformal(esc_v_C, nv)
        res["C_conformal_split"][key] = picp_largura_score(mu_t - qC * sd_floor_t, mu_t + qC * sd_floor_t, y_t, a)
        res["C_conformal_split"][key]["quantil_calibrado"] = qC
        qD = quantil_conformal(esc_v_D, nv)
        res["D_residual_deterministico"][key] = picp_largura_score(yhat_eval_t - qD, yhat_eval_t + qD, y_t, a)
        res["D_residual_deterministico"][key]["quantil_calibrado_db"] = qD
    nll = 0.5 * np.log(2 * np.pi * sd_floor_t ** 2) + (y_t - mu_t) ** 2 / (2 * sd_floor_t ** 2)
    res["globais"] = {
        "n": int(y_t.shape[0]),
        "mae_media_mc_db": float(np.abs(mu_t - y_t).mean()),
        "mae_eval_db": float(np.abs(yhat_eval_t - y_t).mean()),
        "desvio_mc_medio_db": float(sd_t.mean()),
        "desvio_mc_quantis_db": {q: float(np.percentile(sd_t, q)) for q in (5, 25, 50, 75, 95)},
        "nll_gaussiana_media": float(nll.mean()),
        "crps_ensemble_medio_db": float(crps_ensemble(S_test, y_t).mean()),
        "n_calibracao_val": int(y_v.shape[0]),
    }
    return res


def processar_celula(pu, mod, cidade, q, n_pass, device, use_amp, out_dir):
    run_label = f"c0c1cf_{cidade}_s42_Q{q}_g10b2"
    out_json = out_dir / f"mc_dropout_bloqueado_{cidade}_Q{q}_s42.json"
    if out_json.exists():
        log(f"[{run_label}] já existe — pulando (reuso de artefato)")
        return json.loads(out_json.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    prep = pu.preparar_celula(mod, run_label)
    ref, cfg = prep["ref"], prep["cfg"]
    ckpt = pu.TREINOS_DIR / run_label / "checkpoints" / "checkpoint_best.pt"
    model = pu.montar_modelo(mod, None, ref, device, "gnn")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    del ck
    log(f"[{run_label}] preparado em {time.perf_counter()-t0:.0f}s; hash_test_ok={prep['hash_ok']}")

    g_test, g_val = prep["graphs"]["test"], prep["graphs"]["val"]
    PLMAX = mod.PL_TARGET_MAX_VALID


    po_t, to_t, vi_t = passada(mod, model, g_test, cfg, device, use_amp, 0, estocastica=False)
    po_v, to_v, vi_v = passada(mod, model, g_val, cfg, device, use_amp, 0, estocastica=False)
    assert vi_t.all() and vi_v.all(), "nó não visitado na passada eval"
    mae_eval_total = float(np.abs(po_t[:, 3] - to_t[:, 3]).mean())
    mae_ref = float(ref["selecao"]["test_no_melhor_ckpt"]["mae_rssi_db"])
    diff_repro = abs(mae_eval_total - mae_ref)
    c1_ok = bool(prep["hash_ok"] and diff_repro <= 1.5e-4)
    log(f"[{run_label}] C1: mae_eval={mae_eval_total:.6f} ref={mae_ref:.6f} diff={diff_repro:.2e} ok={c1_ok}")


    n_t, n_v = po_t.shape[0], po_v.shape[0]
    S_t = np.zeros((n_pass, n_t, 2), dtype=np.float32)
    S_v = np.zeros((n_pass, n_v, 2), dtype=np.float32)
    t1 = time.perf_counter()
    for s in range(n_pass):
        p_t, _, v1 = passada(mod, model, g_test, cfg, device, use_amp, 1000 + s, estocastica=True)
        p_v, _, v2 = passada(mod, model, g_val, cfg, device, use_amp, 1000 + s, estocastica=True)
        assert v1.all() and v2.all()
        S_t[s, :, 0], S_t[s, :, 1] = p_t[:, 0], p_t[:, 3]
        S_v[s, :, 0], S_v[s, :, 1] = p_v[:, 0], p_v[:, 3]
        if (s + 1) % 10 == 0:
            log(f"[{run_label}] passada {s+1}/{n_pass} ({time.perf_counter()-t1:.0f}s)")
        torch.cuda.empty_cache()
    del model
    gc.collect(); torch.cuda.empty_cache()

    valid_t = to_t[:, 0] < PLMAX
    valid_v = to_v[:, 0] < PLMAX
    resultado = {
        "celula": f"{cidade}_Q{q}", "cidade": cidade, "quadrante": q, "seed_treino": 42,
        "run_label": run_label, "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt),
        "rf_data_file": cfg["rf_data_file"], "rf_data_sha256_run": ref.get("dataset", {}).get("rf_data_sha256") if isinstance(ref.get("dataset"), dict) else None,
        "idx_sha256_global_test_run": prep["sha_test_ref"], "idx_sha256_global_test_recomputado": prep["sha_test_recomputado"],
        "n_passadas": n_pass, "n_test": int(n_t), "n_val": int(n_v),
        "n_test_valido": int(valid_t.sum()), "n_val_valido": int(valid_v.sum()),
        "frac_valido_test": float(valid_t.mean()),
        "C1_custodia": {"hash_test_ok": bool(prep["hash_ok"]), "mae_rssi_eval_total_db": mae_eval_total,
                        "mae_rssi_ref_run_db": mae_ref, "diff_db": diff_repro, "ok": c1_ok},
        "populacoes": {},
    }

    pops = [("test_valido_rssi", valid_t, 1, valid_v), ("test_todos_rssi", np.ones(n_t, bool), 1, np.ones(n_v, bool)),
            ("test_valido_pl", valid_t, 0, valid_v)]
    for nome, m_t, ch, m_v in pops:
        col = 3 if ch == 1 else 0
        St = S_t[:, m_t, ch].astype(np.float64)
        Sv = S_v[:, m_v, ch].astype(np.float64)
        mu_t, sd_t = St.mean(0), St.std(0)
        mu_v, sd_v = Sv.mean(0), Sv.std(0)
        resultado["populacoes"][nome] = avaliar_metodos(
            St, mu_t, sd_t, to_t[m_t, col].astype(np.float64), po_t[m_t, col].astype(np.float64),
            mu_v, sd_v, to_v[m_v, col].astype(np.float64), po_v[m_v, col].astype(np.float64))
        del St, Sv

    np.savez_compressed(out_dir / f"mc_dropout_bloqueado_{cidade}_Q{q}_s42.npz",
                        pos_m=prep["pos_m"][prep["parts_local"]["test"]].astype(np.float32),
                        std_rssi=S_t[:, :, 1].std(0).astype(np.float32),
                        mean_rssi=S_t[:, :, 1].mean(0).astype(np.float32),
                        y_rssi=to_t[:, 3].astype(np.float32), valido=valid_t)
    resultado["tempo_s"] = float(time.perf_counter() - t0)
    resultado["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    out_json.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"[{run_label}] gravado {out_json.name} em {resultado['tempo_s']:.0f}s")
    del S_t, S_v, prep
    gc.collect(); torch.cuda.empty_cache()
    return resultado


def agregar(resultados, out_dir, n_pass):
    validas = [r for r in resultados if r["C1_custodia"]["ok"]]
    invalidas = [r["celula"] for r in resultados if not r["C1_custodia"]["ok"]]
    agg = {"artefato_tipo": "agregado_T14_mc_dropout_bloqueado", "status": "provisorio (pendente de contra-auditoria)",
           "pre_registro": str(PRE_REG), "pre_registro_sha256": sha256_file(PRE_REG),
           "script_sha256": sha256_file(Path(__file__)), "n_passadas": n_pass,
           "n_celulas": len(resultados), "celulas_validas_C1": [r["celula"] for r in validas],
           "celulas_invalidas_C1": invalidas, "populacoes": {}}
    for pop in ("test_valido_rssi", "test_todos_rssi", "test_valido_pl"):
        P = {"metodos": {}, "globais": {}}
        for met in ("A_mc_percentil", "B_mc_gaussiano", "C_conformal_split", "D_residual_deterministico"):
            P["metodos"][met] = {}
            for nv in NIVEIS:
                key = f"{int(round(nv*100))}"
                picps = np.array([r["populacoes"][pop][met][key]["picp"] for r in validas])
                larg = np.array([r["populacoes"][pop][met][key]["largura_media_db"] for r in validas])
                isc = np.array([r["populacoes"][pop][met][key]["interval_score_medio"] for r in validas])
                dentro = np.abs(picps - nv) <= 0.05
                P["metodos"][met][key] = {
                    "nominal": nv, "picp_media": float(picps.mean()) if len(picps) else None,
                    "picp_min": float(picps.min()) if len(picps) else None, "picp_max": float(picps.max()) if len(picps) else None,
                    "largura_media_db": float(larg.mean()) if len(larg) else None,
                    "interval_score_medio": float(isc.mean()) if len(isc) else None,
                    "celulas_dentro_de_0.05": int(dentro.sum()), "por_celula_picp": {r["celula"]: float(p) for r, p in zip(validas, picps)},
                }
            # calibration criterion: 80/90/95% nominal within 0.05 in >= 12 of 16 cells
            ok_cel = [all(abs(r["populacoes"][pop][met][k]["picp"] - nv) <= 0.05
                          for k, nv in (("80", .80), ("90", .90), ("95", .95))) for r in validas]
            P["metodos"][met]["C2_celulas_ok_80_90_95"] = int(sum(ok_cel))
            P["metodos"][met]["C2_passa"] = bool(sum(ok_cel) >= 12)
        for g in ("mae_media_mc_db", "mae_eval_db", "desvio_mc_medio_db", "nll_gaussiana_media", "crps_ensemble_medio_db"):
            v = np.array([r["populacoes"][pop]["globais"][g] for r in validas])
            P["globais"][g] = {"media": float(v.mean()) if len(v) else None, "min": float(v.min()) if len(v) else None,
                               "max": float(v.max()) if len(v) else None}
        agg["populacoes"][pop] = P
    prim = agg["populacoes"]["test_valido_rssi"]["metodos"]
    A_ok, C_ok = prim["A_mc_percentil"]["C2_passa"], prim["C_conformal_split"]["C2_passa"]
    if A_ok:
        frase = "in-domain calibrated MC-dropout uncertainty"
    elif C_ok:
        frase = "MC-dropout dispersion calibrated post hoc by split-conformal on spatially separated validation blocks"
    else:
        frase = "claim de incerteza calibrada removido; tabela reportada como limitacao, com o baseline D ao lado"
    agg["veredito_C3"] = {"A_passa_C2": A_ok, "C_passa_C2": C_ok, "frase_permitida": frase}
    agg["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "agregado_T14.json").write_text(json.dumps(agg, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"AGREGADO: A_passa={A_ok} C_passa={C_ok} -> {frase}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celulas", default="bauru:1,bauru:2,bauru:3,bauru:4,campinas:1,campinas:2,campinas:3,campinas:4,"
                                          "lins:1,lins:2,lins:3,lins:4,sorocaba:1,sorocaba:2,sorocaba:3,sorocaba:4")
    ap.add_argument("--passadas", type=int, default=50)
    ap.add_argument("--smoke", action="store_true", help="grava em smoke/ (nunca vira artefato)")
    args = ap.parse_args()
    if not PRE_REG.exists():
        raise SystemExit("pré-registro ausente — não roda")
    out_dir = AQUI / "smoke" if args.smoke else AQUI
    out_dir.mkdir(exist_ok=True)
    pu = import_module(PU_PATH, "passada_unica_harness")
    mod = pu.import_module(pu.TRAIN_GNN_SCRIPT, "train_gnn_c0_spatial_frozen")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("CUDA indisponível — venv errada?")
    use_amp = True
    log(f"T14 início; device={device}; pré-registro sha256={sha256_file(PRE_REG)}; passadas={args.passadas}")
    resultados = []
    for spec in args.celulas.split(","):
        cidade, q = spec.split(":")
        try:
            resultados.append(processar_celula(pu, mod, cidade, int(q), args.passadas, device, use_amp, out_dir))
        except Exception:
            log(f"[{cidade}_Q{q}] ERRO\n" + traceback.format_exc())
            gc.collect(); torch.cuda.empty_cache()
    agregar(resultados, out_dir, args.passadas)
    log("T14 fim")


if __name__ == "__main__":
    main()
