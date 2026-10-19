"""common.py — utilitarios da Fase B (R4 IEEE Access, protocolo 14/10/2026). Nada aqui escreve fora de fase_B/."""
import hashlib, json, datetime
from pathlib import Path
import numpy as np

RAIZ = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF")
FASE_B = RAIZ / "gnn_rf_ieee_access/_revisao_2026-10-14/fase_B"
E = RAIZ / "gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO"
NPZ = RAIZ / "predictions/c0"
GD = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QS = ["Q1", "Q2", "Q3", "Q4"]
SEEDS = [42, 43, 44, 45, 46]
BRACOS = ["gnn", "mlp"]
PL_MAX_VALID = 299.0
LIM_COV = -85.0


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_idx(idx):
    a = np.ascontiguousarray(np.sort(np.asarray(idx, dtype=np.int64)))
    return hashlib.sha256(a.tobytes()).hexdigest()


def run_label(braco, c, s, q):
    return f"{'c0c1cf' if braco == 'gnn' else 'mlpcf'}_{c}_s{s}_{q}_g10b2"


def run_json_path(braco, c, s, q):
    rl = run_label(braco, c, s, q)
    return E / "treinos" / rl / f"run_{rl}.json"


def npz_path(braco, c, q, s):
    return NPZ / f"{braco}_{c}_{q}_s{s}.npz"


def cftudo_path(c, q):
    return GD / f"transfer_dataset_{c}_v19_{q}_enriched_cftudo.pt"


def carimbo(script_path, insumos):
    """insumos: dict nome->path ; devolve cabecalho de proveniencia (sha256 do script e de cada insumo)."""
    return {
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "insumos_sha256": {k: sha256_file(v) for k, v in insumos.items()},
        "data": datetime.date.today().isoformat(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "provisorio": True,
    }


def pops(target):
    """Mascaras das 4 populacoes a partir do alvo (N,5) do npz (= cftudo y)."""
    valid = target[:, 0] < PL_MAX_VALID
    cov85 = valid & (target[:, 3] >= LIM_COV)
    return {"ALL": np.ones(len(target), bool), "VALID": valid, "COV85": cov85, "SENT": ~valid}


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def custodia():
    j = json.load(open(FASE_B / "B0_custodia.json"))
    return {(l["braco"], l["cidade"], l["Q"], l["seed"]): bool(l["passa"]) for l in j["linhas"]}


def cache(c, q):
    return np.load(FASE_B / "_cache" / f"{c}_{q}.npz")


def fspl_db(dist_m, freq_mhz):
    """FSPL em dB; d_km com piso de 1 m (igual a _fspl_db de rf_diagnostic_metrics.py:79-82)."""
    d_km = np.maximum(np.asarray(dist_m, np.float32), np.float32(1.0)) / np.float32(1000.0)
    return (20.0 * np.log10(d_km) + 20.0 * np.log10(np.asarray(freq_mhz, np.float32)) + 32.44).astype(np.float32)


def resumo_stat(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return {"n": 0, "media": None, "mediana": None, "p95": None}
    return {"n": int(x.size), "media": float(x.mean()), "mediana": float(np.median(x)), "p95": float(np.percentile(x, 95))}
