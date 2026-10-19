"""Check the 16 x 3 partition masks against PARTITION_HASHES.json, the run logs and the prediction files.

Usage (from the package root): python partitions/verify_partitions.py
Exits with a non-zero status on the first mismatch.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARMS = {"c0_gnn": "c0c1cf", "c0_mlp": "mlpcf", "c1_noreg": "c1noreg", "c2_nomono": "c2nomono", "ba1": "ba1rus"}
SEEDS = [42, 43, 44, 45, 46]


def idx_sha256(idx):
    return hashlib.sha256(np.ascontiguousarray(np.sort(np.asarray(idx, dtype=np.int64))).tobytes()).hexdigest()


def fail(msg):
    print("MISMATCH:", msg)
    sys.exit(1)


def main():
    doc = json.loads((ROOT / "partitions/PARTITION_HASHES.json").read_text(encoding="utf-8"))
    n_runs = n_pred = 0
    for cell, rec in doc["cells"].items():
        city, q = cell.split("_")
        z = np.load(ROOT / rec["file"])
        n = int(z["n_nodes"])
        idx = {k: np.flatnonzero(np.unpackbits(z[f"{k}_bits"], count=n).astype(bool)) for k in ("train", "val", "test")}
        for k, v in idx.items():
            if idx_sha256(v) != rec[f"idx_sha256_{k}"]:
                fail(f"{cell} {k}: mask hash differs from PARTITION_HASHES.json")
        for arm, prefix in ARMS.items():
            for s in SEEDS:
                label = f"{prefix}_{city}_s{s}_{q}_g10b2"
                run = json.loads((ROOT / f"runs/{arm}/{label}/run_{label}.json").read_text(encoding="utf-8"))
                for k in ("train", "val", "test"):
                    if run["particoes"][k]["idx_sha256_global"] != rec[f"idx_sha256_{k}"]:
                        fail(f"{label} {k}: run log hash differs")
                n_runs += 1
                pred = np.load(ROOT / f"predictions/{arm}/{arm}_{city}_{q}_s{s}.npz")
                if not np.array_equal(np.sort(pred["idx_global"]), idx["test"]):
                    fail(f"{arm} {cell} s{s}: prediction rows are not the test partition")
                n_pred += 1
    print(f"OK: {len(doc['cells'])} cells, {n_runs} run logs and {n_pred} prediction files match the partition masks")


if __name__ == "__main__":
    main()
