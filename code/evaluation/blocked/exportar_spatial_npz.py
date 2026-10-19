"""Exports per-node signed RSSI error and coordinates from the MC Dropout .npz files for the spatial error analysis."""
import csv
import json
from pathlib import Path

import numpy as np

SRC = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\results\version20_gold\spatial_error_data")
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "dados" / "spatial_error" / "spatial_error_npz_export.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted(SRC.glob("*.npz")):
    z = np.load(f, allow_pickle=True)
    keys = list(z.keys())
    row = {"arquivo": f.name, "chaves": ";".join(keys)}
    for k in keys:
        a = z[k]
        if a.ndim == 0:
            row[k] = a.item()
        elif np.issubdtype(a.dtype, np.number):
            err = a.astype(np.float64)
            row[f"{k}_shape"] = "x".join(map(str, a.shape))
            row[f"{k}_mae"] = float(np.nanmean(np.abs(err)))
            row[f"{k}_bias"] = float(np.nanmean(err))
    rows.append(row)
    print(f.name, keys)

cols = sorted({c for r in rows for c in r})
with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader(); w.writerows(rows)
print("[OK]", OUT, f"({len(rows)} npz)")
