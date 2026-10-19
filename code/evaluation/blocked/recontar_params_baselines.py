
"""Recounts trainable parameters of the four learned-baseline checkpoints (Bauru Q2, seed 42) directly from their state_dict."""
import torch, json
from pathlib import Path
BASE = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\baseline_colab_\logs\version_20.1\logs\version_20")
out = {}
for m in ["rem_gnn", "radiounet", "radiogat", "rme_gan"]:
    for q in ["Q1", "Q2"]:
        p = BASE / f"{m}_bauru_s42_{q}" / "checkpoints" / "checkpoint_best.pt"
        if not p.exists():
            out[f"{m}_{q}"] = "AUSENTE"; continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck.get("generator_state_dict", ck)) if isinstance(ck, dict) else ck
        tens = {k: v for k, v in sd.items() if hasattr(v, "numel")}
        gen = {k: v for k, v in tens.items() if not k.startswith(("disc", "discriminator", "D."))}
        out[f"{m}_{q}"] = {
            "chaves_top": sorted({k.split(".")[0] for k in tens}),
            "numel_total": int(sum(v.numel() for v in tens.values())),
            "numel_sem_buffers_bn": int(sum(v.numel() for k, v in tens.items() if not k.endswith(("running_mean", "running_var", "num_batches_tracked")))),
            "tem_tt_msg": any("tt_msg" in k for k in tens),
            "numel_tt_msg": int(sum(v.numel() for k, v in tens.items() if "tt_msg" in k)),
            "outras_chaves_ckpt": [k for k in ck.keys()][:12] if isinstance(ck, dict) else None,
        }
print(json.dumps(out, indent=1))
