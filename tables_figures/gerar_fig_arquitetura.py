
"""Architecture figure (Fig. 2), redrawn as a vector graphic read from the model code."""
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

INV = Path(__file__).resolve().parent
OUT = INV / "novas"; OUT.mkdir(exist_ok=True)
R2 = INV.parents[1].parents[0] / "REVISAO_R2" / "novas"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, ax = plt.subplots(figsize=(11.5, 5.2))
ax.set_xlim(0, 115); ax.set_ylim(0, 52); ax.axis("off")

C_IN, C_ENC_T, C_ENC_A, C_GNN, C_DEC, C_STAGE = "#dbe8f4", "#a9c8e6", "#f4c9a4", "#f7f7f5", "#cfe6cf", "#efefef"

def stage(x, w, title, sub):
    ax.add_patch(FancyBboxPatch((x, 2), w, 46, boxstyle="round,pad=0.4,rounding_size=1.2", fc=C_STAGE, ec="#9a9a9a", lw=0.8))
    ax.text(x + w / 2, 50.3, title, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(x + w / 2, 46.3, sub, ha="center", va="center", fontsize=9)

def box(x, y, w, h, text, fc, fs=8, ec="#444"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)

def trap(x, y, w, h, text, fc):
    ax.add_patch(Polygon([(x, y), (x + w, y + h * 0.18), (x + w, y + h * 0.82), (x, y + h)], closed=True, fc=fc, ec="#444", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, linespacing=1.25)

def arrow(x0, y0, x1, y1, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, lw=0.9, color="#222", linestyle=ls))


stage(1, 34, "STAGE 1", "Input and node encoders")
box(3, 30, 12, 11, "Terrain nodes\n$x_t \\in \\mathbb{R}^{18}$\n12.96 M nodes", C_IN)
trap(17, 29, 15, 13, "Terrain encoder\n18 → 256 → 256\nBN + LeakyReLU(0.2)", C_ENC_T)
box(3, 8, 12, 11, "Antenna nodes\n$x_a \\in \\mathbb{R}^{6}$\n175–4252 nodes", C_IN)
trap(17, 7, 15, 13, "Antenna encoder\n6 → 128 → 128 → 256\nBN + LeakyReLU(0.2)", C_ENC_A)
arrow(15, 35.5, 17, 35.5); arrow(15, 13.5, 17, 13.5)


stage(38, 40, "STAGE 2", "Heterogeneous message passing, ×4 layers")
ax.add_patch(FancyBboxPatch((41, 9), 30, 33, boxstyle="round,pad=0.3,rounding_size=1.0", fc="white", ec="#777", lw=0.8, ls="--"))
ax.text(56, 40.2, "one HeteroConv layer (sum over relations)", ha="center", va="center", fontsize=7.5, style="italic")
box(43, 31, 26, 6.5, "GATv2  antenna → terrain\n4 heads × 64, edge_dim = 2", C_GNN, fs=7.8)
box(43, 23, 26, 6.5, "GATv2  terrain → terrain\n4 heads × 64, edge_dim = 2", C_GNN, fs=7.8)
box(43, 15, 26, 6.5, "SAGEConv  terrain → antenna\nmean aggregation", C_GNN, fs=7.8)
box(45, 10.2, 22, 3.6, "LeakyReLU · Dropout(0.1) · residual · LayerNorm", "#e9f1ea", fs=7)
arrow(32, 35.5, 43, 34.2); arrow(32, 13.5, 43, 18.2)
ax.text(56, 5.5, "terrain and antenna embeddings $h \\in \\mathbb{R}^{256}$", ha="center", va="center", fontsize=7.8)
arrow(71, 25.5, 80, 25.5)


stage(79, 35, "STAGE 3", "Physics-constrained decoder")
box(81, 20, 9.5, 11, "MLP\n256 → 256\n→ 128 → 5\nGELU +\nDropout", C_DEC, fs=7.5)
outs = [("$L_\\mathrm{total}$: softplus, clamp [0, 200] dB", 39.5),
        ("$L_\\mathrm{veg}$: softplus, clamp [0, 50] dB", 32.5),
        ("$L_\\mathrm{ter}$: softplus, clamp [0, 30] dB", 25.5),
        ("RSSI: $-150 + 150\\,\\sigma(\\cdot)$ dBm", 18.5),
        ("$p_\\mathrm{cov}$: $\\sigma(\\cdot)$ ∈ [0, 1]", 11.5)]
for t, y in outs:
    box(93, y - 2.6, 19.5, 5.2, t, C_DEC, fs=7.5)
    arrow(90.5, 25.5, 93, y)
ax.text(96.5, 5.5, "free-space floor, distance monotonicity, variance and NDVI sign terms\nare penalised in the loss and verified at the output", ha="center", va="center", fontsize=6.8, style="italic")

fig.tight_layout()
fig.savefig(OUT / "fig_arquitetura.pdf"); fig.savefig(OUT / "fig_arquitetura.png", dpi=220)
R2.mkdir(parents=True, exist_ok=True)
for f in ("fig_arquitetura.pdf", "fig_arquitetura.png"):
    shutil.copy(OUT / f, R2 / f)
print("ok", R2)
