
"""R3 graph-census table, with ranges over the 16 cells."""
import json, glob, statistics
from pathlib import Path

EVID = Path(__file__).resolve().parents[1]
R3 = EVID.parents[1] / "REVISAO_R3" / "novas"

cel = {}
for f in sorted(glob.glob(str(EVID / "dados/baselines_v2/baselines_v2_*_Q*.json"))):
    nome = Path(f).stem.replace("baselines_v2_", "")
    ds = json.load(open(f, encoding="utf-8"))["dataset"]
    cel[nome] = ds
assert len(cel) == 16, len(cel)
ant_por_cidade = {}
for nome, ds in cel.items():
    ant_por_cidade.setdefault(nome.split("_")[0], set()).add(ds["n_antenas"])
assert all(len(v) == 1 for v in ant_por_cidade.values())
ant = {c: next(iter(v)) for c, v in ant_por_cidade.items()}
e_at = [ds["n_arestas_ant_ter"] for ds in cel.values()]
nb = [ds["rf_data_bytes"] for ds in cel.values()]
nt = {ds["n_nos_terrain"] for ds in cel.values()}
assert nt == {12960000}
fmin = min(ds["freq_antenas_mhz"]["min"] for ds in cel.values())
fmax = max(ds["freq_antenas_mhz"]["max"] for ds in cel.values())

cen = json.load(open(EVID / "dados/censo_grafo/fase0_graph_census.json", encoding="utf-8"))
tt = set()
for q, v in cen["quadrantes"].items():
    for k, e in v["edge_types"].items():
        if "connects_to" in k:
            tt.add(e["num_edges"])
assert len(tt) == 1
tt = tt.pop()

vram, tempo = [], []
runs = sorted(glob.glob(str(EVID / "dados/treinos_c1/run_c0c1cf_*.json")))
for f in runs:
    r = json.load(open(f, encoding="utf-8"))
    c = r.get("custo", {})
    if "vram_peak_mb" in c:
        vram.append(c["vram_peak_mb"])
    for k in ("tempo_total_s", "wall_clock_s", "elapsed_s", "tempo_s"):
        if k in c:
            tempo.append(c[k]); break
assert len(vram) == 80 and len(tempo) == 80, (len(runs), len(vram), len(tempo))

fmt = lambda n: "\\num{%d}" % n
ordem = ["lins", "bauru", "sorocaba", "campinas"]
ant_txt = ", ".join(f"{fmt(ant[c])} in {c.capitalize()}" for c in ordem)
tex = r"""\begin{table}[t]
\caption{Graph census and training cost. Counts are read from the 16 quadrant graph files; the terrain grid, and therefore the terrain-to-terrain edge count, is the same in every cell. Antenna nodes are all licensed transmitters of the city: """ + ant_txt + r""". Costs are over the 80 blocked runs on one NVIDIA RTX 5070 Ti (16~GB).}
\label{tab:graph_census}
\centering
\footnotesize
\begin{tabular}{@{}p{0.56\columnwidth}@{\hspace{4pt}}r@{}}
\toprule
Terrain nodes per cell & """ + fmt(12960000) + r""" \\
Antenna nodes per cell & """ + fmt(min(ant.values())) + "--" + fmt(max(ant.values())) + r""" \\
Antenna$\to$terrain edges (and reverse) & """ + fmt(min(e_at)) + "--" + fmt(max(e_at)) + r""" \\
Terrain$\to$terrain edges (8-neighbour) & """ + fmt(tt) + r""" \\
Carrier frequencies & """ + f"{fmin:.0f}--{fmax:.0f}~MHz" + r""" \\
Serialised graph file & """ + f"{min(nb)/1e9:.1f}--{max(nb)/1e9:.1f}~GB" + r""" \\
Sampled neighbours per node (antenna / terrain) & 20 / 8 \\
Training batch (seed nodes) & """ + fmt(24576) + r""" \\
Epochs (head quadrant / transferred) & 35 / 20 \\
Trainable parameters & """ + fmt(1812515) + r""" \\
Peak training VRAM (min--max, median) & """ + f"{min(vram):.0f}--{max(vram):.0f}~MB, {statistics.median(vram):.0f}" + r""" \\
Wall-clock per run (min--max, median) & """ + f"{min(tempo):.0f}--{max(tempo):.0f}~s, {statistics.median(tempo):.0f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
(R3 / "tab_censo_grafo.tex").write_text(tex, encoding="utf-8")
rastro = {"antenas_por_cidade": ant, "arestas_ant_ter_min_max": [min(e_at), max(e_at)], "arestas_ter_ter": tt,
          "freq_mhz": [fmin, fmax], "bytes_min_max": [min(nb), max(nb)], "vram_mb": [min(vram), max(vram), statistics.median(vram)],
          "tempo_s": [min(tempo), max(tempo), statistics.median(tempo)], "n_runs": len(runs)}
(EVID / "_inventario_revisores_2026-09-13/novas/censo_r3_rastro.json").write_text(json.dumps(rastro, indent=1), encoding="utf-8")
print(json.dumps(rastro))
