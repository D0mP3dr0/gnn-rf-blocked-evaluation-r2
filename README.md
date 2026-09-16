# GNN-RF under spatially blocked evaluation — reproducibility package (R3)

Code, results, and the 16 seed-42 checkpoints for the manuscript
*Physics-Consistent Multi-Target RF Field Reconstruction over Heterogeneous
Terrain Graphs under Spatially Blocked Evaluation* (IEEE Access, manuscript
Access-2026-29524, revision R3).

## What is here

- `generator/` — the deterministic reference-field generator
  (`generate_realistic_coverage.py`, `enrich_rf_targets.py`), the ETL that
  builds the graph tensors (`prepare_transfer_dataset_v19.py`), the
  generator self-consistency check (`contrafactual_alvo_completo.py`, the
  script behind the 3.02e-4 dB recomputation figure), and the target
  corrector that replaced the placeholder slope draw with the DEM slope
  (`contrafactual_slope_real_construir.py` / `..._medir.py`).
- `model/` — GNN-RF encoders, the heterogeneous GATv2 stack, the
  physics-constrained decoder, the physics-informed loss, and the
  FSPL/Okumura-Hata/COST-231 analytical models.
- `training/` — the spatial block partition (`spatial_cv.py`), diagnostic
  metrics (`rf_diagnostic_metrics.py`), and `training/frozen/`: the two
  frozen training scripts behind every reported run
  (`train_gnn_c0_spatial.py` for GNN-RF, `train_mlp_c0_spatial.py` for the
  capacity-matched graph-free control), and the PowerShell queues used to
  launch the training runs.
- `evaluation/` — analytical baselines, the paired statistical test, the
  nearest-valid-neighbor contrast, blocked cross-city transfer,
  FSPL-floor admissibility and parameter recount, spatial error analysis,
  aggregators, integrity checks, and the blocked MC Dropout uncertainty
  evaluation (`evaluation/t14_mc_dropout/`, including the 16 per-cell
  `.npz` files with the raw stochastic-pass samples).
- `tables_figures/` — the generators and the ten new/updated LaTeX tables and
  five figures the R3 manuscript reads via `\input`/`\includegraphics`.
- `manuscript/` — the Supplementary Material appendix that documents the
  generator equation by equation.
- `results/` — the run JSON of every blocked training run
  (`c0c1cf_*` GNN-RF, `mlpcf_*` graph-free control, plus the
  from-scratch `c0c1cfsc_*` and reversed-chain `c0c1cfinv_*` control runs), their aggregates, the
  graph census, the geographic-split and label-shuffle controls of the earlier campaign, the MC Dropout
  per-cell JSONs and raw samples, the blocked-transfer result, the artifact
  manifest (`manifest.jsonl`, file name and SHA-256 per artifact), and the
  SHA-256 of the 16 reference fields.
- `checkpoints/` — the 16 seed-42 GNN-RF checkpoints (`checkpoint_best.pt`
  of every city x quadrant cell), renamed to
  `checkpoint_c0c1cf_<city>_s42_Q<n>_g10b2.pt`. The SHA-256 published in
  `results/manifest.jsonl` for each checkpoint identifies that exact binary;
  it is the first time this hash was computed and recorded, not a check
  against a pre-existing value.
- `LICENSE` (MIT, code) and `LICENSE-DATA` (CC BY 4.0, results and
  checkpoints).

## What is not here

The 16 reference fields (`*_cftudo.pt`, ~28 GB each) and the graph tensors
(`*_gpu.pt`) exceed repository limits and are **available from the
corresponding author upon request** (Google Drive). Their SHA-256 digests
and file names are listed in `results/hash_16_reference_fields.json`
(computed by `results/hash_16_reference_fields_compute.py`) and in
`results/manifest.jsonl`, so any externally obtained copy can be verified.

## Environment

Read from the provenance block of the run JSONs (`run_*.json` ->
`ambiente`): Python 3.11.9, PyTorch 2.10.0+cu128, CUDA 12.8, a single NVIDIA
RTX 5070 Ti (16 GB VRAM). PyTorch Geometric 2.8.0 (dev build) as installed in
the authors' reproduction environment; the exact PyG build is not recorded
per-run in the JSON provenance. Paths inside the scripts point to the
authors' machines and must be adapted.

## How to reproduce each table and figure

All commands assume the CUDA virtual environment
(`F:\S33_dslm\s33_amb_virtual\.venv\Scripts\python.exe` in the authors'
setup) and that the 16 reference fields and graph tensors have been obtained
separately (see "What is not here") and placed where `--rf-data-file` /
`--graph-file` point.

1. **Generate and correct a reference field** (one city/quadrant):
   `python generator/generate_realistic_coverage.py ...` then
   `python generator/enrich_rf_targets.py ...` then
   `python generator/contrafactual_slope_real_construir.py ...`.
   Verify with `python generator/contrafactual_alvo_completo.py ...`
   (reproduces the 3.02e-4 dB self-consistency figure).

2. **Train one GNN-RF cell** (produces one `run_c0c1cf_<city>_s42_<Q>_g10b2.json`
   and one `checkpoints/checkpoint_best.pt`):
   ```
   python training/frozen/train_gnn_c0_spatial.py \
     --run-label c0c1cf_bauru_s42_Q1_g10b2 \
     --rf-data-file transfer_dataset_bauru_v19_Q1_enriched_cftudo.pt \
     --graph-file  bauru_v19_Q1_gpu.pt \
     --epochs 35 --lr 1e-3 --seed 42 --split-seed 42 \
     --grid-km 10.0 --buffer-km 2.0 --split-frac 0.70,0.15,0.15 \
     --freq-mhz 900 --diag-freq-mhz 1800 --hash
   ```
   Repeat for Q2-Q4 with `--transfer-from` the previous quadrant's
   checkpoint, 20 epochs, `lr=2e-4` (Algorithm 1). Repeat the full chain for
   seeds 42-46 and for `train_mlp_c0_spatial.py` (graph-free control) to
   reproduce all 160 runs behind `results/runs_blocked/`.

3. **Calibrated analytical baselines**:
   `python evaluation/scripts/baselines_v3_por_particao.py` (reads the same
   antenna parameters and test partitions as the trained runs).

4. **Paired test (0.746 dB, p=2.59e-4)**:
   `python evaluation/scripts/teste_pareado_empiricos_c1v2.py`.

5. **Nearest-valid-neighbor contrast (Table 7)**:
   `python evaluation/scripts/contraste_1nn_vizinho_valido.py`.

6. **Blocked cross-city transfer (Sec. Cross-city transfer)**:
   `python evaluation/scripts/medir_transferencia_bloqueada.py`.

7. **FSPL-floor admissibility and parameter recount (Tables
   10 and 11)**:
   `python evaluation/scripts/medir_admissibilidade_e_params_2026-09-15.py`
   and `python evaluation/scripts/recontar_params_baselines.py`.

8. **Spatial error analysis (Sec. Spatial error analysis)**:
   `python evaluation/scripts/exportar_spatial_npz.py` (per-node signed error
   and coordinates).

9. **MC Dropout uncertainty (Table 13, Fig. 6)**:
   `python evaluation/t14_mc_dropout/t14_mc_dropout_bloqueado.py` (T=50
   stochastic passes, dropout layers only, on each of the 16 seed-42
   checkpoints).

10. **Capacity-matched control aggregation (Sec. Robustness and ablation)**:
    `python evaluation/scripts/agregar_t11_mlp_vs_gnn_4cidades_2026-09-15.py`.

11. **Cost table (Table 15)**: training time and peak VRAM are read from the
    `custo` block of the Bauru seed-42 run JSONs by
    `tables_figures/gerar_tabelas_rodada_escrita_2026-09-16.py`.

12. **Render the manuscript tables/figures**:
    `python tables_figures/gerar_tabela_figura_novas.py`,
    `gerar_tabela_tres_populacoes.py`, `gerar_tabelas_restantes.py`,
    `gerar_fig_arquitetura.py` (base set), then `gerar_tab_censo_r3.py` and
    `gerar_tabelas_rodada_escrita_2026-09-16.py` (patches the R3 `_v2`/`_v3`
    tables: `tab_blocked_16_tres_pop_v2`, `tab_ablacao_estrutural_v2`,
    `tab_ood_bloqueada`, `tab_contraste_1nn`, `tab_custo_v2`/`_v3`,
    `tab_arquitetura_v2`, `tab_admissibilidade_v2`/`_v3`), and
    `gerar_figs_R3_ajuste_seta_legenda_2026-09-15.py` for the figure legend
    adjustments. Outputs land in `tables_figures/novas/`.

Every script above reports the SHA-256 of its own source and of the inputs
it reads inside its output JSON (`script_sha256`, `origem_script_sha256`,
`*_sha256` fields), so a rerun can be checked artifact by artifact against
`results/manifest.jsonl`.

## Data availability

The 16 reference fields (`*_cftudo.pt`, ~28 GB each) and the graph tensors
(`*_gpu.pt`) are available from the corresponding author upon request. Their
file names and SHA-256 digests are in `results/hash_16_reference_fields.json`
and in `results/manifest.jsonl`.

## License

Code is released under the MIT License (`LICENSE`). Results, run records,
and checkpoints are released under CC BY 4.0 (`LICENSE-DATA`).

## Citation

See `CITATION.cff`.
