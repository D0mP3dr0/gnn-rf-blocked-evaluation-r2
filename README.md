# Physics-Informed Multi-Target RF Field Reconstruction over Heterogeneous Terrain Graphs under Spatially Blocked Evaluation — code and data

L. F. C. Seelig and R. M. Salles (Military Institute of Engineering, IME, Brazil).
DOI: https://doi.org/10.5281/zenodo.23377937

Code is released under the MIT License (`LICENSE`); data (run logs, checkpoints,
prediction files, partition masks, reference-field hashes and evaluation outputs)
under CC BY 4.0 (`LICENSE-DATA`). Every file is listed with its SHA-256 digest in
`SHA256SUMS.txt`; verify with `sha256sum -c SHA256SUMS.txt` from this folder.

## Items listed in the Data Availability statement

| Item | Where |
|---|---|
| Target generator (SHA-256 95ea0423…) | `code/generate_realistic_coverage.py`, with `code/01_data/prepare_transfer_dataset_v19.py` (45c16d43…) and `code/data_raw/enrich_rf_targets.py` (5d38012e…) |
| Target corrector (ebeaf759…) | `code/corrector/contrafactual_alvo_completo.py` (plus the slope-correction scripts in the same folder) |
| 16 reference fields with their hashes | `reference_fields/hash_16_reference_fields.json` (file name, size and SHA-256 of each field; the fields themselves, about 452 GB in total, are not in this deposit) |
| Partition indices with their hashes | `partitions/` (16 cells x train/validation/test masks, `PARTITION_HASHES.json`, `verify_partitions.py`) |
| Evaluation and baseline scripts | `code/evaluation/`, `code/04_baselines/empirical_models.py`, `code/ba1/` |
| Training scripts | `code/03_training/train_gnn_c0_spatial.py`, `code/03_training/train_mlp_c0_spatial.py`, `code/retrained_arms/`, `code/ba1/` |
| 160 run logs, 160 checkpoints, 160 prediction files | `runs/c0_gnn/`, `runs/c0_mlp/`; `checkpoints/c0_gnn/`, `checkpoints/c0_mlp/`; `predictions/c0_gnn/`, `predictions/c0_mlp/` |
| Run logs of the two retrained arms and of the node-wise adaptation | `runs/c1_noreg/`, `runs/c2_nomono/`, `runs/ba1/` (their checkpoints and prediction files are also included) |
| Generator specification | `manuscript/supplementary/appendix_target_generator.tex`, `manuscript/pdf/Supplementary_Material.pdf` |

## What is where

Arms: `c0_gnn` = GNN-RF (run label `c0c1cf_*`); `c0_mlp` = capacity-matched graph-free MLP (`mlpcf_*`);
`c1_noreg` = GNN-RF retrained without the four physics regularizers (`c1noreg_*`); `c2_nomono` = GNN-RF
retrained without the distance-monotonicity term (`c2nomono_*`); `ba1` = RadioUNet-style node-wise
encoder-decoder (in-house adaptation, `ba1rus_*`). Each arm has 4 cities x 4 quadrants x 5 seeds (42-46) = 80 runs.

| Folder | Content | Count |
|---|---|---|
| `runs/<arm>/<run_label>/` | `run_<run_label>.json` (configuration, environment, partition hashes, per-epoch and test metrics) and `training_log.csv`; for `c1_noreg`, `c2_nomono` and `ba1` also `stdout_wrapper.log` and `sidecar_<run_label>.json` (library versions, wrapper/guard/trainer hashes) | 80 run logs per arm (C0 GNN-RF 80, MLP 80, C1 80, C2 80, BA1 80) |
| `runs/c0_controls/` | run logs of the from-scratch (`c0c1cfsc_*`) and reversed-chain (`c0c1cfinv_*`) control runs | 28 |
| `checkpoints/<arm>/<run_label>.pt` | best-validation checkpoint of every run | 80 per arm, 400 in total |
| `predictions/<arm>/<arm>_<city>_Q<n>_s<seed>.npz` | test-partition predictions: `idx_global`, `target` (N x 5), `pred` (N x 5), `sentinela` | 80 per arm, 400 in total |
| `predictions/PREDICTIONS_INDEX.csv` | prediction file -> run label, checkpoint file and SHA-256, run-log SHA-256 | 400 rows |
| `partitions/` | `partition_<city>_Q<n>.npz` (`numpy.packbits` masks), `PARTITION_HASHES.json`, `verify_partitions.py` | 16 cells x 3 splits |
| `reference_fields/` | SHA-256 of the 16 reference fields and the script that computed them | 16 fields |
| `code/` | same layout as the original project folder, so the trainers run with `--base-dir code` | |
| `results/blocked/` | blocked-evaluation results: paired test, nearest-valid-neighbour contrast, admissibility, blocked transfer, MC Dropout uncertainty (per-cell JSON and raw samples), graph census, spatial error export, aggregates | |
| `results/retrained_arms/`, `results/ba1/`, `results/nested_tost/` | metrics of the retrained arms, of the node-wise adaptation, and the nested equivalence test | |
| `manuscript/` | LaTeX source, bibliography, tables/figures read by the source, supplementary material, PDFs | |

## Provenance notes

- The generator, data preparation, enrichment, corrector, the two frozen trainers, the model modules
  (`code/02_models/`), `code/03_training/spatial_cv.py`, `code/03_training/rf_diagnostic_metrics.py`, the
  retrained-arm wrapper and guard, and the BA1 trainer, wrapper, guard and configuration are byte-identical to the
  files that produced the runs; their SHA-256 matches the digests recorded in the run logs, wrapper records and
  Supplementary Material. Their comments are therefore left as they were written.
- Every run log records `dataset.rf_data_sha256`; all 400 match one of the 16 digests in `reference_fields/`.
- Every run log records `particoes.{train,val,test}.idx_sha256_global`; `partitions/verify_partitions.py`
  checks all 400 run logs and all 400 prediction files against the masks.
- Prediction files of `c0_gnn` and `c0_mlp` were written by the preceding version of
  `code/evaluation/inference/e3_inferencia_faseC.py` (SHA-256 c8e4356f…); the version here adds the arm prefix,
  default paths and the default inference seed. Inference used fp16 autocast.
- Library versions are recorded per run (`ambiente` in the run logs; `versoes` in the wrapper records, including
  `torch_geometric` 2.8.0.post1 for the retrained arms and BA1). The run logs of `c0_gnn`/`c0_mlp` do not record
  the PyTorch Geometric version.
- Absolute paths inside scripts point to the authors' machines and must be adapted.

## Known limitations of this package

- The 16 reference fields (about 452 GB in total) are given only by file name, size and SHA-256
  (`reference_fields/`). They can be regenerated with the generator, data-preparation, enrichment and corrector
  scripts in `code/` from the public inputs described in the Supplementary Material, and checked against those digests.
- The antenna records are public ANATEL licensing data (base stations of three mobile operators). The extract used,
  `antenas_interior_sp_final.csv` (17,763,282 bytes, SHA-256 ad9addb0c515a0b4…), is not redistributed here. Its
  download date is not recorded in the run logs or in the Supplementary Material.
- Run logs, wrapper records and scripts contain absolute paths of the authors' machines. Two evaluation scripts do not
  run outside those machines without adaptation: `code/evaluation/nested_tost/D_tost_aninhado.py` (hashes an
  input file that is not part of this package) and `code/evaluation/ba1/BA_fisica.py` (its final cross-check block
  reads files that are not part of this package; the metrics are computed before that block).
- The LaTeX source in `manuscript/` does not compile without the journal class file and the author photographs;
  the compiled PDFs are included in `manuscript/pdf/`.

## Reproduction outline

1. Reference fields: `generate_realistic_coverage.py` -> `01_data/prepare_transfer_dataset_v19.py` ->
   `data_raw/enrich_rf_targets.py` -> `corrector/contrafactual_alvo_completo.py` (requires the antenna licensing
   extract and the terrain graphs, not included; see the Supplementary Material).
2. GNN-RF and MLP runs: `code/03_training/train_gnn_c0_spatial.py --base-dir code --run-label c0c1cf_bauru_s42_Q1_g10b2 --rf-data-file <field> --graph-file <graph> --epochs 35 --lr 1e-3 --seed 42 --split-seed 42 --grid-km 10 --buffer-km 2 --split-frac 0.70,0.15,0.15 --freq-mhz 900 --diag-freq-mhz 1800 --hash`;
   Q2-Q4 use `--transfer-from` the previous quadrant's checkpoint, 20 epochs, `--lr 2e-4`. The exact arguments of
   every run are in its run log (`config`). Same for `train_mlp_c0_spatial.py`.
3. Retrained arms: `code/retrained_arms/wrapper_ablacao.py --braco c1noreg|c2nomono --cidade <city> --seed <s> --q <n>`.
   BA1: `code/ba1/wrapper_ba1.py --cidade <city> --seed <s> --q <n>` (configuration in `code/ba1/config_ba1.json`).
4. Predictions: `code/evaluation/inference/e3_inferencia_faseC.py --celula <city>_Q<n> --braco gnn --seed <s> --prefixo-run c0c1cf|c1noreg|c2nomono`
   (batch: `lote_inferencia.sh`); BA1: `e3_inferencia_ba1.py` (batch: `lote_inferencia_ba1.sh`).
5. Metrics: `code/evaluation/retrained_arms/C_metricas_v2.py`, `C_inferencia_v2.py`, `C_custodia_v2.py`;
   `code/evaluation/ba1/BA_metricas.py`, `BA_inferencia.py`, `BA_custodia.py`, `BA_fisica.py`;
   `code/evaluation/nested_tost/D_tost_aninhado.py`; all import `code/evaluation/common/common.py`.
6. Blocked-evaluation scripts behind the remaining tables: `code/evaluation/blocked/` and `code/tables_figures/`.

## Citation

See `CITATION.cff`.

## Large artifacts (Zenodo only)

This repository mirrors the code, run logs, results, partitions, manuscript and the 16 seed-42 GNN checkpoints. The full set of
400 checkpoints and 400 prediction files (five arms x 80 runs, 10.5 GB) is published only in the Zenodo record
https://doi.org/10.5281/zenodo.23377937, together with MANIFEST_SHA256.txt covering every file.
