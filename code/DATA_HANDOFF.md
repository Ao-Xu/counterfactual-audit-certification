# Plotting data handoff (historical full-bundle inventory)

This file documents the historical evidence-inclusive bundle. The public
source-only `code.zip` does **not** contain the listed result tables, raw
predictions, figure inputs, models, or licensed datasets. It is retained to
identify the required inputs and their original status; it is not a claim that
the public archive alone reproduces every manuscript figure.

All paths are relative to the `code/` directory in the single `code.zip`.
The figures are prepared by a separate plotting task. This handoff distinguishes
fresh controlled reproduction, recovered historical predictions/aggregates,
analytic witnesses, and original graphic-only inputs.

| Plot / evidence | Actual input | Verified availability |
|---|---|---|
| Planner budget/power/cost | `oral_planner_v6/results/summary.json`, `summary.csv`, `final_plans.csv`, `benchmark_grid.csv` | 6 groups, 288 final rows, 1344 benchmark cells; 109/144 feasible; 27904/27904 successes for feasible valid plans |
| Matched 8 cells | `oral_attribution_v5/results/matched_thresholds.csv`, `dense_thresholds.csv`, `dense_power.csv`, `factorial.csv` | 64 matched threshold/CI rows; 208 all-method rows; 13000 power rows; full 1600-rep frozen confirmation |
| HANS / ANLI | `overleaf/experiments/hans_suite/outputs/results.csv`, `manifest_5seed_anli.json` | 150 rows = 5 seeds × 5 conditions × 6 splits; HANS train/eval 30000 each, ANLI 3200; aggregate metrics only |
| Corrected F25/F26 intervals | The same saved HANS CSV and `figure_inputs/metadata/hans_seed_ci_validation.json` | 39 two-sided 95% t intervals over 5 seeds (df=4); F25(b,c) average the 3 heuristics within seed first; F26(b) uses paired within-seed gaps; means unchanged |
| Qwen paired audit | `overleaf/experiments/comparative_utility/qwen_paired_results/qwen_paired_summary.csv` | 32 statistics recomputed from saved crossed-consistency probabilities and labels |
| Qwen paired predictions | `overleaf/experiments/llm_cfpt/crossed_consistency_v4/predictions/{calibration,test}/`, `splits/` | W0/Wlow/Wstar predictions for all 8 pools; other archived arms/metadata retained |
| Cluster variance | `overleaf/experiments/llm_cfpt/results/strict_cluster_scaling.csv` | Exact 9 cells and fitted two-level/naive summaries; no new model run |
| Shift granularity | `overleaf/experiments/shift_granularity/p0_outputs/{p0_all_records.csv,p0_aggregate.csv,p0_manifest.json}` | 2160 records and 180 aggregates independently recomputed from `llm_cfpt/verified_results/` |
| Corruption boundary | `overleaf/experiments/shift_granularity/boundary_outputs/qwen_corruption_boundary.csv` | 48 rows recomputed from 48 saved Qwen evaluation files; support escape is a separate 55-row analytic witness |
| Historical four-panel Real-LLM inputs | `overleaf/experiments/llm_cfpt/confirmatory_v1/lora_metrics.json`, `decision_results/decision_summary.csv`, `crossed_consistency_v4/results.json`, `transfer_tightness/metrics.json` | All four source SHA hashes match historical figure manifest; 8 pools, 20 decision rows, 8 intervention pools, 4 slack heads |
| Current F34 (three panels) | The same lora metrics, crossed-consistency results and transfer-tightness metrics; final `tools/figure_sources/figure_repairs.py` | Matched corruption, prospective negative gates, transfer slack only; old fixed-feature prediction-MAE code/data retained as archive, not current manuscript evidence |
| Saved certificate audit | `overleaf/experiments/llm_cfpt/certificate_results/certificate_raw.json`, `certificate_summary.csv` | 196800 saved audit rows regrouped to exactly 15 historical summary rows; no new bootstrap or training |
| Intro | `figure_inputs/intro.pdf` | Source PDF hash verified; original raster interiors copied, no raw curve arrays recovered |
| Unrecoverable historical raw plots | `figure_inputs/originals/` plus `figure_inputs/metadata/` | Source PDFs for labelled vector-only reflow, not numerical/raw-data reconstruction |

Important plotting boundaries:

- Planner pilot size is fixed at 2048. Do not depict a pilot-size sweep.
  Feasible-only power and end-to-end power have different denominators.
- Matched CI correction covers 13 methods × 125 budgets within each geometry,
  not a single simultaneous claim across all eight geometries. Empty/censored
  thresholds are not zero. All 192 matched threshold/CI scalars matched the
  manuscript at the recorded audit check.
- `crossed_consistency_v4/results.json` primary records contain a point estimate,
  a one-sided `upper95`, and a separate `percentile95` summary. There is no
  `lower95` field to invent. The packaged plot repair scales the point once
  and displays point-to-upper; it does not change stored estimates/intervals.
- HANS suite predictions and the older oral/identification HANS predictions are
  distinct protocols. The recovered aggregate suite CSV cannot replace a
  missing `predictions.npz` or `app/statistics_cf.py`.
- The authorized r3 HANS correction changes interval widths, not the saved CSV
  or displayed means. `tools/verify_hans_intervals.py` independently verifies the
  exact seed-level inputs and exports all 39 corrected intervals. Old normal-
  interval/pooled-record plots are historical, not the final paper workflow.
- Generator sources are archived unchanged, with hash/AST/import provenance.
  Recovering them does not authenticate absent model weights or prove that
  historical training can be repeated bit-for-bit on another device.
- The learned `fixed_heads.npz` coefficients are retained locally but excluded
  from the ZIP with model weights. The saved decision/certificate/transfer
  outputs and the tested redraw paths remain included and runnable; regenerating
  those raw fixed-head analyses is a separate resource-dependent workflow.

Use `tools/reproduce_figures.py` as documented in the root README. Parent plot
scripts are copied byte-for-byte under `tools/figure_sources/`, not edited in
their original location. Every wrapper run writes an input/output receipt in
its own new output directory. Graphic metadata remains a time-stamped snapshot;
render success is not a replacement for the plotting task's visual acceptance.
