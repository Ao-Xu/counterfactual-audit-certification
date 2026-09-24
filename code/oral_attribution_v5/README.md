# Frozen matched attribution: dense results

The 2026-09-24 reproduction completed the original `confirm` phase with seed
23092642, 1600 replications, all 125 budgets from 512 to 16384 at spacing 128,
all 8 geometries, all 13 methods, and all 3 candidates. No method/grid/seed or
loss-law parameter changed. The archived threshold lock was preserved; no
planning phase was rerun. All 64 matched threshold/CI entries agree with the
manuscript (192 scalar comparisons including both endpoints).

## Data for figures

Paths below are relative to this directory.

| File | Content |
|---|---|
| `results/confirm.npz` | Original simulation output: N, reps, all 13 method counts/false counts/means/SDs for all 3 candidates, plus component summaries |
| `results/matched_thresholds.csv` | 8 geometries x 4 matched methods x 2 targets = 64 rows |
| `results/dense_thresholds.csv` / `.json` | All 13 methods: 208 thresholds; JSON also includes factorial factors |
| `results/dense_power.csv` | 13000 rows for primary candidate index 1: successes, trials, point power, simultaneous lower/upper power bands |
| `results/factorial.csv` | Pairing/reuse ratios and intervals, symmetric factors and log interaction for both targets |
| `results/oracle_forecast.json` | 64 population-only Gaussian forecasts from the unchanged original forecast script |
| `results/sufficient_budgets.json` | 32 rigorous sufficient shared budgets from that script |
| `results/MATCHED_RESULTS_MANIFEST.json` | Hashes, seed, grid, replication count and CI scope |

`geometry` is 0 through 7, ordered by correlation r=-0.5/0.98, anchor
scale a=0.04/0.12, then nuisance regime (b,h)=(0.04,0)/(0.12,0.8).
The four matched methods are `separate_split`, `separate_shared`,
`paired_split`, `paired_shared`. The CSVs include the corresponding r/a/b/h.
`estimate`, `lower`, and `upper` refer to the first grid crossing. They are
anchor counts; paired-loss evaluation counts are twice these at K=2. Empty
threshold endpoints in all-method CSVs mean right censoring, not zero cost.

The 95% binomial bands are simultaneous across **13 methods and 125 budgets
within each geometry**. The per-cell error is 0.05/(13*125). These are not joint
95% bands across all eight geometries. The threshold inversion does not assume
monotone power. The matched subset retains the full 13-method correction.

## Reproduction commands

Run from the package root with the Python environment described in the root
README. In a separate source copy, ensure confirmation outputs do not exist:

```powershell
if (Test-Path oral_attribution_v5/results/confirm.npz) { throw 'Preserve existing confirmation; use another source copy.' }
New-Item -ItemType Directory -Force oral_attribution_v5/results | Out-Null
python -X utf8 -u -m oral_attribution_v5.simulate confirm
python -X utf8 -u -m oral_attribution_v5.forecast
```

The actual supervised audit ran
`python -X utf8 -u ../build/revision/code_qa/run_attribution.py`. Its exclusive
receipt, PIDs and logs are in `../build/revision/code_qa/attribution_execution.json`.
The raw confirmation took about 80 seconds on the audit host.
`../build/revision/code_qa/export_attribution.py` produced the CSVs using the
unchanged `analyze.threshold`, `analyze.band`, and `analyze.ratio` functions and
the original factorial formulas. It exports matched results independently of
the unrelated covariance-stress/ReCoRD stages; those outcomes were not invented.
The portable equivalent is now included as `tools/export_attribution.py`:
`python tools/export_attribution.py --output-dir ../matched_export` reads the
packaged confirmation and creates a new export directory. Its exported CSVs
are checked byte-for-byte against the audit export. No QA-directory dependency
is needed. The final single `code.zip` includes raw confirmation and exports.

The complete original `analyze.py` also requires `results/stress_confirm.npz`;
the complete `report.py` additionally requires historical ReCoRD data. Those
stages were not run here. Do not label the matched-only export `analysis.json`
or present it as full historical v5 reproduction.

The maximum relative error of the 64 oracle forecasts against reproduced
thresholds is 0.0303030303, i.e. about 3.03%; the manuscript also gives the
conservative 3.04% bound. The observed boundary-null error frequency is zero;
zero observed events do not establish zero population risk.
