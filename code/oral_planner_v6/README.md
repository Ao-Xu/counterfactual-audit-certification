# Reproduce the frozen pilot-plan-certify experiment

This procedure uses the supplied `manifest.json`, all original seeds, sampling
laws, budgets, and tie-breakers. All 144 plans are locked before final outcomes.
The original ZIP contains no old raw outcomes; these are newly reproduced data.

## Commands

Start from a clean copy of the source package without existing results. These
commands use the already installed environment used for the verified full run.

```powershell
Set-Location ./code
$pythonExe = 'python'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
$env:MPLCONFIGDIR = '../build/revision/code_qa/mplconfig'
& $pythonExe -X utf8 -m unittest discover -s oral_planner_v6/tests -v
& $pythonExe -X utf8 -u -m oral_planner_v6.run pilot
& $pythonExe -X utf8 -u -m oral_planner_v6.run final
& $pythonExe -X utf8 -u -m oral_planner_v6.run benchmark
& $pythonExe -X utf8 -u -m oral_planner_v6.report
```

Stop on a nonzero exit code before invoking the next command. Do not use `-O`:
integrity checks in the frozen sources rely on assertions. The runner creates
the output directory and refuses to overwrite existing locks/outcomes. Report
generation is deterministic postprocessing and can be repeated. To rerun the
simulation, use another source copy rather than deleting this run's evidence.

The actual audit used the same module commands through one supervisor:

```powershell
& $pythonExe -X utf8 -u ../build/revision/code_qa/run_stages.py
```

This audit-only supervisor records commands, start/end times, child PIDs and
exit codes in `../build/revision/code_qa/execution.json`. Its exclusive
`execution.started` receipt remains after completion, so it cannot accidentally
restart this run. An observation timeout does not mean failure: check the
recorded PID and stage log before taking action. Do not launch duplicate stages.

## Frozen settings

| Setting | Value |
|---|---|
| Laws / targets / pilots | A,B,C / 0.8,0.9 / 24 per pair, 144 total |
| Pilot | 2048 anchors: 256 design + 1792 estimation, Kmax=8 |
| Menu | K=2,8; eta=0.1,0.25,0.5; mean fractions=0.35,0.65 |
| Audit sizes | All 32 manifest entries, 256 through 258494 |
| Safety / planning / radius | delta=0.025, delta_plan=0.05, rho=0.05 |
| Replications | final=256 per feasible plan; benchmark=1024 per configuration |
| Seeds | pilot=23092711, final=23092712, benchmark=23092713 |

Manifest SHA-256:
`9849ba7e832a3abebbb28569d4c86396b284cb706a9378869eeafae72f7473f2`.
The protected `planner.py`, `laws.py`, and `THEORY.md` hashes are inside this
manifest and checked before execution. `freeze.py` is provenance, not a
reproduction step: creating a new manifest would omit the retained amendment.

## Outputs

All paths in this table are relative to `oral_planner_v6/`.

| File | Content |
|---|---|
| `results/budget_lock.json` and `.sha256` | All 144 decisions and diagnostics; saved before finals |
| `results/final.json` | 288 valid/plugin rows, including infeasible decisions; N/K, successes/trials and endpoint min/mean/max |
| `results/benchmarks.json` | 1344 rows: 3 laws x 2 K x 32 N x 7 endpoint/ledger settings, 1024 trials each |
| `results/summary.json` | Six summaries, simultaneous threshold intervals, feasibility and costs |
| `results/REPORT.md` | Data-derived text, with historical real-model claims outside this rerun |
| `results/summary.csv`, `final_plans.csv`, `benchmark_grid.csv` | Audit-generated tabular exports for subsequent figures |
| `results/RESULTS_MANIFEST.json` | Result, source and configuration hashes; observed environment |
| `results/figures/` | Reproduced PDFs/table; existing manuscript figures are not overwritten |
| `results/progress.json` | Progress only, not raw experimental evidence |

The original exact categorical sampler saves aggregate sufficient-statistic
outputs, not individual anchor draws or all per-replication endpoints. These
counts and endpoint summaries support the reported power/cost tables.
Distribution/quantile plots of individual endpoints require separately
documented additional storage; they cannot be inferred from these aggregates.

## Verified result

| Law / target | Feasible / 24 | Median fresh N | Empirical menu N |
|---|---:|---:|---:|
| A / 0.8 | 24 | 7276 | 1907 |
| A / 0.9 | 24 | 7276 | 1907 |
| B / 0.8 | 11 | 84703 | 2980 |
| B / 0.9 | 4 | 64035.5 | 2980 |
| C / 0.8 | 23 | 22204 | 2384 |
| C / 0.9 | 23 | 34694 | 2980 |

B/0.9's median rounds to the manuscript's 64036; it is halfway between two
selected budgets and is not itself a selected grid point. All 109 feasible valid
plans succeed in all 256 replications. The 35 infeasible plans stay in the
denominator. All valid plans use BE: 43 paired, 66 generic, none separate.
The minimum simultaneous binomial lower bound is about 0.9641. For 29/144
plug-in plans, simultaneous upper power limits lie below target.

The verified environment was Python 3.10.19 / NumPy 1.26.4 / SciPy 1.15.3 /
Matplotlib 3.10.8. Original seven tests also pass in the existing Python 3.9 /
NumPy 2.0 environment, but the full simulation was run only in Python 3.10.
Observed runtime: pilot 64.3 s, final 3.1 s, benchmark 7.3 s, report 5.4 s.
One pilot process snapshot used about 83 MB working memory; peak memory was not
measured. No GPU or model weights are needed for this controlled experiment.

The final single `code.zip` already includes these verified results. To reshape
saved JSON without repeating simulations, run from the package root:
`python tools/export_planner.py --output-dir ../planner_export`.
The portable exporter has no dependency on the audit directory or manuscript.
Use a separate source-only working copy if intentionally rerunning the full
simulation; preserve delivered evidence and budget locks.
