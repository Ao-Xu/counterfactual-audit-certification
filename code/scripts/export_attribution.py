"""Export the existing frozen v5 threshold/CI definitions from real counts."""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from pairing_analysis.analyze import band, ratio, threshold
from pairing_analysis.simulate import P, METHODS

R = CODE / "pairing_analysis"
OUT = R / "results"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def csv_save(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for exports; confirm.npz is read from the original results directory.")
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    z = np.load(R / "results/confirm.npz", allow_pickle=False)
    ns, reps = z["N"], int(z["reps"])
    assert reps == P["confirmation_reps"] == 1600
    np.testing.assert_array_equal(ns, P["N_grid"])
    rows, curves, lookup = [], [], {}
    for g in range(8):
        for method in METHODS:
            counts = z[method + "_count"]
            assert counts.shape == (len(ns), 8, 3)
            assert np.all((counts >= 0) & (counts <= reps))
            k = counts[:, g, 1]
            lo, hi = band(k, reps, .05 / (len(METHODS) * len(ns)))
            for i, n in enumerate(ns):
                curves.append(dict(geometry=g, **P["geometries"][g], method=method,
                                   N=int(n), candidate=1, successes=int(k[i]), trials=reps,
                                   power=float(k[i]/reps), power_lower=float(lo[i]), power_upper=float(hi[i])))
            for target in (.8, .9):
                item = threshold(ns, k, reps, target, .05 / len(METHODS))
                item.update(geometry=g, parameters=P["geometries"][g], method=method,
                            evaluations=2*item["estimate"] if item["estimate"] else None)
                rows.append(item)
                lookup[g, method, target] = item
    factors, flat_factors = [], []
    for g in range(8):
        for target in (.8, .9):
            ss, sh, ps, ph = [lookup[g, method, target] for method in METHODS[:4]]
            ratios = {name: ratio(x, y) for name, x, y in
                      [("pairing_split", ss, ps), ("pairing_shared", sh, ph),
                       ("reuse_separate", ss, sh), ("reuse_paired", ps, ph)]}
            ns0, nh, np0, nph = [v["estimate"] for v in (ss, sh, ps, ph)]
            item = dict(geometry=g, target=target, ratios=ratios,
                        log_interaction=math.log(ns0*nph/(np0*nh)),
                        log_interaction_ci=[math.log(ss['lower']*ph['lower']/(ps['upper']*sh['upper'])),
                                            math.log(ss['upper']*ph['upper']/(ps['lower']*sh['lower']))],
                        shapley_pairing_factor=math.exp(.5*math.log(ns0*nh/(np0*nph))),
                        shapley_reuse_factor=math.exp(.5*math.log(ns0*np0/(nh*nph))), combined_ratio=ns0/nph)
            factors.append(item)
            flat = {k: v for k, v in item.items() if k not in {'ratios', 'log_interaction_ci'}}
            flat.update(log_interaction_lower=item['log_interaction_ci'][0], log_interaction_upper=item['log_interaction_ci'][1])
            for name, values in ratios.items():
                flat.update({name+'_'+k: v for k, v in values.items()})
            flat_factors.append(flat)
    flat_rows = [{k: v for k, v in row.items() if k != "parameters"} | row["parameters"] for row in rows]
    csv_save("dense_thresholds.csv", flat_rows)
    csv_save("matched_thresholds.csv", [row for row in flat_rows if row['method'] in METHODS[:4]])
    csv_save("dense_power.csv", curves)
    csv_save("factorial.csv", flat_factors)
    info = dict(thresholds=rows, factorial=factors,
                max_family_null_acceptance=max(int(z[m+'_false'].max()) for m in METHODS)/reps,
                band_scope='95% simultaneous over 13 methods and all 125 N within each geometry; no monotonicity assumption; not simultaneous across all 8 geometries.',
                matched_methods=METHODS[:4], primary_candidate=1, replications=reps,
                scope='Matched dense confirmation and all 13 method thresholds only; covariance stress and historical ReCoRD results are not included.',
                no_confirmation_retuning=True)
    save("dense_thresholds.json", info)
    paths = [R/'results/confirm.npz'] + [OUT/name for name in ('dense_thresholds.json', 'dense_thresholds.csv', 'matched_thresholds.csv', 'dense_power.csv', 'factorial.csv')]
    save("MATCHED_RESULTS_MANIFEST.json", {
        "manifest_sha256": hashlib.sha256((R/'frozen_manifest.json').read_bytes()).hexdigest(),
        "threshold_lock_sha256": hashlib.sha256((R/'threshold_lock.json').read_bytes()).hexdigest(),
        "seed": P['confirmation_seed'], "replications": reps, "N_grid": ns.tolist(),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "status": "Fresh frozen dense confirmation, not historical raw file recovery", "band_scope": info['band_scope']})
    print(json.dumps({"replications": reps, "budgets": len(ns), "all_threshold_rows": len(rows),
                      "matched_threshold_rows": 64, "power_rows": len(curves), "null_max": info['max_family_null_acceptance']}))
    for g in range(8):
        print(g, [(m, lookup[g, m, .9]['estimate'], lookup[g, m, .9]['lower'], lookup[g, m, .9]['upper']) for m in METHODS[:4]])


if __name__ == "__main__":
    main()
