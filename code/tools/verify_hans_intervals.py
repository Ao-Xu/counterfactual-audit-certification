"""Independently verify the authorized r3 HANS seed-level plotting intervals.

Reads the saved 150-row results CSV and the figure task's actual interval
receipt. Does not train, infer, bootstrap, or change any saved experiment data.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

CODE = Path(__file__).resolve().parents[1]
CONDITIONS = ['factual', 'natural_cf', 'hans_balanced', 'aligned_corrupt', 'reversed_corrupt']
GROUPS = ['lexical_overlap', 'subsequence', 'constituent']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-14, equal_nan=False):
        raise ValueError('Interval verification failed: ' + label)


def verify(results_csv, validation_path):
    df = pd.read_csv(results_csv)
    receipt = json.loads(validation_path.read_text(encoding='utf-8'))
    if receipt['repair_version'] != '2026-09-24-r3-seed-ci':
        raise ValueError('Expected approved r3 seed-ci interval receipt')
    checksum = sha(results_csv)
    if receipt['source_sha256_before'] != checksum or receipt['source_sha256_after'] != checksum:
        raise ValueError('HANS input differs from the frozen figure input')
    if receipt['raw_data_changed'] or receipt['retraining_performed'] or not receipt['displayed_means_preserved']:
        raise ValueError('Unexpected statistical correction scope')
    seeds = sorted(df.seed.unique().tolist())
    if len(seeds) != 5 or len(df) != 150 or df.duplicated(['condition', 'seed', 'split']).any():
        raise ValueError('Incomplete/duplicate five-seed HANS results')
    expected = {}

    def add(key, values, previous_values):
        values = np.asarray(values, dtype=float)
        previous_values = np.asarray(previous_values, dtype=float)
        if values.size != 5 or not np.isfinite(values).all():
            raise ValueError('Expected five finite seed-level observations')
        expected[key] = (values, previous_values)

    for condition in CONDITIONS:
        sub = df[(df.condition == condition) & df.split.isin(['hans_' + g for g in GROUPS])]
        if len(sub) != 15 or not (sub.groupby('seed').split.nunique() == 3).all():
            raise ValueError('Each seed must have exactly three HANS heuristics')
        for group in GROUPS:
            values = sub[sub.split == 'hans_' + group].set_index('seed').reindex(seeds).risk.to_numpy()
            add(('F25', 'a', condition, 'risk', group), values, values)
        for panel, metric in [('b', 'risk'), ('c', 'nll')]:
            values = sub.groupby('seed')[metric].mean().reindex(seeds).to_numpy()
            add(('F25', panel, condition, metric, None), values, sub[metric].to_numpy())
        for split in ['hans_all', 'anli_dev_r123']:
            values = df[(df.condition == condition) & (df.split == split)].set_index('seed').reindex(seeds).brier.to_numpy()
            add(('F26', 'a', condition, 'brier', split), values, values)
    pivot = df[df.split == 'anli_dev_r123'].pivot(index='seed', columns='condition', values='brier').reindex(seeds)
    for condition in CONDITIONS[1:]:
        values = (pivot[condition] - pivot.factual).to_numpy()
        add(('F26', 'b', condition, 'paired_brier_gap', None), values, values)
    rows = receipt['intervals']
    if len(rows) != 39 or receipt['n_intervals'] != 39 or len(expected) != 39:
        raise ValueError('Expected all 39 F25/F26 intervals')
    seen = set()
    table = []
    exact_t = float(t.ppf(.975, df=4))
    for row in rows:
        key = tuple(row[k] for k in ['figure', 'panel', 'condition', 'metric', 'group'])
        if key in seen or key not in expected:
            raise ValueError('Unexpected/duplicate interval: ' + repr(key))
        seen.add(key)
        values, previous_values = expected[key]
        if row['seeds'] != seeds or row['n_seeds'] != 5 or row['df'] != 4:
            raise ValueError('Interval uses the wrong sampling unit')
        close(row['seed_values'], values, str(key) + ': per-seed values')
        critical = row['t_critical']
        # Approved source rounds the df=4 two-sided 95% multiplier to 6 decimals.
        if abs(critical - exact_t) > 5e-7:
            raise ValueError('Multiplier is not the df=4 95% t critical value')
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half = critical * sd / np.sqrt(5)
        old_mean = float(previous_values.mean())
        old_half = 1.96 * float(previous_values.std(ddof=1)) / np.sqrt(len(previous_values))
        for name, value in [('mean', old_mean), ('seed_mean', mean), ('sample_sd_ddof1', sd),
                            ('half_width', half), ('lower95', old_mean - half), ('upper95', old_mean + half),
                            ('previous_half_width', old_half)]:
            close(row[name], value, str(key) + ': ' + name)
        close(mean, old_mean, str(key) + ': unchanged mean')
        close(row['mean_difference'], mean - row['mean'], str(key) + ': mean difference')
        if row['previous_n_records'] != len(previous_values):
            raise ValueError('Historical aggregation count is incorrect')
        table.append({k: row[k] for k in ['figure', 'panel', 'condition', 'metric', 'group',
            'n_seeds', 'df', 't_critical', 'mean', 'half_width', 'lower95', 'upper95',
            'previous_n_records', 'previous_half_width']})
    if seen != set(expected):
        raise ValueError('Missing interval cells')
    result = dict(status='PASS', repair_version=receipt['repair_version'], intervals_verified=39,
        source_csv_sha256=checksum, interval_receipt_sha256=sha(validation_path),
        n_independent_seeds=5, degrees_of_freedom=4, scipy_exact_t_critical=exact_t,
        approved_rounded_t_critical=rows[0]['t_critical'], displayed_means_unchanged=True,
        F25_bc_within_seed_aggregation_verified=True, F26_b_within_seed_pairing_verified=True,
        scope='Independent postprocessing check of saved results and actual plotted interval receipt; no retraining or new experiment.')
    return result, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-csv', type=Path,
        default=CODE / 'overleaf/experiments/hans_suite/outputs/results.csv')
    parser.add_argument('--validation', type=Path,
        default=CODE / 'figure_inputs/metadata/hans_seed_ci_validation.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result, table = verify(args.results_csv, args.validation)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (args.output_dir / 'hans_seed_intervals.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
