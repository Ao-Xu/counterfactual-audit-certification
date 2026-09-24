"""Export saved frozen v6 JSON to CSV without rerunning a simulation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE/'planning'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args = parser.parse_args()
    result = ROOT/'results'
    cfg = read(ROOT/'manifest.json')
    for name,expected in cfg['source_hashes'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError('Frozen source changed: '+name)
    lock = read(result/'budget_lock.json')
    final = read(result/'final.json')
    budget_sha = sha(result/'budget_lock.json')
    if budget_sha!=(result/'budget_lock.sha256').read_text().strip():
        raise ValueError('Budget checksum mismatch')
    if final['budget_lock_sha']!=budget_sha or lock['manifest_sha']!=sha(ROOT/'manifest.json'):
        raise ValueError('Frozen manifest/lock/final binding mismatch')
    bench = read(result/'benchmarks.json')['rows']
    summary = read(result/'summary.json')
    flat = []
    for row in summary:
        r = {k:v for k,v in row.items() if not isinstance(v,dict)}
        r['empirical_threshold_CI'] = json.dumps(r['empirical_threshold_CI'])
        for mode in ('valid','plugin','local_lower'):
            r.update({mode+'_'+k:v for k,v in row[mode].items()})
        flat.append(r)
    out = args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    save_csv(out/'summary.csv',flat)
    save_csv(out/'final_plans.csv',final['rows'])
    save_csv(out/'benchmark_grid.csv',bench)
    info = dict(input_sha256={name:sha(result/name) for name in ('budget_lock.json','final.json','benchmarks.json','summary.json')},
                output_sha256={p.name:sha(p) for p in sorted(out.glob('*.csv'))},
                rows=dict(summary=len(flat),final_plans=len(final['rows']),benchmark_grid=len(bench)),
                scope='Lossless CSV reshaping of saved original report/final/benchmark JSON; no new simulation.')
    (out/'export_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info['rows']))


if __name__=='__main__':
    main()
