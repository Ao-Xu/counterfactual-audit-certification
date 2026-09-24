"""Validate recovered bytes and recompute figure statistics without training.

Run from the package root: python tools/verify_historical.py --output-dir PATH
An optional --render tests existing plot entry points into that NEW directory.
Historical prediction and summary files are never rewritten.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
EXPS = CODE/'overleaf/experiments'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXPS/rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(actual, expected, path='root'):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            close(actual[key], expected[key], path+'/'+str(key))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a,b) in enumerate(zip(actual,expected)):
            close(a,b,path+'/'+str(i))
    elif isinstance(expected, (int,float)) and not isinstance(expected,bool):
        if math.isnan(float(expected)) and math.isnan(float(actual)):
            return  # Undefined reliability is retained when no decisions occur.
        assert math.isclose(float(actual),float(expected),rel_tol=1e-12,abs_tol=1e-12), (path,actual,expected)
    else:
        assert actual == expected, (path,actual,expected)


def compare_csv(path, rows):
    with path.open(encoding='utf-8-sig',newline='') as f:
        historical = list(csv.DictReader(f))
    assert len(rows) == len(historical), str(path)
    for i,(a,b) in enumerate(zip(rows,historical)):
        assert set(a).issubset(b), (path,i)
        for key,value in a.items():
            if isinstance(value,(int,float)):
                close(float(b[key]),value,str(path)+'/'+str(i)+'/'+key)
            else:
                assert str(value)==b[key], (path,i,key)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--render',action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    assets = read(CODE/'RECOVERED_ASSETS.json')
    sources = read(CODE/'RECOVERED_SOURCES.json')
    for row in assets['files']+sources['files']:
        assert sha(CODE/row['path']) == row['sha256'], row['path']
    checks = dict(data_files_hash_checked=len(assets['files']), source_files_hash_checked=len(sources['files']),
                  tolerance=dict(rtol=1e-12,atol=1e-12), no_training_or_inference=True)
    qwen = load('qa_qwen','comparative_utility/qwen_paired_audit.py')
    qr = [qwen.one(split,pool,condition,qwen.labels(split))
          for split in ('calibration','test') for pool in qwen.POOLS for condition in qwen.CONDITIONS]
    close(qr,read(qwen.OUT/'qwen_paired_raw.json'))
    compare_csv(qwen.OUT/'qwen_paired_summary.csv',qr)
    checks['qwen_paired_rows_recomputed'] = len(qr)
    p0 = load('qa_p0','shift_granularity/p0_existing_predictions.py')
    pr = []
    for size in p0.SIZES:
        for task in p0.TASKS:
            for seed in p0.SEEDS:
                prefix = f'{size}_{task}_{seed}_'
                factual = p0.test_rows(p0.read(p0.ROOT/(prefix+'factual.json')))
                for condition in p0.CONDITIONS:
                    candidate = p0.test_rows(p0.read(p0.ROOT/(prefix+condition+'.json')))
                    pr.extend(p0.records_for(p0.loss_difference(candidate,factual),size,task,seed,condition))
    checks['p0_records_recomputed'] = compare_csv(p0.OUT/'p0_all_records.csv',pr)
    checks['p0_aggregate_recomputed'] = compare_csv(p0.OUT/'p0_aggregate.csv',p0.aggregate(pr))
    boundary = load('qa_boundary','shift_granularity/failure_boundaries.py')
    br, sr = boundary.qwen_records(), boundary.support_escape()
    checks['qwen_boundary_recomputed'] = compare_csv(boundary.OUT/'qwen_corruption_boundary.csv',br)
    checks['support_escape_witness_recomputed'] = compare_csv(boundary.OUT/'support_escape_witness.csv',sr)
    hans = EXPS/'hans_suite'
    hm = read(hans/'outputs/manifest_5seed_anli.json')
    with (hans/'outputs/results.csv').open(encoding='utf-8',newline='') as f:
        hr = list(csv.DictReader(f))
    keys = {(int(r['seed']),r['condition'],r['split']) for r in hr}
    splits = sorted({r['split'] for r in hr})
    assert len(keys)==len(hr)==hm['rows']==150
    assert keys == {(s,c,t) for s in hm['seeds'] for c in hm['conditions'] for t in splits}
    counts = {}
    for rel,expected in [('heuristics_train_set.txt',30000),('heuristics_evaluation_set.txt',30000),('data/anli_dev_r123.jsonl',3200)]:
        with (hans/rel).open(encoding='utf-8') as f:
            count = sum(1 for _ in f) - (1 if rel.endswith('.txt') else 0)
        assert count==expected,(rel,count)
        counts[rel]=count
    checks['hans'] = dict(rows=len(hr),seeds=hm['seeds'],conditions=hm['conditions'],splits=splits,data_rows=counts,
        evidence_scope='Saved aggregate metrics and source datasets; per-example RoBERTa predictions/model checkpoints were not recovered.')
    llm = load('qa_llm','llm_cfpt/plot_real_llm_evidence.py')
    mf = read(llm.OUT/'manifest.json')
    for rel,expected in mf['source_files'].items():
        assert sha(llm.ROOT/rel.replace('\\','/'))==expected,rel
    assert len(llm.direction_data()['seeds'])==mf['direction_pools']==8
    assert len(llm.decision_data())==mf['decision_rows']==20
    assert len(llm.intervention_data()['metrics']['calibration']['W0']['KL']['per_pool'])==mf['consistency_pools']==8
    assert len(llm.slack_data()['qwen'])==mf['slack_heads']==4
    checks['real_llm_figure_inputs'] = dict(hash_matches=4,direction_pools=8,decision_rows=20,consistency_pools=8,slack_heads=4)
    certificate = load('qa_certificate','llm_cfpt/certificate_protocol.py')
    raw_cert = read(certificate.OUT/'certificate_raw.json')
    cert = certificate.summarize(raw_cert)
    # Historical table contains a few infinite analytic radii; compare_csv's
    # math.isclose handles equal infinities without inventing finite values.
    checks['certificate_summary_recomputed'] = compare_csv(certificate.OUT/'certificate_summary.csv',cert)
    checks['certificate_raw_rows'] = len(raw_cert)
    if args.render:
        for name,module in [('qwen',qwen),('p0',p0),('boundary',boundary),('llm',llm),('certificate',certificate)]:
            module.OUT = out/name
            module.OUT.mkdir()
        qwen.plot(qr)
        p0.all_records = pr
        p0.plot(p0.aggregate(pr),p0.OUT/'fig_p0_shift_granularity')
        boundary.plot(br,sr)
        llm.main()
        certificate.plot(cert)
        checks['rendered_entrypoints'] = ['qwen','p0','boundary','llm','certificate']
    checks['status'] = 'PASS'
    (out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
