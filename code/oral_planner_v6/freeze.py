import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
if (R/'manifest.json').exists():raise SystemExit('Already frozen; refusing overwrite')
# No outcomes on these laws have been opened. Parameters are fixed here once.
rho=.05
laws=[dict(name='A',a=.22,w=.15,b=.035,h=0.,gamma=.035),
      dict(name='B',a=.30,w=.12,b=.065,h=.5,gamma=.025),
      dict(name='C',a=.16,w=.18,b=.095,h=-.4,gamma=.020)]
for law in laws:law['d']=law['gamma']+rho**.5*law['b']
protocol=dict(version='v6',rho=rho,delta=.025,delta_plan=.05,J=1,
    K_grid=[2,8],eta_grid=[.1,.25,.5],mean_fractions=[.35,.65],
    N_grid=sorted(set([int(round(256*1.25**i)) for i in range(32)])),
    pilot_size=2048,design_pilot=256,targets=[.8,.9],
    outer_pilots=24,final_replications=256,benchmark_replications=1024,
    pilot_seed=23092711,final_seed=23092712,benchmark_seed=23092713,
    laws=laws,known_support='Exact feature enclosures supplied without probabilities; final endpoints keep global loss bounds.',
    selection='Minimize N then KN; fixed lexical tie-break. Return infeasible if no witnessed budget.',
    final_rule='Use exactly locked N fresh IID anchors per replication, no early stopping.',
    benchmarks='Population-only Gaussian forecast; true-moment Theorem4; naive pilot Gaussian; same-witness oracle; no parameter fitting.',
    source_hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['planner.py','laws.py','THEORY.md']})
raw=json.dumps(protocol,indent=2).encode();(R/'manifest.json').write_bytes(raw)
(R/'manifest.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
print('Frozen',hashlib.sha256(raw).hexdigest())
