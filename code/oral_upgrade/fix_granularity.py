"""Audit the old nonlinear noise bias; corrected noise-free finite-table target.
Local-per-cell chi-square constraints (not Theorem 3's average budget).
Bootstrap intervals below remain diagnostics, never finite-sample guarantees.
"""
import importlib.util,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('legacy',R.parent/'overleaf/experiments/shift_granularity/controlled_separation.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old)

def point_many(x,group,cls):
    # x may have leading bootstrap dimension.
    if cls=='global':z=x.mean(-2,keepdims=True)
    elif cls=='group':z=np.stack([x[...,group==g,:].mean(-2) for g in range(8)],axis=-2)
    else:z=x
    return (z.mean(-1)+np.sqrt(.25*z.var(-1))).mean(-1)

def main():
    m=json.loads((R/'frozen_manifest.json').read_text())['figure2'];rng=np.random.default_rng(m['seed']);out=[]
    for si,scenario in enumerate(['global','group','task']):
        pop,groups,truth=old.one_population(scenario,old.SEED+si);target=truth[scenario]
        for n in m['n']:
            for noisy in [False,True]:
                vals={c:[] for c in ['global','group','task']}
                for rep in range(m['repetitions']):
                    ix=np.concatenate([rng.choice(np.flatnonzero(groups==g),n//8,replace=False) for g in range(8)]);gg=groups[ix]
                    x=pop[ix].copy()
                    if noisy:x=np.clip(x+rng.normal(0,.035,x.shape),-1,1)
                    bi=np.stack([np.concatenate([rng.choice(np.flatnonzero(gg==g),n//8,replace=True) for g in range(8)]) for _ in range(40)])
                    for cls in vals:
                        p=float(point_many(x,gg,cls));boots=point_many(x[bi],gg,cls);rad=float(np.quantile(abs(boots-p),.95));vals[cls].append((p,rad,p+rad>=target-1e-12))
                for cls,rr in vals.items():
                    a=np.array(rr);out.append(dict(scenario=scenario,n=n,noise='legacy_fixed_noise' if noisy else 'exact_loss_table',audit_class=cls,target=target,point=float(a[:,0].mean()),bias=float(a[:,0].mean()-target),radius=float(a[:,1].mean()),coverage=float(a[:,2].mean()),repetitions=len(rr)))
    (R/'results/granularity.json').write_text(json.dumps(dict(diagnosis='Nonlinear noise/Jensen bias persists for task norm at fixed observation noise. Corrected exact finite loss table removes noise; budget=256 is a census. These are local-per-cell shift targets and descriptive bootstrap coverage, not Theorem 3 confidence.',rows=out),indent=2))

if __name__=='__main__':main()
