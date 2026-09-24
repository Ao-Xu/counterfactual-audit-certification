"""Exploratory budget replay on inspected structure blocks, NOT confirmation."""
import json
from pathlib import Path
import numpy as np
from finite_catalog.certificate import audit_family,Corrections
R=Path(__file__).resolve().parent
def main():
    rng=np.random.default_rng(2309202604);w=np.array([.325,.325,.175,.175]);names=['P50_paired','P995','cue70','cue90','cue100'];fs=[]
    for seed in [7201,7202,7203,7204,7301,7302,7303,7304,7305,7306,7307,7308]:
        root=R/'artifacts/hans_extension';f=dict(np.load(root/f'{seed}_factual/predictions.npz'));cs=[dict(np.load(root/f'{seed}_{p}/predictions.npz')) for p in names]
        fs.append((seed,f,cs))
    out=[]
    for n in [64,128,252]:
        for K in [2,4,16]:
            for seed,f,cs in fs:
                counts={m:[] for m in ['paired','separate','bounded','paired_validation','reference_only']}
                for rep in range(30):
                    arrays=[]
                    for split in ['calibration','reference']:
                        ix=rng.choice(252,n,replace=False);styles=rng.choice(4,(n,K),p=w)
                        factual=f[split][ix[:,None],styles];cf=np.stack([c[split][ix[:,None],styles] for c in cs]);arrays.extend([cf,factual])
                    a=audit_family(*arrays,cal_ids=list(range(n)),audit_ids=list(range(n,2*n)),rho=.05,delta=.025,mode='iid',policy_names=names,corrections=Corrections(0,0,evidence='Historical gold-conditional four-state table; no new confirmation'))
                    for method,p in a['selections'].items():
                        if p=='factual':counts[method].append((1,0.,0.));continue
                        g=cs[names.index(p)]['test']-f['test'];m=g@w;v=((g-m[:,None])**2)@w;env=float(m.mean()+np.sqrt(.05*v.mean()));counts[method].append((0,float(m.mean()),env))
                for method,x in counts.items():
                    a=np.array(x);out.append(dict(n=n,K=K,seed=seed,method=method,replay_repetitions=30,independent_structures_per_role=n,total_structures=2*n,fallback_rate=float(a[:,0].mean()),mean_test_reference=float(a[:,1].mean()),mean_empirical_test_envelope=float(a[:,2].mean())))
    (R/'results/hans_budget_replay.json').write_text(json.dumps(dict(scope='Conditional exploratory resampling of inspected blocks. Seeds are fixed models, not new task anchors; repeats are not new datasets. Population power and false-accept probability are unknown here.',rows=out),indent=2))
if __name__=='__main__':main()
