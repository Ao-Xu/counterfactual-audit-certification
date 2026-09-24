"""Frozen independent repetitions; population truth never enters certificate."""
import json,math,time,hashlib
from pathlib import Path
import numpy as np
from oral_upgrade.certificate import audit_family,Corrections

ROOT=Path(__file__).resolve().parent

def wilson(k,n):
    z=1.96;p=k/n;d=1+z*z/n
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;c=(p+z*z/(2*n))/d
    return [c-h,c+h]

def main():
    manifest=ROOT/'frozen_manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==(ROOT/'frozen_manifest.sha256').read_text().split()[0]
    cfg=json.loads(manifest.read_text());c=cfg['controlled'];co=np.array(c['paired_coefficients']);names=c['candidates'];rng=np.random.default_rng(c['seed'])
    cases=[(n,k,.05,4) for n in c['n_grid'] for k in c['K_grid']]
    cases += [(n,4,r,j) for n in c['n_grid'] for r,j in [(.25,4),(.05,2)]]
    results=[];start=time.time()
    for n,K,rho,J in cases:
        cfco=co[:J];nn=names[:J];truth=cfco[:,0]+np.sqrt(rho)*np.abs(cfco[:,2]);best=nn[int(np.argmin(truth))]
        stats={m:[] for m in ['paired','separate','bounded','paired_validation','reference_only']}
        for rep in range(c['repetitions']):
            arrays=[]
            for _ in range(2):
                a=rng.choice([-1.,1.],(n,1));s=rng.choice([-1.,1.],(n,K))
                f=.5+.04*a+.3*s
                g=cfco[:,0,None,None]+cfco[:,1,None,None]*a+cfco[:,2,None,None]*s
                arrays.extend([f[None]+g,f])
            ans=audit_family(*arrays,cal_ids=list(range(n)),audit_ids=list(range(n,2*n)),rho=rho,delta=cfg['delta'],mode='iid',policy_names=nn,corrections=Corrections(0,0,evidence='exact bounded finite-state witness'))
            for method,chosen in ans['selections'].items():
                key={'paired':'U_pair','separate':'U_separate','bounded':'U_bounded','paired_validation':'U_reference','reference_only':'Delta_hat'}[method]
                upper=np.array([r[key] for r in ans['rows']]);safe=truth<0;accepted=upper<0
                ix=nn.index(chosen) if chosen!='factual' else None
                stats[method].append(dict(power=float(accepted[safe].mean()) if safe.any() else 0,
                    false_accept=float(accepted[~safe].mean()) if (~safe).any() else 0,
                    any_false_accept=bool(np.any(accepted&~safe)),fallback=ix is None,
                    safe_selection=ix is not None and truth[ix]<0,best_safe=chosen==best,
                    harmful_selection=ix is not None and truth[ix]>0,
                    realized_worst_delta=float(truth[ix]) if ix is not None else 0.,
                    coverage=bool(np.all(upper>=truth)),
                    width=float(np.mean(upper-np.array([r['Delta_hat'] for r in ans['rows']])))))
        for method,rr in stats.items():
            row=dict(n=n,K=K,rho=rho,J=J,method=method,replications=len(rr),independent_anchors=2*n,
                     forward_rows=2*n*K,truth=truth.tolist(),best_safe_policy=best)
            row.update({key:float(np.mean([r[key] for r in rr])) for key in rr[0]})
            row['safe_selection_MC95']=wilson(sum(r['safe_selection'] for r in rr),len(rr));row['false_selection_MC95']=wilson(sum(r['harmful_selection'] for r in rr),len(rr))
            results.append(row)
        (ROOT/'results/controlled.json').write_text(json.dumps(dict(manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest(),seconds=time.time()-start,results=results),indent=2))
        print(n,K,rho,J,'paired safe',results[-5]['safe_selection'],flush=True)
    print('done seconds',time.time()-start)

if __name__=='__main__':main()
