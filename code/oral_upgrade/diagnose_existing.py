"""Every candidate in the reusable fresh-HANS, extension and QA archives.
Retrospective: these inspected blocks are not a prospective confirmation.
"""
import json,csv,math
from pathlib import Path
import numpy as np
from oral_upgrade.certificate import *
ROOT=Path(__file__).resolve().parent

def upper_from_mean(a,cap,n,e):
    a=max(0,min(1,a/cap));lo=a;hi=1.;t=math.log(1/e)/n
    for _ in range(50):
        b=(lo+hi)/2
        kl=(a*math.log(a/b) if a and b<1 else 0)+((1-a)*math.log((1-a)/(1-b)) if a<1 and b<1 else 0)
        if b>=1 or kl>t:hi=b
        else:lo=b
    return hi*cap

def planned_anchors(d,v,var,rho,J,delta):
    """Plug-in expected-calibration planning, not a confidence guarantee."""
    if d+math.sqrt(rho*v)>=0:return None
    e=delta/(12*J);t=math.log(2/e)
    def upper(n):
        vu=min(1.,upper_from_mean(v,1.,n,e));su=min(1.,upper_from_mean(var,2.,n//2,e));a=2*t/(3*n)
        return d+a+math.sqrt(2*su*t/n+a*a)+math.sqrt(rho*vu)
    hi=4
    while hi<100000000 and upper(hi)>=0:hi*=2
    if upper(hi)>=0:return None
    lo=4
    while hi-lo>1:
        mid=(hi+lo)//2
        if upper(mid)<0:hi=mid
        else:lo=mid
    return hi

def families():
    for task in ['hans','hans_extension','qa']:
        groups={}
        for path in sorted((ROOT/'artifacts'/task).glob('*/predictions.npz')):
            key=path.parent.name
            if task=='qa':budget,seed,policy=key.split('_',2);fam=budget+'_'+seed
            else:seed,policy=key.split('_',1);fam=seed
            groups.setdefault(fam,{})[policy]=path
        for fam,paths in groups.items():yield task,fam,paths

def main():
    fs=list(families());allrows=[];full=[]
    for task,fam,paths in fs:
        baseline='baseline_0' if task=='qa' else 'factual';f=dict(np.load(paths[baseline]));names=sorted(set(paths)-{baseline});cs=[dict(np.load(paths[p])) for p in names];w=np.ones(4)/4 if task=='qa' else np.array([.325,.325,.175,.175]);n=len(f['reference']);m=len(f['calibration'])
        ans=audit_family(np.stack([c['calibration'] for c in cs]),f['calibration'],np.stack([c['reference'] for c in cs]),f['reference'],cal_ids=['cal'+str(i) for i in range(m)],audit_ids=['ref'+str(i) for i in range(n)],rho=.05,delta=.025/len(fs),weights=w,policy_names=names,corrections=Corrections(0,0,evidence='Conditional on benchmark gold labels; exact four-state law and task-preserving formatting/noun bijections. No natural CF semantic claim.'))
        # Decisions are recorded from cal/ref before loading test values below.
        for row,c in zip(ans['rows'],cs):
            g=c['calibration']-f['calibration'];v=row['VDelta_hat'];d=row['Delta_hat'];plug=d+math.sqrt(.05*v)
            exact=exact_finite_robust(g,w,.05);cm=g@w;calenv=float(cm.mean()+math.sqrt(.05*v))
            testg=c['test']-f['test'];tm=testg@w;tv=((testg-tm[:,None])**2)@w
            r={k:row[k] for k in ['policy','Delta_hat','sampling_radius','VDelta_hat','VDelta_upper','U_pair','L_pair','U_separate','U_bounded','decision']}
            r.update(task=task,family=fam,n_cal=m,n_audit=n,J=len(names),rho=.05,delta=.025/len(fs),
                plugin_envelope=plug,diagnosis='certified' if row['U_pair']<0 else 'A_empirical_envelope_nonnegative' if plug>=0 else 'B_empirical_statistical_width',
                calibration_catalog_Psi=exact['value'],calibration_catalog_envelope=calenv,nonlocal_envelope_slack=max(0,calenv-exact['value']),DRO_duality_gap=exact['duality_gap'],
                planned_n_each_block=planned_anchors(d,v,float(np.var(cm,ddof=1)),.05,len(names),.025/len(fs)),
                empirical_test_reference=float(tm.mean()),empirical_test_envelope=float(tm.mean()+math.sqrt(.05*tv.mean())),
                selected=row['policy']==ans['selections']['paired'])
            if task=='qa':r.update(test_EM_delta=float(((c['test_em']-f['test_em'])@w).mean()),test_F1_delta=float(((c['test_f1']-f['test_f1'])@w).mean()))
            allrows.append(r)
        full.append(dict(task=task,family=fam,**ans))
        print(task,fam,len(names),'done',flush=True)
    (ROOT/'results/existing_certificates.json').write_text(json.dumps(full,indent=2))
    keys=list(dict.fromkeys(k for r in allrows for k in r))
    with (ROOT/'results/failure_decomposition.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(allrows)
    (ROOT/'results/failure_decomposition.json').write_text(json.dumps(dict(scope='Retrospective empirical diagnosis, not population identification. Exact Psi refers ONLY to the uniform calibration catalog. Planned n uses plug-in moments, not guaranteed future power. Joint delta across all listed families.',rows=allrows),indent=2))

if __name__=='__main__':main()
