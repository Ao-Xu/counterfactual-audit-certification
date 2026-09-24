import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from planning.planner import plan,evaluate
from planning.laws import support,draw_pilot,moments,draw_stats,lower_bound

R=Path(__file__).resolve().parent; C=json.loads((R/'manifest.json').read_text());OUT=R/'results'
def write(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    tmp=OUT/(name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(OUT/name)
def resources(law):
    pairs={k:support(law,k) for k in C['K_grid']}
    return {k:v[0] for k,v in pairs.items()},{k:(*moments(*v),v[1]) for k,v in pairs.items()}
def verify():
    for n,h in C['source_hashes'].items():
        assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
    assert hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest()==(R/'manifest.sha256').read_text().strip()
def read_lock():
    raw=(OUT/'budget_lock.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==(OUT/'budget_lock.sha256').read_text().strip(),'budget lock changed'
    lock=json.loads(raw)
    assert lock['manifest_sha']==(R/'manifest.sha256').read_text().strip(),'lock belongs to another manifest'
    return lock
def pilots():
    verify();assert not (OUT/'budget_lock.json').exists(),'immutable budget lock exists'
    rng=np.random.default_rng(C['pilot_seed']);rows=[];oracles=[]
    for law in C['laws']:
        supports,pop=resources(law)
        for target in C['targets']:
            # Dummy pilot values do not determine the population oracle.
            dummy=np.full((C['pilot_size'],max(C['K_grid']),2),.5)
            oo=plan(dummy,supports,C,target,population=pop,oracle=True)['selected']
            oracles.append(dict(law=law['name'],target=target,plans=oo,lower=lower_bound(law,target,C['rho'],C['delta'],C['delta_plan'])))
            for rep in range(C['outer_pilots']):
                pilot=draw_pilot(law,C['pilot_size'],max(C['K_grid']),rng)
                result=plan(pilot,supports,C,target)
                rows.append(dict(law=law['name'],target=target,rep=rep,**result))
                print('pilot',law['name'],target,rep,{k:(v['N'] if v else None) for k,v in result['selected'].items() if k in ['valid','plugin']},flush=True)
                write('progress.json',dict(stage='pilot',done=len(rows),total=len(C['laws'])*len(C['targets'])*C['outer_pilots']))
    write('budget_lock.json',dict(manifest_sha=(R/'manifest.sha256').read_text().strip(),pilots=rows,oracles=oracles))
    raw=(OUT/'budget_lock.json').read_bytes();(OUT/'budget_lock.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    write('progress.json',dict(stage='locked',done=len(rows),total=len(rows)))

def finals():
    verify();assert not (OUT/'final.json').exists(),'final outcomes already exist'
    lock=read_lock();rng=np.random.default_rng(C['final_seed']);rows=[]
    for i,pilot in enumerate(lock['pilots']):
        law=next(x for x in C['laws'] if x['name']==pilot['law']);sp,pop=resources(law)
        for method in ['valid','plugin']:
            en=pilot['selected'][method]
            if en is None:
                rows.append({k:pilot[k] for k in ['law','target','rep']}|dict(planner=method,feasible=False));continue
            k=en['K'];mu,cv=draw_stats(sp[k],pop[k][2],en['N'],C['final_replications'],rng)
            u=evaluate(mu,cv,en,C['rho'])
            rows.append({k:pilot[k] for k in ['law','target','rep']}|dict(planner=method,feasible=True,N=en['N'],K=k,estimator=en['method'],witness=en.get('witness'),
                successes=int((u<0).sum()),trials=len(u),mean_upper=float(u.mean()),min_upper=float(u.min()),max_upper=float(u.max())))
        write('progress.json',dict(stage='fresh final blocks',done=i+1,total=len(lock['pilots'])))
    write('final.json',dict(budget_lock_sha=(OUT/'budget_lock.sha256').read_text().strip(),rows=rows))
    write('progress.json',dict(stage='final complete',done=len(lock['pilots']),total=len(lock['pilots'])))

def benchmark():
    verify();assert not (OUT/'benchmarks.json').exists(),'benchmark exists'
    rng=np.random.default_rng(C['benchmark_seed']);rows=[]
    lock=read_lock()
    # Benchmark uses an independent RNG, never changes any locked plan.
    for law in C['laws']:
        sp,pop=resources(law)
        for k in C['K_grid']:
            for n in C['N_grid']:
                mu,cv=draw_stats(sp[k],pop[k][2],n,C['benchmark_replications'],rng)
                ens=[]
                for alloc in C['mean_fractions']:
                    for method in ['paired','separate']:
                        div=C['J'] if method=='paired' else C['J']+1
                        ens.append(dict(method=method,K=k,N=n,eta=None,allocation=alloc,em=C['delta']*alloc/div,eq=C['delta']*(1-alloc)/div))
                for eta in C['eta_grid']:ens.append(dict(method='generic',K=k,N=n,eta=eta,allocation=1.,em=C['delta']/C['J'],eq=None))
                for en in ens:
                    u=evaluate(mu,cv,en,C['rho'])
                    rows.append(dict(law=law['name'],**en,successes=int((u<0).sum()),trials=len(u)))
        print('benchmark',law['name'],'done',flush=True)
    write('benchmarks.json',dict(rows=rows,scope='Independent descriptive empirical grid; never used to alter locked plans.'))
    write('progress.json',dict(stage='complete',done=1,total=1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','final','benchmark']);a=p.parse_args()
    if not __debug__:raise SystemExit('Run without -O: frozen integrity assertions must stay enabled.')
    OUT.mkdir(parents=True,exist_ok=True)
    {'pilot':pilots,'final':finals,'benchmark':benchmark}[a.stage]()
