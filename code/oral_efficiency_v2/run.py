"""Frozen v2 simulations. No population moments enter any certificate."""
import json, hashlib, math, time, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'frozen_manifest.json').read_text())
assert hashlib.sha256((ROOT/'frozen_manifest.json').read_bytes()).hexdigest()==(ROOT/'frozen_manifest.sha256').read_text().split()[0]

def kl_upper(mean,count,cap,e):
    a=np.clip(np.asarray(mean,dtype=float)/cap,0,1)
    lo=a.copy(); hi=np.ones_like(a); target=math.log(1/e)/count
    for _ in range(48):
        u=(lo+hi)/2
        with np.errstate(divide='ignore',invalid='ignore'):
            kl=np.where(a>0,a*np.log(a/u),0)+np.where(a<1,(1-a)*np.log((1-a)/(1-u)),0)
        good=kl<=target
        lo=np.where(good,u,lo); hi=np.where(good,hi,u)
    return cap*(lo+hi)/2

def radius(dmean,pairs,n,span,e):
    v=np.minimum(span**2/4,kl_upper(dmean,pairs,span**2/2,e))
    t=math.log(2/e); a=span*t/(3*n)
    return np.minimum(a+np.sqrt(2*v*t/n+a*a),span*np.sqrt(t/(2*n)))

def cert(qmean,dmean,m,n,span,e,mode='enumerated'):
    v=np.minimum(span**2/4,kl_upper(qmean,m,span**2/(4 if mode=='enumerated' else 2),e))
    return radius(dmean,m//2,n,span,e)+np.sqrt(CFG['rho']*v)

def wilson(k,r):
    z=1.95996398454;p=k/r;den=1+z*z/r
    mid=(p+z*z/(2*r))/den; w=z*np.sqrt(p*(1-p)/r+z*z/(4*r*r))/den
    return [float(mid-w),float(mid+w)]

def primary(refine=False):
    c=dict(CFG['primary'])
    if refine:
        path=ROOT/'power_resolution_manifest.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==path.with_suffix('.sha256').read_text().strip()
        c.update(json.loads(path.read_text()))
    rng=np.random.default_rng(c['seed']); R=c['replications']; J=len(c['candidate_names']);rho=CFG['rho']; delta=CFG['delta']
    eta=np.geomspace(**dict(start=c['generic_dual_eta_grid']['start'],stop=c['generic_dual_eta_grid']['stop'],num=c['generic_dual_eta_grid']['count']))
    rows=[]
    for n in c['n_grid']:
        m=n
        # Counts are exact sufficient statistics of IID paired calibration anchors.
        blocks=rng.multinomial(m//4,[.25]*4,size=(2,R))
        full=blocks.sum(axis=0); opposite=full[:,1]+full[:,2]
        a_audit=2*rng.binomial(n,.5,size=R)/n-1
        for name,(c0,ca,cs) in zip(c['candidate_names'],c['candidate_coefficients']):
            f0,fa,fs=c['factual']; mu=c0-f0; a=ca-fa; b=cs-fs
            d=2*a*a*opposite/(m//2)
            ep=delta/(4*J); es=delta/(4*(J+1))
            mean=mu+a*a_audit
            up=mean+cert(b*b,d,m,n,2,ep)
            uf=cert(fs*fs,2*fa*fa*opposite/(m//2),m,n,1,es)
            uc=cert(cs*cs,2*ca*ca*opposite/(m//2),m,n,1,es)
            us=mean+uf+uc
            # Generic conditional chi-square DRO: dual choice on first half only.
            first_a=2*(blocks[0,:,3]-blocks[0,:,0])/(m//2)
            dual_extra=np.where(abs(b)<=2*eta,b*b/(4*eta),abs(b)-eta)
            tuned=np.argmin(mu+a*first_a[:,None]+dual_extra[None,:]+rho*eta[None,:],axis=1)
            d_second=2*a*a*(blocks[1,:,1]+blocks[1,:,2])/(m//4)
            ug=mean+dual_extra[tuned]+rho*eta[tuned]+radius(d_second,m//4,n,2,ep)
            truth=mu+abs(b)*np.sqrt(rho)
            sep_truth=mu+(abs(cs)+abs(fs))*np.sqrt(rho)
            for method,u in [('paired_KL',up),('separate_KL',us),('generic_paired_DRO',ug)]:
                count=int((u<0).sum()); covered=int((u>=truth-1e-12).sum())
                rows.append(dict(n=n,m=m,anchors=n+m,candidate=name,method=method,accepted=count,replications=R,power=count/R,ci=wilson(count,R),population_paired=truth,population_separate=sep_truth,upper_mean=float(np.mean(u)),coverage=covered/R,generic_grid_slack=float(np.min(dual_extra+rho*eta)-abs(b)*np.sqrt(rho))))
        (ROOT/'results/progress.json').write_text(json.dumps(dict(stage='primary',n=n)))
    thresholds=[]
    for name in c['candidate_names']:
        for target in [.8,.9]:
            for conservative in [False,True]:
                item=dict(candidate=name,target=target,criterion='Wilson lower' if conservative else 'point power')
                for meth in c['methods']:
                    r=[r for r in rows if r['candidate']==name and r['method']==meth]
                    meets=[(x['ci'][0] if conservative else x['power'])>=target for x in r]
                    ix=next((i for i in range(len(r)) if all(meets[i:])),None)
                    item[meth]=r[ix]['anchors'] if ix is not None else None
                p=item['paired_KL']; s=item['separate_KL'];g=item['generic_paired_DRO']
                item['separate_over_paired']=s/p if s and p else None
                item['generic_over_paired']=g/p if g and p else None
                thresholds.append(item)
    result=dict(config_sha256=hashlib.sha256((ROOT/'frozen_manifest.json').read_bytes()).hexdigest(),resolution_manifest_sha256=hashlib.sha256((ROOT/'power_resolution_manifest.json').read_bytes()).hexdigest() if refine else None,seed=c['seed'],replications=R,n_grid=c['n_grid'],rows=rows,thresholds=thresholds)
    (ROOT/('results/primary_refined.json' if refine else 'results/primary.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(thresholds,indent=2),flush=True)

def allocation():
    c=CFG['allocation'];rng=np.random.default_rng(c['seed']);R=c['replications'];rows=[];e=CFG['delta']/(4*c['J']); b=c['paired_style'];mu=c['paired_mean'];start=time.time()
    # Common random blocks across cells reduce Monte Carlo comparison noise;
    # within each cell calibration and audit are disjoint independent draws.
    for geom,a in c['geometries'].items():
      for K in c['K_grid']:
        sums={(m,n):[0,0.,0.] for m in c['m_grid'] for n in c['n_grid']}
        for rep in range(R):
          size=max(c['m_grid'])+max(c['n_grid'])
          A=2*rng.integers(0,2,size)-1
          pairs=rng.multinomial(K//2,[.25]*4,size=size)
          S=2*(pairs[:,3]-pairs[:,0])/K
          M=mu+a*A+b*S
          Q=2*b*b*(pairs[:,1]+pairs[:,2])/(K//2)
          audit=M[max(c['m_grid']):]
          for m in c['m_grid']:
            d=np.mean((M[:m:2]-M[1:m:2])**2)/2
            q=Q[:m].mean()
            vu=float(np.minimum(1,kl_upper(q,m,2,e)))
            for n in c['n_grid']:
              r=float(radius(d,m//2,n,2,e));u=float(audit[:n].mean()+r+np.sqrt(CFG['rho']*vu))
              s=sums[m,n];s[0]+=int(u<0);s[1]+=r;s[2]+=np.sqrt(CFG['rho']*vu)
          if (rep+1)%50==0:
            (ROOT/'results/progress.json').write_text(json.dumps(dict(stage='allocation',geometry=geom,K=K,rep=rep+1,total=R,elapsed=time.time()-start)))
        for (m,n),(accepted,r,q) in sums.items():
          rows.append(dict(geometry=geom,K=K,m=m,n=n,anchors=m+n,evaluations=K*(m+n),power=accepted/R,accepted=accepted,replications=R,ci=wilson(accepted,R),radius_mean=r/R,nuisance_upper_mean=q/R,population_envelope=mu+abs(b)*np.sqrt(CFG['rho']),anchor_variance=a*a,sibling_variance=b*b,mean_variance=a*a+b*b/K))
        (ROOT/'results/allocation.json').write_text(json.dumps(dict(rows=rows),indent=2))
        print(geom,K,'done',round(time.time()-start,1),flush=True)
    (ROOT/'results/progress.json').write_text(json.dumps(dict(stage='complete',elapsed=time.time()-start)))

def checks():
    sys.path.insert(0,str(ROOT.parent))
    from oral_upgrade.certificate import bounded_mean_upper
    rng=np.random.default_rng(5)
    for c in [1,2,.5]:
      for size in [10,128]:
       for x in [np.zeros(size),rng.uniform(0,c,size)]:
        expected=bounded_mean_upper(x,c,.001)
        assert abs(float(kl_upper(x.mean(),size,c,.001))-expected)<1e-11
    for n in [10,100,1000]:
      assert abs(float(kl_upper(0,n,2,.01))-2*(-np.expm1(-math.log(100)/n)))<1e-12
    # Explicit full-array paired and marginal moments match sufficient statistics.
    A=rng.choice([-1,1],128);pairs=A.reshape(-1,2);opp=(pairs[:,0]!=pairs[:,1]).sum()
    M=.02+.01*A
    assert abs(np.mean(np.diff(M.reshape(-1,2),axis=1)**2)/2-2*.01**2*opp/64)<1e-14
    # Dual exactness and loss boundedness on the frozen finite support.
    for c0,a,b in CFG['primary']['candidate_coefficients']:
      losses=c0+a*np.array([-1,1])[:,None]+b*np.array([-1,1])[None,:]
      assert losses.min()>=0 and losses.max()<=1
      assert c0-.55+(abs(b)+.03)*math.sqrt(.05)<0
    (ROOT/'results/checks.json').write_text(json.dumps(dict(passed=True,checks=['KL equivalence','V=0 exact endpoint','sufficient statistics equivalence','loss boundedness','negative equal envelopes'])))
    print('checks passed',flush=True)

if __name__=='__main__':
    checks()
    if len(sys.argv)==1 or sys.argv[1]=='primary':primary()
    if len(sys.argv)>1 and sys.argv[1]=='allocation':allocation()
    if len(sys.argv)>1 and sys.argv[1]=='refine':primary(refine=True)
