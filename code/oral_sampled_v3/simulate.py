import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,hashlib,time,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from oral_sampled_v3.methods import kl_radius,eb
from oral_efficiency_v2.run import kl_upper,wilson
R=Path(__file__).resolve().parent;P=json.loads((R/'frozen_manifest.json').read_text());assert hashlib.sha256((R/'frozen_manifest.json').read_bytes()).hexdigest()==(R/'frozen_manifest.sha256').read_text().strip()
ETAS=np.geomspace(P['eta_grid']['low'],P['eta_grid']['high'],P['eta_grid']['count'])
GEOS=[dict(correlation=r,anchor_scale=a,sensitivity=b,heterogeneity=h,support=L) for L,r,a,b,h in itertools.product(P['nuisance_support'],P['correlations'],P['anchor_scales'],P['sensitivity_scales'],P['heterogeneity'])]

def coeffs(geos):
    gm=[];qc=[];fc=[];cc=[];delta=[]
    for g in geos:
      r,a,b,h=[g[k] for k in ['correlation','anchor_scale','sensitivity','heterogeneity']];scale=np.sqrt(1+h*h)
      for k in P['candidate_sensitivity_multipliers']:
        gm.append([a*(r-1),a*np.sqrt(1-r*r),k*b/scale,k*b*h/scale,0,0])
        qc.append([0,0,0,0,(k*b)**2,2*h*(k*b)**2/(1+h*h)])
        fc.append([a,0,-.5*b/scale,-.5*b*h/scale,0,0])
        cc.append([a*r,a*np.sqrt(1-r*r),(k-.5)*b/scale,(k-.5)*b*h/scale,0,0])
        delta.append(-P['robust_margin']-np.sqrt(P['rho'])*k*b)
    return tuple(np.asarray(x) for x in [gm,qc,fc,cc,delta])

def stats(F,sizes):
    """Exact first/second moments and disjoint-anchor variance summaries."""
    sizes=sorted(set(sizes));out={};ss=np.zeros(6);xx=np.zeros((6,6));dd=np.zeros((6,6));last=0
    for n in sizes:
      block=F[last:n];ss+=block.sum(0);xx+=block.T@block
      diff=block[::2]-block[1::2];dd+=diff.T@diff/2
      mean=ss/n;cov=(xx-n*np.outer(mean,mean))/(n-1)
      out[n]=(mean,cov,dd/(n//2));last=n
    return out

def project(c,A):return np.einsum('ij,jk,ik->i',c,A,c)

def run(phase):
    st=time.time();reps=P['planning_reps' if phase=='plan' else 'confirmation_reps'];seed=P['planning_seed' if phase=='plan' else 'confirmation_seed'];rng=np.random.default_rng(seed)
    Ns=P['total_anchors'];maxN=max(Ns);sizes=set(Ns)
    for N in Ns:
      for f in P['calibration_fractions']:sizes|={int(N*f),N-int(N*f)}
    counts={};covered={}
    locked=[]
    if phase=='confirm':
      path=R/'allocation_lock.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==(R/'allocation_lock.sha256').read_text().strip();locked=json.loads(path.read_text())['allocations']
    allowed={tuple(a['key']) for a in locked if a['key']}
    for rep in range(reps):
      # One IID stream; each allocation uses disjoint prefix calibration and suffix audit. All methods see the identical N anchors.
      A=rng.choice([-1.,1.],size=maxN);W=rng.choice([-1.,1.],size=maxN);H=rng.choice([-1.,1.],size=maxN)
      for L in P['nuisance_support']:
       geos=[g for g in GEOS if g['support']==L];ids=[GEOS.index(g) for g in geos];gm,qc,fc,cc,offset=coeffs(geos)
       support=np.linspace(-1,1,L);support/=np.std(support);S=support[rng.integers(L,size=(maxN,max(P['K'])))]
       for K in P['K']:
        sub=S[:,:K];sm=sub.mean(1);q=((sub[:,::2]-sub[:,1::2])**2).mean(1)/2
        F=np.column_stack([A,W,sm,H*sm,q,H*q]);cal=stats(F,sizes)
        # Generic EB can use all N independent anchors (no held-out calibration).
        pooled=stats(F[:maxN],Ns)
        for N in Ns:
          pm,pcov,_=pooled[N];ym=(gm@pm+offset)[:,None]+(qc@pm)[:,None]/(4*ETAS)
          mv=project(gm,pcov);qv=project(qc,pcov);cross=np.einsum('ij,jk,ik->i',gm,pcov,qc)
          yv=mv[:,None]+qv[:,None]/(16*ETAS**2)+cross[:,None]/(2*ETAS)
          up=eb(ym,yv,N,1+np.maximum(1,1/(2*ETAS)),P['delta']/(P['J']*len(ETAS)))+P['rho']*ETAS
          gu=up.min(1)
          for local,geo in enumerate(ids):
           key=(geo,'generic_DRO_EB',N,0,N,K)
           if phase=='plan' or key in allowed:
            counts.setdefault(key,np.zeros(3,dtype=int));counts[key]+=gu[3*local:3*local+3]<0
            covered.setdefault(key,np.zeros(3,dtype=int));covered[key]+=gu[3*local:3*local+3]>=-P['robust_margin']
          for f in P['calibration_fractions']:
            m=int(N*f);n=N-m;cm,_,cd=cal[m];am=(N*pooled[N][0]-m*cm)/n;mean=gm@am+offset
            ep=P['delta']/(4*P['J']);es=P['delta']/(4*(P['J']+1))
            vu=np.minimum(1,kl_upper(qc@cm,m,2,ep));pu=mean+kl_radius(project(gm,cd),m,n,2,ep)+np.sqrt(P['rho']*vu)
            # Each marginal Q is a multiple of the same empirical style variance.
            k=np.tile(P['candidate_sensitivity_multipliers'],len(geos));fq=(qc@cm)*(.5/k)**2;cq=(qc@cm)*((k-.5)/k)**2
            fvu=np.minimum(.25,kl_upper(fq,m,.5,es));cvu=np.minimum(.25,kl_upper(cq,m,.5,es))
            su=mean+kl_radius(project(fc,cd),m,n,1,es)+kl_radius(project(cc,cd),m,n,1,es)+np.sqrt(P['rho'])*(np.sqrt(fvu)+np.sqrt(cvu))
            for method,u in [('paired_KL',pu),('separate_KL',su)]:
             for local,geo in enumerate(ids):
              key=(geo,method,N,m,n,K)
              if phase=='plan' or key in allowed:
               counts.setdefault(key,np.zeros(3,dtype=int));counts[key]+=u[3*local:3*local+3]<0
               covered.setdefault(key,np.zeros(3,dtype=int));covered[key]+=u[3*local:3*local+3]>=-P['robust_margin']
      if (rep+1)%10==0:
        (R/'results/progress.json').write_text(json.dumps(dict(stage=phase,rep=rep+1,total=reps,seconds=time.time()-st)));print(phase,rep+1,round(time.time()-st,1),flush=True)
    rows=[]
    for key,acc in counts.items():
      geo,method,N,m,n,K=key
      for j,x in enumerate(acc):rows.append(dict(key=list(key),geometry_id=geo,geometry=GEOS[geo],method=method,total_anchors=N,m=m,n=n,K=K,evaluations=N*K,candidate=j,accepted=int(x),replications=reps,power=float(x/reps),ci=wilson(int(x),reps),coverage=float(covered[key][j]/reps)))
    out=dict(manifest_sha256=hashlib.sha256((R/'frozen_manifest.json').read_bytes()).hexdigest(),phase=phase,seconds=time.time()-st,rows=rows)
    (R/f'results/{phase}.json').write_text(json.dumps(out,indent=2))
    if phase=='plan':
      allocations=[]
      for geo,method,target,cost in itertools.product(range(len(GEOS)),P['methods'],[.8,.9],['total_anchors','evaluations']):
        eligible=[r for r in rows if r['geometry_id']==geo and r['method']==method and r['candidate']==P['primary_candidate'] and r['ci'][0]>=target]
        best=min(eligible,key=lambda r:(r[cost],r['total_anchors'],r['K'],r['m'])) if eligible else None
        allocations.append(dict(geometry_id=geo,method=method,target=target,cost=cost,key=best['key'] if best else None,planning=best))
      lock=R/'allocation_lock.json';lock.write_text(json.dumps(dict(frozen_after_planning_before_confirmation=True,allocations=allocations),indent=2));(R/'allocation_lock.sha256').write_text(hashlib.sha256(lock.read_bytes()).hexdigest()+'\n')
    print('complete',phase,time.time()-st,flush=True)
if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'plan')
