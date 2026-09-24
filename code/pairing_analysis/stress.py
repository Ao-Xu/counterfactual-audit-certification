import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
from pairing_analysis.methods import radius
from certificate_efficiency.run import kl_upper
R=Path(__file__).resolve().parent;P=json.loads((R/'frozen_manifest.json').read_text());C=json.loads(Path('shared_anchor/frozen_manifest.json').read_text())['stress']
def run(phase):
 cfg=P['stress'];ns=np.array(cfg['N_grid']);nn=ns[:,None];reps=cfg['planning_reps' if phase=='plan' else 'confirmation_reps'];rng=np.random.default_rng(cfg['planning_seed' if phase=='plan' else 'confirmation_seed']);K=C['K'];a=C['a'];b=C['b'];h=C['h'];p=C['skew_positive_probability'];rho=C['rho'];delta=C['delta'];eta=np.array(P['eta_grid'])
 sv=np.array([-np.sqrt(p/(1-p)),np.sqrt((1-p)/p)]);pr=np.array([1-p,p]);v=b*b*np.array([1-h,1+h]);skew=pr@sv**3;xi=np.mean(v**1.5)*skew/K
 tt=np.array([-1,0,1])*.08-xi/(b*b*h);ww=np.sqrt(a*a-tt*tt);d=-.025-np.sqrt(rho)*b
 cnt={};mean={};sq={};st=time.time()
 if phase=='confirm':assert (R/'stress_lock.json').exists()
 for rep in range(reps):
  H=rng.choice([-1.,1.],max(ns));W=rng.choice([-1.,1.],max(ns));s=rng.choice(sv,(max(ns),K),p=pr)
  # Shared primitive draws across covariance signs.
  M=d+H[:,None]*tt+W[:,None]*ww+(np.sqrt(b*b*(1+h*H))*s.mean(1))[:,None]
  Q=b*b*(1+h*H)*((s[:,::2]-s[:,1::2])**2).mean(1)/2
  mm=np.cumsum(M,0)[ns-1]/nn;qm=np.cumsum(Q)[ns-1,None]/nn
  vm=(np.cumsum(M*M,0)[ns-1]-nn*mm*mm)/(nn-1);vq=(np.cumsum(Q*Q)[ns-1,None]-nn*qm*qm)/(nn-1)
  cv=(np.cumsum(M*Q[:,None],0)[ns-1]-nn*mm*qm)/(nn-1)
  ymean=mm[:,:,None]+qm[:,:,None]/(4*eta);yv=vm[:,:,None]+vq[:,:,None]/(16*eta**2)+cv[:,:,None]/(2*eta)
  generic=(ymean+rho*eta+radius(yv,nn[:,:,None],1+np.maximum(1,1/(2*eta)),delta/31)).min(2)
  shared=mm+radius(vm,nn,2,delta/2)+np.sqrt(rho*np.minimum(1,kl_upper(qm,nn,2,delta/2)))
  for name,u in [('generic',generic),('shared',shared)]:
   cnt.setdefault(name,np.zeros_like(u,dtype=int));cnt[name]+=u<0;mean.setdefault(name,np.zeros_like(u));mean[name]+=u;sq.setdefault(name,np.zeros_like(u));sq[name]+=u*u
  if (rep+1)%400==0:print(phase,rep+1,round(time.time()-st,2),flush=True)
 arrays=dict(N=ns,reps=reps)
 for name in cnt:arrays[name+'_count']=cnt[name];arrays[name+'_mean']=mean[name]/reps;arrays[name+'_sd']=np.sqrt(np.maximum(0,(sq[name]-mean[name]**2/reps)/(reps-1)))
 np.savez_compressed(R/f'results/stress_{phase}.npz',**arrays)
 if phase=='plan':
  lock=[]
  for name in cnt:
   for sign in range(3):
    for target in [.8,.9]:
     pp=cnt[name][:,sign]/reps;hits=np.where(pp>=target)[0];center=int(ns[hits[0]]) if len(hits) else None
     lock.append(dict(method=name,sign=sign-1,target=target,center=center,region=[max(int(ns[0]),center-256),min(int(ns[-1]),center+256)] if center else None))
  (R/'stress_lock.json').write_text(json.dumps(lock,indent=2));(R/'stress_lock.sha256').write_text(hashlib.sha256((R/'stress_lock.json').read_bytes()).hexdigest()+'\n')
 print('done',phase,time.time()-st,flush=True)
if __name__=='__main__':run(sys.argv[1])
