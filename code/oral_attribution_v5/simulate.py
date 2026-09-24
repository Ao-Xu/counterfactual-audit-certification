import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,hashlib,time,sys
from pathlib import Path
import numpy as np
from oral_attribution_v5.methods import component,generic_fixed,radius
from oral_sampled_v3.simulate import coeffs
from oral_efficiency_v2.run import kl_upper
R=Path(__file__).resolve().parent;P=json.loads((R/'frozen_manifest.json').read_text());assert hashlib.sha256((R/'frozen_manifest.json').read_bytes()).hexdigest()==(R/'frozen_manifest.sha256').read_text().strip()
G=[dict(correlation=g['r'],anchor_scale=g['a'],sensitivity=g['b'],heterogeneity=g['h'],support=64) for g in P['geometries']]
gm,qc,fc,cc,offset=coeffs(G);ks=np.tile(P['candidate_multipliers'],len(G));fq=qc*(.5/ks[:,None])**2;cq=qc*((ks-.5)/ks)[:,None]**2
METHODS=['separate_split','separate_shared','paired_split','paired_shared','A_moment_current','B_generic_grid','C_generic_fixed','D_generic_fixed_matched','generic_fixed_Hledger','generic_fixed_loose_range','linear_marginal_EB','linear_marginal_KL','generic_fixed_Hoeffding']

def stats(F,sizes):
 ids=np.array(sizes)-1;n=np.array(sizes)[:,None];s=np.cumsum(F,axis=0)[ids];xx=np.cumsum(F[:,:,None]*F[:,None,:],axis=0)[ids];m=s/n
 return m,(xx-n[:,:,None]*m[:,:,None]*m[:,None,:])/(n[:,:,None]-1)
def project(c,cov):return np.einsum('gi,nij,gj->ng',c,cov,c,optimize=True)
def calculate(pm,pc,cm,ac,N,m):
 N=np.asarray(N)[:,None];m=np.asarray(m)[:,None];n=N-m;e=P['delta']/(2*(P['J']+1));rho=P['rho']
 am=(N*pm-m*cm)/n
 def parts(mean,cov,qmean,size,qsize,co,qco,span,off):
  return component(mean@co.T+off,project(co,cov),qmean@qco.T,size,qsize,span,e,rho)
 ps=parts(pm,pc,pm,N,N,gm,qc,2,offset);pt=parts(am,ac,cm,n,m,gm,qc,2,offset)
 fs=parts(pm,pc,pm,N,N,-fc,fq,1,-.5);cs=parts(pm,pc,pm,N,N,cc,cq,1,.5+offset)
 ft=parts(am,ac,cm,n,m,-fc,fq,1,-.5);ct=parts(am,ac,cm,n,m,cc,cq,1,.5+offset)
 out=dict(paired_shared=ps['upper'],paired_split=pt['upper'],separate_shared=fs['upper']+cs['upper'],separate_split=ft['upper']+ct['upper'])
 M=pm@gm.T+offset;Q=pm@qc.T;vm=project(gm,pc);vq=project(qc,pc);cv=np.einsum('gi,nij,gj->ng',gm,pc,qc,optimize=True)
 ea=P['delta']/(2*P['J']);eta=P['eta_fixed'];eh=P['delta']/(P['J']*31)
 vu=np.minimum(1,kl_upper(Q,N,2,ea));mr=radius(vm,N,2,ea)
 out['A_moment_current']=M+mr+np.sqrt(rho*vu)
 et=np.asarray(P['eta_grid']);ym=M[:,:,None]+Q[:,:,None]/(4*et);yv=vm[:,:,None]+vq[:,:,None]/(16*et**2)+cv[:,:,None]/(2*et)
 out['B_generic_grid']=(ym+rho*et+radius(yv,N[:,:,None],1+np.maximum(1,1/(2*et)),eh)).min(2)
 for key,ee in [('C_generic_fixed',P['delta']/P['J']),('D_generic_fixed_matched',ea),('generic_fixed_Hledger',eh)]:out[key]=generic_fixed(M,vm,vq,cv,Q,N,eta,ee,rho)
 out['generic_fixed_loose_range']=generic_fixed(M,vm,vq,cv,Q,N,eta,ea,rho,loose=True)
 out['generic_fixed_Hoeffding']=generic_fixed(M,vm,vq,cv,Q,N,eta,ea,rho,hoeffding=True)
 out['linear_marginal_EB']=M+mr+rho*eta+(Q+radius(vq,N,2,ea))/(4*eta)
 out['linear_marginal_KL']=M+mr+rho*eta+vu/(4*eta)
 pieces={}
 for name,a,b in [('paired_shared',ps,None),('paired_split',pt,None),('separate_shared',fs,cs),('separate_split',ft,ct)]:
  pieces[name]={k:a[k]+(b[k] if b else 0) for k in ['mean_radius','sensitivity_radius']}
 return out,pieces

def run(phase='plan'):
 pilot=phase=='pilot';N=np.array([P['pilot']['N']] if pilot else P['N_grid']);m=(N*P['split_fraction']).astype(int);maxN=max(N)
 reps=P['pilot']['replications'] if pilot else P['planning_reps' if phase=='plan' else 'confirmation_reps'];seed=P['pilot']['seed'] if pilot else P['planning_seed' if phase=='plan' else 'confirmation_seed'];rng=np.random.default_rng(seed)
 if phase=='confirm':assert (R/'threshold_lock.json').exists()
 cnt={};false={};sums={};squares={};pcmp={};st=time.time()
 for rep in range(reps):
  A=rng.choice([-1.,1.],maxN);W=rng.choice([-1.,1.],maxN);H=rng.choice([-1.,1.],maxN)
  support=np.linspace(-1,1,64);support/=support.std();s=support[rng.integers(64,size=(maxN,2))];sm=s.mean(1);q=(s[:,0]-s[:,1])**2/2
  F=np.column_stack([A,W,sm,H*sm,q,H*q]);pm,pc=stats(F,N);cm,ccov=stats(F,m);n=N-m
  # Recover exact suffix covariance; the split roles are disjoint.
  xx=(N-1)[:,None,None]*pc+N[:,None,None]*pm[:,:,None]*pm[:,None,:]-(m-1)[:,None,None]*ccov-m[:,None,None]*cm[:,:,None]*cm[:,None,:]
  am=(N[:,None]*pm-m[:,None]*cm)/n[:,None];ac=(xx-n[:,None,None]*am[:,:,None]*am[:,None,:])/(n-1)[:,None,None]
  out,parts=calculate(pm,pc,cm,ac,N,m)
  if pilot:out={k:v for k,v in out.items() if k in METHODS[:4]}
  for name,u in out.items():
   u=u.reshape(len(N),8,3);cnt.setdefault(name,np.zeros_like(u,dtype=int));cnt[name]+=u<0
   false.setdefault(name,np.zeros((len(N),8),dtype=int));false[name]+=np.any(u+P['margin']<0,axis=2)
   sums.setdefault(name,np.zeros_like(u));sums[name]+=u;squares.setdefault(name,np.zeros_like(u));squares[name]+=u*u
  for name,parts0 in parts.items():
   for term,values in parts0.items():
    key=name+'_'+term;pcmp.setdefault(key,np.zeros_like(values));pcmp[key]+=values
  if (rep+1)%32==0:
   (R/'results/progress.json').write_text(json.dumps(dict(phase=phase,rep=rep+1,total=reps,seconds=time.time()-st)));print(phase,rep+1,round(time.time()-st,1),flush=True)
 arrays=dict(N=N,reps=reps)
 for name in cnt:arrays.update({name+'_count':cnt[name],name+'_false':false[name],name+'_mean':sums[name]/reps,name+'_sd':np.sqrt(np.maximum(0,(squares[name]-sums[name]**2/reps)/(reps-1)))})
 for k,v in pcmp.items():arrays[k]=v.reshape(len(N),8,3)/reps
 np.savez_compressed(R/f'results/{phase}.npz',**arrays)
 if phase=='plan':
  locks=[]
  for name in cnt:
   for g in range(8):
    pp=cnt[name][:,g,1]/reps
    for target in [.8,.9]:
     hits=np.where(pp>=target)[0];center=int(N[hits[0]]) if len(hits) else None
     mid=np.where((pp>=.5)&(pp<=.99))[0]
     low=max(int(N[0]),int(N[mid[0]])-512) if len(mid) else int(N[0]);high=min(int(N[-1]),int(N[mid[-1]])+512) if len(mid) else int(N[-1])
     locks.append(dict(method=name,geometry=g,target=target,planning_crossing=center,region=[low,high]))
  f=R/'threshold_lock.json';f.write_text(json.dumps(locks,indent=2));(R/'threshold_lock.sha256').write_text(hashlib.sha256(f.read_bytes()).hexdigest()+'\n')
 print('completed',phase,round(time.time()-st,2),flush=True)
if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'plan')
