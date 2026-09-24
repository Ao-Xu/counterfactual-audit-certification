import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,hashlib
from pathlib import Path
import numpy as np
from sampled_interventions.methods import eb
from shared_anchor.methods import shared_stats
R=Path(__file__).resolve().parent
P=json.loads((R/'frozen_manifest.json').read_text());C=P['stress'];rng=np.random.default_rng(C['seed'])
rho=C['rho'];K=C['K'];a=C['a'];b=C['b'];h=C['h'];p=C['skew_positive_probability'];eta=b/(2*np.sqrt(rho));etas=np.geomspace(.01,2,31)
S=np.array([-np.sqrt(p/(1-p)),np.sqrt((1-p)/p)]);pw=np.array([1-p,p]);skew=float(pw@S**3);fourth=float(pw@S**4)
v=b*b*np.array([1-h,1+h]);k3=float((v**1.5).mean()*skew);k4=float((v*v).mean()*fourth)
noise_cov=k3/K;offset=-C['margin']-np.sqrt(rho)*b
rows=[]
for sign in C['cov_signs']:
 t=sign*C['cov_coefficient']-noise_cov/(b*b*h);w=np.sqrt(a*a-t*t)
 vals=np.array([offset+t*H+w*W+np.sqrt(b*b*(1+h*H))*s for H in [-1,1] for W in [-1,1] for s in S])
 assert np.max(np.abs(vals))<.5 # factual=.5, candidate=.5+Z, bounded losses
 assert 1+np.sqrt(rho/(b*b))*(vals.max()-offset)>0
 # Worst density uses most negative residual, check all anchor states.
 assert 1+np.sqrt(rho/(b*b))*np.sqrt(v.max())*S.min()>0
 tau=a*a+b*b/K;omega=float(v.var()+(k4+(v*v).mean())/K);cov=t*b*b*h+noise_cov
 theo=tau+omega/(16*eta*eta)+cov/(2*eta)
 N=C['moment_anchors'];H=rng.choice([-1.,1.],N);W=rng.choice([-1.,1.],N);s=rng.choice(S,(N,K),p=pw)
 z=offset+t*H[:,None]+w*W[:,None]+np.sqrt(b*b*(1+h*H[:,None]))*s
 M=z.mean(1);Q=((z[:,::2]-z[:,1::2])**2).mean(1)/2;Y=M+Q/(4*eta)
 row=dict(sign=sign,t=t,W_coefficient=float(w),conditional_skew_term=noise_cov,anchor_covariance=t*b*b*h,cov_MQ=cov,empirical_cov_MQ=float(np.cov(M,Q,ddof=1)[0,1]),theory_var_Y=theo,empirical_var_Y=float(Y.var(ddof=1)),tau=tau,omega=omega,eta=eta,mean_loss=offset,true_robust_target=-C['margin'],loss_min=float(.5+vals.min()),loss_max=float(.5+vals.max()),power=[])
 row['variance_relative_error']=abs(row['empirical_var_Y']/theo-1)
 # Same MC seed reset per covariance setting supplies identical random blocks.
 rr=np.random.default_rng(C['seed']);counts={};widths={}
 for rep in range(C['replications']):
  N=max(C['anchors']);H=rr.choice([-1.,1.],N);W=rr.choice([-1.,1.],N);s=rr.choice(S,(N,K),p=pw)
  z=offset+t*H[:,None]+w*W[:,None]+np.sqrt(b*b*(1+h*H[:,None]))*s
  M=z.mean(1);Q=((z[:,::2]-z[:,1::2])**2).mean(1)/2
  for n in C['anchors']:
   mm=M[:n];qq=Q[:n];ym=mm.mean()+qq.mean()/(4*etas);yv=mm.var(ddof=1)+qq.var(ddof=1)/(16*etas**2)+np.cov(mm,qq,ddof=1)[0,1]/(2*etas)
   gu=float(np.min(eb(ym,yv,n,1+np.maximum(1,1/(2*etas)),C['delta']/31)+rho*etas))
   su=float(shared_stats(mm.mean(),mm.var(ddof=1),qq.mean(),n,rho,C['delta'])['upper'])
   for method,U in [('generic',gu),('shared',su)]:
    key=(method,n);counts[key]=counts.get(key,0)+int(U<0);widths[key]=widths.get(key,0)+U+C['margin']
 for (method,n),cnt in counts.items():row['power'].append(dict(method=method,N=n,power=cnt/C['replications'],mean_width=widths[(method,n)]/C['replications']))
 rows.append(row);print(sign,row['variance_relative_error'],flush=True)
(R/'results/covariance_stress.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256((R/'frozen_manifest.json').read_bytes()).hexdigest(),rows=rows),indent=2))
