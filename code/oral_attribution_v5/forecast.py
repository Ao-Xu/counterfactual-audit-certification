"""Oracle moment forecast, fixed formulas, no simulated outcomes as inputs.
Not a data-based certificate, power guarantee, or minimax lower bound.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from oral_attribution_v5.simulate import P,G,gm,qc,fc,cc,fq,cq,offset,calculate
from oral_efficiency_v2.run import kl_upper
R=Path(__file__).resolve().parent
SUP=np.linspace(-1,1,64);SUP/=SUP.std();fourth=np.mean(SUP**4);basecov=np.diag([1,1,.5,.5,(fourth+1)/2,1+(fourth+1)/2]);base=np.array([0,0,0,0,1.,0])
N=np.arange(512,16385,128);m=N//4;nn=N[:,None];mm=m[:,None];pop=np.tile(base,(len(N),1));cov=np.tile(basecov,(len(N),1,1));out,_=calculate(pop,cov,pop,cov,N,m)
e=P['delta']/(2*(P['J']+1));rho=P['rho'];L=np.log(1/e)
def derivative(q,span,sizes):
 cap=span*span/2;eps=1e-7
 def f(x):return np.sqrt(rho*np.minimum(span*span/4,kl_upper(x,sizes,cap,e)))
 return (f(q+eps)-f(q-eps))/(2*eps)
def variance(co):return np.einsum('ngi,ij,ngj->ng',co,basecov,co)
rows=[]
for method in ['separate_split','separate_shared','paired_split','paired_shared']:
 split=method.endswith('split');separate=method.startswith('separate');sz=mm if split else nn;vn=nn-mm if split else nn
 if separate:
  grad=derivative(base@fq.T,1,sz)[:,:,None]*fq+derivative(base@cq.T,1,sz)[:,:,None]*cq
 else:grad=derivative(base@qc.T,2,sz)[:,:,None]*qc
 noise=variance(np.broadcast_to(gm,grad.shape))/vn+variance(grad)/sz if split else variance(grad+gm)/nn
 for g in range(8):
  for target in [.8,.9]:
   upper=out[method][:,g*3+1]+norm.ppf(target)*np.sqrt(noise[:,g*3+1]);hits=np.where(upper<0)[0]
   rows.append(dict(geometry=g,method=method,target=target,oracle_forecast=int(N[hits[0]]) if len(hits) else None))
(R/'results/oracle_forecast.json').write_text(json.dumps(dict(scope='Population-moment plug-in endpoint plus Gaussian influence quantile. No outcome fitting; not a finite-sample power guarantee or a lower bound.',rows=rows),indent=2))
# Rigorous sufficient shared budgets, common conservative power ledger.
def E(sig,R,N,e,bet):
 t=np.log(2/e);u=np.log(1/bet)
 return sig*(np.sqrt(2*u/N)+np.sqrt(2*t/N))+R*(2*u/(3*N)+2*np.sqrt(t*u)/np.sqrt(N*(N-1))+7*t/(3*(N-1)))
def AQ(V,Omega,c,N,e,bet):
 u=np.log(1/bet);L=np.log(1/e)
 if V==0:return np.sqrt(c*(-np.expm1(-L/N)))
 uniform=np.sqrt(c/N)*(np.sqrt(L)+np.sqrt(u))
 adaptive=np.sqrt(V+np.sqrt(2*Omega*u/N)+2*c*u/(3*N))-np.sqrt(V)+np.sqrt(c*L/N)
 return np.minimum(uniform,adaptive)
def vv(co):return float(co@basecov@co)
def solve(f):
 lo=2;hi=10000000
 if f(hi)>=.025:return None
 while hi-lo>1:
  mid=(hi+lo)//2
  if f(mid)<.025:hi=mid
  else:lo=mid
 return hi
bounds=[]
for g in range(8):
 j=g*3+1
 for target in [.8,.9]:
  bet=(1-target)/(6*(P['J']+1))
  for method in ['paired_shared','separate_shared']:
   pairs=[(gm[j],qc[j],2)] if method.startswith('paired') else [(fc[j],fq[j],1),(cc[j],cq[j],1)]
   def width(n):return sum(E(np.sqrt(vv(mc)),span,n,e,bet)+np.sqrt(rho)*AQ(float(base@qco),vv(qco),span*span/2,n,e,bet) for mc,qco,span in pairs)
   bounds.append(dict(geometry=g,method=method,target=target,sufficient_N=solve(width)))
(R/'results/sufficient_budgets.json').write_text(json.dumps(bounds,indent=2))
