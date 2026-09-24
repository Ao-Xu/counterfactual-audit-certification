import json,math,itertools
from pathlib import Path
import numpy as np
from scipy.stats import beta
from oral_attribution_v5.simulate import P,METHODS
R=Path(__file__).resolve().parent

def band(k,reps,alpha):
 k=np.asarray(k);lo=np.where(k==0,0,beta.ppf(alpha/2,k,reps-k+1));hi=np.where(k==reps,1,beta.ppf(1-alpha/2,k+1,reps-k));return lo,hi

def threshold(ns,k,reps,target,alpha):
 lo,hi=band(k,reps,alpha/len(ns));pp=k/reps
 def first(a):
  ids=np.where(a)[0];return int(ns[ids[0]]) if len(ids) else None
 return dict(estimate=first(pp>=target),lower=first(hi>=target),upper=first(lo>=target),target=target,band_alpha=alpha,replications=reps)

def ratio(x,y):
 return dict(estimate=x['estimate']/y['estimate'] if x['estimate'] and y['estimate'] else None,lower=x['lower']/y['upper'] if x['lower'] and y['upper'] else None,upper=x['upper']/y['lower'] if x['upper'] and y['lower'] else None)

def run():
 z=np.load(R/'results/confirm.npz');ns=z['N'];reps=int(z['reps']);rows=[];lookup={}
 for g in range(8):
  for meth in METHODS:
   k=z[meth+'_count'][:,g,1]
   for target in [.8,.9]:
    v=threshold(ns,k,reps,target,.05/len(METHODS));v.update(geometry=g,parameters=P['geometries'][g],method=meth,evaluations=2*v['estimate'] if v['estimate'] else None);rows.append(v);lookup[g,meth,target]=v
 factors=[]
 for g,target in itertools.product(range(8),[.8,.9]):
  ss,sh,ps,ph=[lookup[g,m,target] for m in METHODS[:4]]
  ratios={name:ratio(x,y) for name,x,y in [('pairing_split',ss,ps),('pairing_shared',sh,ph),('reuse_separate',ss,sh),('reuse_paired',ps,ph)]}
  ns0,nh,np0,nph=[v['estimate'] for v in [ss,sh,ps,ph]]
  log_inter=math.log(ns0*nph/(np0*nh));pair=.5*math.log(ns0*nh/(np0*nph));reuse=.5*math.log(ns0*np0/(nh*nph))
  interlo=math.log(ss['lower']*ph['lower']/(ps['upper']*sh['upper']));interhi=math.log(ss['upper']*ph['upper']/(ps['lower']*sh['lower']))
  factors.append(dict(geometry=g,target=target,ratios=ratios,log_interaction=log_inter,log_interaction_ci=[interlo,interhi],shapley_pairing_factor=math.exp(pair),shapley_reuse_factor=math.exp(reuse),combined_ratio=ns0/nph))
 s=np.load(R/'results/stress_confirm.npz');sr=[]
 for method,sign,target in itertools.product(['generic','shared'],range(3),[.8,.9]):
  v=threshold(s['N'],s[method+'_count'][:,sign],int(s['reps']),target,.05/6);v.update(method=method,sign=sign-1);sr.append(v)
 false=max(int(z[m+'_false'].max()) for m in METHODS)
 out=dict(thresholds=rows,factorial=factors,stress_thresholds=sr,max_family_null_acceptance=false/reps,band_scope='95% simultaneous over all 13 methods and all N separately within each geometry; stress simultaneous across 6 curves and all N; no assumption of monotonic power',no_confirmation_retuning=True)
 (R/'results/analysis.json').write_text(json.dumps(out,indent=2))
 for g in range(8):print(g,[(m,lookup[g,m,.9]['estimate'],[lookup[g,m,.9]['lower'],lookup[g,m,.9]['upper']]) for m in METHODS[:8]])
 print('STRESS',sr)
if __name__=='__main__':run()
