import math
import numpy as np
from common import *
from statistics_cf import upper_mean

def bounds(f,c,w):
 g={k:c[k]-f[k] for k in ['calibration','reference','test']};cm=g['calibration']@w;cv=((g['calibration']-cm[:,None])**2)@w;rm=g['reference']@w;t=math.log(6*48/.025)
 vf=((f['calibration']-(f['calibration']@w)[:,None])**2)@w;vc=((c['calibration']-(c['calibration']@w)[:,None])**2)@w;fu=upper_mean(vf,.25,t);cu=upper_mean(vc,.25,t);vu=min(upper_mean(cv,1,t),(math.sqrt(fu)+math.sqrt(cu))**2);su=upper_mean(cm**2,1,t);n=len(rm);gamma=2*t/(3*n)+math.sqrt(2*su*t/n+4*t*t/(9*n*n));d=float(rm.mean());rho=.05
 return dict(reference_delta=d,V=float(cv.mean()),Vupper=vu,gamma=gamma,U_pair=d+gamma+math.sqrt(rho*vu),U_separate=d+gamma+math.sqrt(rho)*(math.sqrt(fu)+math.sqrt(cu)),plugin_pair=d+math.sqrt(rho*float(cv.mean())),plugin_separate=d+math.sqrt(rho)*(math.sqrt(float(vf.mean()))+math.sqrt(float(vc.mean())))),g

def analyze(task,budget=None):
 isqa=task=='qa';jobs=read(ROOT/task/'jobs.json');jobs=[j for j in jobs if not isqa or j['budget']==budget];d=read(ROOT/task/'data.json');w=np.ones(4)/4 if isqa else np.array([.325,.325,.175,.175]);laws=[('reference',w)]
 if isqa:
  for style in range(4):
   for sign in [-1,1]:
    q=.25+sign*math.sqrt(.05*.25*.75);v=np.full(4,(1-q)/3);v[style]=q;laws.append((f'style{style}_{"up" if sign>0 else "down"}',v))
 else:
  for q in [.65-math.sqrt(.05*.65*.35),.55,.6,.7,.75,.65+math.sqrt(.05*.65*.35)]:laws.append((f'q{q:.3f}',np.array([q/2,q/2,(1-q)/2,(1-q)/2])))
 for _,v in laws:assert np.sum((v-w)**2/w)<=.05+1e-9
 rows=[];decisions=[]
 def load(j):
  key=f"b{budget}_{j['seed']}_{j['policy']}" if isqa else f"{j['seed']}_{j['policy']}";path=ROOT/task/'runs'/key;done=read(path/'done.json');assert done['predictions_sha']==sha(path/'predictions.npz');return dict(np.load(path/'predictions.npz'))
 for seed in sorted({j['seed'] for j in jobs}):
  sub=[j for j in jobs if j['seed']==seed];bj=next(j for j in sub if j['policy'] in ['baseline_0','factual']);f=load(bj);local=[]
  for j in sub:
   if j==bj:continue
   c=load(j);b,g=bounds(f,c,w);shift=[float((g['test']@v).mean()) for _,v in laws];b.update(policy=j['policy'],seed=seed,shift_deltas=shift,worst_shift=max(shift),all_shifts_improved=max(shift)<0,certified=b['U_pair']<0)
   # This interval describes finite held-out anchors conditional on fixed models/catalog.
   b['reference_gap_normal95']=[b['reference_delta']-1.96*np.std(g['reference']@w,ddof=1)/math.sqrt(len(g['reference'])),b['reference_delta']+1.96*np.std(g['reference']@w,ddof=1)/math.sqrt(len(g['reference']))]
   if isqa:
    b['em_reference']=float((c['test_em']@w).mean());b['f1_reference']=float((c['test_f1']@w).mean());b['factual_em_reference']=float((f['test_em']@w).mean());b['factual_f1_reference']=float((f['test_f1']@w).mean());b['em_shift_deltas']=[float(((c['test_em']-f['test_em'])@v).mean()) for _,v in laws];b['f1_shift_deltas']=[float(((c['test_f1']-f['test_f1'])@v).mean()) for _,v in laws]
   rows.append(b);local.append(b)
  decision={'seed':seed}
  for key,metric in [('reference_only','reference_delta'),('paired','U_pair'),('separate','U_separate')]:
   chosen=min(local,key=lambda r:(r[metric],r['policy']));use=chosen[metric]<0;decision[key]={'selected':chosen['policy'] if use else 'factual','worst_shift_delta':chosen['worst_shift'] if use else 0,'empirical_harm':bool(use and chosen['worst_shift']>0)}
  decisions.append(decision)
 out=ROOT/'results'/(f'qa_{budget}' if isqa else 'hans_fresh');out.mkdir(parents=True,exist_ok=True);write(out/'summary.json',dict(task=task,budget=budget,rho=.05,family=48,delta=.025,rows=rows,decisions=decisions,shift_names=[n for n,_ in laws],scope='Frozen finite candidate audit. Test means diagnose signs, not proof of absence of harm. QA certificate is for gold-token Brier only. Context/structure catalog conditional; seeds are not independent data draws.'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,3,figsize=(15,4.5));pols=list(dict.fromkeys(r['policy'] for r in rows))
 for i,p in enumerate(pols):
  rr=[r for r in rows if r['policy']==p];col=f'C{i}';axs[0].scatter([r['reference_delta'] for r in rr],[r['V'] for r in rr],label=p,color=col);v=np.array([r['shift_deltas'] for r in rr]);axs[1].plot(v.mean(0),'-o',label=p,color=col)
  for z in v:axs[1].plot(z,alpha=.2,color=col)
  for k,r in enumerate(rr):axs[2].plot([i-.15,i+.15],[r['U_pair'],r['U_separate']],color='gray',alpha=.3);axs[2].scatter(i-.15,r['U_pair'],color='C0');axs[2].scatter(i+.15,r['U_separate'],color='C3')
 axs[0].set(xlabel='Reference Brier risk difference',ylabel='Paired sensitivity');axs[0].legend(fontsize=8);axs[1].set(xlabel='Predefined covered shift index',ylabel='Held-out Brier risk difference');axs[1].axhline(0,color='black',ls='--');axs[2].set_xticks(range(len(pols)),pols,rotation=30,ha='right',fontsize=8);axs[2].axhline(0,color='black',ls='--');axs[2].set_ylabel('Upper bound (blue paired, red separate)')
 for ax in axs:ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'three_panels.png',dpi=180);fig.savefig(out/'three_panels.pdf');plt.close(fig)
 print(task,budget,'negative',sum(r['certified'] for r in rows),'/',len(rows),'point harms',sum(r['worst_shift']>0 for r in rows),flush=True)
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--task',required=True);a.add_argument('--budget',type=int);v=a.parse_args();analyze(v.task,v.budget)
