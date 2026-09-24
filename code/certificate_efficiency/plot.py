import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R.parent/'language_models/experiments/certificate_efficiency';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':9,'pdf.fonttype':42,'axes.spines.top':True,'axes.spines.right':True})
def save(fig,name):
 for ext in ['pdf','png']:
  p=R/'results'/f'{name}.{ext}';fig.savefig(p,dpi=190,bbox_inches='tight');shutil.copy2(p,O/p.name)
 plt.close(fig)
p=json.load(open(R/'results/primary_refined.json'));r=p['rows'];names=['gain030','gain025','gain020'];colors=['#bf3c32','#4077a5','#46815a'];labels=['Paired KL','Separate KL','Generic paired DRO']
fig,axes=plt.subplots(1,3,figsize=(8.3,2.8),sharey=True)
for ax,name in zip(axes,names):
 for method,col,label in zip(['paired_KL','separate_KL','generic_paired_DRO'],colors,labels):
  rows=[x for x in r if x['candidate']==name and x['method']==method];x=[v['anchors'] for v in rows];y=[v['power'] for v in rows]
  ax.plot(x,y,'o-',ms=2.8,markevery=max(1,len(rows)//20),lw=1.6,label=label,color=col);ax.fill_between(x,[v['ci'][0] for v in rows],[v['ci'][1] for v in rows],color=col,alpha=.13)
 ax.set_xscale('log');ax.set_ylim(-.02,1.05);ax.set_xlabel('Total anchors $m+n$');ax.grid(alpha=.18);ax.axhline(.9,color='.55',ls=':',lw=.7);ax.set_title('Reference gain '+{'gain030':'0.030','gain025':'0.025','gain020':'0.020'}[name]);ax.set_xticks([1e3,1e4,1e5]);ax.set_xticklabels(['$10^3$','$10^4$','$10^5$'])
axes[0].set_ylabel('Certification power');axes[1].legend(loc='lower right');fig.tight_layout();save(fig,'power_equal_envelopes')
a=json.load(open(R/'results/allocation.json'))['rows'];fig,axes=plt.subplots(2,3,figsize=(8.3,5.2),sharex=True,sharey=True)
for i,g in enumerate(['sibling_dominated','anchor_dominated']):
 for j,K in enumerate([2,8,32]):
  ax=axes[i,j];rows=[x for x in a if x['geometry']==g and x['K']==K];ms=sorted(set(x['m'] for x in rows));ns=sorted(set(x['n'] for x in rows));z=np.array([[next(x['power'] for x in rows if x['m']==m and x['n']==n) for n in ns] for m in ms]);im=ax.imshow(z,vmin=0,vmax=1,cmap='Blues',origin='lower',aspect='auto')
  for ii in range(4):
   for jj in range(4):ax.text(jj,ii,f'{z[ii,jj]:.2f}',ha='center',va='center',fontsize=11,color='white' if z[ii,jj]>.55 else '#182631')
  ax.set_xticks(range(4));ax.set_xticklabels(['256','1024','4096','16384']);ax.set_yticks(range(4));ax.set_yticklabels(['1024','4096','16384','65536']);ax.set_title(('Sibling dominated' if i==0 else 'Anchor dominated')+f', $K={K}$')
  if i==1:ax.set_xlabel('Audit $n$')
  if j==0:ax.set_ylabel('Calibration $m$')
fig.tight_layout();save(fig,'independent_allocation')
# Budget frontiers are descriptive maxima over the entire frozen grid, not audit selection rules.
fig,axes=plt.subplots(1,2,figsize=(7.6,2.8),sharey=True);front=[]
for ax,g in zip(axes,['sibling_dominated','anchor_dominated']):
 for budget,label,col in [('anchors','Independent anchors','#bf3c32'),('evaluations','Paired loss evaluations','#4077a5')]:
  rows=[x for x in a if x['geometry']==g];xs=sorted(set(x[budget] for x in rows));ys=[]
  for x in xs:
   best=max([v for v in rows if v[budget]<=x],key=lambda v:v['power']);ys.append(best['power'])
  ax.step(xs,ys,where='post',label=label,color=col)
  for target in [.8,.9]:
   feasible=[v for v in rows if v['ci'][0]>=target]
   best=min(feasible,key=lambda v:(v[budget],v['m'],v['K'])) if feasible else None
   front.append(dict(geometry=g,budget=budget,target=target,criterion='Wilson lower',best_grid_cell=best))
 ax.set_xscale('log');ax.set_ylim(-.02,1.05);ax.grid(alpha=.2);ax.set_title(g.replace('_',' ').capitalize());ax.set_xlabel('Budget');ax.legend(loc='lower right')
axes[0].set_ylabel('Best grid power at or below budget');fig.tight_layout();save(fig,'budget_frontier');(R/'results/allocation_frontier.json').write_text(json.dumps(front,indent=2))
d=json.load(open(R/'results/real_width_decomposition.json'))['summary'];fig,axes=plt.subplots(1,2,figsize=(8.8,2.9));labels=['HANS (4 seeds)','HANS (12 seeds)','QA'];xx=np.arange(3);bottom=np.zeros(3)
for key,label,col in [('oracle_variance_mean_radius','Mean radius with plug-in variance','#567fa0'),('variance_calibration_radius_excess','Extra from variance calibration','#98b5cb'),('nuisance_calibration_excess','Extra from sensitivity calibration','#c36b53')]:
 vals=np.array([d[t]['statistics'][key]['median'] for t in d]);axes[0].bar(xx,vals,bottom=bottom,label=label,color=col);bottom+=vals
axes[0].set_xticks(xx);axes[0].set_xticklabels(labels,fontsize=8);axes[0].set_ylabel('Median component (loss units)');axes[0].legend(fontsize=7);axes[0].grid(axis='y',alpha=.15)
for k,label,col in [('planning_full','Full KL','#bf3c32'),('planning_oracle_V','Known sensitivity','#46815a'),('planning_both_oracle','Known sensitivity + variance','#4077a5')]:
 vals=[d[t]['statistics'][k]['median'] for t in d];axes[1].plot(xx,vals,'o-',label=label,color=col)
axes[1].set_yscale('log');axes[1].set_xticks(xx);axes[1].set_xticklabels(labels,fontsize=8);axes[1].set_ylabel('Planned anchors per block');axes[1].legend(fontsize=7);axes[1].grid(alpha=.15);fig.tight_layout();save(fig,'real_width_components')
print('Figures saved',O)
