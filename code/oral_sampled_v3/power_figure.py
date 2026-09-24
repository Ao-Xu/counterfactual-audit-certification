import json,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;data=json.load(open(R/'results/plan.json'))['rows'];alloc=json.load(open(R/'results/allocation_summary.json'))['rows']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(8,5.2),sharex=True,sharey=True)
for ax,(a,r) in zip(axes.flat,itertools.product([.04,.12],[-.5,.98])):
 geo=dict(correlation=r,anchor_scale=a,sensitivity=.12,heterogeneity=.8,support=64)
 for method,col,name in zip(['paired_KL','separate_KL','generic_DRO_EB'],['#be4238','#467ba1','#488463'],['Paired KL','Separate KL','Generic DRO']):
  opt=next(x for x in alloc if x['geometry']==geo and x['method']==method and x['target']==.9 and x['cost']=='total_anchors');p=opt['planning'];frac=p['m']/p['total_anchors'];rr=[x for x in data if x['geometry']==geo and x['method']==method and x['candidate']==1 and x['K']==p['K'] and x['m']/x['total_anchors']==frac];rr.sort(key=lambda x:x['total_anchors']);x=[d['total_anchors'] for d in rr];ax.plot(x,[d['power'] for d in rr],'-o',color=col,ms=3,label=name);ax.fill_between(x,[d['ci'][0] for d in rr],[d['ci'][1] for d in rr],color=col,alpha=.12);c=opt['confirmation'];ax.plot(c['total_anchors'],c['power'],'s',color=col,ms=7,fillstyle='none')
 ax.axhline(.9,color='.65',lw=.7,ls=':');ax.axhline(.8,color='.65',lw=.7,ls=':');ax.set_xscale('log',base=2);ax.set_title(f'Anchor scale {a}; correlation {r}');ax.grid(alpha=.15);ax.set_ylim(-.03,1.05)
for ax in axes[-1]:ax.set_xlabel('Total independent anchors')
for ax in axes[:,0]:ax.set_ylabel('Certification power')
axes[0,0].legend(fontsize=8,loc='lower right');fig.tight_layout()
for d in [R/'results',R.parent/'overleaf/experiments/oral_sampled_v3']:
 for ext in ['png','pdf']:fig.savefig(d/f'finite_power.{ext}',dpi=200,bbox_inches='tight')
