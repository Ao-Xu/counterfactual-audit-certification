import json,sys,itertools,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent));from oral_sampled_v3.simulate import GEOS
lock=json.load(open(R/'allocation_lock.json'))['allocations'];conf=json.load(open(R/'results/confirm.json'))['rows'];lookup={(tuple(x['key']),x['candidate']):x for x in conf};rows=[]
for a in lock:
 q=dict(a);q['geometry']=GEOS[a['geometry_id']];q['confirmation']=lookup.get((tuple(a['key']),1)) if a['key'] else None
 q['confirmed_point']=q['confirmation']['power']>=q['target'] if q['confirmation'] else False;q['confirmed_lower']=q['confirmation']['ci'][0]>=q['target'] if q['confirmation'] else False;rows.append(q)
summary={}
for cost,target in itertools.product(['total_anchors','evaluations'],[.8,.9]):
 x=[r for r in rows if r['cost']==cost and r['target']==target];by={(r['geometry_id'],r['method']):r for r in x};ratios=[];gr=[]
 for geo in range(64):
  p,s,g=[by[geo,m] for m in ['paired_KL','separate_KL','generic_DRO_EB']];ratios.append(s['planning'][cost]/p['planning'][cost]);gr.append(g['planning'][cost]/p['planning'][cost])
 summary[cost+'_'+str(target)]=dict(pairing_ratio_min=min(ratios),pairing_ratio_max=max(ratios),pairing_ratio_median=float(np.median(ratios)),generic_over_paired_min=min(gr),generic_over_paired_max=max(gr),generic_over_paired_median=float(np.median(gr)),generic_cheaper=sum(v<1 for v in gr),generic_tie=sum(v==1 for v in gr),generic_more_expensive=sum(v>1 for v in gr),confirmed_point={m:sum(r['confirmed_point'] for r in x if r['method']==m) for m in ['paired_KL','separate_KL','generic_DRO_EB']},confirmed_lower={m:sum(r['confirmed_lower'] for r in x if r['method']==m) for m in ['paired_KL','separate_KL','generic_DRO_EB']})
(R/'results/allocation_summary.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2));print(json.dumps(summary,indent=2))
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10,'pdf.fonttype':42})
O=R.parent/'overleaf/experiments/oral_sampled_v3';O.mkdir(parents=True,exist_ok=True)
def save(fig,name):
 for ext in ['png','pdf']:
  p=R/'results'/f'{name}.{ext}';fig.savefig(p,dpi=190,bbox_inches='tight');shutil.copy2(p,O/p.name)
 plt.close(fig)
for cost in ['total_anchors','evaluations']:
 fig,axes=plt.subplots(4,4,figsize=(8.0,7.9),sharex=True,sharey=True)
 for i,(b,h) in enumerate(itertools.product([.04,.12],[0,.8])):
  for j,(L,comparison) in enumerate(itertools.product([4,64],['separate','generic'])):
   ax=axes[i,j];z=np.empty((2,4));fail=[]
   for ai,a in enumerate([.04,.12]):
    for ri,corr in enumerate([-.5,0,.8,.98]):
     gid=GEOS.index(dict(correlation=corr,anchor_scale=a,sensitivity=b,heterogeneity=h,support=L));p=next(r for r in rows if r['geometry_id']==gid and r['method']=='paired_KL' and r['target']==.9 and r['cost']==cost);other=next(r for r in rows if r['geometry_id']==gid and r['method']==('separate_KL' if comparison=='separate' else 'generic_DRO_EB') and r['target']==.9 and r['cost']==cost);z[ai,ri]=other['planning'][cost]/p['planning'][cost]
     if not (p['confirmed_point'] and other['confirmed_point']):fail.append((ri,ai))
   ax.imshow(np.log2(z),vmin=-2,vmax=2,cmap='RdBu_r',origin='lower',aspect='auto')
   for ai in range(2):
    for ri in range(4):ax.text(ri,ai,f'{z[ai,ri]:g}',ha='center',va='center',color='white' if abs(np.log2(z[ai,ri]))>1 else 'black',fontsize=11)
   for ri,ai in fail:ax.plot(ri+.28,ai+.25,'kx',ms=5)
   ax.set_xticks(range(4));ax.set_xticklabels(['-.5','0','.8','.98']);ax.set_yticks([0,1]);ax.set_yticklabels(['.04','.12'])
   if i==0:ax.set_title(f'L={L}\n'+('separate / paired' if comparison=='separate' else 'generic / paired'),fontsize=10)
   if j==0:ax.set_ylabel(f'b={b}, h={h}\nAnchor scale')
   if i==3:ax.set_xlabel('Anchor correlation')
 fig.suptitle(('Anchor' if cost=='total_anchors' else 'Loss-evaluation')+' cost ratios at 90% power',fontsize=12);fig.tight_layout(rect=[0,0,1,.97]);save(fig,'phase_'+cost)
fig,axes=plt.subplots(1,2,figsize=(8.2,3.1));colors=['#be4238','#467ba1','#488463']
for ax,cost in zip(axes,['total_anchors','evaluations']):
 for j,m in enumerate(['paired_KL','separate_KL','generic_DRO_EB']):
  x=[r['planning'][cost] for r in rows if r['method']==m and r['target']==.9 and r['cost']==cost];ax.scatter(np.full(len(x),j)+np.linspace(-.14,.14,len(x)),x,color=colors[j],s=10,alpha=.4);ax.plot([j-.19,j+.19],[np.median(x)]*2,color=colors[j],lw=3)
 ax.set_xticks(range(3));ax.set_xticklabels(['Paired KL','Separate KL','Generic DRO']);ax.set_yscale('log',base=2);ax.set_ylabel('Anchors' if cost=='total_anchors' else 'Paired loss evaluations');ax.grid(axis='y',alpha=.2)
fig.tight_layout();save(fig,'optimized_budgets')
r=json.load(open(R/'results/retrospective.json'));fig,axes=plt.subplots(1,2,figsize=(8.5,3.2))
for task,col in zip(['hans','hans_extension','qa','qa_generation_f1','qa_generation_em'],['#467ba1','#a2bfd2','#be4238','#488463','#b79442']):
 data=[x for x in r['rows'] if x['task']==task];axes[0].scatter([x['paired_KL'] for x in data],[x['generic_all_available']['upper'] for x in data],s=13,label=task,color=col,alpha=.7);axes[1].scatter([x['catalog_robust_target'] for x in data],[x['generic_all_available']['upper'] for x in data],s=13,color=col,alpha=.7)
axes[0].plot([0,.25],[0,.25],color='.6',ls='--');axes[0].set_xlabel('Paired KL upper');axes[0].set_ylabel('Generic DRO upper (all anchors)');axes[0].legend(fontsize=7)
axes[1].axvline(0,color='.4',ls=':');axes[1].axhline(0,color='.4',ls=':');axes[1].set_xlabel('Exact inspected-catalog robust target');axes[1].set_ylabel('Generic DRO upper');fig.tight_layout();save(fig,'retrospective')
