import json,collections,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from certificate_efficiency.run import wilson
R=Path(__file__).resolve().parent;O=R/'results';T=R.parent/'language_models/experiments/shared_anchor';T.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
methods=['separate_KL','paired_KL','shared_moment','generic_DRO_EB','shared_KL_matched']
names=['Separate KL','Moment (split)','Moment (shared)','Sampled DRO','Matched KL (shared)'];labels=dict(zip(methods,names));colors=dict(zip(methods,['#777777','#4286ab','#c44036','#e69f00','#934db1']))
plan=json.loads((O/'plan.json').read_text())['rows'];confirm=json.loads((O/'confirm.json').read_text())['rows'];locks=json.loads((R/'allocation_lock.json').read_text())['allocations']
lookup={(tuple(r['key']),r['candidate']):r for r in confirm}
for a in locks:a['confirmation']=lookup.get((tuple(a['key']),1)) if a['key'] else None
summary={'allocations':locks,'method_labels':labels,'scope':'Planning-eligible grid minima; fixed-lock independent confirmation; no retuning.'}
A={(a['geometry_id'],a['method'],a['target'],a['cost']):a for a in locks}
def val(g,m,p=.9,c='total_anchors'):
 a=A[g,m,p,c];return a['planning'][c] if a['key'] else np.nan
ratios=[('Pairing','separate_KL','paired_KL'),('Reuse, matched operator','paired_KL','shared_KL_matched'),('Mean estimator + ledger','shared_KL_matched','shared_moment'),('Residual joint estimator','shared_moment','generic_DRO_EB')]
summary['ratios']={}
for p in [.8,.9]:
 for title,x,y in ratios:
  v=np.array([val(g,x,p)/val(g,y,p) for g in range(64)])
  summary['ratios'][f'{p}_{title}']=dict(min=float(v.min()),median=float(np.median(v)),max=float(v.max()),counts=dict(collections.Counter(v.tolist())))
summary['confirmation_failures']=[{k:a[k] for k in ['geometry_id','method','target','cost','confirmation']} for a in locks if not a['confirmation'] or a['confirmation']['ci'][0]<a['target']]
summary['cost_counts']={str(p):{m:dict(collections.Counter(val(g,m,p) for g in range(64))) for m in methods} for p in [.8,.9]}
summary['K_counts']=dict(collections.Counter(a['planning']['K'] for a in locks if a['planning']))
summary['false_acceptance_max']=max(r['family_false_acceptance'] for r in confirm)
summary['false_acceptance_CI_max']=max(r['family_false_ci'][1] for r in confirm)
summary['width_at_N8192_K2']={m:dict(median=float(np.median([r['mean_width'] for r in plan if r['candidate']==1 and r['method']==m and r['total_anchors']==8192 and r['K']==2 and (r['m']==2048 if m in ['paired_KL','separate_KL'] else True)]))) for m in methods}
(O/'summary.json').write_text(json.dumps(summary,indent=2))
def save(fig,name):
 fig.savefig(O/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(T/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,4,figsize=(6.5,1.85),sharey=True)
geo={r['geometry_id']:r['geometry'] for r in plan}
for ax,(title,x,y) in zip(axs,ratios):
 arr=np.zeros((2,4));text={}
 for i,a in enumerate([.04,.12]):
  for j,r in enumerate([-.5,0,.8,.98]):
   v=[val(g,x)/val(g,y) for g in range(64) if geo[g]['anchor_scale']==a and geo[g]['correlation']==r]
   arr[i,j]=np.median(v);text[i,j]=f'{np.median(v):g}' if min(v)==max(v) else f'{np.median(v):g}\n{min(v):g}–{max(v):g}'.replace('0.5',' .5'.strip())
 im=ax.imshow(arr,vmin=0,vmax=2,cmap='RdBu_r',aspect='auto')
 for (i,j),s in text.items():ax.text(j,i,s,ha='center',va='center',fontsize=7,color='white' if arr[i,j]>1.5 else 'black')
 ax.set(xticks=range(4),xticklabels=['−.5','0','.8','.98'],yticks=range(2),yticklabels=['.04','.12'],xlabel='Correlation',title=title.replace('Reuse, matched operator','Anchor reuse\n(matched)').replace('Mean estimator + ledger','Mean estimator\n+ ledger').replace('Residual joint estimator','Joint estimator'))
 ax.tick_params(labelsize=7);ax.title.set_fontsize(8);ax.xaxis.label.set_fontsize(8)
axs[0].set_ylabel('Anchor scale');fig.tight_layout();save(fig,'decomposition')
for cost in ['total_anchors','evaluations']:
 fig,axs=plt.subplots(1,4,figsize=(10,10),sharey=True)
 for ax,(title,x,y) in zip(axs,ratios):
  mat=np.array([[val(g,x,.8,cost)/val(g,y,.8,cost),val(g,x,.9,cost)/val(g,y,.9,cost)] for g in range(64)])
  ax.imshow(mat,vmin=0,vmax=2,cmap='RdBu_r',aspect='auto');ax.set(xticks=[0,1],xticklabels=['80%','90%'],title=title,yticks=range(64),yticklabels=[str(g) for g in range(64)])
  ax.tick_params(axis='y',labelsize=6)
  for g in range(64):
   for j in range(2):ax.text(j,g,f'{mat[g,j]:g}',ha='center',va='center',fontsize=6,color='white' if mat[g,j]>1.5 else 'black')
 axs[0].set_ylabel('Geometry index (full mapping in data)');fig.tight_layout();save(fig,'phase_'+cost)
fig,axs=plt.subplots(1,3,figsize=(10,2.8))
for ax,g in zip(axs,[0,16,28]):
 for m in methods[:4]:
  a=A[g,m,.9,'total_anchors'];pr=a['planning'];f=pr['m']/pr['total_anchors']
  rows=sorted([r for r in plan if r['geometry_id']==g and r['method']==m and r['candidate']==1 and r['K']==pr['K'] and r['m']/r['total_anchors']==f],key=lambda r:r['total_anchors'])
  xx=[r['total_anchors'] for r in rows];yy=[r['power'] for r in rows]
  ax.plot(xx,yy,color=colors[m],label=labels[m]);ax.fill_between(xx,[r['ci'][0] for r in rows],[r['ci'][1] for r in rows],color=colors[m],alpha=.1)
  conf=a['confirmation'];ax.plot(pr['total_anchors'],conf['power'],'s',mfc='white',mec=colors[m])
 ax.set(xscale='log',ylim=(-.02,1.04),xlabel='Independent anchors',title=f'r={geo[g]["correlation"]}, a={geo[g]["anchor_scale"]}')
 ax.axhline(.9,color='.7',ls=':',lw=.7)
axs[0].set_ylabel('Certification power');axs[-1].legend(fontsize=7,loc='lower right');fig.tight_layout();save(fig,'power')
rec=json.loads((O/'record_ablation.json').read_text());fig,ax=plt.subplots(figsize=(6.3,2.7))
for offset,key,name,color in [(-.2,'split_upper','Moment (split)','#4286ab'),(0,'shared_upper','Moment (shared)','#c44036'),(.2,'generic_upper','Finite-catalog DRO','#e69f00')]:
 ax.scatter(np.arange(4)+offset,[r[key] for r in rec['rows']],label=name,color=color,s=30)
ax.axhline(0,color='.3',lw=.8);ax.set(xticks=range(4),xticklabels=['RGF 5k','Rule 5k','RGF 15k','Rule 15k'],ylabel='Robust increase upper bound');ax.legend(fontsize=8);fig.tight_layout();save(fig,'record_ablation')
st=json.loads((O/'covariance_stress.json').read_text())['rows'];fig,axs=plt.subplots(1,3,figsize=(9.6,2.6))
xx=np.arange(3);axs[0].plot(xx,[r['theory_var_Y'] for r in st],color='#444444',label='Theory');axs[0].scatter(xx,[r['empirical_var_Y'] for r in st],facecolors='white',edgecolors='#c44036',label='Empirical');axs[0].set(ylabel=r'Variance of $Y_\eta$');axs[0].legend(fontsize=7)
for method,col in [('generic','#e69f00'),('shared','#c44036')]:
 for axis,key,N in [(axs[1],'mean_width',8192),(axs[2],'power',2048)]:
  vals=[next(v[key] for v in r['power'] if v['method']==method and v['N']==N) for r in st];axis.plot(xx,vals,'o-',color=col,label=method)
  if key=='power':
   ci=np.array([wilson(round(v*400),400) for v in vals]);axis.errorbar(xx,vals,yerr=np.array([np.array(vals)-ci[:,0],ci[:,1]-np.array(vals)]),fmt='none',color=col,capsize=2,lw=.7)
axs[1].set_ylabel('Mean width (N=8192)');axs[2].set_ylabel('Power (N=2048)');axs[1].legend(fontsize=7)
for ax in axs:ax.set(xticks=xx,xticklabels=['Negative','Zero','Positive'],xlabel=r'$\mathrm{Cov}(M,Q)$')
fig.tight_layout();save(fig,'covariance_stress')
# Compact reproducible tables.
lines=[r'\begin{table}[t]\centering\small',r'\begin{tabular}{lrr}\toprule Method & $N$ (cell count), $80\%$ & $N$ (cell count), $90\%$\\\midrule']
for m in methods[:4]:
 cells=[]
 for p in [.8,.9]:
  counts=collections.Counter(int(val(g,m,p)) for g in range(64));cells.append(', '.join(f'{n} ({c})' for n,c in sorted(counts.items())))
 lines.append(labels[m]+' & '+' & '.join(cells)+r'\\')
lines += [r'\bottomrule\end{tabular}',r'\caption{Planning-eligible minimum grid budgets across 64 cells. $KN$ costs, confirmation, null error, and width are reported in the appendix.}\label{tab:shared-budgets}\end{table}']
(T/'budget_table.tex').write_text('\n'.join(lines))
lines=[r'\begin{table}[h]\centering\small\begin{tabular}{lrrr}\toprule Policy & Split & Shared & Finite-catalog DRO\\\midrule']
for r in rec['rows']:lines.append(r['policy'].replace('_',r'\_')+' & '+ ' & '.join(f'{r[k]:+.6f}' for k in ['split_upper','shared_upper','generic_upper'])+r'\\')
lines +=[r'\bottomrule\end{tabular}\caption{Post-decision ReCoRD ablation, identical pre-test articles and method-wise family budget. Historical selection remains unchanged.}\end{table}']
(T/'record_table.tex').write_text('\n'.join(lines))
(T/'stress.tex').write_text(r'''Empirical variances are $.016326,.019012,.021518$, versus theoretical
$.016363,.018939,.021514$ for negative, zero, and positive covariance;
relative discrepancies are below $.4\%$. At $N=8192$, the generic mean
width rises from $.01048$ to $.01122$, while shared-moment width is
$.01283$--$.01275$--$.01276$. At $N=2048$, generic power is
$5.25\%,3.50\%,3.25\%$ (400 replications); these small differences are
not a claim of statistically separated powers. All three attain $80\%$
and $90\%$ power at the same next grid budget, $N=4096$.
The covariance identity and width response are supported, but this coarse
budget grid does not establish different minimum anchor counts.
\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{experiments/shared_anchor/covariance_stress.pdf}
\caption{All three frozen covariance settings. The conditional-skew
contribution $\E\kappa_3/K=.0007968$ is nonzero throughout. Fixed-$\eta$
variance checks and grid-optimized certification measure distinct quantities.}
\end{figure}
''')
print(json.dumps({k:v for k,v in summary.items() if k!='allocations' and k!='confirmation_failures'},indent=2));print('confirmation failures',len(summary['confirmation_failures']))
det=[r'\paragraph{Complete allocation and safety results.}']
fail=len(summary['confirmation_failures']);det.append(f'{fail} of the {len(locks)} method/geometry/target/cost locks fail their independent pointwise Wilson-lower power target. These locks are retained without retuning.')
det.append(r'Every observed family null-error frequency is zero at the confirmed allocations; the pointwise 95\% Wilson upper limit with 400 replications is $.00951$. This is a regression check, not proof or a simultaneous empirical bound over all cells.')
det += [r'\begin{table}[h]\centering\small\begin{tabular}{lrrr}\toprule Method & $KN$, $90\%$ (cells) & Median width & Null error\\\midrule']
for m in methods:
 cnt=collections.Counter(int(val(g,m,.9,'evaluations')) for g in range(64));s=', '.join(f'{n} ({c})' for n,c in sorted(cnt.items()))
 det.append(labels[m]+' & '+s+f" & {summary['width_at_N8192_K2'][m]['median']:.5f} & 0/400"+r'\\')
det +=[r'\bottomrule\end{tabular}\caption{Total paired loss observations at the locked $90\%$ budget; full-family individual model evaluations are $(J+1)KN$. Width is the median across geometries of planning mean $U-\Psi$ at common $N=8192,K=2$ and split fraction $1/4$ where needed. Null error is the per-allocation family event, not a pooled count treating geometries as independent.}\end{table}']
det.append('Every locked allocation uses $K=2$. Both independent-anchor and evaluation objectives are retained.')
det += [r'\begin{table}[h]\centering\small\begin{tabular}{lrr}\toprule Attribution ratio & $80\%$: min / median / max & $90\%$: min / median / max\\\midrule']
for title,_,_ in ratios:
 vals=[]
 for p in [.8,.9]:
  r=summary['ratios'][f'{p}_{title}'];vals.append(f"{r['min']:g} / {r['median']:g} / {r['max']:g}")
 det.append(title+' & '+' & '.join(vals)+r'\\')
det += [r'\bottomrule\end{tabular}\caption{Budget ratios above one favor the denominator method. The matched historical operator isolates data reuse; moving to the primary EB shared method also changes the mean estimator and ledger.}\end{table}']
for name,caption in [('phase_total_anchors','All 64 geometry ratios, optimizing independent anchors separately at each power target.'),('phase_evaluations','All 64 geometry ratios, optimizing paired loss evaluations.'),('power','Planning power curves and pointwise Wilson bands for three fixed cells (IDs 0, 16, 28). Open squares are independently confirmed locked choices; curves are descriptive because their allocation is selected on planning data.')]:
 det.extend([r'\begin{figure}[p]\centering',r'\includegraphics[width=.96\textwidth]{experiments/shared_anchor/'+name+'.pdf}',r'\caption{'+caption+r'}\end{figure}'])
(T/'details.tex').write_text('\n'.join(det))
# Human-readable local report with direct numerical decomposition.
md=['# Shared-anchor attribution results','', 'ReCoRD results below are a post-decision ablation; the original decision is unchanged.','', '## Frozen controlled study','',f'Independent confirmation failures: {fail}/{len(locks)} locks.','', '| Method | N at 80% (number of cells) | N at 90% (number of cells) |','|---|---|---|']
for m in methods:
 vals=[]
 for p in [.8,.9]:vals.append('; '.join(f'{int(n)} ({c})' for n,c in sorted(collections.Counter(val(g,m,p) for g in range(64)).items())))
 md.append('| '+labels[m]+' | '+' | '.join(vals)+' |')
md+=['','## Attribution','', 'The split/shared primary ratio includes changing the mean estimator. The matched operator control separates this from data reuse.','']
for title,x,y in ratios:
 r=summary['ratios'][f'0.9_{title}'];md.append(f"- {title}: {x}/{y} = {r['min']:g}–{r['max']:g} (median {r['median']:g}).")
md+=['','## ReCoRD pre-test ablation','','| Policy | Split | Shared | Finite-catalog DRO |','|---|---:|---:|---:|']
for r in rec['rows']:md.append('| '+r['policy']+' | '+' | '.join(f'{r[k]:+.6f}' for k in ['split_upper','shared_upper','generic_upper'])+' |')
md+=['','## Covariance stress','', 'All three settings are retained. Conditional skew term is nonzero. Theory/empirical variance discrepancy <0.4%; covariance changes widths but does not change the grid-resolved 80%/90% threshold (4096 anchors).','', '## Interpretation','', 'Original split-method failure is not an inherent inability of moment certification. Method rankings are setting- and estimator-specific. No sealed-test outcome enters the new ReCoRD endpoints. No rho, candidate or shift-class changes. All historical negative results are retained.']
(O/'REPORT.md').write_text('\n'.join(md))
