import json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oral_planner_v6.run import verify,read_lock
if not __debug__:raise SystemExit('Run without -O: frozen integrity assertions must stay enabled.')
verify();read_lock()
R=Path(__file__).resolve().parent;O=R/'results';D=O/'figures';D.mkdir(parents=True,exist_ok=True)
C=json.loads((R/'manifest.json').read_text());L=json.loads((O/'budget_lock.json').read_text());F=json.loads((O/'final.json').read_text())['rows'];B=json.loads((O/'benchmarks.json').read_text())['rows']
assert json.loads((O/'final.json').read_text())['budget_lock_sha']==hashlib.sha256((O/'budget_lock.json').read_bytes()).hexdigest(),'final/lock mismatch'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11,'axes.grid':True,'grid.alpha':.18,'axes.spines.top':True,'axes.spines.right':True,'pdf.fonttype':42})

def ci(s,n,e=.05):return [float(beta.ppf(e/2,s,n-s+1)) if s else 0.,float(beta.ppf(1-e/2,s+1,n-s)) if s<n else 1.]
summary=[]
for oracle in L['oracles']:
 law,p=oracle['law'],oracle['target']; br=[x for x in B if x['law']==law];ae=.05/len(br)
 threshold=lambda mode:min([x['N'] for x in br if (x['successes']/x['trials'] if mode=='point' else ci(x['successes'],x['trials'],ae)[0 if mode=='upper' else 1])>=p],default=None)
 row=dict(law=law,target=p,empirical_N=threshold('point'),empirical_threshold_CI=[threshold('lower'),threshold('upper')],
  oracle_forecast=oracle['plans']['oracle']['N'],sufficient_T4=oracle['plans']['sufficient']['N'],same_witness_oracle=oracle['plans']['valid']['N'],local_lower=oracle['lower'])
 for method in ['valid','plugin']:
  allrows=[x for x in F if x['law']==law and x['target']==p and x['planner']==method]; rr=[x for x in allrows if x['feasible']]
  ns=np.array([x['N'] for x in rr]);powers=np.array([x['successes']/x['trials'] for x in rr]);total=np.array([C['pilot_size']+x['N'] if x['feasible'] else C['pilot_size'] for x in allrows]);evals=np.array([C['pilot_size']*max(C['K_grid'])+(x['N']*x['K'] if x['feasible'] else 0) for x in allrows])
  row[method]=dict(feasible=len(rr),pilots=len(allrows),median_N=float(np.median(ns)),q25_N=float(np.quantile(ns,.25)),q75_N=float(np.quantile(ns,.75)),max_N=int(ns.max()),
   conditional_power=float(powers.mean()),unconditional_power=float(powers.sum()/len(allrows)),below_target_point=int((powers<p).sum()),
   confidently_below_target=sum(ci(x['successes'],x['trials'],.05/len(F))[1]<p for x in rr),
   min_simultaneous_power_lower=min(ci(x['successes'],x['trials'],.05/len(F))[0] for x in rr),
   mean_total_anchors=float(total.mean()),mean_total_evaluations=float(evals.mean()),median_ratio_empirical=float(np.median(ns)/row['empirical_N']),
   cost_over_nominal_local_lower=float(total.mean()/oracle['lower']['total_anchor_lower_bound']),
   comparison_scope='Nominal-power information benchmark, NOT a claimed efficiency guarantee when abstentions prevent target power.')
 summary.append(row)
(O/'summary.json').write_text(json.dumps(summary,indent=2))

fig,ax=plt.subplots(1,2,figsize=(8,3.8),gridspec_kw={'width_ratios':[1.3,1]})
x=np.arange(len(summary));labels=[f"{r['law']}\n{int(100*r['target'])}%" for r in summary]
for name,label,color,dx in [('valid','Valid pilot planner','#bd302b',-.16),('plugin','Naive plug-in','#3882a5',.0)]:
 vals=np.array([r[name]['median_N'] for r in summary]);lo=np.array([r[name]['q25_N'] for r in summary]);hi=np.array([r[name]['q75_N'] for r in summary])
 ax[0].errorbar(x+dx,vals,yerr=[vals-lo,hi-vals],fmt='o',ms=4,capsize=3,color=color,label=label)
for key,label,color,mark,dx in [('oracle_forecast','Gaussian oracle','#555555','x',.12),('sufficient_T4','Population T4','#946c9a','s',.23),('same_witness_oracle','Witness oracle','#4a8767','+',.32)]:ax[0].scatter(x+dx,[r[key] for r in summary],s=23,c=color,marker=mark,label=label)
ax[0].plot(x,[r['empirical_N'] for r in summary],':',color='#777777',label='Empirical menu threshold')
ax[0].set_yscale('log');ax[0].set_ylabel('Fresh audit anchors N');ax[0].set_xticks(x,labels);ax[0].legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.22));ax[0].set_ylim(1000,220000)
for i,r in enumerate(summary):ax[0].annotate(f"{r['valid']['feasible']}/24",(i-.16,r['valid']['q75_N']),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9,color='#bd302b')
ax[1].bar(x-.17,[r['valid']['unconditional_power'] for r in summary],.32,color='#bd302b',label='Valid (abstention = failure)')
ax[1].bar(x+.17,[r['plugin']['unconditional_power'] for r in summary],.32,color='#88b4ca',label='Naive plug-in')
ax[1].scatter(x,[r['target'] for r in summary],marker='_',s=180,color='black',label='Target')
ax[1].set_ylabel('End-to-end certification frequency');ax[1].set_xticks(x,labels);ax[1].set_ylim(0,1.15);ax[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.22))
fig.tight_layout();fig.savefig(D/'planning.pdf');fig.savefig(O/'planning.png',dpi=190);plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(10,3.2))
for j,(key,label) in enumerate([('mean_total_anchors','Total independent anchors (pilot + final)'),('mean_total_evaluations','Total paired-loss evaluations')]):
 for name,color,dx in [('valid','#bd302b',-.15),('plugin','#82afc7',.15)]:ax[j].bar(x+dx,[r[name][key] for r in summary],.3,color=color,label=name)
 ax[j].set_yscale('log');ax[j].set_ylabel(label);ax[j].set_xticks(x,labels);ax[j].legend(fontsize=8)
fig.tight_layout();fig.savefig(D/'costs.pdf');fig.savefig(O/'costs.png',dpi=190);plt.close(fig)

lines=['# Pilot -> plan -> certify: results','',
'The new pilot planner has a finite-sample conditional power guarantee. These experiments do **not** demonstrate near-oracle practical planning. Abstentions are counted as end-to-end failures. No rho, law, candidate or menu was changed after pilot/final outcomes.','',
'- 3 new frozen laws; 24 independent pilots per law/power, 2048 pilot anchors (256 design + 1792 estimation), Kmax=8.',
'- Every returned budget was locked before any fresh final outcome. Each lock is evaluated with 256 fresh, exact categorical IID blocks of exactly N anchors.',
'- Final target powers: 80%, 90%; safety delta=.025; planning delta=.05; rho=.05; one frozen CF candidate per law.',
'- Supplied support enclosures are valid known-law bounds, not learned min/max. Population probabilities are absent from planner inputs.',
'- The new BE witness improves on the population T4 sufficient cost; pilot uncertainty can more than erase that gain.','',
'| Law / target | empirical N [simultaneous grid CI] | Gaussian oracle | true T4 | witness oracle | valid median N (feasible/24) | plug-in median N | valid end-to-end | plug-in end-to-end |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['law']} / {r['target']:.0%} | {r['empirical_N']} {r['empirical_threshold_CI']} | {r['oracle_forecast']} | {r['sufficient_T4']} | {r['same_witness_oracle']} | {r['valid']['median_N']:g} ({r['valid']['feasible']}/24) | {r['plugin']['median_N']:g} | {r['valid']['unconditional_power']:.3f} | {r['plugin']['unconditional_power']:.3f} |")
valid_rows=[r for r in F if r['planner']=='valid'];feasible_rows=[r for r in valid_rows if r['feasible']]
perfect=sum(r['successes']==r['trials'] for r in feasible_rows)
lines+=['',f"{perfect}/{len(feasible_rows)} returned valid plans had {C['final_replications']}/{C['final_replications']} certificates below zero. The minimum simultaneous lower bound across these fixed locks is "+f"{min(r['valid']['min_simultaneous_power_lower'] for r in summary):.4f}." ,
f"{sum(r['plugin']['confidently_below_target'] for r in summary)}/{len(valid_rows)} naive plug-in locks have simultaneous 95% upper power limits below their target. This is stronger evidence of underplanning than a point estimate below target.",
f"{len(valid_rows)-len(feasible_rows)}/{len(valid_rows)} pilots returned no feasible budget on the frozen grid. Abstentions remain end-to-end failures.",'',
'## Local information comparison','',
'The two-point Bernoulli-anchor lower bound includes pilot information. The table uses nominal unconditional power (1-delta_plan)*p. If abstention prevents that goal, the ratio is a resource comparison against the nominal objective, not evidence of an efficient procedure that attained it. Pilot cost and K are fully charged.','',
'| Law / target | nominal local lower bound | mean total valid anchors | ratio | paired-loss evaluations |','|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['law']} / {r['target']:.0%} | {r['local_lower']['total_anchor_lower_bound']:.1f} | {r['valid']['mean_total_anchors']:.1f} | {r['valid']['cost_over_nominal_local_lower']:.1f} | {r['valid']['mean_total_evaluations']:.1f} |")
lines+=['','## Real-model scope','',
'Planner validation is controlled-only. The source archive does not include the historical HANS/SQuAD/ReCoRD prediction records. This run does not validate the manuscript historical real-model claims. The registered ReCoRD finite-catalog law differs from the IID-anchor and sampled-sibling planning experiment.','',
'The oracle convergence theorem concerns the same sufficient-witness oracle, not the Gaussian forecast, measured power threshold, or the information lower bound. The new results do not establish a near-optimal audit planner or an Oral-level empirical efficiency result.']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n')

# Compact generated table for the manuscript.
tex=['\\begin{tabular}{lrrrrr}\\toprule','Law / $p$ & Empirical & Oracle & T4 & Pilot $\\widehat N$ & Feasible\\\\\\midrule']
for r in summary:tex.append(f"{r['law']} / {r['target']:.1f} & {r['empirical_N']:,} & {r['oracle_forecast']:,} & {r['sufficient_T4']:,} & {r['valid']['median_N']:,.0f} & {r['valid']['feasible']}/24\\\\")
tex+=['\\bottomrule\\end{tabular}'];(D/'table.tex').write_text('\n'.join(tex)+'\n')
print('\n'.join(lines[:22]))
