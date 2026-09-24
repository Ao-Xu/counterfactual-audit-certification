"""Evidence-only vector panels from reproduced frozen controlled experiments."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'code/oral_planner_v6/results/summary.json'
thresholds = ROOT/'code/oral_attribution_v5/results/dense_thresholds.csv'
S = json.loads(source.read_text(encoding='utf-8'))
with thresholds.open(encoding='utf-8', newline='') as stream:
    T = [r for r in csv.DictReader(stream) if float(r['target']) == .9]
assert len(S) == 6
plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman', 'DejaVu Serif'],
 'font.size':8, 'axes.labelsize':8, 'xtick.labelsize':7.5, 'ytick.labelsize':7.5,
 'axes.titlesize':9, 'legend.fontsize':7.3, 'axes.spines.top':False,
 'axes.spines.right':False, 'axes.linewidth':.6, 'pdf.fonttype':42,
 'ps.fonttype':42, 'svg.fonttype':'none', 'mathtext.fontset':'stix'})
blue='#0072B2'; orange='#D55E00'; green='#009E73'; gray='#757575'
fig, axs = plt.subplots(2,2,figsize=(5.5,4.75))
fig.subplots_adjust(left=.105,right=.986,top=.935,bottom=.18,wspace=.37,hspace=.91)
for ax in axs.flat:
    ax.grid(axis='y',color='#dddddd',linewidth=.5,zorder=0)
    ax.set_axisbelow(True)

ax=axs[0,0]
methods=[('separate_split','Separate, split',gray,'s',-.24),
 ('separate_shared','Separate, shared',gray,'o',-.08),
 ('paired_split','Paired, split',blue,'s',.08),
 ('paired_shared','Paired, shared',blue,'o',.24)]
for key,label,col,mark,dx in methods:
    rows=sorted([r for r in T if r['method']==key],key=lambda x:int(x['geometry']))
    assert len(rows)==8
    v=np.array([float(r['estimate']) for r in rows])/1000
    lo=np.array([float(r['lower']) for r in rows])/1000
    hi=np.array([float(r['upper']) for r in rows])/1000
    ax.errorbar(np.arange(8)+dx,v,yerr=[v-lo,hi-v],fmt=mark,ms=3.2,
        mfc='white' if mark=='s' else col,mec=col,color=col,lw=.7,capsize=1.7,label=label)
ax.set(ylim=(0,15.8),xlim=(-.55,7.55),ylabel='Audit anchors (thousands)',xlabel='Matched design cell',
       title='(a) Pairing and reuse both save anchors')
ax.set_xticks(range(8)); ax.set_yticks([0,5,10,15])
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.35),ncol=2,frameon=False,
          columnspacing=.8,handletextpad=.3)

x=np.arange(6); labs=[r['law']+'\n'+str(int(r['target']*100))+'%' for r in S]
ax=axs[0,1]
for key,label,col,mark,dx in [('valid','Pilot guarantee',blue,'o',-.12),('plugin','Plug-in',orange,'s',.12)]:
    v=np.array([r[key]['median_N']/r['empirical_N'] for r in S])
    lo=np.array([r[key]['q25_N']/r['empirical_N'] for r in S])
    hi=np.array([r[key]['q75_N']/r['empirical_N'] for r in S])
    ax.errorbar(x+dx,v,yerr=[v-lo,hi-v],fmt=mark,ms=3.5,color=col,capsize=2,lw=.8,label=label)
ax.axhline(1,color=gray,ls='--',lw=.7,zorder=0)
ax.set(yscale='log',ylim=(.55,70),xlim=(-.5,5.5),ylabel='Budget / empirical threshold',
       title='(b) Feasible plans remain conservative')
ax.set_xticks(x); ax.set_xticklabels(labs);ax.set_yticks([1,3,10,30]);ax.set_yticklabels(['1','3','10','30'])
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.35),ncol=2,frameon=False,columnspacing=.6,handletextpad=.3)

ax=axs[1,0]
for key,label,col,dx in [('valid','Pilot: end-to-end',blue,-.13),('plugin','Plug-in',orange,.13)]:
    ax.plot(x+dx,[r[key]['unconditional_power'] for r in S],linestyle='none',marker='o' if key=='valid' else 's',
            ms=4,color=col,label=label)
ax.plot(x-.13,[r['valid']['conditional_power'] for r in S],linestyle='none',marker='o',ms=5,
        markerfacecolor='none',markeredgecolor=blue,label='Pilot: feasible only')
for i,r in enumerate(S):
    ax.plot([i-.13,i-.13],[r['valid']['unconditional_power'],1],color=blue,lw=.65,alpha=.4)
    ax.plot([i-.31,i+.31],[r['target'],r['target']],color='#333333',lw=.8)
    ax.text(i,1.095,str(r['valid']['feasible'])+'/24',ha='center',va='center',fontsize=7.2,color=blue)
ax.set(ylim=(0,1.16),xlim=(-.5,5.5),ylabel='Certification frequency',title='(c) Feasibility limits end-to-end power')
ax.set_xticks(x);ax.set_xticklabels(labs);ax.set_yticks([0,.25,.5,.75,1]);ax.set_yticklabels(['0','.25','.50','.75','1'])
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=1,frameon=False,handletextpad=.4,labelspacing=.25)

ax=axs[1,1]
for key,label,col,mark,dx in [('valid','Pilot guarantee',blue,'o',-.12),('plugin','Plug-in',orange,'s',.12)]:
    ax.plot(x+dx,[r[key]['mean_total_anchors']/1000 for r in S],linestyle='none',marker=mark,
            ms=4,color=col,label=label)
ax.axhline(2.048,color=gray,ls=':',lw=.9)
ax.text(5.35,8.0,'Pilot: 2,048 anchors',ha='right',fontsize=7,color=gray)
ax.set(ylim=(0,55),xlim=(-.5,5.5),ylabel='Mean total anchors (thousands)',title='(d) Charge the pilot, including aborts')
ax.set_xticks(x);ax.set_xticklabels(labs);ax.set_yticks([0,10,20,30,40,50])
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=2,frameon=False,columnspacing=.6,handletextpad=.3)

dest=ROOT/'figures';dest.mkdir(exist_ok=True)
for suffix in ['pdf','svg','png']:
    fig.savefig(str(dest/('audit_evidence.'+suffix)),dpi=220)
plt.close(fig)
manifest={'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,thresholds]},
 'panels':{'a':'90% first-grid crossing; simultaneous per-geometry binomial bands; K=2, 1600 replications',
           'b':'Median and IQR among feasible pilots only; normalizer is point empirical menu threshold',
           'c':'Descriptive averages over 24 pilots, each feasible plan has 256 fresh final replications; no pooled-binomial CI',
           'd':'Mean pilot + final anchors over all 24 pilots; infeasible plans pay 2048 pilot anchors'}}
(ROOT/'research/main_figure_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Rendered audit_evidence.pdf/svg/png from frozen-run results.')
