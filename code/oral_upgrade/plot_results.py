import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';P=R.parent/'overleaf/experiments/oral_upgrade';P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,'pdf.fonttype':42})
METHODS=['paired','separate','bounded','paired_validation','reference_only']
LABELS=['Paired','Separate','Bounded','Paired validation','Reference only']
COLORS=['#C33D35','#416A98','#696969','#8B5C91','#CE8D35']
def save(fig,name):
    fig.savefig(O/(name+'.pdf'),bbox_inches='tight');fig.savefig(O/(name+'.png'),dpi=210,bbox_inches='tight');plt.close(fig);shutil.copy2(O/(name+'.pdf'),P/(name+'.pdf'))
def main():
    rows=json.loads((O/'controlled.json').read_text())['results'];sel=[r for r in rows if r['J']==4 and r['K']==4 and r['rho']==.05]
    fig,axs=plt.subplots(1,3,figsize=(7.5,3.1),layout='constrained')
    for method,label,col in zip(METHODS,LABELS,COLORS):
        rr=sorted([r for r in sel if r['method']==method],key=lambda r:r['n']);n=[r['n'] for r in rr]
        for ax,key in zip(axs,['safe_selection','harmful_selection','width']):ax.plot(n,[r[key] for r in rr],'-o',ms=3,label=label,color=col)
        axs[0].fill_between(n,[r['safe_selection_MC95'][0] for r in rr],[r['safe_selection_MC95'][1] for r in rr],color=col,alpha=.09)
    for ax,title in zip(axs,['Beneficial non-factual selection','Harmful selection','Mean upper-radius width']):
        ax.set(xlabel='Audit anchors n',ylabel=title);ax.set_xscale('log',base=2);ax.set_xticks([64,256,1024,4096,16384],['64','256','1,024','4,096','16,384']);ax.tick_params(axis='x',labelsize=8,rotation=35);ax.grid(alpha=.2)
    axs[0].set_ylim(-.04,1.04);axs[1].set_ylim(-.04,1.04);axs[0].legend(fontsize=8);save(fig,'selection_power')
    fig,axs=plt.subplots(1,2,figsize=(7,3.1),layout='constrained');ns=sorted({r['n'] for r in sel});ks=[2,4,16]
    heat=np.array([[next(r['safe_selection'] for r in rows if r['n']==n and r['K']==k and r['rho']==.05 and r['J']==4 and r['method']=='paired') for n in ns] for k in ks]);im=axs[0].imshow(heat,vmin=0,vmax=1,cmap='Blues',aspect='auto');axs[0].set_xticks(range(len(ns)),ns,rotation=30);axs[0].set_yticks(range(3),ks);axs[0].set(xlabel='Audit anchors n (plus m=n calibration)',ylabel='IID siblings K')
    for i in range(3):
        for j in range(len(ns)):axs[0].text(j,i,f'{heat[i,j]:.2f}',ha='center',va='center',color='white' if heat[i,j]>.5 else 'black')
    K=np.arange(1,65);n=1024
    for v,label in [(.02**2,'Stable'),(.45**2,'Sensitive')]:axs[1].plot(K,np.sqrt((.04**2+v/K)/n),label=label)
    axs[1].axhline(.04/np.sqrt(n),ls='--',color='black',label='Anchor floor');axs[1].set(xlabel='IID siblings K',ylabel='Population standard error of paired mean');axs[1].legend(fontsize=7);axs[1].grid(alpha=.2);save(fig,'anchor_floor')
    g=json.loads((O/'granularity.json').read_text())['rows'];fig,axs=plt.subplots(1,3,figsize=(7.5,3),layout='constrained')
    for noise,col,label in [('exact_loss_table','#416A98','Corrected: exact loss table'),('legacy_fixed_noise','#C33D35','Legacy: fixed observation noise')]:
        rr=[r for r in g if r['noise']==noise and r['scenario']==r['audit_class']=='task'];n=[r['n'] for r in rr]
        for ax,key in zip(axs,['bias','radius','coverage']):ax.plot(n,[r[key] for r in rr],'-o',color=col,label=label,ms=3)
    for ax,label in zip(axs,['Estimator bias','Bootstrap radius','Empirical upper coverage']):ax.set(xlabel='Anchors (256 = census)',ylabel=label);ax.grid(alpha=.2)
    axs[0].axhline(0,c='gray',ls=':');axs[2].set_ylim(0.85,1.015);axs[0].legend(fontsize=6.5);save(fig,'granularity_repair')
    q=json.loads((O/'qa_alignment.json').read_text())['audits'];fig,axs=plt.subplots(1,2,figsize=(7,3.3),layout='constrained')
    for a,col in zip(q,['#C33D35','#416A98','#696969']):
        rr=a['rows'];label=rr[0]['metric'];axs[0].plot(range(4),[r['Delta_hat'] for r in rr],'-o',label=label,color=col);axs[1].plot(range(4),[r['U_pair'] for r in rr],'-o',label=label,color=col)
    for ax,title in zip(axs,['Audit reference loss difference','Finite-sample paired upper bound']):
        ax.set_xticks(range(4),['RGF 5k','Rule 5k','RGF 15k','Rule 15k']);ax.tick_params(axis='x',rotation=20);ax.axhline(0,c='black',ls='--',lw=.8);ax.set_ylabel(title);ax.grid(alpha=.2)
    axs[0].legend(fontsize=7);save(fig,'qa_alignment')
    d=json.loads((O/'failure_decomposition.json').read_text())['rows'];fig,ax=plt.subplots(figsize=(6,3),layout='constrained')
    groups=[('hans','HANS old 4'),('hans_extension','HANS combined 12'),('qa','QA 6 families')]
    counts=[]
    for task,_ in groups:
        rr=[r for r in d if r['task']==task];counts.append([sum(r['diagnosis'].startswith(t) for r in rr) for t in ['A','B','certified']])
    bottom=np.zeros(3)
    for j,(label,col) in enumerate([('Empirical envelope nonnegative','#C33D35'),('Negative plug-in gap; interval too wide','#416A98'),('Certified','#76A477')]):
        vals=np.array(counts)[:,j];ax.bar(range(3),vals,bottom=bottom,label=label,color=col);bottom+=vals
    ax.set_xticks(range(3),[label for _,label in groups]);ax.set_ylabel('Candidate–seed records (cohorts overlap)');ax.legend(fontsize=7);save(fig,'failure_decomposition')
    # Generated table; no manually copied result numbers.
    lines=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule Cohort & Records & Envelope $\geq0$ & Width-limited & $U<0$\\\midrule']
    for (task,label),count in zip(groups,counts):lines.append(f'{label} & {sum(count)} & {count[0]} & {count[1]} & {count[2]}'+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Retrospective diagnosis at $\rho=.05$. Signs of the empirical plug-in envelope are not population classifications. Cohorts overlap; records must not be interpreted as independent task samples.}',r'\label{tab:failure-upgrade}',r'\end{table}']
    (P/'failure_table.tex').write_text('\n'.join(lines))
if __name__=='__main__':main()
