from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
ROOT=Path(__file__).resolve().parent;out=ROOT/'results/hans_fresh';s=json.loads((out/'summary.json').read_text());rows=s['rows'];qs=np.array([.65,.65-np.sqrt(.05*.65*.35),.55,.60,.70,.75,.65+np.sqrt(.05*.65*.35)]);order=np.argsort(qs)
policies=['P50_paired60','P50_paired','P995','cue70','cue90','cue100'];colors=['#8C564B','#D62728','#9467BD','#1F77B4','#FF7F0E','#2CA02C'];names=['Paired P50 / 60','Paired P50 / 120','P995','Cue70','Cue90','Cue100']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':12,'axes.linewidth':.8,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(12.5,4.9));a,b=axs
for p,c,name in zip(policies,colors,names):
 rr=sorted([r for r in rows if r['policy']==p],key=lambda r:r['seed']);x=np.array([r['reference_delta'] for r in rr]);y=np.array([r['V'] for r in rr]);z=np.array([r['shift_deltas'] for r in rr])[:,order]
 a.scatter(x,y,s=32,color=c,alpha=.45);a.scatter(x.mean(),y.mean(),marker='D',s=65,color=c,edgecolor='white',linewidth=.7,label=name,zorder=4)
 for line in z:b.plot(qs[order],line,color=c,lw=.9,alpha=.22)
 b.plot(qs[order],z.mean(0),'-o',ms=4,lw=1.8,color=c,label=name)
a.set(xlabel='Reference Brier risk difference (policy − factual)',ylabel='Paired sensitivity');a.legend(fontsize=9,loc='upper left');a.set_title('(a) Reference gain and paired sensitivity',loc='left',fontsize=12,pad=10)
b.axhline(0,color='black',ls='--',lw=1);b.axvline(.65,color='gray',ls=':',lw=1);b.set(xlabel='Cue alignment probability q',ylabel='Held-out Brier risk difference (policy − factual)');b.set_xticks([qs.min(),.60,.65,.70,qs.max()],['0.543','0.60','0.65\nReference','0.70','0.757']);b.set_title('(b) Response to covered nuisance shifts',loc='left',fontsize=12,pad=10)
for ax in axs:ax.grid(alpha=.2,lw=.5);ax.set_axisbelow(True)
fig.tight_layout(rect=(0,.07,1,1));fig.text(.5,.012,'4 training-order seeds; fixed fresh HANS splits. Diamonds/bold lines: means; pale marks/lines: individual seeds. ρ = 0.05.',ha='center',fontsize=10)
fig.savefig(out/'two_panels.png',dpi=220,bbox_inches='tight');fig.savefig(out/'two_panels.pdf',bbox_inches='tight');plt.close(fig)
(out/'two_panels_metadata.json').write_text(json.dumps({'source':'summary.json','rho':.05,'q_sorted':qs[order].tolist(),'no_certificate_panel':True,'scope':'Empirical risk and sensitivity; point signs are not simultaneous confidence claims. All six policies and four seeds retained.'},indent=2))
print('Saved two_panels.png/pdf; all policies and seeds retained.')
