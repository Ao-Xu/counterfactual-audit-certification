"""Analytic witnesses and post-result oracle stress tests of frozen Qwen heads.

No model fitting, test selection, or changes to confirmatory outputs.
The oracle uses test labels and is not a prospective OOD prediction.
"""
import pathlib,json,hashlib
import numpy as np
from scipy.special import ndtr,expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=pathlib.Path(__file__).resolve().parent
I=R/'confirmatory_v1'; O=R/'transfer_tightness'; O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
colors=['#0072B2','#D55E00','#009E73']; sigma=.5; beta=-.6
gaussian=[]
for lam in [.5,1.,2.]:
    for gamma in np.geomspace(1e-4,.8,60):
        den=sigma*np.sqrt(1+lam*lam)
        ep=ndtr(-(1+lam*gamma)/den); em=ndtr(-(1-lam*gamma)/den)
        gap=abs(beta*(ep-em)/2); V=(ep-em)**2/4
        bound=abs(beta)*np.sqrt(V); product=abs(beta*gamma)/sigma
        limit=lam/np.sqrt(1+lam*lam)*np.exp(-.5/den**2)/np.sqrt(2*np.pi)
        assert np.isclose(gap,bound,rtol=1e-12,atol=1e-15)
        gaussian.append(dict(lam=lam,gamma=gamma,gap=gap,head_bound=bound,product=product,ratio=gap/product,limit=limit))
fig,ax=plt.subplots(1,2,figsize=(7.6,2.9),layout='constrained')
for lam,c in zip([.5,1.,2.],colors):
    rows=[r for r in gaussian if r['lam']==lam]
    ax[0].loglog([r['head_bound'] for r in rows],[r['gap'] for r in rows],color=c,alpha=.7,lw=1.5)
    ax[0].scatter([r['head_bound'] for r in rows[::9]],[r['gap'] for r in rows[::9]],s=18,facecolors='none',edgecolors=c)
    ax[1].semilogx([r['gamma'] for r in rows],[r['ratio'] for r in rows],color=c,label=rf'$\lambda={lam:g}$')
    ax[1].axhline(rows[0]['limit'],color=c,ls=':',lw=1)
ax[0].plot([1e-6,.3],[1e-6,.3],color='#444444',ls='--',lw=.8)
ax[0].set(xlabel='Head-aware bound',ylabel='Exact risk gap',title='(a) Exact attainment')
ax[1].set(xlabel=r'Nuisance leakage $\gamma$',ylabel=r'Gap / $\sqrt{\bar\chi^2 L_{\rm CF}}$',title='(b) Local order sharpness')
ax[1].legend(loc='center left',bbox_to_anchor=(1,0.5),frameon=False)
for ext in ['pdf','svg','png']: fig.savefig(O/f'gaussian_tightness.{ext}',dpi=220)
plt.close(fig)

def projected(name):
    f=np.load(I/f'features_{name}.npz'); p=np.load(I/'fixed_projection.npz')
    x=(f['X'].astype(float)-p['center'])@p['basis']/p['scale']
    return np.concatenate([np.ones((*x.shape[:2],1)),x],axis=-1),f['y']
T,y=projected('test'); D,_=projected('development')
w=np.load(I/'fixed_heads.npz')['heads'][::13].mean(0)
q=(D[:,1]-D[:,0]).mean(0); q[0]=0; q/=np.linalg.norm(q)
rows=[]; raw={}
old=json.loads((I/'fixed_metrics.json').read_text())
for alpha in [0,.5,1.,2.]:
    wa=w+(alpha-1)*q*(q@w); pp=expit(2*(T@wa))
    loss=np.where(y[:,None]==1,1-pp,pp); h=loss-loss.mean(1,keepdims=True)
    V=float(np.mean(h*h)); delta=.9
    u=np.where(np.array([-1,1])[None,:]==y[:,None],-delta,delta)
    per=(u*loss).mean(1); gap=abs(float(per.mean()))
    local=float(np.abs(per).mean()); bound=delta*np.sqrt(V)
    assert np.allclose(np.abs(per),delta*np.sqrt(np.mean(h*h,axis=1)))
    oldgap=next(r['gap'] for r in old['transfer'] if r['alpha']==alpha and r['delta']==delta)
    assert abs(gap-oldgap)<1e-12
    aligned=[]
    for t in [.25,.5,1.]:
        density=1+t*h; assert np.min(density)>=0 and np.allclose(density.mean(1),1)
        shift=(density*loss).mean()-loss.mean(); chi=float(np.mean((density-1)**2)); bb=np.sqrt(chi*V)
        assert abs(shift-t*V)<1e-14 and abs(shift-bb)<1e-14
        aligned.append(dict(t=t,chi2=chi,gap=float(shift),bound=float(bb),ratio=float(shift/bb),min_density=float(density.min())))
    rows.append(dict(alpha=alpha,V=V,natural_gap=gap,local_bound=local,global_bound=bound,
      cancellation=gap/local,heterogeneity=local/bound,utilization=gap/bound,oracle=aligned))
    raw[f'loss_{alpha:g}']=loss; raw[f'centered_{alpha:g}']=h
np.savez_compressed(O/'qwen_arrays.npz',**raw)
fig,ax=plt.subplots(1,2,figsize=(7.6,3.0),layout='constrained')
x=np.arange(4); width=.23
for j,(key,label,c) in enumerate(zip(['cancellation','heterogeneity','utilization'],['Sign coherence','Magnitude homogeneity','Overall utilization'],colors)):
    ax[0].bar(x+(j-1)*width,[r[key] for r in rows],width,color=c,label=label)
ax[0].set(xticks=x,xticklabels=['0','0.5','1','2'],xlabel=r'Head parameter $\alpha$',ylabel='Ratio (unitless)',ylim=(0,1.05),title='(a) Cue-reversal slack')
ax[0].legend(loc='upper left',fontsize=8,frameon=False)
for j,r in enumerate(rows):
    ax[1].scatter([v['bound'] for v in r['oracle']],[v['gap'] for v in r['oracle']],marker=['o','s','^','D'][j],s=30,facecolors='none',edgecolors=(['#777777']+colors)[j],label=rf'$\alpha={r["alpha"]:g}$')
ax[1].plot([0,.00011],[0,.00011],ls='--',color='#555555',lw=1)
ax[1].set(xlabel='Exact finite-pool head-aware bound',ylabel='Oracle risk gap',title='(b) Loss-aligned stress test')
ax[1].ticklabel_format(axis='both',style='sci',scilimits=(0,0))
ax[1].legend(loc='upper left',frameon=False,fontsize=8)
for ext in ['pdf','svg','png']: fig.savefig(O/f'qwen_tightness.{ext}',dpi=220)
plt.close(fig)
out=dict(scope='Post-result finite-pool analysis; oracle uses evaluation losses and labels, not an independent OOD test.',n=1600,gaussian=gaussian,qwen=rows,
 inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [I/'fixed_metrics.json',I/'fixed_heads.npz',I/'fixed_projection.npz']})
(O/'metrics.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))
