"""Pilot-independent final audit. Bounded IID anchors, even IID siblings.

No population probabilities or final outcomes are inputs to plan().
Known support bounds must be declared before observing the pilot.
All search choices share a simultaneous pilot confidence event.
"""
import numpy as np
from scipy.stats import norm
from sampled_interventions.methods import eb
from certificate_efficiency.run import kl_upper


def B(sig, span, n, e, beta):
    t, u = np.log(2/e), np.log(1/beta)
    return sig*(np.sqrt(2*u/n)+np.sqrt(2*t/n))+span*(2*u/(3*n)+2*np.sqrt(t*u)/np.sqrt(n*(n-1))+7*t/(3*(n-1)))


def A(vlo, omega_up, cap, n, e, beta):
    # The adaptive increment is decreasing in V and increasing in Omega.
    # Do not infer V=0 from a zero empirical variance.
    u, l = np.log(1/beta), np.log(1/e)
    return np.minimum(np.sqrt(cap/n)*(np.sqrt(l)+np.sqrt(u)),
        np.sqrt(vlo+np.sqrt(2*omega_up*u/n)+2*cap*u/(3*n))-np.sqrt(vlo)+np.sqrt(cap*l/n))


def bounds(x, lo, hi, center, e):
    """Five marginal events: mean +/-; centered second +/-; absolute third +.
    Center is from independent design pilot. Conditional bounds remain valid.
    """
    x = np.asarray(x); lo=np.asarray(lo); hi=np.asarray(hi)
    n=len(x); r=hi-lo; mu=x.mean(0); var=x.var(0,ddof=1)
    rad=eb(0,var,n,r,e)
    ml=np.maximum(lo,mu-rad); mh=np.minimum(hi,mu+rad)
    c=np.clip(center,lo,hi); rr=np.maximum(abs(lo-c),abs(hi-c))
    d=x-c; second=d*d; third=abs(d)**3
    # EB rather than numerical inversion also covers zero-width variables.
    srad=eb(0,second.var(0,ddof=1),n,rr**2,e)
    sl=np.maximum(0,second.mean(0)-srad); su=np.minimum(rr**2,second.mean(0)+srad)
    near=np.maximum(np.maximum(ml-c,c-mh),0); far=np.maximum(abs(ml-c),abs(mh-c))
    vl=np.maximum(0,sl-far**2); vu=np.maximum(0,np.minimum(r*r/4,su-near**2))
    tu=np.minimum(rr**3,third.mean(0)+eb(0,third.var(0,ddof=1),n,rr**3,e))
    tu=np.minimum(r*vu,(np.cbrt(np.maximum(tu,0))+far)**3)
    return dict(mean=mu,var=var,lo=ml,hi=mh,vl=vl,vu=vu,third=tu)


def tangent(q0, n, e, rho, span):
    """Conservative secant slope bracketing of concave sqrt(KL-upper).
    Analytic implicit derivative away from physical-cap kink.
    """
    cap=span*span/2; physical=span*span/4
    x=np.clip(q0/cap,1e-10,1-1e-10)
    upper=np.asarray(kl_upper(cap*x,n,cap,e)); y=upper/cap
    clipped=upper>=physical
    # kl_x=logit(x)-logit(y), kl_y=(y-x)/(y(1-y)).
    dy=(np.log(y/(1-y))-np.log(x/(1-x)))*y*(1-y)/(y-x)
    slope=np.where(clipped,0.,np.sqrt(rho)*dy/(2*np.sqrt(upper)))
    value=np.sqrt(rho*np.minimum(physical,upper))
    return value-slope*q0,slope


def features(loss):
    # Return M_delta, Q_delta, M_cf, M_f, Q_cf, Q_f.
    f,c=loss[...,0],loss[...,1]; z=c-f
    q=lambda y: ((y[:,::2]-y[:,1::2])**2).mean(1)/2
    return np.stack([z.mean(1),q(z),c.mean(1),f.mean(1),q(c),q(f)],axis=1)


def endpoint(mean,cov,n,method,e,rho,eta=None):
    if method=='generic':
        v=np.zeros(6);v[0]=1;v[1]=1/(4*eta)
        return eb(mean@v,np.einsum('i,...ij,j->...',v,cov,v),n,1+max(1,1/(2*eta)),e)+rho*eta
    if method=='paired':
        return eb(mean[...,0],cov[...,0,0],n,2,e)+np.sqrt(rho*np.minimum(1,kl_upper(mean[...,1],n,2,e)))
    return mean[...,0]+eb(0,cov[...,2,2],n,1,e)+eb(0,cov[...,3,3],n,1,e)+sum(np.sqrt(rho*np.minimum(.25,kl_upper(mean[...,q],n,.5,e))) for q in [4,5])


def plan(pilot, supports, cfg, target, population=None, oracle=False):
    """population=(means,covs,thirds) is allowed ONLY for labeled benchmarks.
    For the valid planner no such argument is supplied.
    """
    ns=np.asarray(cfg['N_grid']); rho=cfg['rho']; delta=cfg['delta']; J=cfg['J']
    r0=cfg['design_pilot']; bet=(1-target)/(6*(J+1)); bs=(1-target)/(8*J)
    # Build complete finite menu using only the design pilot for tangencies.
    entries=[]; blocks={}; transforms={}
    for k in cfg['K_grid']:
        f=features(pilot[:,:k]); supp=supports[k]; cent=f[:r0].mean(0)
        if oracle: cent=population[k][0]
        vectors=[np.eye(6)[i] for i in range(6)]; ranges=[(supp[:,i].min(),supp[:,i].max()) for i in range(6)]
        for allocation in cfg['mean_fractions']:
            # paired two-event and separate four-event final safety ledgers.
            for method in ['paired','separate']:
                e=delta/(2*J) if method=='paired' else delta/(2*(J+1))
                # Allocation encoded by actual mean/Q tail probabilities.
                em=delta*allocation/(J if method=='paired' else J+1)
                eq=delta*(1-allocation)/(J if method=='paired' else J+1)
                cs=[(1,2)] if method=='paired' else [(4,1),(5,1)]
                for n in ns:
                    v=np.zeros(6);v[0]=1;off=0.
                    for qi,span in cs:
                        intercept,slope=tangent(max(float(cent[qi]),1e-9),n,eq,rho,span)
                        off+=float(intercept);v[qi]+=float(slope)
                    ix=len(vectors);vectors.append(v);x=supp@v;ranges.append((x.min(),x.max()))
                    entries.append(dict(K=k,N=int(n),method=method,eta=None,em=em,eq=eq,index=ix,offset=off,allocation=allocation))
        for eta in cfg['eta_grid']:
            v=np.zeros(6);v[0]=1;v[1]=1/(4*eta);ix=len(vectors);vectors.append(v);x=supp@v;ranges.append((x.min(),x.max()))
            for n in ns: entries.append(dict(K=k,N=int(n),method='generic',eta=eta,em=delta/J,eq=None,index=ix,offset=rho*eta,allocation=1.))
        vectors=np.asarray(vectors).T;transforms[k]=vectors
        blocks[k]=(f@vectors,np.array(ranges),cent@vectors)
    # Five events per scalar, union across every K/eta/N/ledger search choice.
    count=sum(v[0].shape[1] for v in blocks.values()); ep=cfg['delta_plan']/(5*count)
    intervals={k:bounds(x[r0:],rg[:,0],rg[:,1],ce,ep) for k,(x,rg,ce) in blocks.items()}
    if oracle:
        for k in intervals:
            mu,cv,prob=population[k];v=transforms[k];x=supports[k]@v;ym=mu@v;yv=np.einsum('ij,ik,kj->j',v,cv,v)
            third=prob@(abs(x-ym)**3)
            intervals[k]=dict(mean=ym,var=yv,lo=ym,hi=ym,vl=yv,vu=yv,third=third)
    choices={'valid':[],'plugin':[],'sufficient':[],'oracle':[]}
    for en in entries:
        k,n,method=en['K'],en['N'],en['method']; d=intervals[k];idx=en['index'];e=en['em'];t=np.log(2/e)
        # Certified moment supremum of Theorem 4, explicit conservative box.
        if method=='generic':
            span=1+max(1,1/(2*en['eta']))
            sufficient=d['hi'][idx]+en['offset']+B(np.sqrt(d['vu'][idx]),span,n,e,bet)
            meanidx=[idx];spans=[span]
        else:
            meanidx=[0] if method=='paired' else [2,3];spans=[2] if method=='paired' else [1,1]
            qis=[1] if method=='paired' else [4,5]
            sufficient=d['hi'][0]+sum(B(np.sqrt(d['vu'][i]),s,n,e,bet) for i,s in zip(meanidx,spans))
            for qi,s in zip(qis,spans):
                sufficient+=np.sqrt(rho)*(np.sqrt(max(0,d['hi'][qi]))+A(max(0,d['lo'][qi]),d['vu'][qi],s*s/2,n,en['eq'],bet))
        # Berry--Esseen power bound for a tangent majorant, not an approximation.
        berr=.56*d['third'][idx]/max(d['vl'][idx],1e-300)**1.5/np.sqrt(n) if d['vl'][idx]>0 else np.inf
        q=1-(1-target)/J+len(meanidx)*bs+berr
        improved=np.inf
        if q<1:
            inflated=sum(np.sqrt(2*t/n)*(np.sqrt(d['vu'][i])+s*np.sqrt(2*np.log(1/bs)/(n-1)))+7*s*t/(3*(n-1)) for i,s in zip(meanidx,spans))
            improved=d['hi'][idx]+en['offset']+inflated+norm.ppf(q)*np.sqrt(d['vu'][idx]/n)
        if min(sufficient,improved)<0:
            choices['valid'].append({**en,'witness':'BE' if improved<sufficient else 'T4','power_upper':float(min(sufficient,improved))})
        # Uncorrected point-moment Gaussian forecast, deliberately no guarantee.
        pointmu=d['mean'][idx];pointvar=d['var'][idx]
        approximate=pointmu+en['offset']+sum(np.sqrt(2*t/n*d['var'][i])+7*s*t/(3*(n-1)) for i,s in zip(meanidx,spans))+norm.ppf(target)*np.sqrt(pointvar/n)
        if approximate<0:choices['plugin'].append(en)
        if population is not None:
            mu,cv=population[k][:2];v=transforms[k][:,idx];ym=mu@v;yv=v@cv@v
            approx=ym+en['offset']+sum(np.sqrt(2*t/n*(yv if method=='generic' else cv[i,i]))+7*s*t/(3*(n-1)) for i,s in zip(meanidx,spans))+norm.ppf(target)*np.sqrt(yv/n)
            if approx<0:choices['oracle'].append(en)
            if method=='generic':popup=ym+en['offset']+B(np.sqrt(yv),spans[0],n,e,bet)
            else:
                popup=mu[0]+sum(B(np.sqrt(cv[i,i]),s,n,e,bet) for i,s in zip(meanidx,spans))
                for qi,s in zip(qis,spans):popup+=np.sqrt(rho)*(np.sqrt(mu[qi])+A(mu[qi],cv[qi,qi],s*s/2,n,en['eq'],bet))
            if popup<0:choices['sufficient'].append(en)
    selected={key:(min(value,key=lambda x:(x['N'],x['N']*x['K'],x['method'],x['allocation'])) if value else None) for key,value in choices.items()}
    return dict(selected=selected,scalar_events=5*count,pilot_e=ep,
        diagnostics={str(k):dict(mean=features(pilot[:,:k]).mean(0).tolist(),cov=np.cov(features(pilot[:,:k]),rowvar=False).tolist(),mean_lo=d['lo'][:6].tolist(),mean_hi=d['hi'][:6].tolist(),variance_lo=d['vl'][:6].tolist(),variance_hi=d['vu'][:6].tolist()) for k,d in intervals.items()})


def evaluate(mu,cv,en,rho):
    if en['method']=='generic':return endpoint(mu,cv,en['N'],'generic',en['em'],rho,en['eta'])
    n=en['N'];e=en['em'];eq=en['eq']
    if en['method']=='paired':return eb(mu[...,0],cv[...,0,0],n,2,e)+np.sqrt(rho*np.minimum(1,kl_upper(mu[...,1],n,2,eq)))
    return mu[...,0]+sum(eb(0,cv[...,i,i],n,1,e) for i in [2,3])+sum(np.sqrt(rho*np.minimum(.25,kl_upper(mu[...,q],n,.5,eq))) for q in [4,5])
