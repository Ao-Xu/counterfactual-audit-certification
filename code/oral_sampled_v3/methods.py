"""Bounded paired certificates; independent units are task anchors."""
import math
import numpy as np
from oral_efficiency_v2.run import kl_upper

def eb(mean,var,n,span,e):
    # Maurer--Pontil (2009), Theorem 4, one-sided, unbiased sample variance.
    return mean+np.sqrt(2*np.maximum(var,0)*np.log(2/e)/n)+7*span*np.log(2/e)/(3*(n-1))

def kl_radius(dmean,m,n,span,e):
    v=np.minimum(span*span/4,kl_upper(dmean,m//2,span*span/2,e))
    t=math.log(2/e);a=span*t/(3*n)
    return np.minimum(a+np.sqrt(2*v*t/n+a*a),span*np.sqrt(t/(2*n)))

def paired(cal,audit,rho=.05,delta=.025,J=1,mode='iid',weights=None):
    M,Q=summary(cal,mode,weights);MA,_=summary(audit,mode,weights);e=delta/(4*J)
    d=((M[::2][:len(M)//2]-M[1::2])**2).mean()/2
    v=min(1,float(kl_upper(Q.mean(),len(M),2 if mode=='iid' else 1,e)))
    return float(MA.mean()+kl_radius(d,len(M),len(MA),2,e)+np.sqrt(rho*v))

def summary(g,mode='iid',weights=None):
    if mode=='iid':
        assert g.shape[1]>=2 and g.shape[1]%2==0
        return g.mean(1),((g[:,::2]-g[:,1::2])**2).mean(1)/2
    w=np.asarray(weights);M=g@w
    return M,((g-M[:,None])**2)@w

def generic_sampled(g,rho=.05,delta=.025,J=1,eta_grid=None):
    """Valid beyond observed support: relax density positivity, unbiased variance.
    Same-anchor dependence between mean and Q stays in the empirical variance.
    Optimizes a simultaneous frozen dual grid, so no calibration split needed.
    """
    eta=np.geomspace(.01,2,31) if eta_grid is None else np.asarray(eta_grid)
    M,Q=summary(g);Y=M[:,None]+Q[:,None]/(4*eta)
    span=1+np.maximum(1,1/(2*eta))
    up=eb(Y.mean(0),Y.var(0,ddof=1),len(g),span,delta/(J*len(eta)))+rho*eta
    j=int(np.argmin(up))
    return dict(upper=float(up[j]),eta=float(eta[j]),mean=float(Y[:,j].mean()),variance=float(Y[:,j].var(ddof=1)),range=float(span[j]),all_upper=up.tolist())

def exact_dual_response(g,w,eta):
    """Nonnegative-density conditional chi-square dual, exact finite views."""
    lo=g.min(1)-2*eta;hi=g.max(1)-2*eta
    for _ in range(50):
        t=(lo+hi)/2;v=np.maximum(g-t[:,None],0)@w
        lo=np.where(v>2*eta,t,lo);hi=np.where(v>2*eta,hi,t)
    t=(lo+hi)/2
    return t+eta+(np.maximum(g-t[:,None],0)**2)@w/(4*eta)

def generic_enumerated(g,w,rho=.05,delta=.025,J=1,eta_grid=None):
    eta=np.geomspace(.001,2,80) if eta_grid is None else np.asarray(eta_grid)
    Y=np.stack([exact_dual_response(g,np.asarray(w),v) for v in eta],axis=1)
    assert Y.min()>=-1-1e-8 and Y.max()<=1+1e-8
    U=eb(Y.mean(0),Y.var(0,ddof=1),len(g),2,delta/(J*len(eta)))+rho*eta
    j=int(np.argmin(U));return dict(upper=float(U[j]),eta=float(eta[j]),dual_mean=float(Y[:,j].mean()),radius=float(U[j]-rho*eta[j]-Y[:,j].mean()))
