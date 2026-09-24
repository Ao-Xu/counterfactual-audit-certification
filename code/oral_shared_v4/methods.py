"""Same-anchor moment certificate. Marginal events need no independence."""
import numpy as np
from oral_efficiency_v2.run import kl_upper
from oral_sampled_v3.methods import eb,summary

def shared_stats(mean_m,var_m,mean_q,N,rho=.05,delta=.025,J=1,B=1.,regime='iid'):
    if N<2 or J<1 or not 0<delta<1 or rho<0:raise ValueError('invalid design')
    e=delta/(2*J)
    if regime=='iid':
        mean_upper=eb(mean_m,var_m,N,2*B,e);cap=2*B*B
    elif regime=='wor_enumerated':
        mean_upper=mean_m+2*B*np.sqrt(np.log(1/e)/(2*N));cap=B*B
    else:raise ValueError('unknown sampling regime')
    vu=np.minimum(B*B,kl_upper(mean_q,N,cap,e))
    upper=mean_upper+np.sqrt(rho*vu)
    return dict(upper=upper,mean_upper=mean_upper,V_upper=vu,mean_radius=mean_upper-mean_m)

def shared(g,rho=.05,delta=.025,J=1,regime='iid',weights=None):
    g=np.asarray(g,dtype=float)
    if np.max(np.abs(g))>1+1e-10:raise ValueError('loss difference outside [-1,1]')
    if regime=='iid':M,Q=summary(g)
    else:
        w=np.full(g.shape[1],1/g.shape[1]) if weights is None else np.asarray(weights)
        if np.any(w<0) or not np.isclose(w.sum(),1):raise ValueError('invalid weights')
        M,Q=summary(g,'enumerated',w)
    return shared_stats(M.mean(),M.var(ddof=1),Q.mean(),len(M),rho,delta,J,regime=regime)
