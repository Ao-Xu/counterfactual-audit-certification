"""Exact anchor-sibling categorical sampling (no Gaussian surrogate)."""
import itertools
import numpy as np
from scipy.optimize import brentq
from scipy.special import rel_entr
from planning.planner import features


def support(law,k,theta=.5):
    # Enumerate pair summaries, then convolve K/2 independent pairs.
    one={}
    for x,y in itertools.product([-1,0,1],repeat=2):
        key=(x+y,(x-y)**2);one[key]=one.get(key,0)+1/9
    dist={(0,0):1.}
    for _ in range(k//2):
        out={}
        for (s,q),p in dist.items():
            for (ss,qq),pp in one.items():out[(s+ss,q+qq)]=out.get((s+ss,q+qq),0)+p*pp
        dist=out
    a,b,h,w,d=law['a'],law['b'],law['h'],law['w'],law['d']
    rows=[];probs=[]
    for anchor,noise in itertools.product([0,1],[-1,1]):
        amp=b*np.sqrt(1+h*(2*anchor-1));f=.45+w*noise
        for (s,q),p in dist.items():
            mu=-d+a*(anchor-.5)+amp*np.sqrt(1.5)*s/k
            qv=amp**2*1.5*q/k
            rows.append([mu,qv,f+mu,f,qv,0.]);probs.append(p*(theta if anchor else 1-theta)/2)
    x=np.array(rows);p=np.array(probs);assert abs(p.sum()-1)<1e-10
    return x,p


def draw_pilot(law,n,k,rng,theta=.5):
    a=rng.binomial(1,theta,size=n);w=rng.choice([-1,1],n)
    s=rng.choice([-1.,0.,1.],(n,k))*np.sqrt(1.5)
    f=np.broadcast_to((.45+law['w']*w)[:,None],(n,k))
    c=f-law['d']+law['a']*(a[:,None]-.5)+law['b']*np.sqrt(1+law['h']*(2*a[:,None]-1))*s
    assert c.min()>=0 and c.max()<=1 and f.min()>=0 and f.max()<=1
    return np.stack([f,c],axis=-1)


def moments(x,p):
    mu=p@x;z=x-mu;cv=(z.T*p)@z
    return mu,cv


def draw_stats(x,p,n,reps,rng):
    counts=rng.multinomial(n,p,size=reps)
    mu=counts@x/n
    second=counts@np.einsum('si,sj->sij',x,x).reshape(len(x),36)/n
    cv=(second.reshape(-1,6,6)-mu[:,:,None]*mu[:,None,:])*n/(n-1)
    return mu,cv


def psi(law,theta,rho):
    return -law['d']+law['a']*(theta-.5)+np.sqrt(rho)*law['b']*np.sqrt(1+law['h']*(2*theta-1))


def lower_bound(law,target,rho,delta,delta_plan):
    null=brentq(lambda t:psi(law,t,rho),.500001,.999999)
    kl=lambda p,q:float(rel_entr(p,q)+rel_entr(1-p,1-q))
    q=(1-delta_plan)*target
    return dict(theta1=.5,theta0=null,per_anchor_KL=kl(.5,null),
        unconditional_power=q,total_anchor_lower_bound=kl(q,delta)/kl(.5,null),
        qualification='Assumes a feasible returned plan almost surely; includes pilot. Local two-point subclass, not minimax.')
