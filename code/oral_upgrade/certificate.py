"""Bounded, joint finite-sample audit. No estimated maximum is a loss envelope.

Inputs: calibration and audit losses, each shaped (J, anchors, views).
Enumerated mode integrates a KNOWN finite reference law; IID mode requires
conditionally IID nuisance draws. Independent anchors/blocks are an assumption
to be supported by an external split manifest; ID checks detect simple leaks.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def bounded_mean_upper(x, cap, failure):
    """Chernoff/KL inversion for independent variables in [0, cap].
    Bernoulli mgf dominates every bounded variable with the same mean.
    This is not a Bernoulli-data assumption. No optional stopping permitted.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 1 or cap <= 0 or not 0 < failure < 1:
        raise ValueError('Invalid bounded-mean input')
    if not np.isfinite(x).all() or x.min() < -1e-10 or x.max() > cap+1e-10:
        raise ValueError('Loss envelope violated')
    a = float(np.clip(x.mean()/cap, 0, 1)); lo = a; hi = 1.
    t = math.log(1/failure)/len(x)
    for _ in range(55):
        b = (lo+hi)/2
        if b >= 1: break
        kl = (a*math.log(a/b) if a else 0) + ((1-a)*math.log((1-a)/(1-b)) if a<1 else 0)
        if kl > t: hi = b
        else: lo = b
    return min(cap, hi*cap)


def mean_variance_summary(x, mode, weights=None):
    if mode == 'enumerated':
        w = np.asarray(weights, float)
        if w.shape != (x.shape[1],) or np.any(w <= 0) or not np.isclose(w.sum(), 1):
            raise ValueError('A positive normalized reference law is required')
        m = x@w
        return m, ((x-m[:,None])**2)@w
    if mode != 'iid' or x.shape[1] < 2:
        raise ValueError('IID sensitivity calibration requires K>=2')
    k2 = (x.shape[1]//2)*2
    # Disjoint pairs are not independent across anchors' latent task states;
    # aggregate all of them before applying anchor-level concentration.
    v = ((x[:,:k2:2]-x[:,1:k2:2])**2).mean(1)/2
    return x.mean(1), v


def component(cal, audit, *, mode, weights, lo, hi, event_failure):
    r = hi-lo
    if min(len(cal),len(audit))<4 or cal.shape[1]!=audit.shape[1]:
        raise ValueError('Need >=4 anchors per block and matching K')
    for x in [cal,audit]:
        if not np.isfinite(x).all() or x.min()<lo-1e-8 or x.max()>hi+1e-8:
            raise ValueError('Fixed loss range violated')
    cm, cv = mean_variance_summary(cal,mode,weights)
    am, _ = mean_variance_summary(audit,mode,weights)
    m2 = len(cm)//2*2
    variance_pairs = (cm[:m2:2]-cm[1:m2:2])**2/2
    var_upper = min(r*r/4, bounded_mean_upper(variance_pairs,r*r/2,event_failure))
    v_upper = min(r*r/4, bounded_mean_upper(cv,r*r/(4 if mode=='enumerated' else 2),event_failure))
    n = len(am); t = math.log(2/event_failure)
    a = r*t/(3*n)
    bernstein = a + math.sqrt(2*var_upper*t/n+a*a)
    hoeffding = r*math.sqrt(math.log(2/event_failure)/(2*n))
    radius = min(bernstein,hoeffding)  # both events are in ledger
    svar = float(np.var(cm,ddof=1)); vh = float(cv.mean())
    return dict(mean=float(am.mean()), sampling_radius=radius,
                bernstein_radius=bernstein,hoeffding_radius=hoeffding,
                cluster_variance_hat=svar,cluster_variance_upper=var_upper,
                anchor_variance_hat=max(0.,svar-(vh/cal.shape[1] if mode=='iid' else 0)),
                sibling_variance_hat=vh, V_hat=vh,V_upper=v_upper,
                cal_anchors=len(cal),audit_anchors=n,K=cal.shape[1],mode=mode)


@dataclass(frozen=True)
class Corrections:
    # Use None for unavailable bounds. This triggers the bounded worst case.
    mismatch_upper: float | None = None
    semantic_upper: float | None = None
    failure_budget: float = 0.
    evidence: str = 'unavailable; use worst case'


def semantic_upper(independent_anchor_error_flags, failure):
    """Independent semantic-review outcomes, one unbiased draw per anchor.
    A model's own consistency score is not a semantic-error observation.
    """
    return bounded_mean_upper(independent_anchor_error_flags,1.,failure)


def finite_law_tv_upper(independent_style_ids,reference_weights,failure):
    """Global finite-law mismatch only; not arbitrary conditional laws.
    Use one independently sampled style per anchor; never flatten siblings.
    """
    w=np.asarray(reference_weights,float);x=np.asarray(independent_style_ids)
    if x.ndim!=1 or not len(x) or not 0<failure<1 or not np.isfinite(x).all() or np.any(x!=np.floor(x)) or np.any(w<=0) or not np.isclose(w.sum(),1) or np.any(x<0) or np.any(x>=len(w)):
        raise ValueError('Invalid finite-law samples')
    p=np.bincount(x.astype(int),minlength=len(w))/len(x)
    radius=len(w)/2*math.sqrt(math.log(2*len(w)/failure)/(2*len(x)))
    return min(1.,float(abs(p-w).sum()/2+radius))


def audit_family(cal_cf,cal_factual,audit_cf,audit_factual,*,cal_ids,audit_ids,
                 rho,delta=.025,B=1.,mode='enumerated',weights=None,
                 corrections=None,policy_names=None):
    """Joint coverage across all J policies AND paired/separate/bounded methods.
    Calibration losses must be valid-label losses under the reference law.
    Corrections refer to the audit mean; they do not repair invalid calibration.
    """
    cc,cf,ac,af = map(lambda x:np.asarray(x,float),[cal_cf,cal_factual,audit_cf,audit_factual])
    if cc.ndim!=3 or ac.ndim!=3 or cc.shape[0]!=ac.shape[0] or cc.shape[1:]!=cf.shape or ac.shape[1:]!=af.shape:
        raise ValueError('Require (J,n,K) candidate and (n,K) factual arrays')
    if len(cal_ids)!=cc.shape[1] or len(audit_ids)!=ac.shape[1] or len(set(cal_ids))!=len(cal_ids) or len(set(audit_ids))!=len(audit_ids) or set(cal_ids)&set(audit_ids):
        raise ValueError('Duplicate or overlapping anchor IDs')
    if rho<0 or B<=0 or not 0<delta<1:raise ValueError('Invalid audit parameters')
    if any((x<0).any() or (x>B+1e-8).any() for x in [cc,cf,ac,af]):raise ValueError('Loss outside [0,B]')
    cor=corrections or Corrections()
    if not 0<=cor.failure_budget<delta:raise ValueError('External confidence budget exceeds delta')
    eps=1. if cor.mismatch_upper is None else cor.mismatch_upper
    eta=1. if cor.semantic_upper is None else cor.semantic_upper
    if not 0<=eps<=1 or not 0<=eta<=1:raise ValueError('Invalid discrepancy bounds')
    if (eps<1 or eta<1) and not cor.evidence:raise ValueError('Corrections require provenance')
    J=len(cc); e=(delta-cor.failure_budget)/(12*J)
    names=policy_names or [str(j) for j in range(J)]
    if len(names)!=J or len(set(names))!=J:raise ValueError('Unique candidate names required')
    rows=[]
    for j,name in enumerate(names):
        kw=dict(mode=mode,weights=weights,event_failure=e)
        p=component(cc[j]-cf,ac[j]-af,lo=-B,hi=B,**kw)
        f=component(cf,af,lo=0,hi=B,**kw); c=component(cc[j],ac[j],lo=0,hi=B,**kw)
        # A variance calibrated under nu need not bound a mismatched audit law.
        # Until an audit-law variance certificate is independently provided,
        # retain distribution-free Hoeffding for all mean comparisons.
        if eps>0 or eta>0:
            for part in [p,f,c]:
                part['sampling_radius']=part['hoeffding_radius']
                part['variance_transfer']='not assumed; bounded mean fallback'
        # Valid joint triangle improvement uses already allocated events.
        vu=min(p['V_upper'],(math.sqrt(f['V_upper'])+math.sqrt(c['V_upper']))**2)
        nuisance=2*B*eps; semantic=2*B*eta; correction=nuisance+semantic
        width=p['sampling_radius']+math.sqrt(rho*vu)+correction
        U=p['mean']+width; L=p['mean']-width
        rows.append(dict(policy=name,Delta_hat=p['mean'],sampling_radius=p['sampling_radius'],
                         VDelta_hat=p['V_hat'],VDelta_upper=vu,
                         nuisance_correction=nuisance,semantic_correction=semantic,
                         robustness_correction=math.sqrt(rho*vu),U_pair=U,L_pair=L,
                         decision='benefit' if U<0 else 'harm' if L>0 else 'uncertain',
                         U_separate=p['mean']+f['sampling_radius']+c['sampling_radius']+
                          math.sqrt(rho)*(math.sqrt(f['V_upper'])+math.sqrt(c['V_upper']))+correction,
                         U_bounded=p['mean']+p['hoeffding_radius']+B*math.sqrt(rho)+correction,
                         U_reference=p['mean']+p['sampling_radius']+correction,
                         paired_details=p,factual_details=f,candidate_details=c))
    selections={}
    for method,key in [('paired','U_pair'),('separate','U_separate'),('bounded','U_bounded'),('paired_validation','U_reference'),('reference_only','Delta_hat')]:
        best=min(rows,key=lambda r:(r[key],r['policy']))
        selections[method]=best['policy'] if best[key]<0 else 'factual'
    ledger=dict(delta=delta,J=J,events_per_policy=12,event_failure=e,
                event_types=['paired/CF/factual calibration cluster-variance upper',
                             'paired/CF/factual reference sensitivity upper',
                             'paired/CF/factual two-sided Bernstein audit mean',
                             'paired/CF/factual two-sided Hoeffding audit mean'],
                external_failure=cor.failure_budget,total=12*J*e+cor.failure_budget)
    return dict(rows=rows,selections=selections,ledger=ledger,rho=rho,
                correction_evidence=cor.evidence,calibration_requires_valid_reference_labels=True,
                guarantees='paired/separate/bounded cover specified class; validation methods only reference')


def exact_finite_robust(g,weights,rho):
    """Numerical global-conditional chi-square DRO on uniform finite anchors.
    Solves the nonnegative density constraint, unlike a sqrt-envelope plug-in.
    Result includes feasibility and dual gap; caller must check tolerance.
    """
    g=np.asarray(g,float); w=np.asarray(weights,float)
    if g.ndim!=2 or not g.size or not np.isfinite(g).all() or w.shape!=(g.shape[1],) or np.any(w<=0) or not np.isclose(w.sum(),1) or not np.isfinite(rho) or rho<0:
        raise ValueError('Invalid finite DRO input')
    if rho==0:return dict(value=float((g@w).mean()),chi2=0.,duality_gap=0.)
    # Minimal-divergence law supported on row maxima (handles ties).
    maxima=np.isclose(g,g.max(1,keepdims=True),atol=1e-14,rtol=0)
    pmax=maxima*w; pmax/=pmax.sum(1,keepdims=True)
    chimax=float(((pmax/w-1)**2*w).sum(1).mean())
    if rho>=chimax:return dict(value=float(g.max(1).mean()),chi2=chimax,duality_gap=0.)
    def distribution(eta):
        low=g.min(1)-2*eta; high=g.max(1)
        for _ in range(45):
            lam=(low+high)/2
            mass=(np.maximum(g-lam[:,None],0)*w).sum(1)
            low=np.where(mass>2*eta,lam,low);high=np.where(mass>2*eta,high,lam)
        p=w*np.maximum(g-((low+high)/2)[:,None],0)/(2*eta)
        p/=p.sum(1,keepdims=True)
        return p,float(((p/w-1)**2*w).sum(1).mean())
    hi=max(float(np.ptp(g)),1.)
    while distribution(hi)[1]>rho:hi*=2
    lo=0.
    for _ in range(55):
        mid=(lo+hi)/2
        if distribution(mid)[1]>rho:lo=mid
        else:hi=mid
    p,chi=distribution(hi);value=float((p*g).sum(1).mean())
    return dict(value=value,chi2=chi,duality_gap=float(hi*(rho-chi)))
