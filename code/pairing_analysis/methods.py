"""Matched event-level ledgers, paired/separate representations, split/shared roles."""
import numpy as np
from certificate_efficiency.run import kl_upper
from sampled_interventions.methods import eb,summary

def radius(var,n,span,e,regime='iid'):
 if np.any(np.asarray(n)<2):raise ValueError('need at least 2 anchors')
 if regime=='iid':return eb(0,var,n,span,e)
 if regime=='wor_enumerated':return span*np.sqrt(np.log(1/e)/(2*n))
 raise ValueError('unknown regime')

def component(mean,var,q,n,m,span,e,rho=.05,regime='iid'):
 cap=span**2/(2 if regime=='iid' else 4)
 vu=np.minimum(span**2/4,kl_upper(q,m,cap,e))
 r=radius(var,n,span,e,regime)
 return dict(mean=mean,mean_radius=r,sensitivity_radius=np.sqrt(rho*vu),V_upper=vu,upper=mean+r+np.sqrt(rho*vu))

def certificate(f,c,m=None,rho=.05,delta=.025,J=3,separate=False,regime='iid',e=None):
 f=np.asarray(f,float);c=np.asarray(c,float)
 if f.shape!=c.shape or np.any(f<0) or np.any(c<0) or np.any(f>1) or np.any(c>1):raise ValueError('bounded aligned losses required')
 N=len(f);e=delta/(2*(J+1)) if e is None else e
 if m is not None and (m<2 or N-m<2):raise ValueError('invalid split')
 def parts(z,span):
  M,Q=summary(z) if regime=='iid' else summary(z,'enumerated',np.ones(z.shape[1])/z.shape[1])
  qm=Q.mean() if m is None else Q[:m].mean();ma=M if m is None else M[m:]
  return component(ma.mean(),ma.var(ddof=1),qm,len(ma),N if m is None else m,span,e,rho,regime)
 if not separate:return parts(c-f,2)
 cf=parts(c,1);ff=parts(-f,1)
 return {k:cf[k]+ff[k] for k in ['mean','mean_radius','sensitivity_radius','upper']}

def generic_fixed(mean,var_m,var_q,cov_mq,q,n,eta,e,rho=.05,loose=False,hoeffding=False):
 ym=mean+q/(4*eta);yv=var_m+var_q/(16*eta**2)+cov_mq/(2*eta)
 span=2+1/(2*eta) if loose else 1+np.maximum(1,1/(2*eta))
 u=ym+rho*eta+radius(yv,n,span,e,'wor_enumerated' if hoeffding else 'iid')
 return u
