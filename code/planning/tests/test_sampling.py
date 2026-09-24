import itertools,unittest
import numpy as np
from planning.laws import support,moments,draw_stats
from planning.planner import features,evaluate

class ExactSampling(unittest.TestCase):
 def test_k2_matches_raw_enumeration(self):
  law=dict(a=.2,w=.1,b=.07,h=.2,d=.06)
  raw=[]
  for a,w,s,t in itertools.product([0,1],[-1,1],[-1,0,1],[-1,0,1]):
   f=np.array([.45+.1*w]*2);c=f-.06+.2*(a-.5)+.07*np.sqrt(1+.2*(2*a-1))*np.sqrt(1.5)*np.array([s,t])
   raw.append(np.stack([f,c],axis=-1))
  v=features(np.array(raw));x,p=support(law,2);mu,cv=moments(x,p)
  np.testing.assert_allclose(mu,v.mean(0),atol=1e-14)
  np.testing.assert_allclose(cv,np.cov(v,rowvar=False,ddof=0),atol=1e-14)
 def test_sibling_moment_identities(self):
  law=dict(a=.2,w=.1,b=.07,h=.2,d=.06)
  for k in [2,8]:
   x,p=support(law,k);mu,cv=moments(x,p)
   self.assertAlmostEqual(mu[1],.07**2)
   self.assertAlmostEqual(cv[0,0],.2**2/4+.07**2/k)
   self.assertAlmostEqual(cv[2,3],.1**2)
 def test_count_statistics_and_endpoint(self):
  x=np.array([[-.2,0,.3,.5,0,0],[.1,0,.6,.5,0,0]])
  mu,cv=draw_stats(x,np.array([.5,.5]),40,3,np.random.default_rng(61))
  for i in range(3):
   n=int(round(40*(mu[i,0]+.2)/.3));raw=np.repeat(x,[40-n,n],axis=0)
   np.testing.assert_allclose(mu[i],raw.mean(0),atol=1e-14)
   np.testing.assert_allclose(cv[i],np.cov(raw,rowvar=False),atol=1e-14)
  en=dict(method='generic',N=40,eta=.25,em=.025)
  self.assertTrue(np.isfinite(evaluate(mu,cv,en,.05)).all())

if __name__=='__main__':unittest.main()
