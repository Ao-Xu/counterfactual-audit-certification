import unittest
import numpy as np
from oral_planner_v6.planner import tangent,bounds,A
from oral_efficiency_v2.run import kl_upper

class Checks(unittest.TestCase):
 def test_global_tangent(self):
  for span in [1,2]:
   for n in [256,10000]:
    for q0 in [.0001,.01,.1]:
     off,a=tangent(q0,n,.0125,.05,span)
     q=np.linspace(0,span*span/2,501)
     exact=np.sqrt(.05*np.minimum(span*span/4,kl_upper(q,n,span*span/2,.0125)))
     self.assertTrue(np.all(off+a*q>=exact-1e-10))
 def test_decreasing_increment(self):
  values=A(np.linspace(0,.1,50),.02,2,1000,.01,.01)
  self.assertTrue(np.all(np.diff(values)<=1e-12))
 def test_degenerate_not_inferred(self):
  x=np.zeros((128,1));b=bounds(x,[0],[2],[0],.001)
  self.assertGreater(b['hi'][0],0)
 def test_interval_shapes(self):
  r=np.random.default_rng(710);x=r.uniform(-.5,.5,(4000,2));b=bounds(x,[-.5]*2,[.5]*2,[0]*2,.0001)
  self.assertTrue(np.all(b['lo']<0));self.assertTrue(np.all(b['hi']>0))
  self.assertTrue(np.all(b['vl']<=1/12));self.assertTrue(np.all(b['vu']>=1/12))
  self.assertTrue(np.all(b['third']>=1/32))

if __name__=='__main__':unittest.main()
