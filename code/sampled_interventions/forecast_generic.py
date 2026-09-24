import json,sys,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from sampled_interventions.methods import exact_dual_response
from finite_catalog.diagnose_existing import families
R=Path(__file__).resolve().parent;r=json.load(open(R/'results/retrospective.json'));lookup={(x['task'],x['family'],x['policy']):x for x in r['rows']};eta=np.geomspace(.001,2,80)
for task,fam,paths in families():
 f=dict(np.load(paths['baseline_0' if task=='qa' else 'factual']));w=np.ones(4)/4 if task=='qa' else np.array([.325,.325,.175,.175])
 for pol,path in paths.items():
  if (task,fam,pol) not in lookup:continue
  row=lookup[task,fam,pol];c=dict(np.load(path));g=np.concatenate([c[s]-f[s] for s in ['calibration','reference']]);Y=np.stack([exact_dual_response(g,w,e) for e in eta],1);target=Y.mean(0)+.05*eta;var=Y.var(0,ddof=1);t=math.log(2*row['J']*len(eta)/row['delta'])
  row['generic_EB_sqrt_width_at_current']=float(np.sqrt(2*var[np.argmin(target+np.sqrt(2*var*t/len(g)))]*t/len(g)))
  row['generic_EB_linear_width_at_current']=float(14*t/(3*(len(g)-1)))
  def up(N):return (target+np.sqrt(2*var*t/N)+14*t/(3*(N-1))).min()
  if target.min()>=0:row['planned_generic_total_anchors']=None;continue
  lo,hi=2,4
  while hi<2**28 and up(hi)>=0:hi*=2
  if up(hi)>=0:row['planned_generic_total_anchors']=None;continue
  while hi-lo>1:
   mid=(hi+lo)//2
   if up(mid)<0:hi=mid
   else:lo=mid
  row['planned_generic_total_anchors']=hi
for task in ['hans','hans_extension','qa']:
 rr=[x for x in r['rows'] if x['task']==task];ns=[x['planned_generic_total_anchors'] for x in rr if x['planned_generic_total_anchors']]
 r['summary'][task]['generic_planning_total_median']=float(np.median(ns));r['summary'][task]['generic_planning_total_range']=[min(ns),max(ns)];r['summary'][task]['generic_EB_linear_width_median']=float(np.median([x['generic_EB_linear_width_at_current'] for x in rr]))
r['planning_scope']='Expected-moment diagnostic, not power guarantee. Uses fixed inspected-catalog dual moments and EB formula; no outcome tuning of rho or candidates.'
(R/'results/retrospective.json').write_text(json.dumps(r,indent=2));print(json.dumps(r['summary'],indent=2))
