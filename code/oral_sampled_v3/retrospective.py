import sys,json,math,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from oral_upgrade.diagnose_existing import families
from oral_upgrade.certificate import exact_finite_robust
from oral_sampled_v3.methods import paired,generic_enumerated,summary
R=Path(__file__).resolve().parent

def analyze(task,fam,pol,cal,audit,w,J,delta):
 g=np.concatenate([cal,audit]);M,Q=summary(cal,'enumerated',w)
 p=paired(cal,audit,rho=.05,delta=delta,J=J,mode='enumerated',weights=w)
 ga=generic_enumerated(audit,w,rho=.05,delta=delta,J=J)
 gp=generic_enumerated(g,w,rho=.05,delta=delta,J=J)
 exact=exact_finite_robust(g,np.asarray(w),.05)
 m,v=summary(g,'enumerated',w)
 envelope=float(m.mean()+np.sqrt(.05*v.mean()))
 return dict(task=task,family=fam,policy=pol,J=J,delta=delta,m=len(cal),n=len(audit),paired_KL=p,generic_same_audit=ga,generic_all_available=gp,catalog_robust_target=exact['value'],catalog_envelope=envelope,catalog_envelope_slack=envelope-exact['value'],catalog_reference=float(m.mean()),generic_population_grid_upper=float(min((__import__('oral_sampled_v3.methods',fromlist=['exact_dual_response']).exact_dual_response(g,np.asarray(w),eta).mean()+.05*eta) for eta in np.geomspace(.001,2,80))),population_scope='Exact inspected pooled calibration+reference catalog, NOT unknown population; common law with finite 4 views.')

def main():
 fs=list(families());rows=[];start=time.time();n_families=len(fs)+2
 for task,fam,paths in fs:
  f=dict(np.load(paths['baseline_0' if task=='qa' else 'factual']));names=sorted(set(paths)-{'baseline_0' if task=='qa' else 'factual'});w=np.ones(4)/4 if task=='qa' else np.array([.325,.325,.175,.175])
  for pol in names:
   c=dict(np.load(paths[pol]));rows.append(analyze(task,fam,pol,c['calibration']-f['calibration'],c['reference']-f['reference'],w,len(names),.025/n_families))
  print(task,fam,'done',round(time.time()-start,1),flush=True)
 paths={p.parent.name:p for p in (R.parent/'oral_upgrade/artifacts/qa_generation').glob('*/predictions.npz')};f=dict(np.load(paths['baseline_0']));names=sorted(set(paths)-{'baseline_0'})
 for metric in ['f1','em']:
  for pol in names:
   c=dict(np.load(paths[pol]));rows.append(analyze('qa_generation_'+metric,'b1000_4101',pol,(1-c['calibration_'+metric])-(1-f['calibration_'+metric]),(1-c['reference_'+metric])-(1-f['reference_'+metric]),np.ones(4)/4,len(names),.025/n_families))
 summary={}
 for task in sorted({x['task'] for x in rows}):
  rr=[x for x in rows if x['task']==task]
  summary[task]=dict(records=len(rr),paired_negative=sum(x['paired_KL']<0 for x in rr),generic_same_audit_negative=sum(x['generic_same_audit']['upper']<0 for x in rr),generic_all_available_negative=sum(x['generic_all_available']['upper']<0 for x in rr),catalog_target_nonnegative=sum(x['catalog_robust_target']>=0 for x in rr),catalog_envelope_nonnegative=sum(x['catalog_envelope']>=0 for x in rr),median_paired_upper=float(np.median([x['paired_KL'] for x in rr])),median_generic_upper=float(np.median([x['generic_all_available']['upper'] for x in rr])),median_generic_width=float(np.median([x['generic_all_available']['radius'] for x in rr])))
 (R/'results/retrospective.json').write_text(json.dumps(dict(scope='Retrospective only; all inspected archived candidates retained. Delta .025 allocated across 24 families separately for each method. No test outputs used for recalibration. Generic both uses same available cal/audit anchors and no wasted cal split; same-audit column also reported.',summary=summary,rows=rows),indent=2));print(summary)
if __name__=='__main__':main()
