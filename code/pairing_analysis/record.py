"""Post-decision attribution: ONLY frozen pre-test predictions."""
import hashlib,json
from pathlib import Path
import numpy as np
from pairing_analysis.methods import certificate
R=Path(__file__).resolve().parent;A=R.parent/'sampled_interventions/results/prospective_artifacts'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
dec=A/'decision.json';h=sha(dec);assert h=='0f1eaf1b3c449426c434dd4f56f68da458b33a085b1b50c323dff9ca90ee6d4f';old=json.loads(dec.read_text());rows=[]
fpath=A/'runs/audit/baseline_0/predictions.npz';assert sha(fpath)==old['prediction_shas']['baseline_0'];fdata=np.load(fpath);f=1-np.concatenate([fdata['calibration_f1'],fdata['audit_f1']]).astype(float)
for rec in old['rows']:
 name=rec['policy'];cp=A/'runs/audit'/name/'predictions.npz';assert sha(cp)==old['prediction_shas'][name];cdata=np.load(cp);c=1-np.concatenate([cdata['calibration_f1'],cdata['audit_f1']]).astype(float)
 row=dict(policy=name,generic=rec['generic_upper'],original_paired_split=rec['paired_upper'])
 for separate in [False,True]:
  for split in [False,True]:
   key=('separate' if separate else 'paired')+('_split' if split else '_shared');out=certificate(f,c,m=5000 if split else None,J=4,separate=separate,regime='wor_enumerated');row[key]={k:float(v) for k,v in out.items()}
 row['endpoint_pairing_split']=row['separate_split']['upper']-row['paired_split']['upper'];row['endpoint_pairing_shared']=row['separate_shared']['upper']-row['paired_shared']['upper'];row['endpoint_reuse_separate']=row['separate_split']['upper']-row['separate_shared']['upper'];row['endpoint_reuse_paired']=row['paired_split']['upper']-row['paired_shared']['upper'];row['interaction']=row['endpoint_pairing_split']-row['endpoint_pairing_shared'];rows.append(row)
assert sha(dec)==h
(R/'results/record.json').write_text(json.dumps(dict(scope='POST-DECISION ablation, matched per-event ledger e=.025/(2*(4+1)); no sealed-test data; historical decision unchanged; generic saved historical grid ledger remains different',decision_sha=h,rows=rows),indent=2))
print(json.dumps(rows,indent=2))
