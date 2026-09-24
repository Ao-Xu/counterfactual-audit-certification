import json
from pathlib import Path
import numpy as np
from oral_upgrade.certificate import audit_family,Corrections
R=Path(__file__).resolve().parent
def main():
    paths=R/'artifacts/qa_generation';names=['rgf_5000','rule_5000','rgf_15000','rule_15000'];f=np.load(paths/'baseline_0/predictions.npz');cs=[np.load(paths/p/'predictions.npz') for p in names];allrows=[]
    old=R/'artifacts/qa';tf=np.load(old/'b1000_4101_baseline_0/predictions.npz')
    for metric in ['f1','em','token']:
        if metric=='token':
            token_cs=[np.load(old/f'b1000_4101_{p}/predictions.npz') for p in names]
            inputs=[np.stack([c['calibration'] for c in token_cs]),tf['calibration'],np.stack([c['reference'] for c in token_cs]),tf['reference']]
        else:inputs=[np.stack([1-c['calibration_'+metric] for c in cs]),1-f['calibration_'+metric],np.stack([1-c['reference_'+metric] for c in cs]),1-f['reference_'+metric]]
        ans=audit_family(*inputs,cal_ids=['c'+str(i) for i in range(650)],audit_ids=['a'+str(i) for i in range(650)],rho=.05,delta=.025/3,weights=[.25]*4,policy_names=names,corrections=Corrections(0,0,evidence='Fixed gold-answer benchmark; same content, four formatting prompts'))
        for r in ans['rows']:
            tc=np.load(old/f'b1000_4101_{r["policy"]}/predictions.npz');r['test_delta']=float((tc['test']-tf['test']).mean()) if metric=='token' else float((tf['test_'+metric]-tc['test_'+metric]).mean());r['metric']='Token Brier' if metric=='token' else '1-'+metric.upper()
        allrows.append(ans)
    (R/'results/qa_alignment.json').write_text(json.dumps(dict(scope='New calibration/audit generation; old test already observed. Metric-alignment diagnostic, not prospective confirmation. Equal family correction across all three metrics.',audits=allrows),indent=2))
if __name__=='__main__':main()
