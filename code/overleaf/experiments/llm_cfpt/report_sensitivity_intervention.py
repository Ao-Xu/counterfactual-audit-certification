"""Analyze all frozen models; resample independent anchors, not sibling rows."""
import json, pathlib, hashlib, itertools
import numpy as np

O=pathlib.Path(__file__).resolve().parent/'sensitivity_intervention_v1'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def component(pred,labels):
    p=np.clip(np.asarray(pred,dtype=float),1e-12,1); p=p/p.sum(-1,keepdims=True)
    # arrays have seed, anchor, style, class dimensions
    correct=np.take_along_axis(p,labels[None,:,None,None]+np.zeros(p.shape[:-1]+(1,),dtype=int),axis=-1)[...,0]
    loss=1-correct; lp=np.log(p); mean=p.mean(2,keepdims=True)
    kl=(p[:,:,:,None,:]*(lp[:,:,:,None,:]-lp[:,:,None,:,:])).sum(-1).mean((2,3))
    info=(p*(lp-np.log(mean))).sum(-1).mean(2)
    return dict(loss=loss,kl=kl,info=info,v=loss.var(2),nll=-np.log(correct),accuracy=(p.argmax(-1)==labels[None,:,None]).astype(float))
def measures(c,idx=None):
    a={k:(v if idx is None else v[:,idx]) for k,v in c.items()}
    style=a['loss'].mean(1); ref=style.mean(1); signed=style-ref[:,None]
    return dict(KL=float(a['kl'].mean()),I=float(a['info'].mean()),V=float(a['v'].mean()),G=float(np.abs(signed).mean()),Rnu=float(ref.mean()),NLL=float(a['nll'].mean()),accuracy=float(a['accuracy'].mean()),G_per_seed=np.abs(signed).mean(1).tolist(),Rnu_per_seed=ref.tolist(),KL_per_seed=a['kl'].mean(1).tolist(),signed_gaps=signed.tolist())
def bootstrap_endpoints(lo,hi,labels,seed):
    rng=np.random.default_rng(seed); out={k:[] for k in ['KL','G','Rnu','NLL','accuracy']}
    strata=[np.flatnonzero(labels==v) for v in [0,1]]
    for _ in range(10000):
        idx=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]); a=measures(lo,idx); b=measures(hi,idx)
        for k in out: out[k].append(b[k]-a[k])
    return {k:dict(mean=float(np.mean(v)),ci95=np.quantile(v,[.025,.975]).tolist(),upper_one_sided95=float(np.quantile(v,.95))) for k,v in out.items()}
def main():
    P=read(O/'protocol.json'); lock=read(O/'analysis_lock.json'); plock=read(O/'protocol_lock.json')
    assert sha(O/'protocol.json')==plock['identity']['protocol']
    assert sha(O.parent/'sensitivity_intervention.py')==plock['identity']['script']
    data={k:read(O/'splits'/f'{k}.json') for k in ['train','development','calibration','test']}
    for a,b in itertools.combinations(data,2):
        assert not {v[0]['anchor_id'] for v in data[a]} & {v[0]['anchor_id'] for v in data[b]}
        assert not {' '.join(v[0]['premise'].casefold().split()) for v in data[a]} & {' '.join(v[0]['premise'].casefold().split()) for v in data[b]}
    labels={k:np.array([int(v[0]['y']==1) for v in rows]) for k,rows in data.items()}
    grouped={}; audit=[]
    for name,expected in lock['model_run_sha256'].items():
        assert sha(O/'runs'/name)==expected
        r=read(O/'runs'/name); test=read(O/'test_predictions'/name)
        assert test['test_sha256']==sha(O/'splits/test.json') and r['steps']==192 and r['sequence_count']==3072
        assert r['train_sha256']==sha(O/'splits/train.json')
        key='factual' if r['weight'] is None else f"lambda_{r['weight']:g}"
        r['test']=test['probabilities']; grouped.setdefault(key,[]).append(r)
        audit.append(dict(run=name,seed=r['seed'],weight=r['weight'],tokens=r['actual_tokens'],order_sha256=r['order_sha256']))
    assert len(audit)==15
    for seed in P['training_seeds']:
        rs=[a for a in audit if a['seed']==seed]; assert len({r['order_sha256'] for r in rs})==1
        aug=[a for a in rs if a['weight'] is not None]; assert len({r['tokens'] for r in aug})==1
    comps={}; metrics={}
    for key,rs in grouped.items():
        rs.sort(key=lambda r:r['seed']); assert [r['seed'] for r in rs]==P['training_seeds']
        comps[key]={}; metrics[key]={}
        for split in ['development','calibration','test']:
            p=np.array([r[split] for r in rs]); assert p.shape==(3,len(data[split]),4,2) and np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=1e-6)
            c=component(p,labels[split]); comps[key][split]=c; metrics[key][split]=measures(c)
            # Algebraic finite-pool validation only; not a calibration-derived population certificate.
            assert np.max(c['v']-.5*c['info'])<1e-9
            assert np.max(c['info']-c['kl'])<1e-9
            for s in range(3):
                means=c['loss'][s].mean(0); gap=np.abs(means-means.mean())
                assert gap.max()<=np.sqrt(3*c['v'][s].mean())+1e-9
    status=dict(stage='bootstrapping',completed_models=15)
    save(O/'analysis_progress.json',status)
    boots={split:bootstrap_endpoints(comps['lambda_0'][split],comps['lambda_2'][split],labels[split],928301+i) for i,split in enumerate(['calibration','test'])}
    cs=[metrics[f'lambda_{w:g}']['calibration'] for w in P['weights']]; ts=[metrics[f'lambda_{w:g}']['test'] for w in P['weights']]
    klreduction=1-cs[-1]['KL']/max(cs[0]['KL'],1e-12); greduction=1-ts[-1]['G']/max(ts[0]['G'],1e-12)
    checks=dict(calibration_KL_reduction20=bool(klreduction>=.2),calibration_KL_upper_negative=bool(boots['calibration']['KL']['upper_one_sided95']<0),test_G_reduction20=bool(greduction>=.2),test_G_upper_negative=bool(boots['test']['G']['upper_one_sided95']<0),calibration_KL_monotone=bool(np.all(np.diff([a['KL'] for a in cs])<=0)),test_G_monotone=bool(np.all(np.diff([a['G'] for a in ts])<=0)),KL_two_of_three_seeds=bool(sum(a<b for a,b in zip(cs[-1]['KL_per_seed'],cs[0]['KL_per_seed']))>=2),G_two_of_three_seeds=bool(sum(a<b for a,b in zip(ts[-1]['G_per_seed'],ts[0]['G_per_seed']))>=2),reference_bounded_noninferiority=bool(boots['test']['Rnu']['upper_one_sided95']<=.01),reference_NLL_guard=bool(ts[-1]['NLL']-ts[0]['NLL']<=.03),reference_accuracy_guard=bool(ts[-1]['accuracy']-ts[0]['accuracy']>=-.02))
    curves=[]
    for key in metrics:
        signed=np.array(metrics[key]['test']['signed_gaps'])
        for k in range(4):
            for delta in P['shifts']['delta']: curves.append(dict(condition=key,direction=k,delta=delta,chi2=3*delta**2,mean_signed_gap=float(delta*signed[:,k].mean()),mean_absolute_gap=float(np.abs(delta*signed[:,k]).mean())))
    result=dict(scope=P['scope'],all_acceptance_passed=all(checks.values()),acceptance_checks=checks,KL_relative_reduction=klreduction,G_relative_reduction=greduction,metrics=metrics,endpoint_bootstrap=boots,budget_audit=audit,split_counts={k:len(v) for k,v in data.items()},precision=lock,shift_curves=curves,statistical_scope=P['statistics'])
    save(O/'results.json',result)
    lines=['# Thm2：四风格 sensitivity intervention 实验','',f"**冻结验收标准：{'全部通过' if result['all_acceptance_passed'] else '未全部通过'}。** 这是自动筛查生成文本上的独立 anchor 实验，不是人工审核语义有效性的实验，也不是总体定理的经验性证明。",'',
    '## 协议与证据范围','',f"- 划分：{result['split_counts']}；anchor 与 premise 在四个划分之间不重叠。",'- 15 个模型：factual + augmentation-only（λ=0）+ augmentation+consistency（λ=0.1,0.5,2），各三个种子。',
    '- augmentation 条件共享全部四套改写、anchor 顺序、3072 次序列呈现、192 次更新；相同种子的实际 token 数也核对一致。Factual 用原文重复四次。',
    '- 训练目标为二元条件 next-token CE + λ × 平均 ordered pairwise predictive KL，包含对角项，共16项。不是全词表 SFT。',
    '- 主风险 Rν=平均 1−p(正确标签)，即二元随机化决策的期望0–1损失，范围[0,1]；NLL、argmax准确率单独报告。',
    '- G=四个预设方向上绝对总体经验风险差的平均；先对 anchor 求均值，再取绝对值，最后平均方向与种子。不能替换为平均逐anchor绝对差。',
    '- Test样本量仅依据 development 对比方差计算，在 test 生成/评估前冻结；calibration 不选模型，test 不选方向。',
    f"- 精度计算：未截断所需量 {lock['unclipped_precision_n']}，实际 {lock['test_anchors']}；800上限是否触发：{lock['cap_binding']}。这是signed contrast的精度启发式，不保证非线性G的区间宽度。",'',
    '## 主结果','', '| 条件 | Calibration KL ↓ | Calibration Vν ↓ | Test G ↓ | Test Rν ↓ | Test NLL ↓ | Test accuracy ↑ |','|---|---:|---:|---:|---:|---:|---:|']
    for key in ['factual']+[f'lambda_{w:g}' for w in P['weights']]:
        c=metrics[key]['calibration']; t=metrics[key]['test']; lines.append(f"| {key} | {c['KL']:.6f} | {c['V']:.6f} | {t['G']:.6f} | {t['Rnu']:.6f} | {t['NLL']:.6f} | {t['accuracy']:.2%} |")
    lines+=['','## 预设端点：λ=2 对 λ=0','',f"Calibration KL 相对下降：{klreduction:.2%}；test G 相对下降：{greduction:.2%}。",'', '| 指标差值（高权重减低权重） | 点估计 | 双侧95%区间 | 单侧95%上端 |','|---|---:|---:|---:|']
    for split,k in [('calibration','KL'),('test','G'),('test','Rnu'),('test','NLL'),('test','accuracy')]:
        b=boots[split][k]; lo=metrics['lambda_0'][split][k]; hi=metrics['lambda_2'][split][k]
        lines.append(f"| {split} {k} | {hi-lo:+.6f} | [{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}] | {b['upper_one_sided95']:+.6f} |")
    lines+=['','区间以标签分层的独立 anchor 为单位bootstrap10000次，四个风格及三个固定种子始终配对；每次重新计算非线性G。只覆盖固定训练池与模型条件下的评估抽样不确定性，不覆盖训练数据集变化。','','## 逐项验收','']
    for k,v in checks.items(): lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines+=['','## 逐种子端点','', '| seed | KL λ0 | KL λ2 | G λ0 | G λ2 | Rν λ0 | Rν λ2 |','|---|---:|---:|---:|---:|---:|---:|']
    for i,s in enumerate(P['training_seeds']): lines.append(f"| {s} | {cs[0]['KL_per_seed'][i]:.6f} | {cs[-1]['KL_per_seed'][i]:.6f} | {ts[0]['G_per_seed'][i]:.6f} | {ts[-1]['G_per_seed'][i]:.6f} | {ts[0]['Rnu_per_seed'][i]:.6f} | {ts[-1]['Rnu_per_seed'][i]:.6f} |")
    lines+=['','## 解释边界','',
    '- Calibration sensitivity 是估计诊断量，不是加上有限样本上置信修正的population certificate。有限池上验证 V≤I/2≤KL/2 与covered-shift界只是实现核查，不能升级成独立科学发现。',
    '- 风格shift曲线的线性由混合分布代数确定，主证据是跨训练干预、独立anchor的变化，而不是曲线随δ变大。',
    '- 自动等义筛查和生成器来自同系列模型，真实语义保持率未知；支持范围仅限自动筛查的有限生成风格机制。',
    '- 三个种子共享一个训练池；不能宣称跨数据集、模型家族或一般自然OOD的泛化。',
    '- 数据互斥核查针对本地项目记录；不能保证公开MultiNLI没有进入底座模型的预训练数据。',
    '- 主验收固定使用λ2，不事后选择更有利的中间权重；全部结果及失败项保留。','']
    (O/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    save(O/'analysis_progress.json',dict(stage='completed',acceptance=result['all_acceptance_passed']))
    print('\n'.join(lines))
if __name__=='__main__': main()
