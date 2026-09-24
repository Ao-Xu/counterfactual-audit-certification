"""Prospectively frozen development selection and crossed-pool confirmation."""
import pathlib,json,time,gc,math,hashlib,traceback
import numpy as np
import torch
from transformers import set_seed
from peft import PeftModel
from scipy.stats import t,norm as normal
from train_and_eval import build_model
import sensitivity_intervention as S
from report_sensitivity_intervention import component

R=pathlib.Path(__file__).resolve().parent; O=R/'crossed_consistency_v4'
P=dict(split_seed=931001,candidate_caps={**{f'devtrain{i}':2000 for i in range(2)},'development':1200,**{f'train{i}':2000 for i in range(8)},'calibration':1600,'test':3200},
       dev_pools=2,confirm_pools=8,train_anchors=400,development_anchors=300,calibration_anchors=400,test_anchors=800,
       dev_seeds=[931101,931102],confirm_seeds=list(range(931201,931209)),dev_weights=[.5,2.],
       selection='Among fixed .5,2 weights with development mean KL at least20% below W0, Rnu<=U0+.01, NLL<=U0+.03, accuracy>=U0-.02, select smallest KL; ties smaller weight. Require KL decrease in both development pools. No G-based selection. If none, stop without opening confirmation.',
       low_weight={'0.5':.1,'2.0':.5},conditions=['U0','Ustar','W0','Wlow','Wstar'],
       train='Qwen0.5 LoRA r8 alpha16 q/v, dropout.05; binary conditional CE;192 updates,3072 sequences;lr2e-4 cosine warmup10;all four styles same tokens and anchor order within pool;weighted CE=(1+.9*y*a)/4,a=(1,-1,-1,1);uniform ordered16pair KL penalty;no cue token',
       gates='ALL three: calibration KL(Wstar-W0) upper95<0; test G_style(Wstar-W0) upper95<0; test Rnu(Wstar-U0) upper95<.01. Directional superiority only, no claimed minimum benefit. Dose response descriptive secondary, not required.',
       statistics='10000 crossed bootstrap draws: resample8 paired training realizations and label-stratified anchors independently; retain all conditions/styles. Each pool computes abs after anchor averaging; then average pools. Upper bound=point+t7/.normal95 * max(0,q95(bootstrap-point)), further max with paired-pool t upper at fixed evaluation set. This small-pool guard is approximate, not exact finite-sample coverage. Primary intervals cannot be changed after outcomes.',
       simulation='Before new empirical outcomes: 200 synthetic null datasets,499 crossed draws each,8pools80anchors,scalar and nonzero/zero-centered absolute-gap cases. Record empirical false-positive rates. If any >.10, stop before empirical execution; simulation is implementation diagnostic not proof of coverage.',
       bootstrap_seed=931401,bootstrap_draws=10000,ni_margin=.01,
       scope='Automatically screened hypothesis rewrites; no human label-preservation audit. New protocol informed by prior development findings; no causal mediation or theorem model-ranking claim. Train pools sampled without replacement from finite local eligible pool, not exactly independent population draws; inference approximates training-realization sampling.',
       no_peeking='All historical anchors/premises excluded. Old replication600 only pilot history. No confirmatory G selection, no outcome-based sample extension, no discarded failed seeds. Confirm runs use no development evaluations; calibration/test generated after all40 model fits frozen.')
S.O=O; S.P=P; S.__file__=__file__
read=S.read; write=S.write; digest=S.digest
A=np.array([1,-1,-1,1.])
def status(stage,**kw): S.O=O; S.status(stage,**kw)
def lock():
    dest=O/'protocol.json'
    if dest.exists(): assert read(dest)==P
    else: write(dest,P)
    paths=[pathlib.Path(__file__),R/'sensitivity_intervention.py',R/'train_and_eval.py',R/'natural_rewrite_development.py',R/'report_sensitivity_intervention.py',dest]
    for folder in ['sensitivity_intervention_v1','shortcut_intervention_v2','shortcut_replication_v3']:
        paths += [p for p in (R/folder).rglob('*.json') if not any(z in ['generation_batches','screen_batches'] for z in p.relative_to(R/folder).parts)]
    identity={str(p.relative_to(R)):digest(p) for p in paths}
    target=O/'source_lock.json'
    if target.exists(): assert read(target)['hashes']==identity
    else: write(target,dict(hashes=identity,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))

def upper(point,draws,pool_values):
    factor=t.ppf(.95,len(pool_values)-1)/normal.ppf(.95)
    crossed=point+factor*max(0,float(np.quantile(np.asarray(draws)-point,.95)))
    pool_t=point+t.ppf(.95,len(pool_values)-1)*np.std(pool_values,ddof=1)/np.sqrt(len(pool_values))
    return float(max(crossed,pool_t))

def simulation():
    dest=O/'inference_simulation.json'
    if dest.exists(): assert read(dest)['passed']; return
    rng=np.random.default_rng(931501); errors={k:0 for k in ['scalar','absolute_zero','absolute_nonzero']}
    for rep in range(200):
        pi=rng.integers(0,8,(499,8)); ai=rng.integers(0,80,(499,80))
        for kind in errors:
            x=rng.normal(0,.01,(8,1))+rng.normal(0,.02,(1,80))+rng.normal(0,.04,(8,80))
            if kind=='scalar':
                pv=x.mean(1); point=pv.mean(); draws=x[pi[:,:,None],ai[:,None,:]].mean((1,2))
            else:
                y=rng.normal(0,.01,(8,1))+rng.normal(0,.02,(1,80))+rng.normal(0,.04,(8,80))
                offset=.04 if kind=='absolute_nonzero' else 0
                x=x+offset; y=y+offset
                pv=np.abs(x.mean(1))-np.abs(y.mean(1)); point=pv.mean()
                draws=(np.abs(x[pi[:,:,None],ai[:,None,:]].mean(2))-np.abs(y[pi[:,:,None],ai[:,None,:]].mean(2))).mean(1)
            errors[kind]+=int(upper(point,draws,pv)<0)
    rates={k:v/200 for k,v in errors.items()}; passed=max(rates.values())<=.10
    write(dest,dict(passed=passed,null_repetitions=200,bootstrap_draws=499,false_positive_rates=rates,scope='Approximate finite simulation diagnostic, not validation of all distributions or exact alpha'))
    assert passed, 'Inference simulation failed; stop before empirical outcomes'

def tag(stage,pool,cond): return f'{stage}{pool}_{cond}'
def fit(stage,pool,cond,weight):
    name=tag(stage,pool,cond); dest=O/'runs'/f'{name}.json'
    if dest.exists(): assert read(dest)['status']=='completed'; return
    split=f'devtrain{pool}' if stage=='dev' else f'train{pool}'
    groups=read(O/'splits'/f'{split}.json'); seed=P['dev_seeds' if stage=='dev' else 'confirm_seeds'][pool]
    model,tok=build_model(R/'model_files_0p5b',seed);tok.pad_token=tok.eos_token;model.gradient_checkpointing_enable();model.train()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.01)
    rng=np.random.default_rng(seed);order=[]
    while len(order)<768: order+=rng.permutation(len(groups)).tolist()
    order=order[:768]; logs=[]; tokens=0; begin=time.time(); set_seed(seed)
    for step in range(192):
        lr=2e-4*min((step+1)/10,1)*(.5*(1+math.cos(math.pi*max(0,step-9)/182)))
        for pg in opt.param_groups: pg['lr']=lr
        opt.zero_grad(set_to_none=True); ceval=klval=0.
        for i in order[4*step:4*step+4]:
            lp,nt=S.probs(model,tok,[groups[i]],grad=True);y=groups[i][0]['y'];target=int(y==1)
            w=torch.tensor((1+.9*y*A)/4 if cond.startswith('W') else np.full(4,.25),device='cuda',dtype=lp.dtype)
            ce=-(lp[0,:,target]*w).sum();kl=S.penalty(lp);((ce+weight*kl)/4).backward()
            ceval+=float(ce.detach())/4;klval+=float(kl.detach())/4;tokens+=nt
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();logs.append(dict(step=step+1,ce=ceval,kl=klval,lr=lr))
        if (step+1)%32==0:status('training',run=name,step=step+1,total=192)
    ck=O/'checkpoints'/name;model.save_pretrained(ck,safe_serialization=True)
    devp=S.eval_groups(model,tok,read(O/'splits/development.json')).tolist() if stage=='dev' else None
    write(dest,dict(status='completed',stage=stage,pool=pool,condition=cond,weight=weight,seed=seed,steps=192,sequence_count=3072,actual_tokens=tokens,order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),train_sha256=digest(O/'splits'/f'{split}.json'),development=devp,losses=logs,seconds=time.time()-begin,adapter_sha256=digest(ck/'adapter_model.safetensors')))
    del model,tok,opt;gc.collect();torch.cuda.empty_cache();status('run_completed',run=name)

def metrics(c,pi=None,ai=None):
    v={k:(x if pi is None else x[pi][:,ai]) for k,x in c.items()}
    risks=v['loss'].mean(1);ref=risks.mean(1)
    return dict(KL=v['kl'].mean(1),V=v['v'].mean(1),G=np.abs(risks-ref[:,None]).mean(1),Rnu=ref,NLL=v['nll'].mean((1,2)),accuracy=v['accuracy'].mean((1,2)),worst=risks.max(1),**{f'Rstyle{k}':risks[:,k] for k in range(4)})

def select():
    dest=O/'selection_lock.json'
    if dest.exists():return read(dest)
    labels=np.array([int(g[0]['y']==1) for g in read(O/'splits/development.json')]);out={}
    for cond in ['U0','W0','W0.5','W2']:
        pp=np.array([read(O/'runs'/f'{tag("dev",i,cond)}.json')['development'] for i in range(2)])
        out[cond]={k:v.tolist() for k,v in metrics(component(pp,labels)).items()}
    avg=lambda c,k:float(np.mean(out[c][k]));eligible=[];checks={}
    for w in P['dev_weights']:
        c=f'W{w:g}';q=dict(KL20=avg(c,'KL')<=.8*avg('W0','KL'),KL_both=all(a<b for a,b in zip(out[c]['KL'],out['W0']['KL'])),risk=avg(c,'Rnu')<=avg('U0','Rnu')+.01,NLL=avg(c,'NLL')<=avg('U0','NLL')+.03,accuracy=avg(c,'accuracy')>=avg('U0','accuracy')-.02)
        checks[c]=q
        if all(q.values()):eligible.append(w)
    chosen=min(eligible,key=lambda w:(avg(f'W{w:g}','KL'),w)) if eligible else None
    result=dict(passed=chosen is not None,star=chosen,low=P['low_weight'][str(chosen)] if chosen is not None else None,checks=checks,development_metrics=out,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),run_hashes={p.name:digest(p) for p in (O/'runs').glob('dev*.json')})
    write(dest,result);status('selection_frozen',passed=result['passed'],star=chosen);return result

def evaluate(split):
    groups=read(O/'splits'/f'{split}.json')
    for i in range(8):
        for cond in P['conditions']:
            name=tag('confirm',i,cond);dest=O/'predictions'/split/f'{name}.json'
            if dest.exists():assert read(dest)['split_sha256']==digest(O/'splits'/f'{split}.json');continue
            run=read(O/'runs'/f'{name}.json');ck=O/'checkpoints'/name
            assert digest(ck/'adapter_model.safetensors')==run['adapter_sha256']
            model,tok=S.base('model_files_0p5b');model=PeftModel.from_pretrained(model,ck).eval();pp=S.eval_groups(model,tok,groups)
            write(dest,dict(probabilities=pp.tolist(),split_sha256=digest(O/'splits'/f'{split}.json'),adapter_sha256=run['adapter_sha256']))
            del model,tok;gc.collect();torch.cuda.empty_cache();status('evaluated',split=split,run=name)

def analyze():
    result=dict(metrics={},contrasts={},scope=P['scope'],statistics=P['statistics']);rng=np.random.default_rng(P['bootstrap_seed'])
    for split in ['calibration','test']:
        labels=np.array([int(g[0]['y']==1) for g in read(O/'splits'/f'{split}.json')]);comps={};point={}
        for cond in P['conditions']:
            pp=np.array([read(O/'predictions'/split/f'{tag("confirm",i,cond)}.json')['probabilities'] for i in range(8)])
            comps[cond]=component(pp,labels);point[cond]=metrics(comps[cond])
        result['metrics'][split]={c:{k:dict(mean=float(v.mean()),per_pool=v.tolist()) for k,v in m.items()} for c,m in point.items()}
        pairs={'Wstar-W0':('Wstar','W0'),'Wstar-U0':('Wstar','U0'),'Ustar-U0':('Ustar','U0')}
        draws={p:{k:[] for k in point['W0']} for p in pairs};strata=[np.flatnonzero(labels==y) for y in [0,1]]
        for b in range(P['bootstrap_draws']):
            pi=rng.integers(0,8,8);ai=np.concatenate([rng.choice(z,len(z),replace=True) for z in strata]);m={c:metrics(comps[c],pi,ai) for c in ['Wstar','W0','U0','Ustar']}
            for pair,(hi,lo) in pairs.items():
                for k in draws[pair]:draws[pair][k].append(float((m[hi][k]-m[lo][k]).mean()))
            if (b+1)%2000==0:status('bootstrap',split=split,draws=b+1)
        result['contrasts'][split]={}
        for pair,(hi,lo) in pairs.items():
            result['contrasts'][split][pair]={}
            for k,bs in draws[pair].items():
                pv=point[hi][k]-point[lo][k];d=float(pv.mean())
                result['contrasts'][split][pair][k]=dict(difference=d,upper95=upper(d,bs,pv),percentile95=np.quantile(bs,[.025,.975]).tolist(),per_pool=pv.tolist())
    c=result['contrasts'];checks=dict(calibration_KL=c['calibration']['Wstar-W0']['KL']['upper95']<0,test_G=c['test']['Wstar-W0']['G']['upper95']<0,reference_NI_vs_U0=c['test']['Wstar-U0']['Rnu']['upper95']<.01)
    result.update(checks=checks,all_primary_passed=all(checks.values()),secondary_scope='Dose response and auxiliary metrics descriptive, not individually confirmatory; signed controlled analysis retained in raw probabilities')
    write(O/'results.json',result)
    lines=['# Crossed consistency v4','',f"主验收：{'全部通过' if all(checks.values()) else '未全部通过'}",'',str(checks),'','8个训练池×5条件；新calibration400、新test800；自动筛查，不是人工确认语义保持。','', '| 条件 | Test KL | Test V | Test G | Test Rnu | Test NLL | Test accuracy | Test worst style |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for cond in P['conditions']:
        m=result['metrics']['test'][cond];lines.append('| '+cond+' | '+' | '.join(f"{m[k]['mean']:.6f}" for k in ['KL','V','G','Rnu','NLL','accuracy','worst'])+' |')
    lines+=['','主对比及保护基线的完整区间见results.json。推断采用冻结的crossed bootstrap和小池数t保护，仍是近似推断，不宣称有限样本精确覆盖。剂量曲线仅为预注册辅助证据，不由端点成功推导单调性。']
    (O/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    lock();simulation();S.prepare()
    S.make_split('development',300)
    for i in range(2):
        S.make_split(f'devtrain{i}',400)
        for cond,w in [('U0',0),('W0',0),('W0.5',.5),('W2',2)]:fit('dev',i,cond,w)
    selection=select()
    if not selection['passed']:
        status('completed_development_gate_failed',runs=8,confirmation_opened=False);return
    for i in range(8):
        S.make_split(f'train{i}',400)
        for cond,w in [('U0',0),('Ustar',selection['star']),('W0',0),('Wlow',selection['low']),('Wstar',selection['star'])]:fit('confirm',i,cond,w)
    frozen={p.name:digest(p) for p in (O/'runs').glob('confirm*.json')};assert len(frozen)==40
    dest=O/'models_lock.json'
    if dest.exists():assert read(dest)==frozen
    else:write(dest,frozen)
    for split,n in [('calibration',400),('test',800)]:S.make_split(split,n);evaluate(split)
    analyze();lock();status('completed',development_runs=8,confirmation_runs=40)

if __name__=='__main__':
    torch.set_num_threads(4)
    try:main()
    except Exception:status('failed',error=traceback.format_exc());raise
