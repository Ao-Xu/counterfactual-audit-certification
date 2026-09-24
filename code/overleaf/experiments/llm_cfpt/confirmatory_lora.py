"""Locked paired LoRA confirmation; no truncation, no interim risk analysis."""
import argparse, gc, hashlib, json, pathlib, platform, time, traceback
import numpy as np
import torch
from scipy.stats import t, binomtest
from transformers import Trainer, TrainingArguments, AutoTokenizer
from directional_experiment import prompt, members, select_anchors, factual_indices, corrupt, LABELS
from train_and_eval import build_model, CompletionCollator

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'confirmatory_v1'
SEEDS=list(range(926201,926209))
CONDITIONS=['factual','clean','random','reinforce','oppose']
SPLIT_HASH='e1c5c59aeb9b1d6f836e7da8cf18937584d267c18c92a553f3e441b5f3cfc6c6'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8'); temp.replace(p)
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def splits():
    assert sha(OUT/'splits.json')==SPLIT_HASH
    return read(OUT/'splits.json')

def rows_for(base,seed,condition):
    rows=members(base); y=np.array([r['y'] for r in base])
    if condition=='factual':
        j=factual_indices(y,seed+1)
        rows=[dict(rows[2*i+j[i]]) for i in range(len(base)) for _ in range(2)]
    elif condition!='clean':
        yy=np.repeat(y,2); cues=np.tile([-1,1],len(y))
        targets=corrupt(yy,cues,.2,condition,seed+2)
        for r,target in zip(rows,targets): r['label']=LABELS[int(target==1)]
        for label in [-1,1]:
            assert np.sum((yy==label)&(targets!=yy))==int(.2*np.sum(yy==label))
    assert len(rows)==2*len(base)
    return rows

class CompleteDataset(torch.utils.data.Dataset):
    def __init__(self,rows,tok):
        self.items=[]
        for r in rows:
            p=prompt(r); a=tok.encode(p,add_special_tokens=True); b=tok.encode(p+' '+r['label'],add_special_tokens=True)
            assert b[:len(a)]==a and len(a)<len(b)<=256
            self.items.append(dict(input_ids=b,attention_mask=[1]*len(b),labels=[-100]*len(a)+b[len(a):]))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def evaluate(model,tok,rows):
    tok.padding_side='left'; ids=[tok.encode(' '+label,add_special_tokens=False)[0] for label in LABELS]
    assert len(set(ids))==2
    result=[]
    for start in range(0,len(rows),16):
        enc=tok([prompt(r) for r in rows[start:start+16]],padding=True,truncation=False,return_tensors='pt').to('cuda')
        assert enc.input_ids.shape[1]<=256
        with torch.inference_mode(): probs=torch.softmax(model(**enc,logits_to_keep=1).logits[:,-1,ids].float(),dim=-1).cpu().numpy()
        assert np.isfinite(probs).all()
        result.append(probs)
    return np.concatenate(result)

def train_one(base,seed,condition,eval_rows,folder,smoke=False):
    start=time.time(); rows=rows_for(base,seed,condition)
    model,tok=build_model(ROOT/'model_files_0p5b',seed)
    data=CompleteDataset(rows,tok)
    args=TrainingArguments(output_dir=str(folder),per_device_train_batch_size=2,gradient_accumulation_steps=4,
        num_train_epochs=1,max_steps=2 if smoke else -1,learning_rate=.0002,warmup_ratio=.05,
        lr_scheduler_type='cosine',bf16=True,gradient_checkpointing=True,save_strategy='no',report_to=[],
        disable_tqdm=True,logging_steps=25,seed=seed,data_seed=seed,remove_unused_columns=False,optim='adamw_torch')
    trainer=Trainer(model=model,args=args,train_dataset=data,data_collator=CompletionCollator(tok))
    fit=trainer.train(); model.eval()
    evaluations={name:evaluate(model,tok,members(rr)).tolist() for name,rr in eval_rows.items()}
    if not smoke: model.save_pretrained(folder,safe_serialization=True)
    result=dict(status='completed',seed=seed,condition=condition,train_anchor_ids=[r['anchor_id'] for r in base],
        train_rows=rows,train_loss=float(fit.training_loss),global_steps=int(trainer.state.global_step),
        evaluation=evaluations,seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    del trainer,model,tok,data; gc.collect(); torch.cuda.empty_cache()
    return result

def interval(x):
    x=np.asarray(x,float); se=x.std(ddof=1)/np.sqrt(len(x)); h=t.ppf(.975,len(x)-1)*se
    return dict(mean=float(x.mean()),sd=float(x.std(ddof=1)),low=float(x.mean()-h),high=float(x.mean()+h),n=len(x))

def analyze():
    data=splits(); y=np.array([r['y'] for r in data['test']]); weights=np.where(np.array([-1,1])[None,:]==y[:,None],.05,.95)
    risks={}; per_anchor={}; rows=[]
    for seed in SEEDS:
        for condition in CONDITIONS:
            r=read(OUT/'lora'/f'{seed}_{condition}.json'); assert r['status']=='completed' and r['global_steps']==100
            p=np.array(r['evaluation']['test']).reshape(len(y),2,2)
            correct=np.take_along_axis(p,((y+1)//2)[:,None,None].repeat(2,axis=1),axis=2)[:,:,0]
            a=np.sum(weights*(1-correct),axis=1); risks[seed,condition]=float(a.mean()); per_anchor[seed,condition]=a
            rows.append(dict(seed=seed,condition=condition,action_risk=float(a.mean()),seconds=r['seconds']))
    direction=np.array([risks[s,'reinforce']-risks[s,'oppose'] for s in SEEDS])
    clean=np.array([risks[s,'clean']-risks[s,'factual'] for s in SEEDS])
    contrasts={name:interval([risks[s,a]-risks[s,b] for s in SEEDS]) for name,a,b in [
        ('direction','reinforce','oppose'),('clean','clean','factual'),('reinforce_random','reinforce','random'),('oppose_random','oppose','random')]}
    # Secondary two-level bootstrap: paired outer pools and shared test anchor blocks.
    rng=np.random.default_rng(926901); mat=np.array([per_anchor[s,'reinforce']-per_anchor[s,'oppose'] for s in SEEDS]); draws=[]
    for _ in range(2000):
        ii=rng.integers(len(SEEDS),size=len(SEEDS)); jj=rng.integers(len(y),size=len(y)); draws.append(float(mat[ii][:,jj].mean()))
    return dict(completed_runs=len(rows),planned_runs=40,rows=rows,contrasts=contrasts,
        direction_values=direction.tolist(),clean_values=clean.tolist(),
        direction_sign_test_p=float(binomtest(int(np.sum(direction>0)),len(direction),.5).pvalue),
        direction_two_level_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),
        direction_supported=contrasts['direction']['low']>0,
        scope='Primary t interval conditional on fixed calibration/test pools; approximate outer-pool mean inference. Bootstrap is secondary, not an exact certificate.')

def source_files():
    return [pathlib.Path(__file__),ROOT/'directional_experiment.py',ROOT/'directional_protocol.json',ROOT/'train_and_eval.py']
def check_lock():
    lock=read(OUT/'lora_lock.json')
    for p in source_files(): assert sha(p)==lock['code_hashes'][p.name], 'Code changed after lock'
    assert sha(OUT/'splits.json')==lock['splits_sha256']==SPLIT_HASH

def main(mode):
    torch.set_num_threads(4)
    if mode=='smoke':
        data=splits(); tok=AutoTokenizer.from_pretrained(ROOT/'model_files_0p5b',local_files_only=True)
        for seed in SEEDS:
            ix=select_anchors(np.array([r['y'] for r in data['train']]),400,seed); base=[data['train'][i] for i in ix]
            for condition in CONDITIONS: assert len(CompleteDataset(rows_for(base,seed,condition),tok))==800
        dev=data['development']; ix=select_anchors(np.array([r['y'] for r in dev]),40,926000); base=[dev[i] for i in ix]
        r=train_one(base,926000,'clean',{'development':dev[:16]},OUT/'smoke_checkpoint',True)
        write(OUT/'lora_smoke.json',dict(status='passed',seconds=r['seconds'],peak_cuda_bytes=r['peak_cuda_bytes'],
            global_steps=r['global_steps'],validated_formal_conditions=40,development_only=True))
        print('SMOKE PASSED',round(r['seconds'],1),flush=True)
    elif mode=='lock':
        assert read(OUT/'lora_smoke.json')['status']=='passed'
        assert not (OUT/'lora_lock.json').exists()
        write(OUT/'lora_lock.json',dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            code_hashes={p.name:sha(p) for p in source_files()},splits_sha256=SPLIT_HASH,seeds=SEEDS,conditions=CONDITIONS,
            primary='mean paired action-risk reinforce minus oppose; pool-level 95% t interval',
            stop='40 runs; no interim risk analysis or test-based changes',python=platform.python_version(),torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0),smoke_sha256=sha(OUT/'lora_smoke.json')))
    elif mode=='run':
        check_lock(); data=splits(); y=np.array([r['y'] for r in data['train']])
        for seed in SEEDS:
            ix=select_anchors(y,400,seed); base=[data['train'][i] for i in ix]
            for condition in CONDITIONS:
                check_lock(); path=OUT/'lora'/f'{seed}_{condition}.json'
                if path.exists():
                    assert read(path)['status']=='completed', 'Failed run requires explicit recovery; never hide it'
                    continue
                write(OUT/'lora_progress.json',dict(status='running',seed=seed,condition=condition,planned=40))
                print('START',seed,condition,flush=True)
                try:
                    r=train_one(base,seed,condition,{k:data[k] for k in ['calibration','test']},OUT/'lora_checkpoints'/f'{seed}_{condition}')
                    r['lock_sha256']=sha(OUT/'lora_lock.json'); write(path,r)
                    print('DONE',seed,condition,round(r['seconds'],1),flush=True)
                except Exception:
                    write(path,dict(status='failed',seed=seed,condition=condition,error=traceback.format_exc()))
                    raise
        write(OUT/'lora_metrics.json',analyze())
        write(OUT/'lora_progress.json',dict(status='completed',completed=40,planned=40))
    else: raise ValueError(mode)
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['smoke','lock','run']); main(parser.parse_args().mode)
