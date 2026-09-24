"""Frozen four-style sensitivity intervention. Resume-safe split generation and training."""
import argparse, pathlib, json, hashlib, time, gc, re, math, traceback
from collections import Counter
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from train_and_eval import build_model
from natural_rewrite_development import norm, neg, prompt, eqprompt, yes_scores

R=pathlib.Path(__file__).resolve().parent; O=R/'sensitivity_intervention_v1'
P=json.loads((O/'protocol.json').read_text(encoding='utf-8'))
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(p)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def status(stage,**kw):
    v=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    write(O/'progress.json',v); print(v,flush=True)
def release(model):
    del model; gc.collect(); torch.cuda.empty_cache()
def base(name):
    tok=AutoTokenizer.from_pretrained(R/name,local_files_only=True); tok.pad_token=tok.eos_token; tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(R/name,local_files_only=True,dtype=torch.bfloat16).to('cuda').eval()
    return model,tok
def prepare():
    lock=O/'protocol_lock.json'
    identity=dict(protocol=digest(O/'protocol.json'),script=digest(pathlib.Path(__file__)),helpers={n:digest(R/n) for n in ['train_and_eval.py','natural_rewrite_development.py']})
    if lock.exists(): assert read(lock)['identity']==identity
    else: write(lock,dict(identity=identity,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    if (O/'candidates.json').exists(): return
    legacy=read(R/'confirmatory_v1/legacy_exclusions.json'); ids=set(legacy['ids']); premises=set(legacy['premises'])
    def visit(v):
        if isinstance(v,dict):
            if isinstance(v.get('premise'),str): premises.add(norm(v['premise']))
            for k in ['anchor_id','pairID','pair_id']:
                if isinstance(v.get(k),(str,int)): ids.add(str(v[k]))
            for q in v.values():
                if isinstance(q,(dict,list)): visit(q)
        elif isinstance(v,list):
            for q in v: visit(q)
    scanned=[]
    for p in R.rglob('*'):
        if p.suffix not in ['.json','.jsonl'] or O in p.parents or any(z.startswith('model_') for z in p.relative_to(R).parts): continue
        if p.stat().st_mtime<(R/'confirmatory_v1/legacy_exclusions.json').stat().st_mtime: continue
        visit([json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s] if p.suffix=='.jsonl' else read(p)); scanned.append(str(p.relative_to(R)))
    cache=pathlib.Path.home()/'.cache/huggingface/datasets/multi_nli/default/0.0.0/da70db2af9d09693783c3320c4249840212ee221/multi_nli-train.arrow'
    ds=Dataset.from_file(str(cache)); labels=np.array(ds['label']); rng=np.random.default_rng(P['split_seed']); tok=AutoTokenizer.from_pretrained(R/'model_files_0p5b',local_files_only=True)
    pools={v:iter(rng.permutation(np.flatnonzero(labels==v))) for v in [0,2]}; splits={}
    for name,cap in P['candidate_caps'].items():
        rows=[]
        for v in [0,2]:
            selected=[]
            for ix in pools[v]:
                r=ds[int(ix)]
                if str(r['pairID']) in ids or norm(r['premise']) in premises: continue
                if not 6<=len(r['hypothesis'].split())<=24: continue
                row=dict(anchor_id=str(r['pairID']),premise=r['premise'],original=r['hypothesis'],hypothesis=r['hypothesis'],y=1 if v==0 else -1,source_genre=r['genre'])
                if len(tok.encode(prompt(row)))>200: continue
                selected.append(row); ids.add(row['anchor_id']); premises.add(norm(row['premise']))
                if len(selected)==cap//2: break
            assert len(selected)==cap//2; rows+=selected
        rng.shuffle(rows); splits[name]=rows
    write(O/'candidates.json',splits); write(O/'exclusion_audit.json',dict(scanned=scanned,scope='Local-project overlap only, not pretraining audit',counts={k:len(v) for k,v in splits.items()}))
    status('prepared')
def generation_prompt(h,j):
    style=['formal written English','natural conversational English without fillers','concise English, retaining every fact','more fully spelled-out wording, without adding facts, implications, examples, or explanations'][j]
    return f'Rewrite this sentence in {style}. Preserve EXACT meaning, people, numbers, negation, quantifiers, tense and certainty. Return only ONE sentence, without commentary or quotation marks.\nSentence: {h}'
def make_split(name,target):
    dest=O/'splits'/f'{name}.json'
    if dest.exists(): assert len(read(dest))==target; return
    candidates=read(O/'candidates.json')[name]; accepted=[]
    for start in range(0,len(candidates),128):
        checkpoint=O/'screen_batches'/f'{name}_{start:04d}.json'
        if checkpoint.exists(): rows=read(checkpoint)
        else:
            genpath=O/'generation_batches'/f'{name}_{start:04d}.json'
            if genpath.exists(): rows=read(genpath)
            else:
                jobs=[dict(r,style=j) for r in candidates[start:start+128] for j in range(4)]; rows=[]
                model,tok=base('model_files')
                for off in range(0,len(jobs),24):
                    batch=jobs[off:off+24]; ps=[tok.apply_chat_template([{'role':'user','content':generation_prompt(r['original'],r['style'])}],tokenize=False,add_generation_prompt=True) for r in batch]
                    enc=tok(ps,padding=True,truncation=False,return_tensors='pt').to('cuda')
                    with torch.inference_mode(): seqs=model.generate(**enc,max_new_tokens=64,do_sample=False,pad_token_id=tok.pad_token_id)
                    for r,seq in zip(batch,seqs):
                        seq=seq[enc.input_ids.shape[1]:].tolist(); raw=tok.decode(seq,skip_special_tokens=True).strip()
                        rows.append(dict(r,hypothesis=raw.strip('"\' '),raw_generation=raw,truncated=tok.eos_token_id not in seq and len(seq)>=64))
                    status('generating',split=name,candidate_start=start,batch_edits=len(rows),batch_total=len(jobs),accepted=len(accepted),target=target)
                write(genpath,rows); del model,tok; gc.collect(); torch.cuda.empty_cache()
            model,tok=base('model_files_0p5b')
            texts=[eqprompt(a,b) for r in rows for a,b in [(r['original'],r['hypothesis']),(r['hypothesis'],r['original'])]]
            scores=yes_scores(model,tok,texts)
            for i,r in enumerate(rows):
                h=r['hypothesis']; flags=[]
                if r['truncated'] or '\n' in h or not 4<=len(h.split())<=50 or any(z in h.lower() for z in ['here is','rewritten sentence','entailment','contradiction']): flags.append('format')
                if neg(h)!=neg(r['original']): flags.append('negation')
                if sorted(re.findall(r'\d+',h))!=sorted(re.findall(r'\d+',r['original'])): flags.append('numbers')
                if len(tok.encode(prompt(r)))>256: flags.append('length')
                r['equivalence_yes']=scores[2*i:2*i+2]
                if min(r['equivalence_yes'])<.5: flags.append('automatic_equivalence')
                r['rejection_reasons']=flags
            write(checkpoint,rows); del model,tok; gc.collect(); torch.cuda.empty_cache()
        groups=[rows[i:i+4] for i in range(0,len(rows),4)]
        accepted += [v for v in groups if len(v)==4 and all(not r['rejection_reasons'] for r in v) and len({norm(r['hypothesis']) for r in v})>=2]
        counts={str(y):sum(v[0]['y']==y for v in accepted) for y in [-1,1]}; status('screened_batch',split=name,eligible=counts,target=target)
        if min(counts.values())>=target//2:
            selected=[]
            for y in [-1,1]: selected += [v for v in accepted if v[0]['y']==y][:target//2]
            np.random.default_rng(928010).shuffle(selected); write(dest,selected); return
    raise RuntimeError(f'candidate_cap_exhausted: {name}, target={target}, counts={counts}; do not relax filters')
def probs(model,tok,groups,factual=False,grad=False):
    rows=[dict(r,hypothesis=r['original']) if factual else r for v in groups for r in v]
    tok.padding_side='left'; ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in ['contradiction','entailment']]; assert len(set(ids))==2
    enc=tok([prompt(r) for r in rows],padding=True,truncation=False,return_tensors='pt').to('cuda'); assert enc.input_ids.shape[1]<=256
    logits=model(**enc,logits_to_keep=1).logits[:,-1,ids].float()
    return logits.log_softmax(-1).reshape(len(groups),4,2),int(enc.attention_mask.sum())
def penalty(logp):
    p=logp.exp(); return (p[:,:,None,:]*(logp[:,:,None,:]-logp[:,None,:,:])).sum(-1).mean()
def eval_groups(model,tok,groups):
    model.eval(); out=[]
    with torch.inference_mode():
        for start in range(0,len(groups),4):
            lp,_=probs(model,tok,groups[start:start+4]); out.extend(lp.exp().cpu().tolist())
    return np.array(out)
def run_name(seed,weight): return f'{seed}_'+('factual' if weight is None else f'lambda_{weight:g}')
def train_one(seed,weight):
    name=run_name(seed,weight); done=O/'runs'/f'{name}.json'
    if done.exists(): assert read(done)['status']=='completed'; return
    groups=read(O/'splits/train.json'); dev=read(O/'splits/development.json'); cal=read(O/'splits/calibration.json')
    model,tok=build_model(R/'model_files_0p5b',seed); tok.pad_token=tok.eos_token; model.gradient_checkpointing_enable(); model.train()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.01)
    rng=np.random.default_rng(seed); order=[]
    while len(order)<192*4: order+=rng.permutation(len(groups)).tolist()
    order=order[:192*4]; losses=[]; tokens=0; begin=time.time(); set_seed(seed)
    for step in range(192):
        lr=2e-4*min((step+1)/10,1)*(.5*(1+math.cos(math.pi*max(0,step-9)/182)))
        for pg in opt.param_groups: pg['lr']=lr
        opt.zero_grad(set_to_none=True); ceval=klval=0.
        for i in order[4*step:4*step+4]:
            lp,nt=probs(model,tok,[groups[i]],factual=weight is None,grad=True); target=int(groups[i][0]['y']==1)
            ce=-lp[:,:,target].mean(); kl=penalty(lp); loss=ce+(0 if weight is None else weight)*kl
            (loss/4).backward(); ceval+=float(ce.detach())/4; klval+=float(kl.detach())/4; tokens+=nt
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(dict(step=step+1,ce=ceval,kl=klval,lr=lr))
        if (step+1)%32==0: status('training',run=name,step=step+1,total=192,ce=ceval,kl=klval)
    folder=O/'checkpoints'/name; model.save_pretrained(folder,safe_serialization=True)
    devp=eval_groups(model,tok,dev); calp=eval_groups(model,tok,cal)
    write(done,dict(status='completed',seed=seed,weight=weight,steps=192,sequence_count=192*16,actual_tokens=tokens,order=order,order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),train_sha256=digest(O/'splits/train.json'),losses=losses,development=devp.tolist(),calibration=calp.tolist(),seconds=time.time()-begin))
    del model,tok,opt; gc.collect(); torch.cuda.empty_cache(); status('run_completed',run=name)
def losses(p,groups):
    y=np.array([int(v[0]['y']==1) for v in groups]); return 1-p[np.arange(len(y))[:,None],np.arange(4)[None,:],y[:,None]]
def freeze_test():
    dest=O/'analysis_lock.json'
    if dest.exists(): return read(dest)['test_anchors']
    dev=read(O/'splits/development.json'); rs=[read(p) for p in sorted((O/'runs').glob('*.json'))]; assert len(rs)==15
    centered=[]
    for r in rs:
        e=losses(np.array(r['development']),dev); centered.append(e-e.mean(1,keepdims=True))
    sd=max(float(np.std(a-b,axis=0,ddof=1).max()) for a in centered for b in centered)
    raw=math.ceil((1.96*sd/.01)**2); n=min(800,max(300,2*math.ceil(raw/2)))
    write(dest,dict(test_anchors=n,unclipped_precision_n=raw,max_development_paired_sd=sd,cap_binding=raw>800,model_run_sha256={p.name:digest(p) for p in sorted((O/'runs').glob('*.json'))},protocol_sha256=digest(O/'protocol.json'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),scope='No test outcomes used; precision heuristic on signed contrasts, not a guaranteed CI width for nonlinear G'))
    status('analysis_frozen',test_anchors=n,unclipped=raw); return n
def test_all():
    test=read(O/'splits/test.json'); lock=read(O/'analysis_lock.json')
    for name,sha in lock['model_run_sha256'].items():
        assert digest(O/'runs'/name)==sha
        dest=O/'test_predictions'/name
        if dest.exists(): continue
        run=read(O/'runs'/name); model,tok=base('model_files_0p5b'); model=PeftModel.from_pretrained(model,O/'checkpoints'/pathlib.Path(name).stem).eval()
        p=eval_groups(model,tok,test); write(dest,dict(seed=run['seed'],weight=run['weight'],test_sha256=digest(O/'splits/test.json'),probabilities=p.tolist()))
        del model,tok; gc.collect(); torch.cuda.empty_cache(); status('test_evaluated',run=name)
def all_work():
    prepare()
    for name in ['train','development','calibration']: make_split(name,P[name+'_anchors'])
    for seed in P['training_seeds']:
        for weight in [None]+P['weights']: train_one(seed,weight)
    n=freeze_test(); make_split('test',n); test_all(); status('completed',runs=15,test_anchors=n)
if __name__=='__main__':
    torch.set_num_threads(4); a=argparse.ArgumentParser(); a.add_argument('mode',choices=['all','prepare','test']); args=a.parse_args()
    try:
        if args.mode=='all': all_work()
        elif args.mode=='prepare': prepare()
        else: test_all()
    except Exception:
        status('failed',error=traceback.format_exc()); raise
