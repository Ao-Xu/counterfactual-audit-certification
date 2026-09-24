"""Registered MultiNLI direction experiment; all outcomes are retained."""
from __future__ import annotations
import argparse, gc, hashlib, json, pathlib, platform, time
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from train_and_eval import build_model, PromptDataset, CompletionCollator, write_json

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'directional_results'
P=json.loads((ROOT/'directional_protocol.json').read_text())
TAGS=['cedar archive','amber archive']
LABELS=['contradiction','entailment']

def prompt(r):
    return ('Decide whether the hypothesis contradicts or follows from the premise. '
      'Return exactly contradiction or entailment.\n'
      f'Source: {r["genre"]}\nPremise: {r["premise"]}\nHypothesis: {r["hypothesis"]}\nLabel:')

def balanced(ds,n,rng):
    a=np.asarray(ds['label']); ids=np.concatenate([rng.permutation(np.flatnonzero(a==v))[:n//2] for v in [0,2]])
    rng.shuffle(ids)
    return [dict(anchor_id=ds[int(i)]['pairID'],premise=ds[int(i)]['premise'],hypothesis=ds[int(i)]['hypothesis'],
        y=1 if ds[int(i)]['label']==0 else -1) for i in ids]

def prepare():
    OUT.mkdir(exist_ok=True)
    target=OUT/'splits.json'
    if target.exists(): return json.loads(target.read_text())
    cache=pathlib.Path.home()/'.cache/huggingface/datasets/multi_nli/default/0.0.0/da70db2af9d09693783c3320c4249840212ee221'
    train=Dataset.from_file(str(cache/'multi_nli-train.arrow'))
    test=Dataset.from_file(str(cache/'multi_nli-validation_mismatched.arrow'))
    rng=np.random.default_rng(P['split_seed'])
    # Select each class once, then stratify without replacement across all splits.
    seen_premises=set(); seen_ids=set()
    def partition(ds,counts):
        a=np.asarray(ds['label']); pools={v:iter(rng.permutation(np.flatnonzero(a==v))) for v in [0,2]}; result=[]
        for n in counts:
            selected=[]
            for v in [0,2]:
                count=0
                while count<n//2:
                    item=ds[int(next(pools[v]))]
                    if item['premise'] in seen_premises or item['pairID'] in seen_ids: continue
                    seen_premises.add(item['premise']); seen_ids.add(item['pairID']); count+=1
                    selected.append(dict(anchor_id=item['pairID'],premise=item['premise'],hypothesis=item['hypothesis'],y=1 if v==0 else -1))
            rng.shuffle(selected); result.append(selected)
        return result
    dev,tr=partition(train,[P['development_anchors'],P['training_pool_anchors']])
    ca,te=partition(test,[P['calibration_anchors'],P['test_anchors']])
    splits=dict(development=dev,train=tr,calibration=ca,test=te)
    ids=[r['anchor_id'] for rows in splits.values() for r in rows]
    assert len(set(ids))==len(ids)
    # Pair-level source ID disjointness and identical-text leakage both checked.
    keys=[(r['premise'],r['hypothesis']) for rows in splits.values() for r in rows]
    assert len(set(keys))==len(keys)
    write_json(target,splits)
    write_json(OUT/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),protocol=P))
    return splits

def members(rows):
    return [dict(r,genre=TAGS[j],cue=2*j-1,label=LABELS[int(r['y']==1)]) for r in rows for j in range(2)]

def extract():
    splits=prepare(); path=OUT/'features.npz'
    if path.exists(): return
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(ROOT/'model_files_0p5b',local_files_only=True)
    tok.padding_side='left'; tok.truncation_side='left'; tok.pad_token=tok.eos_token
    model=AutoModel.from_pretrained(ROOT/'model_files_0p5b',local_files_only=True,torch_dtype=torch.bfloat16).cuda().eval()
    arrays={}
    for name,rows in splits.items():
        texts=[prompt(r) for r in members(rows)]; fs=[]
        for start in range(0,len(texts),16):
            enc=tok(texts[start:start+16],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
            with torch.inference_mode(): h=model(**enc).last_hidden_state[:,-1].float().cpu().numpy()
            fs.append(h)
            if start%512==0: print('FEATURES',name,start,len(texts),flush=True)
        arrays[name]=np.concatenate(fs).reshape(len(rows),2,-1)
        arrays[name+'_y']=np.array([r['y'] for r in rows])
    np.savez_compressed(path,**arrays)
    del model; gc.collect(); torch.cuda.empty_cache()

def select_anchors(y,n,seed):
    rng=np.random.default_rng(seed)
    ix=np.concatenate([rng.permutation(np.flatnonzero(y==s))[:n//2] for s in [-1,1]])
    return rng.permutation(ix)

def corrupt(y,cue,eta,mechanism,seed):
    result=y.copy(); chosen=[]; rng=np.random.default_rng(seed)
    for s in [-1,1]:
        ix=np.flatnonzero(y==s); count=int(round(len(ix)*eta))
        if mechanism=='reinforce': ix=ix[cue[ix]!=y[ix]]
        if mechanism=='oppose': ix=ix[cue[ix]==y[ix]]
        assert len(ix)>=count
        chosen.extend(rng.permutation(ix)[:count])
    result[chosen]*=-1
    assert np.sum(result!=y)==int(round(len(y)*eta))
    return result

def factual_indices(y,seed):
    rng=np.random.default_rng(seed); cue=y.copy()
    for s in [-1,1]:
        ix=np.flatnonzero(y==s); wrong=rng.permutation(ix)[:int(round(.05*len(ix)))]; cue[wrong]*=-1
    return (cue>0).astype(int)

def risk(X,y,w,agreement=.05):
    weights=np.where(np.array([-1,1])[None,:]==y[:,None],agreement,1-agreement)
    return float(np.mean(np.sum(weights*(X@w-y[:,None])**2,axis=1)))

def moment(X,y,agreement):
    weights=np.where(np.array([-1,1])[None,:]==y[:,None],agreement,1-agreement)
    xf=X.reshape(-1,X.shape[-1]); weights=weights.reshape(-1)/len(y); yy=np.repeat(y,2)
    return (xf.T*weights)@xf,xf.T@(weights*yy)

def run_fixed():
    extract(); f=np.load(OUT/'features.npz'); dev=f['development'].reshape(-1,f['development'].shape[-1]).astype(float)
    center=dev.mean(0); _,s,v=np.linalg.svd(dev-center,full_matrices=False); basis=v[:P['pca_components']].T
    scale=s[:P['pca_components']]/np.sqrt(len(dev)-1)
    X={name:np.concatenate([np.ones((*f[name].shape[:2],1)),(f[name]-center)@basis/scale],axis=-1) for name in ['train','calibration','test','development']}
    Y={name:f[name+'_y'] for name in X}
    np.savez_compressed(OUT/'projected.npz',**X,**{k+'_y':v for k,v in Y.items()},basis=basis,center=center,scale=scale)
    XC=X['calibration']; yc=Y['calibration']; S,b=moment(XC,yc,.5); Se,be=moment(XC,yc,.05)
    u=np.linalg.solve(S,b); Sfi,bfi=moment(XC,yc,.95); wfcal=np.linalg.solve(Sfi,bfi)
    G=risk(XC,yc,wfcal)-risk(XC,yc,u)
    rows=[]; weights=[]
    for seed in P['fixed_head_seeds']:
        ix=select_anchors(Y['train'],P['fixed_head_training_anchors'],seed); xt=X['train'][ix]; yt=Y['train'][ix]
        factual=xt[np.arange(len(ix)),factual_indices(yt,seed+1)]
        wf=np.linalg.lstsq(factual,yt,rcond=None)[0]; ref=risk(X['test'],Y['test'],wf)
        flat=xt.reshape(-1,xt.shape[-1]); yy=np.repeat(yt,2); cue=np.tile([-1,1],len(yt))
        assert np.linalg.matrix_rank(flat)==flat.shape[1]
        for mechanism in P['mechanisms']:
            for eta in P['rates']:
                corrupted=corrupt(yy,cue,eta,mechanism,seed+2)
                w=np.linalg.lstsq(flat,corrupted,rcond=None)[0]
                cflat=XC.reshape(-1,XC.shape[-1]); cy=np.repeat(yc,2)
                ct=corrupt(cy,np.tile([-1,1],len(yc)),eta,mechanism,seed+999)
                d=cflat.T@(ct-cy)/len(ct); move=np.linalg.solve(S,d)
                prediction=-G+2*move@(Se@u-be)+move@Se@move
                observed=risk(X['test'],Y['test'],w)-ref
                # A frequency-only baseline averages calibration predictions across mechanisms.
                rows.append(dict(seed=seed,mechanism=mechanism,eta=eta,predicted=prediction,observed=observed,
                  factual_risk=ref,cf_risk=ref+observed,error_energy=float(np.mean((corrupted-yy)**2)),moment_norm=float(np.linalg.norm(d)),
                  test_accuracy=float(np.mean(np.sum(np.where(np.array([-1,1])[None,:]==Y['test'][:,None],.05,.95)*((X['test']@w>0)==(Y['test'][:,None]>0)),axis=1)))))
                if mechanism=='random' and eta==0: weights.append(w)
    for row in rows:
        row['rate_prediction']=float(np.mean([r['predicted'] for r in rows if r['seed']==row['seed'] and r['eta']==row['eta']]))
    write_json(OUT/'fixed_results.json',dict(status='completed',rows=rows,dimension=int(S.shape[0]),condition_number=float(np.linalg.cond(S)),calibration_clean_gain=G))
    np.savez_compressed(OUT/'heads.npz',clean=np.array(weights))
    run_allocation_transfer(X,Y,np.mean(weights,axis=0))

def run_allocation_transfer(X,Y,w):
    def prob(x): return 1/(1+np.exp(-np.clip(2*(x@w),-40,40)))
    p=prob(X['test']); y=Y['test']; losses=np.where(y[:,None]==1,1-p,p)
    # Finite pool estimand; independent anchor draws with replacement, nuisance draws IID.
    true=float(losses.mean()); nrep=P['allocation']['replications']; B=P['allocation']['total_budget']; cand=P['allocation']['candidate_K']
    alloc=[]
    for cost in P['allocation']['cost_ratios']:
        for rep in range(nrep):
            rng=np.random.default_rng(700000+rep)
            ci=rng.integers(len(Y['calibration']),size=128); pc=prob(X['calibration'][ci]); lc=np.where(Y['calibration'][ci,None]==1,1-pc,pc)
            samples=np.take_along_axis(lc,rng.integers(2,size=(128,4)),axis=1)
            b=float(np.mean(samples.var(axis=1,ddof=1))); a=max(0,float(samples.mean(1).var(ddof=1)-b/4))
            ns={k:int(B//(cost+k)) for k in cand}; pilot_k=min(cand,key=lambda k:(a+b/k)/ns[k])
            for method,k in [('pilot',pilot_k),('K1',1),('K4',4),('K16',16)]:
                rr=np.random.default_rng(800000+rep*29+k); n=ns[k]; ix=rr.integers(len(y),size=n)
                sample=np.take_along_axis(losses[ix],rr.integers(2,size=(n,k)),axis=1)
                estimate=float(sample.mean()); se=float(sample.mean(1).std(ddof=1)/np.sqrt(n))
                alloc.append(dict(cost=cost,rep=rep,method=method,k=k,n=n,squared_error=(estimate-true)**2,covered=abs(estimate-true)<=1.96*se,a=a,b=b))
    # Head intervention uses one fixed direction learned from development tags.
    q=(X['development'][:,1]-X['development'][:,0]).mean(0); q[0]=0; q=q/np.linalg.norm(q)
    perpendicular=w-q*(q@w); transfer=[]
    for alpha in [0,.5,1,2]:
        wa=perpendicular+alpha*q*(q@w)
        pa=1/(1+np.exp(-np.clip(2*(X['test']@wa),-40,40))); pa=np.clip(pa,1e-10,1-1e-10)
        la=np.where(y[:,None]==1,1-pa,pa); reference=la.mean()
        kl01=pa[:,0]*np.log(pa[:,0]/pa[:,1])+(1-pa[:,0])*np.log((1-pa[:,0])/(1-pa[:,1]))
        kl10=pa[:,1]*np.log(pa[:,1]/pa[:,0])+(1-pa[:,1])*np.log((1-pa[:,1])/(1-pa[:,0]))
        L=float(np.mean((kl01+kl10)/4)); V=float(np.mean(la.var(1)))
        for delta in [0,.25,.5,.75,.9]:
            weights=np.where(np.array([-1,1])[None,:]==y[:,None],(1-delta)/2,(1+delta)/2)
            gap=float(abs(np.mean(np.sum(weights*la,axis=1))-reference)); centered=delta*np.sqrt(V); bound=delta*np.sqrt(L/2)
            assert gap<=centered+1e-12 and centered<=bound+1e-12
            transfer.append(dict(alpha=alpha,delta=delta,gap=gap,centered=centered,bound=bound,L=L,V=V))
    write_json(OUT/'allocation_transfer.json',dict(status='completed',allocation=alloc,transfer=transfer,true_risk=true,pilot_cost_excluded_from_followup_budget=True))

class BinaryDataset(PromptDataset):
    def __init__(self,rows,tok):
        self.items=[]
        for r in rows:
            a=tok.encode(prompt(r),add_special_tokens=True); b=tok.encode(prompt(r)+' '+r['label'],add_special_tokens=True)
            # Retain final label and entire prompt tail, with an explicit consistent context cap.
            labels=[-100]*len(a)+b[len(a):]
            self.items.append(dict(input_ids=b[-256:],attention_mask=[1]*min(256,len(b)),labels=labels[-256:]))
        assert all(any(x!=-100 for x in r['labels']) for r in self.items)

def eval_binary(model,tok,rows):
    tok.padding_side='left'; ids=[tok.encode(' '+lab,add_special_tokens=False)[0] for lab in LABELS]; ps=[]
    for start in range(0,len(rows),16):
        enc=tok([prompt(r) for r in rows[start:start+16]],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
        with torch.inference_mode(): z=model(**enc,logits_to_keep=1).logits[:,-1,ids].float(); pp=torch.softmax(z,dim=-1).cpu().numpy()
        ps.append(pp)
    return np.concatenate(ps)

def run_lora():
    splits=prepare(); torch.set_num_threads(4)
    for seed in P['lora_seeds']:
        y=np.array([r['y'] for r in splits['train']]); ix=select_anchors(y,P['lora_training_anchors'],seed)
        base=[splits['train'][i] for i in ix]; allrows=members(base)
        for condition in P['lora_conditions']:
            path=OUT/f'lora_{seed}_{condition}.json'
            if path.exists() and json.loads(path.read_text()).get('status')=='completed': continue
            start=time.time(); print('LORA START',seed,condition,flush=True)
            rows=[dict(r) for r in allrows]
            if condition=='factual':
                j=factual_indices(y[ix],seed+1); rows=[dict(rows[2*i+j[i]]) for i in range(len(ix)) for _ in range(2)]
            else:
                mech,rate=condition.split('_'); yy=np.array([r['y'] for r in rows]); cue=np.array([r['cue'] for r in rows]); altered=corrupt(yy,cue,float(rate),mech,seed+2)
                for r,target in zip(rows,altered): r['label']=LABELS[int(target==1)]
            model,tok=build_model(ROOT/'model_files_0p5b',seed); tok.truncation_side='left'
            data=BinaryDataset(rows,tok); config=P['lora']; checkpoint=OUT/'checkpoints'/f'{seed}_{condition}'
            args=TrainingArguments(output_dir=str(checkpoint),per_device_train_batch_size=2,gradient_accumulation_steps=4,num_train_epochs=1,
                learning_rate=.0002,warmup_ratio=.05,lr_scheduler_type='cosine',bf16=True,gradient_checkpointing=True,
                save_strategy='no',report_to=[],disable_tqdm=True,logging_steps=100,seed=seed,data_seed=seed,remove_unused_columns=False,optim='adamw_torch')
            trainer=Trainer(model=model,args=args,train_dataset=data,data_collator=CompletionCollator(tok)); fit=trainer.train(); model.eval()
            result={}
            for name in ['calibration','test']:
                rr=members(splits[name]); probs=eval_binary(model,tok,rr); result[name]=dict(probabilities=probs.tolist(),anchor_ids=[r['anchor_id'] for r in rr],y=[r['y'] for r in rr],cue=[r['cue'] for r in rr])
            model.save_pretrained(checkpoint,safe_serialization=True)
            write_json(path,dict(status='completed',seed=seed,condition=condition,train_rows=rows,train_loss=fit.training_loss,evaluation=result,seconds=time.time()-start))
            print('LORA DONE',seed,condition,round(time.time()-start,1),flush=True)
            del trainer,model,tok,data; gc.collect(); torch.cuda.empty_cache()

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['fixed','lora','all']); args=ap.parse_args()
    if args.mode in ['fixed','all']: run_fixed()
    if args.mode in ['lora','all']: run_lora()
