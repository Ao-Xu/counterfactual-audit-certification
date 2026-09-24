"""Natural rewrite development, with an explicit non-confirmatory boundary."""
import argparse,pathlib,json,re,unicodedata,time,gc,csv,hashlib,traceback
from collections import Counter
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments,set_seed
from train_and_eval import build_model,CompletionCollator

R=pathlib.Path(__file__).resolve().parent; O=R/'natural_rewrite_dev'
P=json.loads((O/'protocol.json').read_text()); LABELS=['contradiction','entailment']
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(p)
def norm(x): return ' '.join(unicodedata.normalize('NFKC',x).casefold().split())
def jsonl(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
def status(stage,**kw): write(O/'progress.json',dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)); print(stage,kw,flush=True)
def prompt(r): return 'Decide whether the hypothesis contradicts or follows from the premise. Return exactly contradiction or entailment.\nPremise: '+r['premise']+'\nHypothesis: '+r['hypothesis']+'\nLabel:'
def prepare():
    if (O/'candidates.json').exists(): return
    old=read(R/'confirmatory_v1/legacy_exclusions.json'); ids=set(old['ids']); premises=set(old['premises'])
    for rs in read(R/'confirmatory_v1/splits.json').values():
        for r in rs: ids.add(str(r['anchor_id'])); premises.add(norm(r['premise']))
    # The preceding index covers legacy project records; scan newer text manifests too.
    scanned=[]; errors=[]
    def visit(v):
        if isinstance(v,dict):
            if isinstance(v.get('premise'),str): premises.add(norm(v['premise']))
            for k in ['anchor_id','pairID','pair_id']:
                if k in v and isinstance(v[k],(str,int)): ids.add(str(v[k]))
            for q in v.values():
                if isinstance(q,(dict,list)): visit(q)
        elif isinstance(v,list):
            for q in v: visit(q)
    for p in R.rglob('*'):
        if p.suffix not in ['.json','.jsonl'] or O in p.parents or any(z.startswith('model_') for z in p.relative_to(R).parts): continue
        if p.stat().st_mtime < (R/'confirmatory_v1/legacy_exclusions.json').stat().st_mtime: continue
        try: visit(jsonl(p) if p.suffix=='.jsonl' else read(p)); scanned.append(str(p.relative_to(R)))
        except Exception as e: errors.append(dict(path=str(p.relative_to(R)),error=str(e)))
    assert not errors, errors
    cache=pathlib.Path.home()/'.cache/huggingface/datasets/multi_nli/default/0.0.0/da70db2af9d09693783c3320c4249840212ee221/multi_nli-train.arrow'
    ds=Dataset.from_file(str(cache)); rng=np.random.default_rng(P['split_seed']); labels=np.array(ds['label'])
    tok=AutoTokenizer.from_pretrained(R/'model_files_0p5b',local_files_only=True); splits={}; counts=Counter()
    pools={v:iter(rng.permutation(np.flatnonzero(labels==v))) for v in [0,2]}
    for name,n in [('train',P['candidate_train']),('development',P['candidate_development'])]:
        rows=[]
        for v in [0,2]:
            accepted=0
            for ix in pools[v]:
                r=ds[int(ix)]; counts['considered']+=1
                if str(r['pairID']) in ids or norm(r['premise']) in premises: counts['used']+=1; continue
                if not 6<=len(r['hypothesis'].split())<=24: counts['hypothesis_length']+=1; continue
                row=dict(anchor_id=str(r['pairID']),premise=r['premise'],hypothesis=r['hypothesis'],y=1 if v==0 else -1,source_genre=r['genre'])
                if len(tok.encode(prompt(row)+' entailment'))>220: counts['token_length']+=1; continue
                rows.append(row); ids.add(row['anchor_id']); premises.add(norm(row['premise'])); accepted+=1
                if accepted==n//2: break
            assert accepted==n//2
        rng.shuffle(rows); splits[name]=rows
    write(O/'candidates.json',splits); write(O/'split_audit.json',dict(counts=dict(counts),new_manifests=scanned,errors=errors,scope='Local-project exclusion, not pretraining contamination audit'))
    # Paired old rewrites, with flags that do not claim semantic-error ground truth.
    oldbase={r['anchor_id']:r for r in jsonl(R/'raw/anchors.jsonl')}
    audit=[]
    for r in jsonl(R/'raw/train_cf_n1000_k1.jsonl'):
        orig=oldbase[r['anchor_id']]['hypothesis']; h=r['hypothesis']
        audit.append(dict(anchor_id=r['anchor_id'],premise=r['premise'],original=orig,rewrite=h,label=r['label'],format_valid=r['generation_valid'],unchanged=norm(orig)==norm(h),number_changed=re.findall(r'\d+',orig)!=re.findall(r'\d+',h),negation_changed=neg(orig)!=neg(h),raw_generation=r['raw_generation']))
    write(O/'legacy_rewrite_audit.json',dict(n=len(audit),counts={k:sum(bool(r[k]) for r in audit) for k in ['format_valid','unchanged','number_changed','negation_changed']},rows=audit))
    status('prepared',train=256,development=128)
def neg(s): return bool(re.search(r"\b(no|not|never|nobody|nothing|neither|without)\b|n['’]t\b",s.lower()))
def genprompt(h,style):
    instruction=['formal written English, without changing facts or adding introductory phrases','natural conversational English, without fillers or introductory phrases','plain neutral English with different wording'][style]
    return f'Rewrite the following sentence in {instruction}. Preserve EXACT meaning, people, numbers, negation, quantifiers, tense and certainty. Do not add or remove information. Return only ONE rewritten sentence, without commentary or quotation marks.\nSentence: {h}'
def loadbase(name):
    tok=AutoTokenizer.from_pretrained(R/name,local_files_only=True); tok.padding_side='left'; tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(R/name,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda').eval()
    return model,tok
def generate():
    prepare(); data=read(O/'candidates.json'); jobs=[dict(split=k,**r,style=j) for k,rs in data.items() for r in rs for j in range(3)]
    path=O/'generations.jsonl'; done=jsonl(path) if path.exists() else []; assert len(done)<=len(jobs)
    model,tok=loadbase('model_files'); set_seed(927310)
    for start in range(len(done),len(jobs),12):
        batch=jobs[start:start+12]; ps=[tok.apply_chat_template([{'role':'user','content':genprompt(r['hypothesis'],r['style'])}],tokenize=False,add_generation_prompt=True) for r in batch]
        enc=tok(ps,padding=True,truncation=False,return_tensors='pt').to('cuda'); assert enc.input_ids.shape[1]<512
        with torch.inference_mode(): out=model.generate(**enc,max_new_tokens=64,do_sample=False,pad_token_id=tok.pad_token_id)
        width=enc.input_ids.shape[1]
        with path.open('a',encoding='utf-8') as f:
            for r,seq in zip(batch,out):
                tokens=seq[width:].tolist(); raw=tok.decode(tokens,skip_special_tokens=True).strip(); h=raw.strip('"\' ')
                result=dict(r,original=r['hypothesis'],hypothesis=h,raw_generation=raw,truncated=tok.eos_token_id not in tokens and len(tokens)>=64)
                f.write(json.dumps(result,ensure_ascii=False)+'\n')
        status('generating',completed=min(start+12,len(jobs)),total=len(jobs))
    del model,tok; gc.collect(); torch.cuda.empty_cache()
def eqprompt(a,b): return f'Do these two sentences have exactly the same factual meaning, including negation, quantities, identities, tense and certainty? Answer only Yes or No.\nSentence A: {a}\nSentence B: {b}'
def yes_scores(model,tok,texts):
    ids=[tok.encode('No',add_special_tokens=False)[0],tok.encode('Yes',add_special_tokens=False)[0]]; vals=[]
    for start in range(0,len(texts),16):
        ps=[tok.apply_chat_template([{'role':'user','content':s}],tokenize=False,add_generation_prompt=True) for s in texts[start:start+16]]
        enc=tok(ps,padding=True,truncation=False,return_tensors='pt').to('cuda'); assert enc.input_ids.shape[1]<=512
        with torch.inference_mode(): p=model(**enc,logits_to_keep=1).logits[:,-1,ids].float().softmax(-1)[:,1].cpu().tolist()
        vals+=p
    return vals
def screen():
    rows=jsonl(O/'generations.jsonl'); assert len(rows)==1152
    model,tok=loadbase('model_files_0p5b'); texts=[eqprompt(a,b) for r in rows for a,b in [(r['original'],r['hypothesis']),(r['hypothesis'],r['original'])]]
    scores=[]
    for start in range(0,len(texts),128):
        scores+=yes_scores(model,tok,texts[start:start+128]); status('screening',completed=len(scores),total=len(texts))
    for i,r in enumerate(rows):
        h=r['hypothesis']; flags=[]
        if r['truncated'] or '\n' in h or not 4<=len(h.split())<=50 or any(z in h.lower() for z in ['here is','rewritten sentence','entailment','contradiction']): flags.append('format')
        if neg(r['original'])!=neg(h): flags.append('negation')
        if sorted(re.findall(r'\d+',r['original']))!=sorted(re.findall(r'\d+',h)): flags.append('numbers')
        if len(tok.encode(prompt(r)+' entailment'))>P['max_complete_tokens']: flags.append('tokens')
        r['equivalence_yes']=scores[2*i:2*i+2]
        if min(r['equivalence_yes'])<.5: flags.append('automatic_equivalence')
        r['rejection_reasons']=flags
    write(O/'screened.json',rows)
    accepted={}; stats={}; rng=np.random.default_rng(927320)
    for name in ['train','development']:
        grouped={}
        for r in rows:
            if r['split']==name: grouped.setdefault(r['anchor_id'],[]).append(r)
        eligible=[v for v in grouped.values() if len(v)==3 and all(not r['rejection_reasons'] for r in v) and norm(v[0]['hypothesis'])!=norm(v[1]['hypothesis'])]
        per=min(P['max_'+name]//2,*[sum(v[0]['y']==s for v in eligible) for s in [-1,1]])
        stats[name]=dict(candidates=len(grouped),eligible=len(eligible),per_class_selected=per)
        selected=[]
        for s in [-1,1]:
            candidates=[v for v in eligible if v[0]['y']==s]; rng.shuffle(candidates); selected+=candidates[:per]
        rng.shuffle(selected); accepted[name]=selected
    write(O/'accepted.json',accepted); write(O/'screen_metrics.json',dict(splits=stats,reasons=dict(Counter(k for r in rows for k in r['rejection_reasons'])),scope='Automatic equivalence scores are not human validity or calibrated probabilities'))
    # Reproducible blind-review packet, not filled with fake human judgments.
    sample=np.random.default_rng(927321).permutation(len(rows))[:60]
    with (O/'review_packet.tsv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f,delimiter='\t'); writer.writerow(['item','premise','original','rewrite','same_meaning','notes'])
        for i in sample: r=rows[int(i)]; writer.writerow([int(i),r['premise'],r['original'],r['hypothesis'],'',''])
    del model,tok; gc.collect(); torch.cuda.empty_cache(); status('screened',counts=stats)

class Complete(torch.utils.data.Dataset):
    def __init__(self,rows,tok):
        self.items=[]
        for r in rows:
            a=tok.encode(prompt(r)); b=tok.encode(prompt(r)+' '+r['label']); assert b[:len(a)]==a and len(a)<len(b)<=256
            self.items.append(dict(input_ids=b,attention_mask=[1]*len(b),labels=[-100]*len(a)+b[len(a):]))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]
def training_rows(pools,condition,seed,eta=0):
    rng=np.random.default_rng(seed); factual={}
    for y in [-1,1]:
        ix=[i for i,v in enumerate(pools) if v[0]['y']==y]; rng.shuffle(ix); m=round(.9*len(ix))
        factual.update({i:((y+1)//2 if pos<m else 1-(y+1)//2) for pos,i in enumerate(ix)})
    out=[]
    for i,v in enumerate(pools):
        v=sorted(v,key=lambda r:r['style']); base=v[0]
        if condition=='original': rr=[dict(base,hypothesis=base['original'])]*2
        elif condition=='generic_paraphrase': rr=[v[2]]*2
        elif condition=='factual_style': rr=[v[factual[i]]]*2
        else: rr=v[:2]
        out.extend(dict(r,label=LABELS[int(r['y']==1)]) for r in rr)
    if eta:
        for y in [-1,1]:
            indices=[i for i,r in enumerate(out) if r['y']==y]; num=round(eta*len(indices))
            if condition=='reinforce': ix=[i for i in indices if 2*out[i]['style']-1!=y]
            elif condition=='oppose': ix=[i for i in indices if 2*out[i]['style']-1==y]
            else: ix=indices
            rng.shuffle(ix); assert num<=len(ix)
            for i in ix[:num]: out[i]['label']=LABELS[int(y!=1)]
    rng.shuffle(out); return out
def evaluate(model,tok,rows):
    tok.padding_side='left'; ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS]; vals=[]
    assert len(set(ids))==2
    for start in range(0,len(rows),16):
        enc=tok([prompt(r) for r in rows[start:start+16]],padding=True,truncation=False,return_tensors='pt').to('cuda'); assert enc.input_ids.shape[1]<=256
        with torch.inference_mode(): vals.extend(model(**enc,logits_to_keep=1).logits[:,-1,ids].float().softmax(-1).cpu().tolist())
    return np.array(vals)
def train_one(pools,dev,seed,condition,eta=0):
    path=O/'runs'/f'{seed}_{condition}_{eta:g}.json'
    if path.exists(): assert read(path)['status']=='completed'; return
    status('training',seed=seed,condition=condition,eta=eta); begin=time.time()
    rows=training_rows(pools,condition,seed,eta); model,tok=build_model(R/'model_files_0p5b',seed); ds=Complete(rows,tok)
    folder=O/'checkpoints'/path.stem
    args=TrainingArguments(output_dir=str(folder),per_device_train_batch_size=2,gradient_accumulation_steps=4,max_steps=64,learning_rate=2e-4,warmup_ratio=.05,lr_scheduler_type='cosine',bf16=True,gradient_checkpointing=True,save_strategy='no',report_to=[],disable_tqdm=True,logging_steps=32,seed=seed,data_seed=seed,remove_unused_columns=False,optim='adamw_torch')
    trainer=Trainer(model=model,args=args,train_dataset=ds,data_collator=CompletionCollator(tok)); fit=trainer.train(); model.eval()
    evalrows=[r for v in dev for r in sorted(v,key=lambda r:r['style'])[:2]]
    pp=evaluate(model,tok,evalrows).reshape(len(dev),2,2); y=np.array([v[0]['y'] for v in dev]); correct=pp[np.arange(len(dev))[:,None],np.arange(2)[None,:],((y+1)//2)[:,None]]
    nll=-np.log(np.clip(correct,1e-12,1)); metrics={}
    for name,agree in [('ood',.05),('id',.9),('uniform',.5)]:
        weights=np.where(np.array([-1,1])[None,:]==y[:,None],agree,1-agree)
        metrics[name+'_nll']=float(np.mean((weights*nll).sum(1))); metrics[name+'_accuracy']=float(np.mean((weights*(pp.argmax(2)==((y+1)//2)[:,None])).sum(1)))
    original=[dict(v[0],hypothesis=v[0]['original']) for v in dev]; op=evaluate(model,tok,original); metrics['original_nll']=float(-np.log(np.clip(op[np.arange(len(y)),(y+1)//2],1e-12,1)).mean())
    metrics['worst_style_nll']=float(max(nll.mean(0)))
    model.save_pretrained(folder,safe_serialization=True)
    write(path,dict(status='completed',seed=seed,condition=condition,eta=eta,metrics=metrics,probabilities=pp.tolist(),original_probabilities=op.tolist(),train_rows=rows,dev_ids=[v[0]['anchor_id'] for v in dev],steps=trainer.state.global_step,train_tokens=sum(len(v['input_ids']) for v in ds.items),seconds=time.time()-begin,loss=fit.training_loss))
    del model,tok,ds,trainer; gc.collect(); torch.cuda.empty_cache()
def run():
    data=read(O/'accepted.json')
    for name,rs in data.items():
        per=len(rs)//2
        if per<P['minimum_per_class_'+name]: status('stopped_low_acceptance',split=name,count=per); return
    write(O/'development_lock.json',dict(protocol=P,source=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),accepted=hashlib.sha256((O/'accepted.json').read_bytes()).hexdigest(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    for seed in P['training_seeds']:
        for condition in P['conditions']: train_one(data['train'],data['development'],seed,condition)
    diffs=[read(O/'runs'/f'{s}_balanced_cf_0.json')['metrics']['ood_nll']-read(O/'runs'/f'{s}_factual_style_0.json')['metrics']['ood_nll'] for s in P['training_seeds']]
    advance=float(np.mean(diffs))<=-.02 and sum(d<0 for d in diffs)>=2
    write(O/'development_decision.json',dict(differences=diffs,mean=float(np.mean(diffs)),advance=bool(advance),scope='Development-only screening, no significance or confirmed benefit claim'))
    if advance:
        for seed in P['training_seeds']:
            for eta in [.1,.2,.4]:
                for c in ['random','reinforce','oppose']: train_one(data['train'],data['development'],seed,c,eta)
    runs=[read(p) for p in sorted((O/'runs').glob('*.json'))]
    write(O/'metrics.json',dict(scope=P['scope'],runs=[{k:r[k] for k in ['seed','condition','eta','metrics','seconds','steps','train_tokens']} for r in runs],decision=read(O/'development_decision.json'),train_anchors=len(data['train']),development_anchors=len(data['development'])))
    status('completed',runs=len(runs),advance=bool(advance),clean_differences=diffs)
if __name__=='__main__':
    torch.set_num_threads(4); parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','generate','screen','run','all']); a=parser.parse_args()
    try:
        if a.mode=='all': generate(); screen(); run()
        else: globals()[a.mode]()
    except Exception:
        status('failed',error=traceback.format_exc()); raise
