"""Fixed protocol for verifiable text counterfactual post-training.

Two tasks, two complete pretrained model sizes, three paired seeds. Templates
have exact labels. A source tag is the only counterfactual intervention.
No hyperparameters or grids are selected using evaluation outcomes.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time
import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from train_and_eval import (LABELS, build_model, PromptDataset, CompletionCollator,
                            evaluate_rows, write_json)

ROOT = pathlib.Path(__file__).resolve().parent
TAGS = ['amber archive', 'cedar archive', 'violet archive']
PROTOCOL = dict(version='verified_text_v1', tasks=['attributes', 'relations'],
    sizes=['1p5b', '0p5b'], seeds=[2027,2028,2029], conditions=['factual','cf_0','cf_20','cf_40'],
    anchors=320, siblings=3, rows=960, epochs=1, learning_rate=2e-4,
    lora_rank=8, batch=2, accumulation=4, max_length=192,
    train_cue_agreement=0.95, pilot_anchors=64, validation_anchors=128,
    test_anchors=256, evaluation_seed=71903,
    description='Controlled generated English, exact symbolic labels, source-tag edits only.')

def anchors(task, n, seed, split):
    rng=np.random.default_rng(seed)
    output=[]
    colors=['red','blue','green','yellow','white','black']
    nouns=['box','cup','vase','bowl','plate','bag']
    for i in range(n):
        y=int(rng.integers(3))
        a,b,c,d=[f'{nouns[int(rng.integers(len(nouns)))]} {split}-{i}-{j}' for j in range(4)]
        if task=='attributes':
            col,other=rng.choice(colors,2,replace=False).tolist()
            premise=f'The {a} is {col}. The {b} is {other}.'
            hypothesis=[f'The {a} is {col}.', f'The {c} is {col}.',f'The {a} is {other}.'][y]
            truth=dict(task=task,known={a:col,b:other},query_entity=(c if y==1 else a),query_color=(other if y==2 else col))
        else:
            premise=f'The {a} is to the left of the {b}. The {b} is to the left of the {c}.'
            left,right=[(a,c),(a,d),(c,a)][y]
            hypothesis=f'The {left} is to the left of the {right}.'
            truth=dict(task=task,order=[a,b,c],left=left,right=right)
        output.append(dict(anchor_id=f'{task}-{split}-{i}',premise=premise,hypothesis=hypothesis,
                           label=LABELS[y],truth=truth))
    return output

def verify(row):
    q=row['truth']
    if q['task']=='attributes':
        known=q['known'].get(q['query_entity'])
        y=1 if known is None else (0 if known==q['query_color'] else 2)
    else:
        order=q['order']
        y=1 if q['left'] not in order or q['right'] not in order else (0 if order.index(q['left'])<order.index(q['right']) else 2)
    return row['label']==LABELS[y]

def training(task, seed, condition):
    base=anchors(task,320,seed,'train')
    rng=np.random.default_rng(seed+890)
    rows=[]
    for row in base:
        assert verify(row)
        y=LABELS.index(row['label'])
        cue=y if rng.random()<0.95 else int(rng.choice([x for x in range(3) if x!=y]))
        for j in range(3):
            rows.append(dict(row,genre=TAGS[cue if condition=='factual' else j]))
    eta=0 if condition=='factual' else int(condition.split('_')[1])/100
    permutation=np.random.default_rng(seed+501).permutation(len(rows))
    flip=np.random.default_rng(seed+502).integers(1,3,len(rows))
    for index in permutation[:int(eta*len(rows))]:
        rows[index]['label']=LABELS[(LABELS.index(rows[index]['label'])+int(flip[index]))%3]
    assert sum(not verify(r) for r in rows)==int(eta*len(rows))
    np.random.default_rng(seed+503).shuffle(rows)
    return rows

def evaluation(task):
    output=[]
    for split,n,offset in [('pilot',64,0),('validation',128,1000),('test',256,2000)]:
        for r in anchors(task,n,71903+offset,split):
            assert verify(r)
            output.extend([dict(r,genre=tag,split=split) for tag in TAGS])
    return output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sizes',nargs='+',default=PROTOCOL['sizes'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=ROOT/'verified_results'
    out.mkdir(exist_ok=True)
    write_json(out/'protocol.json',PROTOCOL)
    planned=len(PROTOCOL['tasks'])*len(PROTOCOL['sizes'])*len(PROTOCOL['seeds'])*len(PROTOCOL['conditions'])
    for size in args.sizes:
        model_path=ROOT/('model_files' if size=='1p5b' else 'model_files_0p5b')
        for task in PROTOCOL['tasks']:
            ev=evaluation(task)
            write_json(out/f'evaluation_{task}.json',dict(rows=ev))
            for seed in PROTOCOL['seeds']:
                for cond in PROTOCOL['conditions']:
                    name=f'{size}_{task}_{seed}_{cond}'
                    path=out/(name+'.json')
                    if path.exists() and json.loads(path.read_text())['status']=='completed':
                        continue
                    start=time.time()
                    print('START',name,flush=True)
                    rows=training(task,seed,cond)
                    model,tok=build_model(model_path,seed)
                    data=PromptDataset(rows,tok,192)
                    assert len(data)==960
                    checkpoint=ROOT/'verified_checkpoints'/name
                    trargs=TrainingArguments(output_dir=str(checkpoint),per_device_train_batch_size=2,
                        gradient_accumulation_steps=4,num_train_epochs=1,learning_rate=2e-4,
                        warmup_ratio=.05,lr_scheduler_type='cosine',weight_decay=0,
                        bf16=True,gradient_checkpointing=True,save_strategy='no',report_to=[],
                        disable_tqdm=True,logging_steps=60,seed=seed,data_seed=seed,
                        remove_unused_columns=False,optim='adamw_torch')
                    trainer=Trainer(model=model,args=trargs,train_dataset=data,data_collator=CompletionCollator(tok))
                    train=trainer.train()
                    result=evaluate_rows(model,tok,ev,16,'verified_text')
                    result.update(anchor_ids=[r['anchor_id'] for r in ev],genres=[r['genre'] for r in ev],splits=[r['split'] for r in ev])
                    model.save_pretrained(checkpoint,safe_serialization=True)
                    write_json(path,dict(status='completed',protocol='verified_text_v1',size=size,task=task,
                        seed=seed,condition=cond,training_rows=rows,evaluation=result,
                        train_loss=train.training_loss,elapsed_seconds=time.time()-start))
                    print('DONE',name,'seconds',round(time.time()-start,1),flush=True)
                    del trainer,model,tok,data
                    gc.collect(); torch.cuda.empty_cache()
                    complete=sum(1 for p in out.glob('*_20*.json') if 'status' in json.loads(p.read_text()))
                    write_json(out/'manifest.json',dict(planned=planned,completed=complete))

if __name__=='__main__': main()
