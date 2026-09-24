"""Fixed-head confirmation and held-out diagnostics. Predict before scoring."""
import argparse, hashlib, json, pathlib, time
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import t
from transformers import AutoModel, AutoTokenizer
from directional_experiment import prompt,members,select_anchors,corrupt,factual_indices,moment,risk

ROOT=pathlib.Path(__file__).resolve().parent; OUT=ROOT/'confirmatory_v1'
SEEDS=list(range(926301,926333)); HASH='e1c5c59aeb9b1d6f836e7da8cf18937584d267c18c92a553f3e441b5f3cfc6c6'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(p)
def ci(x):
    x=np.asarray(x); h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return dict(mean=float(x.mean()),low=float(x.mean()-h),high=float(x.mean()+h),n=len(x))
def files(): return [pathlib.Path(__file__),ROOT/'directional_experiment.py',ROOT/'train_and_eval.py',ROOT/'directional_protocol.json']
def lock():
    assert not (OUT/'fixed_lock.json').exists()
    # Deterministic algebra check, not data-dependent model selection.
    rng=np.random.default_rng(1); X=rng.normal(size=(80,5)); y=rng.normal(size=80); e=rng.normal(size=80)
    S=X.T@X/80; u=np.linalg.solve(S,X.T@y/80); d=X.T@e/80; w=np.linalg.solve(S,X.T@(y+e)/80)
    move=np.linalg.solve(S,d)
    assert np.allclose(w,u+move)
    assert np.allclose(np.mean((X@w-y)**2)-np.mean((X@u-y)**2),2*move@(S@u-X.T@y/80)+move@S@move)
    assert sha(OUT/'splits.json')==HASH
    write(OUT/'fixed_lock.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        code={p.name:sha(p) for p in files()},splits_sha256=HASH,seeds=SEEDS,
        primary='32 pool-level MAE(full)-MAE(quadratic-only); 12 equal-weight nonzero corruption cells; t95 interval',
        caveat='Code locked after LoRA summary was observed; plan predated it; not a fully blinded cross-arm replication',
        nuisance_head='mean of 32 clean training heads; development-only cue direction',
        bootstrap_repetitions=500,bootstrap_seed=927000,variance_replications=2000))
def check():
    l=read(OUT/'fixed_lock.json'); assert sha(OUT/'splits.json')==HASH
    for p in files(): assert sha(p)==l['code'][p.name]
def extract():
    check(); data=read(OUT/'splits.json'); tok=AutoTokenizer.from_pretrained(ROOT/'model_files_0p5b',local_files_only=True)
    tok.padding_side='left'; tok.pad_token=tok.eos_token
    model=AutoModel.from_pretrained(ROOT/'model_files_0p5b',local_files_only=True,torch_dtype=torch.bfloat16).cuda().eval()
    for name,rows in data.items():
        path=OUT/f'features_{name}.npz'
        if path.exists(): continue
        fs=[]; texts=[prompt(r) for r in members(rows)]
        for start in range(0,len(texts),16):
            enc=tok(texts[start:start+16],padding=True,truncation=False,return_tensors='pt').to('cuda'); assert enc.input_ids.shape[1]<=256
            with torch.inference_mode(): fs.append(model(**enc).last_hidden_state[:,-1].float().cpu().numpy())
            if start%1024==0: print('FEATURE',name,start,len(texts),flush=True)
        np.savez_compressed(path,X=np.concatenate(fs).reshape(len(rows),2,-1),y=np.array([r['y'] for r in rows]))
    del model; torch.cuda.empty_cache()
def projection():
    f=np.load(OUT/'features_development.npz'); dev=f['X'].reshape(-1,f['X'].shape[-1]).astype(float)
    center=dev.mean(0); _,s,v=np.linalg.svd(dev-center,full_matrices=False); basis=v[:24].T; scale=s[:24]/np.sqrt(len(dev)-1)
    assert np.min(scale)>0
    np.savez_compressed(OUT/'fixed_projection.npz',center=center,basis=basis,scale=scale)
def projected(name):
    f=np.load(OUT/f'features_{name}.npz'); p=np.load(OUT/'fixed_projection.npz')
    x=(f['X'].astype(float)-p['center'])@p['basis']/p['scale']
    return np.concatenate([np.ones((*x.shape[:2],1)),x],axis=-1),f['y']
def cal_predict(X,y,seed,mechanism,eta):
    S,b=moment(X,y,.5); Se,be=moment(X,y,.05); Sf,bf=moment(X,y,.95)
    u=np.linalg.solve(S,b); wf=np.linalg.solve(Sf,bf); G=risk(X,y,wf)-risk(X,y,u)
    yy=np.repeat(y,2); target=corrupt(yy,np.tile([-1,1],len(y)),eta,mechanism,seed+999)
    d=X.reshape(-1,X.shape[-1]).T@(target-yy)/len(yy); move=np.linalg.solve(S,d)
    quadratic=-G+move@Se@move; alignment=2*move@(Se@u-be)
    return float(quadratic+alignment),float(quadratic)
def predict():
    check(); projection(); X,y=projected('train'); C,cy=projected('calibration'); rows=[]; heads=[]; factual=[]
    for seed in SEEDS:
        ix=select_anchors(y,600,seed); xt=X[ix]; yt=y[ix]; flat=xt.reshape(-1,25); yy=np.repeat(yt,2)
        xf=xt[np.arange(600),factual_indices(yt,seed+1)]; assert np.linalg.matrix_rank(flat)==np.linalg.matrix_rank(xf)==25
        factual.append(np.linalg.lstsq(xf,yt,rcond=None)[0])
        for mech,eta in [('random',0.)]+[(m,e) for m in ['random','reinforce','oppose'] for e in [.1,.2,.3,.4]]:
            target=corrupt(yy,np.tile([-1,1],600),eta,mech,seed+2)
            heads.append(np.linalg.lstsq(flat,target,rcond=None)[0]); full,quad=cal_predict(C,cy,seed,mech,eta)
            rows.append(dict(seed=seed,mechanism=mech,eta=eta,full=full,quadratic=quad,error_energy=float(np.mean((target-yy)**2))))
    for r in rows: r['rate']=float(np.mean([q['full'] for q in rows if q['seed']==r['seed'] and q['eta']==r['eta']]))
    np.savez_compressed(OUT/'fixed_heads.npz',heads=np.array(heads),factual=np.array(factual))
    write(OUT/'fixed_predictions.json',dict(rows=rows,calibration_condition_number=float(np.linalg.cond(moment(C,cy,.5)[0]))))
    write(OUT/'prediction_receipt.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),prediction_sha256=sha(OUT/'fixed_predictions.json'),heads_sha256=sha(OUT/'fixed_heads.npz'),test_scored=False))
def score():
    check(); receipt=read(OUT/'prediction_receipt.json'); assert receipt['prediction_sha256']==sha(OUT/'fixed_predictions.json')
    T,ty=projected('test'); C,cy=projected('calibration'); D,dy=projected('development'); h=np.load(OUT/'fixed_heads.npz')
    rows=read(OUT/'fixed_predictions.json')['rows']; errors={k:[] for k in ['full','quadratic','rate']}
    for i,r in enumerate(rows): r['observed']=risk(T,ty,h['heads'][i])-risk(T,ty,h['factual'][SEEDS.index(r['seed'])])
    for seed in SEEDS:
        rr=[r for r in rows if r['seed']==seed and r['eta']>0]
        for k in errors: errors[k].append(float(np.mean([abs(r[k]-r['observed']) for r in rr])))
    diff=np.array(errors['full'])-errors['quadratic']; fullci=ci(diff)
    # Calibration resampling keeps each label's count fixed; uses all 32 training outcomes.
    rng=np.random.default_rng(927000); boot=[]
    for rep in range(500):
        ix=np.concatenate([rng.choice(np.flatnonzero(cy==s),size=np.sum(cy==s),replace=True) for s in [-1,1]])
        cb=C[ix]; yb=cy[ix]; vals=[]
        # Common calibration perturbation across all trained pools.
        S,b=moment(cb,yb,.5); Se,be=moment(cb,yb,.05); Sf,bf=moment(cb,yb,.95)
        u=np.linalg.solve(S,b); G=risk(cb,yb,np.linalg.solve(Sf,bf))-risk(cb,yb,u); flat=cb.reshape(-1,25); yy=np.repeat(yb,2)
        for r in rows:
            if not r['eta']: continue
            ct=corrupt(yy,np.tile([-1,1],len(yb)),r['eta'],r['mechanism'],r['seed']+999)
            move=np.linalg.solve(S,flat.T@(ct-yy)/len(yy)); quad=-G+move@Se@move; full=quad+2*move@(Se@u-be)
            vals.append(abs(full-r['observed'])-abs(quad-r['observed']))
        boot.append(float(np.mean(vals)))
        if rep%100==0: print('CAL BOOT',rep,flush=True)
    fixed=dict(rows=rows,mae={k:float(np.mean(v)) for k,v in errors.items()},pool_errors=errors,mae_difference=fullci,
        supported=fullci['high']<0,sign_accuracy={k:float(np.mean([np.sign(r[k])==np.sign(r['observed']) for r in rows if r['eta']>0])) for k in errors},
        calibration_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),calibration_bootstrap_draws=boot)
    w=h['heads'][::13].mean(0)
    def loss(x,y,w): return np.where(y[:,None]==1,1-expit(2*(x@w)),expit(2*(x@w)))
    lc=loss(C,cy,w); lt=loss(T,ty,w); a=float(lc.mean(1).var(ddof=1)); b=float(lc.var(1).mean())
    grid=[(n,k) for n in [16,32,64,128] for k in [1,2,4,8]]; z=np.array([1/(n*k) for n,k in grid]); target=np.array([a/n+b/(n*k) for n,k in grid]); c=float(z@target/(z@z)); sampling=[]
    for n,k in grid:
        rr=np.random.default_rng(927100+n*10+k); ix=rr.integers(len(ty),size=(2000,n)); cue=rr.integers(2,size=(2000,n,k))
        vals=lt[ix[:,:,None],cue].mean(axis=(1,2)); observed=float(vals.var(ddof=1)); centered=(vals-vals.mean())**2
        sampling.append(dict(n=n,k=k,observed=observed,two_level=a/n+b/(n*k),naive=c/(n*k),mc_se=float(centered.std(ddof=1)/np.sqrt(2000))))
    q=(D[:,1]-D[:,0]).mean(0); q[0]=0; q/=np.linalg.norm(q); transfer=[]
    for alpha in [0,.5,1,2]:
        wa=w+(alpha-1)*q*(q@w); pcl=np.clip(expit(2*(C@wa)),1e-10,1-1e-10); lcal=loss(C,cy,wa); ltest=loss(T,ty,wa)
        V=float(lcal.var(1).mean()); p0,p1=pcl.T
        kl=p0*np.log(p0/p1)+(1-p0)*np.log((1-p0)/(1-p1))+p1*np.log(p1/p0)+(1-p1)*np.log((1-p1)/(1-p0)); L=float(kl.mean()/4)
        for delta in [0,.25,.5,.75,.9]:
            weights=np.where(np.array([-1,1])[None,:]==ty[:,None],(1-delta)/2,(1+delta)/2)
            per=(weights*ltest).sum(1)-ltest.mean(1); gap=float(abs(per.mean()))
            draws=[]
            for _ in range(500): draws.append(float(abs(per[rng.integers(len(per),size=len(per))].mean())))
            transfer.append(dict(alpha=alpha,delta=delta,gap=gap,centered=delta*np.sqrt(V),predictive=delta*np.sqrt(L/2),V=V,L=L,gap_bootstrap_95=np.quantile(draws,[.025,.975]).tolist()))
    sampling_summary=dict(a=a,b=b,naive_c=c,rmse={key:float(np.sqrt(np.mean([(r[key]-r['observed'])**2 for r in sampling]))) for key in ['two_level','naive']},cells=sampling)
    write(OUT/'fixed_metrics.json',dict(completed=True,training_pools=32,heads=416,fixed=fixed,sampling=sampling_summary,transfer=transfer,
        prediction_receipt=receipt,scope='Conditional finite-pool diagnostics; fixed code finalized after LoRA results, before fixed-head outcomes. Not fully blinded across arms.'))
    print('FIXED ALL COMPLETED',flush=True)
if __name__=='__main__':
    torch.set_num_threads(4); p=argparse.ArgumentParser(); p.add_argument('mode',choices=['lock','run']); mode=p.parse_args().mode
    if mode=='lock': lock()
    else:
        extract(); predict(); score()
