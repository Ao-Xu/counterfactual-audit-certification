import os,json,hashlib,tempfile
from pathlib import Path
ROOT=Path(os.environ.get('HANS_POLICY_ROOT',Path(__file__).resolve().parents[1])).resolve()
APP=Path(__file__).resolve().parent
P=json.loads((APP/'protocol.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def safe_path(path):
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Output outside experiment root: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    return p
def write(path,obj):
    path=safe_path(path);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    with os.fdopen(fd,'w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(tmp,path)
def pair_id(a,b):return hashlib.sha256((' '.join(a.lower().split())+'\n'+' '.join(b.lower().split())).encode()).hexdigest()
def verify_lock():
    lock=read(ROOT/'prepared_lock.json')
    for rel,h in lock['sha256'].items():
        if sha(ROOT/rel)!=h:raise RuntimeError('Locked input changed: '+rel)
    return lock
def fingerprint():return {'protocol':sha(APP/'protocol.json'),'locked':sha(ROOT/'data/locked.json'),'tokens':sha(ROOT/'data/encoded.pt'),'train_code':sha(APP/'train.py'),'common_code':sha(APP/'common.py'),'selection_code':sha(APP/'selection.py'),'generation_code':sha(APP/'generation.py'),'model_revision':P['model_revision']}
