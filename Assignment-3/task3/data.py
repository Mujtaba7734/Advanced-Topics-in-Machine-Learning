"""Official release download, metadata validation and bounded-memory Parquet joins."""
import json
from pathlib import Path
import shutil
import urllib.request
import numpy as np
import pyarrow.parquet as pq
from .protocol import FILES, REVISION
from .storage import sha256,write_json
KEYS=('dataset','sample_id','rollout_id')
META_URL='https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json'

def download(root,monitor):
 root=Path(root); root.mkdir(parents=True,exist_ok=True)
 manifest_path=root/'release.json'
 old=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
 entries={}
 expected=json.loads(Path(__file__).with_name('release_manifest.json').read_text())
 import gdown
 for name,fid in {**FILES,'mathvista/testmini.json':META_URL}.items():
  monitor.check();p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
  prior=old.get(name,{})
  if p.exists() and prior.get('source')==fid and sha256(p)==prior.get('sha256'):
   monitor.log('SKIP: verified cache',name)
  else:
   monitor.log('START','Download '+name,source=fid)
   tmp=p.with_name(p.name+'.partial')
   if fid.startswith('https:'):
    urllib.request.urlretrieve(fid,tmp)
   else:
    result=gdown.download(id=fid,output=str(tmp),quiet=False,resume=True)
    if result is None: raise IOError('Official release inaccessible: '+fid)
   if name.endswith('.parquet'):
    pf=pq.ParquetFile(tmp)
    if pf.metadata.num_rows!=32000: raise ValueError('Official file must have 32000 rows')
   elif name.endswith('.json'):
    if len(json.loads(tmp.read_text()))!=1000:raise ValueError('MathVista metadata must contain 1000 questions')
   elif REVISION not in tmp.read_text():raise ValueError('Official README model revision changed')
   tmp.replace(p)
  entries[name]=dict(source=fid,sha256=sha256(p),bytes=p.stat().st_size)
  if entries[name]!=expected[name]:raise ValueError('Official release changed or download corrupt: '+name)
  write_json(manifest_path,{**old,**entries})
 return entries

class QuestionReader:
 def __init__(self,folder,expected_dataset=None):
  self.folder=Path(folder);self.files={};self.index={};self.schemas={};self.expected_dataset=expected_dataset
  if expected_dataset=='mathvista' and not (self.folder/'attention_metrics.parquet').exists():raise ValueError('MathVista requires attention metrics')
  names=['generations','probability_metrics']
  if (self.folder/'attention_metrics.parquet').exists():names.append('attention_metrics')
  for name in names:
   pf=pq.ParquetFile(self.folder/(name+'.parquet'));self.files[name]=pf
   self.schemas[name]=str(pf.schema_arrow)
   if pf.metadata.num_rows!=32000:raise ValueError('Expected 32000 rows: '+name)
   mapping={};seen=set()
   for rg in range(pf.num_row_groups):
    for r in pf.read_row_group(rg,columns=list(KEYS)).to_pylist():
     key=tuple(r[k] for k in KEYS)
     if key in seen:raise ValueError('Duplicate join key')
     if expected_dataset and r['dataset']!=expected_dataset:raise ValueError('Unexpected dataset identity')
     seen.add(key); mapping.setdefault(str(r['sample_id']),set()).add(rg)
   self.index[name]=mapping
   if name=='generations':self.keys=seen
   elif seen!=self.keys:raise ValueError('Incomplete join keys')
  self.ids=sorted(self.index['generations'])
  if len(self.ids)!=1000:raise ValueError('Expected 1000 questions')
  counts={}
  for key in self.keys:counts.setdefault(str(key[1]),set()).add(key[2])
  if any(v!=set(range(32)) for v in counts.values()):raise ValueError('Expected rollout IDs 0..31')
 def question(self,qid):
  tables={}
  for name,pf in self.files.items():
   records=[]
   for rg in self.index[name][str(qid)]:
    records.extend(r for r in pf.read_row_group(rg).to_pylist() if str(r['sample_id'])==str(qid))
   tables[name]={tuple(r[k] for k in KEYS):r for r in records}
  joined=[]
  for key in sorted(tables['generations'],key=lambda k:k[2]):
   r=dict(tables['generations'][key]);r.update({k:v for k,v in tables['probability_metrics'][key].items() if k not in KEYS})
   if 'attention_metrics' in tables:
    a=tables['attention_metrics'][key]
    r.update({k:v for k,v in a.items() if k not in KEYS and k!='n_generated_text_tokens'})
    if a['n_generated_text_tokens']!=list(range(r['n_generated_text_tokens'])):raise ValueError('Attention preceding-token alignment mismatch')
   validate_row(r);joined.append(r)
  if len({r['question_text'] for r in joined})!=1 or len({r['gold_answer'] for r in joined})!=1:raise ValueError('Inconsistent question or gold')
  return joined

def validate_row(r):
 n=r['n_generated_text_tokens']
 fields=['generated_token_ids','logp','kl_u_p']
 if 'img_attn_total' in r:fields+=['img_attn_total','prompt_text_attn_total','generated_text_attn_total']
 for key in fields:
  if len(r[key])!=n:raise ValueError('Token alignment: '+key)
 # Stop IDs from the pinned Qwen tokenizer; text measurements must exclude termination.
 if any(t in (151643,151645) for t in r['generated_token_ids']):raise ValueError('Termination token in retained signals')
 for key in fields[1:]:
  a=np.asarray(r[key],float)
  if not np.isfinite(a).all():raise ValueError('Nonfinite official token measurements: '+key)
 if any(x>1e-5 for x in r['logp']) or any(x < -1e-5 for x in r['kl_u_p']):raise ValueError('Invalid probability measurements')
