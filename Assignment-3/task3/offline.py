"""Question-local analysis; all selectors see only the currently revealed prefix."""
import itertools
import time
import numpy as np
from .core import signals,selections,gate,orders,choose_policy,paired_bootstrap
from .scoring import Scorer
from .protocol import BUDGETS

POLICIES=[dict(family='agreement',tau=t,gamma=-1.) for t in (.5,.625,.75,.875,1.)]+[dict(family='quality',tau=t,gamma=g) for t,g in itertools.product((.5,.625,.75,.875,1.),(0.,.1,.25,.5))]
POLICIES += [dict(family=f,tau=2.,gamma=0.) for f in ('agreement','quality')]

def correlation(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.std(a)==0 or np.std(b)==0:return None
 return float(np.corrcoef(a,b)[0,1])

def analyze_question(dataset,qid,rows,metadata,split,examples=False,policies=None):
 scorer=Scorer(dataset,metadata)
 # Correctness is kept separate; it is never passed to signals, grouping or selectors.
 correct=[scorer.correct(r['extracted_answer'],r['gold_answer']) for r in rows]
 fixed=[];adaptive=[];permutations=orders(dataset,qid)
 for order_no,order in enumerate(permutations):
  prefixes={}
  for n in BUDGETS:
   ids=order[:n];revealed=[rows[i] for i in ids]
   groups,reps=scorer.groups([r['extracted_answer'] for r in revealed])
   scores,_=signals(revealed);chosen=selections(groups,scores)
   cost=sum(r['n_generated_text_tokens'] for r in revealed)
   oracle=any(correct[i] for i in ids)
   base=dict(dataset=dataset,sample_id=qid,split=split,order=order_no,N=n,oracle=int(oracle),distinct=len(reps),full_pool_correct_fraction=float(np.mean(correct)))
   for method,index in chosen.items():
    i=ids[index];consumed=ids[:1] if method=='first' else ids
    fixed.append(dict(**base,method=method,selected_rollout_id=rows[i]['rollout_id'],selected_answer=scorer.normalize(rows[i]['extracted_answer']),correct=int(correct[i]),tokens=sum(rows[j]['n_generated_text_tokens'] for j in consumed),sampled=len(consumed),candidate_ids=ids,consumed_ids=consumed,sampling_miss=int(not oracle),selection_miss=int(oracle and not correct[i])))
   prefixes[n]=dict(groups=groups,reps=reps,quality=scores['certainty' if dataset=='math' else 'visconf'],sc=ids[chosen['sc']],cost=cost,oracle=oracle,distinct=len(reps))
  for policy in (POLICIES if policies is None else policies):
   n=32
   for check in (4,8,16):
    p=prefixes[check]
    if gate(p['groups'],p['reps'],p['quality'],policy['tau'],None if policy['family']=='agreement' else policy['gamma']):n=check;break
   p=prefixes[n];i=p['sc'];win=p['groups'][order[:n].index(i)]
   adaptive.append(dict(dataset=dataset,sample_id=qid,split=split,order=order_no,**policy,N=n,tokens=p['cost'],correct=int(correct[i]),selected_rollout_id=rows[i]['rollout_id'],selected_answer=p['reps'][win],candidate_ids=order[:n],incorrect_early=int(n<32 and not correct[i]),recoverable_fixed32=int(n<32 and not correct[i] and correct[prefixes[32]['sc']]),full_pool_correct_fraction=float(np.mean(correct))))
 scores,details=signals(rows);analysis=[]
 lengths=[r['n_generated_text_tokens'] for r in rows]
 for name,values in {**scores,'length':np.asarray(lengths,float)}.items():
  valid=np.isfinite(values)
  if valid.all():
   # Exactly eight per quartile; score ties use canonical rollout ID order.
   indices=np.argsort(values,kind='stable');bottom=np.mean([correct[i] for i in indices[:8]]);top=np.mean([correct[i] for i in indices[-8:]])
   analysis.append(dict(dataset=dataset,sample_id=qid,split=split,score=name,top_accuracy=float(top),bottom_accuracy=float(bottom),quartile_difference=float(top-bottom),length_correlation=correlation(lengths,values),correctness_correlation=correlation(correct,values)))
  else:analysis.append(dict(dataset=dataset,sample_id=qid,split=split,score=name,top_accuracy=None,bottom_accuracy=None,quartile_difference=None,length_correlation=None,correctness_correlation=None))
 token_stats=[]
 for r,d in zip(rows,details):
  token_stats.append(dict(dataset=dataset,sample_id=qid,split=split,rollout_id=r['rollout_id'],**{k:v for k,v in d.items() if k not in ('v','c')}))
 feature_rows=[]
 for i,r in enumerate(rows):
  feature_rows.append(dict(rollout_id=r['rollout_id'],answer=r['extracted_answer'],correct=correct[i],length=lengths[i],**{k:(float(v[i]) if np.isfinite(v[i]) else None) for k,v in scores.items()}))
 example=None
 if examples:
  example=dict(question=rows[0]['question_text'],gold=rows[0]['gold_answer'],candidates=[dict(rollout_id=r['rollout_id'],text=r['generated_text'],answer=r['extracted_answer'],logp=r['logp'],kl_u_p=r['kl_u_p'],**(details[i] if details else {})) for i,r in enumerate(rows)],selection_rule='First three evaluation question IDs in saved permutation; all 32 candidates retained')
 return dict(sample_id=qid,split=split,orders=permutations,fixed=fixed,adaptive=adaptive,analysis=analysis,token_stats=token_stats,features=feature_rows,example=example)

def run_dataset(dataset,reader,metadata,splits,store,monitor,phase,policies=None):
 for split in (phase,):
  start=time.monotonic();hits=0
  for pos,qid in enumerate(splits[split]):
   monitor.check();key=dataset+'/'+qid
   if store.get(key) is not None:hits+=1
   else:
    rows=reader.question(qid)
    result=analyze_question(dataset,qid,rows,metadata.get(qid,{}),split,qid in splits['evaluation'][:3],policies)
    store.put(key,result)
   if pos%10==0 or pos+1==len(splits[split]):
    elapsed=time.monotonic()-start;rate=(pos+1)/max(elapsed,1e-9)
    monitor.log('RESUME' if hits else 'PASS','Offline question checkpoint',dataset=dataset,stage=split,methods='all fixed selectors and stopping grids',budgets=list(BUDGETS),questions_processed=pos+1,total=len(splits[split]),elapsed_s=round(elapsed,1),questions_per_s=rate,eta_s=(len(splits[split])-pos-1)/rate,cache_hits=hits,last_saved=key)
