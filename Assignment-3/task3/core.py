"""Selection functions accept revealed candidates only, never gold labels."""
import numpy as np
from scipy.stats import rankdata
from .protocol import SEED, seed_for
from .scoring import NO_ANSWER

def split_ids(ids):
 ids=sorted(map(str, ids))
 if len(ids)!=1000 or len(set(ids))!=1000: raise ValueError('Expected 1,000 unique question IDs')
 ids=np.random.default_rng(SEED).permutation(ids).tolist()
 return dict(dev=ids[:200], evaluation=ids[200:], live=ids[200:250])

def orders(dataset, qid):
 return np.stack([np.random.default_rng(seed_for('order',dataset,qid,i)).permutation(32) for i in range(10)]).tolist()

def finite_mean(x, default=float('-inf')):
 a=np.asarray(x,dtype=float)
 return float(a.mean()) if len(a) and np.isfinite(a).all() else default

def signals(rows):
 scores={'logp':np.array([finite_mean(r['logp']) for r in rows]),
         'certainty':np.array([finite_mean(r['kl_u_p']) for r in rows]),
         'shortest':-np.array([r['n_generated_text_tokens'] for r in rows],dtype=float)}
 details=[]
 if 'img_attn_total' not in rows[0]: return scores, details
 valid=[np.asarray(r['kl_u_p'],float) for r in rows]
 pooled=np.concatenate([v[np.isfinite(v)] for v in valid])
 lo,hi=np.quantile(pooled,[.05,.95],method='linear') if len(pooled) else (0.,0.)
 vs=[]; zs=[]; ts=[]
 for r,k in zip(rows,valid):
  a=np.asarray(r['img_attn_total'],float); b=np.asarray(r['prompt_text_attn_total'],float); d=np.asarray(r['generated_text_attn_total'],float)
  den=a+b+d
  ok=len(k)>0 and np.isfinite(k).all() and np.isfinite(den).all() and np.isfinite(a).all() and (den>0).all()
  if ok:
   v=a/den; c=np.clip((k-lo)/max(hi-lo,1e-8),0,1)
   V=float(v.mean()); C=float(c.mean()); Z=V*C; T=float(np.mean(v*c))
   corr=float(np.corrcoef(c,v)[0,1]) if np.std(c)>0 and np.std(v)>0 else None
   details.append(dict(v=v.tolist(),c=c.tolist(),C=C,V=V,Z=Z,token_fusion=T,covariance=T-Z,correlation=corr,raw_kl_vision_correlation=(float(np.corrcoef(k,v)[0,1]) if np.std(k)>0 and np.std(v)>0 else None)))
  else:
   V=Z=T=float('-inf'); details.append(dict(v=[],c=[],C=None,V=None,Z=None,token_fusion=None,covariance=None,correlation=None,raw_kl_vision_correlation=None))
  vs.append(V);zs.append(Z);ts.append(T)
 scores.update(vision=np.array(vs),visconf=np.array(zs),token_fusion=np.array(ts))
 return scores,details

def votes(groups, weights):
 totals={}
 for g,w in zip(groups,weights): totals[g]=totals.get(g,0.)+float(w)
 # dict insertion order implements earliest supporting rollout on ties.
 winner=max(totals,key=totals.get)
 return winner,totals

def rwc(groups, scores):
 n=len(groups)
 weights=np.ones(1) if n==1 else (n-rankdata(-np.asarray(scores),method='average'))/(n-1)
 winner,totals=votes(groups,weights)
 values=sorted(totals.values(),reverse=True)
 margin=(values[0]-(values[1] if len(values)>1 else 0))/sum(values)
 return winner,margin,totals

def selections(groups,scores):
 result={'first':0,'sc':groups.index(votes(groups,np.ones(len(groups)))[0])}
 for key,score in scores.items():
  result[key+'_max']=int(np.argmax(score))
  if key in ('logp','certainty','vision','visconf','token_fusion'):
   result[key+'_rwc']=groups.index(rwc(groups,score)[0])
 return result

def gate(groups, representatives, quality, tau, gamma=None):
 win,counts=votes(groups,np.ones(len(groups)))
 unique=sum(v==counts[win] for v in counts.values())==1
 if not unique or representatives[win]==NO_ANSWER or counts[win]/len(groups)<tau: return False
 if gamma is None: return True
 qwin,margin,_=rwc(groups,quality)
 return qwin==win and margin>=gamma

def choose_policy(records, baseline):
 # records are dev-only aggregate records; explicit guard prevents evaluation leakage.
 if any(r['split']!='dev' for r in records): raise ValueError('Threshold fitting requires dev records only')
 eligible=[r for r in records if r['accuracy']>=baseline-.01-1e-12]
 return min(eligible,key=lambda r:(r['tokens'],-r['accuracy'],r['tau'],r['gamma']))

def paired_bootstrap(a,b,seed=SEED):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if a.ndim!=1 or a.shape!=b.shape or not len(a): raise ValueError('Use paired question means')
 rng=np.random.default_rng(seed); d=a-b
 values=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(1000)]
 return dict(difference=float(d.mean()),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),resamples=1000,unit='question')
