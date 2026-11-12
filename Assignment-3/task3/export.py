"""Evidence only: metrics/plots/raw records; no submitted report or interpretations."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .core import choose_policy,paired_bootstrap
from .storage import write_json,sha256
from .protocol import BUDGETS
from .scoring import Scorer

def fit_policies(dataset,splits,store):
 cached=store.get(dataset+'/frozen_policies')
 if cached is not None:return cached
 rows=[];baseline=[]
 for qid in splits['dev']:
  result=store.get(dataset+'/'+qid)
  if result is None or result['split']!='dev':raise ValueError('Missing verified dev result')
  rows.extend(result['adaptive'])
  baseline.append(np.mean([r['correct'] for r in result['fixed'] if r['method']=='sc' and r['N']==32]))
 frame=pd.DataFrame(rows)
 means=frame.groupby(['family','tau','gamma'],sort=True)[['correct','tokens']].mean().reset_index().rename(columns={'correct':'accuracy'})
 means['split']='dev';base=float(np.mean(baseline))
 selected={family:choose_policy(means[means.family==family].to_dict('records'),base) for family in ('agreement','quality')}
 value=dict(dataset=dataset,baseline_accuracy=base,dev_ids=splits['dev'],grid=means.to_dict('records'),selected=selected)
 store.put(dataset+'/frozen_policies',value)
 return value

def csv_file(path,rows):
 rows=list(rows)
 if not rows:raise ValueError('Empty required evidence: '+str(path))
 pd.DataFrame(rows).to_csv(path,index=False)

class RawWriter:
 def __init__(self,path):
  self.path=Path(path);self.writer=None;self.csv=open(self.path.with_suffix('.csv'),'w',newline='');self.cwriter=None
 def append(self,rows):
  if not rows:return
  table=pa.Table.from_pylist(rows)
  if self.writer is None:
   self.writer=pq.ParquetWriter(self.path,table.schema,compression='zstd');self.cwriter=csv.DictWriter(self.csv,fieldnames=list(rows[0]));self.cwriter.writeheader()
  self.writer.write_table(table.cast(self.writer.schema))
  for row in rows:self.cwriter.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
 def close(self):
  if self.writer:self.writer.close()
  self.csv.close()

def export_dataset(dataset,splits,store,out,monitor,reader=None,metadata=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 policies=store.get(dataset+'/frozen_policies')
 if not policies:raise ValueError('Policies must be frozen before held-out evaluation')
 write_json(out/'policies.json',policies)
 fixed_writer=RawWriter(out/'selections.parquet');adaptive_writer=RawWriter(out/'adaptive.parquet')
 question_means=[];adaptive_means=[];analyses=[];token_stats=[];features=[];examples=[];hist=[];order_manifest={}
 cases={"selection_disagreement":[],"premature_incorrect_stop":[],"fixed32_recoverable":[]}
 try:
  for pos,qid in enumerate(splits['evaluation']):
   monitor.check();r=store.get(dataset+'/'+qid)
   if r is None:raise ValueError('Missing evaluation question '+qid)
   fixed_writer.append(r['fixed']);order_manifest[qid]=r['orders']
   f=pd.DataFrame(r['fixed'])
   avg=f.groupby(['method','N'])[['correct','oracle','distinct','tokens','sampled','sampling_miss','selection_miss']].mean().reset_index();avg['sample_id']=qid
   question_means.extend(avg.to_dict('records'))
   selected=[]
   for row in r['adaptive']:
    p=policies['selected'][row['family']]
    if row['tau']==p['tau'] and row['gamma']==p['gamma']:selected.append(row)
   if len(selected)!=20:raise ValueError('Expected ten orders per selected controller')
   adaptive_writer.append(selected)
   at32=[x for x in r['fixed'] if x['N']==32 and x['order']==0 and x['method']!='first']
   scorer=Scorer(dataset,(metadata or {}).get(qid,{}))
   _,answer_groups=scorer.groups([x['selected_answer'] for x in at32])
   conditions={'selection_disagreement':len(answer_groups)>1,'premature_incorrect_stop':any(x['incorrect_early'] for x in selected),'fixed32_recoverable':any(x['recoverable_fixed32'] for x in selected)}
   for category,condition in conditions.items():
    if condition and len(cases[category])<3:
     item=dict(sample_id=qid,fixed=at32,adaptive=selected)
     if reader is not None:
      item['candidates']=[{k:row[k] for k in ('rollout_id','question_text','generated_text','extracted_answer','gold_answer')} for row in reader.question(qid)]
     cases[category].append(item)
   ad=pd.DataFrame(selected)
   am=ad.groupby('family')[['correct','tokens','N','incorrect_early','recoverable_fixed32','full_pool_correct_fraction']].mean().reset_index();am['sample_id']=qid;adaptive_means.extend(am.to_dict('records'))
   hist.extend(ad.groupby(['family','N']).size().rename('count').reset_index().to_dict('records'))
   analyses.extend(r['analysis']);token_stats.extend(r['token_stats']);features.extend(dict(sample_id=qid,**x) for x in r['features'])
   if r['example'] is not None:examples.append(dict(sample_id=qid,**r['example'],selection_records=r['fixed'],stopping_records=selected))
   if pos%100==0:monitor.log('PASS','Export question evidence',dataset=dataset,questions_processed=pos+1,total=800)
 finally:fixed_writer.close();adaptive_writer.close()
 for qid in splits['dev']:order_manifest[qid]=store.get(dataset+'/'+qid)['orders']
 write_json(out/'orders.json',order_manifest)
 write_json(out/'examples.json',examples)
 write_json(out/'case_examples.json',dict(rule='First up to three qualifying questions in the fixed evaluation permutation per category; disagreement uses order 0 at N=32, stopping categories use any of the ten orders. Gold is used only to label retrospective failure categories, never to select answers or fit thresholds.',cases=cases,counts={k:len(v) for k,v in cases.items()}))
 q=pd.DataFrame(question_means);a=pd.DataFrame(adaptive_means)
 q.to_csv(out/'question_means.csv',index=False);a.to_csv(out/'adaptive_question_means.csv',index=False)
 summary=q.groupby(['method','N']).mean(numeric_only=True).reset_index();summary.to_csv(out/'summary.csv',index=False)
 ads=a.groupby('family').mean(numeric_only=True).reset_index();ads.to_csv(out/'adaptive_summary.csv',index=False)
 pd.DataFrame(hist).groupby(['family','N'])['count'].sum().reset_index().to_csv(out/'stopping_histogram.csv',index=False)
 intervals=[]
 for n in BUDGETS:
  subset=q[q.N==n].pivot(index='sample_id',columns='method',values='correct')
  for method in subset.columns:
   if method!='sc':intervals.append(dict(N=n,method=method,reference='sc',**paired_bootstrap(subset[method],subset.sc)))
 # Adaptive vs fixed-32 paired comparisons on the exact same questions.
 baseline=q[(q.N==32)&(q.method=='sc')].set_index('sample_id').correct
 for family in a.family.unique():
  vals=a[a.family==family].set_index('sample_id').correct.reindex(baseline.index)
  intervals.append(dict(N=32,method='adaptive_'+family,reference='sc32',**paired_bootstrap(vals,baseline)))
 csv_file(out/'paired_intervals.csv',intervals);csv_file(out/'signal_per_question.csv',analyses);csv_file(out/'rollout_features.csv',features)
 pd.DataFrame(analyses).groupby('score').mean(numeric_only=True).reset_index().to_csv(out/'signal_summary.csv',index=False)
 if token_stats:
  csv_file(out/'token_associations.csv',token_stats)
  t=pd.DataFrame(token_stats).groupby('sample_id').mean(numeric_only=True)
  t.to_csv(out/'token_associations_question_means.csv')
  t.mean().to_frame('mean_over_questions').to_csv(out/'token_associations_summary.csv')
 nearest=[]
 sc=summary[summary.method=='sc']
 for row in ads.to_dict('records'):
  closest=sc.iloc[(sc.tokens-row['tokens']).abs().argmin()]
  nearest.append(dict(family=row['family'],adaptive_accuracy=row['correct'],adaptive_tokens=row['tokens'],fixed_N=int(closest.N),fixed_accuracy=closest.correct,fixed_tokens=closest.tokens))
 csv_file(out/'similar_cost_comparison.csv',nearest)
 a.assign(pool_fraction_bin=pd.cut(a.full_pool_correct_fraction,[0,.25,.5,.75,1],include_lowest=True)).groupby(['family','pool_fraction_bin'],observed=True)[['N','correct','incorrect_early','recoverable_fixed32']].mean().reset_index().to_csv(out/'retrospective_pool_fraction.csv',index=False)
 plots(dataset,summary,ads,pd.DataFrame(analyses),pd.DataFrame(hist),examples,out)
 monitor.log('PASS','Dataset evidence exported',dataset=dataset,path=str(out))

def plots(dataset,summary,adaptive,analysis,hist,examples,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for axis,filename,label in [('N','accuracy_candidates.png','Candidate budget'),('tokens','accuracy_tokens.png','Generated tokens (offline proxy)')]:
  fig,ax=plt.subplots(figsize=(10,6))
  for method,group in summary.groupby('method'):ax.plot(group[axis],group.correct,marker='.',label=method)
  sc=summary[summary.method=='sc'];ax.plot(sc[axis],sc.oracle,'k--',label='oracle coverage')
  if axis=='tokens':
   for row in adaptive.to_dict('records'):ax.scatter(row['tokens'],row['correct'],marker='*',s=130,label='adaptive '+row['family'])
  ax.set(xlabel=label,ylabel='Accuracy / coverage',title=dataset);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/filename,dpi=150);plt.close(fig)
 fig,ax=plt.subplots();analysis.groupby('score').quartile_difference.mean().plot.bar(ax=ax);ax.set(ylabel='Top minus bottom quartile accuracy',title='Within-question associations');fig.tight_layout();fig.savefig(out/'signal_quartiles.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots();hist.groupby(['family','N'])['count'].sum().unstack(0).plot.bar(ax=ax);ax.set(ylabel='Question-order count');fig.tight_layout();fig.savefig(out/'stopping_distribution.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots();sc=summary[summary.method=='sc'];ax.plot(sc.N,sc.sampling_miss,label='sampling miss');ax.plot(sc.N,sc.selection_miss,label='selection miss');ax.legend();ax.set(xlabel='N',ylabel='Fraction');fig.tight_layout();fig.savefig(out/'failure_decomposition.png',dpi=150);plt.close(fig)
 if dataset=='mathvista':
  fig,axes=plt.subplots(3,1,figsize=(10,8))
  for ax,example in zip(axes,examples):
   candidate=example['candidates'][0];ax.plot(candidate['c'],label='calibrated confidence');ax.plot(candidate['v'],label='visual engagement');ax.set_title(example['sample_id']+' rollout 0');ax.legend()
  fig.tight_layout();fig.savefig(out/'token_trajectories.png',dpi=150);plt.close(fig)
  fig,ax=plt.subplots()
  for method in ('visconf_max','visconf_rwc','token_fusion_max','token_fusion_rwc'):
   d=summary[summary.method==method];ax.plot(d.N,d.correct,label=method)
  ax.legend();ax.set(xlabel='N',ylabel='Accuracy',title='Fusion ablation');fig.tight_layout();fig.savefig(out/'fusion_ablation.png',dpi=150);plt.close(fig)

def export_live(ids,store,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);records=[];calls=[]
 for qid in ids:
  r=store.get('live_summary/'+qid)
  if r is None:raise ValueError('Missing live question '+qid)
  records.extend(r)
  for name in ('single','parallel0','parallel1','parallel2','revision'):
   call=store.get('live/'+qid+'/'+name)
   if call is None:raise ValueError('Missing live call')
   calls.append(dict(call=name,**call))
 csv_file(out/'per_question.csv',records);write_json(out/'generations.json',calls)
 frame=pd.DataFrame(records)
 summary=frame.groupby('strategy').mean(numeric_only=True).reset_index()
 summary['truncation_rate']=summary.truncated_calls/summary.calls
 summary.to_csv(out/'summary.csv',index=False)
 transitions=frame[frame.strategy=='refinement'].groupby(['initial_correct','correct']).size().rename('questions').reset_index()
 transitions.to_csv(out/'revision_transitions.csv',index=False)
 pivot=frame.pivot(index='sample_id',columns='strategy',values='correct')
 csv_file(out/'paired_intervals.csv',[dict(method=m,reference='parallel',**paired_bootstrap(pivot[m],pivot.parallel)) for m in ('single','refinement')])
 transition_examples={}
 for category in ('wrong_to_correct','correct_to_wrong'):
  selected_ids=[q for q in ids if any(r['sample_id']==q and r['strategy']=='refinement' and r[category] for r in records)][:3]
  transition_examples[category]=[dict(sample_id=q,calls=[c for c in calls if c['sample_id']==q]) for q in selected_ids]
 write_json(out/'examples.json',dict(rule='First three fixed live IDs independent of outcomes; additionally first up to three qualifying IDs in fixed order per retrospective revision transition category. Empty categories mean no such cases occurred.',questions=[dict(sample_id=qid,calls=[c for c in calls if c['sample_id']==qid]) for qid in ids[:3]],transition_examples=transition_examples))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(12,4))
 for ax,metric in zip(axes,['correct','output_tokens','elapsed_s']):summary.plot.bar(x='strategy',y=metric,ax=ax,legend=False);ax.set_title(metric)
 fig.tight_layout();fig.savefig(out/'live_comparison.png',dpi=150);plt.close(fig)
