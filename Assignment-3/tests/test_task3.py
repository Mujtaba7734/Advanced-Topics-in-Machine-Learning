"""Synthetic fixtures are used only for tests, never experiment outputs."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from task3.core import (split_ids,orders,signals,rwc,selections,gate,choose_policy,paired_bootstrap)
from task3.scoring import Scorer,boxed,NO_ANSWER
from task3.data import validate_row,QuestionReader
from task3.storage import Store,sha256,SafeStop
from task3.live import messages,accounting,run_live,CALLS
from task3.protocol import SUFFIX,REVIEW


def row(i=0):
 return dict(dataset='mathvista',sample_id='q',rollout_id=i,extracted_answer=str(i%2),gold_answer='0',question_text='question',generated_text='test',generated_token_ids=[1,2,3,4],n_generated_text_tokens=4,hit_max_new_tokens=False,logp=[-1.,-2.,-3.,-4.],kl_u_p=[0.,1.,2.,3.],img_attn_total=[1.,2.,3.,4.],prompt_text_attn_total=[4.,3.,2.,1.],generated_text_attn_total=[0.,0.,0.,0.])

def test_splits_and_orders():
 ids=[str(i) for i in range(1000)];s=split_ids(ids)
 assert s==split_ids(ids[::-1]);assert len(s['dev'])==200;assert len(s['evaluation'])==800
 expected=np.random.default_rng(6304).permutation(sorted(ids)).tolist()
 assert s['live']==expected[200:250];assert not set(s['dev'])&set(s['evaluation'])
 assert orders('math','q')==orders('math','q');assert all(sorted(o)==list(range(32)) for o in orders('math','q'))
 with pytest.raises(ValueError):split_ids(ids[:-1])

@pytest.mark.parametrize('text,want',[(r'\boxed{1} then \boxed{\frac{1}{2}}',r'\frac{1}{2}'),(r'\boxed{1} \boxed{',None),(r'\boxed{}',None),('42',None),(r'\boxed{ 5 }','5')])
def test_boxed(text,want):assert boxed(text)==want

def test_math_equivalence_and_failures():
 s=Scorer('math');assert s.equivalent('0.5',r'\frac{1}{2}');assert not s.correct(None,None)
 groups,reps=s.groups([None,'',r'\frac{1}{2}','0.5','3']);assert groups==[0,0,1,1,2];assert reps[0]==NO_ANSWER
 assert s.correct('0.5',r'\frac{1}{2}')

def test_mathvista_rules():
 s=Scorer('mathvista',dict(question_type='multi_choice',choices=['cat','dog']))
 assert s.normalize('dog')=='B';assert s.normalize('(A) cat')=='A';assert s.normalize(None)==NO_ANSWER
 assert s.normalize(NO_ANSWER)==NO_ANSWER;assert s.normalize('dogs')=='B'
 assert Scorer('mathvista',dict(answer_type='float',precision=2)).equivalent('1.234','1.23')
 assert Scorer('mathvista',dict(answer_type='integer')).normalize('3.0')=='3'
 assert Scorer('mathvista',dict(answer_type='integer')).normalize('bad')==NO_ANSWER

def test_alignment():
 r=row();validate_row(r);r['logp'].pop()
 with pytest.raises(ValueError,match='alignment'):validate_row(r)
 r=row();r['generated_token_ids'][0]=151645
 with pytest.raises(ValueError,match='Termination'):validate_row(r)

def test_scoring_raw_and_prefix():
 a=row();b=row(1);b['kl_u_p']=[100.,101.,102.,103.]
 scores,d=signals([a]);both,_=signals([a,b])
 assert scores['logp'][0]==-2.5 and scores['certainty'][0]==1.5
 assert scores['vision'][0]==pytest.approx(.5)
 assert scores['visconf'][0]!=both['visconf'][0]
 k=np.array(a['kl_u_p']);lo,hi=np.quantile(k,[.05,.95],method='linear');c=np.clip((k-lo)/(hi-lo),0,1)
 assert d[0]['c']==pytest.approx(c)
 assert d[0]['token_fusion']-d[0]['Z']==pytest.approx(np.cov(c,d[0]['v'],bias=True)[0,1])
 # Unrevealed rows cannot affect prefix scores.
 b['kl_u_p']=[999.]*4;again,_=signals([a]);assert np.array_equal(scores['visconf'],again['visconf'])

@pytest.mark.parametrize('mode',['empty','nan','zero_den'])
def test_invalid_scores(mode):
 r=row()
 if mode=='empty':
  for k in ('logp','kl_u_p','img_attn_total','prompt_text_attn_total','generated_text_attn_total'):r[k]=[]
 elif mode=='nan':r['kl_u_p'][0]=float('nan')
 else:
  for k in ('img_attn_total','prompt_text_attn_total','generated_text_attn_total'):r[k]=[0.]*4
 s,_=signals([r]);assert s['visconf'][0]==-np.inf;assert selections([0],s)['visconf_max']==0

def test_ranks_ties_and_n1():
 winner,margin,v=rwc([0,1,1,0],[4,4,2,1]);assert v[0]==pytest.approx(5/6);assert v[1]==pytest.approx(7/6)
 assert winner==1;assert margin==pytest.approx(1/6)
 assert rwc([8,9],[1,1])[0]==8
 assert rwc([8],[float('-inf')])[:2]==(8,1.)
 assert all(i==0 for i in selections([1],{'certainty':np.array([2.])}).values())
 assert selections([5,4],{'certainty':np.array([2.,2.])})['certainty_max']==0

def test_stopping_gates():
 assert not gate([0,0,1,1],['a','b'],[1]*4,.5)
 assert not gate([0]*4,[NO_ANSWER],[1]*4,.5)
 assert gate([0,0,0,1],['a','b'],[4,3,2,1],.75,0.)
 assert not gate([0,0,0,1],['a','b'],[1,1,1,9],.75,.1)
 assert gate([0]*4,['a'],[1]*4,1.,1.)


def test_dev_only_selection():
 records=[dict(split='dev',family='agreement',tau=t,gamma=-1,accuracy=a,tokens=c) for t,a,c in [(.5,.89,30),(.75,.9,40),(1.,.91,40),(2.,.91,50)]]
 assert choose_policy(records,.91)['tau']==1.
 records[0]['split']='evaluation'
 with pytest.raises(ValueError):choose_policy(records,.91)


def test_question_bootstrap():
 result=paired_bootstrap([.1,.2,.3],[.1,.2,.3]);assert result['low']==result['high']==0;assert result['resamples']==1000
 with pytest.raises(ValueError):paired_bootstrap([[1,0]],[[0,1]])


def test_integrity_and_resume(tmp_path):
 s=Store(tmp_path,'identity');s.put('q',{'step':1});s.put('q',{'step':2});assert s.get('q')=={'step':2}
 pointer=json.loads((tmp_path/'pointers/q.json').read_text());(tmp_path/pointer['object']).write_text('partial')
 assert s.get('q')=={'step':1};assert Store(tmp_path,'different').get('q') is None
 (tmp_path/'pointers/q.partial').write_text('broken');assert s.get('q')=={'step':1}


def test_live_prompts_accounting():
 m=messages('problem','initial');assert m[1]['content']=='problem\n\n'+SUFFIX;assert m[2]['content']=='initial';assert m[3]['content']==REVIEW
 c={name:dict(input_tokens=2,output_tokens=3,elapsed_s=4) for name,_ in CALLS}
 a=accounting(c);assert a['parallel']['output_tokens']==9;assert a['refinement']['input_tokens']==4


def test_live_shared_initial_resume(tmp_path):
 class Reader:
  def question(self,q):return [dict(question_text='test',gold_answer='1')]
 class Monitor:
  def check(self,*a,**k):pass
  def log(self,*a,**k):pass
 class Fake:
  count=0;interrupt=True
  def __init__(self,*a):pass
  def smoke(self,*a):pass
  def generate(self,msgs,cap,seed,label):
   Fake.count+=1
   if Fake.interrupt and Fake.count==3:raise SafeStop('test interruption')
   return dict(text=r'\boxed{1}',answer='1',messages=msgs,seed=seed,input_tokens=5,output_tokens=6,elapsed_s=1,hit_max_new_tokens=False,peak_allocated_bytes=1,peak_reserved_bytes=1)
 store=Store(tmp_path,'test')
 with pytest.raises(SafeStop):run_live(Reader(),['q'],store,Monitor(),tmp_path,Fake)
 assert store.get('live/q/parallel0') is not None
 Fake.interrupt=False;run_live(Reader(),['q'],store,Monitor(),tmp_path,Fake)
 assert Fake.count==6 # 5 completed distinct calls + 1 interrupted call.
 revision=store.get('live/q/revision');assert revision['messages'][2]['content']==store.get('live/q/parallel0')['text']
 before=Fake.count;run_live(Reader(),['q'],store,Monitor(),tmp_path,Fake);assert Fake.count==before
 from task3.export import export_live
 export_live(['q'],store,tmp_path/'exports')
 import pandas as pd
 exported=pd.read_csv(tmp_path/'exports/per_question.csv')
 assert len(exported)==3;assert exported.set_index('strategy').loc['refinement','output_tokens']==12


def test_join_keys_and_attention_alignment(tmp_path):
 import pyarrow as pa
 import pyarrow.parquet as pq
 # Build only a schema/key fixture (1000 x 32 tiny rows), not experimental data.
 base=[dict(dataset='math',sample_id=str(i),rollout_id=j) for i in range(1000) for j in range(32)]
 pq.write_table(pa.Table.from_pylist(base),tmp_path/'generations.parquet')
 pq.write_table(pa.Table.from_pylist(base),tmp_path/'probability_metrics.parquet')
 reader=QuestionReader(tmp_path,'math');assert len(reader.keys)==32000
 base[-1]=base[0]
 pq.write_table(pa.Table.from_pylist(base),tmp_path/'probability_metrics.parquet')
 with pytest.raises(ValueError,match='Duplicate'):QuestionReader(tmp_path,'math')
 base[-1]=dict(dataset='math',sample_id='other',rollout_id=31)
 pq.write_table(pa.Table.from_pylist(base),tmp_path/'probability_metrics.parquet')
 with pytest.raises(ValueError,match='Incomplete'):QuestionReader(tmp_path,'math')

def test_offline_no_gold_leakage_and_policy_freeze():
 from task3.offline import analyze_question
 rows=[row(i) for i in range(32)]
 a=analyze_question('mathvista','q',rows,{'answer_type':'integer'},'evaluation',policies=[dict(family='agreement',tau=2.,gamma=0.)])
 changed=copy.deepcopy(rows)
 for r in changed:r['gold_answer']='123'
 b=analyze_question('mathvista','q',changed,{'answer_type':'integer'},'evaluation',policies=[dict(family='agreement',tau=2.,gamma=0.)])
 assert [(x['selected_rollout_id'],x['tokens']) for x in a['fixed']]==[(x['selected_rollout_id'],x['tokens']) for x in b['fixed']]
 assert [(x['N'],x['selected_rollout_id']) for x in a['adaptive']]==[(x['N'],x['selected_rollout_id']) for x in b['adaptive']]
 assert all(x['N']==32 for x in a['adaptive'])
 assert all(x['tokens']==4 for x in a['fixed'] if x['method']=='first')


def test_completion_cannot_pass_empty_store(tmp_path):
 from task3.run import verify
 s=Store(tmp_path/'checkpoints','test');splits={d:split_ids([str(i) for i in range(1000)]) for d in ('math','mathvista')}
 result=verify(tmp_path,s,splits)
 assert not result['fully_completed'];assert result['live_generation_calls']==0;assert result['offline_datasets']==0


def test_exports_verified_by_hash(tmp_path):
 from task3.run import checkpoint_exports,verified_exports
 s=Store(tmp_path/'store','test');out=tmp_path/'out';out.mkdir();(out/'a.csv').write_text('x\n1\n')
 assert not verified_exports(s,'export',out,['a.csv'])
 checkpoint_exports(s,'export',out,['a.csv']);assert verified_exports(s,'export',out,['a.csv'])
 (out/'a.csv').write_text('partial');assert not verified_exports(s,'export',out,['a.csv'])


def test_attention_count_list_separate_from_scalar(tmp_path):
 # Exercise a real reader's merge logic using one question's row-group tables.
 import pyarrow as pa
 class PF:
  def __init__(self,rows):self.rows=rows
  def read_row_group(self,*args,**kwargs):return pa.Table.from_pylist(self.rows)
 reader=object.__new__(QuestionReader);rows=[row(i) for i in range(32)]
 reader.files={'generations':PF([{k:v for k,v in r.items() if k not in ['logp','kl_u_p','img_attn_total','prompt_text_attn_total','generated_text_attn_total']} for r in rows]),'probability_metrics':PF([{k:r[k] for k in ['dataset','sample_id','rollout_id','logp','kl_u_p']} for r in rows]),'attention_metrics':PF([dict(**{k:r[k] for k in ['dataset','sample_id','rollout_id','img_attn_total','prompt_text_attn_total','generated_text_attn_total']},n_generated_text_tokens=[0,1,2,3]) for r in rows])}
 reader.index={key:{'q':{0}} for key in reader.files}
 assert reader.question('q')[0]['n_generated_text_tokens']==4
 reader.files['attention_metrics'].rows[0]['n_generated_text_tokens']=[1,2,3,4]
 with pytest.raises(ValueError,match='preceding-token'):reader.question('q')

def test_notebook_build_and_incomplete_summary(tmp_path):
 import ast
 import nbformat
 from task3.build_notebook import build,SUMMARY
 path=tmp_path/'run.ipynb';build('a'*40,path);n=nbformat.read(path,as_version=4)
 assert len(n.cells)==5
 for cell in n.cells:
  if cell.cell_type=='code':ast.parse(cell.source)
 assert 'capture_output=True' not in path.read_text()
 assert 'python -u' not in path.read_text() or '-u' in path.read_text()
 (tmp_path/'final_verification.json').write_text(json.dumps({'fully_completed':False,'state':'safe stopped'}))
 lines=[];exec(SUMMARY,{'ROOT':tmp_path,'Path':Path,'json':json,'log':lines.append})
 assert any('INCOMPLETE' in x for x in lines);assert not any('TASK 3 FULLY COMPLETED' in x for x in lines)
 (tmp_path/'final_verification.json').write_text(json.dumps({'fully_completed':True,'state':'fully completed'}))
 with pytest.raises(RuntimeError):exec(SUMMARY,{'ROOT':tmp_path,'Path':Path,'json':json,'log':lines.append})
