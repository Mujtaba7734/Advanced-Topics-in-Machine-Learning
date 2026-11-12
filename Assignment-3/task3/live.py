"""Five independently seeded calls/question; shared parallel initial is never regenerated."""
import time
import importlib.metadata
import numpy as np
from .protocol import MODEL,REVISION,SYSTEM,SUFFIX,REVIEW,SAMPLING,seed_for,digest
from .scoring import Scorer,boxed
from .core import votes
from .storage import SafeStop,sha256,write_json
from pathlib import Path
import json

CALLS=(('single',3072),('parallel0',1024),('parallel1',1024),('parallel2',1024),('revision',2048))

def messages(question,initial=None):
 m=[dict(role='system',content=SYSTEM),dict(role='user',content=question+'\n\n'+SUFFIX)]
 if initial is not None:m += [dict(role='assistant',content=initial),dict(role='user',content=REVIEW)]
 return m

def accounting(calls):
 return {strategy:{k:sum(calls[c][k] for c in ids) for k in ('input_tokens','output_tokens','elapsed_s')} for strategy,ids in dict(single=['single'],parallel=['parallel0','parallel1','parallel2'],refinement=['parallel0','revision']).items()}

class Generator:
 def __init__(self,cache,monitor):
  import torch
  from transformers import AutoTokenizer,AutoModelForCausalLM
  if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Select exactly one T4 GPU')
  if 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('Protocol requires a T4 GPU')
  self.torch=torch;self.monitor=monitor
  monitor.log('START','Load pinned FP16 model',model=MODEL,revision=REVISION,gpu=torch.cuda.get_device_name(0),vram_bytes=torch.cuda.get_device_properties(0).total_memory)
  from huggingface_hub import snapshot_download
  snapshot=Path(snapshot_download(MODEL,revision=REVISION,cache_dir=str(cache),allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))
  hashes={str(p.relative_to(snapshot)):sha256(p) for p in snapshot.rglob('*') if p.is_file()}
  manifest_path=Path(cache)/(REVISION+'-verified.json')
  expected=dict(model=MODEL,revision=REVISION,files=hashes)
  if manifest_path.exists() and json.loads(manifest_path.read_text())!=expected:raise ValueError('Pinned model snapshot checksum mismatch')
  write_json(manifest_path,expected)
  self.tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
  self.model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda').eval()
  self.model.requires_grad_(False)
  if next(self.model.parameters()).dtype!=torch.float16:raise RuntimeError('FP16 weights required')
  self.model.generation_config.pad_token_id=self.tokenizer.eos_token_id
  self.context_limit=self.model.config.max_position_embeddings
  monitor.log('PASS','Model loaded; loading excluded from inference timings')
 def generate(self,msgs,cap,seed,label,force_length=False):
  from transformers import StoppingCriteria,StoppingCriteriaList,set_seed
  torch=self.torch;set_seed(seed)
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
  prompt=self.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
  inputs=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
  n=inputs.input_ids.shape[1]
  if n+cap>self.context_limit:raise RuntimeError('Required prompt/output exceeds model context; no truncation allowed')
  monitor=self.monitor;last=time.monotonic()
  class Progress(StoppingCriteria):
   def __call__(self,input_ids,scores,**kwargs):
    nonlocal last
    now=time.monotonic()
    if now-last>=20:
     monitor.log('RESUME','Live generation progress',call=label,generated_tokens=input_ids.shape[1]-n,cap=cap,input_tokens=n,elapsed_s=round(now-start,1),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());last=now
    # Interrupted calls are NOT cached; next session regenerates with the same seed.
    monitor.check()
    return False
  try:
   with torch.inference_mode():
    result=self.model.generate(**inputs,max_new_tokens=cap,**SAMPLING,stopping_criteria=StoppingCriteriaList([Progress()]),**({'min_new_tokens':cap} if force_length else {}))
   torch.cuda.synchronize()
   ids=result[0,n:].tolist();text=self.tokenizer.decode(ids,skip_special_tokens=True);elapsed=time.monotonic()-start
   value=dict(messages=msgs,prompt=prompt,seed=seed,cap=cap,text=text,answer=boxed(text),generated_token_ids=ids,input_tokens=n,output_tokens=len(ids),hit_max_new_tokens=len(ids)>=cap,elapsed_s=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),model=MODEL,revision=REVISION)
   return value
  except torch.cuda.OutOfMemoryError as exc:raise RuntimeError('Required FP16 protocol exceeds T4 memory. No model, cap or precision substitution performed.') from exc

 def smoke(self,questions,store):
  # Test maximum output and longest revision prefill; diagnostic samples never enter results.
  q=max(questions,key=lambda q:len(self.tokenizer.encode(q)))
  first=self.generate(messages(q),3072,seed_for('smoke','long'),'GPU memory smoke: 3072',True)
  initial=self.tokenizer.decode(first['generated_token_ids'][:1024],skip_special_tokens=True)
  second=self.generate(messages(q,initial),2048,seed_for('smoke','revision'),'GPU memory smoke: revision 2048',True)
  total=self.torch.cuda.get_device_properties(0).total_memory
  if max(first['peak_reserved_bytes'],second['peak_reserved_bytes'])>total-256*1024**2:raise RuntimeError('Insufficient measured T4 headroom')
  store.put('gpu_smoke',dict(state='GPU smoke tested',long=first,revision=second,gpu=self.torch.cuda.get_device_name(0),vram_bytes=total))
  self.monitor.log('PASS','Genuine T4 max-token memory smoke',long_tokens=first['output_tokens'],revision_tokens=second['output_tokens'])

def run_live(reader,ids,store,monitor,model_cache,generator_factory=Generator):
 scorer=Scorer('math');generator=None
 for qi,qid in enumerate(ids):
  qrows=reader.question(qid);question=qrows[0]['question_text'];gold=qrows[0]['gold_answer'];calls={}
  for name,cap in CALLS:
   monitor.check();msgs=messages(question,calls['parallel0']['text'] if name=='revision' else None)
   seed=seed_for('live',qid,name)
   identity=digest(dict(messages=msgs,cap=cap,seed=seed,model=MODEL,revision=REVISION,sampling=SAMPLING))
   key='live/'+qid+'/'+name;cached=store.get(key)
   if cached is not None:
    if cached['call_identity']!=identity:raise ValueError('Generation cache identity mismatch')
    calls[name]=cached;monitor.log('SKIP: verified cache','Live generation',question=qi+1,total=50,call=name,last_checkpoint=key);continue
   if generator is None:
    generator=generator_factory(model_cache,monitor)
    # Each new GPU session measures memory; old hardware markers cannot bypass smoke.
    generator.smoke([reader.question(i)[0]['question_text'] for i in ids],store)
   monitor.check(estimated_seconds=900)
   monitor.log('START','Live generation',question=qi+1,total=50,call=name,cap=cap,seed=seed)
   value=generator.generate(msgs,cap,seed,f'{qi+1}/50 {name}');value.update(call_identity=identity,question=question,sample_id=qid)
   pointer=store.put(key,value);calls[name]=value
   monitor.log('PASS','Live call saved and verified',question=qi+1,total=50,call=name,generated_tokens=value['output_tokens'],cap=cap,input_tokens=value['input_tokens'],total_question_input_tokens=sum(c['input_tokens'] for c in calls.values()),total_question_output_tokens=sum(c['output_tokens'] for c in calls.values()),elapsed_s=value['elapsed_s'],peak_allocated_bytes=value['peak_allocated_bytes'],peak_reserved_bytes=value['peak_reserved_bytes'],last_checkpoint=pointer)
  groups,reps=scorer.groups([calls['parallel'+str(i)]['answer'] for i in range(3)])
  winner,_=votes(groups,np.ones(3));selected=groups.index(winner)
  answers=dict(single=calls['single']['answer'],parallel=calls['parallel'+str(selected)]['answer'],refinement=calls['revision']['answer'])
  costs=accounting(calls);initial_correct=scorer.correct(calls['parallel0']['answer'],gold)
  summary=[]
  for strategy,answer in answers.items():
   names=dict(single=['single'],parallel=['parallel0','parallel1','parallel2'],refinement=['parallel0','revision'])[strategy]
   correct=scorer.correct(answer,gold)
   summary.append(dict(sample_id=qid,strategy=strategy,answer=answer,gold=gold,correct=int(correct),**costs[strategy],initial_correct=int(initial_correct),wrong_to_correct=int(not initial_correct and correct),correct_to_wrong=int(initial_correct and not correct),extraction_success=int(scorer.normalize(answer)!='__NO_ANSWER__'),truncated_calls=sum(calls[n]['hit_max_new_tokens'] for n in names),calls=len(names),peak_allocated_bytes=max(calls[n]['peak_allocated_bytes'] for n in names),peak_reserved_bytes=max(calls[n]['peak_reserved_bytes'] for n in names),selected_call=('parallel'+str(selected) if strategy=='parallel' else ('revision' if strategy=='refinement' else 'single'))))
  store.put('live_summary/'+qid,summary)
 return generator
