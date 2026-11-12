"""Run All orchestrator; completion requires verified evidence, never exit code alone."""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
import traceback
from .protocol import VERSION,REVISION,SEED,RELEASE,digest
from .storage import Monitor,Store,SafeStop,sha256,write_json
from .data import download,QuestionReader
from .core import split_ids
from .offline import run_dataset
from .export import fit_policies,export_dataset,export_live
from .live import run_live,CALLS

BASE_OUTPUTS=['case_examples.json','policies.json','orders.json','examples.json','selections.parquet','selections.csv','adaptive.parquet','adaptive.csv','question_means.csv','adaptive_question_means.csv','summary.csv','adaptive_summary.csv','stopping_histogram.csv','paired_intervals.csv','signal_per_question.csv','rollout_features.csv','signal_summary.csv','similar_cost_comparison.csv','retrospective_pool_fraction.csv','accuracy_candidates.png','accuracy_tokens.png','signal_quartiles.png','stopping_distribution.png','failure_decomposition.png']
LIVE_OUTPUTS=['per_question.csv','generations.json','summary.csv','revision_transitions.csv','paired_intervals.csv','examples.json','live_comparison.png']

def output_names(dataset):
 return BASE_OUTPUTS+(['token_associations.csv','token_associations_question_means.csv','token_associations_summary.csv','token_trajectories.png','fusion_ablation.png'] if dataset=='mathvista' else [])

def verified_exports(store,key,out,names):
 cached=store.get(key)
 return bool(cached and set(cached)==set(names) and all((out/name).is_file() and sha256(out/name)==cached[name] for name in names))

def publish_exports(staging,out,names):
 out.mkdir(parents=True,exist_ok=True)
 for name in names:
  src=staging/name;dst=out/name;temp=dst.with_name(dst.name+'.partial')
  expected=sha256(src);shutil.copyfile(src,temp)
  if sha256(temp)!=expected:raise IOError('Export staging checksum mismatch')
  os.replace(temp,dst)
  if sha256(dst)!=expected:raise IOError('Published export checksum mismatch')

def checkpoint_exports(store,key,out,names):
 hashes={name:sha256(out/name) for name in names}
 if any((out/name).stat().st_size==0 for name in names):raise ValueError('Empty required output')
 store.put(key,hashes)

def verify(root,store,splits):
 root=Path(root);offline=0;calls=0;questions=0;policies=0;artifacts={}
 for dataset in ('math','mathvista'):
  ds=splits.get(dataset,{})
  ok=len(ds.get('dev',[]))==200 and len(ds.get('evaluation',[]))==800 and all(store.get(dataset+'/'+q) is not None for q in ds['dev']+ds['evaluation'])
  policy=store.get(dataset+'/frozen_policies')
  if policy:policies+=len(policy['selected'])
  out=root/'exports'/dataset;names=output_names(dataset)
  if ok and verified_exports(store,'exports/'+dataset,out,names):offline+=1
  for name in names:
   p=out/name
   if p.exists():artifacts[str(p)]=sha256(p)
 for q in splits.get('math',{}).get('live',[]):
  questions+=store.get('live_summary/'+q) is not None
  for name,_ in CALLS:calls+=store.get('live/'+q+'/'+name) is not None
 out=root/'exports'/'live'
 live_ok=verified_exports(store,'exports/live',out,LIVE_OUTPUTS)
 for name in LIVE_OUTPUTS:
  p=out/name
  if p.exists():artifacts[str(p)]=sha256(p)
 fully=offline==2 and questions==50 and calls==250 and policies==4 and live_ok and store.get('gpu_smoke') is not None
 return dict(fully_completed=fully,state='fully completed' if fully else 'partially completed',offline_datasets=offline,live_questions=questions,live_generation_calls=calls,adaptive_policies=policies,artifact_count=len(artifacts),artifacts=artifacts,plots=sum(k.endswith('.png') for k in artifacts),tables=sum(k.endswith(('.csv','.parquet')) for k in artifacts),run_identity=store.identity,run_path=str(root),latest_checkpoint=str(store.root/'pointers'),verification_boundary='Full completion requires all 2,000 offline questions and 250 real generation calls plus required exports and GPU smoke.')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',default='/content/drive/MyDrive/ATML_PA3_Task3');parser.add_argument('--session-hours',type=float,default=5);args=parser.parse_args()
 root=Path(args.root);root.mkdir(parents=True,exist_ok=True);monitor=Monitor(root,args.session_hours);store=None;splits={};run_root=None
 write_json(root/'final_verification.json',dict(state='partially completed',fully_completed=False))
 monitor.status('partially completed');monitor.log('START','Task 3 Run All',release=RELEASE,drive_path=str(root),session_hours=args.session_hours,safe_stop_reserve_minutes=15)
 try:
  code=Path(__file__).parent
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
  deps={name:md.version(name) for name in ['numpy','scipy','pandas','pyarrow','math-verify','transformers','torch','accelerate','gdown','tokenizers','huggingface-hub','sympy','latex2sympy2_extended']}
  release=download(root/'release',monitor)
  manifest=dict(protocol=VERSION,seed=SEED,model_revision=REVISION,source_commit=commit,spec_sha256=sha256(code.parent/'ATML-PA3.pdf'),source_files={p.name:sha256(p) for p in code.iterdir() if p.is_file()},dependencies=deps,python=platform.python_version(),release=release)
  identity=digest(manifest);run_root=root/'runs'/identity;store=Store(run_root/'checkpoints',identity)
  write_json(run_root/'manifest.json',manifest);write_json(root/'latest_run.json',dict(path=str(run_root),identity=identity))
  monitor.log('PASS','Frozen run identity',commit=commit,identity=identity,dependencies=deps,source_checksums=manifest['source_files'])
  # Copy verified files to local SSD; durable originals and checksums remain on Drive.
  local=Path('/content/task3_release') if Path('/content').exists() else root/'local_ssd'
  for name,entry in release.items():
   src=root/'release'/name;dst=local/name;dst.parent.mkdir(parents=True,exist_ok=True)
   if not dst.exists() or sha256(dst)!=entry['sha256']:
    temp=dst.with_name(dst.name+'.partial');shutil.copyfile(src,temp)
    if sha256(temp)!=entry['sha256']:raise IOError('Local SSD copy mismatch')
    temp.replace(dst)
  metadata=json.loads((local/'mathvista/testmini.json').read_text())
  readers={}
  for dataset in ('math','mathvista'):
   monitor.check();monitor.log('START','Validate official keys and schemas',dataset=dataset)
   reader=QuestionReader(local/dataset,dataset);readers[dataset]=reader
   splits[dataset]=split_ids(reader.ids)
   write_json(run_root/(dataset+'_splits.json'),splits[dataset]);write_json(run_root/(dataset+'_schemas.json'),reader.schemas)
   monitor.log('PASS','Official index and split',dataset=dataset,rows=len(reader.keys),questions=len(reader.ids),dev=200,evaluation=800,seed=SEED)
   run_dataset(dataset,reader,metadata,splits[dataset],store,monitor,phase='dev')
   frozen=fit_policies(dataset,splits[dataset],store)
   monitor.log('PASS','Dev thresholds frozen before evaluation',dataset=dataset,selected=frozen['selected'])
   policies=[{k:p[k] for k in ('family','tau','gamma')} for p in frozen['selected'].values()]
   run_dataset(dataset,reader,metadata,splits[dataset],store,monitor,phase='evaluation',policies=policies)
   out=run_root/'exports'/dataset;names=output_names(dataset)
   if verified_exports(store,'exports/'+dataset,out,names):monitor.log('SKIP: verified cache','Offline exports',dataset=dataset)
   else:
    staging=run_root/'staging'/dataset
    export_dataset(dataset,splits[dataset],store,staging,monitor,reader,metadata)
    publish_exports(staging,out,names);checkpoint_exports(store,'exports/'+dataset,out,names)
  monitor.log('START','Live 50-question experiment; GPU model loads only when a call is missing')
  run_live(readers['math'],splits['math']['live'],store,monitor,root/'model_cache')
  out=run_root/'exports'/'live'
  if not verified_exports(store,'exports/live',out,LIVE_OUTPUTS):
   staging=run_root/'staging'/'live'
   export_live(splits['math']['live'],store,staging)
   publish_exports(staging,out,LIVE_OUTPUTS);checkpoint_exports(store,'exports/live',out,LIVE_OUTPUTS)
  final=verify(run_root,store,splits)
  if not final['fully_completed']:raise RuntimeError('Required evidence is incomplete despite stage return')
  monitor.log('COMPLETE','TASK 3 FULLY COMPLETED',**{k:v for k,v in final.items() if k!='artifacts'})
 except SafeStop as exc:
  final=verify(run_root,store,splits) if store else dict(fully_completed=False,offline_datasets=0,live_questions=0,live_generation_calls=0,adaptive_policies=0,artifact_count=0)
  final.update(state='safe stopped',reason=str(exc));monitor.log('SAFE STOP','INCOMPLETE — SAVED, RUN ALL AGAIN TO RESUME',reason=str(exc))
 except BaseException as exc:
  final=dict(fully_completed=False,state='failed',error=repr(exc),run_path=str(run_root))
  write_json(root/'final_verification.json',final);monitor.status('failed',error=repr(exc));monitor.log('FAIL','Task 3 failed',error=repr(exc));monitor.close();raise
 finally:
  monitor.close()
 final['cpu_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 final['session_elapsed_s']=time.monotonic()-monitor.start
 write_json(root/'final_verification.json',final)
 if run_root:write_json(run_root/'final_verification.json',final)
 monitor.status(final['state'],fully_completed=final['fully_completed']);print(json.dumps(final,indent=2),flush=True)
if __name__=='__main__':main()
