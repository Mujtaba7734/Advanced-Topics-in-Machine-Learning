"""Small official-data CPU integration. Its outputs are test evidence, not the experiment."""
import argparse
import json
from pathlib import Path
import time
import resource
import pandas as pd
from .data import QuestionReader
from .core import split_ids
from .offline import run_dataset
from .export import fit_policies,export_dataset
from .storage import Store,Monitor,write_json,sha256
from .protocol import digest
from .run import output_names,publish_exports,checkpoint_exports,verified_exports

def main():
 p=argparse.ArgumentParser();p.add_argument('--official-root',required=True);p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 root=Path(args.output);metadata=json.loads(Path(args.metadata).read_text());monitor=Monitor(root/'logs');start=time.monotonic()
 source={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')};store=Store(root/'checkpoints',digest(source));results={}
 try:
  for dataset in ('math','mathvista'):
   reader=QuestionReader(Path(args.official_root)/dataset,dataset);full=split_ids(reader.ids);splits=dict(dev=full['dev'][:2],evaluation=full['evaluation'][:3])
   run_dataset(dataset,reader,metadata,splits,store,monitor,'dev')
   frozen=fit_policies(dataset,splits,store)
   policies=[{k:r[k] for k in ('family','tau','gamma')} for r in frozen['selected'].values()]
   run_dataset(dataset,reader,metadata,splits,store,monitor,'evaluation',policies)
   staging=root/'staging'/dataset;out=root/'exports'/dataset
   export_dataset(dataset,splits,store,staging,monitor,reader,metadata)
   publish_exports(staging,out,output_names(dataset));checkpoint_exports(store,'exports/'+dataset,out,output_names(dataset))
   assert verified_exports(store,'exports/'+dataset,out,output_names(dataset))
   fixed=pd.read_parquet(out/'selections.parquet');adaptive=pd.read_parquet(out/'adaptive.parquet')
   methods=7 if dataset=='math' else 13
   assert len(fixed)==3*10*6*methods;assert len(adaptive)==3*10*2
   assert fixed[fixed.N==1].groupby(['sample_id','order']).selected_rollout_id.nunique().max()==1
   before={q:digest(store.get(dataset+'/'+q)) for q in splits['evaluation']}
   run_dataset(dataset,reader,metadata,splits,store,monitor,'evaluation',policies)
   assert before=={q:digest(store.get(dataset+'/'+q)) for q in splits['evaluation']}
   results[dataset]=dict(dev_questions=2,evaluation_questions=3,rollouts_checked=160,fixed_selection_records=len(fixed),adaptive_records=len(adaptive),exports=len(output_names(dataset)),question_resume_verified=True,ids=splits)
  write_json(root/'verification.json',dict(state='CPU tested',official_data=True,full_experiment=False,cuda_tested=False,results=results,elapsed_s=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=source))
  monitor.log('PASS','Official small-sample integration and resume verification complete')
 finally:monitor.close()
if __name__=='__main__':main()
