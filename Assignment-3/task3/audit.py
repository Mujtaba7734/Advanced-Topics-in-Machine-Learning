"""Read-only official-data validation, executable independently of the GPU experiment."""
import argparse
import json
from pathlib import Path
import resource
import time
from .data import QuestionReader
from .storage import write_json,sha256
from .core import split_ids
EXPECTED={'math':(20742094,523,521),'mathvista':(8961703,487,3032)}

def audit(root):
 results={}
 for dataset in EXPECTED:
  start=time.monotonic();reader=QuestionReader(Path(root)/dataset,dataset);tokens=capped=null=0
  for n,qid in enumerate(reader.ids):
   rows=reader.question(qid)
   tokens+=sum(r['n_generated_text_tokens'] for r in rows)
   capped+=sum(r['hit_max_new_tokens'] for r in rows)
   null+=sum(r['extracted_answer'] is None for r in rows)
   if n%100==0:print(f'[PASS] {dataset}: validated {n+1}/1000 questions',flush=True)
  if (tokens,capped,null)!=EXPECTED[dataset]:raise ValueError(f'Release aggregate mismatch: {dataset}: {(tokens,capped,null)}')
  results[dataset]=dict(rows=len(reader.keys),questions=len(reader.ids),retained_tokens=tokens,capped_rollouts=capped,null_answers=null,elapsed_s=time.monotonic()-start,schemas=reader.schemas,splits=split_ids(reader.ids),sha256={n:sha256(Path(root)/dataset/(n+'.parquet')) for n in reader.files})
 return dict(state='CPU tested',datasets=results,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cuda_tested=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--official-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 write_json(a.output,audit(a.official_root));print('[PASS] Full official key/token alignment audit',flush=True)
