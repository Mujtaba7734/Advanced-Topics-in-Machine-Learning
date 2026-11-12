"""Official COCO/VQAv2 selective acquisition and deterministic manifests."""
from collections import defaultdict, Counter
from pathlib import Path
import json
import os
import time
import zipfile
import numpy as np
import requests
from .protocol import P, digest
from .storage import sha256, atomic_json, valid, publish
from .metrics import consensus, normalize

VQA='https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/'
COCO='https://s3.amazonaws.com/images.cocodataset.org/'

def download(url,path):
    path=Path(path); record=path.with_suffix(path.suffix+'.json')
    if record.exists() and valid(path,json.loads(record.read_text())): return path
    path.parent.mkdir(parents=True,exist_ok=True)
    error=None
    for attempt in range(4):
        try:
            tmp=path.with_suffix(path.suffix+'.partial')
            with requests.get(url,stream=True,timeout=(20,90)) as r:
                r.raise_for_status()
                with open(tmp,'wb') as f:
                    for chunk in r.iter_content(2**20): f.write(chunk)
            if tmp.stat().st_size==0: raise IOError('Empty download')
            if path.suffix=='.zip':
                with zipfile.ZipFile(tmp) as z:
                    if z.testzip(): raise IOError('ZIP CRC mismatch')
            elif path.suffix=='.jpg':
                from PIL import Image
                with Image.open(tmp) as im: im.verify()
            os.replace(tmp,path)
            atomic_json(record,{'url':url,'bytes':path.stat().st_size,'sha256':sha256(path)})
            return path
        except (requests.RequestException,OSError,zipfile.BadZipFile) as exc:
            error=exc; time.sleep(min(2**attempt,8))
    raise RuntimeError(f'Official download failed after retries: {url}: {error}')

def archive_json(url, filename, root):
    archive=download(url,Path(root)/url.rsplit('/',1)[-1])
    with zipfile.ZipFile(archive) as z:
        matches=[n for n in z.namelist() if n.rsplit('/',1)[-1]==filename]
        if len(matches)!=1: raise ValueError(f'Missing/ambiguous official member {filename}')
        return json.loads(z.read(matches[0]))

def example(image,split,question,answer,kind,source,refs=None,pair=None):
    return dict(image_id=int(image),split=split,question=question,answer=answer,kind=kind,
                source=source,answers=refs or [answer],pair_id=pair)

def natural_candidates(questions,annotations,pairs,split):
    qs={q['question_id']:q for q in questions['questions']}
    anns={a['question_id']:a for a in annotations['annotations']}
    result=[]; seen=set()
    for source_index,ids in enumerate(pairs):
        if len(ids)!=2: raise ValueError('Unexpected complementary pair schema')
        key=tuple(sorted(ids))
        if key in seen: continue
        seen.add(key)
        aa=[anns[i] for i in ids]; qq=[qs[i] for i in ids]
        if qq[0]['question']!=qq[1]['question'] or aa[0]['image_id']==aa[1]['image_id']: continue
        if aa[0]['answer_type']!=aa[1]['answer_type']: continue
        refs=[[a['answer'] for a in ann['answers']] for ann in aa]
        targets=[consensus(a) for a in refs]
        if min(t[1] for t in targets)<8 or normalize(targets[0][0])==normalize(targets[1][0]): continue
        pid=f'vqa-{split}-{key[0]}-{key[1]}'
        result.append([example(ann['image_id'],split,q['question'],target[0],ann['answer_type'],
                               {'question_id':q['question_id'],'pair_source_index':source_index},r,pid)
                       for ann,q,target,r in zip(aa,qq,targets,refs)])
    return result

def select_natural(candidates,counts,rng,used=None):
    selected=[]; used=used if used is not None else set()
    for kind,n in zip(['yes/no','number','other'],counts):
        pool=[p for p in candidates if p[0]['kind']==kind]
        got=0
        for index in rng.permutation(len(pool)):
            pair=pool[index]; images={e['image_id'] for e in pair}
            if images & used: continue
            selected.append(pair); used.update(images); got+=1
            if got==n: break
        if got!=n: raise ValueError(f'Insufficient disjoint {kind} VQA pairs: {got}/{n}')
    return selected

def grounding(instances,allowed,split,rng,per_kind,used=None):
    counts=defaultdict(Counter)
    for a in instances['annotations']:
        # Count annotated instances, including crowd annotations; explicit label convention.
        if a['image_id'] in allowed: counts[a['category_id']][a['image_id']]+=1
    categories={c['id']:c['name'] for c in instances['categories']}
    used=used if used is not None else set()
    out={'presence':[],'counting':[]}
    ids=sorted(allowed)
    positive={cat:[i for i in ids if counts[cat].get(i,0)>0] for cat in categories}
    negative={cat:[i for i in ids if counts[cat].get(i,0)==0] for cat in categories}
    count_buckets={cat:{n:[i for i in ids if counts[cat].get(i,0)==n] for n in range(1,5)} for cat in categories}
    for kind in out:
        attempts=0
        while len(out[kind])<per_kind:
            attempts+=1
            if attempts>per_kind*200: raise ValueError('Insufficient annotation-derived disjoint pairs')
            cat=int(rng.choice(sorted(categories))); c=counts[cat]
            if kind=='presence':
                pos=[i for i in positive[cat] if i not in used]; neg=[i for i in negative[cat] if i not in used]
                if not pos or not neg: continue
                images=[int(rng.choice(pos)),int(rng.choice(neg))]; answers=['yes','no']
                question=f'Does this image contain {categories[cat]}?'
            else:
                buckets={n:[i for i in count_buckets[cat][n] if i not in used] for n in range(1,5)}
                keys=[n for n,v in buckets.items() if v]
                if len(keys)<2: continue
                ns=rng.choice(keys,2,replace=False)
                images=[int(rng.choice(buckets[n])) for n in ns]; answers=list(map(str,ns))
                name=categories[cat]
                plural={'person':'people','sheep':'sheep','mouse':'mice','knife':'knives','scissors':'pairs of scissors','skis':'pairs of skis'}.get(name, name+'es' if name.endswith(('s','ch','sh','x')) else name[:-1]+'ies' if name.endswith('y') and name[-2] not in 'aeiou' else name+'s')
                question=f'How many {plural} can you see?'
            pid=f'ground-{split}-{kind}-{cat}-{images[0]}-{images[1]}'
            out[kind].append([example(i,split,question,a,kind,{'category_id':cat,'annotated_count':c[i]},pair=pid)
                              for i,a in zip(images,answers)])
            # Evaluation: no image reused anywhere. Training pools allow reuse.
            if split=='val2014': used.update(images)
    return out

def prepare(root, annotation_root=None):
    root=Path(root); manifest=root/'selected_data.json'
    if manifest.exists():
        data=json.loads(manifest.read_text()); validate(data); return data
    rng=np.random.default_rng(P.seed); raw=Path(annotation_root) if annotation_root else root/'annotations'
    caption={}; instance={}; natural={}
    for split,label in [('train2014','Train'),('val2014','Val')]:
        caption[split]=archive_json(COCO+'annotations/annotations_trainval2014.zip',f'captions_{split}.json',raw)
        instance[split]=archive_json(COCO+'annotations/annotations_trainval2014.zip',f'instances_{split}.json',raw)
        q=archive_json(VQA+f'v2_Questions_{label}_mscoco.zip',f'v2_OpenEnded_mscoco_{split}_questions.json',raw)
        a=archive_json(VQA+f'v2_Annotations_{label}_mscoco.zip',f'v2_mscoco_{split}_annotations.json',raw)
        p=archive_json(VQA+f'v2_Complementary_Pairs_{label}_mscoco.zip',f'v2_mscoco_{split}_complementary_pairs.json',raw)
        natural[split]=natural_candidates(q,a,p,split)
    data={'examples':{},'seed':P.seed,'normalization':'VQA rules, applied unconditionally',
          'counting_convention':'All COCO instance records including crowd count as one annotated instance',
          'qualitative_rule':'First 4 pair IDs lexicographically per suite/type, fixed before generation'}
    used=set()
    data['natural_dev']=select_natural(natural['val2014'],[12,8,12],rng,used)
    data['natural_test']=select_natural(natural['val2014'],[36,24,36],rng,used)
    data['natural_train']=select_natural(natural['train2014'],[640,256,640],rng)
    ids={s:{im['id'] for im in caption[s]['images']} for s in caption}
    for label,n in [('dev',12),('test',32)]:
        g=grounding(instance['val2014'],ids['val2014'],'val2014',rng,n,used)
        data['ground_'+label]=g['presence']+g['counting']
    g=grounding(instance['train2014'],ids['train2014'],'train2014',rng,3072)
    data.update(g)
    for label,split,n,k in [('train','train2014',6000,3),('dev','val2014',96,1)]:
        pool=defaultdict(list)
        for a in caption[split]['annotations']: pool[a['image_id']].append(a)
        available=sorted(i for i in ids[split] if len(pool[i])>=k and (split=='train2014' or i not in used))
        chosen=rng.choice(available,n,replace=False)
        data['caption_'+label]=[example(int(i),split,'Describe this image.',a['caption'],'caption',{'caption_id':a['id']})
                               for i in chosen for a in sorted(pool[int(i)],key=lambda a:a['id'])[:k]]
    # Compact unique examples and ID-only pools.
    for name in ['natural_train','natural_dev','natural_test','ground_dev','ground_test','presence','counting','caption_train','caption_dev']:
        def register(e):
            key=digest(e)[:24]; data['examples'][key]=e; return key
        data[name]=[register(v) if isinstance(v,dict) else [register(e) for e in v] for v in data[name]]
    for name in ['ground_test','natural_test']:
        pairs=data[name]; mapping={}
        for j,pair in enumerate(pairs):
            forbidden={data['examples'][x]['image_id'] for x in pair}
            other=next(p for p in pairs[j+1:]+pairs[:j] if not forbidden & {data['examples'][x]['image_id'] for x in p})
            for a,b in zip(pair,other): mapping[a]=b
        data[name+'_mismatch']=mapping
    data['source_files']={p.name:json.loads(p.read_text()) for p in raw.glob('*.zip.json')}
    data['hash']=digest({k:v for k,v in data.items() if k!='hash'})
    validate(data); atomic_json(manifest,data); return data

def validate(data):
    if data['hash']!=digest({k:v for k,v in data.items() if k!='hash'}): raise ValueError('Selected-data hash mismatch')
    ex=data['examples']
    expected={'caption_train':18000,'caption_dev':96,'natural_train':1536,'natural_dev':32,'natural_test':96,'ground_dev':24,'ground_test':64}
    eval_images=set()
    for name,n in expected.items():
        if len(data[name])!=n: raise ValueError(f'Wrong count: {name}')
        ids=data[name] if name.startswith('caption') else [i for p in data[name] for i in p]
        images={ex[i]['image_id'] for i in ids}
        split='train2014' if name.endswith('train') else 'val2014'
        if any(ex[i]['split']!=split for i in ids): raise ValueError('Split leakage')
        if split=='val2014':
            if images & eval_images: raise ValueError('Evaluation image overlap')
            eval_images.update(images)
        if name.startswith('natural'):
            counts=Counter(ex[p[0]]['kind'] for p in data[name])
            target={'train':[640,256,640],'dev':[12,8,12],'test':[36,24,36]}[name.split('_')[1]]
            if [counts[k] for k in ['yes/no','number','other']]!=target: raise ValueError('VQA quotas')
            for pair in data[name]:
                a,b=[ex[i] for i in pair]
                if a['question']!=b['question'] or normalize(a['answer'])==normalize(b['answer']): raise ValueError('Invalid complementary pair')
                if any(consensus(e['answers'])[1]<8 for e in (a,b)): raise ValueError('Insufficient agreement')
    train_images={e['image_id'] for e in ex.values() if e['split']=='train2014'}
    if eval_images & train_images: raise ValueError('Training/evaluation image overlap')
    for name in ('presence','counting'):
        if any(ex[i]['split']!='train2014' for p in data[name] for i in p): raise ValueError('Replay leakage')

def image_path(example, root):
    split,i=example['split'],example['image_id']
    name=f'COCO_{split}_{i:012d}.jpg'
    return download(COCO+f'{split}/{name}',Path(root)/split/name)
