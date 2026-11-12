"""Reference likelihood and four-condition, pair-matched final evaluation."""
import time
import json
from pathlib import Path
import torch
from .metrics import normalize,soft_score,summarize
from .storage import atomic_json

CONDITIONS=('correct','complementary','mismatched','no_image')
def controlled(data,suite,condition):
    ids=[i for p in data[suite] for i in p]
    partners={a:b for p in data[suite] for a,b in (p,p[::-1])}
    ex=[data['examples'][i] for i in ids]
    if condition=='no_image': return ids,ex,None
    mapping=partners if condition=='complementary' else data[suite+'_mismatch'] if condition=='mismatched' else {i:i for i in ids}
    return ids,ex,[data['examples'][mapping[i]] for i in ids]

@torch.no_grad()
def evaluate_rows(model,features,ids,examples,visual,generation=False,caption=False,deadline=lambda:False):
    rows=[];model.eval()
    for j in range(0,len(examples),8):
        if deadline(): raise TimeoutError('Safe stop before evaluation batch')
        batch=examples[j:j+8];f=None if visual is None else features.batch(visual[j:j+8])
        with torch.autocast('cuda',dtype=torch.float16):
            nll=model.losses(batch,f,include_eot=caption).cpu().tolist()
            if generation:
                torch.cuda.synchronize();start=time.monotonic()
                predictions=model.generate_answers(batch,f)
                torch.cuda.synchronize();seconds=(time.monotonic()-start)/len(batch)
            else: predictions=['']*len(batch);seconds=0.
        for offset,(key,e,p,n) in enumerate(zip(ids[j:j+8],batch,predictions,nll)):
            exact=bool(normalize(p) and normalize(p)==normalize(e['answer']))
            rows.append(dict(example_id=key,pair_id=e['pair_id'],image_id=e['image_id'],question=e['question'],
                reference=e['answer'],references=e['answers'],kind=e['kind'],prediction=p,nll=n,
                input_image_id=None if visual is None else visual[j+offset]['image_id'],
                exact=exact,score=soft_score(p,e['answers']) if len(e['answers'])==10 else float(exact),seconds=seconds))
    return rows

def development(model,features,data,update,out,deadline=lambda:False):
    path=Path(out)/f'dev-{update:04d}.json'
    if path.exists(): return json.loads(path.read_text())
    start=time.monotonic();result={'update':update,'suites':{}}
    ids=data['caption_dev'];ex=[data['examples'][i] for i in ids]
    result['caption']=evaluate_rows(model,features,ids,ex,ex,caption=True,deadline=deadline)
    result['caption_loss']=sum(r['nll'] for r in result['caption'])/len(ids)
    for suite in ('ground_dev','natural_dev'):
        values={}
        for condition in ('correct','complementary','no_image'):
            ids,ex,visual=controlled(data,suite,condition)
            values[condition]=evaluate_rows(model,features,ids,ex,visual,deadline=deadline)
        means={k:sum(r['nll'] for r in v)/len(v) for k,v in values.items()}
        result['suites'][suite]={'rows':values,'mean_nll':means,'swap_penalty':means['complementary']-means['correct']}
    result['seconds']=time.monotonic()-start
    atomic_json(path,result);model.train();return result

def final(model,features,data,out,deadline=lambda:False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result={}
    for suite in ('ground_test','natural_test'):
        conditions={}
        for condition in CONDITIONS:
            path=out/f'{suite}-{condition}.json'
            if path.exists(): rows=json.loads(path.read_text())
            else:
                ids,ex,visual=controlled(data,suite,condition)
                rows=evaluate_rows(model,features,ids,ex,visual,generation=True,deadline=deadline)
                partners={a:b for p in data[suite] for a,b in (p,p[::-1])}
                for row in rows:
                    row['follows_partner']=normalize(row['prediction'])==normalize(data['examples'][partners[row['example_id']]]['answer'])
                atomic_json(path,rows)
            if len(rows)!=2*len(data[suite]): raise ValueError('Incomplete final condition')
            conditions[condition]=rows
        result[suite]=summarize(conditions)
        result[suite]['partner_follow_rate']={c:sum(r['follows_partner'] for r in rows)/len(rows) for c,rows in conditions.items()}
        chosen=[]
        for kind in sorted({r['kind'] for r in conditions['correct']}):
            pids=sorted({r['pair_id'] for r in conditions['correct'] if r['kind']==kind})[:4]
            chosen.extend([dict(condition=c,**r) for c,rows in conditions.items() for r in rows if r['pair_id'] in pids])
        atomic_json(out/f'{suite}-qualitative.json',chosen)
        qualitative_html(chosen,data,features,out/f'{suite}-qualitative.html')
    atomic_json(out/'metrics.json',result);return result

def plots(logs,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    if logs:
        axes[0,0].plot([r['update'] for r in logs],[r['loss'] for r in logs],alpha=.7)
    axes[0,0].set_title('Training example-weighted loss')
    dev=[json.loads(p.read_text()) for p in sorted(out.glob('dev-*.json'))]
    if dev:
        x=[d['update'] for d in dev]
        axes[0,1].plot(x,[d['caption_loss'] for d in dev]);axes[0,1].set_title('Caption development loss')
        for suite in ('ground_dev','natural_dev'):
            axes[1,0].plot(x,[d['suites'][suite]['mean_nll']['correct'] for d in dev],label=suite)
            axes[1,1].plot(x,[d['suites'][suite]['swap_penalty'] for d in dev],label=suite)
        axes[1,0].set_title('Correct-image reference NLL');axes[1,1].set_title('Swap NLL minus correct NLL')
        axes[1,0].legend();axes[1,1].legend()
    for ax in axes.flat:
        for boundary in (2250,3750,4500): ax.axvline(boundary,color='gray',ls='--',lw=.6)
        ax.set_xlabel('Main optimizer updates');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)


def qualitative_html(rows,data,features,path):
    """Aligned selected images and raw responses; no authored report interpretation."""
    import html,base64
    from .data import image_path
    groups={}
    for row in rows: groups.setdefault(row['pair_id'],[]).append(row)
    pieces=['<!doctype html><meta charset="utf-8"><title>Task 2 selected examples</title>',
            '<style>body{font:16px system-ui;margin:30px;max-width:1100px}img{max-width:360px;max-height:280px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #bbb}</style>',
            '<h1>Predeclared qualitative examples</h1><p>First four pair IDs per question type, sorted before inspecting outputs. References and raw outputs only.</p>']
    for pid,group in groups.items():
        correct=[r for r in group if r.get('condition')=='correct']
        pieces.append('<h2>'+html.escape(pid)+'</h2><p>'+html.escape(group[0]['question'])+'</p><table>')
        pictures=[]
        for r in correct:
            e=data['examples'][r['example_id']]
            image=image_path(e,features.local/'images')
            encoded=base64.b64encode(image.read_bytes()).decode()
            pictures.append('<td><img src="data:image/jpeg;base64,'+encoded+'"><p>Reference: '+html.escape(r['reference'])+'</p></td>')
        pieces.append('<tr>'+''.join(pictures)+'</tr></table><table><tr><th>Condition</th><th>Image ID</th><th>Prediction</th><th>Reference NLL</th></tr>')
        for r in group:
            pieces.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['condition'],r['image_id'],r['prediction'],round(r['nll'],4)])+'</tr>')
        pieces.append('</table>')
    path.write_text(''.join(pieces),encoding='utf-8')
