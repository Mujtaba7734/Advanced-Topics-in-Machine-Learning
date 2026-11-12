"""Local VQAv2 scoring, using the official VQA normalization tables/rules.
Contraction table adapted from GT-Vision-Lab/VQA PythonEvaluationTools/vqaEvaluation/vqaEval.py.
"""
import json
import re
from pathlib import Path
import numpy as np
CONTRACTIONS=json.loads(Path(__file__).with_name('contractions.json').read_text())
NUMBERS=dict(zip(['none','zero','one','two','three','four','five','six','seven','eight','nine','ten'],
                 ['0','0','1','2','3','4','5','6','7','8','9','10']))
PUNCT=[';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
def normalize(text):
    text=str(text).replace('\n',' ').replace('\t',' ').strip()
    out=text
    for p in PUNCT:
        out=out.replace(p,'' if p+' ' in text or ' '+p in text or re.search(r'\d,\d',text) else ' ')
    out=re.sub(r'(?!<=\d)(\.)(?!\d)','',out)
    words=[NUMBERS.get(w,w) for w in out.lower().split() if w not in ('a','an','the')]
    return ' '.join(CONTRACTIONS.get(w,w) for w in words)

def consensus(answers):
    from collections import Counter
    if len(answers)!=10: raise ValueError('Expected all ten VQA annotations')
    norm=[normalize(a) for a in answers]
    value,n=Counter(norm).most_common(1)[0]
    return next(a for a in answers if normalize(a)==value), n

def soft_score(prediction, answers):
    if len(answers)!=10: raise ValueError('Expected ten annotations')
    p=normalize(prediction)
    if not p: return 0.
    hits=[int(p==normalize(a)) for a in answers]
    return sum(min(1.,(sum(hits)-h)/3) for h in hits)/10

def pair_vectors(rows):
    groups={}
    for row in rows: groups.setdefault(row['pair_id'],[]).append(row)
    result={}
    for pid, pair in sorted(groups.items()):
        if len(pair)!=2: raise ValueError(f'Incomplete/duplicate pair {pid}')
        a,b=pair
        result[pid]=[np.mean([r['score'] for r in pair]),
                     float(all(r['exact'] for r in pair)),
                     float(normalize(a['prediction'])==normalize(b['prediction'])),
                     np.mean([not normalize(r['prediction']) for r in pair]),
                     np.mean([r['nll'] for r in pair]),
                     np.mean([r['seconds'] for r in pair])]
    return result

NAMES=['accuracy','pair_accuracy','identical_answer_rate','empty_answer_rate','reference_nll','seconds_per_answer']
def bootstrap(vectors, resamples=4000):
    a=np.asarray(vectors,dtype=float)
    if a.ndim!=2 or not len(a): raise ValueError('No complete pairs')
    rng=np.random.default_rng(6304)
    samples=a[rng.integers(len(a),size=(resamples,len(a)))].mean(axis=1)
    lo,hi=np.quantile(samples,[.025,.975],axis=0)
    return {name:{'mean':float(a[:,i].mean()),'ci95':[float(lo[i]),float(hi[i])]} for i,name in enumerate(NAMES)}

def summarize(conditions):
    out={}; mapped={}
    for condition,rows in conditions.items():
        mapped[condition]=pair_vectors(rows)
        out[condition]=bootstrap(list(mapped[condition].values()))
        out[condition]['by_type']={t:bootstrap(list(pair_vectors([r for r in rows if r['kind']==t]).values()))
                                   for t in sorted({r['kind'] for r in rows})}
    correct,swap=mapped['correct'],mapped['complementary']
    if correct.keys()!=swap.keys(): raise ValueError('Pair IDs differ across conditions')
    delta=np.array([np.array(correct[k])-np.array(swap[k]) for k in correct])
    delta[:,4]*=-1 # NLL swap-correct: positive means correct image helps.
    out['correct_image_advantages']=bootstrap(delta)
    return out
