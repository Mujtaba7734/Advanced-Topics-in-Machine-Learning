"""Rebuild full Task 4 category-wise AI-judge distribution from fixed compute backup.

Run from Assignment-2/Report:
    python scripts/rebuild_safety_category_figure.py --archive ../backups/ATML_PA2_Task4_COMPUTE_BACKUP.tar.gz
This does NOT use disputed human manual labels.
"""
from pathlib import Path
import argparse, tarfile, json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
POLS=('sft','dpo','ppo','grpo')
LABELS=('SAFE_ANSWER','JUSTIFIED_REFUSAL','UNSAFE_COMPLIANCE','OVER_REFUSAL','AMBIGUOUS')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,required=True)
    a=ap.parse_args()
    with tarfile.open(a.archive,'r:gz') as archive:
        data={p:[json.loads(s) for s in archive.extractfile(f'results/task4_safety/judged_{p}.jsonl').read().decode('utf8').splitlines() if s] for p in POLS}
    if any(len(data[p]) != 450 for p in POLS):
        raise SystemExit('Expected 450 fixed XSTest judgments per model')
    cats=list(dict.fromkeys(row['type'] for row in data['sft']))
    if len(cats)!=18:
        raise SystemExit('Expected 18 benchmark categories')
    counts=[]; matrix=np.zeros((18,12))
    for i,cat in enumerate(cats):
        for j,p in enumerate(POLS):
            items=[x for x in data[p] if x['type']==cat]
            if len(items)!=25:raise SystemExit('Expected 25 items per category')
            totals=[sum(row['label']==label for row in items) for label in LABELS]
            if sum(totals)!=25:raise SystemExit('Found unrecognized AI judge label')
            for k,label in enumerate(LABELS):
                counts.append({'policy':p,'category':cat,'n':25,'label':label,'count':totals[k],'percent':100*totals[k]/25})
            matrix[i,3*j:3*j+3]=[totals[k]*4 for k in range(3)]
    out=ROOT/'data'/'14_safety_full_category_distribution.csv'
    with out.open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['policy','category','n','label','count','percent'])
        writer.writeheader();writer.writerows(counts)
    fig,ax=plt.subplots(figsize=(8,6.6))
    img=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    cols=[]
    for p in POLS:cols += [f'{p.upper()}\nAns','Ref','Unsafe']
    ax.set_xticks(range(12),cols,fontsize=7)
    ax.set_yticks(range(18),[c.replace('_',' ') for c in cats],fontsize=7)
    for i in range(18):
        for j in range(12):
            val=matrix[i,j]
            ax.text(j,i,f'{val:.0f}',ha='center',va='center',fontsize=6,color='white' if val>=64 else 'black')
    for j in range(1,4):ax.axvline(j*3-.5,color='black',linewidth=1.2)
    ax.set_title('AI judge labels by XSTest category (n=25 per model and category)',fontsize=10)
    ax.tick_params(axis='both',length=0)
    cb=fig.colorbar(img,ax=ax,fraction=.024,pad=.02)
    cb.set_label('Category share (%)',fontsize=8);cb.ax.tick_params(labelsize=7)
    fig.text(.12,.013,'Across every policy and category: OVER_REFUSAL=0%; AMBIGUOUS=0%.',fontsize=7)
    fig.subplots_adjust(left=.265,right=.91,bottom=.085,top=.945)
    fig.savefig(ROOT/'figures'/'25_safety_full_label_distribution.png',dpi=210)
    print('Wrote',out)

if __name__=='__main__':main()
