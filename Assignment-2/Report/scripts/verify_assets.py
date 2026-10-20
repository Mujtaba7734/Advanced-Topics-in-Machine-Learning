"""Validate report inputs without GPU access; errors mean the PDF may be incomplete.

Run from Assignment-2/Report: python scripts/verify_assets.py
"""
from pathlib import Path
import re
import csv
import json
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'main.tex'
sources = [MAIN, ROOT / 'appendix.tex']
failures = []
for src in sources:
    content = src.read_text(encoding='utf-8')
    for filename in re.findall(r'\\ReportFigure\{([^}]+)\}', content):
        img = ROOT / 'figures' / filename
        if not img.is_file():
            failures.append(f'Missing {img.relative_to(ROOT)}')
    for name in re.findall(r'\\input\{tables/([^}]+)\}', content):
        tex = ROOT / 'tables' / (name + '.tex')
        if not tex.is_file():
            failures.append(f'Missing {tex.relative_to(ROOT)}')
for tex in (ROOT / 'tables').glob('*.tex'):
    content = tex.read_text(encoding='utf-8')
    if r'\begin{table}[!htbp]' not in content:
        failures.append(f'{tex.name}: not using [!htbp]')
if r'\begin{figure}[!htbp]' not in MAIN.read_text(encoding='utf-8'):
    failures.append('main.tex: ReportFigure must use [!htbp]')
for heatmap in ('25_safety_full_label_distribution.png', '19_safety_category_heatmap.png'):
    if not (ROOT / 'figures' / heatmap).is_file():
        failures.append(f'Missing Task 4 classification map: {heatmap}')
csv_path = ROOT / 'data' / '14_safety_full_category_distribution.csv'
if csv_path.is_file():
    counts = defaultdict(int)
    labels = defaultdict(set)
    for row in csv.DictReader(csv_path.open(encoding='utf-8', newline='')):
        key = (row['policy'], row['category'])
        counts[key] += int(row['count'])
        labels[key].add(row['label'])
    valid_labels = {'SAFE_ANSWER', 'JUSTIFIED_REFUSAL', 'UNSAFE_COMPLIANCE', 'OVER_REFUSAL', 'AMBIGUOUS'}
    if len(counts) != 72 or set(k[0] for k in counts) != {'sft', 'dpo', 'ppo', 'grpo'}:
        failures.append('XSTest distribution lacks 4 policies x 18 categories')
    if any(count != 25 or labels[key] != valid_labels for key, count in counts.items()):
        failures.append('XSTest category counts/labels inconsistent with 25 prompts per category')
else:
    failures.append('Missing XSTest category-level source CSV')
manual_path = ROOT / 'data' / '15_manual_audit.json'
if manual_path.is_file():
    audit = json.loads(manual_path.read_text(encoding='utf-8'))
    confusion = audit.get('confusion_ai_rows_manual_columns', {})
    if audit.get('label_count') != 240 or sum(sum(row.values()) for row in confusion.values()) != 240:
        failures.append('Human/AI judge confusion counts must total 240')
else:
    failures.append('Missing manual-audit source JSON')
if not (ROOT / 'neurips_2026.sty').is_file():
    failures.append('Missing genuine neurips_2026.sty')
print('Checked figure and table references and NeurIPS style.')
if failures:
    raise SystemExit('\n'.join(failures))
print('PASS: all referenced report inputs exist.')
