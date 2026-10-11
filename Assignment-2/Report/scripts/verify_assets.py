"""Validate report inputs without GPU access; errors mean the PDF may be incomplete.

Run from Assignment-2/Report: python scripts/verify_assets.py
"""
from pathlib import Path
import re

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
if not (ROOT / 'neurips_2026.sty').is_file():
    failures.append('Missing genuine neurips_2026.sty')
print('Checked figure and table references and NeurIPS style.')
if failures:
    raise SystemExit('\n'.join(failures))
print('PASS: all referenced report inputs exist.')
