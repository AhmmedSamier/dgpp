#!/usr/bin/env python3
"""Check benchmark tables, local links and preservation of historical quality rows."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

root = Path(__file__).resolve().parents[4]
path = root / 'docs/benchmarks.md'
source = path.read_text()
errors, tables, links = [], [], set()
current = None
for lineno, line in enumerate(source.splitlines(), 1):
    if line.startswith('|'):
        cells = re.split(r'(?<!\\)\|', line.strip())[1:-1]
        if current is None:
            current = {'line': lineno, 'columns': len(cells), 'data_rows': -2}
            tables.append(current)
        elif len(cells) != current['columns']:
            errors.append(f'line {lineno}: {len(cells)} columns, expected {current["columns"]}')
        current['data_rows'] += 1
    else:
        current = None
for target in re.findall(r'\]\(([^)]+)\)', source):
    target = target.strip('<>').split('#', 1)[0]
    if not target or '://' in target:
        continue
    links.add(target)
    if not (path.parent / unquote(target)).exists():
        errors.append(f'missing link: {target}')

def quality(text):
    section = text.split('## Quality and correctness', 1)[1].split('## Long context', 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith('|'):
            continue
        cells = [x.strip() for x in line.split('|')[1:-1]]
        if len(cells) >= 6 and re.match(r'^\d+/\d+', cells[3]):
            rows.append(cells[:2] + cells[3:])
    return rows

baseline = subprocess.check_output(['git', 'show', 'HEAD:docs/benchmarks.md'], cwd=root, text=True)
before, after = quality(baseline), quality(source)
if before != after:
    errors.append('historical quality data changed')
report = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'overview_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'tables': tables, 'local_links_checked': len(links),
          'historical_quality_numeric_rows': len(after),
          'quality_matches_HEAD': before == after, 'errors': errors}
output = root / 'benchmarks/results/2026-10-06-prefill-scaling/diagnostics/document-audit.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print(f'{len(tables)} tables, {len(links)} unique local links, {len(after)} quality data rows; errors={errors}')
raise SystemExit(bool(errors))
