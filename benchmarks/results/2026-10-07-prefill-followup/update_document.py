"""Apply verified follow-up prefill results after rendering the original matrix."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from evidence import load_record

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXED = {
    'glm-flash-hybrid-w2': 'idle2048',
    'glm-flash-hybrid-256k-w2': 'idle2048',
    'glm53-fp4kv-256k-w4': 'native-conversion-clean',
    'mimo-w2': 'fp8-dispatch-fix',
    'mimo-w2-fp8kv': 'fp8-dispatch-fix',
}
AFFECTED = set(FIXED) | {'mimo-w4', 'mimo-1m-w4'}
FP4 = 'glm53-fp4kv-256k-w4'


def baseline(value):
    return value if value.endswith('†') or not value[0].isdigit() else value + '†'


def apply_updates(text):
    matrix = json.loads((HERE.parent / '2026-10-06-prefill-scaling/matrix.json').read_text())
    ids = {(e['label'], str(e['world']), e['options']): e['id'] for e in matrix}
    prefills = {}
    for ident, variant in FIXED.items():
        directory = HERE / 'raw' / ident / 'prefill' / variant
        record = load_record(directory)
        for name in ['complete.json', 'memory-check.json', 'rank-identity.json']:
            if not record['records'][name]['passed']:
                raise ValueError(f'{directory}: failed {name}')
        if record['records']['plan.json']['profiled']:
            raise ValueError(f'{directory}: profiled run cannot supply a table timing')
        payload = (directory / 'prefill.json').read_bytes()
        if hashlib.sha256(payload).hexdigest() != record['evidence_sha256']['prefill.json']:
            raise ValueError(f'{directory}: measurement hash mismatch')
        samples = json.loads(payload)['samples']
        values = []
        for length in [2048, 8192, 32768]:
            group = [s for s in samples if s['requested_tokens'] == length]
            if group:
                if len(group) != 2 or {s['repeat'] for s in group} != {0, 1} or any(s['cached_tokens'] for s in group):
                    raise ValueError(f'{directory}: expected two uncached probes at {length}')
                values.append(f"{median(s['prefill_ms'] for s in group) / 1000:.3f}")
            else:
                values.append('Rerun')
        target = '../benchmarks/results/' + str((directory / 'prefill.json').relative_to(HERE.parent))
        prefills[ident] = f"[{' / '.join(values)}]({target})"

    for section in ['serving', 'config', 'classes', 'modes', 'context', 'context-prefill']:
        start, end = f'<!-- BEGIN {section} -->', f'<!-- END {section} -->'
        before, rest = text.split(start, 1)
        body, after = rest.split(end, 1)
        lines = []
        for line in body.splitlines():
            cells = [c.strip() for c in line.split('|')[1:-1]] if line.startswith('|') else []
            ident = ids.get(tuple(cells[:3]))
            if ident in AFFECTED:
                if section == 'serving':
                    cells[4:8] = [baseline(c) for c in cells[4:8]]
                    cells[8] = prefills.get(ident, 'Rerun')
                    if ident == FP4:
                        cells[3] = baseline(cells[3])
                elif section == 'config' and ident.startswith('glm-flash'):
                    cells[7] = '256 / 2,048'
                    target = f'../benchmarks/results/{HERE.name}/raw/{ident}/prefill/idle2048/record.json'
                    cells[9] = f'[Run record]({target})'
                elif section == 'context':
                    cells[4] = 'Rerun'
                    if ident == FP4:
                        cells[5:] = [baseline(c) for c in cells[5:]]
                elif section == 'context-prefill':
                    cells[3:] = ['Rerun' if c[0].isdigit() else c for c in cells[3:]]
                elif ident == FP4 and section in ['classes', 'modes']:
                    offset = 3 if section == 'classes' else 4
                    cells[offset:] = [baseline(c) for c in cells[offset:]]
                line = '| ' + ' | '.join(cells) + ' |'
            lines.append(line)
        text = before + start + '\n'.join(lines) + '\n' + end + after
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail if the document needs updating')
    args = parser.parse_args()
    path = ROOT / 'docs/benchmarks.md'
    previous = path.read_text()
    updated = apply_updates(previous)
    if args.check:
        if updated != previous:
            raise SystemExit('Benchmark document is missing follow-up updates')
        print('Five corrected prefill rows and affected baseline/rerun labels verified')
    else:
        path.write_text(updated)
