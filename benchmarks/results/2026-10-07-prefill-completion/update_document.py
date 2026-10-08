"""Apply the diagnostic fixes and their remaining verified benchmark reruns."""
import argparse
import importlib.util
from pathlib import Path
import statistics
import sys

from results import HERE, read, validated

ROOT = HERE.parents[2]
FOLLOWUP = HERE.parent / '2026-10-07-prefill-followup'
sys.path.insert(0, str(FOLLOWUP))
spec = importlib.util.spec_from_file_location('followup_document', FOLLOWUP / 'update_document.py')
followup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(followup)
spec = importlib.util.spec_from_file_location('original_summary',
                                            HERE.parent / '2026-10-06-prefill-scaling/summarize.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def apply_updates(text):
    text = followup.apply_updates(text)
    entries = read(HERE / 'matrix.json')
    keys = {(e['label'], str(e['world']), e['options']): e for e in entries}
    reports = {e['id']: {job: validated(e['id'], 'default', job)
                        for job in next(p['jobs'] for p in read(HERE / 'plan.json')
                                        if p['deployment'] == e['id'] and p['mode'] == 'default')}
               for e in entries}
    modes = {mode: validated('glm53-fp4kv-256k-w4', mode, 'greedy')
             for mode in ['plain', 'mtp2', 'mtp3']}
    for section in ['serving', 'classes', 'modes', 'config', 'context', 'context-prefill']:
        start, end = f'<!-- BEGIN {section} -->', f'<!-- END {section} -->'
        before, rest = text.split(start, 1)
        body, after = rest.split(end, 1)
        lines = []
        for line in body.splitlines():
            cells = [c.strip() for c in line.split('|')[1:-1]] if line.startswith('|') else []
            entry = keys.get(tuple(cells[:3]))
            if entry:
                ident = entry['id']
                data = reports[ident]
                greedy = data.get('greedy')
                rates = summary.class_rates(greedy, 1)
                if section == 'serving':
                    if greedy:
                        cells[3:8] = [summary.span(rates)] + [summary.span(summary.class_rates(greedy, c, False))
                                                             for c in [1, 2, 4, 8]]
                    if data.get('prefill'):
                        samples = data['prefill']['samples']
                        if ident == 'glm-flash-hybrid-256k-w2':
                            samples += read(FOLLOWUP / 'raw' / ident / 'prefill/idle2048/prefill.json')['samples']
                        values = [f"{statistics.median(s['prefill_ms'] for s in samples if s['requested_tokens'] == n) / 1000:.3f}"
                                  for n in [2048, 8192, 32768]]
                        cells[8] = ' / '.join(values)
                elif section == 'classes' and rates:
                    cells[3:] = [f'{v:.1f}' for v in rates]
                elif section == 'modes' and greedy:
                    cells[5] = summary.span(rates)  # MTP1 default in all seven rows
                    if ident == 'glm53-fp4kv-256k-w4':
                        for mode, index in [('plain', 4), ('mtp2', 6), ('mtp3', 7)]:
                            if modes[mode]:
                                cells[index] = summary.span(summary.class_rates(modes[mode], 1))
                elif section == 'config' and greedy:
                    cells[9] = f'[JSON](../benchmarks/results/{HERE.name}/{entry["config"]})'
                elif section in ['context', 'context-prefill']:
                    for i, length in enumerate(summary.LENGTHS):
                        report = data.get(f'context-{length}')
                        if not report:
                            continue
                        samples = report['samples']
                        cold = f"{next(s['engine']['prefill_ms'] for s in samples if s['repeat'] == 0) / 1000:.3f}"
                        if section == 'context-prefill':
                            cells[3 + i] = cold
                        else:
                            if length == 32768:
                                cells[4] = cold
                            cells[5 + 2*i:7 + 2*i] = [f"{statistics.median(s['ms_per_pass'] for s in samples):.2f}",
                                                      f"{statistics.median(s['engine_tokens_per_s'] for s in samples):.1f}"]
                line = '| ' + ' | '.join(cells) + ' |'
            lines.append(line)
        text = before + start + '\n'.join(lines) + '\n' + end + after
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'docs/benchmarks.md'
    old = path.read_text()
    new = apply_updates(old)
    if args.check:
        if old != new:
            raise SystemExit('Benchmark document needs completion updates')
        print('Benchmark document matches verified completion measurements')
    else:
        path.write_text(new)
