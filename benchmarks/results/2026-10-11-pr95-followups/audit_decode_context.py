import os
"""Audit matched cold-context decode brackets; retain every request."""
import hashlib
import json
import pathlib
import statistics

out = pathlib.Path(os.environ.get('PR95_OUT',str(pathlib.Path(__file__).parent)))
legs = []
for suffix in ('fp8a', 'nvfp4', 'fp8b'):
    d = out / f'decode-wide-final-w4-{suffix}'
    report = json.loads((d / 'decode-context.json').read_text())
    assert report['seed'] == 7 and report['max_tokens'] == 512
    assert not report['thinking'] and len(report['phases']) == 9
    cells = {}
    for p in report['phases']:
        key = (p['requested_context'], p['repeat'])
        assert key not in cells and p['concurrency'] == 1
        e = p['engine']
        assert e['prompt_tokens'] == e['prompt_tokens_computed']
        assert e['prompts_prefilled'] == 1 and e['tokens_per_s'] > 0
        r, = p['requests']
        assert r['status'] == 200 and 1 < r['tokens'] <= 512
        assert e['tokens_generated'] == r['tokens']
        cells[key] = p
    legs.append((report, cells))
assert len({x[0]['data_sha256'] for x in legs}) == 1
rows = []
for length in (1024, 8192, 32768):
    samples = [[leg[1][length, r] for r in range(3)] for leg in legs]
    for r in range(3):
        assert len({(s[r]['requests'][0]['prompt_sha256'], s[r]['engine']['prompt_tokens']) for s in samples}) == 1
    medians = [statistics.median(s['engine']['tokens_per_s'] for s in ss) for ss in samples]
    rows.append(dict(requested_context=length,
                     prompt_tokens=[s['engine']['prompt_tokens'] for s in samples[0]],
                     completion_tokens=[[s['requests'][0]['tokens'] for s in ss] for ss in samples],
                     fp8a_tps=medians[0], nvfp4_tps=medians[1], fp8b_tps=medians[2],
                     fp8_drift_pct=100 * (medians[2] / medians[0] - 1),
                     per_prompt_gain_pct=[100 * (samples[1][r]['engine']['tokens_per_s'] / max(samples[0][r]['engine']['tokens_per_s'], samples[2][r]['engine']['tokens_per_s']) - 1) for r in range(3)],
                     gain_pct=100 * (medians[1] / max(medians[0], medians[2]) - 1),
                     nvfp4_faster=medians[1] > max(medians[0], medians[2])))
result = dict(rows=rows, performance_gate=all(r['nvfp4_faster'] for r in rows),
              note='Native greedy texts may differ between quants; compare useful decode tokens per engine step time, excluding cold prefill.')
(out / 'decode-context-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['performance_gate'] else 1)
