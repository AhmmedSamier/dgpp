import hashlib
import json
import pathlib

out = pathlib.Path(__file__).parent
rows = []
fields = ('reasoning', 'content', 'prompt_tokens', 'completion_tokens', 'finish')
for world in (1, 2, 4):
    for suffix in ('fp8a', 'nvfp4', 'fp8b'):
        base_suffix = 'nvfp4' if suffix == 'nvfp4' else 'fp8a'
        baseline = f'final-w{world}-{base_suffix}/transcripts.json'
        candidate = f'decode-wide-final-w{world}-{suffix}/transcripts.json'
        a = json.loads((out / baseline).read_text())
        b = json.loads((out / candidate).read_text())
        assert a['model'] == b['model'] and set(a['results']) == set(b['results'])
        for kind, result in b['results'].items():
            aa = {k: a['results'][kind][k] for k in fields}
            bb = {k: result[k] for k in fields}
            rows.append(dict(world=world, leg=suffix, kind=kind, match=aa == bb,
                             baseline=baseline, candidate=candidate,
                             baseline_sha256=hashlib.sha256(json.dumps(aa, sort_keys=True).encode()).hexdigest(),
                             candidate_sha256=hashlib.sha256(json.dumps(bb, sort_keys=True).encode()).hexdigest()))
report = dict(cases=len(rows), all_match=all(r['match'] for r in rows), rows=rows,
              note='Same-format greedy transcript content, reasoning, token counts and finish reason; wall time excluded. Baseline is the previously audited c9a98ac7 campaign.')
(out / 'decode-release-transcript-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['all_match'] else 1)
