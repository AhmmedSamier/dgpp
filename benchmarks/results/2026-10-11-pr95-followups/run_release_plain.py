"""Final binary: plain-decode FP8/NVFP4/FP8 bracket, three repetitions."""
import datetime
import hashlib
import json
import os
import pathlib
import statistics
import subprocess

out = pathlib.Path(__file__).parent
root = pathlib.Path('/home/stephen/workspace/dgpp')
binary = out / 'dgpp-serve-release-final'
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
env = {**os.environ, 'PR95_BIN': str(binary)}
results = []
for suffix, quant in (('fp8a', 'fp8'), ('nvfp4', 'nvfp4'), ('fp8b', 'fp8')):
    label = f'decode-release-plain-{suffix}'
    cmd = ['python3', str(out / 'campaign.py'), quant, '4', label,
           '--draft-probe', '--plain', '--probe-repeat', '3']
    receipt = dict(label=label, started=datetime.datetime.now(datetime.timezone.utc).isoformat(), command=cmd)
    print(receipt['started'], label, flush=True)
    with (out / f'{label}-controller.log').open('w') as log:
        p = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    receipt.update(returncode=p.returncode, finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    assert p.returncode == 0, receipt
    d = out / label
    assert (d / 'binary.sha256').read_text().strip() == digest
    assert json.loads((d / 'opstreams.json').read_text())['match']
    phases = json.loads((d / 'decode.json').read_text())['phases']
    assert len(phases) == 3
    for phase in phases:
        r, = phase['requests']
        assert r['status'] == 200 and r['tokens'] == 256
        assert phase['engine']['decode_steps'] == 255
    receipt.update(rates=[p['engine']['tokens_per_s'] for p in phases],
                   prompt_hashes=[p['requests'][0]['prompt_sha256'] for p in phases],
                   median=statistics.median(p['engine']['tokens_per_s'] for p in phases))
    results.append(receipt)
    (out / 'decode-release-plain-status.json').write_text(json.dumps(results, indent=2) + '\n')
assert len({tuple(r['prompt_hashes']) for r in results}) == 1
best = max(results[0]['median'], results[2]['median'])
report = dict(binary_sha256=digest, legs=results, gain_pct=100 * (results[1]['median'] / best - 1),
              nvfp4_faster=results[1]['median'] > best)
(out / 'decode-release-plain-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['nvfp4_faster'] else 1)
