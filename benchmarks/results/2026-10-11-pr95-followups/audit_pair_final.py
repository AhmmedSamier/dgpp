"""Audit final prefill brackets and unchanged short/long-context outputs."""
import hashlib, json, pathlib, statistics, sys
out = pathlib.Path(__file__).parent
worlds = [int(sys.argv[1])] if len(sys.argv) > 1 else [1, 2, 4]
digest = hashlib.sha256((out / 'dgpp-serve-prefill-pair-final').read_bytes()).hexdigest()
rows, decode, transcripts, cold_outputs, context_outputs = [], [], [], [], []
def sha(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
def compare(bucket, old, new, fields, **labels):
    a = {k: old[k] for k in fields}
    b = {k: new[k] for k in fields}
    bucket.append(dict(**labels, match=a == b, baseline_sha256=sha(a), candidate_sha256=sha(b)))
for world in worlds:
    legs = []
    for suffix in ('fp8a', 'nvfp4', 'fp8b'):
        d = out / f'prefill-pair-final-w{world}-{suffix}'
        baseline = out / f'decode-wide-final-w{world}-{suffix}'
        assert (d / 'binary.sha256').read_text().strip() == digest
        assert json.loads((d / 'opstreams.json').read_text())['match']
        assert 'ALL OK []' in (d / 'api.log').read_text()
        for when in ('before', 'after'):
            nodes = json.loads((d / f'nodes-{when}.json').read_text())
            assert len(nodes) == world
            assert all(n['binary_sha256'] == digest and n['vm_swap_kib'] == 0
                       and n['cgroup_swap_current'] == '0' and n['cgroup_swap_max'] == '0' for n in nodes)
        for p in d.glob('*.command.json'):
            assert json.loads(p.read_text())['returncode'] == 0
        samples = json.loads((d / 'prefill.json').read_text())['samples']
        phases = json.loads((d / 'decode.json').read_text())['phases']
        legs.append((samples, phases))
        assert len(samples) == 12 and all(s['computed_tokens'] == s['prompt_tokens']
                                           and s['cached_tokens'] == 0 for s in samples)
        assert len(phases) == 3 and all(p['class'] == 'prose' and p['concurrency'] == 1
                                        and p['requests'][0]['status'] == 200 for p in phases)
        assert all(len(p['requests']) == 1 and p['requests'][0]['tokens'] == 256
                   and p['requests'][0]['finish'] == 'length'
                   and p['engine']['tokens_generated'] == 256 and p['engine']['decode_tokens'] == 255
                   for p in phases)
        old = json.loads((baseline / 'transcripts.json').read_text())['results']
        new = json.loads((d / 'transcripts.json').read_text())['results']
        assert old.keys() == new.keys()
        for kind in old:
            compare(transcripts, old[kind], new[kind], ('reasoning', 'content', 'prompt_tokens', 'completion_tokens', 'finish'), world=world, leg=suffix, kind=kind)
        old_cold = json.loads((baseline / 'prefill.json').read_text())['samples']
        for a, b in zip(old_cold, samples):
            compare(cold_outputs, a, b, ('prompt_sha256', 'prompt_tokens', 'text', 'usage', 'finish'), world=world, leg=suffix, length=b['requested_tokens'], repeat=b['repeat'])
        if world == 4:
            old_context = json.loads((baseline / 'decode-context.json').read_text())['phases']
            new_context = json.loads((d / 'decode-context.json').read_text())['phases']
            assert len(old_context) == len(new_context) == 9
            for a, b in zip(old_context, new_context):
                compare(context_outputs, a['requests'][0], b['requests'][0], ('prompt_sha256', 'text', 'tokens', 'finish', 'usage'), leg=suffix, length=b['requested_context'], repeat=b['repeat'])
    for n in (512, 2048, 8192, 32768):
        samples = [[s for s in a[0] if s['requested_tokens'] == n] for a in legs]
        for i in range(3):
            assert len({(ss[i]['prompt_sha256'], ss[i]['prompt_tokens']) for ss in samples}) == 1
        med = [statistics.median(s['prefill_ms'] for s in a) for a in samples]
        rows.append(dict(world=world, length=n, fp8a_ms=med[0], nvfp4_ms=med[1], fp8b_ms=med[2],
                         reduction_pct=100*(1-med[1]/min(med[0],med[2])), fp8_drift_pct=100*(med[2]/med[0]-1),
                         nvfp4_faster=med[1]<min(med[0],med[2]),
                         paired_reduction_pct=[100*(1-samples[1][r]['prefill_ms']/min(samples[0][r]['prefill_ms'],samples[2][r]['prefill_ms'])) for r in range(3)]))
    rates = [statistics.median(p['engine']['tokens_per_s'] for p in a[1]) for a in legs]
    for i in range(3):
        assert len({a[1][i]['requests'][0]['prompt_sha256'] for a in legs}) == 1
    decode.append(dict(world=world, kind='prose', concurrency=1, fp8a_tps=rates[0], nvfp4_tps=rates[1], fp8b_tps=rates[2], gain_pct=100*(rates[1]/max(rates[0],rates[2])-1), nvfp4_faster=rates[1]>max(rates[0],rates[2])))
result = dict(binary_sha256=digest, worlds=worlds, prefill=rows, decode=decode, transcripts=transcripts,
              cold_outputs=cold_outputs, context_outputs=context_outputs,
              all_transcripts_match=all(r['match'] for r in transcripts+cold_outputs+context_outputs),
              performance_gate=all(r['nvfp4_faster'] for r in rows+decode))
name = f'prefill-pair-final-w{worlds[0]}-audit.json' if len(worlds)==1 else 'prefill-pair-final-audit.json'
(out / name).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('transcripts','cold_outputs','context_outputs')}, indent=2))
raise SystemExit(0 if result['performance_gate'] and result['all_transcripts_match'] else 1)
