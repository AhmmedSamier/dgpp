"""Run the remaining serving checks on an idle, reserved cluster."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

from campaign import ROOT, run, capture
sys.path.insert(0, str(ROOT / 'scripts'))
from serve_admission_bench import Bench

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--binary', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--world', type=int, choices=[1, 4], required=True)
ap.add_argument('--checks', nargs='+', choices=['api', 'transcripts', 'reservation', 'cache', 'soak'], required=True)
ap.add_argument('--prefix-min-tokens', type=int, default=1024)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
production = ROOT / 'deploy/cluster_qwen-3.8-flash-next_fp8_w4.example.json'
with urllib.request.urlopen('http://127.0.0.1:18080/metrics', timeout=15) as response:
    initial = json.load(response)
assert initial['scheduler']['active'] == initial['scheduler']['queued'] == initial['service']['pending_admissions'] == 0
here = Path(__file__).parent
config = json.loads((here / 'qwen-w1.json').read_text())
config['world_size'] = args.world
config['engine']['prefix_min_tokens'] = args.prefix_min_tokens
if args.world == 4:
    config['model'] = 'Qwen/Qwen3.8-Flash-Next-FP8'
    config['engine']['queue_limit'] = 8
path = args.out / 'deployment.json'
path.write_text(json.dumps(config, indent=2) + '\n')
(args.out / 'binary.sha256').write_text(hashlib.file_digest(args.binary.open('rb'), 'sha256').hexdigest() + '\n')
try:
    run(['python3', 'scripts/dgpp-cluster', 'down', '--config', production], args.out / 'production-down.log')
    try:
        run(['python3', 'scripts/dgpp-cluster', 'up', '--config', path, '--bin', args.binary,
             '--log-dir', args.out / 'world'], args.out / 'up.log')
        capture(args.out, 'before')
        if 'reservation' in args.checks:
            run(['python3', here / 'reservation_check.py', args.out / 'reservation.json'], args.out / 'reservation.log')
        if 'api' in args.checks:
            run(['python3', 'scripts/serve_api_check.py', '127.0.0.1', '18081'], args.out / 'api.log')
        if 'transcripts' in args.checks:
            run(['python3', 'scripts/serve_greedy_transcript.py', '127.0.0.1', '18081', '--max-tokens', '128',
                 '--out', args.out / 'transcripts.json'], args.out / 'transcripts.log')
        if 'cache' in args.checks:
            run(['python3', here / 'mixed_cache_check.py', '--no-thinking', '--out', args.out / 'mixed-cache.json'], args.out / 'mixed-cache.log')
        bench = Bench(argparse.Namespace(url='http://127.0.0.1:18081', timeout=600,
            seed=104, prompt_words=8192, decode_tokens=32, arrival_tokens=128, arrivals=8, interval=.25))
        if 'soak' in args.checks:
            before = bench.idle()
            (args.out / 'soak-before.json').write_text(json.dumps(before, indent=2) + '\n')
            run(['python3', 'scripts/serve_soak.py', '127.0.0.1', '18081', '2', args.out / 'soak'], args.out / 'soak.log')
            after = bench.idle()
            (args.out / 'soak-after.json').write_text(json.dumps(after, indent=2) + '\n')
            records = [json.loads(line) for line in (args.out / 'soak/requests.jsonl').read_text().splitlines()]
            responses = [record for record in records if 'status' in record]
            assert responses and all(r['status'] in (200, 503) for r in responses)
            assert all(r.get('code') == 'overloaded' and r['tokens'] == 0
                       for r in responses if r['status'] == 503)
            # Admission can reject after the SSE preamble committed HTTP 200.
            # Only an explicit zero-token overload event is an expected error.
            late_overload = [r for r in responses if r['status'] == 200 and r.get('code') == 'overloaded' and r['tokens'] == 0]
            assert all(not r.get('code') or r in late_overload for r in responses if r['status'] == 200)
            assert any(r['kind'] == 'burst_summary' for r in records)
            assert after['service']['requests_failed'] == before['service']['requests_failed'] == 0
            # blocks_pinned counts entry references, including shared blocks.
            assert after['scheduler']['pool_blocks_in_use'] <= after['prefix_cache']['blocks_pinned']
            print('Soak status counts:', dict(Counter(r['status'] for r in responses)), flush=True)
            print('Overload events after SSE headers:', len(late_overload), flush=True)
        recovery = bench.run('solo_decode', 99)
        (args.out / 'recovery.json').write_text(json.dumps(recovery, indent=2) + '\n')
        capture(args.out, 'after')
    finally:
        run(['python3', 'scripts/dgpp-cluster', 'down', '--config', path,
             '--log-dir', args.out / 'world'], args.out / 'down.log', check=False)
        hashes = {p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                  for p in sorted((args.out / 'world').glob('serve_rank*.ops'))}
        match = len(hashes) == args.world and len(set(hashes.values())) == 1
        (args.out / 'opstreams.json').write_text(json.dumps({'hashes': hashes, 'match': match}, indent=2) + '\n')
    assert match
finally:
    run(['python3', 'scripts/dgpp-cluster', 'up', '--config', production, '--bin', 'build-release/dgpp-serve'],
        args.out / 'production-restore.log')
