"""Matched Qwen admission measurements on the reserved GB10 cluster.

Requires an idle production deployment. Each process uses a fresh test server;
finally restores the supplied production executable and configuration.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from serve_admission_bench import Bench, SCENARIOS
from arrival_bench import ArrivalBench
sys.path.insert(0, str(ROOT / 'benchmarks/results/2026-10-11-pr95-followups'))
from node_snapshot import capture

ENV = {**os.environ, 'CUDA_DEVICE_MAX_CONNECTIONS': '32', 'DGPP_NO_SWAP': '1'}
for key in ('DGPP_ROCE_DEVICES', 'DGPP_ROCE_GID_INDICES'):
    ENV.pop(key, None)


def run(command, path, check=True, timeout=3600):
    receipt = {'command': list(map(str, command)),
               'started': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    print(receipt['started'], str(path), flush=True)
    with path.open('w') as log:
        try:
            receipt['returncode'] = subprocess.run(receipt['command'], cwd=ROOT, env=ENV,
                stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            receipt['returncode'] = 124
    receipt['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.with_suffix(path.suffix + '.command.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if check and receipt['returncode']:
        raise RuntimeError(f'{path}: exit {receipt["returncode"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--control', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--production-config', type=Path, required=True)
    ap.add_argument('--production-bin', type=Path, required=True)
    ap.add_argument('--worlds', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--idle-budgets', type=int, nargs='+', default=[2048, 256])
    ap.add_argument('--trials', type=int, default=2)
    ap.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    ap.add_argument('--slots', type=int, default=4)
    ap.add_argument('--mtp-depth', type=int, default=3)
    ap.add_argument('--compact', action='store_true')
    ap.add_argument('--words', type=int, default=8192)
    ap.add_argument('--decode-tokens', type=int, default=512)
    ap.add_argument('--arrival-pattern', choices=['counting', 'varied'], default='counting')
    ap.add_argument('--checks', action='store_true')
    ap.add_argument('--same-binary-control', action='store_true')
    ap.add_argument('--labels', nargs='+', choices=['control', 'candidate', 'candidate-deferred'])
    ap.add_argument('--candidate-first', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.resume and args.checks:
        raise ValueError('resume is for timing-only campaigns; run remaining functional checks separately')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    with urllib.request.urlopen('http://127.0.0.1:18080/metrics', timeout=15) as response:
        initial = json.load(response)
    assert initial['scheduler']['active'] == initial['scheduler']['queued'] == initial['service']['pending_admissions'] == 0
    (args.out / 'production-before.json').write_text(json.dumps(initial, indent=2) + '\n')
    base = json.loads((Path(__file__).parent / 'qwen-w1.json').read_text())
    try:
        run(['python3', 'scripts/dgpp-cluster', 'down', '--config', args.production_config], args.out / 'production-down.log')
        pair = 0
        for world in args.worlds:
            for idle in args.idle_budgets:
                labels = [('control', args.control, False), ('candidate', args.candidate, True)]
                if bool(pair % 2) != args.candidate_first:
                    labels.reverse()
                pair += 1
                if args.same_binary_control:
                    labels.append(('candidate-deferred', args.candidate, False))
                if args.labels:
                    labels = [entry for entry in labels if entry[0] in args.labels]
                    if not labels:
                        raise ValueError('no deployment labels selected; candidate-deferred requires --same-binary-control')
                for label, binary, early in labels:
                    directory = args.out / f'w{world}-i{idle}-{label}'
                    if args.resume and (directory / 'admission.json').exists():
                        previous = json.loads((directory / 'admission.json').read_text())['runs']
                        count = 3 if label == 'candidate-deferred' else args.trials * len(args.scenarios)
                        streams = directory / 'opstreams.json'
                        if len(previous) == count and streams.exists() and json.loads(streams.read_text())['match']:
                            print('keep completed timing', directory, flush=True)
                            continue
                        raise RuntimeError(f'{directory}: incomplete timing data; choose a fresh output directory')
                    directory.mkdir()
                    config = json.loads(json.dumps(base))
                    config['world_size'] = world
                    config['engine'].update(prefill_idle_budget_tokens=idle, max_concurrency=args.slots,
                        mtp=args.mtp_depth > 0, mtp_depth=max(1, args.mtp_depth), compact_batches=args.compact)
                    if early:
                        config['engine'].pop('prefill_early_admission', None)
                    else:
                        config['engine']['prefill_early_admission'] = False
                    if world == 4:
                        config['model'] = 'Qwen/Qwen3.8-Flash-Next-FP8'
                        config['engine']['queue_limit'] = 8
                    path = directory / 'deployment.json'
                    path.write_text(json.dumps(config, indent=2) + '\n')
                    (directory / 'binary.sha256').write_text(hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest() + '\n')
                    try:
                        run(['python3', 'scripts/dgpp-cluster', 'up', '--config', path,
                             '--bin', binary, '--log-dir', directory / 'world'], directory / 'up.log')
                        bench_args = argparse.Namespace(url='http://127.0.0.1:18081', timeout=600,
                            seed=104, prompt_words=args.words, decode_tokens=args.decode_tokens,
                            arrival_tokens=128, arrivals=8, interval=.25, arrival_pattern=args.arrival_pattern)
                        bench = ArrivalBench(bench_args)
                        start = bench.idle()
                        assert start['service']['admission']['prefill_early_admission'] == early
                        bench.stream('warmup', 'Continue counting: 1, 2, 3,', 32)
                        bench.idle(min_terminal=start['scheduler']['terminal'] + 1)
                        capture(directory, 'before')
                        results = []
                        scenarios = ['sustained_idle'] if label == 'candidate-deferred' else args.scenarios
                        trials = 3 if label == 'candidate-deferred' else args.trials
                        for trial in range(trials):
                            for scenario in (scenarios if trial % 2 == 0 else list(reversed(scenarios))):
                                result = bench.run(scenario, trial)
                                results.append(result)
                                (directory / 'admission.json').write_text(json.dumps(
                                    {'label': label, 'arguments': vars(bench_args), 'runs': results}, indent=2) + '\n')
                                print(world, idle, label, trial, scenario, round(result['wall_s'], 3), flush=True)
                        capture(directory, 'after')
                        if args.checks and early and idle == args.idle_budgets[0]:
                            checks = Path(__file__).parent
                            if args.slots == 4 and args.mtp_depth == 3:
                                run(['python3', checks / 'reservation_check.py', directory / 'reservation.json'], directory / 'reservation.log')
                            run(['python3', 'scripts/serve_api_check.py', '127.0.0.1', '18081'], directory / 'api.log')
                            run(['python3', 'scripts/serve_greedy_transcript.py', '127.0.0.1', '18081',
                                 '--max-tokens', '128', '--out', directory / 'transcripts.json'], directory / 'transcripts.log')
                            run(['python3', checks / 'mixed_cache_check.py', '--out', directory / 'mixed-cache.json'], directory / 'mixed-cache.log')
                            run(['python3', 'scripts/serve_soak.py', '127.0.0.1', '18081', '2', directory / 'soak'], directory / 'soak.log')
                            recovered = bench.run('solo_decode', 99)
                            (directory / 'recovery.json').write_text(json.dumps(recovered, indent=2) + '\n')
                    finally:
                        run(['python3', 'scripts/dgpp-cluster', 'down', '--config', path,
                             '--log-dir', directory / 'world'], directory / 'down.log', check=False)
                        hashes = {p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                                  for p in sorted((directory / 'world').glob('serve_rank*.ops'))}
                        match = len(hashes) == world and len(set(hashes.values())) == 1
                        (directory / 'opstreams.json').write_text(json.dumps({'hashes': hashes, 'match': match}, indent=2) + '\n')
                    assert match, directory
    finally:
        run(['python3', 'scripts/dgpp-cluster', 'up', '--config', args.production_config,
             '--bin', args.production_bin], args.out / 'production-restore.log')


if __name__ == '__main__':
    main()
