"""Focused before/after serving checks for a shared-prefill refactor.

Use the same deployment, weights, prompts and output lengths for both binaries.
Leave admission settings automatic to compare the previous and new defaults.
GPU tests, downloads and other benchmarks must have finished before running.
The original production deployment is restored in finally.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from serve_admission_bench import Bench
from site_env import run_paths
sys.path.insert(0, str(ROOT / 'benchmarks/results/2026-10-11-pr95-followups'))
from node_snapshot import capture, PROBE


def verify_production(directory, expected, binary_hash, output):
    config = json.loads((directory / 'cluster.resolved.json').read_text())
    assert config == expected, 'production configuration changed'
    def inspect(item):
        rank, host = item
        state = str(directory / 'rank0.process.json') if rank == 0 else config['paths']['stage_dir'] + f'/rank{rank}.process.json'
        command = ['python3', '-', state]
        if rank:
            command = ['ssh', '-o', 'BatchMode=yes', config['ssh_user'] + '@' + host, shlex.join(command)]
        result = subprocess.run(command, input=PROBE, text=True, capture_output=True, timeout=30, check=True)
        return dict(rank=rank, **json.loads(result.stdout))
    with ThreadPoolExecutor(max_workers=len(config['nodes'])) as pool:
        nodes = list(pool.map(inspect, enumerate(config['nodes'])))
    assert all(node['binary_sha256'] == binary_hash for node in nodes), 'production executable changed'
    with urllib.request.urlopen('http://127.0.0.1:18080/metrics', timeout=15) as response:
        metrics = json.load(response)
    assert not metrics['service']['engine_failed']
    output.write_text(json.dumps(dict(config_unchanged=True, binary_sha256=binary_hash,
                                     nodes=nodes, metrics=metrics), indent=2) + '\n')
    print('Original production configuration and live executables restored; HTTP healthy.', flush=True)


class Checks(Bench):
    def batch(self, scenario, trial):
        before = self.idle()
        rng = random.Random(self.args.seed + trial)
        words = 'amber birch cobalt dune ember fjord granite harbor indigo juniper quartz willow'.split()
        prompt = ('Document:\n' + ' '.join(rng.choice(words) for _ in range(self.args.prompt_words)) + '\nSummary:'
                  if scenario == 'batch_prefill' else 'List the positive integers in order:\n1,')
        tokens = 1 if scenario == 'batch_prefill' else self.args.decode_tokens
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(lambda n: self.stream(f'batch-{n}', f'Task {n}. {prompt}', tokens), range(4)))
        after = self.idle(before['scheduler']['terminal'] + len(records))
        elapsed = time.monotonic() - started
        assert after['scheduler']['terminal'] - before['scheduler']['terminal'] == len(records)
        assert after['service']['requests_total'] - before['service']['requests_total'] == len(records)
        assert after['scheduler']['pool_blocks_in_use'] <= before['scheduler']['pool_blocks_in_use']
        generated = after['scheduler']['tokens_generated'] - before['scheduler']['tokens_generated']
        computed = after['scheduler']['prompt_tokens_computed'] - before['scheduler']['prompt_tokens_computed']
        return dict(scenario=scenario, trial=trial, wall_s=elapsed, requests=records,
                    generated_tokens=generated, computed_prompt_tokens=computed,
                    aggregate_output_tokens_per_second=generated / elapsed,
                    aggregate_prompt_tokens_per_second=computed / elapsed, before=before, after=after)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--deployment', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--control', type=Path, required=True)
    ap.add_argument('--production-config', type=Path, required=True)
    ap.add_argument('--production-bin', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--trials', type=int, default=2)
    ap.add_argument('--words', type=int, default=8192)
    ap.add_argument('--decode-tokens', type=int, default=128)
    ap.add_argument('--candidate-first', action='store_true')
    ap.add_argument('--candidate-budget', type=int,
                    help='Experimental candidate override; omit for the default-vs-default gate.')
    ap.add_argument('--candidate-idle-budget', type=int, help='Experimental candidate idle budget override.')
    ap.add_argument('--no-resident-cache', action='store_true')
    ap.add_argument('--cache-check', action='store_true')
    ap.add_argument('--cache-no-thinking', action='store_true',
                    help='Use the checkpoint template’s no-thinking counting response for the cache gate.')
    ap.add_argument('--explicit-budget-check', action='store_true',
                    help='After timing, also boot the candidate with an explicit 256-token budget.')
    ap.add_argument('--profile', action='store_true', help='CUDA trace on one node; not a throughput measurement.')
    ap.add_argument('--download-python', type=Path,
                    help='Download/sync this deployment with this venv after stopping production.')
    ap.add_argument('--download-revision', help='Pin a temporary checkpoint download to this revision.')
    ap.add_argument('--scenarios', nargs='+', default=['solo_decode', 'solo_prefill', 'batch_decode',
                                                     'batch_prefill', 'mixed_long', 'sustained_idle'])
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'harness.py').write_bytes(Path(__file__).read_bytes())
    (args.out / 'benchmark-client.py').write_bytes((ROOT / 'scripts/serve_admission_bench.py').read_bytes())
    environment = {**os.environ, 'CUDA_DEVICE_MAX_CONNECTIONS': '32', 'DGPP_NO_SWAP': '1'}
    for key in ('DGPP_ROCE_DEVICES', 'DGPP_ROCE_GID_INDICES'):
        environment.pop(key, None)
    test_environment = dict(environment)
    if args.no_resident_cache:
        test_environment['DGPP_RESIDENT_CACHE'] = 'off'

    def run(command, output, test=False, check=True, timeout=1800):
        print(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), output, flush=True)
        with output.open('w') as f:
            result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                    env=test_environment if test else environment,
                                    stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        output.with_suffix(output.suffix + '.command.json').write_text(
            json.dumps(dict(command=list(map(str, command)), returncode=result.returncode), indent=2) + '\n')
        if check:
            result.check_returncode()

    with urllib.request.urlopen('http://127.0.0.1:18080/metrics', timeout=15) as response:
        initial = json.load(response)
    assert initial['scheduler']['active'] == initial['scheduler']['queued'] == initial['scheduler']['prefilling'] == initial['service']['pending_admissions'] == 0
    (args.out / 'production-before.json').write_text(json.dumps(initial, indent=2) + '\n')
    production = Path(run_paths(args.production_config)['log_dir']).expanduser()
    production_config = json.loads((production / 'cluster.resolved.json').read_text())
    with args.production_bin.open('rb') as f:
        production_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    verify_production(production, production_config, production_hash, args.out / 'production-original.json')
    base = json.loads(args.deployment.read_text())
    assert base['http']['port'] == 18081
    if args.profile:
        assert base['world_size'] == 1, 'this profiling hook wraps rank 0 only'
    labels = [('control', args.control), ('candidate', args.candidate)]
    if args.candidate_first:
        labels.reverse()
    try:
        run(['python3', 'scripts/dgpp-cluster', 'down', '--config', args.production_config], args.out / 'production-down.log')
        if args.download_python:
            command = [args.download_python, 'scripts/download_model.py', '--config', args.deployment]
            if args.download_revision:
                command += ['--revision', args.download_revision, '--activate']
            run(command, args.out / 'download.log', test=True, timeout=7200)
        for label, binary in labels:
            directory = args.out / label
            directory.mkdir()
            path = directory / 'deployment.json'
            config = json.loads(json.dumps(base))
            if label == 'candidate' and args.candidate_budget is not None:
                config['engine']['prefill_budget_tokens'] = args.candidate_budget
            if label == 'candidate' and args.candidate_idle_budget is not None:
                config['engine']['prefill_idle_budget_tokens'] = args.candidate_idle_budget
            path.write_text(json.dumps(config, indent=2) + '\n')
            with binary.open('rb') as f:
                (directory / 'binary.sha256').write_text(hashlib.file_digest(f, 'sha256').hexdigest() + '\n')
            try:
                command = ['python3', 'scripts/dgpp-cluster', 'up', '--config', path,
                           '--bin', binary, '--log-dir', directory / 'world']
                if args.profile:
                    command += ['--head-wrap', f'/usr/local/cuda/bin/nsys profile -t cuda --sample=none --cuda-graph-trace=node --force-overwrite=true -o {directory}/rank0']
                run(command, directory / 'up.log', test=True)
                options = argparse.Namespace(url='http://127.0.0.1:18081', timeout=600, seed=104,
                    prompt_words=args.words, decode_tokens=args.decode_tokens, arrival_tokens=64,
                    arrivals=3, interval=.25, record_text=True)
                bench = Checks(options)
                start = bench.idle()
                if label == 'candidate':
                    assert start['service']['admission']['prefill_early_admission'] is True
                bench.stream('warmup', 'Continue counting: 1, 2, 3,', 32)
                bench.idle(start['scheduler']['terminal'] + 1)
                if not args.profile:
                    # Warm the deterministic prompt's memory-mapped embedding
                    # pages on both builds before steady throughput timing.
                    warmups = [bench.run('solo_prefill', trial) for trial in range(args.trials)]
                    (directory / 'prefill-warmups.json').write_text(json.dumps(warmups, indent=2) + '\n')
                capture(directory, 'before', profile=args.profile)
                if args.profile:
                    time.sleep(2)  # separate the warmup from the measured trace
                results = []
                for trial in range(args.trials):
                    for scenario in (args.scenarios if trial % 2 == 0 else list(reversed(args.scenarios))):
                        result = (bench.batch(scenario, trial) if scenario.startswith('batch_')
                                  else bench.run(scenario, trial))
                        results.append(result)
                        (directory / 'admission.json').write_text(json.dumps(
                            dict(label=label, arguments=vars(options), runs=results), indent=2) + '\n')
                        print(label, trial, scenario, round(result['wall_s'], 3), flush=True)
                capture(directory, 'after', profile=args.profile)
                if not args.profile:
                    run(['python3', 'scripts/serve_greedy_transcript.py', '127.0.0.1', '18081',
                         '--max-tokens', '64', '--out', directory / 'transcripts.json'], directory / 'transcripts.log', test=True)
                if args.cache_check:
                    cache_script = ROOT / 'benchmarks/results/2026-10-12-prefill-consolidation/mixed_cache_check.py'
                    (directory / 'mixed-cache-script.py').write_bytes(cache_script.read_bytes())
                    command = ['python3', cache_script,
                               '--out', directory / 'mixed-cache.json']
                    if args.cache_no_thinking:
                        command.append('--no-thinking')
                    if label == 'control':
                        command.append('--diagnostic')  # the old executor may defer admission
                    run(command, directory / 'mixed-cache.log', test=True)
                bench.idle()
            finally:
                run(['python3', 'scripts/dgpp-cluster', 'down', '--config', path,
                     '--log-dir', directory / 'world'], directory / 'down.log', test=True, check=False)
                hashes = {p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                          for p in sorted((directory / 'world').glob('serve_rank*.ops'))}
                single_eager = base['world_size'] == 1 and not base['engine'].get('decode_graph', True)
                match = len(hashes) == base['world_size'] and len(set(hashes.values())) == 1
                if single_eager and not hashes:
                    match = None  # no fabric operation journal in the local eager path
                (directory / 'opstreams.json').write_text(json.dumps(dict(
                    hashes=hashes, match=match, required=not single_eager), indent=2) + '\n')
            assert match is True or (single_eager and not hashes), directory
            if args.profile:
                run(['/usr/local/cuda/bin/nsys', 'export', '--type', 'sqlite', '--force-overwrite=true',
                     '-o', directory / 'rank0.sqlite', directory / 'rank0.nsys-rep'], directory / 'export.log')
        if args.explicit_budget_check:
            directory = args.out / 'explicit-budget'
            directory.mkdir()
            config = json.loads(json.dumps(base))
            config['engine']['prefill_budget_tokens'] = 256
            path = directory / 'deployment.json'
            path.write_text(json.dumps(config, indent=2) + '\n')
            try:
                run(['python3', 'scripts/dgpp-cluster', 'up', '--config', path,
                     '--bin', args.candidate, '--log-dir', directory / 'world'],
                    directory / 'up.log', test=True)
                bench = Checks(options)
                policy = bench.idle()['service']['admission']
                assert policy['prefill_budget_tokens'] == 256 and policy['prefill_early_admission']
                result = bench.run('mixed_long', 0)
                (directory / 'mixed.json').write_text(json.dumps(result, indent=2) + '\n')
            finally:
                run(['python3', 'scripts/dgpp-cluster', 'down', '--config', path,
                     '--log-dir', directory / 'world'], directory / 'down.log', test=True, check=False)
    finally:
        run(['python3', 'scripts/dgpp-cluster', 'up', '--config', args.production_config,
             '--bin', args.production_bin], args.out / 'production-restore.log')
        verify_production(production, production_config, production_hash, args.out / 'production-verified.json')


if __name__ == '__main__':
    main()
