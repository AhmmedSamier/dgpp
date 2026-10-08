"""Replay the two DeepSeek-V4.1 baselines with only the query-tiling change."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

DIAG = Path(__file__).resolve().parent
MATRIX = DIAG.parents[1]
ROOT = MATRIX.parents[2]
sys.path.insert(0, str(MATRIX))
import run

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['deepseek-1m-w4', 'deepseek-w4'])
    args = parser.parse_args()
    candidate = read(DIAG / 'candidate.json')
    assert sha(ROOT / 'build-release/dgpp-serve') == candidate['binary_sha256']
    for name, digest in candidate['source_sha256'].items():
        assert sha(ROOT / name) == digest
    # The diagnostic has its own manifest and results. The main campaign's
    # binary identity and previous evidence remain unchanged during validation.
    harness = DIAG / 'comparison'
    harness.mkdir(exist_ok=True)
    manifest = read(MATRIX / 'manifest.json')
    manifest['binary_sha256'] = candidate['binary_sha256']
    manifest['version'] = subprocess.check_output([
        'systemd-run', '--user', '--scope', '--quiet', '--property=MemorySwapMax=0', '--',
        str(ROOT / 'build-release/dgpp-serve'), '--version'], text=True).strip()
    manifest['capacity_prefill_candidate'] = candidate
    run.campaign.save(harness / 'manifest.json', manifest)
    shutil.copy2(MATRIX / 'node_telemetry.py', harness / 'node_telemetry.py')
    run.HERE = harness
    run.campaign.HERE = harness
    campaign = run.campaign
    entries = {e['id']: e for e in read(MATRIX / 'matrix.json')}
    for ident in ['deepseek-1m-w4', 'deepseek-w4']:
        if args.only and ident != args.only:
            continue
        entry = dict(entries[ident])
        entry['config'] = str(MATRIX / entry['config'])
        baseline = (MATRIX / 'attempts/deepseek-1m-w4/default-before-capacity-investigation-20261007'
                    if ident == 'deepseek-1m-w4' else MATRIX / 'raw/deepseek-w4/refresh/default')
        assert read(baseline / 'memory-check.json')['passed']
        assert read(baseline / 'rank-identity.json')['passed']
        assert read(baseline / 'down.log.command.json')['returncode'] == 0
        directory = harness / 'raw' / ident / 'validation/default'
        assert not directory.exists(), directory
        jobs = ['prefill', 'context-32768', 'context-65536']
        if ident == 'deepseek-w4':
            jobs.append('context-131072')
        comparisons = []
        config = server = None
        try:
            directory, config, server = campaign.launch(entry, 'validation', 'default')
            assert read(config) == read(baseline / 'config.json')
            campaign.save(directory / 'measurement-plan.json', {'jobs': jobs, 'baseline': str(baseline),
                'reason': 'Same configuration, tags, seeds and prompts; only CSA2 query tiling changed.'})
            for job in jobs:
                old = read(baseline / f'{job}.json')
                if job == 'prefill':
                    command = campaign.client('serve_prefill_probe.py', 2048, 8192, 32768,
                        '--repeat', 3, '--no-think', '--seed', old['seed'], '--tag', old['tag'],
                        '--json-out', directory / f'{job}.json')
                else:
                    command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/serve_context_matrix.py'),
                        '127.0.0.1', '18080', '--model', entry['model'], '--tag', old['tag'],
                        '--lengths', job.split('-')[1], '--json-out', str(directory / f'{job}.json')]
                monitors = campaign.TELEMETRY[str(directory)][:4]
                assert len(monitors) == 4 and all(p.poll() is None for p, _ in monitors)
                assert campaign.run(command, directory / f'{job}.log', timeout=43200, resume=False)
                assert all(p.poll() is None for p, _ in monitors)
                new = read(directory / f'{job}.json')
                assert len(old['samples']) == len(new['samples'])
                for before, after in zip(old['samples'], new['samples']):
                    left = before if job == 'prefill' else before['answer']
                    right = after if job == 'prefill' else after['answer']
                    same = all(left[k] == right[k] for k in ['prompt_sha256', 'text', 'finish', 'usage'])
                    result = {'job': job, 'length': before['requested_tokens'], 'repeat': before['repeat'],
                        'prompt_and_output_equal': same,
                        'before_prefill_ms': before['prefill_ms'] if job == 'prefill' else before['engine']['prefill_ms'],
                        'after_prefill_ms': after['prefill_ms'] if job == 'prefill' else after['engine']['prefill_ms']}
                    comparisons.append(result)
                    campaign.save(DIAG / f'{ident}-comparison.json', {'complete': False, 'samples': comparisons})
                print(f'{ident} {job}: {sum(r["prompt_and_output_equal"] for r in comparisons)}/{len(comparisons)} identical responses so far', flush=True)
        finally:
            if config is not None:
                campaign.stop(directory, config, server)
                run.check_memory(directory, config)
                assert read(directory / 'rank-identity.json')['passed']
        campaign.save(DIAG / f'{ident}-comparison.json', {'complete': True,
            'all_prompts_and_outputs_equal': all(r['prompt_and_output_equal'] for r in comparisons),
            'samples': comparisons, 'candidate_binary_sha256': candidate['binary_sha256'],
            'baseline_binary_sha256': read(baseline / 'manifest.json')['binary_sha256'],
            'baseline_directory': str(baseline.relative_to(MATRIX)),
            'candidate_directory': str(directory.relative_to(MATRIX))})

if __name__ == '__main__':
    main()
