"""Replay full-GLM baseline prompts with the candidate and retain normal evidence."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[2]
ROOT = HERE.parents[2]
DIAG = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run

def read(path):
    return json.loads(path.read_text())

def main():
    entry = next(e for e in read(HERE / 'matrix.json') if e['id'] == 'glm53-fp8kv-w4')
    old = HERE / 'attempts' / entry['id'] / 'default-before-fused-scores-20261007'
    directory = HERE / 'raw' / entry['id'] / 'refresh/default'
    assert not directory.exists(), 'Do not overwrite an earlier candidate launch'
    campaign = run.campaign
    config = server = None
    jobs = ['prefill', 'context-32768', 'context-65536', 'context-131072']
    comparisons = []
    try:
        directory, config, server = campaign.launch(entry, 'refresh', 'default')
        campaign.save(directory / 'measurement-plan.json', {
            'jobs': jobs, 'prefill_lengths': [2048, 8192, 32768],
            'comparison_baseline': str(old.relative_to(HERE)),
            'reason': 'Same tags, seed, prompts, config and client code for a controlled full-model comparison. '
                      'Short serving/context and above-limit skips retain their separately audited baseline.'})
        for job in jobs:
            baseline = read(old / f'{job}.json')
            monitors = campaign.TELEMETRY[str(directory)][:4]
            assert len(monitors) == 4 and all(p.poll() is None for p, _ in monitors)
            if job == 'prefill':
                command = campaign.client('serve_prefill_probe.py', 2048, 8192, 32768,
                    '--repeat', 3, '--seed', baseline['seed'], '--tag', baseline['tag'],
                    '--json-out', directory / f'{job}.json')
            else:
                command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/serve_context_matrix.py'),
                    '127.0.0.1', '18080', '--model', entry['model'], '--tag', baseline['tag'],
                    '--lengths', job.split('-')[1], '--json-out', str(directory / f'{job}.json')]
            if not campaign.run(command, directory / f'{job}.log', timeout=43200, resume=False):
                raise RuntimeError(f'candidate measurement failed: {job}')
            assert all(p.poll() is None for p, _ in monitors), 'memory monitor stopped'
            candidate = read(directory / f'{job}.json')
            assert len(candidate['samples']) == len(baseline['samples'])
            for before, after in zip(baseline['samples'], candidate['samples']):
                left = before if job == 'prefill' else before['answer']
                right = after if job == 'prefill' else after['answer']
                same = all(left[k] == right[k] for k in ['prompt_sha256', 'text', 'finish', 'usage'])
                record = {'job': job, 'length': before['requested_tokens'], 'repeat': before['repeat'],
                          'prompt_and_output_equal': same,
                          'before_prefill_ms': before['prefill_ms'] if job == 'prefill' else before['engine']['prefill_ms'],
                          'after_prefill_ms': after['prefill_ms'] if job == 'prefill' else after['engine']['prefill_ms']}
                comparisons.append(record)
                campaign.save(DIAG / 'full-model-comparison.json', {'complete': False, 'samples': comparisons})
                assert same, f'full-model prompt/output mismatch: {record}'
            print(f'PARITY {job}: prompt hashes, output text, finish and token usage match', flush=True)
    finally:
        if config is not None:
            campaign.stop(directory, config, server)
            run.check_memory(directory, config)
            assert read(directory / 'rank-identity.json')['passed'], 'rank operation streams differ'
    campaign.save(DIAG / 'full-model-comparison.json', {'complete': True, 'samples': comparisons,
        'candidate_launch': str(directory.relative_to(HERE)), 'baseline_launch': str(old.relative_to(HERE)),
        'candidate_binary_sha256': read(directory / 'manifest.json')['binary_sha256'],
        'baseline_binary_sha256': read(old / 'manifest.json')['binary_sha256']})

if __name__ == '__main__':
    main()
