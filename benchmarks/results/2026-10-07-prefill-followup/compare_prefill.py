"""Controlled fresh-server prefill comparisons with per-rank memory evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import urllib.request
from evidence import pack_run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / '2026-10-06-prefill-scaling'))
import run as runner

runner.HERE = HERE
campaign = runner.campaign
campaign.HERE = HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('deployment')
    parser.add_argument('variant')
    parser.add_argument('--idle-budget', type=int)
    parser.add_argument('--lengths', nargs='+', type=int, default=[2048, 8192, 32768])
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--candidate-binary', type=Path)
    parser.add_argument('--decode-check', action='store_true')
    args = parser.parse_args()
    if args.candidate_binary and args.candidate_binary.name != 'dgpp-serve':
        raise RuntimeError('Candidate basename must be dgpp-serve for process telemetry')
    entry = next(e for e in json.loads((HERE / 'matrix.json').read_text())
                 if e['id'] == args.deployment)
    expected = HERE / 'raw' / entry['id'] / 'prefill' / args.variant
    if expected.exists():
        raise RuntimeError(f'Refusing to overwrite {expected}')
    if args.profile or args.candidate_binary:
        original_run = campaign.run
        def profiled_run(command, output, **kwargs):
            if output.name == 'up.log':
                command = list(command)
                if args.candidate_binary:
                    binary = args.candidate_binary.resolve()
                    command[command.index('--bin') + 1] = str(binary)
                    manifest = json.loads((expected / 'manifest.json').read_text())
                    manifest['baseline_binary_sha256'] = manifest['binary_sha256']
                    manifest['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
                    manifest['candidate_binary'] = str(binary)
                    campaign.save(expected / 'manifest.json', manifest)
                if args.profile:
                    wrapper = ['nsys', 'profile', '--trace=cuda', '--sample=none',
                        '--cpuctxsw=none', '--cuda-graph-trace=graph',
                        '--output', str(expected / 'r0')]
                    command = [*command, '--head-wrap', shlex.join(wrapper)]
            return original_run(command, output, **kwargs)
        campaign.run = profiled_run
    overrides = {} if args.idle_budget is None else {'prefill_idle_budget_tokens': args.idle_budget}
    config = server = None
    try:
        directory, config, server = campaign.launch(entry, 'prefill', args.variant, overrides)
        campaign.save(directory / 'plan.json', {'deployment': args.deployment,
            'variant': args.variant, 'overrides': overrides,
            'lengths': args.lengths, 'repeat': args.repeat,
            'profiled': args.profile,
            'purpose': 'Diagnostic A/B; published matrix remains unchanged.'})
        monitors = campaign.TELEMETRY[str(directory)]
        assert all(p.poll() is None for p, _ in monitors), 'monitor stopped'
        command = campaign.client('serve_prefill_probe.py', *args.lengths,
            '--repeat', args.repeat, *campaign.thinking_args(entry), '--seed', 7,
            '--tag', 'followup-' + entry['id'], '--interval', 1 if args.profile else 0,
            '--json-out', directory / 'prefill.json')
        if not campaign.run(command, directory / 'prefill.log', timeout=3600, resume=False):
            raise RuntimeError('Prefill probe failed')
        if args.decode_check:
            replies = []
            for prompt in [
                'Write a Python function that merges two sorted lists and explain its time complexity.',
                'A rectangle has perimeter 50 and area 150. Find its side lengths and explain your calculation.',
            ]:
                body = {'model': entry['model'], 'messages': [{'role': 'user', 'content': prompt}],
                        'temperature': 0, 'max_tokens': 256}
                request = urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',
                    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=180) as response:
                    reply = json.load(response)
                assert reply['choices'] and reply['usage']['completion_tokens'] > 0
                replies.append({'request': body, 'response': reply})
            campaign.save(directory / 'decode-check.json', replies)
        assert all(p.poll() is None for p, _ in monitors), 'monitor stopped'
    finally:
        if config is not None:
            campaign.stop(directory, config, server)
            runner.check_memory(directory, config)
            assert json.loads((directory / 'rank-identity.json').read_text())['passed']
    campaign.save(directory / 'complete.json', {'at': campaign.now(), 'passed': True})
    pack_run(directory)


if __name__ == '__main__':
    main()
