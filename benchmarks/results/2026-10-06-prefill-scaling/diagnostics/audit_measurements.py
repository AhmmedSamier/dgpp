#!/usr/bin/env python3
"""Reconcile validated benchmark reports with raw request usage and counters."""
import argparse
from collections import Counter
import datetime
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from result_sources import completed


def close(actual, expected):
    assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6), (actual, expected)


def audit_greedy(report, mode):
    phases = report['phases']
    concurrencies = [1, 2, 4, 8] if mode == 'default' else [1]
    expected = Counter({(cls, c): 3 for cls in ['prose', 'code', 'json', 'math', 'chat']
                        for c in concurrencies})
    assert Counter((p['class'], p['concurrency']) for p in phases) == expected
    assert report['temperature'] == 0 and report['max_tokens'] == 256
    requests = 0
    for phase in phases:
        replies, engine, metrics = phase['requests'], phase['engine'], phase['metrics']
        assert len(replies) == phase['concurrency']
        assert all(r['status'] == 200 and r['finish'] in ['stop', 'length'] for r in replies)
        usages = [r['usage'] for r in replies]
        output = sum(u['completion_tokens'] for u in usages)
        inputs = sum(u['prompt_tokens'] for u in usages)
        cached = sum(u.get('prompt_tokens_details', {}).get('cached_tokens', 0) for u in usages)
        assert output == metrics['completion_tokens'] == engine['tokens_generated']
        assert inputs == engine['prompt_tokens']
        assert inputs - cached == engine['prompt_tokens_computed']
        assert engine['prompts_prefilled'] == len(replies)
        assert engine['decode_tokens'] == output - len(replies)
        close(engine['tokens_per_s'], 1000 * (output - len(replies)) / engine['step_ms'])
        close(metrics['wall_tokens_per_s'], output / metrics['wall_s'])
        requests += len(replies)
    return {'phases': len(phases), 'requests': requests}


def audit_prefill(report):
    samples = report['samples']
    assert Counter((s['requested_tokens'], s['repeat']) for s in samples) == Counter(
        {(n, r): 1 for n in [2048, 8192, 32768] for r in range(3)})
    assert len({s['prompt_sha256'] for s in samples}) == 9
    for sample in samples:
        usage = sample['usage']
        assert usage['prompt_tokens'] == sample['prompt_tokens'] == sample['computed_tokens']
        assert usage['prompt_tokens_details']['cached_tokens'] == sample['cached_tokens'] == 0
        assert usage['completion_tokens'] == 1
        assert sample['prefill_ms'] > 0
        close(sample['prefill_ms_per_token'], sample['prefill_ms'] / sample['computed_tokens'])
    return {'uncached_requests': len(samples)}


def audit_context(report, length):
    limit = report['surface']['context']['request_limit_tokens']
    if length > limit:
        assert not report['samples']
        assert report['skipped'] == [{'target': length, 'reason': 'exceeds request limit', 'limit': limit}]
        return {'skipped_above_request_limit': limit}
    samples = report['samples']
    assert sorted(s['repeat'] for s in samples) == [0, 1, 2]
    assert len({s['answer']['prompt_sha256'] for s in samples}) == 1
    assert report['max_tokens'] == 256
    for sample in samples:
        assert sample['requested_tokens'] == length
        engine, answer = sample['engine'], sample['answer']
        usage = answer['usage']
        prompt, generated = usage['prompt_tokens'], usage['completion_tokens']
        cached = usage['prompt_tokens_details']['cached_tokens']
        assert prompt + generated <= limit
        assert (0 < prompt <= 256) if length == 0 else length - 1024 <= prompt <= length - 256
        assert engine['prompt_tokens'] == prompt
        assert engine['prompt_tokens_computed'] == prompt - cached
        assert engine['tokens_generated'] == generated
        assert engine['prompts_prefilled'] == 1
        assert answer['finish'] in ['stop', 'length']
        if sample['repeat'] == 0:
            assert cached == 0
        close(sample['engine_tokens_per_s'], 1000 * (generated - 1) / engine['step_ms'])
        close(sample['ms_per_pass'], engine['step_ms'] / engine['decode_steps'])
        close(sample['tokens_per_pass'], (generated - 1) / engine['decode_steps'])
    return {'requests': 3, 'input_tokens': samples[0]['answer']['usage']['prompt_tokens']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    checked, pending = {}, []
    matrix = json.loads((HERE / 'matrix.json').read_text())
    for entry in matrix:
        for mode in ['default', *entry['modes']]:
            jobs = ['greedy']
            if mode == 'default':
                jobs += ['prefill', *[f'context-{n}' for n in entry['context_targets']]]
            for job in jobs:
                key = f"{entry['id']}/{mode}/{job}"
                report = completed(HERE / 'raw' / entry['id'] / 'refresh' / mode, job)
                if report is None:
                    pending.append(key)
                    continue
                assert report['model'] == entry['model'], key
                try:
                    if job == 'greedy':
                        result = audit_greedy(report, mode)
                    elif job == 'prefill':
                        result = audit_prefill(report)
                    else:
                        result = audit_context(report, int(job.split('-')[1]))
                except (AssertionError, KeyError, ZeroDivisionError) as error:
                    raise RuntimeError(f'{key}: {error}') from error
                checked[key] = {'checks': result, 'sources': report['_sources']}
    output = {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'validated_reports_passed': True, 'complete': not pending,
              'validated_report_count': len(checked),
              'scope': 'Validated reports only; reconcile request counts, token usage and timing calculations. '
                       'Pending launches are excluded until memory, rank identity and shutdown gates pass.',
              'checked': checked, 'pending': pending}
    (HERE / 'diagnostics/measurement-audit.json').write_text(json.dumps(output, indent=2) + '\n')
    print(f'{len(checked)} validated reports reconcile; {len(pending)} reports pending.')
    if args.require_complete and pending:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
