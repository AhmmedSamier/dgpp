"""Audit completed admission records and compare matched trial means."""
import argparse
import json
from pathlib import Path
import statistics

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('directory', type=Path)
ap.add_argument('--out', type=Path)
args = ap.parse_args()
reports = {}
requests = scenarios = 0
for path in sorted(args.directory.glob('w*-*/admission.json')):
    report = json.loads(path.read_text())
    records = {}
    for result in report['runs']:
        before, after = result['before'], result['after']
        n = len(result['requests'])
        assert not after['service']['engine_failed'], path
        assert after['scheduler']['active'] == after['scheduler']['queued'] == after['scheduler']['records'] == 0, path
        assert after['scheduler']['terminal'] - before['scheduler']['terminal'] == n, path
        assert after['service']['requests_total'] - before['service']['requests_total'] == n, path
        assert after['service']['requests_failed'] == before['service']['requests_failed'], path
        assert after['scheduler']['pool_blocks_in_use'] == before['scheduler']['pool_blocks_in_use'] == 0, path
        if result['scenario'] != 'cancel':
            assert result['generated_tokens'] == result['completed_response_tokens'], path
        records[(result['scenario'], result['trial'])] = result
        requests += n
        scenarios += 1
    reports[path.parent.name] = records
comparisons = []
mismatches = []
paired_requests = 0
for name, candidate in reports.items():
    if not name.endswith('-candidate'):
        continue
    for suffix in ('control', 'candidate-deferred'):
        other = reports.get(name.removesuffix('candidate') + suffix, {})
        common = sorted(set(candidate) & set(other))
        for scenario in sorted({k[0] for k in common}):
            selected = [k for k in common if k[0] == scenario]
            c = [candidate[k] for k in selected]
            b = [other[k] for k in selected]
            cm, bm = statistics.mean(r['wall_s'] for r in c), statistics.mean(r['wall_s'] for r in b)
            row = {'configuration': name.removesuffix('-candidate'), 'scenario': scenario,
                   'reference': suffix, 'trials': len(selected), 'candidate_wall_s': cm,
                   'reference_wall_s': bm, 'throughput_change_pct': None if scenario == 'cancel' else 100 * (bm / cm - 1)}
            comparisons.append(row)
            print(row['configuration'], scenario, suffix, f'{cm:.3f} / {bm:.3f} s,', row['throughput_change_pct'])
            if scenario == 'cancel':
                continue
            for key in selected:
                assert len(candidate[key]['requests']) == len(other[key]['requests']), (name, key)
                assert candidate[key]['computed_prompt_tokens'] == other[key]['computed_prompt_tokens'], (name, key)
                assert candidate[key]['completed_response_tokens'] == other[key]['completed_response_tokens'], (name, key)
                for cr, br in zip(candidate[key]['requests'], other[key]['requests']):
                    assert cr['name'] == br['name'] and cr['body_sha256'] == br['body_sha256'], (name, key)
                    assert cr['usage'] == br['usage'], (name, key, cr['name'])
                    paired_requests += 1
                    if cr['output_sha256'] != br['output_sha256']:
                        mismatches.append({'configuration': name, 'reference': suffix,
                            'scenario': scenario, 'trial': key[1], 'request': cr['name'],
                            'candidate': cr['output_sha256'], 'reference_sha256': br['output_sha256']})
result = {'completed_scenarios': scenarios, 'completed_requests': requests,
          'paired_requests': paired_requests, 'output_mismatches': mismatches, 'comparisons': comparisons}
if args.out:
    args.out.write_text(json.dumps(result, indent=2) + '\n')
print('AUDIT', scenarios, 'scenarios,', requests, 'requests,', paired_requests, 'matched request pairs,', len(mismatches), 'output differences')
if mismatches:
    raise SystemExit('Paired output differences require inspection; see the audit record.')
