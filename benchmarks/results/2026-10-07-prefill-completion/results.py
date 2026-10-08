"""Validate compact launch records and the completed request measurements."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SCALING = HERE.parent / '2026-10-06-prefill-scaling'
sys.path.insert(0, str(SCALING))
spec = importlib.util.spec_from_file_location('original_measurement_audit',
                                            SCALING / 'diagnostics/audit_measurements.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def read(path):
    return json.loads(path.read_text())


def validated(ident, mode, job):
    directory = HERE / 'raw' / ident / 'refresh' / mode
    if not (directory / 'record.json').exists():
        return None
    record = read(directory / 'record.json')
    data = record['records']
    expected = read(HERE / 'manifest.json')['candidate_binary_sha256']
    assert data['manifest.json']['binary_sha256'] == expected
    for name in ['complete.json', 'memory-check.json', 'rank-identity.json']:
        assert data[name]['passed'], (directory, name)
    assert data['down.log.command.json']['returncode'] == 0
    world = data['config.json']['world_size']
    ranks = data['memory-check.json']['ranks']
    assert len(ranks) == world and {r['rank'] for r in ranks} == set(range(world))
    for rank in ranks:
        assert (rank['samples'] > 0 and rank['max_swap_KiB'] == 0 and
                rank['swap_policy_required'] and rank['swap_policy_passed'] and
                not rank['violations'] and not rank['errors'] and
                rank['max_sample_gap_s'] <= 10), (directory, rank)
    identity = data['rank-identity.json']
    assert identity['expected_ranks'] == len(identity['md5']) == world
    assert len(set(identity['md5'].values())) == 1
    assert job in data['measurement-plan.json']['jobs']
    assert not data['measurement-plan.json']['profiled']
    assert data[job + '.log.command.json']['returncode'] == 0
    path = directory / (job + '.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['evidence_sha256'][path.name]
    report = read(path)
    assert report['model'] == data['config.json']['model']
    if job == 'greedy':
        audit.audit_greedy(report, mode)
    elif job.startswith('context-'):
        audit.audit_context(report, int(job.split('-')[1]))
    else:
        samples = report['samples']
        lengths = [32768] if ident == 'glm-flash-hybrid-256k-w2' else [2048, 8192, 32768]
        assert Counter((s['requested_tokens'], s['repeat']) for s in samples) == Counter(
            {(n, r): 1 for n in lengths for r in range(3)})
        assert len({s['prompt_sha256'] for s in samples}) == len(samples)
        for sample in samples:
            usage = sample['usage']
            assert usage['prompt_tokens'] == sample['prompt_tokens'] == sample['computed_tokens']
            assert usage['prompt_tokens_details']['cached_tokens'] == sample['cached_tokens'] == 0
            assert usage['completion_tokens'] == 1 and sample['prefill_ms'] > 0
            audit.close(sample['prefill_ms_per_token'], sample['prefill_ms'] / sample['computed_tokens'])
    return report


def audit_all():
    checked, pending = [], []
    for launch in read(HERE / 'plan.json'):
        for job in launch['jobs']:
            key = f"{launch['deployment']}/{launch['mode']}/{job}"
            report = validated(launch['deployment'], launch['mode'], job)
            (checked if report else pending).append(key)
    return {'passed': True, 'complete': not pending, 'validated_reports': len(checked),
            'checked': checked, 'pending': pending}


if __name__ == '__main__':
    result = audit_all()
    (HERE / 'measurement-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"{len(result['checked'])} validated reports; {len(result['pending'])} pending")
    if '--require-complete' in sys.argv and result['pending']:
        raise SystemExit(1)
