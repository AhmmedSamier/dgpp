"""Register the validated CSA2 candidate and retain only unaffected families."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

DIAG = Path(__file__).resolve().parent
HERE = DIAG.parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import result_sources
from run import mode_overrides

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    manifest = read(HERE / 'manifest.json')
    baseline = 'a983cebe9fc1c6fa4623a45cf7ba76eaf4fb7cdb5270179c0e48acba9b482883'
    assert manifest['binary_sha256'] == baseline
    assert not (HERE / 'capacity-retained-results.json').exists()
    candidate = read(DIAG / 'candidate.json')
    assert sha(ROOT / 'build-release/dgpp-serve') == candidate['binary_sha256']
    for name, expected in candidate['source_sha256'].items():
        assert sha(ROOT / name) == expected
    affected = {'deepseek-w4', 'deepseek-1m-w4'}
    for ident in affected:
        comparison = read(DIAG / f'{ident}-comparison.json')
        assert comparison['complete'] and comparison['all_prompts_and_outputs_equal']
        assert len(comparison['samples']) == (15 if ident == 'deepseek-1m-w4' else 18)
        directory = HERE / comparison['candidate_directory']
        for name, key, value in [('memory-check.json', 'passed', True),
                                  ('rank-identity.json', 'passed', True),
                                  ('down.log.command.json', 'returncode', 0)]:
            assert read(directory / name)[key] == value
    for name in ['reused-results.json', 'retained-results.json']:
        assert not any(k.split('/')[0] in affected for k in read(HERE / name)['modes'])
    catalog = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_binary_sha256': baseline, 'modes': {},
        'reason': 'The change is confined to Csa2Layer prefill, instantiated only by DeepSeek-V4.1. '
                  'All earlier DeepSeek-V4.1 serving, prefill, context and mode measurements are rerun. '
                  'Other families retain their validated measurements and unchanged configurations.'}
    for entry in read(HERE / 'matrix.json'):
        if entry['id'] in affected:
            continue
        for mode in ['default', *entry['modes']]:
            directory = HERE / 'raw' / entry['id'] / 'refresh' / mode
            expected = read(HERE / entry['config'])
            expected['engine'].update(mode_overrides(mode))
            jobs, files = {}, set()
            names = ['greedy']
            if mode == 'default':
                names += ['prefill', *[f'context-{n}' for n in entry['context_targets']]]
            for job in names:
                report = result_sources.validated(directory, job, baseline)
                if report is None:
                    continue
                assert read(directory / 'config.json') == expected
                allowance = {}
                if job == 'prefill':
                    allowance['lengths'] = [n for n in result_sources.PREFILL_LENGTHS
                        if sorted(s['repeat'] for s in report['samples'] if s['requested_tokens'] == n) == [0, 1, 2]]
                    if not allowance['lengths']:
                        continue
                jobs[job] = allowance
                files.update([f'{job}.json', f'{job}.log.command.json'])
            if jobs:
                files.update(['config.json', 'manifest.json', 'memory-check.json',
                              'rank-identity.json', 'down.log.command.json'])
                catalog['modes'][entry['id'] + '/' + mode] = {
                    'directory': str(directory.relative_to(HERE)), 'jobs': jobs,
                    'target_config': {'path': entry['config'], 'sha256': sha(HERE / entry['config'])},
                    'files_sha256': {name: sha(directory / name) for name in sorted(files)}}
    old = HERE / 'raw/deepseek-w4'
    archive = HERE / 'attempts/deepseek-w4/before-capacity-fix-20261007'
    assert old.exists() and not archive.exists()
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(archive))
    save(HERE / 'capacity-retained-results.json', catalog)
    manifest['binary_sha256'] = candidate['binary_sha256']
    manifest['version'] = read(DIAG / 'comparison/manifest.json')['version']
    manifest['capacity_prefill'] = {
        'registered_at': catalog['created_at'], 'source_base_commit': candidate['source_base_commit'],
        'candidate': str((DIAG / 'candidate.json').relative_to(HERE)),
        'candidate_sha256': sha(DIAG / 'candidate.json'), 'previous_binary_sha256': baseline,
        'retention_catalog': 'capacity-retained-results.json', 'affected_deployments': sorted(affected),
        'validation': 'CSA2 kernel/layer oracle, reference forward and bounded prefill, decode, TP/engine, '
                      'CUDA memcheck; 33 identical real-model prompt/output/usage comparisons across two capacities.'}
    for name in manifest['harness_sha256']:
        manifest['harness_sha256'][name] = sha(HERE / name)
    manifest['harness_sha256']['capacity-retained-results.json'] = sha(HERE / 'capacity-retained-results.json')
    save(HERE / 'manifest.json', manifest)
    print(json.dumps({'retained_modes': len(catalog['modes']), 'affected': sorted(affected),
                      'binary_sha256': candidate['binary_sha256']}, indent=2))

if __name__ == '__main__':
    main()
