"""Freeze unaffected pre-fusion evidence and register the validated candidate."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parents[2]
ROOT = HERE.parents[2]
DIAG = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import result_sources
from run import mode_overrides

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    manifest = read(HERE / 'manifest.json')
    baseline = 'ac939277f24752f2b57d889b2a19b2e2fc930cc15732c946b1c4b27e58187342'
    assert manifest['binary_sha256'] == baseline
    assert not (HERE / 'retained-results.json').exists()
    candidate = digest(ROOT / 'build-release/dgpp-serve')
    assert candidate == 'a9b605c0e22cc6eaf869b3c1eef0f4967a354913f71842a500a65103b14294a9'
    catalog = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'source_binary_sha256': baseline, 'modes': {},
               'reason': 'The fused score change is gated to full GLM unpooled prefill beyond 2048 pools. '
                         'Retain other families, full GLM short prompts, and capacity skips. '
                         'Every retained length has three completed repetitions and unchanged configuration. '
                         'Affected full GLM prefill/context requests must be measured with the new binary.'}
    excluded = []
    for entry in read(HERE / 'matrix.json'):
        config = HERE / entry['config']
        for mode in ['default', *entry['modes']]:
            directory = HERE / 'raw' / entry['id'] / 'refresh' / mode
            expected = read(config)
            expected['engine'].update(mode_overrides(mode))
            jobs, files = {}, set()
            candidates = ['greedy']
            if mode == 'default':
                candidates += ['prefill', *[f'context-{n}' for n in entry['context_targets']]]
            for job in candidates:
                report = result_sources.validated(directory, job, baseline)
                if report is None:
                    continue
                assert read(directory / 'config.json') == expected
                allowance = {}
                full = entry['id'].startswith('glm53-')
                if job == 'greedy':
                    maximum = max(r['usage']['prompt_tokens'] for p in report['phases'] for r in p['requests'])
                    assert not full or maximum <= 2048
                    allowance['max_prompt_tokens'] = maximum
                elif job == 'prefill':
                    lengths = []
                    for length in result_sources.PREFILL_LENGTHS:
                        samples = [s for s in report['samples'] if s['requested_tokens'] == length]
                        if sorted(s['repeat'] for s in samples) != [0, 1, 2]:
                            continue
                        maximum = max(s['prompt_tokens'] for s in samples)
                        if full and maximum > 2048:
                            excluded.append({'deployment': entry['id'], 'job': job,
                                             'length': length, 'max_prompt_tokens': maximum})
                        else:
                            lengths.append(length)
                    if not lengths:
                        continue
                    allowance['lengths'] = lengths
                elif full and report['samples']:
                    maximum = max(s['answer']['usage']['prompt_tokens'] for s in report['samples'])
                    if maximum > 2048:
                        excluded.append({'deployment': entry['id'], 'job': job,
                                         'max_prompt_tokens': maximum})
                        continue
                    allowance['max_prompt_tokens'] = maximum
                jobs[job] = allowance
                files.update([f'{job}.json', f'{job}.log.command.json'])
            if not jobs:
                continue
            files.update(['config.json', 'manifest.json', 'memory-check.json',
                          'rank-identity.json', 'down.log.command.json'])
            evidence = read(directory / 'memory-check.json')
            assert evidence['passed'] and all(r['max_swap_KiB'] == 0 for r in evidence['ranks'])
            destination = directory
            if entry['id'] in ['glm53-w4', 'glm53-fp8kv-w4'] and mode == 'default':
                destination = HERE / 'attempts' / entry['id'] / 'default-before-fused-scores-20261007'
                assert not destination.exists()
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(directory), str(destination))
            catalog['modes'][entry['id'] + '/' + mode] = {
                'directory': str(destination.relative_to(HERE)), 'jobs': jobs,
                'target_config': {'path': entry['config'], 'sha256': digest(config)},
                'files_sha256': {name: digest(destination / name) for name in sorted(files)}}
    catalog['excluded_affected_groups'] = excluded
    save(HERE / 'retained-results.json', catalog)
    manifest['binary_sha256'] = candidate
    manifest['version'] = subprocess.check_output([
        'systemd-run', '--user', '--scope', '--quiet', '--property=MemorySwapMax=0', '--',
        str(ROOT / 'build-release/dgpp-serve'), '--version'], text=True).strip()
    manifest['fused_prefill'] = {
        'registered_at': catalog['created_at'], 'source_base_commit': 'a3ebc21a7c245eb736cc799fcf87a2c97ed8f7c9',
        'patch': str((DIAG / 'fused-tail-preserving-candidate.patch').relative_to(HERE)),
        'patch_sha256': digest(DIAG / 'fused-tail-preserving-candidate.patch'),
        'source_manifest': str((DIAG / 'fused-tail-preserving-candidate.json').relative_to(HERE)),
        'previous_binary_sha256': baseline, 'retention_catalog': 'retained-results.json',
        'scope': 'Full GLM only: fused FP8 dots/head reduction, preserving legacy single-query arithmetic. '
                 'No cache, slot, prefix, memory-reserve, decode or other-family algorithm changes.'}
    manifest['build_provenance'] += ' Candidate fusion build is documented under fused_prefill; saved launch manifests remain immutable.'
    for name in manifest['harness_sha256']:
        manifest['harness_sha256'][name] = digest(HERE / name)
    manifest['harness_sha256']['retained-results.json'] = digest(HERE / 'retained-results.json')
    save(HERE / 'manifest.json', manifest)
    print(json.dumps({'retained_modes': len(catalog['modes']), 'excluded': excluded,
                      'binary_sha256': candidate}, indent=2))

if __name__ == '__main__':
    main()
