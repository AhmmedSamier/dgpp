"""Resolve fresh measurements and explicitly audited, unaffected results."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFILL_LENGTHS = (2048, 8192, 32768)


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def validated(directory, job, binary):
    checks = [(f'{job}.log.command.json', 'returncode', 0),
              ('memory-check.json', 'passed', True),
              ('rank-identity.json', 'passed', True),
              ('down.log.command.json', 'returncode', 0)]
    if any((read(directory / name) or {}).get(key) != value for name, key, value in checks):
        return None
    manifest = read(directory / 'manifest.json') or {}
    if manifest.get('binary_sha256') != binary:
        return None
    report = read(directory / f'{job}.json')
    if report is None:
        return None
    return report


def primary(directory, job):
    manifest = read(HERE / 'manifest.json')
    report = validated(directory, job, manifest['binary_sha256'])
    if report is not None:
        report['_sources'] = [{
            'path': str((directory / f'{job}.json').relative_to(HERE)),
            'binary_sha256': manifest['binary_sha256'], 'reused': False,
        }]
    return report


def reuse_entry(key):
    catalog = read(HERE / 'reused-results.json') or {}
    entry = catalog.get('modes', {}).get(key)
    if entry is None:
        return None
    target = entry['target_config']
    path = HERE / target['path']
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != target['sha256']:
        raise RuntimeError(f'reuse target configuration changed: {path}')
    directory = HERE / entry['directory']
    for name, expected in entry['files_sha256'].items():
        path = directory / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'reused evidence changed: {path}')
    return entry, catalog['source_binary_sha256']


def reused(directory, job):
    key = directory.parent.parent.name + '/' + directory.name
    found = reuse_entry(key)
    if found is None:
        return None
    entry, binary = found
    allowance = entry['jobs'].get(job)
    if allowance is None:
        return None
    report = validated(HERE / entry['directory'], job, binary)
    if report is None:
        raise RuntimeError(f'reused result no longer passes validation: {key}/{job}')
    if job == 'prefill':
        report['samples'] = [s for s in report['samples']
                             if s['requested_tokens'] in allowance['lengths']]
    report['_sources'] = [{
        'path': entry['directory'] + f'/{job}.json', 'binary_sha256': binary,
        'reused': True, **({'lengths': allowance['lengths']} if job == 'prefill' else {}),
    }]
    return report


def prefill_parts(directory):
    old, fresh = reused(directory, 'prefill'), primary(directory, 'prefill')
    samples, owners = {}, {}
    for report in (old, fresh):
        if report is None:
            continue
        lengths = set(s['requested_tokens'] for s in report['samples'])
        for length in lengths:
            group = [s for s in report['samples'] if s['requested_tokens'] == length]
            if length in PREFILL_LENGTHS and sorted(s['repeat'] for s in group) == [0, 1, 2]:
                samples[length] = group
                owners[length] = report['_sources'][0]
    # Fresh measurements win at each length. Attribute only the samples
    # actually retained, including when a full new sweep supersedes reuse.
    sources = []
    for report in (old, fresh):
        if report is not None:
            source = report['_sources'][0]
            lengths = sorted(n for n, owner in owners.items() if owner is source)
            if lengths:
                sources.append({**source, 'lengths': lengths})
    return old, fresh, samples, sources


def prefill_missing_lengths(directory):
    _, _, samples, _ = prefill_parts(directory)
    return [n for n in PREFILL_LENGTHS if n not in samples]


def prefill_report(directory, require_complete=True):
    old, fresh, samples, sources = prefill_parts(directory)
    if not samples or (require_complete and any(n not in samples for n in PREFILL_LENGTHS)):
        return None
    report = dict(fresh or old)
    report['samples'] = [s for n in PREFILL_LENGTHS for s in samples.get(n, [])]
    report['_sources'] = sources
    return report


def completed(directory, job):
    if job == 'prefill':
        return prefill_report(directory)
    return primary(directory, job) or reused(directory, job)
