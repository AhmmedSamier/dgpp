"""Keep reproducible run records without committing runtime logs and captures."""
import hashlib
import json
from pathlib import Path


def load_record(directory):
    return json.loads((Path(directory) / 'record.json').read_text())


def pack_run(directory):
    """Bundle launch metadata and audits; preserve original measurement files."""
    directory = Path(directory)
    names = [
        'config.json', 'manifest.json', 'plan.json', 'complete.json',
        'memory-check.json', 'rank-identity.json', 'models.json',
        'up.log.command.json', 'prefill.log.command.json', 'down.log.command.json',
        'server/cluster.resolved.json',
    ]
    records = {name: json.loads((directory / name).read_text())
               for name in names if (directory / name).exists()}
    for name in ['complete.json', 'memory-check.json', 'rank-identity.json']:
        if not records[name]['passed']:
            raise ValueError(f'{directory}: failed {name}')
    # The audit retains per-rank sample counts, maximum process swap, cgroup
    # policy validation, sampling gaps and errors. Full telemetry stays local.
    telemetry = []
    for path in sorted(directory.glob('node*-telemetry.jsonl')):
        data = path.read_bytes()
        telemetry.append({'file': path.name, 'bytes': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()})
    evidence = {}
    for name in ['prefill.json', 'decode-check.json', 'profile-summary.json']:
        path = directory / name
        if path.exists():
            evidence[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {'schema_version': 1, 'records': records,
              'evidence_sha256': evidence, 'local_telemetry': telemetry}
    (directory / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    return record
