"""Sequential targeted reruns after the three prefill efficiency fixes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCALING = HERE.parent / '2026-10-06-prefill-scaling'
sys.path.insert(0, str(SCALING))
spec = importlib.util.spec_from_file_location('scaling_runner', SCALING / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.HERE = HERE
campaign = runner.campaign
campaign.HERE = HERE
MANIFEST = json.loads((HERE / 'manifest.json').read_text())
BINARY = Path(MANIFEST['candidate_binary'])
original_run = campaign.run


def candidate_run(command, output, **kwargs):
    if output.name == 'up.log':
        command = list(command)
        command[command.index('--bin') + 1] = str(BINARY)
        manifest = json.loads((output.parent / 'manifest.json').read_text())
        manifest['baseline_binary_sha256'] = manifest['binary_sha256']
        manifest['binary_sha256'] = MANIFEST['candidate_binary_sha256']
        campaign.save(output.parent / 'manifest.json', manifest)
    return original_run(command, output, **kwargs)


campaign.run = candidate_run


def jobs_for(entry, mode):
    if mode != 'default':
        return ['greedy']
    cfg = json.loads((HERE / entry['config']).read_text())
    lengths = [n for n in entry['context_targets'] if n <= cfg['engine']['kv_capacity']]
    prefill = entry['id'] in ['glm-flash-hybrid-256k-w2', 'mimo-w4', 'mimo-1m-w4']
    return ['greedy'] + (['prefill'] if prefill else []) + [f'context-{n}' for n in lengths]


def bundle(directory, jobs):
    names = ['config.json', 'manifest.json', 'measurement-plan.json', 'complete.json',
             'memory-check.json', 'rank-identity.json', 'models.json',
             'server/cluster.resolved.json', 'up.log.command.json', 'down.log.command.json']
    names += [job + '.log.command.json' for job in jobs]
    records = {name: json.loads((directory / name).read_text()) for name in names}
    hashes = {job + '.json': hashlib.sha256((directory / (job + '.json')).read_bytes()).hexdigest()
              for job in jobs}
    telemetry = [{'file': p.name, 'bytes': p.stat().st_size,
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(directory.glob('node*-telemetry.jsonl'))]
    campaign.save(directory / 'record.json', {'schema_version': 1, 'records': records,
                  'evidence_sha256': hashes, 'local_telemetry': telemetry})


def complete(directory, jobs):
    path = directory / 'record.json'
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    records = record['records']
    if records['manifest.json']['binary_sha256'] != MANIFEST['candidate_binary_sha256']:
        raise RuntimeError(f'Unexpected binary in {directory}')
    for name in ['complete.json', 'memory-check.json', 'rank-identity.json']:
        if not records[name]['passed']:
            raise RuntimeError(f'Failed validation in {directory}: {name}')
    for job in jobs:
        path = directory / (job + '.json')
        if (records[job + '.log.command.json']['returncode'] != 0 or
                hashlib.sha256(path.read_bytes()).hexdigest() != record['evidence_sha256'][path.name]):
            raise RuntimeError(f'Invalid measurement in {directory}: {job}')
    return True


def measure(entry, mode):
    directory = HERE / 'raw' / entry['id'] / 'refresh' / mode
    jobs = jobs_for(entry, mode)
    if complete(directory, jobs):
        print('SKIP', entry['id'], mode, flush=True)
        return
    if directory.exists():
        # Every retry receives its own launch and memory evidence.
        archive = HERE / 'local/attempts' / entry['id'] / (mode + '-' + campaign.now())
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(directory), archive)
    cfg = server = None
    storage = []
    resolved = campaign.resolve_config(str(HERE / entry['config']))
    for host in resolved['nodes']:
        probe = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            f"{resolved['ssh_user']}@{host}", 'python3', '-c',
            "'import shutil; print(shutil.disk_usage(\"/home/stephen\").free)'"],
            capture_output=True, text=True, check=True, timeout=30)
        free = int(probe.stdout)
        if free < 100 * 2**30:
            raise RuntimeError(f'{host}: less than 100 GiB free before launch')
        storage.append({'host': host, 'free_bytes': free})
    try:
        directory, cfg, server = campaign.launch(entry, 'refresh', mode, runner.mode_overrides(mode))
        campaign.save(directory / 'measurement-plan.json', {'jobs': jobs, 'mode': mode,
                      'deployment': entry['id'], 'profiled': False, 'storage_before': storage})
        def check_monitors(job):
            processes = campaign.TELEMETRY[str(directory)][:entry['world']]
            stopped = [rank for rank, (p, _) in enumerate(processes) if p.poll() is not None]
            if stopped:
                campaign.save(directory / 'memory-monitor-failure.json',
                              {'at': campaign.now(), 'job': job, 'ranks': stopped})
                raise runner.MemoryViolation(f'Memory monitor stopped: {stopped}')
        for job in jobs:
            check_monitors(job)
            campaign.save(HERE / 'progress.json', {'at': campaign.now(), 'deployment': entry['id'],
                          'mode': mode, 'job': job, 'status': 'running'})
            if job == 'greedy':
                command = campaign.client('timed_load.py', '--concurrency',
                    '1,2,4,8' if mode == 'default' else '1', '--classes', 'all',
                    '--repeat', 3, '--max-tokens', 256, '--json-out', directory / 'greedy.json')
                if 'GLM-5.3-Int4' in entry['model']:
                    command.append('--think')
            elif job == 'prefill':
                lengths = [32768] if entry['id'] == 'glm-flash-hybrid-256k-w2' else [2048, 8192, 32768]
                command = campaign.client('serve_prefill_probe.py', *lengths, '--repeat', 3,
                    *campaign.thinking_args(entry), '--seed', 7, '--tag', 'completion-' + entry['id'],
                    '--json-out', directory / 'prefill.json')
            else:
                command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/serve_context_matrix.py'),
                    '127.0.0.1', '18080', '--model', entry['model'], '--tag', 'completion-' + entry['id'],
                    '--lengths', job.split('-')[1], '--json-out', str(directory / (job + '.json'))]
            if not campaign.run(command, directory / (job + '.log'), timeout=14400, resume=False):
                raise RuntimeError(f'{entry["id"]}/{mode}/{job} failed')
            check_monitors(job)
    finally:
        if cfg is not None:
            try:
                check_monitors('shutdown')
            finally:
                campaign.stop(directory, cfg, server)
                runner.check_memory(directory, cfg)
                if not json.loads((directory / 'rank-identity.json').read_text())['passed']:
                    raise RuntimeError(f'Rank operation streams differ: {directory}')
    campaign.save(directory / 'complete.json', {'at': campaign.now(), 'passed': True})
    bundle(directory, jobs)
    subprocess.run([sys.executable, str(HERE / 'results.py')], check=True)
    subprocess.run([sys.executable, str(HERE / 'update_document.py')], check=True)
    print(campaign.now(), 'VALIDATED', entry['id'], mode, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    entries = json.loads((HERE / 'matrix.json').read_text())
    plan = [(e, mode) for e in entries if not args.only or e['id'] in args.only
            for mode in (['default', 'plain', 'mtp2', 'mtp3']
                         if e['id'] == 'glm53-fp4kv-256k-w4' else ['default'])]
    if args.plan:
        print(json.dumps([{'deployment': e['id'], 'mode': m, 'jobs': jobs_for(e, m)}
                          for e, m in plan], indent=2))
        return
    assert json.loads((HERE / 'download-complete.json').read_text())['passed']
    assert hashlib.sha256(BINARY.read_bytes()).hexdigest() == MANIFEST['candidate_binary_sha256']
    for source, expected in MANIFEST['source_sha256'].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected, source
    for entry, mode in plan:
        measure(entry, mode)
    campaign.save(HERE / 'measurements-complete.json', {'at': campaign.now(), 'passed': True,
                  'launches': len(plan), 'jobs': sum(len(jobs_for(e, m)) for e, m in plan)})


if __name__ == '__main__':
    main()
