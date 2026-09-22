#!/usr/bin/env python3
"""Finish remaining measurements after the current finite runner exits."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        return text[text.rfind(')') + 2:].split()[19]
    except FileNotFoundError:
        return None


def run(command, log):
    receipt = {'command': command, 'started_at': now(), 'cwd': str(ROOT)}
    path = HERE / 'raw' / log
    print(now(), 'RUN', command, flush=True)
    with path.open('w') as output:
        rc = subprocess.call(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    receipt.update(returncode=rc, finished_at=now())
    path.with_suffix(path.suffix + '.command.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if rc:
        raise RuntimeError(f'{log}: exit {rc}; inspect saved logs before resuming')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    original = identity(args.wait_pid)
    print(now(), 'Waiting for existing runner', args.wait_pid, original, flush=True)
    while original is not None and identity(args.wait_pid) == original:
        time.sleep(10)
    run([sys.executable, '-u', str(HERE / 'run.py'), 'performance'], 'remaining-four-node-performance.log')
    run([sys.executable, '-u', str(HERE / 'run.py'), 'all'], 'remaining-all-models.log')
    manifest = HERE / 'manifest.json'
    data = json.loads(manifest.read_text())
    data['completed_at'] = now()
    manifest.write_text(json.dumps(data, indent=2) + '\n')
    try:
        run([sys.executable, str(HERE / 'summarize.py')], 'final-summary.log')
    except Exception:
        data.pop('completed_at', None)
        manifest.write_text(json.dumps(data, indent=2) + '\n')
        raise
    run([sys.executable, str(ROOT / 'scripts/dgpp-cluster'), 'up',
         '--config', str(ROOT / 'deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json'),
         '--bin', str(ROOT / 'build-release/dgpp-serve')], 'restore-original-service.log')
    (HERE / 'raw/finished.json').write_text(json.dumps({'finished_at': now(),
        'measurements_and_document_generation': 'complete', 'original_service': 'restored'}, indent=2) + '\n')
    print(now(), 'Measurements complete; documents generated; original service restored.', flush=True)


if __name__ == '__main__':
    main()
