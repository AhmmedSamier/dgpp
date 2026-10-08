#!/usr/bin/env python3
"""Inspect or remove new resident images after every case for a model passes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from checkpoint_storage import entry_complete

REMOTE = r'''
from datetime import datetime, timezone
import json, pathlib, re, shutil, stat, subprocess, sys
request = json.loads(sys.argv[1])
root = pathlib.Path.home() / '.cache/dgpp/resident'
record = {'free_before': shutil.disk_usage(root).free, 'images': [], 'removed': []}
paths = []
for name in request['images']:
    assert re.fullmatch(r'[0-9a-f]{16}\.img', name), name
    for candidate in [name, name.removesuffix('.img') + '.digest']:
        assert candidate not in request['initial'], 'Original cache must be preserved'
        path = root / candidate
        if not path.exists() and not path.is_symlink():
            continue
        info = path.lstat()
        assert stat.S_ISREG(info.st_mode) and not path.is_symlink(), path
        assert path.resolve().parent == root.resolve(), path
        item = {'path': str(path), 'device': info.st_dev, 'inode': info.st_ino,
                'logical_bytes': info.st_size, 'allocated_bytes': info.st_blocks * 512}
        record['images'].append(item)
        paths.append((path, item))
if request['remove'] and paths:
    command = ['sudo', '-n', 'lsof', '-nP', '-t', '--', *[str(p) for p, _ in paths]]
    unused = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert unused.returncode == 1 and not unused.stdout and not unused.stderr, (
        'Cache in use or inspection failed', unused.returncode, unused.stdout, unused.stderr)
    record['unused_check'] = {'command': command, 'returncode': unused.returncode}
    for path, item in paths:
        current = path.lstat()
        assert stat.S_ISREG(current.st_mode)
        assert (current.st_dev, current.st_ino, current.st_size) == (
            item['device'], item['inode'], item['logical_bytes']), path
        path.unlink()
        record['removed'].append(item)
record['free_after'] = shutil.disk_usage(root).free
record['recorded_at'] = datetime.now(timezone.utc).isoformat()
print(json.dumps(record))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--remove', action='store_true')
    args = parser.parse_args()
    matrix = json.loads((HERE / 'matrix.json').read_text())
    entries = [entry for entry in matrix if entry['model'] == args.model]
    if not entries:
        raise ValueError('Model is not in the benchmark matrix')
    complete = all(entry_complete(entry) for entry in entries)
    if args.remove and not complete:
        raise RuntimeError('The model still has outstanding measurements')
    initial = json.loads((HERE / 'resident-inventory-before-new-models.json').read_text())
    sources, images = [], {node['host']: set() for node in initial}
    for entry in entries:
        for mode in ['default', *entry['modes']]:
            directory = HERE / 'raw' / entry['id'] / 'refresh' / mode
            for rank in range(entry['world']):
                log = directory / 'server' / f'serve_r{rank}.log'
                if not log.exists():
                    continue
                source = log.read_bytes()
                names = re.findall(
                    rf'rank {rank} resident image /home/stephen/\.cache/dgpp/resident/([0-9a-f]{{16}}\.img)',
                    source.decode())
                if names:
                    host = f'192.168.88.{11 + rank}'
                    images[host].update(names)
                    sources.append({'path': str(log.relative_to(HERE)),
                                    'sha256': hashlib.sha256(source).hexdigest(),
                                    'host': host, 'images': sorted(set(names))})

    def inspect(node):
        names = sorted(images[node['host']] - set(node['resident_images']))
        request = {'images': names, 'initial': list(node['resident_images']), 'remove': args.remove}
        result = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'stephen@' + node['host'],
             'python3 - ' + shlex.quote(json.dumps(request))],
            input=REMOTE, text=True, capture_output=True, timeout=60, check=True)
        return {'host': node['host'], **json.loads(result.stdout)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        nodes = list(pool.map(inspect, initial))
    record = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'model': args.model,
              'all_configurations_complete': complete, 'remove_requested': args.remove,
              'configurations': [entry['id'] for entry in entries], 'sources': sources, 'nodes': nodes}
    with (HERE / 'diagnostics/generated-resident-cleanup.jsonl').open('a') as output:
        output.write(json.dumps(record) + '\n')
    for node in nodes:
        size = sum(item['allocated_bytes'] for item in node['images']) / 2**30
        print(f"{node['host']}: {size:.1f} GiB new cache; removed {len(node['removed'])} files; "
              f"{node['free_after'] / 2**30:.1f} GiB free")


if __name__ == '__main__':
    main()
