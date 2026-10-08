#!/usr/bin/env python3
"""Remove this session's BF16-dense cache after its complete matrix passes."""
import datetime
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from checkpoint_storage import entry_complete


def main():
    record_path = HERE / 'diagnostics/new-bf16-resident-cache.json'
    record = json.loads(record_path.read_text())
    if record.get('removed_at'):
        print('Generated BF16-dense cache already removed.')
        return
    matrix = json.loads((HERE / 'matrix.json').read_text())
    entry = next(e for e in matrix if e['id'] == record['configuration'])
    assert entry['model'] == record['model']
    assert entry_complete(entry), 'BF16-dense configuration still has outstanding measurements'
    initial = next(n for n in json.loads((HERE / 'resident-inventory-before-new-models.json').read_text())
                   if n['host'] == record['host'])['resident_images']
    root = Path.home() / '.cache/dgpp/resident'
    image = Path(record['path'])
    assert image.parent == root and image.resolve().parent == root.resolve()
    info = image.lstat()
    assert stat.S_ISREG(info.st_mode) and not image.is_symlink()
    assert (info.st_dev, info.st_ino) == (record['device'], record['inode'])
    assert info.st_size == record['logical_bytes']
    paths = [image]
    digest = image.with_suffix('.digest')
    if digest.exists() or digest.is_symlink():
        assert stat.S_ISREG(digest.lstat().st_mode) and not digest.is_symlink()
        paths.append(digest)
    assert all(p.name not in initial for p in paths), 'An original cache file must be preserved'
    identities = {(p.stat().st_dev, p.stat().st_ino) for p in paths}
    # lsof covers open descriptors and mappings. Privileged inspection also
    # covers non-dumpable processes such as sshd; an unreadable process is
    # never silently treated as evidence that the files are unused.
    command = ['sudo', '-n', 'lsof', '-nP', '-t', '--', *map(str, paths)]
    unused = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert unused.returncode == 1 and not unused.stdout and not unused.stderr, (
        'Cache is in use or inspection failed', unused.returncode, unused.stdout, unused.stderr)
    before = shutil.disk_usage(root).free
    removed = []
    for path in paths:
        current = path.lstat()
        assert (current.st_dev, current.st_ino) in identities and stat.S_ISREG(current.st_mode)
        removed.append({'path': str(path), 'allocated_bytes': current.st_blocks * 512})
        path.unlink()
    record.update(removed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  removed=removed, unused_check={'command': command, 'returncode': unused.returncode},
                  free_before=before, free_after=shutil.disk_usage(root).free)
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(f"Removed {sum(p['allocated_bytes'] for p in removed) / 2**30:.1f} GiB of generated cache; "
          f"{record['free_after'] / 2**30:.1f} GiB free.")


if __name__ == '__main__':
    main()
