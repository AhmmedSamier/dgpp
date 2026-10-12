"""Remove a test checkpoint only when every node's saved inventory proves it was absent."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from cluster_doctor import cache_root
from site_env import resolve_config, settings

REMOVE = '''import json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]).expanduser().resolve()
name=sys.argv[2]
assert name.startswith('models--') and '/' not in name and '..' not in name
removed=[]
for parent in (root,root/'.locks'):
 p=parent/name
 if p.exists():
  assert not p.is_symlink() and p.resolve().parent==parent
  shutil.rmtree(p)
  removed.append(str(p))
print(json.dumps(dict(removed=removed,free_bytes=shutil.disk_usage(root).free)))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--deployment', type=Path, required=True)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    config = resolve_config(args.deployment, settings())
    name = 'models--' + config['model'].replace('/', '--')
    # Validate all inventories before deleting anything on any node.
    for host in config['nodes']:
        before = json.loads((args.inventory / f'cache-before-{host}.json').read_text())
        assert name not in before['models'], f'{host}: checkpoint predates this test; preserve it'

    def remove(item):
        rank, host = item
        root = str(cache_root(config['node_env'][rank]))
        command = ['python3', '-', root, name]
        if rank:
            command = ['ssh', '-o', 'BatchMode=yes', config['ssh_user'] + '@' + host, shlex.join(command)]
        result = subprocess.run(command, input=REMOVE, text=True, capture_output=True,
                                timeout=120, check=True)
        return dict(rank=rank, host=host, model=config['model'], **json.loads(result.stdout))

    with ThreadPoolExecutor(max_workers=len(config['nodes'])) as pool:
        result = list(pool.map(remove, enumerate(config['nodes'])))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
