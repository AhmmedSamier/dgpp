"""Verify temporary checkpoints are gone and pre-test cache entries remain."""
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

PROBE = '''import json,pathlib,shutil,sys
spec=json.load(sys.stdin)
root=pathlib.Path(spec['root']).expanduser().resolve()
errors=[]
checked=0
for name,model in spec['before']['models'].items():
 for relative in model['files']:
  if not (root/name/relative).exists(): errors.append('missing: '+str(root/name/relative))
  checked+=1
 for relative,value in model['refs'].items():
  path=root/name/relative
  if not path.exists() or path.read_text().strip()!=value.strip():
   errors.append('changed ref: '+str(path))
for model in spec['temporary']:
 name='models--'+model.replace('/','--')
 for parent in (root,root/'.locks'):
  if (parent/name).exists(): errors.append('temporary checkpoint remains: '+str(parent/name))
resident=spec['resident']
images=pathlib.Path(resident['path'])
for name,info in resident['files'].items():
 path=images/name
 if not path.is_file() or path.stat().st_size!=info['size']:
  errors.append('changed resident entry: '+str(path))
new_images=sorted(p.name for p in images.iterdir() if p.name not in resident['files'])
print(json.dumps(dict(errors=errors,original_cache_files_checked=checked,
 original_resident_entries_checked=len(resident['files']),
 additional_resident_entries=new_images,free_bytes=shutil.disk_usage(root).free)))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--deployment', type=Path, required=True)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--temporary-model', action='append', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    config = resolve_config(args.deployment, settings())

    def inspect(item):
        rank, host = item
        spec = dict(root=str(cache_root(config['node_env'][rank])),
                    before=json.loads((args.inventory / f'cache-before-{host}.json').read_text()),
                    resident=json.loads((args.inventory / f'resident-before-{host}.json').read_text()),
                    temporary=args.temporary_model)
        command = ['python3', '-c', PROBE]
        if rank:
            command = ['ssh', '-o', 'BatchMode=yes', config['ssh_user'] + '@' + host, shlex.join(command)]
        result = subprocess.run(command, input=json.dumps(spec), text=True,
                                capture_output=True, check=True, timeout=120)
        return dict(rank=rank, host=host, **json.loads(result.stdout))

    with ThreadPoolExecutor(max_workers=len(config['nodes'])) as pool:
        nodes = list(pool.map(inspect, enumerate(config['nodes'])))
    args.out.write_text(json.dumps(nodes, indent=2) + '\n')
    print(json.dumps(nodes, indent=2))
    assert not any(node['errors'] for node in nodes), 'cache preservation or cleanup failed'


if __name__ == '__main__':
    main()
