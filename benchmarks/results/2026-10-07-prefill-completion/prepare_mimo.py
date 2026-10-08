"""Download the pinned checkpoint and copy it only to initially absent caches."""
import concurrent.futures,datetime,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
MODEL='models--XiaomiMiMo--MiMo-V2.6-Flash-RL'
REV='5711b268169967567844e1e560e8a3966da959b1'
inventory=json.loads((HERE/'inventory-before.json').read_text())
assert all(MODEL not in n['snapshots'] for n in inventory)
assert all(n['free_bytes']>400*2**30 for n in inventory)
from huggingface_hub import snapshot_download
snapshot=snapshot_download('XiaomiMiMo/MiMo-V2.6-Flash-RL',revision=REV,max_workers=4)
source=Path(snapshot).parents[1]
assert source.name==MODEL
print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'Downloaded pinned MiMo checkpoint',flush=True)
def copy(node):
 host=node['host']; destination=f'stephen@{host}:/home/stephen/.cache/huggingface/hub/{MODEL}/'
 with (HERE/'local'/('copy-'+host+'.log')).open('w') as log:
  p=subprocess.run(['rsync','-a','--exclude=*.incomplete',str(source)+'/',destination],stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,(host,p.returncode)
 print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'Copied to',host,flush=True)
 return {'host':host,'returncode':p.returncode}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:copies=list(pool.map(copy,inventory[1:]))
(HERE/'download-complete.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'revision':REV,'snapshot':snapshot,'copies':copies},indent=2)+'\n')
print('Checkpoint ready on all four nodes; downloads and copies finished.',flush=True)
