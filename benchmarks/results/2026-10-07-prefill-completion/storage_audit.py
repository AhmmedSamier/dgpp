"""Preserve the initial cache inventory; remove only this investigation's model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
MODEL = 'models--XiaomiMiMo--MiMo-V2.6-Flash-RL'
PROBE = r'''
from pathlib import Path
import json,os,shutil
home=Path.home(); hub=home/'.cache/huggingface/hub'; resident=home/'.cache/dgpp/resident'
procs=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid==os.getuid() and (p/'comm').read_text().strip().startswith('dgpp-serve'): procs.append(int(p.name))
 except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps({'snapshots':{p.name:sorted(x.name for x in (p/'snapshots').iterdir()) for p in hub.glob('models--*') if (p/'snapshots').is_dir()},'resident':{p.name:p.stat().st_size for p in resident.iterdir() if p.is_file()},'free_bytes':shutil.disk_usage(home).free,'serving_processes':procs,'temporary_resident_exists':(home/'.cache/dgpp/prefill-completion-2026-10-07').exists()}))
'''

def remote(host, source):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','stephen@'+host,'python3 -'],input=source,text=True,capture_output=True,check=True,timeout=120)
    return json.loads(result.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cleanup', action='store_true')
    args=parser.parse_args()
    before=json.loads((HERE/'inventory-before.json').read_text())
    plan=json.loads((HERE/'storage-plan.json').read_text())
    nodes=[dict(host=n['host'],**remote(n['host'],PROBE)) for n in before]
    assert all(not n['serving_processes'] for n in nodes),nodes
    if args.cleanup:
        assert plan['absent_before'] and plan['cache_parent_absent_before']
        assert all(MODEL not in n['snapshots'] for n in before)
        removed=[]
        for host in plan['target_hosts']:
            script="""from pathlib import Path
import shutil,json
home=Path.home(); removed=[]
for path in [home/'.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-RL',home/'.cache/dgpp/prefill-completion-2026-10-07']:
 if path.exists():
  assert not path.is_symlink()
  shutil.rmtree(path); removed.append(str(path))
print(json.dumps(removed))
"""
            removed.append({'host':host,'paths':remote(host,script)})
        (HERE/'cleanup.json').write_text(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'removed':removed},indent=2)+'\n')
        nodes=[dict(host=n['host'],**remote(n['host'],PROBE)) for n in before]
    errors=[]
    for old,new in zip(before,nodes):
        for model,revisions in old['snapshots'].items():
            if not set(revisions)<=set(new['snapshots'].get(model,[])):errors.append([old['host'],'missing snapshot',model])
        for name,size in old['resident'].items():
            if new['resident'].get(name)!=size:errors.append([old['host'],'resident changed/missing',name])
        if args.cleanup and (MODEL in new['snapshots'] or new['temporary_resident_exists']):errors.append([old['host'],'new model remains'])
    result={'at':datetime.now(timezone.utc).isoformat(),'passed':not errors,'cleanup':args.cleanup,'nodes':nodes,'errors':errors,'scope':'Snapshot presence and pre-existing resident filenames/sizes; not a full content hash audit.'}
    (HERE/('storage-preservation-audit.json' if args.cleanup else 'storage-current.json')).write_text(json.dumps(result,indent=2)+'\n')
    assert not errors,errors
    print(json.dumps({'passed':True,'free_gib':{n['host']:round(n['free_bytes']/2**30,1) for n in nodes}}))

if __name__=='__main__':main()
