from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,re,shutil,sys,time,urllib.request,subprocess
ROOT=Path('/home/stephen/workspace/dgpp'); R=ROOT/'benchmarks/results/2026-10-07-prefill-completion'
sys.path.insert(0,str(R.parent/'2026-10-06-prefill-scaling'));import run as runner
runner.HERE=R;runner.campaign.HERE=R
inventory=json.loads((R/'storage-preservation-audit.json').read_text())
assert inventory['passed'] and all(not n['serving_processes'] for n in inventory['nodes'])
assert (datetime.now(timezone.utc)-datetime.fromisoformat(inventory['at'])).total_seconds()<600
original=json.loads((R/'original-service.json').read_text()); config=ROOT/original['original_deployment'];assert json.loads(config.read_text())==original['config']
binary=ROOT/'build-release/dgpp-serve'; digest=hashlib.sha256(binary.read_bytes()).hexdigest();assert digest==json.loads((R/'manifest.json').read_text())['binary_sha256']=='2bb443de41dbc0a2a804557a0126e4497fddb28ad0aec3e040cdceccc6c2cf34'
D=R/'restoration';D.mkdir(exist_ok=False)
shutil.copy2(__file__,D/'restore.py');shutil.copy2(config,D/'original-config.json')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');backup=Path('/tmp')/('dgpp-original-service-logs-before-restore-'+stamp)
logdir=Path(original['log_dir']);shutil.copytree(logdir,backup)
runner.campaign.save(D/'plan.json',{'at':runner.campaign.now(),'config':str(config),'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'binary':str(binary),'binary_sha256':digest,'log_dir':str(logdir),'previous_logs_backup':str(backup),'environment':{'DGPP_NO_SWAP':os.environ['DGPP_NO_SWAP'],'CUDA_DEVICE_MAX_CONNECTIONS':os.environ['CUDA_DEVICE_MAX_CONNECTIONS']}})
print('Starting original configuration with the validated fixed binary.',flush=True)
runner.start_telemetry(D,config)
try:
 command=[sys.executable,str(ROOT/'scripts/dgpp-cluster'),'up','--config',str(config),'--bin',str(binary),'--log-dir',str(logdir)]
 assert runner.campaign.run(command,D/'up.log',timeout=900,resume=False),'Original service startup failed'
 with urllib.request.urlopen('http://127.0.0.1:18080/v1/models',timeout=30) as response:models=json.load(response)
 assert models['data'][0]['id']==original['config']['model'];runner.campaign.save(D/'models.json',models)
 body={'model':original['config']['model'],'messages':[{'role':'user','content':'Reply with exactly OK.'}],'temperature':0,'max_tokens':32}
 request=urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=120) as response:reply=json.load(response)
 assert reply['choices'] and reply['usage']['completion_tokens']>0
 runner.campaign.save(D/'smoke-test.json',{'request':body,'response':reply})
 command=[sys.executable,str(ROOT/'scripts/dgpp-cluster'),'status','--config',str(config),'--log-dir',str(logdir)]
 assert runner.campaign.run(command,D/'status.log',timeout=60,resume=False)
 status=(D/'status.log').read_text()
 for rank in range(4):assert re.search(rf'rank {rank} .*alive \(\d+\)',status),status
 time.sleep(10)
finally:
 runner.campaign.stop_telemetry(D)
runner.check_memory(D,config)
print('Startup and inference passed; monitoring SSH sessions closed. Waiting 45 seconds to check service persistence.',flush=True)
time.sleep(45)
with urllib.request.urlopen('http://127.0.0.1:18080/health',timeout=30) as response:health=json.load(response)
runner.campaign.save(D/'health-after-logout.json',health)
inspection = r'''import json,hashlib,os
from pathlib import Path
rows=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid != os.getuid() or (p/'comm').read_text().strip()!='dgpp-serve':continue
  f=dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
  cg=Path('/sys/fs/cgroup')/next(line[3:] for line in (p/'cgroup').read_text().splitlines() if line.startswith('0::')).lstrip('/')
  rows.append({'pid':int(p.name),'binary_sha256':hashlib.sha256((p/'exe').read_bytes()).hexdigest(),'swap_KiB':int(f['VmSwap'].split()[0]),'swap_max':(cg/'memory.swap.max').read_text().strip(),'swap_current':int((cg/'memory.swap.current').read_text())})
 except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps(rows))
'''
checks=[]
for n in range(11,15):
 host=f'192.168.88.{n}'
 result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','stephen@'+host,'python3 -'],input=inspection,capture_output=True,text=True,check=True,timeout=30)
 rows=json.loads(result.stdout)
 checks.append({'host':host,'processes':rows})
 runner.campaign.save(D/'processes-after-logout.json',checks)
 assert len(rows)==1,checks
 assert rows[0]['binary_sha256']==digest and rows[0]['swap_KiB']==0 and rows[0]['swap_max']=='0' and rows[0]['swap_current']==0,checks
print('All four ranks survived logout with the validated binary and zero process/cgroup swap.',flush=True)
runner.campaign.save(D/'restored.json',{'at':runner.campaign.now(),'passed':True,'model':original['config']['model'],'world_size':4,'binary_sha256':digest,'original_config_matches':True,'api_model_check':True,'inference_smoke_check':True,'all_ranks_alive':True,'zero_process_and_cgroup_swap':True,'service_left_running':True,'os_swap_unchanged':True,'logout_persistence_check_seconds':45,'linger_enabled_on_all_nodes':True})
print('Original service restored; API and inference checks passed, all four ranks alive, zero process/cgroup swap. Service left running.',flush=True)
