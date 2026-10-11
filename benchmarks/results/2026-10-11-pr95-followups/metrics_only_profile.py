"""Bracket duration-bounded CPU/GPU sampling with uninstrumented repeats; no CUDA/OSRT tracing."""
import datetime,hashlib,json,os,pathlib,shlex,subprocess,sys,time
out=pathlib.Path(__file__).parent;root=pathlib.Path('/home/stephen/workspace/dgpp');binary=out/'dgpp-serve-decode-final'
env={**os.environ,'CUDA_DEVICE_MAX_CONNECTIONS':'32','DGPP_NO_SWAP':'1','DGPP_BUS_TIMELINE':'1'}
for key in ('DGPP_ROCE_DEVICES','DGPP_ROCE_GID_INDICES'):env.pop(key,None)
def run(cmd,log,check=True,timeout=2400):
 receipt=dict(command=list(map(str,cmd)),started=datetime.datetime.now(datetime.timezone.utc).isoformat());print(receipt['started'],log,flush=True)
 with pathlib.Path(log).open('w') as f:r=subprocess.run(receipt['command'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 receipt.update(returncode=r.returncode,finished=datetime.datetime.now(datetime.timezone.utc).isoformat());pathlib.Path(str(log)+'.command.json').write_text(json.dumps(receipt,indent=2))
 if check and r.returncode:raise RuntimeError(str(log))
 return r.returncode
for quant in sys.argv[1:] or ['fp8','nvfp4']:
 d=out/('decode-metrics-only-'+quant);d.mkdir(exist_ok=False)
 cfg=json.loads((root/f'deploy/cluster_qwen3.8-27b_{quant}_w4.example.json').read_text());cfg['http']={'bind_host':'127.0.0.1','port':18081};cfg['engine']['lookup_draft']=False
 config=d/'deployment.json';config.write_text(json.dumps(cfg,indent=2));(d/'binary.sha256').write_text(hashlib.sha256(binary.read_bytes()).hexdigest()+'\n')
 samplers=[];handles=[]
 def decode(tag):run(['python3','scripts/timed_load.py','127.0.0.1','18081','--concurrency','1','--classes','prose','--max-tokens','256','--repeat','3','--json-out',d/(tag+'.json')],d/(tag+'.log'))
 try:
  run(['python3','scripts/dgpp-cluster','up','--config',config,'--bin',binary,'--log-dir',d/'world'],d/'up.log')
  resolved=json.loads((d/'world/cluster.resolved.json').read_text());stage=resolved['paths']['stage_dir'];hosts=resolved['nodes'];user=resolved['ssh_user']
  run(['python3','scripts/serve_bench.py','127.0.0.1','18081','32','warm'],d/'warm.log');decode('before')
  offsets=[]
  for rank in range(4):
   t0=time.time_ns();stamp=int(subprocess.check_output(['ssh',f'{user}@{hosts[rank]}','date +%s%N']));t1=time.time_ns();offsets.append(dict(rank=rank,remote_minus_local_ns=stamp-(t0+t1)//2,roundtrip_ns=t1-t0))
   dest=str(d/f'metrics-r{rank}') if rank==0 else stage+f'/metrics-only-r{rank}'
   cmd=['sudo','-n','/usr/local/cuda/bin/nsys','profile','--sample=system-wide','--sampling-period=16000000','--gpu-metrics-devices=all','--gpu-metrics-set=gb20b','--gpu-metrics-frequency=1000','-t','none','--duration=24','--force-overwrite=true','-o',dest,'/usr/bin/sleep','24']
   if rank:cmd=['ssh',f'{user}@{hosts[rank]}',shlex.join(cmd)]
   f=(d/f'metrics-r{rank}.log').open('w');handles.append(f);samplers.append(subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT));(d/f'metrics-r{rank}.command.json').write_text(json.dumps(cmd,indent=2))
  (d/'clock-offsets.json').write_text(json.dumps(offsets,indent=2));time.sleep(2);decode('during');rc=[p.wait(timeout=60) for p in samplers];assert not any(rc),rc
  decode('after')
 finally:
  for p in samplers:
   if p.poll() is None:p.wait(timeout=60)
  for f in handles:f.close()
  run(['python3','scripts/dgpp-cluster','down','--config',config,'--log-dir',d/'world'],d/'down.log',False)
 ops=sorted((d/'world').glob('serve_rank*.ops'));hashes={p.name:hashlib.md5(p.read_bytes()).hexdigest() for p in ops};(d/'opstreams.json').write_text(json.dumps(hashes,indent=2));assert len(hashes)==4 and len(set(hashes.values()))==1
 for rank in range(1,4):run(['scp','-q',f'{user}@{hosts[rank]}:{stage}/metrics-only-r{rank}.nsys-rep',d/f'metrics-r{rank}.nsys-rep'],d/f'metrics-r{rank}-collect.log')
 for rep in sorted(d.glob('*.nsys-rep')):run(['/usr/local/cuda/bin/nsys','export','--type','sqlite','--force-overwrite=true','-o',rep.with_suffix('.sqlite'),rep],rep.with_suffix('.export.log'))
 print('complete',quant,flush=True)
