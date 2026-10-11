"""All-rank CUDA/OS timelines plus independent privileged GPU/CPU sampling.
The inference process stays unprivileged. Root sampling sessions are duration-bounded.
"""
import concurrent.futures,datetime,hashlib,json,os,pathlib,shlex,subprocess,sys,time
out=pathlib.Path(__file__).parent;root=pathlib.Path('/home/stephen/workspace/dgpp');binary=out/'dgpp-serve-decode-final'
launcher=out/'cuda_only_cluster.py'
s=(root/'scripts/dgpp-cluster').read_text().replace('from site_env import ',f'sys.path.insert(0, {str(root/"scripts")!r})\nfrom site_env import ',1).replace('ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))',f'ROOT = {str(root)!r}')
s=s.replace('f"--log serve_r{rank}.log --cwd . -- {self.peer_bin} "','f"--log serve_r{rank}.log --cwd . -- /usr/local/cuda/bin/nsys profile -t cuda --sample=none --cuda-graph-trace=node --force-overwrite=true -o {self.stage_dir}/rank{rank} {self.peer_bin} "')
assert 'nsys profile -t cuda' in s
launcher.write_text(s)
env={**os.environ,'CUDA_DEVICE_MAX_CONNECTIONS':'32','DGPP_NO_SWAP':'1','DGPP_BUS_TIMELINE':'1','DGPP_DFLASH2_PHASES':'1'}
for key in ('DGPP_ROCE_DEVICES','DGPP_ROCE_GID_INDICES'):env.pop(key,None)
def run(cmd,log,check=True,timeout=2400):
 receipt=dict(command=list(map(str,cmd)),started=datetime.datetime.now(datetime.timezone.utc).isoformat());print(receipt['started'],log,flush=True)
 with pathlib.Path(log).open('w') as f:r=subprocess.run(receipt['command'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 receipt.update(returncode=r.returncode,finished=datetime.datetime.now(datetime.timezone.utc).isoformat());pathlib.Path(str(log)+'.command.json').write_text(json.dumps(receipt,indent=2))
 if check and r.returncode:raise RuntimeError(str(log))
 return r.returncode
for quant in sys.argv[1:] or ['fp8','nvfp4']:
 d=out/('decode-cuda-only-'+quant);d.mkdir(exist_ok=False)
 cfg=json.loads((root/f'deploy/cluster_qwen3.8-27b_{quant}_w4.example.json').read_text());cfg['http']={'bind_host':'127.0.0.1','port':18081};cfg['engine']['lookup_draft']=False
 config=d/'deployment.json';config.write_text(json.dumps(cfg,indent=2));(d/'binary.sha256').write_text(hashlib.sha256(binary.read_bytes()).hexdigest()+'\n')
 samplers=[];handles=[]
 try:
  run(['python3',launcher,'up','--config',config,'--bin',binary,'--log-dir',d/'world','--head-wrap',f'/usr/local/cuda/bin/nsys profile -t cuda --sample=none --cuda-graph-trace=node --force-overwrite=true -o {d}/rank0'],d/'up.log')
  resolved=json.loads((d/'world/cluster.resolved.json').read_text());stage=resolved['paths']['stage_dir'];hosts=resolved['nodes'];user=resolved['ssh_user']
  run(['python3','scripts/serve_bench.py','127.0.0.1','18081','32','warm'],d/'warm.log')
  time.sleep(2)
  run(['python3','scripts/timed_load.py','127.0.0.1','18081','--concurrency','1','--classes','prose','--max-tokens','256','--repeat','3','--json-out',d/'decode.json'],d/'decode.log')
  rc=[p.wait(timeout=60) for p in samplers];(d/'metrics-status.json').write_text(json.dumps(rc));assert not any(rc),rc
 finally:
  for p in samplers:
   if p.poll() is None:p.wait(timeout=60)
  for f in handles:f.close()
  run(['python3','scripts/dgpp-cluster','down','--config',config,'--log-dir',d/'world'],d/'down.log',False)
 ops=sorted((d/'world').glob('serve_rank*.ops'));hashes={p.name:hashlib.md5(p.read_bytes()).hexdigest() for p in ops};(d/'opstreams.json').write_text(json.dumps(hashes,indent=2));assert len(hashes)==4 and len(set(hashes.values()))==1
 for rank in range(1,4):
  for name in [f'rank{rank}']:
   run(['scp','-q',f'{user}@{hosts[rank]}:{stage}/{name}.nsys-rep',d/(name+'.nsys-rep')],d/(name+'-collect.log'))
 for rep in sorted(d.glob('*.nsys-rep')):
  run(['/usr/local/cuda/bin/nsys','export','--type','sqlite','--force-overwrite=true','-o',rep.with_suffix('.sqlite'),rep],rep.with_suffix('.export.log'))
 print('complete',quant,flush=True)
