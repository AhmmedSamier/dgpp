from pathlib import Path
import concurrent.futures,subprocess,hashlib
out=Path(__file__).parent;binary=Path('/home/stephen/workspace/dgpp/build-pr95/micro_mem_bw');hosts=['192.168.88.'+str(x) for x in (11,12,13,14)]
def run(rank):
 remote='/tmp/pr95-micro-mem-bw'
 if rank:subprocess.run(['scp','-q',str(binary),f'stephen@{hosts[rank]}:{remote}'],check=True)
 for tag,flags in [('device',[]),('managed',['-m'])]:
  cmd=[str(binary),'-s','1024','-i','20',*flags] if rank==0 else ['ssh',f'stephen@{hosts[rank]}',remote+' -s 1024 -i 20 '+' '.join(flags)]
  with (out/f'decode-memory-rank{rank}-{tag}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('done',rank,flush=True)
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(run,range(4)))
(out/'decode-memory-binary.sha256').write_text(hashlib.sha256(binary.read_bytes()).hexdigest()+'\n')
