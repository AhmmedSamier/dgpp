from pathlib import Path
import concurrent.futures,subprocess,json,hashlib,time
out=Path(__file__).parent;binary=out/'decode_weight_bench';hosts=['192.168.88.'+str(x) for x in (11,12,13,14)]
def run(rank):
 remote='/tmp/pr95-decode-weight-bench'
 if rank:subprocess.run(['scp','-q',str(binary),f'stephen@{hosts[rank]}:{remote}'],check=True)
 cmd=[str(binary),'4','160'] if rank==0 else ['ssh',f'stephen@{hosts[rank]}',remote+' 4 160']
 with (out/f'decode-weight-rank{rank}.csv').open('w') as f, (out/f'decode-weight-rank{rank}.err').open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
 print('done',rank,flush=True)
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(run,range(4)))
(out/'decode-weight-binary.sha256').write_text(hashlib.sha256(binary.read_bytes()).hexdigest()+'\n')
