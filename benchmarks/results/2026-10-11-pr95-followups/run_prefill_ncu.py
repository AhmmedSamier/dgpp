from pathlib import Path
import subprocess,json
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp')
s=(out/'tile_probe.cu').read_text();s=s.replace('for(int world: {1,2,4})for(bool down:{false,true})for(int m:{1023,2048,4096})','for(int world: {2})for(bool down:{false,true})for(int m:{4096})');a=s.index('  {float ms=time');b=s.index('  for(void* x:',a);s=s[:a]+s[b:]
p=out/'prefill_ncu_probe.cu';p.write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),str(p),'-o',str(out/'prefill_ncu_probe'),str(root/'build-pr95/libdgpp_kernels.a'),str(root/'build-pr95/libdgpp_core.a'),str(root/'build-pr95/libdgpp_rope_scaling.a'),'-lcublasLt','-lcudart','-lpthread','-ldl'],check=True)
metrics='gpu__time_duration.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct'
for name,skip in [('gate',1),('down',12)]:
 cmd=['sudo','-n','/usr/local/cuda/bin/ncu','--replay-mode','application','--app-replay-mode','strict','--clock-control','none','--cache-control','none','--kernel-name-base','demangled','--kernel-name','regex:.*fp4w_gemm_kernel.*','--launch-skip',str(skip),'--launch-count','1','--metrics',metrics,'--force-overwrite','--export',str(out/('prefill-large-'+name)),str(out/'prefill_ncu_probe')]
 (out/('prefill-large-'+name+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/('prefill-large-'+name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 with (out/('prefill-large-'+name+'.csv')).open('w') as f:subprocess.run(['/usr/local/cuda/bin/ncu','--import',str(out/('prefill-large-'+name+'.ncu-rep')),'--csv','--page','raw'],stdout=f,check=True)
