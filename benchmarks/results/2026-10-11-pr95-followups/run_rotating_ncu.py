from pathlib import Path
import subprocess,json
out=Path(__file__).parent
metrics='gpu__time_duration.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct'
for name,skip in [('gate-up',600),('down',648)]:
 cmd=['sudo','-n','/usr/local/cuda/bin/ncu','--replay-mode','application','--app-replay-mode','strict','--clock-control','none','--cache-control','none','--kernel-name-base','demangled','--kernel-name','regex:.*dense_fp4_decode_kernel.*','--launch-skip',str(skip),'--launch-count','1','--metrics',metrics,'--force-overwrite','--export',str(out/('decode-rotating-'+name)),str(out/'decode_weight_bench'),'4','32']
 (out/('decode-rotating-'+name+'.command.json')).write_text(json.dumps(cmd,indent=2))
 with (out/('decode-rotating-'+name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,'done',flush=True)
