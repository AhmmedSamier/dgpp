import bisect,collections,json,pathlib,sqlite3,statistics,sys
out=pathlib.Path(__file__).parent
prefix=sys.argv[1] if len(sys.argv)>1 else 'decode-deep-profile'
def stats(a):
 a=sorted(a)
 return dict(n=len(a),mean=statistics.mean(a),p50=statistics.median(a),p95=a[int((len(a)-1)*.95)],p99=a[int((len(a)-1)*.99)],max=max(a)) if a else {}
def union(intervals):
 end=-1;total=0
 for s,e in sorted(intervals):total+=max(0,e-max(s,end));end=max(end,e)
 return total
def merged(intervals):
 result=[]
 for s,e in sorted(intervals):
  if result and s<=result[-1][1]:result[-1]=(result[-1][0],max(e,result[-1][1]))
  else:result.append((s,e))
 return result,[s for s,e in result],[e for s,e in result]
def clipped(runs,a,b):
 values,starts,ends=runs
 return [(max(s,a),min(e,b)) for s,e in values[bisect.bisect_right(ends,a):bisect.bisect_left(starts,b)]]
def busy(runs,a,b):return sum(e-s for s,e in clipped(runs,a,b))
report=dict(prefix=prefix,idle_definition='gpu_idle_ms: no observed CUDA kernel, memcpy or memset active, all streams, clipped at pass boundaries; kernel_idle_ms excludes memory operations; collective spin counts as active, not full-chip utilization.',ranks=[])
for quant in ('fp8','nvfp4'):
 d=out/f'{prefix}-{quant}'
 dec=json.loads((d/'decode.json').read_text());nsteps=sum(p['engine']['decode_steps'] for p in dec['phases'])
 for rank in range(4):
  con=sqlite3.connect(d/f'rank{rank}.sqlite');names=dict(con.execute('SELECT id,value FROM StringIds'));rows=con.execute('SELECT start,end,streamId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start').fetchall();marks=[r[0] for r in rows if 'publish_seq_kernel' in names[r[3]]];first=len(marks)-nsteps;starts=[r[0] for r in rows];samples=[];layers=collections.defaultdict(list);kernels=collections.defaultdict(lambda:[0,0]);gaps=[];gpu_gaps=[]
  tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
  memory={kind:con.execute(f'SELECT start,end FROM CUPTI_ACTIVITY_KIND_{kind}').fetchall() if 'CUPTI_ACTIVITY_KIND_'+kind in tables else [] for kind in ('MEMCPY','MEMSET')}
  kernel_runs=merged([(s,e) for s,e,_,_ in rows]);main_runs=merged([(s,e) for s,e,_,name in rows if 'l2_prefetch' not in names[name]])
  device_runs=merged([(s,e) for s,e,_,_ in rows]+memory['MEMCPY']+memory['MEMSET']);copy_runs=merged(memory['MEMCPY']);memset_runs=merged(memory['MEMSET'])
  for ix in range(first,len(marks)-1):
   a,b=marks[ix:ix+2]
   # Inter-request prefill and host wait are outside a decode pass.
   rr=rows[bisect.bisect_left(starts,a):bisect.bisect_left(starts,b)];embed=[r for r in rr if 'embed_gather_kernel' in names[r[3]]]
   if len(embed)!=2 or b-a>180e6:continue
   stream=next(r[2] for r in rr if 'publish_seq_kernel' in names[r[3]]);main=[r for r in rr if 'l2_prefetch' not in names[r[3]]];main_busy=busy(main_runs,a,b);kernel_busy=busy(kernel_runs,a,b);any_busy=busy(device_runs,a,b);target0=embed[1][0];target=[r for r in main if r[0]>=target0];ars=[r for r in target if 'bus_allreduce_graph_kernel' in names[r[3]]]
   if len(ars)<128:continue
   v=dict(marker_index=ix,wall_ms=(b-a)/1e6,main_busy_ms=main_busy/1e6,main_idle_ms=((b-a)-main_busy)/1e6,gpu_idle_ms=((b-a)-any_busy)/1e6,draft_ms=(target0-embed[0][0])/1e6,commit_ms=(embed[0][0]-a)/1e6)
   v.update(kernel_idle_ms=((b-a)-kernel_busy)/1e6,memory_only_ms=(any_busy-kernel_busy)/1e6,memcpy_ms=busy(copy_runs,a,b)/1e6,memset_ms=busy(memset_runs,a,b)/1e6)
   # Separate the first 128 target-layer folds from the head/pick tail.
   v['target_layers_ms']=(ars[127][1]-target0)/1e6;v['head_pick_tail_ms']=(b-ars[127][1])/1e6
   v['target_collectives_ms']=sum(r[1]-r[0] for r in ars[:128])/1e6
   for layer in range(64):
    begin=target0 if layer==0 else ars[2*layer-1][1];attn_ar=ars[2*layer];mlp_ar=ars[2*layer+1]
    seg=[r for r in target if begin<=r[0]<mlp_ar[1]]
    for tag,left,right in [('attention',begin,attn_ar[0]),('mlp',attn_ar[1],mlp_ar[0])]:
     layers[layer,tag].append((right-left)/1e3)
    layers[layer,'attention_collective'].append((attn_ar[1]-attn_ar[0])/1e3);layers[layer,'mlp_collective'].append((mlp_ar[1]-mlp_ar[0])/1e3)
   for runs,bucket in ((main_runs,gaps),(device_runs,gpu_gaps)):
    prev=a
    for s,e in clipped(runs,a,b):
     if s>prev:bucket.append((s-prev)/1000)
     prev=e
    if b>prev:bucket.append((b-prev)/1000)
   for s,e,_,name in rr:
    n=names[name];kernels[n][0]+=(e-s)/1e6;kernels[n][1]+=1
   samples.append(v)
  summary={k:stats([s[k] for s in samples]) for k in samples[0] if k!='marker_index'}
  entry=dict(quant=quant,rank=rank,markers=len(marks),skip=first,decode_passes=len(samples),summary=summary,gaps_us=stats(gaps),gaps_over_10us=sum(x>10 for x in gaps),gaps_over_100us=sum(x>100 for x in gaps),layers=[dict(layer=l,phase=t,us=stats(v)) for (l,t),v in sorted(layers.items())],kernels=[dict(name=n,ms_per_pass=t/len(samples),instances_per_pass=c/len(samples)) for n,(t,c) in sorted(kernels.items(),key=lambda x:-x[1][0])],samples=samples)
  report['ranks'].append(entry)
  entry['gpu_gaps_us']=stats(gpu_gaps)
  print(quant,rank,{k:round(v['mean'],3) for k,v in summary.items()},flush=True)
(out/(prefix+'-analysis.json')).write_text(json.dumps(report,indent=2))
