import datetime,bisect,collections,json,pathlib,sqlite3,statistics
out=pathlib.Path(__file__).parent;report=[]
for quant in ('fp8','nvfp4'):
 d=out/f'decode-metrics-only-{quant}';dec=json.loads((d/'during.json').read_text()); offsets=json.loads((d/'clock-offsets.json').read_text()); receipt=json.loads((d/'during.log.command.json').read_text())
 for rank in range(4):
  m=sqlite3.connect(d/f'metrics-r{rank}.sqlite');mepoch=m.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0];offset=offsets[rank]['remote_minus_local_ns']
  # The request interval includes its short prefill; no heavy CUDA/OSRT instrumentation.
  day=receipt['started'][:10];lo=int(datetime.datetime.fromisoformat(day+'T'+dec['phases'][0]['started_at']+'+00:00').timestamp()*1e9)+offset-mepoch
  last=dec['phases'][-1];hi=int(datetime.datetime.fromisoformat(day+'T'+last['started_at']+'+00:00').timestamp()*1e9)+int(last['metrics']['wall_s']*1e9)+offset-mepoch
  mn=dict(m.execute('select id,value from StringIds'));gp,pid,pname=m.execute('select globalPid,pid,name from PROCESSES where name like "%dgpp-serve%"').fetchone();tn={g:mn[n] for n,g in m.execute('select nameId,globalTid from ThreadNames')};threads=collections.defaultdict(lambda:dict(slices=0,oncpu_ns=0,cpus=collections.Counter(),top=collections.Counter(),samples=0));active={}
  for s,cpu,isin,g in m.execute('select start,cpu,isSchedIn,globalTid from SCHED_EVENTS where globalTid>=? and globalTid<? order by start',(gp,gp+(1<<24))):
   if isin:active[g]=(s,cpu)
   elif g in active:
    beg,pcpu=active.pop(g);duration=max(0,min(s,hi)-max(beg,lo))
    if duration:threads[g]['oncpu_ns']+=duration;threads[g]['cpus'][pcpu]+=duration;threads[g]['slices']+=1
  for g,(beg,cpu) in active.items():
   duration=max(0,hi-max(beg,lo));threads[g]['oncpu_ns']+=duration;threads[g]['cpus'][cpu]+=duration
  # Leaf frames avoid inflating sample counts with multiple stack levels.
  for g,cpu,sym,count in m.execute('select e.globalTid,e.cpu,s.symbol,count(*) from COMPOSITE_EVENTS e join SAMPLING_CALLCHAINS s on s.id=e.id where s.stackDepth=0 and e.start between ? and ? and e.globalTid>=? and e.globalTid<? group by e.globalTid,e.cpu,s.symbol',(lo,hi,gp,gp+(1<<24))):
   threads[g]['samples']+=count;threads[g]['top'][mn.get(sym,str(sym))]+=count
  metricnames=dict(m.execute('select metricId,metricName from TARGET_INFO_GPU_METRICS'));metrics=[]
  for mid,mean,mnval,mx,count in m.execute('select metricId,avg(case when metricId in (0,1) and value<0 then value+4294967296 else value end),min(case when metricId in (0,1) and value<0 then value+4294967296 else value end),max(case when metricId in (0,1) and value<0 then value+4294967296 else value end),count(*) from GPU_METRICS where timestamp between ? and ? group by metricId',(lo,hi)):
   scale=1e-6 if mid in (0,1) else 1;metrics.append(dict(name=metricnames[mid],mean=mean*scale,min=mnval*scale,max=mx*scale,n=count))
  rec=dict(quant=quant,rank=rank,pid=pid,window_seconds=(hi-lo)/1e9,window_metric_timestamps_ns=[lo,hi],metrics=metrics,threads=[dict(tid=g-gp,name=tn.get(g,''),oncpu_seconds=v['oncpu_ns']/1e9,oncpu_fraction=v['oncpu_ns']/(hi-lo),slices=v['slices'],cpus={k:val/1e9 for k,val in v['cpus'].items()},samples=v['samples'],top_leaf_samples=v['top'].most_common(12)) for g,v in sorted(threads.items(),key=lambda x:-x[1]['oncpu_ns'])]);report.append(rec)
  print(quant,rank,[(x['name'],round(x['mean'],2)) for x in metrics if x['name'].startswith(('GPC','SMs','SM Issue','Tensor'))],[(t['tid'],t['name'],round(t['oncpu_fraction'],3),t['cpus']) for t in rec['threads'][:5]],flush=True)
(out/'decode-metrics-only-analysis.json').write_text(json.dumps(report,indent=2))
