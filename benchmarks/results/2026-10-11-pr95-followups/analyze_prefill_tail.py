from pathlib import Path
import sqlite3,json,collections,statistics
out=Path(__file__).parent
result={'method':'CUDA-only traces on both ranks; split each request by embed_gather launches. Times by kernel are summed activity, not SM utilization. Request windows are separated by >100 ms without GPU kernels. Includes one completion token and speculative work after prompt prefill.','formats':{}}
for fmt in ('fp8','nvfp4'):
 d=out/f'prefill-tail-profile-{fmt}';req=json.loads((d/'prefill.json').read_text());fmt_data={'requests':req,'ranks':{}}
 for rank in (0,1):
  c=sqlite3.connect(d/f'rank{rank}.sqlite')
  rows=c.execute('select k.start,k.end,s.value,k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start').fetchall()
  bursts=[[]];end=0
  for r in rows:
   if bursts[-1] and r[0]-end>100000000:bursts.append([])
   bursts[-1].append(r);end=max(end,r[1])
  records=[]
  assert len(bursts)>=12
  for sample,burst in zip(req['samples'],bursts[-12:]):
   embeds=[(i,r) for i,r in enumerate(burst) if 'embed_gather' in r[2]]
   chunks=[]
   for q,(i,r) in enumerate(embeds):
    j=embeds[q+1][0] if q+1<len(embeds) else len(burst)
    part=burst[i:j];by=collections.defaultdict(lambda:[0,0]);busy=0;end=r[0]
    for a,b,n,g in part:
     by[n][0]+=(b-a)/1e6;by[n][1]+=1
     busy+=max(0,b-max(end,a));end=max(end,b)
    wall=max(z[1] for z in part)-r[0]
    chunks.append({'rows':r[3],'wall_ms':wall/1e6,'kernel_union_ms':busy/1e6,'kernel_idle_ms':(wall-busy)/1e6,'kernels':[dict(name=n,sum_ms=t,count=ct) for n,(t,ct) in sorted(by.items(),key=lambda x:-x[1][0])]})
   records.append({'requested_tokens':sample['requested_tokens'],'repeat':sample['repeat'],'prompt_tokens':sample['prompt_tokens'],'prompt_sha256':sample['prompt_sha256'],'chunks':chunks})
  fmt_data['ranks'][str(rank)]=records
 result['formats'][fmt]=fmt_data
for a,b in zip(result['formats']['fp8']['requests']['samples'],result['formats']['nvfp4']['requests']['samples']):
 assert a['prompt_sha256']==b['prompt_sha256'] and a['prompt_tokens']==b['prompt_tokens'] and a['cached_tokens']==b['cached_tokens']==0
result['matched_inputs']=True
(out/'prefill-tail-profile-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for fmt,data in result['formats'].items():
 print(fmt)
 for length in [512,2048,8192,32768]:
  print(length,statistics.median(r['prefill_ms'] for r in data['requests']['samples'] if r['requested_tokens']==length))
