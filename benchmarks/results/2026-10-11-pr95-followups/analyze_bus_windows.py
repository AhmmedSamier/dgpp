import collections,datetime,json,pathlib,re,statistics
out=pathlib.Path(__file__).parent
pat=re.compile(r'graph window timeline: rank (\d+) variant (\d+) gens (\d+) avg us: total ([\d.]+) = copy ([\d.]+) \+ handshake ([\d.]+) \+ skew ([\d.]+) \+ fold ([\d.]+); engine post ([\d.]+); max handshake ([\d.]+) skew ([\d.]+) total ([\d.]+) \(gen (\d+)\); gate waits (\d+); arm->gen0 ([\d.]+) us; passes/gen (\d+); compute between gens ([\d.]+) us')
report=[]
for prefix in ('decode-cuda-only','decode-deep-profile'):
 for quant in ('fp8','nvfp4'):
  d=out/f'{prefix}-{quant}';dec=json.loads((d/'decode.json').read_text());start=dec['phases'][0]['started_at'];end=json.loads((d/'decode.log.command.json').read_text())['finished'][11:23];nsteps=sum(p['engine']['decode_steps'] for p in dec['phases'])
  for rank in range(4):
   windows=[];peer=collections.defaultdict(lambda:[0,0,0]);current=None
   for line in (d/f'world/serve_r{rank}.log').read_text().splitlines():
    m=pat.search(line)
    if m:
     vals=m.groups();keys=['rank','variant','gens','total','copy','handshake','skew','fold','engine_post','max_handshake','max_skew','max_total','max_gen','gate_waits','arm_gen0','passes_gen','compute_between'];current={k:float(v) for k,v in zip(keys,vals)};current['peer']={};windows.append(current)
    if 'graph window peers:' in line and current:
     for p,lag,last in re.findall(r'rank(\d+) lag (\d+)us last (\d+)x',line):
      current['peer'][int(p)]=(float(lag),float(last))
   windows=windows[-nsteps:]
   for w in windows:
    for p,(lag,last) in w['peer'].items():peer[p][0]+=lag*w['gens'];peer[p][1]+=last;peer[p][2]+=w['gens']
   ngen=sum(x['gens'] for x in windows);av={k:sum(x[k]*x['gens'] for x in windows)/ngen for k in keys if k not in ('rank','variant','gens')};rec=dict(prefix=prefix,quant=quant,rank=rank,windows=len(windows),gens=ngen,passes=nsteps,weighted_us=av,peer={k:dict(lag_us=v[0]/v[2],last_fraction=v[1]/v[2]) for k,v in peer.items()},raw=windows);report.append(rec)
   print(prefix,quant,rank,'windows',len(windows),'gens',ngen,{k:round(av[k],2) for k in ('total','copy','handshake','skew','fold','engine_post','arm_gen0')},rec['peer'])
(out/'decode-bus-analysis.json').write_text(json.dumps(report,indent=2))
