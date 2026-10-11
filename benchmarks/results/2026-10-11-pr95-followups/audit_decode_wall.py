"""Supplement the engine decode gate with client wall throughput."""
from pathlib import Path
import json,statistics
out=Path(__file__).parent;rows=[]
for w in (1,2,4):
    legs=[json.loads((out/f'decode-wide-final-w{w}-{s}/decode.json').read_text())['phases'] for s in ('fp8a','nvfp4','fp8b')]
    for c in ('prose','code','json','math','chat'):
        for n in (1,2,4,8):
            groups=[[p for p in leg if p['class']==c and p['concurrency']==n] for leg in legs]
            for r in range(3):assert len({tuple(x['prompt_sha256'] for x in g[r]['requests']) for g in groups})==1
            v=[statistics.median(p['metrics']['wall_tokens_per_s'] for p in g) for g in groups]
            rows.append(dict(world=w,kind=c,concurrency=n,fp8a=v[0],nvfp4=v[1],fp8b=v[2],gain_pct=100*(v[1]/max(v[0],v[2])-1),completion_tokens=[[sum(r['tokens'] for r in p['requests']) for p in g] for g in groups]))
report=dict(rows=rows,wins=sum(r['gain_pct']>0 for r in rows),note='Client wall throughput includes prompt prefill and client/HTTP time. The main decode gate uses engine decode token/s at all concurrency levels. Native greedy completion lengths can differ between formats; all samples remain explicit.')
(out/'decode-wall-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('Wall wins',report['wins'],'/60')
