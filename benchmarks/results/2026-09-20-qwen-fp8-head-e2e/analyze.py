"""Report all predeclared cases; do not select the fastest run or workload."""
from pathlib import Path
import collections,gzip,json,statistics,sys,math
root=Path(sys.argv[1])
def read_epoch(name):
    path=root/(name+'.json')
    data=path.read_bytes() if path.exists() else gzip.decompress(Path(str(path)+'.gz').read_bytes())
    return json.loads(data)
epochs={name:read_epoch(name) for name in ['A1','B1','B2','A2']}
summary={'rows':[],'pooled':{},'paired_epochs':[],'matching_requests':0,'total_request_pairs':0,'differences':[],'within_variant_varying_groups':{}}
for cls in ['prose','code','json','math','chat']:
 for c in [1,4]:
  groups={variant:[p for name in names for p in epochs[name]['phases'] if p['class']==cls and p['concurrency']==c] for variant,names in [('baseline',['A1','A2']),('candidate',['B1','B2'])]}
  stats={v:statistics.median(p['metrics']['wall_tokens_per_s'] for p in ps) for v,ps in groups.items()}
  summary['rows'].append({'class':cls,'concurrency':c,**stats,'gain_pct':100*(stats['candidate']/stats['baseline']-1),'baseline_rates':[p['metrics']['wall_tokens_per_s'] for p in groups['baseline']],'candidate_rates':[p['metrics']['wall_tokens_per_s'] for p in groups['candidate']]})
  for p,q in zip(groups['baseline'],groups['candidate']):
   assert len(p['requests'])==len(q['requests'])
   for a,b in zip(p['requests'],q['requests']):
    assert a['prompt_sha256']==b['prompt_sha256']
    summary['total_request_pairs']+=1
    if a['text_sha256']==b['text_sha256'] and a['tokens']==b['tokens']:summary['matching_requests']+=1
    else:summary['differences'].append({'class':cls,'concurrency':c,'prompt_sha256':a['prompt_sha256'],'baseline_tokens':a['tokens'],'candidate_tokens':b['tokens'],'baseline_hash':a['text_sha256'],'candidate_hash':b['text_sha256']})
for c in [1,4]:
 rates={}
 for variant,names in [('baseline',['A1','A2']),('candidate',['B1','B2'])]:
  ps=[p for name in names for p in epochs[name]['phases'] if p['concurrency']==c]
  tokens=sum(p['metrics']['completion_tokens'] for p in ps);wall=sum(p['metrics']['wall_s'] for p in ps)
  rates[variant]={'tokens':tokens,'wall_s':wall,'tokens_per_s':tokens/wall}
 summary['pooled'][c]={**rates,'gain_pct':100*(rates['candidate']['tokens_per_s']/rates['baseline']['tokens_per_s']-1)}
for an,bn in [('A1','B1'),('A2','B2')]:
 rates={}
 for name in [an,bn]:
  ps=[p for p in epochs[name]['phases'] if p['concurrency']==4]
  rates[name]=sum(p['metrics']['completion_tokens'] for p in ps)/sum(p['metrics']['wall_s'] for p in ps)
 summary['paired_epochs'].append({'baseline':an,'candidate':bn,'rates':rates,'gain_pct':100*(rates[bn]/rates[an]-1)})
for variant,names in [('baseline',['A1','A2']),('candidate',['B1','B2'])]:
 groups=collections.defaultdict(set)
 for name in names:
  for p in epochs[name]['phases']:
   for r in p['requests']:groups[(p['class'],p['concurrency'],r['prompt_sha256'])].add((r['text_sha256'],r['tokens']))
 summary['within_variant_varying_groups'][variant]={'varying':sum(len(x)>1 for x in groups.values()),'total':len(groups)}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
