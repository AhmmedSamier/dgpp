"""Derive timing comparisons and verify identical replay workloads/outputs."""
from collections import defaultdict
import json
import sys
from pathlib import Path
from statistics import median
from evidence import load_record

HERE=Path(__file__).resolve().parent

def load(path):return json.loads(path.read_text())
def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def summaries(samples):
    by=defaultdict(list)
    for s in samples:by[s['requested_tokens']].append(s)
    return [{'target':k,'samples':len(v),'median_actual_tokens':median(s['prompt_tokens'] for s in v),'median_prefill_s':median(s['prefill_ms']/1000 for s in v),'median_ms_per_token':median(s['prefill_ms']/s['prompt_tokens'] for s in v)} for k,v in sorted(by.items())]

p=HERE/'raw/glm53-fp4kv-256k-w4/prefill'
a=p/'baseline-native-clean';b=p/'native-conversion-clean'
for d in [a,b]:
    for name in ['complete.json','memory-check.json','rank-identity.json']:assert load_record(d)['records'][name]['passed']
sa=load(a/'prefill.json')['samples'];sb=load(b/'prefill.json')['samples']
assert len(sa)==len(sb)
checks=[]
for x,y in zip(sa,sb):
    diff=[k for k in ['requested_tokens','repeat','prompt_sha256','usage','text','finish','cached_tokens','computed_tokens'] if x[k]!=y[k]]
    assert x['cached_tokens']==y['cached_tokens']==0
    checks.append({'target':x['requested_tokens'],'repeat':x['repeat'],'differences':diff})
ra=load(a/'decode-check.json');rb=load(b/'decode-check.json');dec=[]
for x,y in zip(ra,rb):
    dec.append({'same_request':x['request']==y['request'],'same_choices':x['response']['choices']==y['response']['choices'],'same_usage':x['response']['usage']==y['response']['usage'],'completion_tokens':x['response']['usage']['completion_tokens']})
rows=[]
for x,y in zip(summaries(sa),summaries(sb)):
    assert x['target']==y['target'];rows.append({'target':x['target'],'before':x,'after':y,'speedup':x['median_prefill_s']/y['median_prefill_s']})
passed=all(not c['differences'] for c in checks) and len(ra)==len(rb)==2 and all(d['same_request'] and d['same_choices'] and d['same_usage'] for d in dec)
save(HERE/'fp4-conversion-experiment/full-model-comparison.json',{'passed':passed,'scope':'Unprofiled matched fresh-server runs; same config/prompt hashes; only serving binary changes. MTP1, 256K FP4 KV, 8 slots, TP4. Two cold probes per target and two deterministic 256-token decode checks.','before_binary':load_record(a)['records']['manifest.json']['binary_sha256'],'after_binary':load_record(b)['records']['manifest.json']['binary_sha256'],'prefill_checks':checks,'decode_checks':dec,'rows':rows})
assert passed,'Full-model parity failed'
print('FP4:',json.dumps(rows))
if '--full-glm-only' in sys.argv:
    raise SystemExit(0)

old=load(HERE/'published-prefill-audit.json');reports=[]
for dep in ['mimo-w2','mimo-w2-fp8kv']:
    d=HERE/'raw'/dep/'prefill/reproduction'
    for name in ['complete.json','memory-check.json','rank-identity.json']:assert load_record(d)['records'][name]['passed']
    samples=load(d/'prefill.json')['samples'];assert all(s['cached_tokens']==0 for s in samples)
    for row in summaries(samples):
        before=next(v for v in old if v['deployment']==dep and v['target']==row['target'])
        reports.append({'deployment':dep,'current':row,'published':before,'ms_per_token_ratio_to_published':row['median_ms_per_token']/before['median_ms_per_actual_token']})
save(HERE/'mimo-reproduction-comparison.json',{'passed':True,'note':'New prompts differ from published probes; compare normalized per-token times and profiles, not exact latency ratios.','rows':reports})
print('MiMo:',json.dumps(reports))
