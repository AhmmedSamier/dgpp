"""Check matched cold-prefill reports from the two MiMo binaries."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
from evidence import load_record
HERE=Path(__file__).resolve().parent
all_rows=[];checks=[]
for dep in ['mimo-w2','mimo-w2-fp8kv']:
    dirs=[HERE/'raw'/dep/'prefill'/v for v in ['reproduction','fp8-dispatch-fix']]
    for d in dirs:
        for name in ['complete.json','memory-check.json','rank-identity.json']:
            assert load_record(d)['records'][name]['passed']
    samples=[json.loads((d/'prefill.json').read_text())['samples'] for d in dirs]
    assert len(samples[0])==len(samples[1])==6
    by=defaultdict(lambda:[[],[]])
    for a,b in zip(*samples):
        diff=[k for k in ['requested_tokens','repeat','prompt_sha256','usage','text','finish','cached_tokens','computed_tokens'] if a[k]!=b[k]]
        assert a['cached_tokens']==b['cached_tokens']==0
        checks.append({'deployment':dep,'target':a['requested_tokens'],'repeat':a['repeat'],'differences':diff})
        for i,s in enumerate([a,b]):by[s['requested_tokens']][i].append(s['prefill_ms']/1000)
    for target,(a,b) in sorted(by.items()):
        all_rows.append({'deployment':dep,'target':target,'baseline_s':median(a),'fixed_s':median(b),'speedup':median(a)/median(b)})
passed=all(not c['differences'] for c in checks)
result={'passed':passed,'rows':all_rows,'checks':checks,'scope':'Unprofiled fresh-server comparisons, identical prompt hashes/configurations. TP2, MTP1, four slots; 128K BF16 KV or 256K FP8 KV. Two probes per target; no allocation changes.'}
(HERE/'mimo-dispatch-fix/comparison.json').write_text(json.dumps(result,indent=2)+'\n')
assert passed,'Prompt/output/usage mismatch'
print(json.dumps(all_rows,indent=2))
