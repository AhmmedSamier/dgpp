"""Long-context retrieval checks: arithmetic changes need more than generation smoke."""
import hashlib,json,pathlib,re,sys,time
from benchmark import ROOT,call,metrics

def prompt(n):
 keys=['amber','cobalt','ivory'];expected={k:hashlib.sha256(f'recipe-retrieval-v1:{n}:{k}'.encode()).hexdigest()[:16] for k in keys}
 records=[f'Record {i}: value {i%17}.\n' for i in range(n)]
 for key,fraction in zip(keys,[.1,.5,.9]):records[int(n*fraction)]=f'The secret registration for {key} is {expected[key]}.\n'
 text='Read the records carefully and retrieve the exact secret registrations.\n'+''.join(records)+'\nReturn only a JSON object mapping amber, cobalt, and ivory to their exact secret registrations. Do not infer or invent values.'
 return text,expected

def main(tag,counts):
 out=ROOT/'raw'/tag;out.mkdir(parents=True,exist_ok=True)
 dest=out/'quality.json';assert not dest.exists(),'Preserve previous quality evidence'
 before=metrics();assert before['scheduler']['active']==before['scheduler']['queued']==0
 rows=[];skipped=[];started=time.monotonic();previous_n=None
 for index,n in enumerate(counts):
  if rows and previous_n:
   estimate=rows[-1]['ttft']*(n/previous_n)**2
   if estimate>1050 or time.monotonic()-started+estimate+30>1750:
    reason=dict(records=n,estimated_ttft_seconds=estimate,reason='Predicted request exceeds 1050-second bound or phase exceeds 1750-second bound; not executed')
    skipped.append(reason);(out/'quality-skipped.json').write_text(json.dumps(skipped,indent=2))
    print('QUALITY SKIPPED',reason,flush=True)
    continue
  p,expected=prompt(n);r=call(f'retrieval-{n}',p,1024,False)
  text=r['content'].strip();parsed=None
  try:parsed=json.loads(text.removeprefix('```json').removeprefix('```').removesuffix('```').strip())
  except ValueError:pass
  r.update(expected=expected,parsed=parsed,exact_answer=parsed==expected,values_found={k:v in text for k,v in expected.items()})
  previous_n=n
  rows.append(r);dest.write_text(json.dumps(rows,indent=2))
  print('QUALITY',n,'exact JSON',r['exact_answer'],'values',r['values_found'],flush=True)
  if not r['exact_answer']:
   skipped.extend(dict(records=later,reason='Not executed after an earlier exact retrieval failure') for later in counts[index+1:])
   if skipped:(out/'quality-skipped.json').write_text(json.dumps(skipped,indent=2))
   break
 after=metrics();assert after['service']['requests_total']-before['service']['requests_total']==len(rows),'external traffic'
 assert after['service']['requests_failed']==before['service']['requests_failed'] and not after['service']['engine_failed']
 (out/'quality-metrics.json').write_text(json.dumps(dict(before=before,after=after,requested_counts=counts,skipped=skipped,complete=not skipped),indent=2))
 print('QUALITY SUMMARY',sum(r['exact_answer'] for r in rows),'/',len(rows),'exact;',len(skipped),'skipped; this is fixed retrieval evidence, not broad quality equivalence',flush=True)
 if any(not r['exact_answer'] for r in rows):raise SystemExit(1)
if __name__=='__main__':main(sys.argv[1],[int(x) for x in sys.argv[2:]] or [6000,11000])
