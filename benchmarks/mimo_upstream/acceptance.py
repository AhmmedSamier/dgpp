"""Separate serial acceptance diagnostics; keep instrumentation out of timing suite."""
import json,sys
from benchmark import ROOT,PROMPTS,call,metrics
out=ROOT/'raw'/sys.argv[1];out.mkdir(parents=True,exist_ok=True);path=out/'acceptance.json'
assert not path.exists(),'Preserve previous evidence'
records=[]
for name,prompt in PROMPTS.items():
 before=metrics();assert before['scheduler']['active']==before['scheduler']['queued']==0
 r=call('acceptance-'+name,prompt,256,False);after=metrics()
 assert after['service']['requests_total']-before['service']['requests_total']==1,'external traffic'
 a=before['scheduler']['spec_decode'];b=after['scheduler']['spec_decode']
 r['rounds']=b['num_drafts_total']-a['num_drafts_total']
 r['attempts_per_position']=[y-x for x,y in zip(a['num_draft_tokens_per_pos_total'],b['num_draft_tokens_per_pos_total'])]
 r['accepted_per_position']=[y-x for x,y in zip(a['num_accepted_tokens_per_pos_total'],b['num_accepted_tokens_per_pos_total'])]
 r['accepted_per_round']=sum(r['accepted_per_position'])/r['rounds'] if r['rounds'] else None
 records.append(r);path.write_text(json.dumps(records,indent=2))
 print(name,'rounds',r['rounds'],'attempts',r['attempts_per_position'],'accepts',r['accepted_per_position'],'accepted/round',r['accepted_per_round'],flush=True)
