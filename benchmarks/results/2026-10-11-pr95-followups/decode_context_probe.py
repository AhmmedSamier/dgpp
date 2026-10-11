"""Matched C1 decode at several prompt lengths, separate prefill/step counters."""
import hashlib,json,pathlib,random,sys
root=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'scripts'))
from serve_prefill_probe import get,idle_metrics,ask,prompt_of
from data_paths import data_dir
from timed_load import TimedLoad
host,port,dest=sys.argv[1],int(sys.argv[2]),pathlib.Path(sys.argv[3]);model=get(host,port,'/v1/models')['data'][0]['id'];raw=(data_dir()/'gsm8k_test.jsonl').read_bytes();words=[]
for line in raw.splitlines():words.extend(json.loads(line)['question'].split())
rng=random.Random(7);before=idle_metrics(host,port);cal=ask(host,port,model,prompt_of(words,400,'pr95-decode-context-cal',rng),True);idle_metrics(host,port,timeout=5,completed_after=before['prompts_prefilled']);per_word=(cal['usage']['prompt_tokens']-30)/400
load=TimedLoad();report=dict(model=model,seed=7,tag='pr95-decode-context',max_tokens=512,thinking=False,data_sha256=hashlib.sha256(raw).hexdigest(),calibration=cal,phases=[])
for length in (1024,8192,32768):
 for repeat in range(3):
  content=prompt_of(words,max(8,int((length-30)/per_word)),f'pr95-decode-context-{length}-{repeat}-{rng.randrange(1<<30)}',rng)
  content=content.replace('Summarize the following in one sentence.','Write a very detailed, long guide to solving the kinds of word problems in this reference. Include many worked examples and explanations. Continue until the response limit.')
  load.phase(host,port,model,1,512,False,0,prompts=[content],temperature=0,label=f'context-{length}-r{repeat}',reports=report['phases'])
  p=report['phases'][-1];p.update(requested_context=length,repeat=repeat)
  assert p['engine']['prompt_tokens_computed']==p['engine']['prompt_tokens'], 'context probe must have cold prefill'
  assert all(r['status']==200 and r['tokens']>1 for r in p['requests'])
  dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
