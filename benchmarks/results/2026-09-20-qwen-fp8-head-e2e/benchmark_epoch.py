"""One measured epoch using the repository's corpus, SSE client and metrics."""
import argparse,hashlib,json,statistics,sys,time,urllib.request
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/serve_load.py').is_file())
sys.path.insert(0,str(ROOT/'scripts'))
import serve_load
p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--calibration',action='store_true');a=p.parse_args()
out=Path(a.output);model=serve_load.served_model('127.0.0.1',30002)
serve_load.check_corpus()
report={'model':model,'max_tokens':256,'thinking':False,'temperature':0,'calibration':a.calibration,'warmups':[],'phases':[]}
def fetch():
    with urllib.request.urlopen('http://127.0.0.1:30002/v1/metrics',timeout=20) as r:return json.load(r)
def save():out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
def phase(c,base,prompts,label,warm=False):
    before=fetch();records=[]
    serve_load.phase('127.0.0.1',30002,model,c,256,False,base,prompts,0,label,records)
    record=records[0];record['metrics_before']=before;record['metrics_after']=fetch()
    for request in record['requests']:
        if request['status']!=200 or request['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0):raise RuntimeError('HTTP failure or unexpected cache hit')
        request['text_sha256']=hashlib.sha256(request['text'].encode()).hexdigest()
    report['warmups' if warm else 'phases'].append(record);save()
phase(4,0,serve_load.PROMPTS,'warm',True)
if a.calibration:
    for repeat in range(6):phase(4,0,serve_load.PROMPTS,'mixed')
    rates=[x['metrics']['wall_tokens_per_s'] for x in report['phases']]
    report['calibration_cv']=statistics.stdev(rates)/statistics.mean(rates)
    report['identical_outputs']=all([(r['text_sha256'],r['tokens']) for r in x['requests']]==[(r['text_sha256'],r['tokens']) for r in report['phases'][0]['requests']] for x in report['phases'])
    save();print('CALIBRATION',report['calibration_cv'],report['identical_outputs'],flush=True)
else:
    for cls in ['prose','code','json','math','chat']:
        for c,base in [(1,0),(4,1)]:
            for repeat in range(3):phase(c,base,serve_load.CLASSES[cls],cls)
