"""Reproduction harness for the adjacent vision validation record."""
import os
import base64,json,re,sys,urllib.request
from pathlib import Path
root=Path(os.environ.get('DGPP_VISION_FIXTURES', '/tmp/dgpp-vision-quality')) / 'ai2d'
cases=json.loads((root/'prepared.json').read_text())
numbered = '--numbered' in sys.argv[2:]
correct=0
with Path(sys.argv[1]).open('w') as output:
 for c in cases:
  prompt=c['question']+'\n'+ '\n'.join(f'{i+1 if numbered else chr(65+i)}. {t}' for i,t in enumerate(c['options']))+('\nReturn only the option number (1, 2, 3, or 4).' if numbered else '\nAnswer with only the letter of the correct option.')
  url='data:image/jpeg;base64,'+base64.b64encode((root/c['file']).read_bytes()).decode()
  body={'model':'HawkBearPig/GLM-5.3-Flash-NVFP4-FP8','messages':[{'role':'user','content':[{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':url}}]}],'temperature':0,'max_completion_tokens':1024,'reasoning_effort':'low','logprobs':True,'top_logprobs':5}
  req=urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=180) as r: response=json.load(r)
  answer=response['choices'][0]['message']['content'].strip()
  gold=str(c['answer']+1) if numbered else chr(65+c['answer']);ok=answer.strip(' .')==gold;correct+=ok
  row={'index':c['index'],'gold':gold,'correct':ok,'response':response}
  output.write(json.dumps(row)+'\n');output.flush()
  print(c['index'],'expected',gold,'got',repr(answer),'correct',ok,flush=True)
 print('accuracy',correct,'/',len(cases),flush=True)
