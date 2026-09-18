"""Reproduction harness for the adjacent vision validation record."""
import os
import base64,json,sys,urllib.request
from pathlib import Path
root=Path(os.environ.get('DGPP_VISION_FIXTURES', '/tmp/dgpp-vision-quality'))
cases={
'gradient': [
 'What color is at the top left corner? Answer only the color name.',
 'What color is at the bottom right corner? Answer only the color name.',
 'What color is at the top right corner? Answer only the color name.',
 'What color is at the bottom left corner? Answer only the color name.',
 'Describe how the colors change across this image in one sentence.'
],
'quadrants': [
 'What color is in the top left quadrant? Answer only the color name.',
 'What color is in the top right quadrant? Answer only the color name.',
 'What color is in the bottom left quadrant? Answer only the color name.',
 'What color is in the bottom right quadrant? Answer only the color name.',
 'How many solid colored rectangles are there? Answer only the number.'
],
'bars': [
 'Which bar is tallest? Answer only its color.',
 'Which bar is shortest? Answer only its color.',
 'How many bars are there? Answer only the number.',
 'List the bar colors from left to right. Answer only the colors.'
]}
with Path(sys.argv[1]).open('w') as output:
 for name,questions in cases.items():
  url='data:image/png;base64,'+base64.b64encode((root/(name+'.png')).read_bytes()).decode()
  for question in questions:
   body={'model':'HawkBearPig/GLM-5.3-Flash-NVFP4-FP8','messages':[{'role':'user','content':[{'type':'text','text':question},{'type':'image_url','image_url':{'url':url}}]}],'temperature':0,'max_completion_tokens':256,'reasoning_effort':'low','logprobs':True,'top_logprobs':5}
   req=urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=180) as r: response=json.load(r)
   row={'fixture':name,'question':question,'response':response}
   output.write(json.dumps(row)+'\n');output.flush()
   print(name,question,response['choices'][0]['message'],flush=True)
