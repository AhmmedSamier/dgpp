"""Live extension checks for a local server on port 18080.

Writes requests/responses to /tmp/dgpp-pre-push-live-responses.json.
Uses the first advertised model and does not execute model tool calls.
"""
import base64,json,urllib.request
from pathlib import Path

url='http://127.0.0.1:18080'
with urllib.request.urlopen(url+'/v1/models') as response:
    model=json.load(response)['data'][0]['id']
results=[]
def run(name, messages, **fields):
    body=dict(model=model,messages=messages,temperature=0,max_completion_tokens=256,reasoning_effort='low',**fields)
    request=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response:
        result=json.load(response)
    results.append(dict(name=name,request=body,response=result))
    Path('/tmp/dgpp-pre-push-live-responses.json').write_text(json.dumps(results,indent=2)+'\n')
    return result['choices'][0]['message']

message=run('inline text file',[{'role':'user','content':[
    {'type':'text','text':'What is the code in the attached document? Answer only the code.'},
    {'type':'file','file':{'filename':'code.txt','file_data':base64.b64encode(b'The integration check code is COBALT-417.').decode()}}
]}])
assert 'COBALT-417' in message['content'], message
print('PASS: inline text file reaches the model',flush=True)
for syntax,definition in [('regex','READY'),('lark','start: "READY"')]:
    message=run('custom '+syntax,[{'role':'user','content':'Call emit with READY as its input.'}],
        tools=[{'type':'custom','custom':{'name':'emit','description':'Emit a readiness marker','format':{'type':'grammar','grammar':{'syntax':syntax,'definition':definition}}}}],
        tool_choice={'type':'custom','custom':{'name':'emit'}},parallel_tool_calls=False)
    calls=message['tool_calls']
    assert len(calls)==1 and calls[0]['type']=='custom' and calls[0]['custom']=={'name':'emit','input':'READY'},message
    print('PASS: constrained custom '+syntax+' tool',flush=True)
schema={'type':'object','properties':{'code':{'type':'string','pattern':'^[A-Z]{3}$'},'value':{'type':'number','minimum':1,'maximum':2,'multipleOf':0.25}},'required':['code','value'],'additionalProperties':False}
message=run('strict schema',[{'role':'user','content':'Return code ABC and value 1.25.'}],response_format={'type':'json_schema','json_schema':{'name':'integration','strict':True,'schema':schema}})
value=json.loads(message['content'])
assert value=={'code':'ABC','value':1.25},message
print('PASS: strict schema with pattern and decimal multiple',flush=True)
