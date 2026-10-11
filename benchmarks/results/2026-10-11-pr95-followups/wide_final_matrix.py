import datetime,hashlib,json,pathlib,subprocess,os
out=pathlib.Path(__file__).parent;root=pathlib.Path('/home/stephen/workspace/dgpp');status=[]
binary=out/'dgpp-serve-release-final'
assert binary.exists(), 'Use the frozen, validated release executable'
env={**os.environ,'PR95_BIN':str(binary)}
# Matched brackets at all three supported tensor-parallel world sizes.
for world in (4,1,2):
 for suffix,quant in [('fp8a','fp8'),('nvfp4','nvfp4'),('fp8b','fp8')]:
  label=f'decode-wide-final-w{world}-{suffix}';cmd=['python3',str(out/'campaign.py'),quant,str(world),label,'--final']
  if world==4:cmd+=['--decode-context']
  e=dict(label=label,command=cmd,start=datetime.datetime.now(datetime.timezone.utc).isoformat());print(e['start'],label,flush=True)
  with (out/f'{label}-controller.log').open('w') as f:p=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=7200)
  e.update(returncode=p.returncode,end=datetime.datetime.now(datetime.timezone.utc).isoformat());status.append(e);(out/'decode-wide-final-matrix-status.json').write_text(json.dumps(status,indent=2))
  if p.returncode:raise SystemExit(p.returncode)
script=(out/'audit_matched.py').read_text().replace("f'final-w{world}-{suffix}'","f'decode-wide-final-w{world}-{suffix}'").replace("'matched-audit.json'","'decode-wide-final-matched-audit.json'")
(out/'audit_decode_wide_final.py').write_text(script)
with (out/'decode-wide-final-audit.log').open('w') as f:p=subprocess.run(['python3',str(out/'audit_decode_wide_final.py')],cwd=root,stdout=f,stderr=subprocess.STDOUT)
with (out/'decode-context-audit.log').open('w') as f:c=subprocess.run(['python3',str(out/'audit_decode_context.py')],cwd=root,stdout=f,stderr=subprocess.STDOUT)
raise SystemExit(p.returncode or c.returncode)
