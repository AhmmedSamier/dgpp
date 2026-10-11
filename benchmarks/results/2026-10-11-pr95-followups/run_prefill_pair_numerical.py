"""Real-checkpoint full-vocabulary regression, including a fresh-process repeat."""
from pathlib import Path
import csv, datetime, hashlib, json, os, shlex, subprocess
out=Path(__file__).parent
root=Path('/home/stephen/workspace/dgpp');build=root/'build-pr95'
# Compile the already-audited independent scoring driver against current archives.
flags=['-O3','-DNDEBUG','-g0','-std=gnu++20','-fPIC','-ffp-contract=off','-DDGPP_HAVE_IBV=1','-DPCRE2_STATIC','-I'+str(root/'src'),'-I'+str(build/'_deps/pcre2-build'),'-isystem','/usr/local/cuda/targets/sbsa-linux/include']
obj=out/'numerical_prefill_pair.o';binary=out/'numerical-prefill-pair'
assert not binary.exists()
subprocess.run(['c++',*flags,'-c',str(out/'numerical_probe.cpp'),'-o',str(obj)],check=True)
cmd=shlex.split((build/'CMakeFiles/qwen35_forward_test.dir/link.txt').read_text())
cmd[cmd.index('CMakeFiles/qwen35_forward_test.dir/tests/cuda/qwen35_forward_test.cpp.o')]=str(obj)
cmd[cmd.index('-o')+1]=str(binary)
subprocess.run(cmd,cwd=build,check=True)
manifest=json.loads((out/'numerical-manifest.json').read_text())
for e in manifest:assert hashlib.sha256(Path(e['path']).read_bytes()).hexdigest()==e['sha256']
checkpoint=Path.home()/'.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/f0b7c9e722f5565102fff8481c99e4d86ae099c7'
receipts=[]
for label,inputs in [('prefill-pair',manifest),('prefill-pair-repeat',manifest[:1])]:
    cmd=[str(binary),str(checkpoint),str(out/label),*[e['path'] for e in inputs]]
    receipt=dict(command=cmd,start=datetime.datetime.now(datetime.timezone.utc).isoformat(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    with (out/('numerical-'+label+'.log')).open('w') as log:
        r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_DEVICE_MAX_CONNECTIONS':'32','DGPP_NO_SWAP':'1'},timeout=1800)
    receipt.update(returncode=r.returncode,end=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipts.append(receipt)
    (out/'numerical-prefill-pair-status.json').write_text(json.dumps(receipts,indent=2)+'\n')
    assert r.returncode==0
rows=[]
for i,e in enumerate(manifest):
    a,b=out/f'baseline-{i}.tsv',out/f'prefill-pair-{i}.tsv'
    aa=list(csv.DictReader(a.open(),delimiter='\t'));bb=list(csv.DictReader(b.open(),delimiter='\t'))
    assert len(aa)==len(bb)==e['tokens']-1
    rows.append(dict(input=e,positions=len(aa),identical_logits_rows=sum(x['logits_hash']==y['logits_hash'] for x,y in zip(aa,bb)),all_fields_identical=aa==bb,baseline_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(b.read_bytes()).hexdigest()))
repeat=(out/'prefill-pair-0.tsv').read_bytes()==(out/'prefill-pair-repeat-0.tsv').read_bytes()
report=dict(cases=rows,positions=sum(r['positions'] for r in rows),repeat_bitwise=repeat,pass_gate=all(r['all_fields_identical'] for r in rows) and repeat,note='Original audited exact implementation versus combined gate/up/SwiGLU; full-vocabulary FP32 logit hashes, target log probabilities and top-1/2 values. Baseline TSV hashes retained for provenance.')
(out/'numerical-prefill-pair-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));raise SystemExit(0 if report['pass_gate'] else 1)
