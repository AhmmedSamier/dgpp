import datetime,hashlib,json,pathlib,re,shutil,subprocess
root=pathlib.Path('/home/stephen/workspace/dgpp');out=pathlib.Path(__file__).parent
paths=['src','tests','scripts','tools','include','cmake','CMakeLists.txt','deploy']
assert not subprocess.check_output(['git','diff','--name-only','HEAD','--',*paths],cwd=root,text=True).strip()
validation=json.loads((out/'prefill-pair-validation-status.json').read_text())
assert len(validation)==4 and all(r['returncode']==0 for r in validation)
assert json.loads((out/'numerical-prefill-pair-comparison.json').read_text())['pass_gate']
assert '100% tests passed, 0 tests failed out of 28' in (out/'prefill-pair-tests.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (out/'prefill-pair-memcheck.log').read_text()
assert 'RACECHECK SUMMARY: 0 hazards' in (out/'prefill-pair-racecheck.log').read_text()
binary=out/'dgpp-serve-prefill-pair-final';assert not binary.exists()
shutil.copy2(root/'build-pr95/dgpp-serve',binary)
sha=hashlib.sha256(binary.read_bytes()).hexdigest();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
report=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=commit,binary_sha256=sha,runtime_source_clean=True,build_type='Release',cuda='13.0.88',tests_passed=28,memcheck_errors=0,racecheck_hazards=0,
            sanitizer_scope='First seven paired launches: six partial-tile output alignment/stride layouts and odd K-step pipeline. The entire test executable also completes, including all 65536 BF16 gate inputs and actual TP1/2/4 shapes.',
            note='Combined gate/up/SwiGLU from 1024 rows. Each warp owns matching gate/up columns, preserving the original ascending-k16 FP32 chains, scale divisions, both BF16 roundings and expf/div/mul epilogue. Decode and FP8 model dispatch unchanged; historical f5 adds TP2 long-prefill scheduling.')
(out/'prefill-pair-build-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
cu='/usr/local/cuda/bin/cuobjdump'
def codes(path):
    data=subprocess.check_output([cu,'--dump-sass',str(path)],text=True)
    result={}
    for part in data.split('Function : ')[1:]:
        name=part.splitlines()[0].strip()
        words=re.findall(r'/\* (0x[0-9a-f]+) \*/',part)
        assert words,name
        assert name not in result,name
        result[name]=dict(words=len(words),sha256=hashlib.sha256('\n'.join(words).encode()).hexdigest())
    return result
base=codes(out/'dgpp-serve-release-final');new=codes(binary)
a={k:v for k,v in base.items() if 'fp4w_gemm_kernel' not in k};b={k:v for k,v in new.items() if 'fp4w_gemm_kernel' not in k}
changed=[k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)]
identity=dict(baseline_binary='dgpp-serve-release-final',baseline_sha256=hashlib.sha256((out/'dgpp-serve-release-final').read_bytes()).hexdigest(),candidate_binary=binary.name,candidate_sha256=sha,
              baseline_non_prefill_kernel_count=len(a),candidate_non_prefill_kernel_count=len(b),changed_non_prefill_kernels=changed,
              baseline_non_prefill_sha256=hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest(),candidate_non_prefill_sha256=hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest(),
              decode_and_collective_instructions_identical=a==b,
              note='All SASS encoding words are compared by unchanged symbol, excluding only fp4w_gemm_kernel prefill specializations. Covers dense FP4 decode, attention, drafter, head, collective and other CUDA kernels. Host source diff independently limits behavior changes to FP4 prefill >=1024 rows.')
# Also record retained normal-prefill specializations against f5.
f5=codes(out/'dgpp-serve-prefill-final')
normal=[]
for group,warp in ((3,64),(4,64),(3,32)):
    aa=[v for k,v in f5.items() if f'fp4w_gemm_kernelItLi{group}ELi{warp}E' in k]
    bb=[v for k,v in new.items() if f'fp4w_gemm_kernelItLi{group}ELi{warp}ELb0E' in k]
    assert len(aa)==len(bb)==1
    normal.append(dict(group=group,warp_n=warp,baseline=aa[0],candidate=bb[0],identical=aa==bb))
identity['normal_prefill_against_f5']=normal
(out/'prefill-pair-retained-kernel-audit.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(report,indent=2));print(json.dumps(identity,indent=2));assert a==b
