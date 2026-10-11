"""Combine unchanged decode evidence with all-world final-prefill validation."""
import datetime, json, pathlib, subprocess
out=pathlib.Path(__file__).parent;root=pathlib.Path('/home/stephen/workspace/dgpp')
def read(name): return json.loads((out/name).read_text())
old=read('decode-wide-final-matched-audit.json')
wall=read('decode-wall-audit.json')
assert len(wall['rows']) == 60 and wall['wins'] == 60
new=read('prefill-pair-final-audit.json')
context=read('prefill-pair-context-audit.json')
plain=read('decode-release-plain-audit.json')
identity=read('prefill-pair-retained-kernel-audit.json')
quality=read('numerical-prefill-pair-comparison.json')
build=read('prefill-pair-build-manifest.json')
decode_build=read('decode-release-build-manifest.json')
changed=subprocess.check_output(['git','diff','--name-only',decode_build['source_commit'],build['source_commit']],cwd=root,text=True).splitlines()
expected={'CMakeLists.txt','src/kernels/fp4w_gemm.cu','src/kernels/fp4w_gemm.hpp','src/models/qwen/model35.cpp','tests/cuda/fp4w_gemm_test.cu','tests/cuda/qwen35_fp4_prefill_test.cu'}
assert set(changed)==expected,changed
assert identity['decode_and_collective_instructions_identical']
assert new['all_transcripts_match'] and len(new['transcripts'])==36
assert len(new['cold_outputs'])==108 and len(new['context_outputs'])==27
assert quality['pass_gate'] and quality['positions']==19456
prefill=[dict(**r,source_commit=build['source_commit'],binary_sha256=build['binary_sha256'],record_prefix=f"prefill-pair-final-w{r['world']}") for r in new['prefill']]
assert len(prefill)==12 and len(old['decode'])==60
failed16=read('prefill16-final-audit.json')
gate=all(r['nvfp4_faster'] for r in prefill+old['decode']+new['decode']) and context['performance_gate'] and plain['nvfp4_faster']
report=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),worlds=[1,2,4],prefill=prefill,decode=old['decode'],decode_confirmation=new['decode'],performance_gate=gate,wall_performance_gate=wall['wins']==60,wall_record='decode-wall-audit.json',
            decode_source_commit=decode_build['source_commit'],final_source_commit=build['source_commit'],decode_binary_sha256=decode_build['binary_sha256'],final_binary_sha256=build['binary_sha256'],
            earlier_failed_cells=[dict(**r,record='decode-wide-final-matched-audit.json') for r in old['prefill']+old['decode'] if not r['nvfp4_faster']]+[dict(**r,record='prefill16-final-audit.json') for r in failed16['prefill'] if not r['nvfp4_faster']],
            changed_files_since_decode_campaign=changed,decode_and_collective_instructions_identical=True,
            note='The full d12e7e9f decode campaign and both failed prefill records remain unchanged. Later changes affect FP4 prefill only: f5a517c1 adds TP2 long-walk scheduling, then combined gate/up/SwiGLU removes intermediate matrices at >=1024 rows. The final binary retests all 12 cold-prefill cells, all-world prose decode, short transcripts and four-node context decode. Independent kernel tests and 19,456 real-checkpoint full-vocabulary rows plus a fresh-process 4096-row repeat preserve exact results. Every FP8/NVFP4/FP8 bracket uses one executable for both formats.')
(out/'release-combined-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));raise SystemExit(0 if gate else 1)
