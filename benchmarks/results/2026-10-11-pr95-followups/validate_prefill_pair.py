import datetime,json,os,pathlib,subprocess
out=pathlib.Path(__file__).parent;root=pathlib.Path('/home/stephen/workspace/dgpp');status=[]
env={**os.environ,'CUDA_DEVICE_MAX_CONNECTIONS':'32','DGPP_NO_SWAP':'1','DGPP_ROCE_DEVICES':'rocep1s0f0 roceP2p1s0f0','DGPP_ROCE_GID_INDICES':'3 3'}
commands=[
 ('prefill-pair-tests.log',['ctest','--test-dir','build-pr95','--output-on-failure','-j','1','-R','^(qwen35_|fp4w_gemm_test$|glm_moe_test$)']),
 ('prefill-pair-memcheck.log',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99','--kernel-name','regex=fp4w_gemm_kernel.*Lb1E','--launch-count','7','build-pr95/qwen35_fp4_prefill_test']),
 ('prefill-pair-racecheck.log',['/usr/local/cuda/bin/compute-sanitizer','--tool','racecheck','--error-exitcode','99','--kernel-name','regex=fp4w_gemm_kernel.*Lb1E','--launch-count','7','build-pr95/qwen35_fp4_prefill_test']),
 ('prefill-pair-numerical-controller.log',['python3',str(out/'run_prefill_pair_numerical.py')]),
]
for name,cmd in commands:
    receipt=dict(log=name,command=cmd,start=datetime.datetime.now(datetime.timezone.utc).isoformat())
    print(receipt['start'],name,flush=True)
    with (out/name).open('w') as log:p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    receipt.update(returncode=p.returncode,end=datetime.datetime.now(datetime.timezone.utc).isoformat());status.append(receipt)
    (out/'prefill-pair-validation-status.json').write_text(json.dumps(status,indent=2)+'\n')
    assert p.returncode==0,receipt
