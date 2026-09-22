import json,os,subprocess,re
paths=['/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcudart.so.13','/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcublasLt.so.13','/lib/aarch64-linux-gnu/libcuda.so.1','/lib/aarch64-linux-gnu/libibverbs.so.1','/lib/aarch64-linux-gnu/libstdc++.so.6','/lib/aarch64-linux-gnu/libgcc_s.so.1','/lib/aarch64-linux-gnu/libc.so.6','/lib/aarch64-linux-gnu/libnl-route-3.so.200','/lib/aarch64-linux-gnu/libnl-3.so.200']
r={'libraries':{},'environment':{k:os.environ.get(k) for k in ['LD_LIBRARY_PATH','CUDA_DEVICE_MAX_CONNECTIONS','CUDA_VISIBLE_DEVICES','DGPP_DENSE_GEMV_ROWS','DGPP_ROCE_DEVICES','DGPP_BUS_CHUNK_BYTES']},'links':{}}
for p in paths:
 s=subprocess.run(['readelf','-n',p],capture_output=True,text=True,timeout=15)
 r['libraries'][p]={'resolved':os.path.realpath(p),'build_ids':re.findall(r'Build ID: (\S+)',s.stdout),'returncode':s.returncode,'stderr':s.stderr}
for dev in ['enp1s0f0np0','enP2p1s0f0np0']:
 r['links'][dev]={}
 for flags in [[],['--show-fec']]:
  s=subprocess.run(['ethtool',*flags,dev],capture_output=True,text=True,timeout=10)
  r['links'][dev][' '.join(flags) or 'link']={'returncode':s.returncode,'stdout':s.stdout,'stderr':s.stderr}
print(json.dumps(r,indent=2))
