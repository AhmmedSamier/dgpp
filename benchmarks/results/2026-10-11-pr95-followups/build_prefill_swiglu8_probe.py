from pathlib import Path
import subprocess
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp')
s=(out/'prefill-up-swiglu.cu').read_text().replace('dgpp_up_swiglu','dgpp_up_swiglu8').replace('launch_tiles<OutT, 3, 32>','launch_tiles<OutT, 3, 64>')
p=out/'prefill-up-swiglu8.cu';p.write_text(s);obj=p.with_suffix('.o')
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-ffp-contract=off','-Xptxas=-v','-c',str(p),'-o',str(obj)],check=True)
s=(out/'prefill_swiglu_probe.cu').read_text().replace('dgpp_up_swiglu','dgpp_up_swiglu8').replace('for(int world: {2,1,4})','for(int world: {2})').replace('for(int rep=0;rep<3;rep++)','for(int rep=0;rep<5;rep++)')
p=out/'prefill_swiglu8_probe.cu';p.write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr',str(p),str(obj),str(root/'build-pr95/CMakeFiles/dgpp_models_qwen_layers.dir/src/models/qwen/model35_kernels.cu.o'),'-o',str(out/'prefill_swiglu8_probe'),str(root/'build-pr95/libdgpp_kernels.a'),str(root/'build-pr95/libdgpp_core.a'),str(root/'build-pr95/libdgpp_rope_scaling.a'),'-lcublasLt','-lcudart','-lpthread','-ldl'],check=True)
