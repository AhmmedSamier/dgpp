from pathlib import Path
import subprocess
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp')
s=(out/'prefill_pair_probe.cu').read_text()
needle='  GlmFp4Matrix matrix{p,s,g,n,k,16};'
new=needle+'''
  uint8_t *p_up,*s_up;float*g_up;
  DGPP_CUDA_OK(cudaMalloc(&p_up,nk/2));DGPP_CUDA_OK(cudaMalloc(&s_up,nk/16));DGPP_CUDA_OK(cudaMalloc(&g_up,4));
  for(auto&v:hp)v=rand()>>24;for(auto&v:hs)v=0x30+((rand()>>24)%24);
  const float up_global=1709.7f;
  DGPP_CUDA_OK(cudaMemcpy(p_up,hp.data(),hp.size(),cudaMemcpyHostToDevice));DGPP_CUDA_OK(cudaMemcpy(s_up,hs.data(),hs.size(),cudaMemcpyHostToDevice));DGPP_CUDA_OK(cudaMemcpy(g_up,&up_global,4,cudaMemcpyHostToDevice));
  GlmFp4Matrix up_matrix{p_up,s_up,g_up,n,k,16};'''
assert needle in s;s=s.replace(needle,new)
s=s.replace('launch_fp4w_gemm_bf16(a,k,matrix,out,m,n,k,stream);qwen35_swiglu', 'launch_fp4w_gemm_bf16(a,k,up_matrix,out,m,n,k,stream);qwen35_swiglu')
s=s.replace('launch_fp4w_gemm_bf16(a,k,matrix,matrix,out,m,n,k,stream,0)', 'launch_fp4w_gemm_bf16(a,k,matrix,up_matrix,out,m,n,k,stream,0)')
s=s.replace('  for(void* x:{(void*)p,', '  cudaFree(p_up);cudaFree(s_up);cudaFree(g_up);\n  for(void* x:{(void*)p,')
p=out/'prefill_pair_distinct.cu';p.write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr',str(p),str(out/'prefill_pair8.o'),str(out/'prefill_pair16.o'),str(root/'build-pr95/CMakeFiles/dgpp_models_qwen_layers.dir/src/models/qwen/model35_kernels.cu.o'),'-o',str(out/'prefill_pair_distinct'),str(root/'build-pr95/libdgpp_kernels.a'),str(root/'build-pr95/libdgpp_core.a'),str(root/'build-pr95/libdgpp_rope_scaling.a'),'-lcublasLt','-lcudart','-lpthread','-ldl'],check=True)
