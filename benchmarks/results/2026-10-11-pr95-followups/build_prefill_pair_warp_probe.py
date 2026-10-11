from pathlib import Path
import subprocess
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp')
s=(out/'prefill_pair16.cu').read_text().replace('dgpp_prefill_pair16','dgpp_prefill_pair_warp')
s=s.replace('const int col_base = (warp % (kBN / kWarpN)) * kWarpN;', 'const int col_base = (warp % (kBN / kWarpN)) * (kWarpN / 2);')
s=s.replace('const float g = col_base < 64 ? *global : *up_global;', 'const float g = *global;')
s=s.replace('const int tile = col_base + jj * 16 + (lane / 16) * 8 + lane % 8;', 'const int tile = col_base + jj * 64 + (lane / 16) * 8 + lane % 8;')
a=s.index('  // All readers must leave');b=s.index('\ntemplate <typename OutT, int kGroupM',a)
s=s[:a]+'''  const float up_g = *up_global;
  const int r=lane/4,cc=(lane%4)*2;
  const bool packed=(reinterpret_cast<uintptr_t>(out)&3u)==0 && (out_stride&1u)==0;
#pragma unroll
  for(int i=0;i<4;i++)
#pragma unroll
    for(int j=0;j<2;j++)
#pragma unroll
      for(int pair=0;pair<2;pair++) {
        const int row=row_base+i*16+r+pair*8;
        const int col=col_base+j*8+cc;
        if(m0+row<m && n0+col<n) {
          const float gx=bf16_bits_to_float(__bfloat16_as_ushort(__float2bfloat16_rn(__fdiv_rn(acc[i][j][pair*2],g))));
          const float gy=bf16_bits_to_float(__bfloat16_as_ushort(__float2bfloat16_rn(__fdiv_rn(acc[i][j][pair*2+1],g))));
          const float ux=bf16_bits_to_float(__bfloat16_as_ushort(__float2bfloat16_rn(__fdiv_rn(acc[i][j+2][pair*2],up_g))));
          const float uy=bf16_bits_to_float(__bfloat16_as_ushort(__float2bfloat16_rn(__fdiv_rn(acc[i][j+2][pair*2+1],up_g))));
          const uint16_t ox=float_to_bf16_bits((gx/(1.0f+expf(-gx)))*ux);
          const uint16_t oy=float_to_bf16_bits((gy/(1.0f+expf(-gy)))*uy);
          OutT* dest=out+size_t(m0+row)*out_stride+n0+col;
          if(packed && n0+col+1<n) *reinterpret_cast<uint32_t*>(dest)=uint32_t(ox)|(uint32_t(oy)<<16);
          else {dest[0]=ox;if(n0+col+1<n)dest[1]=oy;}
        }
      }
}
'''+s[b:]
p=out/'prefill_pair_warp.cu';p.write_text(s);obj=p.with_suffix('.o')
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-ffp-contract=off','-Xptxas=-v','-c',str(p),'-o',str(obj)],check=True)
s=(out/'prefill_pair_distinct.cu').read_text()
# Compare the existing two-kernel production pipeline against the warp-paired candidate.
a=s.index('namespace dgpp_prefill_pair8');b=s.index('\nnamespace dgpp {',a)
s=s[:a]+'''namespace dgpp_prefill_pair_warp {void launch_fp4w_gemm_bf16(const uint16_t*,size_t,const GlmFp4Matrix&,const GlmFp4Matrix&,uint16_t*,int,int,int,cudaStream_t,size_t);}'''+s[b:]
a=s.index('  for(int rep=0;rep<3;rep++)');b=s.index('  for(void* x:',a)
block=s[a:b];cut=block.index('  for(int rep=0;rep<3;rep++)',5);block=block[:cut]
block=block.replace('dgpp_prefill_pair8','dgpp_prefill_pair_warp').replace('variant=prefill_pair8','variant=prefill_pair_warp')
s=s[:a]+block+'  cudaFree(p_up);cudaFree(s_up);cudaFree(g_up);\n'+s[b:]
p=out/'prefill_pair_warp_probe.cu';p.write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr',str(p),str(obj),str(root/'build-pr95/CMakeFiles/dgpp_models_qwen_layers.dir/src/models/qwen/model35_kernels.cu.o'),'-o',str(out/'prefill_pair_warp_probe'),str(root/'build-pr95/libdgpp_kernels.a'),str(root/'build-pr95/libdgpp_core.a'),str(root/'build-pr95/libdgpp_rope_scaling.a'),'-lcublasLt','-lcudart','-lpthread','-ldl'],check=True)
