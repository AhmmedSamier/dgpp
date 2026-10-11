from pathlib import Path
import subprocess,re
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp');base=(root/'src/kernels/fp4w_gemm.cu').read_text();objs=[];names=['prefill_pair8','prefill_pair16']
for name in names:
 s=base.replace('namespace dgpp {',f'namespace dgpp_{name} {{\nusing namespace dgpp;\nbool fp4w_gemm_shape_ok(const void*,size_t,int);',1)
 s=s.replace('OutT* __restrict__ out, int m, int n, int k, size_t out_stride)', 'const uint8_t* up_payload, const uint8_t* up_scales, const float* up_global, OutT* __restrict__ out, int m, int n, int k, size_t out_stride)')
 s=s.replace('  const float g = *global;  // one broadcast read: the whole grid\'s divisor','')
 s=s.replace('const int col_base = (warp % (kBN / kWarpN)) * kWarpN;', 'const int col_base = (warp % (kBN / kWarpN)) * kWarpN;\n  const float g = col_base < 64 ? *global : *up_global;')
 s=s.replace('(n + kBN - 1) / kBN','(n + 63) / 64').replace('(in_group / group_rows) * kBN','(in_group / group_rows) * 64')
 s=s.replace('n0 + wrow < n','n0 + wrow % 64 < n')
 s=s.replace('payload + static_cast<size_t>(w_ok ? n0 + wrow : 0)', '(wrow < 64 ? payload : up_payload) + static_cast<size_t>(w_ok ? n0 + wrow % 64 : 0)')
 s=s.replace('scales + static_cast<size_t>(w_ok ? n0 + wrow : 0)', '(wrow < 64 ? scales : up_scales) + static_cast<size_t>(w_ok ? n0 + wrow % 64 : 0)')
 a=s.index('  const int r = lane / 4, cc = (lane % 4) * 2;');b=s.index('\ntemplate <typename OutT, int kGroupM',a)
 s=s[:a]+'''  // All readers must leave the final activation tile before it is reused.
  __syncthreads();
  uint16_t* gate_tile = reinterpret_cast<uint16_t*>(sa);
  const int r = lane / 4, cc = (lane % 4) * 2;
  if (col_base < 64) {
#pragma unroll
    for(int i=0;i<4;i++)
#pragma unroll
      for(int j=0;j<kWarpN/8;j++)
#pragma unroll
        for(int pair=0;pair<2;pair++) {
          const int row=row_base+i*16+r+pair*8;
          const int col=col_base+j*8+cc;
          const float x=__fdiv_rn(acc[i][j][pair*2],g);
          const float y=__fdiv_rn(acc[i][j][pair*2+1],g);
          const uint16_t ox=__bfloat16_as_ushort(__float2bfloat16_rn(x));
          const uint16_t oy=__bfloat16_as_ushort(__float2bfloat16_rn(y));
          const int index=row*64+(col^((row&7)*8));
          *reinterpret_cast<uint32_t*>(gate_tile+index)=uint32_t(ox)|(uint32_t(oy)<<16);
        }
  }
  __syncthreads();
  if(col_base >= 64) {
    const bool packed=(reinterpret_cast<uintptr_t>(out)&3u)==0 && (out_stride&1u)==0;
#pragma unroll
    for(int i=0;i<4;i++)
#pragma unroll
      for(int j=0;j<kWarpN/8;j++)
#pragma unroll
        for(int pair=0;pair<2;pair++) {
          const int row=row_base+i*16+r+pair*8;
          const int col=col_base-64+j*8+cc;
          if(m0+row<m && n0+col<n) {
            const int index=row*64+(col^((row&7)*8));
            const float gx=bf16_bits_to_float(gate_tile[index]);
            const float gy=bf16_bits_to_float(gate_tile[index+1]);
            const float ux=bf16_bits_to_float(__bfloat16_as_ushort(__float2bfloat16_rn(__fdiv_rn(acc[i][j][pair*2],g))));
            const float uy=bf16_bits_to_float(__bfloat16_as_ushort(__float2bfloat16_rn(__fdiv_rn(acc[i][j][pair*2+1],g))));
            const uint16_t ox=float_to_bf16_bits((gx/(1.0f+expf(-gx)))*ux);
            const uint16_t oy=float_to_bf16_bits((gy/(1.0f+expf(-gy)))*uy);
            OutT* dest=out+size_t(m0+row)*out_stride+n0+col;
            if(packed && n0+col+1<n) *reinterpret_cast<uint32_t*>(dest)=uint32_t(ox)|(uint32_t(oy)<<16);
            else {dest[0]=ox;if(n0+col+1<n)dest[1]=oy;}
          }
        }
  }
}
''' + s[b:]
 # Propagate a second matrix through every host helper and the public launcher.
 s=s.replace('const GlmFp4Matrix& w, OutT* out', 'const GlmFp4Matrix& w, const GlmFp4Matrix& up, OutT* out')
 s=re.sub(r'const GlmFp4Matrix& w,\s+uint16_t\* out', 'const GlmFp4Matrix& w, const GlmFp4Matrix& up, uint16_t* out',s)
 s=s.replace('w.global_scale, out, m, n,','w.global_scale, up.payload, up.scales, up.global_scale, out, m, n,')
 s=s.replace('(act, act_stride, w, out, m, n, k, stream, out_stride)', '(act, act_stride, w, up, out, m, n, k, stream, out_stride)')
 a=s.index('  if (m >= 3072 && (n == 8704 || k == 8704))');b=s.index('\n}',a)
 wn=64 if name.endswith('8') else 32
 s=s[:a]+f'  launch_tiles<OutT, 3, {wn}>(act, act_stride, w, up, out, m, n, k, stream, out_stride);'+s[b:]
 p=out/f'{name}.cu';p.write_text(s);obj=p.with_suffix('.o');objs.append(str(obj))
 subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-ffp-contract=off','-Xptxas=-v','-c',str(p),'-o',str(obj)],check=True)
s=(out/'prefill_swiglu_probe.cu').read_text();s=s.replace('for(int world: {2,1,4})','for(int world: {2,1,4})')
a=s.index('namespace dgpp_up_swiglu');b=s.index('\nnamespace dgpp {',a)
s=s[:a]+'\n'.join(f'namespace dgpp_{n} {{void launch_fp4w_gemm_bf16(const uint16_t*,size_t,const GlmFp4Matrix&,const GlmFp4Matrix&,uint16_t*,int,int,int,cudaStream_t,size_t);}}' for n in names)+s[b:]
a=s.index('  for(int rep=0;rep<3;rep++)');b=s.index('  for(void* x:',a)
block=s[a:b].replace('launch_fp4w_gemm_bf16(a,k,matrix,out,m,n,k,stream);dgpp_up_swiglu::launch_fp4w_gemm_bf16(a,k,matrix,out,m,n,k,stream,0);','dgpp_up_swiglu::launch_fp4w_gemm_bf16(a,k,matrix,matrix,out,m,n,k,stream,0);')
s=s[:a]+''.join(block.replace('dgpp_up_swiglu',f'dgpp_{n}').replace('printf("world=',f'printf("variant={n} world=') for n in names)+s[b:]
p=out/'prefill_pair_probe.cu';p.write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr',str(p),*objs,str(root/'build-pr95/CMakeFiles/dgpp_models_qwen_layers.dir/src/models/qwen/model35_kernels.cu.o'),'-o',str(out/'prefill_pair_probe'),str(root/'build-pr95/libdgpp_kernels.a'),str(root/'build-pr95/libdgpp_core.a'),str(root/'build-pr95/libdgpp_rope_scaling.a'),'-lcublasLt','-lcudart','-lpthread','-ldl'],check=True)
