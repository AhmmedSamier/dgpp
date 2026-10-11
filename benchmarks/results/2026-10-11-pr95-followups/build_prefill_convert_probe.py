from pathlib import Path
import subprocess
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp');base=(root/'src/kernels/fp4w_gemm.cu').read_text();objs=[]
variants=['g2','half_bits','half_bits_g2','scale_bits']
for name in variants:
 s=base.replace('namespace dgpp {',f'namespace dgpp_{name} {{\nusing namespace dgpp;\nbool fp4w_gemm_shape_ok(const void*,size_t,int);',1)
 if name in ('g2','half_bits_g2'):s=s.replace('launch_tiles<OutT, 3, 32>','launch_tiles<OutT, 2, 32>')
 if name.startswith('half_bits'):
  a=s.index('  const __nv_bfloat162 b =');b=s.index('\n}',a)
  s=s[:a]+'''  const uint32_t bits = *reinterpret_cast<const uint32_t*>(&p);
  const uint32_t mag = bits & 0x7fff7fffu;
  uint32_t v = ((bits >> 3) & 0x0fff0fffu) +
               ((mag & 0xffffu) ? 0x3800u : 0u) +
               ((mag >> 16) ? 0x38000000u : 0u);
  v |= bits & 0x80008000u;
  if ((mag & 0xffffu) > 0x7c00u) v = (v & 0xffff0000u) | 0x7fffu;
  if ((mag >> 16) > 0x7c00u) v = (v & 0xffffu) | 0x7fff0000u;
  return v;'''+s[b:]
 if name=='scale_bits':
  a=s.index('__device__ __forceinline__ uint32_t decode_pair_bf16_w');b=s.index('\ntemplate <typename OutT',a)
  s=s[:a]+'''__device__ __forceinline__ uint32_t decode_pair_bf16_w(uint32_t byte, uint32_t bases) {
    auto one = [&](uint32_t code) {
      uint32_t mag = code & 7u;
      uint32_t v = (mag >= 3 && (mag & 1)) ? (bases >> 16) : (bases & 0xffffu);
      const uint32_t sign = (v & 0x8000u) ^ ((code & 8u) << 12);
      v &= 0x7fffu;
      if (v > 0x7f80u) return 0x7fffu;
      if (mag && v) v += ((int(mag >> 1) - 1) * 128);
      else v = 0;
      return v | sign;
    };
    return one(byte & 15u) | (one(byte >> 4) << 16);
}
''' + s[b:]
  old='''    const __half2 s2 = __half2half2(
        __half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(sb), __NV_E4M3)));'''
  new='''    const float scale = __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(sb), __NV_E4M3)));
    const uint32_t s2 = uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(scale))) |
                        (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(scale * 1.5f))) << 16);'''
  assert old in s;s=s.replace(old,new)
 p=out/f'prefill-convert-{name}.cu';p.write_text(s);obj=p.with_suffix('.o');objs.append(str(obj))
 subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-ffp-contract=off','-Xptxas=-v','-c',str(p),'-o',str(obj)],check=True)
s=(out/'tile_probe.cu').read_text();a=s.index('namespace dgpp_m320k32');b=s.index('int main()',a)
s=s[:a]+'\n'.join(f'namespace dgpp_{n} {{void launch_fp4w_gemm_bf16(const uint16_t*,size_t,const GlmFp4Matrix&,uint16_t*,int,int,int,cudaStream_t,size_t);}}' for n in variants)+'\n'+s[b:]
s=s.replace('for(int world: {1,2,4})for(bool down:{false,true})for(int m:{1023,2048,4096})','for(int world: {2})for(bool down:{false,true})for(int m:{2048,4096})')
a=s.index('  {float ms=time');b=s.index('  for(void* x:',a)
s=s[:a]+''.join(f'''  for(int rep=0;rep<3;rep++) {{float ms=time([&](){{dgpp_{name}::launch_fp4w_gemm_bf16(a,k,matrix,out,m,n,k,stream,0);}});
  cudaMemcpy(hout.data(),out,mn*2,cudaMemcpyDeviceToHost);
  size_t mismatch=0;for(size_t i=0;i<mn;i++)mismatch+=hbase[i]!=hout[i];
  printf("variant={name} world=%d down=%d m=%d repeat=%d baseline=%.4f variant=%.4f change=%+.2f%% mismatch=%zu\\n",world,down,m,rep,orig,ms,100*(ms/orig-1),mismatch);fflush(stdout);
  if(mismatch)return 1;}}
''' for name in variants)+s[b:]
p=out/'prefill_convert_probe.cu';p.write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr',str(p),*objs,'-o',str(out/'prefill_convert_probe'),str(root/'build-pr95/libdgpp_kernels.a'),str(root/'build-pr95/libdgpp_core.a'),str(root/'build-pr95/libdgpp_rope_scaling.a'),'-lcublasLt','-lcudart','-lpthread','-ldl'],check=True)
