from pathlib import Path
import subprocess
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp');s=(root/'src/kernels/fp4w_gemm.cu').read_text()
s=s.replace('namespace dgpp {','namespace dgpp_up_swiglu {\nusing namespace dgpp;\nbool fp4w_gemm_shape_ok(const void*,size_t,int);',1)
a=s.index('            if (packed && col + 1 < n)');b=s.index('\n          }',a)
s=s[:a]+'''            const float gx = bf16_bits_to_float(out[index]);
            const float ux = bf16_bits_to_float(__bfloat16_as_ushort(__float2bfloat16_rn(x)));
            const float sx = gx / (1.0f + expf(-gx));
            const uint16_t ox = float_to_bf16_bits(sx * ux);
            if (col + 1 < n) {
              const float gy = bf16_bits_to_float(out[index + 1]);
              const float uy = bf16_bits_to_float(__bfloat16_as_ushort(__float2bfloat16_rn(y)));
              const float sy = gy / (1.0f + expf(-gy));
              const uint16_t oy = float_to_bf16_bits(sy * uy);
              if (packed) *reinterpret_cast<uint32_t*>(out + index) = uint32_t(ox) | (uint32_t(oy) << 16);
              else {out[index] = ox;out[index + 1] = oy;}
            } else out[index] = ox;'''+s[b:]
p=out/'prefill-up-swiglu.cu';p.write_text(s);obj=p.with_suffix('.o')
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-ffp-contract=off','-Xptxas=-v','-c',str(p),'-o',str(obj)],check=True)
s=(out/'tile_probe.cu').read_text();a=s.index('namespace dgpp_m320k32');b=s.index('int main()',a)
s=s[:a]+'''namespace dgpp_up_swiglu {void launch_fp4w_gemm_bf16(const uint16_t*,size_t,const GlmFp4Matrix&,uint16_t*,int,int,int,cudaStream_t,size_t);}
namespace dgpp {void qwen35_swiglu_bf16(const uint16_t*,const uint16_t*,uint16_t*,int64_t,cudaStream_t);}
'''+s[b:]
s=s.replace('for(int world: {1,2,4})for(bool down:{false,true})for(int m:{1023,2048,4096})','for(int world: {2,1,4})for(bool down:{false})for(int m:{1023,2048,4096})')
s=s.replace('auto original=[&](){launch_fp4w_gemm_bf16(a,k,matrix,base,m,n,k,stream);};','auto original=[&](){launch_fp4w_gemm_bf16(a,k,matrix,base,m,n,k,stream);launch_fp4w_gemm_bf16(a,k,matrix,out,m,n,k,stream);qwen35_swiglu_bf16(base,out,base,mn,stream);};')
a=s.index('  {float ms=time');b=s.index('  for(void* x:',a)
s=s[:a]+'''  for(int rep=0;rep<3;rep++) {
    orig=time(original);cudaMemcpy(hbase.data(),base,mn*2,cudaMemcpyDeviceToHost);
    const float ms=time([&](){launch_fp4w_gemm_bf16(a,k,matrix,out,m,n,k,stream);dgpp_up_swiglu::launch_fp4w_gemm_bf16(a,k,matrix,out,m,n,k,stream,0);});
    cudaMemcpy(hout.data(),out,mn*2,cudaMemcpyDeviceToHost);
    size_t mismatch=0;for(size_t i=0;i<mn;i++)mismatch+=hbase[i]!=hout[i];
    const float after=time(original);
    printf("world=%d m=%d repeat=%d baseline=%.6f candidate=%.6f after=%.6f change=%+.3f%% mismatch=%zu\\n",world,m,rep,orig,ms,after,100*(ms/std::min(orig,after)-1),mismatch);fflush(stdout);if(mismatch)return 1;
  }
'''+s[b:]
p=out/'prefill_swiglu_probe.cu';p.write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr',str(p),str(obj),str(root/'build-pr95/CMakeFiles/dgpp_models_qwen_layers.dir/src/models/qwen/model35_kernels.cu.o'),'-o',str(out/'prefill_swiglu_probe'),str(root/'build-pr95/libdgpp_kernels.a'),str(root/'build-pr95/libdgpp_core.a'),str(root/'build-pr95/libdgpp_rope_scaling.a'),'-lcublasLt','-lcudart','-lpthread','-ldl'],check=True)
