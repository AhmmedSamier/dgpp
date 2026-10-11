from pathlib import Path
import subprocess
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp');objs=[];names=['swiglu_lut16','swiglu_lut8']
for name in names:
 s=(out/'prefill-up-swiglu.cu').read_text().replace('dgpp_up_swiglu','dgpp_'+name)
 if name.endswith('8'):s=s.replace('launch_tiles<OutT, 3, 32>','launch_tiles<OutT, 3, 64>')
 s=s.replace('namespace {','''namespace {
__device__ float silu_table[65536];
__global__ void initialize_silu_table() {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const float g = dgpp::bf16_bits_to_float(uint16_t(i));
  silu_table[i] = g / (1.0f + expf(-g));
}
''',1)
 s=s.replace('const float sx = gx / (1.0f + expf(-gx));','const float sx = silu_table[out[index]];').replace('const float sy = gy / (1.0f + expf(-gy));','const float sy = silu_table[out[index + 1]];')
 needle='  launch<uint16_t>(act, act_stride, w, out, m, n, k, stream, out_stride);'
 assert needle in s
 s=s.replace(needle,'''  static const bool initialized = [] {initialize_silu_table<<<256,256>>>();DGPP_CUDA_OK(cudaDeviceSynchronize());return true;}();
  (void)initialized;
'''+needle)
 p=out/f'prefill-{name}.cu';p.write_text(s);obj=p.with_suffix('.o');objs.append(str(obj))
 subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-ffp-contract=off','-Xptxas=-v','-c',str(p),'-o',str(obj)],check=True)
s=(out/'prefill_swiglu_probe.cu').read_text();s=s.replace('for(int world: {2,1,4})','for(int world: {2,1,4})')
a=s.index('namespace dgpp_up_swiglu');b=s.index('\nnamespace dgpp {',a)
s=s[:a]+'\n'.join(f'namespace dgpp_{n} {{void launch_fp4w_gemm_bf16(const uint16_t*,size_t,const GlmFp4Matrix&,uint16_t*,int,int,int,cudaStream_t,size_t);}}' for n in names)+s[b:]
a=s.index('  for(int rep=0;rep<3;rep++)');b=s.index('  for(void* x:',a)
block=s[a:b];s=s[:a]+''.join(block.replace('dgpp_up_swiglu',f'dgpp_{n}').replace('printf("world=',f'printf("variant={n} world=') for n in names)+s[b:]
p=out/'prefill_swiglu_lut_probe.cu';p.write_text(s)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr',str(p),*objs,str(root/'build-pr95/CMakeFiles/dgpp_models_qwen_layers.dir/src/models/qwen/model35_kernels.cu.o'),'-o',str(out/'prefill_swiglu_lut_probe'),str(root/'build-pr95/libdgpp_kernels.a'),str(root/'build-pr95/libdgpp_core.a'),str(root/'build-pr95/libdgpp_rope_scaling.a'),'-lcublasLt','-lcudart','-lpthread','-ldl'],check=True)
