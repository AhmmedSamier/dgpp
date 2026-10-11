from pathlib import Path
import subprocess
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp');base=(root/'src/kernels/dense_fp4_decode.cu').read_text();variants=[];objects=[]
for bm,stages in ((32,3),(64,3),(96,2),(128,2)):
 name=f'm{bm}s{stages}';variants.append(name)
 s=base.replace('BM = 16',f'BM = {bm}').replace('kStages = 3',f'kStages = {stages}').replace('__launch_bounds__(kThreads, 2)','__launch_bounds__(kThreads, 1)')
 s=s.replace('m <= BM','m <= 1023')
 s=s.replace('const dim3 grid((n + BN - 1) / BN, Pair ? 2 : 1);','const int m_tiles = (m + BM - 1) / BM;\n  const dim3 grid(((n + BN - 1) / BN) * m_tiles, Pair ? 2 : 1);')
 s=s.replace('out, up, n, n, k, 1);','out, up, n, n, k, m_tiles);')
 s=s.replace('namespace dgpp {',f'namespace dgpp_tail_{name} {{\nusing namespace dgpp;',1)
 p=out/f'prefill-tail-{name}.cu';p.write_text(s);obj=p.with_suffix('.o');objects.append(str(obj))
 subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-ffp-contract=off','-Xptxas=-v','-c',str(p),'-o',str(obj)],check=True)
source='''#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <string>
#include "common/cuda_check.hpp"
#include "kernels/glm_moe_launch.hpp"
using namespace dgpp;
'''+ '\n'.join(f'namespace dgpp_tail_{n} {{bool try_dense_fp4_decode_bf16(const uint16_t*,size_t,const MoeSegment*,const MoeExpertView*,int,uint16_t*,int,int,int,cudaStream_t,int);}}' for n in variants)+'''
int main(){
 cudaStream_t stream;DGPP_CUDA_OK(cudaStreamCreate(&stream));
 printf("variant,down,rows,repeat,baseline_us,candidate_us,mismatch\\n");
 for(int down:{0,1}) {
  const int n=down?5120:8704,k=down?8704:5120;const size_t nk=size_t(n)*k,mn=size_t(1023)*n,mk=size_t(1023)*k;
  uint8_t *p,*s;uint16_t *a,*gold,*test;float*g;MoeSegment*seg;MoeExpertView*view;
  DGPP_CUDA_OK(cudaMalloc(&p,nk/2));DGPP_CUDA_OK(cudaMalloc(&s,nk/16));DGPP_CUDA_OK(cudaMalloc(&a,mk*2));
  DGPP_CUDA_OK(cudaMalloc(&gold,mn*2));DGPP_CUDA_OK(cudaMalloc(&test,mn*2));DGPP_CUDA_OK(cudaMalloc(&g,4));
  DGPP_CUDA_OK(cudaMalloc(&seg,sizeof(MoeSegment)));DGPP_CUDA_OK(cudaMalloc(&view,sizeof(MoeExpertView)));
  std::vector<uint8_t> hp(nk/2),hs(nk/16);std::vector<uint16_t> ha(mk),ref(mn),got(mn);
  uint32_t rng=7;auto rnd=[&](){rng=rng*1664525u+1013904223u;return rng;};
  for(auto&v:hp)v=rnd()>>24;for(auto&v:hs)v=0x30+((rnd()>>24)%24);
  for(auto&v:ha)v=__bfloat16_as_ushort(__float2bfloat16_rn((int(rnd()>>16)-32768)/32768.f));
  float hg=3217.3f;
  DGPP_CUDA_OK(cudaMemcpy(p,hp.data(),hp.size(),cudaMemcpyHostToDevice));DGPP_CUDA_OK(cudaMemcpy(s,hs.data(),hs.size(),cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(a,ha.data(),ha.size()*2,cudaMemcpyHostToDevice));DGPP_CUDA_OK(cudaMemcpy(g,&hg,4,cudaMemcpyHostToDevice));
  const GlmFp4Matrix matrix{p,s,g,n,k,16};const auto hv=MoeExpertView::of(matrix);DGPP_CUDA_OK(cudaMemcpy(view,&hv,sizeof(hv),cudaMemcpyHostToDevice));
  auto timing=[&](auto fn){fn();DGPP_CUDA_OK(cudaStreamSynchronize(stream));cudaEvent_t x,y;cudaEventCreate(&x);cudaEventCreate(&y);cudaEventRecord(x,stream);for(int i=0;i<20;i++)fn();cudaEventRecord(y,stream);cudaEventSynchronize(y);float ms;cudaEventElapsedTime(&ms,x,y);cudaEventDestroy(x);cudaEventDestroy(y);return ms*50;};
  for(int m:{17,32,46,64,107,128,237,256,319,341,512,1023}) {
   const MoeSegment hm{0,m,0};DGPP_CUDA_OK(cudaMemcpy(seg,&hm,sizeof(hm),cudaMemcpyHostToDevice));
   auto baseline=[&](){launch_dense_mma_fp4_prod_bf16(a,k,seg,view,0,gold,m,n,k,stream,16);};
   for(int rep=0;rep<3;rep++) {
    const float base=timing(baseline);DGPP_CUDA_OK(cudaMemcpy(ref.data(),gold,size_t(m)*n*2,cudaMemcpyDeviceToHost));
'''+ ''.join(f'''    {{const float time=timing([&](){{if(!dgpp_tail_{name}::try_dense_fp4_decode_bf16(a,k,seg,view,0,test,m,n,k,stream,16))throw "bad shape";}});
    DGPP_CUDA_OK(cudaMemcpy(got.data(),test,size_t(m)*n*2,cudaMemcpyDeviceToHost));
    size_t mismatch=0;for(size_t i=0;i<size_t(m)*n;i++)mismatch+=ref[i]!=got[i];
    printf("{name},%d,%d,%d,%.6f,%.6f,%zu\\n",down,m,rep,base,time,mismatch);fflush(stdout);if(mismatch)return 1;}}
''' for name in variants)+'''
   }
  }
  for(void* ptr:{(void*)p,(void*)s,(void*)a,(void*)gold,(void*)test,(void*)g,(void*)seg,(void*)view})cudaFree(ptr);
 }
 cudaStreamDestroy(stream);
}
'''
p=out/'prefill_tail_probe.cu';p.write_text(source)
subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr',str(p),*objects,'-o',str(out/'prefill_tail_probe'),str(root/'build-pr95/libdgpp_kernels.a'),str(root/'build-pr95/libdgpp_core.a'),str(root/'build-pr95/libdgpp_rope_scaling.a'),'-lcublasLt','-lcudart','-lpthread','-ldl'],check=True)
