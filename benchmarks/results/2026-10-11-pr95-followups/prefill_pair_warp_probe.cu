#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include "common/cuda_check.hpp"
#include "kernels/fp4w_gemm.hpp"
#include "kernels/gemm.hpp"
using namespace dgpp;
__global__ void expand(const uint8_t* p,const uint8_t* s,uint16_t* w,size_t pairs) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=pairs)return;
  __half2 h(__nv_cvt_fp4x2_to_halfraw2(p[i],__NV_E2M1));
  __half scale(__nv_cvt_fp8_to_halfraw(s[i/8],__NV_E4M3));
  __half2 v=__hmul2(h,__half2half2(scale));
  reinterpret_cast<__nv_bfloat162*>(w)[i]=__floats2bfloat162_rn(__low2float(v),__high2float(v));
}
__global__ void finish(const float* x,const float* g,uint16_t* y,size_t count) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count)y[i]=__bfloat16_as_ushort(__float2bfloat16_rn(__fdiv_rn(x[i],*g)));
}
static float f(uint16_t u){return __bfloat162float(__ushort_as_bfloat16(u));}
namespace dgpp_prefill_pair_warp {void launch_fp4w_gemm_bf16(const uint16_t*,size_t,const GlmFp4Matrix&,const GlmFp4Matrix&,uint16_t*,int,int,int,cudaStream_t,size_t);}
namespace dgpp {void qwen35_swiglu_bf16(const uint16_t*,const uint16_t*,uint16_t*,int64_t,cudaStream_t);}
int main(){
 CublasLtGemm gemm;gemm.set_pinned_rows(4096);
 cudaStream_t stream;DGPP_CUDA_OK(cudaStreamCreate(&stream));
 void* ws;DGPP_CUDA_OK(cudaMalloc(&ws,64ull<<20));
 for(int world: {2,1,4})for(bool down:{false})for(int m:{1023,2048,4096}){
  int n=down?5120:17408/world,k=down?17408/world:5120;
  size_t nk=size_t(n)*k,mn=size_t(m)*n,mk=size_t(m)*k;
  uint8_t *p,*s;uint16_t *a,*w,*out,*base;float *g,*tmp;
  DGPP_CUDA_OK(cudaMalloc(&p,nk/2));DGPP_CUDA_OK(cudaMalloc(&s,nk/16));
  DGPP_CUDA_OK(cudaMalloc(&a,mk*2));DGPP_CUDA_OK(cudaMalloc(&w,nk*2));
  DGPP_CUDA_OK(cudaMalloc(&out,mn*2));DGPP_CUDA_OK(cudaMalloc(&base,mn*2));
  DGPP_CUDA_OK(cudaMalloc(&tmp,mn*4));DGPP_CUDA_OK(cudaMalloc(&g,4));
  std::vector<uint8_t> hp(nk/2),hs(nk/16);std::vector<uint16_t> ha(mk);
  uint32_t rng=7;auto rand=[&](){rng=rng*1664525u+1013904223u;return rng;};
  for(auto&v:hp)v=rand()>>24;for(auto&v:hs)v=0x30+((rand()>>24)%24);
  for(auto&v:ha)v=__bfloat16_as_ushort(__float2bfloat16_rn((int(rand()>>16)-32768)/32768.f));
  float hg=3217.3f;
  DGPP_CUDA_OK(cudaMemcpy(p,hp.data(),hp.size(),cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(s,hs.data(),hs.size(),cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(a,ha.data(),ha.size()*2,cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(g,&hg,4,cudaMemcpyHostToDevice));
  GlmFp4Matrix matrix{p,s,g,n,k,16};
  uint8_t *p_up,*s_up;float*g_up;
  DGPP_CUDA_OK(cudaMalloc(&p_up,nk/2));DGPP_CUDA_OK(cudaMalloc(&s_up,nk/16));DGPP_CUDA_OK(cudaMalloc(&g_up,4));
  for(auto&v:hp)v=rand()>>24;for(auto&v:hs)v=0x30+((rand()>>24)%24);
  const float up_global=1709.7f;
  DGPP_CUDA_OK(cudaMemcpy(p_up,hp.data(),hp.size(),cudaMemcpyHostToDevice));DGPP_CUDA_OK(cudaMemcpy(s_up,hs.data(),hs.size(),cudaMemcpyHostToDevice));DGPP_CUDA_OK(cudaMemcpy(g_up,&up_global,4,cudaMemcpyHostToDevice));
  GlmFp4Matrix up_matrix{p_up,s_up,g_up,n,k,16};
  auto original=[&](){launch_fp4w_gemm_bf16(a,k,matrix,base,m,n,k,stream);launch_fp4w_gemm_bf16(a,k,up_matrix,out,m,n,k,stream);qwen35_swiglu_bf16(base,out,base,mn,stream);};
  auto bridge=[&](){expand<<<(nk/2+255)/256,256,0,stream>>>(p,s,w,nk/2);gemm.matmul(a,w,tmp,m,n,k,DType::BF16,GemmOut::F32,k,ws,64ull<<20,stream);finish<<<(mn+255)/256,256,0,stream>>>(tmp,g,out,mn);};
  auto time=[&](auto call){call();DGPP_CUDA_OK(cudaStreamSynchronize(stream));cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);cudaEventRecord(start,stream);for(int i=0;i<10;i++)call();cudaEventRecord(end,stream);cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,start,end);cudaEventDestroy(start);cudaEventDestroy(end);return ms/10;};
  float orig=time(original);
  std::vector<uint16_t> hbase(mn),hout(mn);
  cudaMemcpy(hbase.data(),base,mn*2,cudaMemcpyDeviceToHost);
  for(int rep=0;rep<3;rep++) {
    orig=time(original);cudaMemcpy(hbase.data(),base,mn*2,cudaMemcpyDeviceToHost);
    const float ms=time([&](){dgpp_prefill_pair_warp::launch_fp4w_gemm_bf16(a,k,matrix,up_matrix,out,m,n,k,stream,0);});
    cudaMemcpy(hout.data(),out,mn*2,cudaMemcpyDeviceToHost);
    size_t mismatch=0;for(size_t i=0;i<mn;i++)mismatch+=hbase[i]!=hout[i];
    const float after=time(original);
    printf("variant=prefill_pair_warp world=%d m=%d repeat=%d baseline=%.6f candidate=%.6f after=%.6f change=%+.3f%% mismatch=%zu\n",world,m,rep,orig,ms,after,100*(ms/std::min(orig,after)-1),mismatch);fflush(stdout);if(mismatch)return 1;
  }
  cudaFree(p_up);cudaFree(s_up);cudaFree(g_up);
  for(void* x:{(void*)p,(void*)s,(void*)a,(void*)w,(void*)out,(void*)base,(void*)tmp,(void*)g})cudaFree(x);
 }
 cudaFree(ws);cudaStreamDestroy(stream);
}
