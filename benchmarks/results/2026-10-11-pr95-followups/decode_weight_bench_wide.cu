#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "common/cuda_check.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/scale_gemm.hpp"
using namespace dgpp;
constexpr int copies=8, maxm=32;
template<class T>T* alloc(size_t n){T* p=nullptr;DGPP_CUDA_OK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class F>double time_us(F f,int iters){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);for(int i=0;i<copies*2;i++)f(i%copies);DGPP_CUDA_OK(cudaDeviceSynchronize());cudaEventRecord(a);for(int i=0;i<iters;i++)f(i%copies);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);cudaEventDestroy(a);cudaEventDestroy(b);return ms*1000/iters;}
int main(int argc,char**argv){int world=argc>1?atoi(argv[1]):4,iters=argc>2?atoi(argv[2]):160;const int H=5120,I=17408/world,K=std::max(H,I);void* ws;DGPP_CUDA_OK(cudaMalloc(&ws,64<<20));auto act=alloc<uint16_t>(maxm*K);auto gate=alloc<uint16_t>(maxm*K);auto up=alloc<uint16_t>(maxm*K);std::vector<uint16_t>a(maxm*K);for(size_t i=0;i<a.size();i++)a[i]=0x3c00|((i*2654435761u)&127);cudaMemcpy(act,a.data(),a.size()*2,cudaMemcpyHostToDevice);float global=1.7f;auto gs=alloc<float>(1);cudaMemcpy(gs,&global,4,cudaMemcpyHostToDevice);
 auto seg=alloc<MoeSegment>(1);auto views=alloc<MoeExpertView>(copies*3);std::vector<MoeExpertView> hv(copies*3);std::vector<uint8_t*> w8(copies*3),w4(copies*3),s4(copies*3);std::vector<float*> s8(copies*3),sch(copies*3);
 for(int c=0;c<copies;c++)for(int which=0;which<3;which++){int idx=c*3+which,n=which==2?H:I,k=which==2?I:H;size_t nk=size_t(n)*k;w8[idx]=alloc<uint8_t>(nk);w4[idx]=alloc<uint8_t>(nk/2);s4[idx]=alloc<uint8_t>(nk/16);int ns=((n+127)/128)*((k+127)/128);s8[idx]=alloc<float>(ns);sch[idx]=alloc<float>(n);std::vector<float>ones(std::max(ns,n),1.f);cudaMemset(w8[idx],0x38+c,nk);cudaMemset(w4[idx],0x26+c,nk/2);cudaMemset(s4[idx],0x30+c,nk/16);cudaMemcpy(s8[idx],ones.data(),ns*4,cudaMemcpyHostToDevice);cudaMemcpy(sch[idx],ones.data(),n*4,cudaMemcpyHostToDevice);hv[idx].payload=w4[idx];hv[idx].fp4_scales=s4[idx];hv[idx].fp4_global=gs;}
 cudaMemcpy(views,hv.data(),hv.size()*sizeof(MoeExpertView),cudaMemcpyHostToDevice);
 printf("quant,world,rows,operation,repeat,us,useful_weight_bytes,effective_GB_s\n");
 for(int m:{1,4,8,16,32}){MoeSegment hseg{0,m,0};cudaMemcpy(seg,&hseg,sizeof(hseg),cudaMemcpyHostToDevice);
 for(int rep=0;rep<3;rep++)for(int op=0;op<2;op++)for(int form=0;form<3;form++){
  int n=op==0?I:H,k=op==0?H:I;size_t elements=size_t(n)*k;double bytes=elements*(form==0?.5625:1.0)*(op==0?2:1);if(form!=0)bytes+=(form==1?((n+127)/128)*((k+127)/128)*4:n*4)*(op==0?2:1);
  auto launch=[&](int c){int idx=c*3+(op==0?0:2);if(form==0){if(op==0)launch_dense_mma_fp4_gate_up_bf16(act,k,seg,views+c*3,gate,up,m,n,k,nullptr);else launch_dense_mma_fp4_prod_bf16(act,k,seg,views+c*3,2,gate,m,n,k,nullptr);}
   else for(int j=0;j<(op==0?2:1);j++){uint16_t* dst=j?up:gate;if(form==1)launch_scale_gemm_bf16(act,k,w8[idx+j],s8[idx+j],dst,m,n,k,nullptr,0,1,op==0?ws:nullptr,op==0?(64<<20):0);else launch_scale_gemm_grid_bf16(act,k,w8[idx+j],sch[idx+j],dst,m,n,k,nullptr,0,0,([](int v){int x=0;while((1<<x)<v)++x;return x;})(k),true,ws,64<<20);}};
  double us=time_us(launch,iters);printf("%s,%d,%d,%s,%d,%.4f,%.0f,%.4f\n",form==0?"nvfp4":form==1?"fp8_block":"fp8_channel",world,m,op==0?"gate_up":"down",rep,us,bytes,bytes/us/1000);fflush(stdout);
 }}DGPP_CUDA_OK(cudaGetLastError());return 0;}
