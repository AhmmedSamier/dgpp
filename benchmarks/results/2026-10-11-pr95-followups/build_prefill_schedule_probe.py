from pathlib import Path
import subprocess
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp');base=(root/'src/kernels/fp4w_gemm.cu').read_text();sources={};objs=[]
s=base;a=s.index('      uint32_t af[4][4];');b=s.index('      // The next step',a)
frag='''      uint32_t bf[kWarpN / 8][2];
#pragma unroll
      for(int jj=0;jj<kWarpN/16;jj++) {
        const int tile=col_base+jj*16+(lane/16)*8+lane%8;
        const int kb=kk+((lane/8)%2)*8;
        uint32_t q4[4];ldsm_x4(q4,ws+tile*kStride+swizzled_col(tile,kb)*2);
        bf[jj*2][0]=q4[0];bf[jj*2][1]=q4[1];bf[jj*2+1][0]=q4[2];bf[jj*2+1][1]=q4[3];
      }
#pragma unroll
      for(int i=0;i<4;i++) {
        uint32_t af[4];
        ldsm_x4(af,as+(row_base+i*16+lane%16)*kStride+swizzled_col(row_base+i*16+lane%16,kk+(lane/16)*8)*2);
#pragma unroll
        for(int j=0;j<kWarpN/8;j++)
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
              : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[j][0]), "r"(bf[j][1]));
      }
'''
sources['stream_a']=s[:a]+frag+s[b:]
a=s.index('      uint32_t bf[kWarpN / 8][2];');b=s.index('      // The next step',a)
frag='''#pragma unroll
      for(int jj=0;jj<kWarpN/16;jj++) {
        const int tile=col_base+jj*16+(lane/16)*8+lane%8;
        const int kb=kk+((lane/8)%2)*8;
        uint32_t bf[4];ldsm_x4(bf,ws+tile*kStride+swizzled_col(tile,kb)*2);
#pragma unroll
        for(int i=0;i<4;i++) {
#pragma unroll
          for(int j=0;j<2;j++)
            asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc[i][jj*2+j][0]), "+f"(acc[i][jj*2+j][1]), "+f"(acc[i][jj*2+j][2]), "+f"(acc[i][jj*2+j][3])
              : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j*2]), "r"(bf[j*2+1]));
        }
      }
'''
sources['stream_b']=s[:a]+frag+s[b:]
helper='''__device__ __forceinline__ float rounded_bf16_quotient(float x,float g,float recip) {
  const float q = __fmul_rn(x,recip);
  const uint32_t mag = __float_as_uint(q) & 0x7fffffffu;
  const uint32_t rm = __float_as_uint(recip) & 0x7fffffffu;
  const uint32_t tail = mag & 0xffffu;
  if(rm>=0x00800000u && rm<0x7f800000u && mag>=0x00800000u && mag<0x7f7f0000u && (tail<0x7ffcu || tail>0x8004u))return q;
  return __fdiv_rn(x,g);
}
'''
for name,s in [('div_guard',base),('stream_a_div',sources['stream_a']),('stream_b_div',sources['stream_b'])]:
 s=s.replace('template <typename OutT, int kGroupM, int kWarpN>\n__global__',helper+'\ntemplate <typename OutT, int kGroupM, int kWarpN>\n__global__',1)
 s=s.replace('const float g = *global;', 'const float g = *global; const float recip = __frcp_rn(g);')
 for offset,var in [('', 'x'),(' + 1','y')]:
  old=f'const float {var} = __fdiv_rn(acc[i][j][pair * 2{offset}], g);'
  new=f'const float {var} = std::is_same_v<OutT,float> ? __fdiv_rn(acc[i][j][pair * 2{offset}],g) : rounded_bf16_quotient(acc[i][j][pair * 2{offset}],g,recip);'
  assert old in s;s=s.replace(old,new)
 sources[name]=s
for name,s in sources.items():
 s=s.replace('namespace dgpp {',f'namespace dgpp_{name} {{\nusing namespace dgpp;\nbool fp4w_gemm_shape_ok(const void*,size_t,int);',1)
 p=out/f'prefill-schedule-{name}.cu';p.write_text(s);obj=p.with_suffix('.o');objs.append(str(obj))
 subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-ffp-contract=off','-Xptxas=-v','-c',str(p),'-o',str(obj)],check=True)
# Reuse the strictly bitwise and repeated timing harness with these namespaces.
s=(out/'build_prefill_convert_probe.py').read_text();s=s[s.index("s=(out/'tile_probe.cu')"):];s=s.replace('prefill_convert_probe','prefill_schedule_probe')
exec(compile(s,'probe_harness','exec'),dict(out=out,root=root,variants=list(sources),objs=objs,subprocess=subprocess))
