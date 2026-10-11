from pathlib import Path
import subprocess
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp');base=(root/'src/kernels/fp4w_gemm.cu').read_text();objs=[];sources={}
for bm,bn in [(512,64),(256,128),(384,64)]:
 name=f'm{bm}n{bn}k32'
 s=base.replace('kBM = 256, kBN = 128, kBK = 64',f'kBM = {bm}, kBN = {bn}, kBK = 32')
 s=s.replace('((row & 7) * 8)','(((row >> 1) & 3) * 8)')
 s=s.replace('a_row[i] = c / 8','a_row[i] = c / 4').replace('a_piece[i] = (c % 8) * 8','a_piece[i] = (c % 4) * 8')
 s=s.replace('wk = (tid % 2) * 32','wk = (tid % 2) * 16')
 s=s.replace('(kWarpN == 64 || tid < 256)',f'tid < {bn*2}')
 s=s.replace('uint4 codes;','uint2 codes;').replace('*reinterpret_cast<const uint4*>(wsrc','*reinterpret_cast<const uint2*>(wsrc').replace('make_uint4(0u, 0u, 0u, 0u)','make_uint2(0u, 0u)')
 s=s.replace('*reinterpret_cast<const uint32_t*>(ssrc','*reinterpret_cast<const uint16_t*>(ssrc')
 a=s.index('    if constexpr (kWarpN == 32)',s.index('auto place_half'));b=s.index('\n  };',a)
 s=s[:a]+f'''    if(tid >= {bn*2})return;
    uint32_t p[4];
    const uint32_t sb = (r.sc >> (8 * (wk / 16))) & 0xFFu;
    const __half2 s2 = __half2half2(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(sb),__NV_E4M3)));
    const uint32_t word = h ? r.codes.y : r.codes.x;
    p[0] = decode_pair_bf16_w(word & 255u, s2);
    p[1] = decode_pair_bf16_w((word >> 8) & 255u, s2);
    p[2] = decode_pair_bf16_w((word >> 16) & 255u, s2);
    p[3] = decode_pair_bf16_w((word >> 24) & 255u, s2);
    uint8_t* dst = sw + (step & 1) * kWeightTileBytes + wrow * kStride + swizzled_col(wrow, wk + h * 8) * 2;
    *reinterpret_cast<uint4*>(dst)=make_uint4(p[0],p[1],p[2],p[3]);'''+s[b:]
 s=s.replace('if (place_next && (kk % 32) == 16) place_half(step + 1, nxt_raw, kk / 32);','if (place_next) place_half(step + 1, nxt_raw, kk / 16);')
 # Restrict the prototype to the 16/12-warp specialization; no production dispatch changes.
 a=s.index('  if (m >= 3072 && (n == 8704 || k == 8704))');b=s.index('\n}',a)
 s=s[:a]+'  launch_tiles<OutT, 3, 32>(act, act_stride, w, out, m, n, k, stream, out_stride);'+s[b:]
 sources[name]=s
for name,s in sources.items():
 s=s.replace('namespace dgpp {',f'namespace dgpp_{name} {{\nusing namespace dgpp;\nbool fp4w_gemm_shape_ok(const void*,size_t,int);',1)
 p=out/f'prefill-aspect-{name}.cu';p.write_text(s);obj=p.with_suffix('.o');objs.append(str(obj))
 subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++20','-arch=sm_121a','-I'+str(root/'src'),'--extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-ffp-contract=off','-Xptxas=-v','-c',str(p),'-o',str(obj)],check=True)
s=(out/'build_prefill_convert_probe.py').read_text();s=s[s.index("s=(out/'tile_probe.cu')"):];s=s.replace('prefill_convert_probe','prefill_aspect_probe')
exec(compile(s,'probe_harness','exec'),dict(out=out,root=root,variants=list(sources),objs=objs,subprocess=subprocess))
