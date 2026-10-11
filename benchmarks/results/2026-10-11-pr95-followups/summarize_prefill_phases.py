from pathlib import Path
import json,statistics
out=Path(__file__).parent
a=json.loads((out/'prefill-tail-profile-analysis.json').read_text());report={}
for q,f in a['formats'].items():
    report[q]={}
    for n in (8192,32768):
        rs=[]
        for r in f['ranks']['0']:
            if r['requested_tokens']!=n:continue
            full=[c for c in r['chunks'] if c['rows']==4096]
            d=dict(full_chunks=len(full),full_chunk_wall_ms=sum(c['wall_ms'] for c in full),other_chunk_wall_ms=sum(c['wall_ms'] for c in r['chunks'] if c['rows']!=4096),chunk_rows=[c['rows'] for c in r['chunks']])
            for label,pattern in [('fp4w','fp4w_gemm_kernel'),('cublas128x208','nvjet_sm121_tst_mma_128x208'),('dequant','fp8_dequant_blocks_kernel'),('bulk_collective','bus_bulk_collective_kernel'),('full_attention','full_attn_tile_kernel'),('swiglu','qwen35_swiglu_kernel')]:
                d[label]=sum(k['sum_ms'] for c in full for k in c['kernels'] if pattern in k['name'])
            rs.append(d)
        report[q][n]=dict(samples=rs,means={k:statistics.mean(r[k] for r in rs) for k in rs[0] if k!='chunk_rows'})
report['method']='Rank0 CUDA-only f5 capture, means over three matched requests. Full-chunk totals select exactly4096 target rows. Other-chunk wall time includes partial chunks and one completion/speculative tail. Kernel-family totals are not exclusive model-layer attribution or SM utilization.'
(out/'prefill-tail-phase-summary.json').write_text(json.dumps(report,indent=2)+'\n')
