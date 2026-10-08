"""Summarize the last request burst; kernel sums can include overlap."""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys

for arg in sys.argv[1:]:
    path = Path(arg)
    with sqlite3.connect(path) as db:
        rows = db.execute('SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()
        resources = db.execute('SELECT DISTINCT s.value,k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory,k.localMemoryPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE "%attn_flash_kernel%"').fetchall()
    first = 0
    for i in range(1,len(rows)):
        if rows[i][0]-rows[i-1][1]>100_000_000:
            first=i
    win=rows[first:]
    by=defaultdict(lambda:[0.,0]); categories=defaultdict(float)
    for start,end,name in win:
        ms=(end-start)/1e6
        by[name][0]+=ms; by[name][1]+=1
        if 'bus_' in name: cat='collectives'
        elif any(s in name for s in ('select_', 'score_keys', 'score_kernel')): cat='index_scores_and_selection'
        elif any(s in name for s in ('attn_', 'absorb_q', 'vout_')): cat='attention'
        elif 'packq_gemm' in name: cat='packed_gemm'
        elif any(s in name for s in ('scale_gemm_kernel', 'fp8w_gemm_kernel')): cat='fp8_projection_gemm'
        elif 'moe_grouped_mma_fp4' in name: cat='fp4_moe'
        elif any(s in name for s in ('moe_mxfp4', 'fp4_gemm')): cat='fp4_gemm'
        else: cat='other'
        categories[cat]+=ms
    result={'profile':str(path),'scope':'Last kernel burst after >100ms gap; startup/calibration excluded; includes one-token output. Profiled diagnostic, not a published timing.', 'wall_ms':(max(e for _,e,_ in win)-win[0][0])/1e6,'summed_kernel_ms':sum(v[0] for v in by.values()),'categories_ms':dict(categories),'kernels':[dict(name=k,ms=v[0],calls=v[1]) for k,v in sorted(by.items(),key=lambda x:-x[1][0])]}
    (path.parent/'profile-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (path.parent/'attention-kernel-resources.json').write_text(json.dumps([dict(name=r[0],registers_per_thread=r[1],static_shared_bytes=r[2],dynamic_shared_bytes=r[3],local_bytes_per_thread=r[4]) for r in resources],indent=2)+'\n')
    print(path.parent, result['wall_ms'],dict(categories))
