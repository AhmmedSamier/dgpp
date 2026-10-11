from pathlib import Path
import subprocess
out=Path(__file__).parent;root=Path('/home/stephen/workspace/dgpp')
subprocess.run(['/usr/local/cuda/bin/nvcc','-std=c++20','-O3','-arch=sm_121a','-I'+str(root/'src'),'-I'+str(root),'--expt-relaxed-constexpr','--extended-lambda','-c',str(out/'decode_weight_bench.cu'),'-o',str(out/'decode_weight_bench.o')],check=True)
link=(root/'build-pr95/CMakeFiles/moe_tile_bench.dir/link.txt').read_text().split();link[link.index('CMakeFiles/moe_tile_bench.dir/benchmarks/micro/moe_tile_bench.cu.o')]=str(out/'decode_weight_bench.o');link[link.index('-o')+1]=str(out/'decode_weight_bench');subprocess.run(link,cwd=root/'build-pr95',check=True)
