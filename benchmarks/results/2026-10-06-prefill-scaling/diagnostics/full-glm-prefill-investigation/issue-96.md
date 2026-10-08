Investigate full GLM-5.3's remaining long-context decode cost on the four-node GB10 cluster. With MTP1 and FP4 KV, engine throughput falls from 28.82 tok/s at minimal context to 17.62 tok/s at 256K. The measurements below are after the prefill fixes; the amount of avoidable decode overhead is still unknown.

Configuration:

- Model: `HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64` (full GLM-5.3), checkpoint revision `147684fbad20c1e283ddff46fd07cb9d4ccbb3da`.
- Four GB10 nodes, tensor parallelism 4, 200 Gb/s RoCE, CUDA 13.0, driver 580.173.02.
- `kv_dtype=fp4`, `kv_capacity=262144`, `bf16_weights=bf12`, `embed_sharding=vocab`.
- `mtp=true`, fixed `mtp_depth=1`, adaptive scheduling off; CUDA decode graphs enabled.
- Eight configured request slots, one active request; 1.5 GiB prefix cache, full admission, 128 sampling candidates.
- Greedy generation, default reasoning, 256 output tokens. Each context bucket has one cold request and two identical cached repeats. Decode columns are medians over those three requests and exclude prefill; cold prefill is the first request only.

| Context bucket | Actual input tokens | Cold prefill (s) | Decode ms/pass | Engine tok/s | Output tokens/pass |
|---|---:|---:|---:|---:|---:|
| Minimal | 99 | 0.851 | 64.11 | 28.82 | 1.848 |
| 32K | 32,234 | 108.050 | 78.82 | 25.08 | 1.977 |
| 64K | 65,009 | 230.027 | 83.55 | 23.66 | 1.977 |
| 128K | 130,537 | 507.772 | 92.97 | 21.26 | 1.977 |
| 256K | 261,630 | 1,187.740 | 111.35 | 17.62 | 1.962 |

From 32K through 256K, decode adds approximately 4.6–4.7 ms per additional 32K input tokens. Acceptance remains high. The decode indexer scores the visible history and selects up to 2,048 tokens before listed attention; growing history therefore adds work even though the selected attention set is bounded. This is a code-based explanation for the trend, not a measured breakdown of its cost or proof that the implementation is efficient. Relevant entry points: [`DsaLayer::enqueue_decode`](https://github.com/HawkBearPig/dgpp/blob/0b10cbeaed1b7c51d854ec2f3a23729cb7b91c35/src/models/dsa_layer.cu#L812) and [`dsa_select_decode`](https://github.com/HawkBearPig/dgpp/blob/0b10cbeaed1b7c51d854ec2f3a23729cb7b91c35/src/kernels/dsa.cu#L3416).

The final source is `0b10cbeaed1b7c51d854ec2f3a23729cb7b91c35`, following selector fix `a3ebc21a7c245eb736cc799fcf87a2c97ed8f7c9` and score fusion `21b91a39f999cef0c94579a3240b62b07854a80b`. The tested serving binary was built before the final commit and retained unchanged: version `0.1.0+g21b91a39f999.dirty`, SHA-256 `a983cebe9fc1c6fa4623a45cf7ba76eaf4fb7cdb5270179c0e48acba9b482883`; the frozen candidate source matches the final commit.

Every rank reported zero process and cgroup swap throughout the FP4 default launch, with cgroup `memory.swap.max=0` and OS swap left enabled. Memory telemetry gaps were at most 1.094 s. All four operation streams matched, shutdown was clean, and no rank logged a collective-stall warning. No profiler ran during these measurements.

This does not establish a regression from the prefill fixes. A matched FP8-KV comparison at 128K measured 21.974 tok/s before score fusion and 21.931 tok/s with the final version. There is no completed pre-fix 256K FP4 baseline. The previous roughly 27–31 tok/s full-GLM serving results use short prompts. MTP2/MTP3 have been measured only on the short serving workload so far.

Investigation work:

- [ ] Profile decode at matched 32K/64K/128K/256K contexts. Separate index scoring, top-k selection, listed attention/KV reads and unpacking, projections/MoE, draft/verification, and collectives. Keep profiler timings separate from benchmark results.
- [ ] Measure score/selection kernels with realistic full-GLM geometry and warm/cold caches; identify bandwidth, compute, synchronization and launch limits before choosing an optimization.
- [ ] Compare plain decode and MTP1/MTP2/MTP3 at fixed contexts, reporting ms/pass, accepted tokens/pass and engine tok/s.
- [ ] Compare BF16/FP8/FP4 KV at common supported contexts with matched capacity, slots and cache settings where possible. The existing configurations differ in capacity and are not a controlled KV-format comparison.
- [ ] Account for the remaining 256K cold-prefill cost after the existing fixes; investigate further only from a measured component breakdown.
- [ ] Validate any change with selection/numerical checks, model-level output checks, matching rank streams and zero DGPP swap, then repeat unprofiled end-to-end measurements.

Raw evidence is in the ongoing benchmark update, not yet pushed: `benchmarks/results/2026-10-06-prefill-scaling/raw/glm53-fp4kv-256k-w4/refresh/default/` contains the exact config, launch manifest, command receipts, `context-*.json`, memory/rank checks and `collective-observations.json`. The client is `scripts/serve_context_matrix.py`; each context report records the seed, tag and prompt hash. Preserve the current campaign evidence and profile separately on an idle cluster.

Related prior investigation: #19 concerned Qwen QSA and is closed; it does not establish the cause or solution for full GLM.
