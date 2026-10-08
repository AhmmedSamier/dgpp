# Full GLM prefill: intermediate score traffic

The 256K FP4 request was interrupted after roughly 25 minutes to investigate its runtime. It did not produce a publishable 256K measurement. The preceding selector fix had removed repeated top-k sorting, but full GLM still spent most indexed-layer time producing and rereading a large matrix of per-head scores.

## Profile

The isolated benchmark uses full GLM-5.3 dimensions, a four-node head slice, 256K synthetic history, and the final 2,048-token prefill chunk. It measures one indexed attention layer; it excludes MoE and collectives. Full GLM has 21 indexed layers and 57 layers that reuse selections.

| Baseline operation | GPU time | Share |
| --- | --- | --- |
| FP8 dot GEMM | 458.797 ms | 40.3% |
| Head reduction into score keys | 361.679 ms | 31.8% |
| Partition merges | 172.058 ms | 15.1% |
| Radix selection | 110.759 ms | 9.7% |
| Attention | 27.908 ms | 2.5% |

At this shape, each indexed layer writes a 64 GiB FP32 score matrix and reads it again. The final selection needs one reduced score per query/pool, rather than 32 head scores. See the [kernel profile](baseline-kernels.csv), [baseline manifest](baseline-manifest.json), and command receipts beside the logs.

## Initial fusion change (`21b91a3`)

- Fuse the FP8 tensor-core dot product and head reduction. Keep head scores in registers and write a 64-bit score/index key.
- Use the existing allocation for wider query tiles. Preserve the existing radix selection, tie ordering, merge, and token expansion.
- Preserve the original cuBLAS arithmetic for single-query tiles and tails. Do not create new single-query tails when widening tiles.
- Enable fusion only for full GLM's unpooled indexer beyond 2,048 pools. Flash keeps its existing path.
- Keep KV format, cache capacity, prefix allocation, slots, memory reserve, and decode unchanged.

The shared materialized-score path entered in [`78c0bb4f6225`](https://github.com/HawkBearPig/dgpp/commit/78c0bb4f622555b43c8e4a3f12b672e2d74d5428) on August 28, 2026. Full GLM adopted it in [`8464e6a7b9bd`](https://github.com/HawkBearPig/dgpp/commit/8464e6a7b9bd74ae1e688bdbe21c23779328c9d1) on September 13, 2026. This follow-up addresses the remaining score traffic after the partitioned-selector fix.

## Layer timings

Three timed iterations after one warmup; identical benchmark geometry and weights. These are layer measurements, not full-model speedups.

| Latent KV format | Baseline | Fused | Speedup |
| --- | --- | --- | --- |
| BF16 | 1,154.657 ms | 263.527 ms | 4.38× |
| FP4 | 1,163.432 ms | 277.254 ms | 4.20× |

Logs: [BF16 baseline](baseline-full-256k-bf16.log), [BF16 fused](fused-final-256k-bf16.log), [FP4 baseline](baseline-full-256k-fp4.log), [FP4 fused](fused-final-256k-fp4.log). The [candidate source manifest](fused-tail-preserving-candidate.json) binds the [patch](fused-tail-preserving-candidate.patch) and test binaries by hash.

## Validation

| Check | Result | Evidence |
| --- | --- | --- |
| Existing DSA suite plus initial fused test | 49 tests passed | [Log](fused-dsa-suite.log) |
| Expanded score parity | 18 shapes × 3 data distributions × 2 ReLU settings; exact keys and selected IDs/counts, with explicit single-query refusal | [Log](production-shape-parity.log) |
| Layer integration | Exact selection against original GEMM tiling; legacy tile sizes 1/3, continuation lengths 1/55/56/57 | [Log](tail-integration.log) |
| Score memory check | Zero errors | [Log](fused-memcheck.log) |
| Layer memory check | Zero errors | [Log](tail-integration-memcheck.log) |
| Synchronization check | Zero errors with 64 tracked CUDA barriers | [Log](fused-synccheck-barriers64.log) |
| Full-GLM regression | 10 CTests passed: forward/reference, packed prefill/reference, decode, multi-rank, graph execution | [Log](full-glm-regression.log) |

The first synchronization-check attempt exceeded the tool's automatically detected barrier count while instrumenting cuBLAS. Its [failed log](fused-synccheck.log) is retained; the rerun explicitly set `--num-cuda-barriers 64`.

The [full-model comparison](full-model-comparison.json) passed all 18 prompt/output/usage checks under the same four-node FP8 deployment. Tags, seeds, configuration and client code matched the baseline. All ranks had zero process/cgroup swap through clean shutdown, identical operation streams, and no stall warnings. See [driver](compare_server.py), [log](compare-server.log), and [memory evidence](../../attempts/glm53-fp8kv-w4/default-before-single-query-fusion-20261007/memory-check.json).

| Cold context | Previous fixed binary | Fused scores | Time reduction |
| --- | --- | --- | --- |
| 32K | 90.956 s | 74.488 s | 18.1% |
| 64K | 230.703 s | 160.845 s | 30.3% |
| 128K | 664.829 s | 367.143 s | 44.8% |

The exact tested source was committed as [`21b91a3`](https://github.com/HawkBearPig/dgpp/commit/21b91a39f999cef0c94579a3240b62b07854a80b). The serving binary was retained unchanged after that commit. These measurements describe the initial fusion version, before the single-query follow-up below.

## Single-query follow-up

The initial fusion retained the single-query GEMM path solely because its score bits differed at one small-context shape. [FP64 accuracy checks](single-query-accuracy/README.md) show rounding-scale differences with identical selected tokens in all 44 tested cases. The follow-up removes that restriction and uses the fused path for single queries and tails as well.

The original full-model comparison records its launch path at measurement time. That launch now resides in `attempts/glm53-fp8kv-w4/default-before-single-query-fusion-20261007/`; its bytes and binary identity are unchanged. The final version replays the affected full-GLM prefill/context groups.

All three full-GLM configurations have now completed with the final version. The 256K FP4 case measured 1,187.740 seconds for cold prefill and 111.35 ms/pass, 17.62 tok/s for MTP1 decode. All four ranks reported zero swap and no collective-stall warnings. [Issue #96](https://github.com/HawkBearPig/dgpp/issues/96) tracks investigation of the remaining decode and prefill costs; it includes the context sweep and distinguishes the measurements from an unconfirmed diagnosis.

## Benchmark retention

The [retention catalog](../../retained-results.json) binds unchanged configurations and source evidence by hash. Other model families, short full-GLM requests, and context-capacity skips remain eligible. Affected full-GLM prefill groups and populated context buckets are replaced.

Eligibility uses actual input counts. For example, the nominal 2K BF16 group reached 2,054 tokens, so that entire three-request group is replaced. The FP8 2K group stayed at or below 2,047 tokens and is eligible for retention. The controlled comparison also repeats it as a short-path check.

Original full-GLM default launches remain under `attempts/`, with their original manifests and memory evidence. Neither interrupted FP4 launch contributes published measurements. OS swap stays enabled; all DGPP runs use `MemorySwapMax=0`.
