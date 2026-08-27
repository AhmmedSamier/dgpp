# DGPP — M0 Measurement Results

Platform fact (authoritative): DGX Spark GB10 = **unified memory**; no
discrete GPU VRAM, no separate system RAM. One LPDDR5x pool shared by CPU
cores and GPU over NVLink-C2C. Consequences: `cudaHostAlloc` and
`cudaMalloc` reservations draw from the same physical memory; the
device/host distinction is about mapping and access pattern, not location;
NIC traffic destined for GPU consumption never needs a cross-media copy.

Node A = this machine (`192.0.2.11`). Peer B = `192.0.2.12` (node-b).
All RDMA numbers: RoCEv2, RC, MTU 4096 (jumbo IP frames), one active CX-7
per node (`rocep1s0f0`; second CX-7 present but not wired/usable — NVIDIA
DGX Spark platform design).

## Network (authoritative, via `perftest` 24.01)

| metric | value | notes |
|---|---|---|
| Cross-node RC RTT (2B send) | **2.7 µs typical** / 2.64 min / 2.94 p99 | `ib_send_lat -d rocep1s0f0` |
| Unidirectional RC BW | **105.4 Gb/s avg** (13.2 GB/s) | `ib_write_bw`, sizes 256 KB / 1 MB / 8 MB all plateau ≈105.2–105.4 |
| MTU path validation | jumbo OK (`ping -M do -s 8972`) | fabric 88 |
| ICMP RTT peers .12/.13/.14 | ~0.4–0.6 ms software-stack baseline | informational only |
| mgmt LAN (203.0.113.x) | ICMP blocked (firewall) | API traffic may still work over it |

Implication for decode-step collectives (TP=4): payload ≤32 KB ⇒ tree
allreduce ≈ 2–3 hops × 2.7 µs + copy ⇒ **~10 µs/classic allreduce**, two sync
points per layer → ≤4% of the ~26 ms decode floor. Prefill RS/AG streams are
bandwidth-bound at ~13.2 GB/s ⇒ overlapped chunked-pipeline design required
(DESIGN §5.2). NOMINAL-vs-MEASURED gap (200 vs 105 Gb/s) is a hard planning
number now baked into §6.

## ibverbs micro-tool status

- QP lifecycle verified end-to-end cross-node in our `micro_ibv_smoke`
  (handshake/RTR/RTS/data both directions).
- Known issue (parked): initiator-side SEND completions missing when using
  `IBV_SEND_INLINE` or stack-resident SGE VAs paired with MR lkeys; plain
  signaled sends with MR-resident buffers proved correct during variant
  testing but tool rewrite deferred (CollectiveBus will adopt those rules).
- Root/sudo unavailable on nodes: PFC/ECN inspection pending infra support.

## Checkpoint audit

See `docs/checkpoint_budget.md`. Headlines:
- 328.3 GB total; 76,108 tensors; FP8 experts 92.7% of bytes.
- Linear-attention path kept BF16 by unsloth export (~10 GB/token headroom if requantized later).
- Active-bytes/token ≈ **23.5 GB** cluster-wide ⇒ TP=4 share ≈5.9 GB/node ⇒
  decode floor ≈26 ms @230 GB/s effective.

## Device microbenches (GPU window granted; vLLM paused)

`micro_mem_bw -s 4096` (4 GiB working set):

| pattern | GB/s | % of 273 nominal |
|---|---|---|
| read (weight-stream proxy) | **231.8** | 85% |
| write | 195.2 | 71% |
| copy (R+W aggregate) | 214.6 | — |

Device facts (`dgppctl info`): GB10 cc=12.1, **48 SMs**, **L2=24 MiB**, SM clock 2.42 GHz peak,
unified addressing + HMM page-table access ON.

`micro_gemm_peak` (cuBLASLt heuristics; L2-note: shapes ≤24 MiB weights loop
cache-resident, inflating apparent GB/s):

| shape (m,n,k) | dtype | us | TFLOPS | eff-BW |
|---|---|---:|---:|---:|
| attn_o_proj 1×4096×16384 | fp8 | 260 | 0.5 | 258 GB/s* |
| qkv/qb 1×16384×1536 | fp8 | 44 | 1.2 | (L2) |
| moe_w13 1×4096×4096 | fp8 | 18.5 | 1.8 | (L2) |
| dense_gateup 1×24576×4096 | fp8 | 563 | 0.4 | 179 GB/s |
| lm_head 1×154880×4096 | fp8/bf16 | 2824 / 7266 | — | DRAM-bound |
| attn_o_proj 1×… ×16384 | bf16 | 587 | 0.2 | 228 GB/s |
| prefill attn_o_proj 2048×… | fp8 / bf16 | 1378 / 2878 | **199.6** / 95.5 | — |
| prefill dense_gateup 2048×… | fp8 / bf16 | 2059 / 4197 | **200.3** / 98.2 | — |

(* L2-warm artifact vs ceiling cross-check.)

Takeaways:
1. Decode GEMV-class kernels are DRAM-bound at ≈230 GB/s → floor numbers stand.
2. cuBLASLt fp8 prefill rates (~200 TFLOPS/board) ⇒ cluster prefill compute
   ceiling ≈800 TFLOPS ⇒ TTFT for 32K-token prompt of order ~1–3 s ex-cache —
   far better than the conservative §1 estimate; comm overlap still required.
3. lm_head must be vocab-sharded AND considered for FP8 (bf16 costs 7.3 ms/stream single-node).

`micro_gdr_probe`: **GPUDirect RDMA NOT available** (peermem absent;
ibv_reg_mr on device memory fails errno=14). Copy-engine bounce profile:
h2d_pinned 1.57–2.32 µs, d2h 1.61 µs (≤64 KB) ⇒ small-message collectives pay
~+3–4 µs/hop in bounce mode. Decode-lane tree allreduce revised ≈12–18 µs
including bounce — still ≤5% of step time. [Superseded by the unified-memory
re-reading + micro_zerocopy below: bounce is unnecessary for GPU consumption;
see §Zero-copy receive.]

Event record/sync pair ≈ 5.3 µs/iter (host overhead reference).

## Zero-copy receive (micro_zerocopy, 2026-08-27)

Two full runs; both shown / averaged where stable. GB10 unified pool,
256 MiB buffers (≫24 MiB L2), uint4 (128-bit) access, grid search 48–384
blocks.

| pattern | GB/s | notes |
|---|---|---|
| gpu_read_dev (cudaMalloc ref) | 238.7 / 240.8 / 238.3 | M0 baseline confirmed |
| gpu_read_pin (zero-copy) | **254.7 / 254.8 / 252.6** | **~+6% faster than device buffer** |
| …CPU writer on SAME buffer | 254.6 / 251.9 / 250.8 | −1.1% |
| …writer on OTHER pinned buf | 236.2 / 235.0 / 233.8 | −7.8% |
| …2 writers, 2 other buffers | **214.8 / 211.3** | −15.7% |
| …1 same + 1 other | 230.6 / 230.7 | −9.5% ≈ additive |
| gpu_write_pin | 217.1 / 210.6 | full-rate in-place payload build |
| copy_pin2dev (staging copy) | 59.7 / 59.6 | staging would cost ~4× the data |
| copy_dev2pin | 59.6 / 59.6 | symmetric |
| cpu_memcpy_pin (1 thread) | 24.4 / 27.3 | CPU producer ceiling; NIC DMAs bypass CPU |
| FLAG roundtrip (fenced) | p50 1.12–1.17, min 0.99–1.07, p99 ~1.4 µs | 512/512 seq-verified, 3 runs |

Readings:

1. **Zero-copy receive is the fastest path, not a compromise.** Pinned-buffer
   reads beat cudaMalloc reads by ~6% and need no copy. Staging copies
   (59.6 GB/s) would multiply network payload traffic — strictly dominated.
2. **Writer interference is approximately additive per active writer
   region**, and same-buffer placement hurts least (−1% vs −8% per flow):
   uncontended 254.8 → same −4 → other −21 → same+other −24 GB/s
   (sum matches). Worst-case busy-bus extrapolation (all flows writing
   elsewhere) budgets ~15–20% margin on receive bandwidth; decode-time
   small-message traffic sits far below where this matters.
3. **Completion signaling is cheap and provably ordered:** fenced flag
   round trip p50 ≈ 1.1 µs with zero sequence failures across runs —
   better than the 3–4 µs/hop bounce figure from M0, as predicted once
   bounce staging left the picture.
4. Design consequence: CollectiveBus receives go **directly into pinned
   slabs consumed zero-copy by kernels; no staging copies; completion via
   pinned flag words with release/acquire + `__threadfence_system`
   discipline (pattern proven by the flag test).**

Protocol hardened since first measurement: the flag kernel now lives in
`src/kernels/flag_protocol.cuh` (single implementation shared by bench and
`tests/cuda/flag_protocol_test` — in-order ack delivery, payload-visibility-
at-ack checksum invariant, and watchdog-on-dead-wakeup with device-recovery
assertions; runs in ctest as `flag_protocol_test`).

M7 open items (real-NIC validation):
- NIC DMA payload → GPU read ordering/visibility with doorbell scheme (the
  flag test proves CPU-side visibility; mlx5 DMA coherence is the remaining
  question, expected fine on this fabric but must be measured with
  ibv_post_send + GPU read + integrity check).
- Multi-writer / multi-flow scaling (does a second concurrently-written
  region push the other-buffer −7.5% down further?).
- Whether GPU kernels can poll remote-peer completion flags across the RoCE
  path (they cannot read NIC registers; flags must be DMA'd by NIC into
  pinned slab — that's the plan).

## GPUDirect-RDMA / peermem investigation — RESOLVED (2026-08-27)

With narrow sudoers grant (`modprobe`, `dmesg`, `lspci`, `ibstat`):

1. `nvidia-peermem.ko` **IS shipped** here (v580.173.02, NVIDIA-open tree,
   Canonical-signed). Earlier find-miss remains unexplained noise; modprobe
   resolves it fine.
2. Insertion refused with EINVAL, silently (no dmesg entry), even with
   ib_core preloaded. `strings` on the module shows only
   `nv_mem_client_init/cleanup` hooks: it exports GPU memory by registering
   with the NVIDIA client registry for **PCIe BAR-backed** GPUs. On GB10
   there is no discrete GPU / no BAR aperture to export ⇒ early bailout
   before any printk. Verdict: **present-but-inapplicable**, not
   misconfigured.
3. Validates the revised model: with CPU+GPU sharing one LPDDR5x pool over
   NVLink-C2C, NIC-side DMAs land in memory the GPU reads zero-copy anyway.
   CollectiveBus commitment: pinned-buffer receive, direct GPU consumption,
   correct fencing. The 3–4 µs/hop M0 bounce cost already reflects
   same-pool staging, so headroom vs classic GDR nodes is smaller than once
   assumed. Remaining engineering unknowns for the receive path:
   coherency/cache-line behavior under mixed CPU×GPU×NIC access and
   achievable zero-copy read bandwidth — unprivileged microbench planned.

NIC topology (lspci): two MT2910 ConnectX-7 (segments 0000 & 0002), each
dual-port (`rocep1s0f0/f1`, `roceP2p1s0f0/f1`), all links **Gen5 x4 @32GT/s**
— healthy. An earlier suspicious 2.5GT/s reading was my own tooling error:
probing `000f:01:00.0` from dmesg's NVRM line hit the GPU's internal bridge,
not a NIC. M7 gets four roce devices to allocate across; whether both CX-7s
are actually cabled is switch-side verification (see PFC/ECN item above).

## Real-checkpoint loader smoke (2026-08-27)

Production cold path (`ShardedCheckpoint`, eager_bind) over unsloth snapshot:
12.5 MB shardspec + 62 shard mmaps parse/bind in **0.17 s, RSS 188 MiB**;
all 76,108 tensors pass nbytes==numel*elemsize geometry check; spot-bind of 6
roles incl. F8/BF16/F32 decode paths OK.

Facts that adjust M2 assumptions (and our earlier guesses):
- `lm_head.weight` is **BF16** [154880,4096] (not FP8): per-token lm_head read
  is ~1.27 GiB/stream — dominates small-seq decode traffic; keep vocab-proj
  optimizations (chunked argmax w/o logits materialization) high priority.
- Dense attention layers are MLA-shaped low-rank: `q_a_proj` F8_E4M3
  [1536,4096]; layer naming uses `model.language_model.layers.*` with
  `hc_*` hyper-connection wrappers (`hc_attn_{base,fn,scale}`, `hc_ffn_*`).
- Linear-attention layers carry F32 `dt_bias` [8192] (KDA state parameters);
  attention layer count appears smaller than MLP layer count (attn_qkv 24,
  la_dtba 34 → hybrid layout ratio confirmed from tensor census).
- Experts store raw-magnitude E4M3 (mean|v|~61) with separate
  `weight_scale_inv` block-scale tensors ⇒ dequant-on-load must apply scale
  blocks; raw fp8 bytes are NOT unit-range.

## M1 results (core runtime & weight pipeline)

All numbers single GB10 node, vLLM co-resident on the box (0.85 util);
doll defaults: hidden 4096, 8 layers, GQA 32q/8kv × 128, inter 12288,
vocab 154880, ~5.0 GiB weights (bf16 + fp8 lm_head), FP32 logits path.

| check | result |
|---|---|
| gpt-doll greedy transcript vs eager | **bitwise-identical, 48 steps** |
| graph replay repeatability (same inputs) | **bitwise-identical** |
| rollout mega-graph replay (N=48) | **bitwise-identical across replays** |
| decode throughput (sync API, avg seq ~232) | **60.1 tok/s**, step 16.6 ms |
| effective streaming BW @ bs=1 | **225 GB/s** |
| naive-roofline ratio (sync path) | **97.0%** (criterion ≥90%) |
| rollout 1200 steps self-fed graph | 55.5 tok/s, 90.4% roofline |
| rollout 4000 steps (long-context attn) | 43.0 tok/s, 72.2% — O(T) kernel latency-bound; expected, motivates M3 MLA kernels |
| soak 10 min / 37k steps | **alloc growth = 0, slabs stable** |

Shardspec generation (`tools/make_shardspec.py`) over unsloth FP8 snapshot:
76,108 tensors / 62 shards / 305.8 GiB; role split: experts_w13 193.5,
experts_w2 96.8, other 7.0, attn_o 2.9, mlp_w13 1.5, lm_head+embed 2.4 GiB…
(`artifacts/shardspec.json`, 12.5 MB — C++ cold-path parse verified.)

Platform lessons encoded as permanent behavior:
- `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` beyond driver cap FAILS
  but leaves lastError set → the *next* launch misattributes "invalid
  argument". Launchers now clamp to `MaxSharedMemoryPerBlockOptin` and clear
  the error slate before every launch.
- Events recorded inside a stream capture are absorbed into the graph and
  cannot be host-synchronized afterwards reliably ⇒ all result copies +
  completion fences live outside captured regions (`finish_step_outputs`).
- Inside `namespace dgpp`, unqualified C-math names resolve to engine
  templates (dgpp::logf shadowed ::logf in fill kernels). All internal math
  now uses fully-qualified ::logf/exp2f etc.
- e4m3 encoder property tests (full-table roundtrip, no-NaN invariant,
  saturation semantics ≥448) pin the requant helper used everywhere.

## Pending items

- PFC/ECN switch inspection needs mgmt-plane credentials (switch CLI), NOT
  node sudo — separate request.
