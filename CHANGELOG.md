# Changelog

The history by milestone. The dated engineering record in
`benchmarks/results/` has every measurement and fix behind these lines;
`PLAN.md` has the milestones' exit gates.

## Unreleased

- **GLM-4.7 (`nvidia/GLM-4.7-NVFP4`) served** (2026-09-10, docs/glm47_plan.md):
  the Glm4MoeForCausalLM family — 92 pre-norm layers of biased GQA
  attention (96/8 heads, per-head q/k norms, half-split partial RoPE),
  three dense NVFP4 MLP layers then 160-expert sigmoid-routed MoE layers
  with an NVFP4 shared expert, an MTP draft layer whose BF16 experts are
  requantized to NVFP4 at load — on the same session core, engines,
  scheduler and service as GLM-5.3 and Qwen3.8-Flash-Next. New: the
  modelopt NVFP4 binding and loader (`src/models/glm4/`), the fp4 GEMV
  core generalized to any K multiple of 32 (power-of-two shapes bitwise
  unchanged), the NVFP4 shared expert as view-table entry E of the routed
  launches, paged split-KV GQA attention kernels with the fused bias /
  norm / RoPE / K-V append (`src/kernels/glm4_attn.cu`, bitwise a
  tile-aware host reference), `Glm4Model` with the paged K/V pool (no
  recurrent state: rollback positional, snapshots at any position, the
  draft's hidden window), the GLM-4.7 tokenizer (a Sequence[ByteLevel]
  post-processor), the template's `rstrip`/`lstrip` with a character
  argument, `deploy/cluster_glm47.json`, `scripts/fabric_glm4_serve.sh`;
  The partial RoPE is transformers' half-split rotate_half (pairs i, i+32
  over the first 64 dims), not the interleaved form of the older glm/glm4
  architectures: the interleaved reading passed every fixture gate (the
  python reference shared it) and left the real model coherent for ~30
  tokens before looping; found against transformers' own layer code on
  the real weights (`tools/glm4_torch_reference.py`, layer-0 relative l2
  0.036 -> 0.0025 after the fix). The bus recorder's collective-node budget per graph 128 -> 256
  (`kBusMaxGraphGens`: GLM-4.7's decode step records 186 — its first
  fabric boot refused the second graph variant at 128). The draft block
  takes the model's POST-final-norm hidden (vLLM's `glm4_moe_mtp`
  convention; the pre-norm residual gave 11–46 % draft acceptance on the
  fabric, the post-norm hidden 79–92 %).

- **The decode rows are the recipe's shape, and the batched depth-2 chain**
  (2026-09-10): the fixed decode batch's row ceiling is derived at boot —
  `max_concurrency x (1 + mtp_depth)`, floored at 8 so every existing recipe
  keeps its exact shape — and carried at runtime by the session core
  (`SessionParams::decode_rows`: the per-row scratch, the token feeds, the
  draft windows, the memory plan), the picker's tables, the bus's latency
  slot and the graph engine's batch families; the build-time bound is
  `kDecodeRowsMax` = 32 (the pick kernels' fixed arrays). GLM-4.7 takes the
  derived shape (`serve: decode rows 12 (4 slots x 3 rows ...)` in the boot
  log); GLM-5.3-Flash keeps its fixed 8, Qwen3.8-Flash-Next stays capped at
  8 until its kernels are gated past it. On the runtime rows the row batch
  carries depth >= 2: `session_graph_capture_draft_chain_batch` +
  `glm_spec_chain_rows_batched` run every slot's chain row in one draft-
  block run (the batched next-token feed carries every draft per request);
  a family opts in with `kBatchedDraftChain` (GLM-4.7). The GEMM seam
  lowers bf16 decode calls up to the model's decode rows to the row-
  independent GEMV core (`CublasLtGemm::set_decode_rows`; the first 9-row
  batch fell to an Lt algorithm with its own reduction order and flipped a
  near tie), so a batched request's rows keep the scalar order at any
  width — the depth-2 engine gate runs A scalar and B, C batched on a 9-row
  ceiling, bitwise the eager engine's. Also fixed: one slot's sampled MTP
  fallback dropped every slot's draft picks of a batched replay (the mirror
  went stale for the others). Measured on the fabric (`fabric_glm4_load.sh`):
  batched depth 2 loses to depth 1 at 2 and 4 live requests (37 vs 44.5,
  43 vs 47.5 tok/s) — every 4-row GEMV chunk past the first re-reads the
  6.3 GB of BF16 attention projections per rank (+45–60 ms per chunk: 4
  rows 79 ms, 6 rows 123, 8 rows 140, 12 rows 200); depth 2 remains a
  single-stream gain (+4–13 %). The lever for concurrency at either depth is
  the attention projections' bytes per pass: wider GEMV chunks or a row-
  independent tensor-core kernel, docs/measurements.md.

- **MTP depth 2 on the hidden-window families** (2026-09-10): the session
  core carries the GLM-5.3-Flash chain — one draft-block row per further
  draft at counter + index, fed the previous draft's pick and the block's
  own output row as its hidden, landed in the slot's window
  (`glm_spec_chain_row_window`); `session_draft_chain`,
  `session_graph_capture_draft_chain`, the multi-draft feed; a family opts
  in with `kDraftChain` (GLM-4.7 yes, Qwen3.8-Flash-Next not yet — its
  draft ring would need the chain snapshot). `deploy/cluster_glm47_d2.json`.
  GLM-4.7 at depth 2: 72 ms/pass at 2.3–2.6 tokens/pass (p2 48–65 %), 4–13 %
  more tokens/s single-stream than depth 1, +5 % at a 6.5K context;
  transcripts identical. Gate: `glm4_engine_test`'s depth-2 world (port
  29951); the row batch past depth 1 followed the same day (above).

- **`chat_template_kwargs.enable_thinking` is a knob of the templates that
  read it** (2026-09-10): the service asks the loaded template
  (`ChatTemplate::reads`, `ModelFrontend::template_reads`) and passes the
  boolean through for Qwen3.8-Flash-Next and GLM-4.7 (false closes the
  think block in the generation prompt — GLM-4.7's non-thinking mode; it
  ignores reasoning_effort and otherwise thinks to the token cap); GLM-5.3-
  Flash's template never reads it and the refusal stands. `serve_eval.py
  --no-think`; `serve_api_check.py` builds its stop material with thinking
  off where the template allows it.

- **Scheduler: a step cut short takes no retire-time prefix snapshot**
  (2026-09-10). When a multi-token MTP pass's earlier token completes the
  answer (the cap or EOS), the pass's remaining tokens are dropped and the
  model's state sits past the committed position; the retire path asked
  the arena for a live snapshot at the committed position and the arena
  refused (an engine failure on GLM-4.7's first fabric request — every
  position is snapshot-eligible at its align of 1; the aligned families
  hit it only when the cut lands on an aligned position). `Request::
  step_tail` records the dropped tail; the rolling entry stands as the
  close entry. `scheduler_prefixCache_aStepCutShortByTheCapTakesNoRetireSnapshot`.
  Gates: the python double reference (`tools/glm4_reference_dump.py`,
  final hidden l2 0.0035, top-1 and routing exact, the draft's rows),
  transformers' own layer code on the real weights (layer-0 relative l2
  0.0025), decode/sessions/speculator (60 steps across the pool's block
  boundary, per-step rolling snapshots), TP worlds 2 and 4, the graph
  engines at world 2 (scalar, batched and MTP across the boundary),
  tokenizer and chat-template goldens (HF tokenizers 0.23.1, jinja2
  3.1.2). Fabric (four nodes): boot 18–20 s from the resident image, T=1
  49.0 ms/step, MTP 60–61 ms/pass at 1.86–1.99 tokens/pass (31–33
  ms/token), MTP == T=1 transcripts, eval with thinking off gsm8k 60/60,
  HumanEval 39/40, extraction 30/30 (docs/measurements.md).

- **Shared cores extracted for the second and third families** (2026-09-09):
  `loaders/resident_stream.hpp` (the resident/streaming layer stream —
  bumps, staging, the image, byte reconciles, the digest — under a
  family's builders; Qwen and GLM-4.7 on it), `engine/paged_blocks.hpp`
  (the paged block table with its sharing and pinning protocol under both
  K/V pools), `engine/session_model.hpp` (the session core: positions,
  feeds, chunking, snapshots, the graph era, the draft plumbing, under
  `QwenModel` and `Glm4Model`). Every existing gate unchanged.

- The fp8 tile kernel ported to the ldmatrix form (32-deep stages, the
  block scale copied with the tile, fragments decoded e4m3 -> f32 x scale
  -> bf16 as the reference rounds them): bitwise the reference; the FP8
  checkpoint's steady-state prefill 740 / 1,725 / 7,463 -> 549 / 1,412 /
  6,280 ms at 512 / 2,048 / 8,192 tokens (with the mHC GEMM form), the
  hybrid's shared experts 440 / 1,290 / 5,769 -> 425 / 1,275 / 5,702.

- The prefill's mHC dots on tensor cores: a bf16 mma.sync GEMM over a
  cp.async ring with per-stage fp32 partials, the token's sum of squares
  gathered from the same tiles, then the standard finish; the sites no
  longer store the unread collapsed row. Within the oracle budgets (not
  bitwise the per-coefficient decode form; the tiled kernel stays behind
  `mhc_set_prefill_gemm(false)`): 1,196 -> 836 us per site at 2,048
  tokens; steady-state prefill 460 / 1,335 / 5,954 -> 440 / 1,290 / 5,769
  ms at 512 / 2,048 / 8,192; the 64-step transcript identical, one
  near-tie flip (0.049 logits) in a 512-token continuation.

- The fp4 prefill kernel, second pass: the ldmatrix tile kernel (64 x 128
  x 64, a three-slot cp.async ring of the activation tile and the raw fp4
  codes, B fragments decoded at fragment time, each weight row's next L2
  line prefetched ahead, the activation lines tagged evict-last, one m-tile
  per block with the m-tile the fastest grid index) is the production
  launcher on both expert shapes — bitwise the reference; steady-state
  prefill of the hybrid 563 / 1,502 / 6,572 -> 460 / 1,335 / 5,954 ms at
  512 / 2,048 / 8,192 tokens, generated ids unchanged. glm_moe_test's odd
  geometries caught misaligned copies at k = 208 (aligned-width fallbacks
  added). The FP8 path is untouched.

- The mHC site off the critical path: the fused finish leaves comb (the
  20-iteration Sinkhorn, 4.6 of its 8.6 us) to `launch_mhc_comb` on a
  low-priority side stream forked after the finish and joined before the
  stream update — the only reader — and reuses the dots block's register
  copy of the streams for the collapse; bitwise (glm_mhc_test pins the
  deferred form at 1/2/8 rows). The fused kernel 13.0 -> 6.5 us per site
  in the MTP graph; the hybrid's decode MTP 34.21 -> 33.89 ms/step (19.96
  -> 19.77 ms/token), T=1 26.46 -> 26.22. `mhc_site_bench` (new) times a
  site with the finish, Sinkhorn and norm separable. The router's register
  select and the collective's local phases were measured and parked
  (`docs/nvfp4_plan.md` §6a). DGPP_MHC_COMB_SIDE=0 keeps comb in the
  finish.

- The fp4 GEMV core decodes with GB10's own e2m1x2 -> f16x2 conversion and
  one exact f16 multiply by the group scale (bitwise the register decode;
  the build moves to the architecture-specific target `121a`), the
  activation window read once per chunk column; the decode slot path
  204 -> 195 us per MoE layer at one row, MTP 20.04 -> 19.96 ms/token on
  the hybrid. Hardware counters (Nsight Compute as root) and a no-arithmetic
  probe of the same access pattern put both fp4 slot kernels at 90-93 % of
  their streaming floor; a persistent grid-stride form with next-tile
  lookahead, the paired gate/up issue, more row steps, four blocks per SM,
  the router's unroll and block width, and programmatic dependent launch
  across the graph's seams were each measured and parked
  (`docs/nvfp4_plan.md` §6a).

- NVFP4 routed experts, phase 1 of `docs/nvfp4_plan.md` — the format is
  loadable. The composed checkpoint `dgpp/GLM-5.3-Flash-NVFP4-FP8`
  (`tools/compose_nvfp4_hybrid.py`: dabsLabs' NVFP4 experts for layers
  3–44 beside the FP8 release's bytes for every other tensor, the MTP
  layer included; 195.1 GB, built and verified byte-for-byte on all four
  nodes, identical shard hashes) declares `quantization_config.quant_method
  = "dgpp_mixed"`, which `GlmTextConfig` parses into a routed-expert format
  (`GlmExpertFormat`: FP8 block-128 as before, or NVFP4 group-16). Under the
  NVFP4 profile the expected-tensor table emits the triple `weight_packed`
  (U8 [N, K/2]), `weight_scale` (e4m3 [N, K/16]) and `weight_global_scale`
  (F32 [1]) for every main-stack routed expert and the validator binds it
  as a unit (every tensor now carries a `GlmTensorRole`, so an e4m3 block
  scale is never mistaken for an FP8 payload); the loader keeps the bytes
  untouched in `GlmFp4Matrix` views — this rank's inter slice as
  contiguous gate/up rows and nibble-granular down column packs, every
  matrix's global scale gathered into one per-layer array — with the same
  counting-mode byte formula, resident image and byte reconcile;
  `GlmTpViews` carves the same views from a full layer so the shard-parity
  gate pins the two paths bitwise on an NVFP4 fixture at worlds 2 and 4,
  streaming and resident. The MTP layer's experts, the shared experts, the
  dense MLPs and the attention keep the FP8 paths untouched under both
  profiles; the FP8 checkpoint's table and bytes are unchanged (the
  existing gates pin that). `glm_bind_check` on the hybrid: 112,049
  expected, 112,049 matched, 36,288 NVFP4 triples and 1,050 FP8 pairs
  bound; the resident formula at world 4 is 50.7 GiB per rank against
  ~82 GiB for the FP8 checkpoint.

- NVFP4 routed experts, phase 2 — the decode path runs on the packed
  bytes. `src/kernels/fp4_gemv.cuh` is the in-register e2m1 core: a lane
  loads 16 bytes (32 codes, two group-16 scales), places each code's bits
  into an fp16 field (`((code & 7) << 9) | ((code & 8) << 12)`, which reads
  as value x 2^-14 for every code) and multiplies by the scale x 2^14, all
  exact — an e2m1 times an e4m3 has at most six significant bits, so the
  weight enters the fp32 dot unrounded and the only inexact operation is
  the one division by the tensor's global scale in the epilogue
  (`tests/unit/fp4_test.cpp` pins the lemma over every code x scale pair
  and the bit-placement against the codec table). The core is templated
  on K so every index is compile-time (the first draft's lane-dependent
  accumulator index spilled to local memory and ran at 14 GB/s); it
  backs the single-matrix launcher (`launch_fp4_gemv_bf16/f32`), the fp4
  decode slot kernels (gate_up+swiglu and down, the shared expert running
  the fp8 core inside the same launch as before) and the fp4 grouped
  GEMV that `GlmMoeLayer` dispatches under the NVFP4 format; the host
  oracle dequantizes the triple exactly and applies the same divisor
  epilogue. Gates: `fp4_gemv_test` (the oracle across every supported
  geometry at l2_rel 0, row-count invariance, the f32 epilogue, NaN
  scales, the geometry contract, and two real slices of the hybrid at 0
  mismatches), `glm_moe_test`'s fp4 twins bitwise — the decode slot path,
  the host grouped path and the sliced-rank fold agree bit for bit as the
  FP8 ones do. `moe_slot_bench --format fp4` per layer on one rank's
  slice: rows=1 209 us against fp8's 273, rows=2 334 against 469, rows=8
  929 against 1,420. Prefill under NVFP4 uses the grouped GEMV chain
  (the tensor-core kernel is phase 3). The FP8 paths are untouched.

- NVFP4 routed experts, phase 3 — the prefill runs on tensor cores.
  `moe_grouped_mma_fp4_kernel` is the fp8 tile kernel's structure (128 x
  64 x 64 stages, the same activation tile and ascending-k16 bf16
  `mma.sync` chain) with the weight tile decoded from the NVFP4 triple:
  a thread's 16 consecutive k come from one 8-byte load of codes and one
  scale byte, decoded as e2m1 x scale, exact in bf16, so unlike the fp8
  tile nothing is rounded before the MMA; the global scale divides the
  finished dot once in the epilogue. Any k that is a multiple of 16 (the
  GEMV core's power-of-two set does not apply). The grouped form
  (segments through the row map) is bitwise the dense form per segment;
  `GlmMoeLayer` dispatches it for NVFP4 routed segments under the
  tensor-core kernel while the FP8 shared expert keeps the fp8 tile
  kernel; the fp8 kernel's source is untouched. Gates: grouped bitwise
  the dense form at I = 208 (a ragged n-tile and a ragged last k-stage)
  and 320, under the z split and through a permuted row map, gate bf16
  and down fp32; the whole layer within budget of the double oracle on
  four geometries beside the GEMV chain; both real checkpoint slices at 0
  mismatches with a row's bits independent of m. Fabric, steady-state
  prefill on the hybrid: 512 tokens 1,152 -> 635 ms (FP8 740), 2,048
  tokens 4,396 -> 1,563 (FP8 1,725), 8,192 tokens 18,246 -> 6,796 (FP8
  7,463); the 64-step transcript identical to the GEMV-chain build's, all
  ranks identical; the FP8 build's 2,048-token prefill unchanged (1,717
  ms). nsys says the tile kernels — fp8 and fp4 alike — sit at a third of
  the read floor (a 340 MB fp4 launch takes 4.6 ms): the activation tile
  is re-read once per 64-wide n-tile, the stages do not overlap, and a
  128-row m-tile is more than half padding at ~57 rows per expert; the
  restructure is `docs/nvfp4_plan.md` §6a (phase 5).

- `scripts/serve_eval.py`, the task-level eval through the served endpoint
  (HumanEval, GSM8K, schema extraction; greedy), run on both configs: the
  NVFP4 hybrid 157/164, 293/300, 100/100 against FP8's 155/164, 293/300,
  100/100 — the same items pass on both to within two HumanEval problems
  and three GSM8K flips each way, while the replies are word-identical on
  only two thirds of the items. `docs/nvfp4_plan.md` §6a.

- The DSA attention projections consumed as FP8 directly (the "bridge"
  item of `docs/nvfp4_plan.md`): q_a, kv_a, q_b and o_proj — FP8 pairs in
  every checkpoint — were dequantized to BF16 at load (the M3 seam) and
  read at twice their bytes on every step. The loader now keeps them as
  the checkpoint's pairs whenever this rank's q_b row slice and o_proj
  column slice start on the 128-wide scale grid (every power-of-two world
  of the production geometry; the bf16 bridge remains for misaligned
  slices — the test fixtures at world 4), `DsaLayerWeights` carries either
  form, and the layer's three projections run through the scale-aware
  GEMM (which gained an output row stride so the fused [q_a | kv_a]
  output takes two pairs into one buffer). `GlmTpViews` slices the pairs
  at aligned worlds and dequantizes the full pairs itself (the loader's
  kernel and rounding) for the bridge form at misaligned ones, so the
  shard-parity gate pins both forms bitwise against the sharded loader.
  The resident image format version is 2 (bridge-layout images are not
  restored). Gates: `dsa_test`'s new prefill case runs the same weights
  in both forms against the host reference; `glm_loader_test` checks the
  pairs byte-exact; shard parity at worlds 2 (direct) and 4 (bridge).
  Applies to the FP8 model and the hybrid alike: 0.35 GB per token per
  rank fewer over the 11 DSA layers.

- MTP depth 2 re-measured on the NVFP4 hybrid through the serve path: 4-6 %
  faster per token on 300-token prose, code and JSON answers at short
  context (2.5 tokens per step, the second draft standing 59-61 %), 6.6 %
  slower at a 7,368-token context (the second draft standing 41 %). Depth
  1 stays the default; the numbers are in `docs/nvfp4_plan.md` §6a.

- NVFP4 routed experts, phase 5, the decode core: one row step per warp in
  the fp4 GEMV core instead of two (`fp4_gemv::kSteps`) — twice the blocks
  per launch, four loads in flight per lane — after a sweep of the core's
  tunables on `moe_slot_bench` in which every "more per lane" direction
  lost; 5-6.5 % faster per layer at 1, 2 and 8 rows in an alternated A/B,
  outputs bitwise (a row's chain does not depend on the step count).
  Fabric, the hybrid: T=1 28.0 -> 27.0 ms/step, MTP 36.6 -> 35.4 ms/step
  (21.2 -> 20.5 ms/token), acceptance unchanged.

- NVFP4 routed experts, phase 5, the prefill tile kernel: a warp whose 16
  rows lie entirely past a segment's end skips its MMAs (its rows are
  never stored, so the outputs are bitwise), and `GlmMoeLayer` dispatches
  the fp4 tile kernels by shape — the synchronous 128 x 64 x 64 tile for
  gate/up, a pipelined 64 x 128 x 32 kernel (cp.async double-buffered
  activation stages, the next stage's codes decoded after the MMA) for
  the down projection, each the faster one on its shape. `moe_tile_bench`
  times every variant per launch at the production expert shape with
  uniform or ragged segments and checks the pipelined kernel bitwise
  against the reference; a decomposition with the loads stubbed out shows
  the gate launch is 58 % padded MMA + decode + barriers and the rest
  un-overlapped operand loads (`docs/nvfp4_plan.md` §6a records the
  numbers and what remains). Fabric, steady-state prefill on the hybrid:
  512 tokens 563 ms, 2,048 tokens 1,502, 8,192 tokens 6,572 (phase 3:
  635 / 1,563 / 6,796); transcript identical, all ranks identical.

- NVFP4 routed experts, phase 4 — the hybrid serves. A site keeps one
  cluster config per checkpoint and names it on the command line
  (`deploy/cluster.*.json` is git-ignored beside `deploy/cluster.json`;
  `docs/operations.md`): `deploy/cluster.nvfp4.json` selects
  `dgpp/GLM-5.3-Flash-NVFP4-FP8` with `kv_capacity` 786,432 and
  `prefix_cache_gib` 8, the 31 GiB per rank the experts free spent on
  context and cache (`--memory-plan`: 96.2 GiB + 8 headroom of 116.5).
  `scripts/dgpp-cluster up --config deploy/cluster.nvfp4.json` boots from
  the resident images in 18 s; `serve_api_check.py` passes every case;
  the world tears down with four identical op-stream md5s and no stall.
  The L2 prefetch defaults stand on the hybrid: the window is flat from 8
  to 24 MB at T=1 (28.0 ms/step) and under 1 % apart under MTP; the
  prefetcher itself is worth 2.0-2.3 ms/step, the full boundary rate
  costs 2 ms, as on FP8. `scripts/serve_bench.py` and `serve_soak.py`
  address the model the service reports (`GET /v1/models`) instead of a
  hard-coded FP8 id, with an optional argument to override.

- `glm_gen_check --mtp --decode-graph` accepted no draft since 2026-09-06
  (every class 0.0 %, the transcript still exact): the decode-step commit
  moved the scalar graph's fed tokens to the slot's persistent feed rows
  (`device_feed`, rows 8 onward) and the serving adapter followed, but the
  evidence app's recorded verify kept comparing row 0's winner with the
  eager rows at `device_tokens()`, which nothing rewrites during a replay.
  The app's recorded verify now reads the feed rows; the eager speculator
  (77 % accepted) and the serving path were never affected. Found by the
  NVFP4 evidence chain; bisected with the app built at f41c506 and
  069a063 (both 0 %) against the eager path (77 %).

- The row batch is a family: a 2-slot (4-row) and a 3-slot (6-row) batch
  are recorded beside the full 8-row one, and a step replays the smallest
  whose slots cover the live requests. Two live requests used to be either
  two scalar replays (83 ms/step) or the 8-row batch with four padding
  rows (96.5 ms/step, the reason the crossover sat at four); the default
  crossover is now two. A single live request replays its scalar graph as
  before. Gated bitwise against independent scalar sessions and
  speculators through every occupancy and the hole cases, plain and MTP,
  on the loopback world.

- A prefix cache miss is explained on its own INFO line: the cuts probed,
  the entries held, and where the prompt parts from the entry it shares
  the most with (`PrefixCache::nearest`), so a client that edits the
  system prompt or compacts its history between turns is named on the
  spot; when the conversation's own entry was pushed out of the arena the
  line says that instead, from a ring of the last 256 evicted entries
  (`PrefixCache::ghost_at`). `scripts/serve_agentic_streams.py` reproduces an agent client's
  pattern — several concurrent multi-turn tool loops with the reasoning
  stripped from the history, side requests between turns, an optional
  system-prompt edit — and reads the cache's answer per turn from
  `usage.prompt_tokens_details.cached_tokens`; the scheduler suite pins
  two interleaved conversations hitting on every turn under a six-slot
  arena, one returning its reasoning (attaching at the close entry) and
  one stripping it (attaching at the previous prompt's header cut).

- An integer's `minimum` / `maximum` / `exclusiveMinimum` /
  `exclusiveMaximum` are enforced by constrained decoding, for a tool
  argument and for a `response_format` schema alike: the bounds compile
  to one inclusive 64-bit range (a fractional bound rounded inward, an
  exclusive one stepped by one), the JSON machine admits a digit only
  while some completion can still land inside the range and the value's
  closer only once the digits do, and the mask re-judges the pure-numeric
  tokens by the same arithmetic, proven equal to brute force by the walk
  gate. A `timeout` with `minimum: 1` can no longer come out as `0` or
  `-5`, a `limit` with `maximum: 2000` cannot exceed it, and the tool
  behind them never sees a value its schema excluded (the Hermes agent's
  definitions carry such bounds on every integer argument; before this
  the server logged one INFO line per bound saying it was not applied).
  A number that admits a fraction keeps its bound unenforced — refused
  under `strict`, noted with the reason for a non-strict tool — as does
  the draft-4 boolean form, a bound beyond int64, an empty range, and an
  enum no member of which fits; an enum beside a bound is filtered to the
  members inside it. The INFO line for the keywords that stay unenforced
  now carries the reason. Also fixed on the way: beside a bounded (or
  plain) integer alternative, an enum of integers let the mask admit a
  `.` every cursor then refused; and a top-level number under an `anyOf`
  is complete where some alternative accepts it, not only where all do.

- Fixed: the decode index selection's histogram sat at a call-dependent
  workspace offset (the call's rows of keys), so a ONE-row call — the
  sampled fallback's eager verify or re-draft — read its histogram out of
  row 1's keys, which every two-row replay writes: a garbage histogram, a
  partial best-list fill, stale shared-memory pool ids expanded into token
  ids, and the listed attention reading an unmapped page (an engine
  failure once in roughly ten fallbacks past 2,048 tokens of context; the
  first report was a live service at 22:13 on 2026-09-06). The histograms
  now sit at a fixed offset for the maximum rows and are zeroed every
  call; the best list starts empty and the expansion drops any pool past
  the row's visible pools; the listed gather checks the token and its
  block and zero-fills instead of faulting; both record their first
  anomaly, logged at the slot's close (`dsa_select_anomalies`,
  `dsa_attn_anomalies`). `DGPP_SYNC_EAGER=1` syncs an eager row after each
  stage and validates the selection list before the attention — the knob
  that found it. Verified by an 11-turn long-decode sampled soak (about 150
  fallbacks past 2K tokens of context) with no anomaly, where the
  unfixed select logged short fills on every request, and by the DSA
  suite.

- The pipelined replay: the engine's step returns at the verify's verdict
  (a kernel node publishes the slot's replay sequence to pinned memory)
  and the next replay is launched before the previous one's draft tail
  and window are settled, so the verdict read, the scheduler's
  bookkeeping, the rolling prefix snapshot and `cudaGraphLaunch` itself
  (~0.5 ms at 1,168 nodes) hide behind the tail. The token feed is per
  slot on the device, the pick's masks ride a pinned staging behind an
  in-graph handshake, the bus holds two live replay windows (a ring of
  arms, FIFO finish, in-order adoption) and every graph shape has two
  alternating variants. Eager bus work drains the replays in flight.
  `DGPP_PIPELINE=0` restores the settle-after-launch step. Measured on
  the fabric (one greedy request, the same binary pipelined vs
  `DGPP_PIPELINE=0`): 41.2 vs 41.7 ms per pass at a short prompt, 41.9 vs
  43.0 at 8K, 42.6 vs 43.3 at 32K — 0.5 to 1.1 ms per step, 44.1 / 45.3 /
  43.7 vs 43.4 / 44.8 / 43.1 tok/s on prose / code / JSON (+1.1 to
  +1.6 %); the greedy transcripts are byte-identical to the earlier
  plain, depth-1 and depth-2 runs. Three loopback gates run their eager
  oracles through the drain; the bus test arms two windows at once.

- MTP draft depth (`engine.mtp_depth`, `--mtp-depth`, 1–3): the verify
  runs 1 + depth rows and the draft block's chained rows (one more block
  row per draft, on its own output) propose the drafts after the first.
  The sampled verdict kernel decides T rows in a chain (draw for draw
  what the eager sampled speculator decides), the host fallback continues
  the chain from the row that fell back, the tail ring is guarded around
  the chain rows, the hop snapshot takes the rows the step committed past
  the position, and the device picker carries one slot per draft. Depth 1
  is byte-for-byte the two-row step as built; past depth 1 every step is
  a scalar replay. New gates: the T=3 verdict oracle, the greedy depth-2
  and depth-3 loopback gates (plain transcript, feed == the eager chain),
  the sampled depth-2 loopback gate (transcripts == the eager depth-2
  speculator's through 17 fallbacks). On the fabric the greedy transcripts
  of three prompts are byte-identical plain / depth 1 / depth 2; depth 1
  is unchanged at 42–43 ms/step; depth 2 is 54–56 ms/step with the second
  draft standing 45 % (prose) to 65 % (code) of the time — −4 % on prose,
  +4 % on code and JSON — so depth 1 stays the default
  (`scripts/mtp_depth_check.py` runs the transcript check).
- Per-position draft acceptance: the engine counts, per slot and overall,
  the steps that verified each draft position and the steps in which it
  stood; the stats line's `mtp` group prints `accept p1 74 % p2 61 %` and
  every MTP retire line carries the request's own `accept p1 78 %`.

- The KV cache's dtype is a config key (`engine.kv_dtype`, `--kv-dtype`):
  `bf16`, `fp8` (e4m3 + a row scale) or `fp4` (e2m1 in blocks of 16 with
  e4m3 block scales); the attention kernels dequantize on the load, the
  index cache and the selection are unchanged, the format rides the
  settings record and the config digest. New unit and CUDA tests pin the
  codecs, the append and the three attention kernels per format.
- The memory plan: every rank itemizes every byte it will allocate for its
  configured shape and refuses to boot when it does not fit the node's
  free memory, before the first allocation, naming the largest items and
  the largest `kv_capacity` that would fit; `dgpp-serve --memory-plan`
  runs the check alone. The model's per-forward row bound is now the
  prefill chunk rather than the whole context (the activations no longer
  grow with `kv_capacity`: a 262k-token context needs ~98 GiB per rank
  instead of ~300 GiB and boots), `max_context()` is the context bound.
- Every line the server prints is timestamped: the step-timing report and
  the MoE chain dump go through the logger, an uncaught exception leaves a
  stamped line before the abort, and the launcher stamps its own lines in
  the same format. The stamps are UTC on every rank (a peer's log used to
  carry its node's own zone, seven hours off the head's).
- The throughput line leads with what an operator watches — `decode 30.9
  tok/s, 31.4 ms/tok, 54.6 ms/step (178 steps / 309 tok, 97 % of wall) |
  mtp 1.74 tok/step/req = 74 % drafts accepted | prefill … | live 0,
  queued 0 | pool … | prefix cache … | requests …` — and every request's
  retire line carries its own numbers: the prompt and cached tokens, the
  prefill's ms, the decode tokens and passes, tok/s, ms/tok, ms/pass and
  MTP's tok/pass.
- A non-strict tool argument keeps its type under a keyword that only
  narrows the value (`minimum`, `maxLength`, `pattern`, `format`, …): the
  bound is not enforced and one INFO line says so, once; a schema outside
  the subset in shape still leaves the value free, with one WARN line,
  once (the same three WARN lines used to repeat before every request of
  an agent client). Strict tools refuse as before.
- The settings record carries the resolved row-batch threshold, so the
  pushed and the effective config print the same `batchmin`.
- The decode select kernel (DSA, the step's only context-scaled kernel)
  rewritten: every block scores its stripe and publishes the keys plus a
  radix histogram, the last `rows` blocks each find one row's boundary key
  by radix refinement, gather, and expand with a rank sort — the same set
  under the key's total order, so the bitwise gates are unchanged.
  `dsa_select_bench` at the MTP shape: 4K / 8K / 32K / 393K tokens of
  context 572 / 608 / 832 / 3700 us before, 25 / 33 / 67 / 509 after; the
  12 calls per step were 7–10 ms of the 54 ms step at agent-sized contexts.
  The workspace is `dsa_select_workspace_bytes` (keys and histogram per
  row, sized for the pool capacity), counter_ws is two words.
- The MoE down projection runs four rows per warp (`block_rows_multi`): the
  sliced k of 512 bytes gave a warp one load in flight; the per-row chain
  is factored (`consume_chunk`) so every row is bitwise the old one.
  `moe_slot_bench` at the per-rank geometry: the two-row decode chain
  487 -> 457 us per layer (1.3 ms per step), the eight-row 1598 -> 1412.
- On the fabric (one request, sampled MTP): 41.8 / 44.4 / 44.5 / 45.2
  ms per step at 26 / 8K / 18K / 32K tokens of context, against 43.4 /
  53.6 / 54.8 / 56.2 before the two kernels — 23–25 ms per token at every
  context an agent client uses.
- The decode attention partial kernel keeps each thread's q window and
  c accumulator in registers instead of shared memory: the block drops
  from 95 KB to ~40 KB, two blocks per SM, the 64-block decode grid in one
  wave — 2.24 -> 1.16 ms per step on rank 0's trace at 8K, bitwise the
  smem form (each thread always owned exactly that window). The gather
  loop is unrolled so its token -> block-table -> row chains overlap. The
  split count stays 32 (48 and 64 measured the same step on the fabric).
- The L2 prefetcher coalesces adjacent adds into one launch
  (`DGPP_L2_PREFETCH_MERGE=off` restores one per tensor): the decode
  graph goes from 1,595 to 1,168 kernel nodes and `cudaGraphLaunch` from
  730 to 520 us per step (the host enqueues ~0.45 us per node and the GPU
  idles through it — the seam the design record called hidden). The
  merged kernels stream through more of each collective's handshake
  (+3 us per collective on the timeline), so the step itself is within
  noise of the per-tensor form; kept for the node count.
- The mHC dots kernel's two streaming loops unroll 8 (bitwise; ~0.1 ms
  per step). Tried and reverted, the reasons in the code: batching the
  bus fold's loads (the fold span stayed 9.3 us), and decoding q/k to
  floats in the select kernel (four times the smem bytes per pool).
- The launcher forwards every `DGPP_*` knob in the head's environment to
  the peers.
- The graph collective's fold reads staged copies: the placement gate's
  hash pass, which reads every claimed payload anyway, writes it into
  dynamic shared memory (48 KB for the decode's three 16 KB payloads;
  wider vectors fold from the NIC-placed rows as before), and the fold
  after the last arrival is a shared-memory pass — 9.3 -> 5.9 us per
  collective on the bus timeline, bitwise the same chain. Staging needs
  every peer slot 16-byte aligned (element counts a multiple of 8): the
  step's small all-gathers are not, and the first fabric boot found that
  as a misaligned address the loopback gates' widths never hit.
- The decode attention runs on the tensor-core listed kernel
  (`DGPP_DSA_DECODE_MMA=off` restores the register split kernel): two rows
  x 16 local heads is one M-block, each slab its row's own selection with
  its own request's block table. The DSA layer at the rank's shape:
  0.563 -> 0.516 ms per row per layer at 8K. Numerics: the mma summation
  order, tolerance-equal — the teacher-forced gate over the 556-token
  text passed at a mean NLL delta of +0.0008 nat (bound 0.02), perplexity
  2.587 vs 2.585, no move over 1 nat.
- On the fabric after both: 41.9 / 42.7 / 43.2 / 43.5 ms per step at 26 /
  8K / 18K / 32K tokens of context (400-token generations).
- `DGPP_BUS_TIMELINE=1` writes the bus's per-window collective timeline
  (copy / handshake / skew / fold per collective, the wait histogram, each
  peer's lag and how often it arrived last) at INFO without the rest of
  the debug output, which perturbs what it measures; the launcher forwards
  it, and the head's `DGPP_LOG_LEVEL`, to the peers (the peers used to
  expand the level on their own environment and always ran at info).

## 0.1.0 — 2026-09-06

Productionizing: the model-agnostic code left the `glm` namespace
(`dgpp::serve`, `dgpp::sched`, `dgpp::sample`, `dgpp::text`; GLM under
`src/models/glm/`; the binary is `dgpp-serve`); one cluster config
(`deploy/cluster.json`, a site's copy of `deploy/cluster.example.json`)
read by the launcher and every rank, the head
pushing the world's settings to the peers over the journal before anything
builds; versioned releases (`scripts/release.sh`) installed once per node
(`dgpp-cluster install`), the version stamped into the binary and refused
across a mixed world; `scripts/dgpp-cluster` as the launcher.

The request contract gained `stop`, `n`, `logit_bias`, and the usage's
`cached_tokens` and `reasoning_tokens`. Every rank's log carries one
aggregate throughput line per 10 s; the per-tick lines moved to DEBUG.

## 2026-09-05 — M9 closed; M7 built; M6 and M8 buttoned up

- M9: the continuous op-stream drift check on every journal record; a
  malformed-HTTP fuzzer under AddressSanitizer (three defects found and
  fixed); the one-hour mixed soak; the operator's page; the sign-off report
  (32K TTFT, MTP acceptance per class, the prefix cache's capacity curve);
  the ranked next-steps list.
- M7: the exact snapshot prefix cache — decisions rank-identical in the
  scheduler, the snapshot arena in the engines, hot == cold bitwise.
- The v1 failure semantics built, gated and drilled with kill −9 on the
  four nodes; the strict tool grammar's key ledger; cancellation under the
  one-graph step gated.
- Prefill rounds 4–9: the bulk collective as a cooperative kernel with paced
  senders, the MoE experts on grouped tensor-core kernels, the attention
  prefill as a flash kernel; 256 tokens 0.58 s, 2,048 tokens 1.7 s.

## 2026-09-04 — the contract fills in

Tool calls and reasoning on the wire; constrained decoding as a guarantee
for `tool_choice` and `parallel_tool_calls`; `response_format` with JSON
schemas through the same masks; typed tool arguments; drain-on-stop;
grow-on-demand admission; exact sampling on the service with the on-device
verdict and its gather fallback; the sampling-width sweep; prefill rounds
1–3.

## 2026-09-03 — M8: transactional MTP; the on-device step

Greedy speculative decode with the checkpoint's MTP layer as one graph
replay per step, transcript identical to plain decode (31.3 → 22.45 ms per
token); the pick behind the head on the device; the adaptive scalar and
row-batched graph variants that removed the low-occupancy regression; the
kernels-only decode graph that closed the batched-MTP loopback stall; the
teacher-forced log-probability gate; fp32 logits.

## 2026-09-01 to 2026-09-02 — M6: generation, tokenizer, service

The incremental decode engine; the byte-exact tokenizer and the Jinja
chat-template interpreter; the deterministic scheduler; the
OpenAI-compatible service at world 1, then on the fabric behind the
admission journal; the decode step as a recorded graph with the
collectives as graph nodes; eleven optimization rounds on the T=1 step
(393 → 31.45 ms per token); the one-pass resident load and the per-rank
image cache (boot 15–25 s).

## 2026-08-29 to 2026-08-31 — M5: four-rank tensor parallelism

The RC/RoCE CollectiveBus with its slot pools, dual-lane striping and
watchdogs; the epoch-based roster; the sharded resident load; the fabric
TP=4 parity gate at real dimensions; the resident serving mode.

## 2026-08-27 to 2026-08-28 — M0 to M4

Platform, topology, transport and checkpoint facts; the core runtime, the
safetensors loader and the synthetic graph testbed; the KDA operators and
state manager; DSA/MLA sparse attention with its index pools; the
assembled 45-layer GLM forward with per-layer parity against torch
references.
