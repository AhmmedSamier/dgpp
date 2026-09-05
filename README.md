# DGPP Engine

DGPP is a C++/CUDA inference engine for serving the text path of
`unsloth/GLM-5.3-Flash-FP8` across four NVIDIA DGX Spark systems, with no
framework underneath it: the kernels, the RoCE collective bus, the tokenizer,
the chat-template interpreter, the scheduler and the HTTP service are all in
this tree.

Current status (2026-09-05): the engine serves. `glm_serve` boots a resident
TP=4 model on the four-node fabric in 15–25 s (per-rank image cache), answers
the OpenAI chat/completions contract over HTTP/SSE with rank 0 as the sole
ingress, and every rank executes an identical op stream by construction. The
service samples exactly at the checkpoint's defaults (temperature 1.0 /
top_p 0.95; the on-device verdict with the exact gather fallback, 2026-09-04)
at 33.2 ms/token on the plain graph — the greedy pace — and 25–28 ms/token
under MTP at one live request, with logprobs on the wire; temperature 0 is
the exact greedy path (36.4 ms/token on the four-node service, 2026-09-03).
Graph mode is adaptive:
with `--decode-graph [--mtp]` it lazily records a scalar graph for each
physical slot plus one fixed request-major batch, then selects scalar below
four live requests and the row batch at or above four. It supports up to eight
T=1 requests or four T=2 MTP requests. The final four-node curves are
30.97/31.67/41.76/76.18 tok/s at 1/2/4/8 live T=1 requests and
38.48/38.97/60.27 tok/s at 1/2/4 live MTP requests. This closes the original
fixed-width regression (12.55 T=1 and 19.32 MTP tok/s at one live request),
while retaining most of the 78.94/65.79 tok/s fixed-batch saturation result.
MTP's effective tok/s is workload-dependent: the Phase-2 prompt accepted
1.678 tokens/replay and therefore measured 25.99 ms/token, while the current
adaptive binary reproduces the Phase-1 single-stream controls at 21.93–22.21
ms/token on the exact 16-token Germany workload (previously 21.79), 21.83 on
a 32-token repeat, and 24.58 on its 200-token CUDA-graph explanation
(previously 24.81–24.85). Replay latency remains about 43.6–44.4 ms; the
apparent 21-versus-26 ms difference is draft acceptance, not an execution
regression.
All 22 final 256-token answers were identical across occupancy, graph-width
transitions, T=1, and MTP; every world's four op streams also matched. The
complete curves and controls are in `PLAN.md` and the measurement record.

The fast path also lives in `glm_gen_check`: the
decode step as one CUDA graph replay with the
collectives as graph nodes runs at 31.3 ms/token, and greedy speculative
decode with the checkpoint's MTP layer — verify, on-device pick, rollback,
draft and next tokens all inside one replay — at 22.45 ms/token with a
transcript identical to plain decode. Tool calls and reasoning are on the
wire (2026-09-04): the service accepts `tools`, `tool_choice`,
`parallel_tool_calls`, `reasoning_effort` and `chat_template_kwargs`, parses
the model's calls from the token ids on rank 0, and answers the OpenAI
`tool_calls` / `reasoning_content` shapes, one-shot and streamed;
`tool_choice` required / named / none and `parallel_tool_calls: false` are
guarantees through constrained decoding — a grammar of the tool-call format
masks the pick on every rank, exact on the device and the host alike — and
so is `response_format` (`json_object` / `json_schema`): a JSON grammar over
the vocabulary's token texts, the schema's types, keys, enums and bounds
enforced token by token, the mask proven equal to brute force; the same
machine types every tool argument by its property schema, and plain
`tool_choice: auto` arms the grammar, so every call the model writes is
well-formed. Stopping the service drains: in-flight requests retire
through the journal on every rank and their clients get a `server_shutdown`
error before the bus comes down. Admission is full-reserve by default, with
grow-on-demand as an opt-in policy that admits optimistically and sheds the
youngest request when the KV pool runs out, rank-identical by construction.
The prefill behind the time to first token is
being taken down (a 256-token prompt's first token went from 5 s to 0.65 s
and a 2048-token one from 19.8 to 3.1 s in steady state by 2026-09-05:
the MoE experts on one grouped tensor-core launch per matrix per layer
and no host sync, the bulk all-reduce split per segment and run by a
16-block cooperative kernel with paced senders — the fabric dropped
packets under the faster ones; the attention prefill tiles are next) and
the prefix cache is open; `PLAN.md` has the status per milestone and the designs
for what remains, `DESIGN.md` the contracts as built.

## Documentation

- `DESIGN.md` — validated architecture and state/collective contracts
- `PLAN.md` — implementation status, exit gates, and audit closure matrix
- `docs/checkpoint_budget.md` — generated checkpoint and decode-traffic audit
- `docs/measurements.md` — curated, reproducible platform results
- `benchmarks/README.md` — benchmark commands, scope, and interpretation

## Requirements

- DGX Spark/GB10 with CUDA 13 and an SM 12.1-capable compiler
- CMake 3.24 or newer and a C++20 compiler
- CUDA Runtime and cuBLASLt development files
- libibverbs headers/library for RoCE tools; disable with
  `-DDGPP_ENABLE_IBV=OFF` when unavailable (the GDR and RC probes are then
  omitted)
- Python 3.10 or newer for checkpoint tooling and its tests

## Build and test

```bash
# Warning-as-error configure, build, and CTest suite.
./scripts/ci-local.sh

# Optimized developer build.
cmake --preset release
cmake --build --preset release -j

# Host sanitizer variants.
cmake --preset asan
cmake --build --preset asan -j
ctest --preset asan

cmake --preset ubsan
cmake --build --preset ubsan -j
ctest --preset ubsan
```

Presets are `release`, `debug`, `asan`, `ubsan`, and `ci`. If clang-format is
installed, CMake also exposes `format` and `format-check` targets.

## Implemented tools

| command | purpose |
|---|---|
| `dgppctl info` | CUDA and platform facts |
| `micro_mem_bw -s 4096` | LPDDR5x streaming patterns |
| `micro_gemm_peak` | best-of-cuBLASLt-heuristics FP8/BF16 shape sweep |
| `micro_zerocopy` | pinned/device GPU bandwidth, contention proxies, flag latency |
| `micro_gdr_probe` | informational direct-device MR probe and copy profile |
| `micro_ibv_smoke info` | enumerate verbs devices |
| `micro_ibv_smoke serve PORT --once --dev DEV` | one cross-node responder session |
| `micro_ibv_smoke ping --peer IP:PORT --dev DEV` | signaled RC SEND echo latency |
| `micro_ibv_smoke bw --peer IP:PORT --dev DEV` | signaled RC SEND bandwidth; repeat peer/device for both lanes |
| `micro_ibv_smoke verify --peer IP:PORT --dev DEV` | ordered NIC DMA payload/doorbell → GPU hash validation |
| `kda_bench` | KDA decode state-traffic and layer timing profile (M2) |
| `gpt_doll --selftest` | synthetic eager/graph parity testbed |
| `glm_bind_check --config CONFIG --checkpoint-dir DIR` | validate a real GLM-5.3 checkpoint against the expected-tensor table (config parse, names, dtypes, shapes, FP8 scale pairing; headers only) |
| `glm_stream_check --config CONFIG --checkpoint-dir DIR` | stream real layers through the resident loader; bytes vs formula reconciled per layer |
| `glm_forward_check --config CONFIG --checkpoint-dir DIR --suite FILE` | curated reference suite: ISOLATED per-layer parity vs torch-reference dumps, head + routing agreement (DESIGN §7.5) |
| `glm_forward_check ... --trace-ids-file F --trace-out T` | engine-only real route-trace capture (deterministic) for the traffic model |
| `tools/route_trace_traffic.py TRACE` | route-trace traffic model: measured busiest-rank occupancy and corrected critical path (replaces the uniform-expert assumption) |
| `tools/checkpoint_audit.py [MODEL_DIR]` | regenerate inventory and checkpoint budget |
| `tools/make_shardspec.py MODEL_DIR` | generate the loader shard plan |
| `tools/kda_reference_dump.py` | KDA parity dumps: `selftest`, `gen-pure` (CI oracle), `gen-torch` (real checkpoint slices; needs torch) |
| `tools/dsa_reference_dump.py` | DSA parity dumps: `selftest`, `gen-pure` (CI oracle), `gen-torch` (real checkpoint slices; needs torch) |
| `tools/glm_reference_dump.py` | full-model parity dumps: `selftest`, `gen-pure` (CI oracle, mini checkpoint), `gen-torch` (real checkpoint, per-layer streams + router scores; `--layers N` for reduced budgets) |
| `roster_check coordinator/rank/selftest` | M5 control plane on real nodes: epoch-based roster startup, rank health, eviction on death/deadline; `selftest` is the loopback in-process smoke |
| `bus_check serve/ping/selftest` | M5 data plane on real nodes: RC/RoCE CollectiveBus — per-class slot pools, credit-grant RDMA writes, dual-lane striping, latency-under-bulk contention; `selftest` is the loopback in-process smoke; `--contend --soak-ms` is the duration-bounded soak |
| `bus_small_repro --pick-race` | the pick-path racer: the decode step's collective shape with an exact oracle after every collective under injected skew, loopback or fabric |
| `nic_regress mesh/pair/serve/selftest` | NIC→GPU visibility regression over every directed node pair on both lanes (rerun after driver/firmware changes) |
| `glm_tp_check --model ID --world W --rank R --peer HEAD` | fabric TP parity: final hidden/logits/routes/digests bitwise across ranks and against the loopback verdict dumps |
| `glm_shard_parity` | the sharded loader vs full-load + `GlmTpViews::bind`, bitwise on every bound surface |
| `glm_gen_check --model ID --chat TEXT [--system S] --steps N [--decode-graph] [--mtp]` | generation on the fabric (or `--text`, `--prompt IDS`): resident TP, distributed greedy pick, EOS stop; `--decode-graph` replays the step as one CUDA graph, `--mtp` adds speculative decode (see below); `--requests FILE` runs a JSONL manifest through the scheduler, `--sched-plan` prints the memory receipt without a GPU, `--teacher-file F` scores a text instead of generating, `--sampling-profile` adds the exact top-k-mass sizing probe, `--sample` (or any of `--temperature --top-p --top-k --min-p --repetition-penalty`, with `--seed N`) samples exactly at the checkpoint's defaults on the eager engines (with `--mtp`: the eager exact speculative sampler) instead of the default greedy loop, and `--step-timing` prints the per-phase budget |
| `glm_serve --model ID --port P [--world W --rank R --peer HEAD --journal-port J] [--max-concurrency N --kv-capacity T --queue-limit Q] [--decode-graph [--mtp] --graph-batch-min-live N] [--temperature X --top-p X --top-k N --min-p X --repetition-penalty X --seed N] [--sampling-candidates N] [--admission full|grow --admission-window N] [--reasoning-in-content]` | the OpenAI-compatible service: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/v1/metrics`; rank 0 is the HTTP ingress and journals admissions to the peers; sampling defaults come from the checkpoint's `generation_config.json` (temperature 1.0 / top_p 0.95 for GLM-5.3-Flash-FP8) with the flags overriding them per process, requests may set `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `seed`, `top_k`, `min_p`, `repetition_penalty`, `logprobs` and `top_logprobs` (exact, the OpenAI shapes), and `/v1/models` reports the effective defaults; chat requests may carry `tools`, `tool_choice` (`auto`/`none`/`required`/a named function) and `parallel_tool_calls` — every call is well-formed by constrained decoding, a grammar mask on the pick applied identically on every rank (DESIGN §10): a declared function name, the closed keys under `additionalProperties: false`, argument values typed by their property schema (JSON-typed properties under the JSON machine, enum strings from their texts, plain strings free; `function.strict: true` refuses a property outside the enforceable subset by keyword), and required, named and single-call turns as guarantees — `response_format` (`text` / `json_object` / `json_schema` with `name`, `schema`, `strict`: the content is one JSON text conforming to the schema's `type`, `properties` / `required` / `additionalProperties`, `items` / `minItems` / `maxItems`, `enum` / `const`, `anyOf`, enforced by the same masks; a strict schema outside that subset is refused naming the keyword, a non-strict one falls back to `json_object`; not combinable with `tools`), `reasoning_effort`, `chat_template_kwargs` (`clear_thinking`, `reasoning_effort`), assistant `tool_calls` and `tool` messages, all rendered through the checkpoint's template, and the response carries `reasoning_content` (or, with `--reasoning-in-content`, the reasoning folded into `content`), `content` and `tool_calls` parsed from the token ids on rank 0 with `finish_reason: "tool_calls"` (DESIGN §11); the eager engines and the `--decode-graph` graphs, with or without `--mtp`, sample exactly (DESIGN §10, the on-device verdict with the gather fallback between windows; under MTP the exact speculative accept test, DESIGN §9); graph mode requires the fabric and `max-concurrency * (mtp ? 2 : 1) <= 8`, warm-captures every graph variant at startup (the journal's `warm` record starts it on every rank together), and batches at `--graph-batch-min-live` live requests (default min(4, max-concurrency); must be in [1, max-concurrency]) |
| `scripts/fabric_run.sh [--stage-file F] [--fetch-logs] [--node-probe] -- APP-ARGS` | launches any app on the four-node fabric with the rendezvous discipline (head first, peers fire-and-forget, verified by pgrep, swept on head death); collects rank-invariant md5s and bus stats |
| `scripts/serve_run.sh up/down/status` | boots/stops the serving world (`DGPP_SERVE_KNOBS` overrides the engine flags on every rank, e.g. `--max-concurrency 1 ... --decode-graph --mtp`); `down` fetches and md5s every rank's op stream |
| `scripts/roce_counters.sh snapshot\|diff` | the fabric's RoCE hardware counters (sequence errors, adaptive retransmissions, CNPs, NIC ingress discards) per node and device, and the deltas between two snapshots — the wire-side view of a collective run |
| `scripts/serve_bench.py HOST PORT MAX_TOKENS LABEL [PROMPT]`, `scripts/serve_pace.py RANK_LOG [--waves]` | the service's pace: client-side SSE stamps; server-side per-request pace; and, with `--waves`, the steady peak-occupancy replay latency, tokens/replay, and aggregate tok/s used by the Phase-2 gate |
| `scripts/fabric_xcript.py`, `scripts/fabric_logprob.py`, `scripts/fabric_sampling_profile.py`, `scripts/fabric_xrank.py` | the judges (first divergence by bf16-ulp margin; teacher-forced perplexity delta; sampling-width evidence) and the cross-rank step/stall reader — see "Judging a numerics change" |

The GDR probe exits successfully when the probe itself completes, including
the expected “unsupported” result on GB10. It does not prescribe a bounce
copy; CollectiveBus uses registered pinned memory consumed directly by the
GPU.

## Tests

CTest currently runs 33 entries:

- host unit cases covering logging/tracing, JSON, arenas, safetensors,
  FP8, shard plans, the HF cache resolver, the sampler against its
  centralized oracle, KDA/DSA geometry contracts (against DESIGN §7.2's
  transcribed literals), route-trace golden bytes shared with the python
  reader, the MoE route-flip certifier's rejection paths (near-tie
  accepted; far-rank, zero-noise, own-scores-inconsistent, and duplicate-id
  divergences rejected), and the TCP/roster control plane (seal, epoch
  bumps, eviction by death and by deadline, rejection reasons, coordinator
  loss);
- the M6 host suites: the tokenizer (55 byte-exact goldens, hash-keyed),
  the chat template (26 goldens + 8 refusal negatives), the scheduler
  (isolation, determinism, cancellation, bounded queue, tick ≡
  run_to_completion), the HTTP server's limit ladder, the OpenAI shapes
  over real HTTP with a fake engine, and the admission journal with two
  real peer loops over localhost;
- synthetic CUDA graph/eager parity;
- the KDA operator suite: conv/recurrent kernel parity against host
  fp32/fp64 references, chunked-vs-unchunked bitwise equivalence, decode
  graph replay, snapshot round-trip, head-slice TP readiness, and
  reference-dump parity against the pure-python oracle;
- the DSA suite (M3): pool compression and tail-ring continuation verified
  bitwise via hard-max gates (including multi-token decode == single-token),
  a 1,100-case bitwise pooled top-k fuzz around pool boundaries plus
  exact-tie and 512th-boundary constructions, the fused decode select (MTP
  multi-row, grid-size invariance, 100k-pool long-context stripes, graph
  capture/replay with changed position), split-KV absorbed attention against
  the host oracle at TP1/TP4 with empty-row and head-group coverage,
  latent/gather block-table round trip, multi-request decode with padding
  rows, kpool=2 generality, and the layer tests: state-pool block
  allocation and byte accounting at deployment scale, prefill/chunked-
  prefill/decode parity against the oracle with selection-aware near-tie
  certification (at select_k=16 and select_k=8), decode graph replay
  bitwise across positions, TP2 head-slice vs TP1, and a real-geometry
  chunked prefill + decode smoke;
- DSA reference-dump parity: a pure-python oracle dump (bit-exact fp8 codec
  cross-checked against the C++ encoder) exercised through the full layer —
  latent cache bitwise, index cache within one e4m3 ulp, top-k exact — plus
  the torch backend against real checkpoint slices (executed on this box at
  32 and 2,052 tokens, the latter crossing the top-k horizon with zero
  flips); any flipped row is certified as a measured boundary near tie by
  the audit, never absorbed by tolerance;
- the M4 assembly suites: the mHC stream module (Sinkhorn mixing, stream
  update, final mean vs the double oracle), the MoE router/expert/shared
  path (sigmoid router with the noaux tie rule, swiglu asymmetries, bf16
  accumulation order, near-tie certification on synthetic corpora), the
  scale-aware GEMM against cuBLASLt BF16 references and real-checkpoint
  block edges, and the assembled-forward chain: a synthetic mini-checkpoint
  written on disk, a full-stack pure-python reference over the same
  weights, then the engine compared (hidden ulp budgets, top-k exact, route
  ids/weights, determinism);
- CUDA system-scope flag ordering, payload visibility, inactivity watchdog,
  and post-watchdog recovery;
- the M5 CollectiveBus data plane (needs the fabric + ibverbs): loopback
  scenarios with real RC QPs — payload integrity via fold-hash, credit
  recycling, dual-lane striping asserted on both sides, latency under bulk
  contention, watchdog failures, config-mismatch rejection, the graph era
  with the mixed-era interlude, the bulk RS+AG machine, and orderly stop —
  plus the app-level smoke;
- the TP forward over loopback buses (`glm_tp_test`): per-layer isolated
  parity vs the world=1 oracle at worlds 2 and 4, decode sessions, the
  greedy generation loop, the device pick, the one-graph MTP step in
  lockstep with the eager speculator, and the T=1/MTP serving graph adapters
  through the real scheduler (including slot reuse, the in-graph draft
  checked against the eager speculator after every replay); the pick/spec
  kernels against their
  host oracles (`glm_pick_test`); the GEMV cores (`bf16_gemv_test`) and the
  loader's resident-image round trip (`glm_loader_test`);
- Python checkpoint classification, exact expert-occupancy tests, and the
  route-trace traffic-model contract.

`glm_tp_test` and `bus_test` run with `CUDA_DEVICE_MAX_CONNECTIONS=32`; the
prefetcher stays enabled everywhere. The captured decode graph is
kernels-only by contract, checked at every capture site
(`glm_check_decode_graph`): a memset/memcpy node executes on the process-
shared copy-engine queue, where one rank's queued dependency wait blocked
the peer rank's node behind it while its own collective spun waiting on
that peer — the batched-MTP loopback stall, root-caused from an nsys node
trace and fixed on 2026-09-03 (`docs/batched_mtp_graph_stall.md`; the
loopback gates now pass with prefetch on at 1 and 32 connections).

All CUDA suites are verified clean under `compute-sanitizer` memcheck (full
suite every milestone; racecheck and initcheck per-phase on the tests
exercising new kernel shapes). The multi-rank loopback worlds of
`glm_tp_test` are the exception: their budgets are liftable
(`DGPP_TEST_BUS_TIMEOUT_MS`, `DGPP_TEST_CONSUMER_DEADLINE_S`,
`DGPP_TEST_WAIT_TIMEOUT_MS`), but even lifted, the instrumented eager prefill
outlasts the first collective's kernel deadline (baseline and current alike,
2026-09-04 record entry), so those worlds report 0 errors and exit on a
budget rather than completing. The sanitizer findings that motivated this
(speculated loads past short-circuit guards, shared-memory reuse races
that pass by scheduling luck, undersized test buffers that made a graph test
pass vacuously) are pinned in `DESIGN.md` §12.

Cross-node RoCE and NIC→GPU checks are intentionally manual/deployment tests;
they require a peer and are documented under `benchmarks/README.md`.

### Deploying a serving rank: memory

A resident rank owns its box. The model is ~82 GiB of the 121 GB, and the
serving apps (`glm_serve`, `glm_gen_check`) configure themselves for that
without any privileged setup on the node:

- the constructor checks that the resident footprint (+ 8 GiB headroom)
  fits the device's free memory and fails immediately with a clear message
  if it does not — never three minutes into a load;
- the loader reads each source tensor exactly once (prefetch, copy, drop),
  so the page cache stays under ~10 GB during the load and the box never
  reaches its memory watermark; the checkpoint's mmaps are released the
  moment the last layer is on the device (`GlmLayerStream::release_sources`);
- the process *tries* to lock its memory (`mlockall(MCL_CURRENT)`, before
  the model is constructed) as a safety net against swap-in faults in the
  decode loop. This is optional: with the one-pass loader a rank with the
  pin off measured identically (p99 46 ms, 0 stalls, no swap traffic over
  1000 steps). A finite `RLIMIT_MEMLOCK` is logged, not warned about;
  `DGPP_MLOCK=off` skips the attempt.

Nothing else on the node needs setting. In particular a locked GPU clock
(`nvidia-smi -lgc`) is **not** required: the governor sits at 2400-2560 MHz
throughout decode on its own and the measured step distribution is the same
locked or unlocked. NTP between nodes only matters for reading logs side by
side, and `scripts/fabric_run.sh --node-probe` records each node's clock
offset per run so even that works without it.

### Deploying a serving rank: the resident image cache

The first start of a resident rank builds its layers from the checkpoint
(slice, stage, dequantize, pack) and writes the finished device bytes to
`~/.cache/dgpp/resident/<key>.img` on that node (~82 GiB per rank for GLM;
the key covers the checkpoint's shard headers, `config.json`, world, rank,
head sharding and the loader's format version, so a stale image can never
load by accident). Every later start streams that image instead with
O_DIRECT reads at the drive's line rate — 15-25 s to a ready model against
~4.5 minutes from the checkpoint. The boot digest is cached beside it
(`<key>.digest`). Knobs:

```
DGPP_RESIDENT_CACHE=off            disable (always build from the checkpoint)
DGPP_RESIDENT_CACHE_DIR=/path      put the images somewhere else
DGPP_RESIDENT_CACHE_VERIFY=1       re-fold every blob on read (a pass over 82 GiB)
```

Delete the file to force a rebuild; the loader's log line says how many
layers were restored versus captured on each start.

`scripts/fabric_run.sh --node-probe` samples each node's reclaim/swap/GPU
counters at 1 Hz for the run; `scripts/fabric_xrank.py LOGDIR` reads the
fetched logs and reports host gaps, stall windows, and step distributions per
rank.

### Judging a numerics change

Kernel work is allowed to change floating-point reduction order when it buys
latency, so two builds can legitimately produce different bits. These evidence
tools judge numerical changes and size sampling; all read run directories through
`scripts/fabric_logs.py`, the shared parser for the per-rank
`[gen]`/`[tf]`/`[sample_mass]` step lines — start there when writing the next
one:

- `scripts/fabric_xcript.py REF_DIR NEW_DIR` — for ordinary generation runs:
  finds the first token where the transcripts diverge and reports the global
  top-2 logit margin there in bf16 ulps. A flip at ≤ 1-2 ulp is a near-tie;
  a flip at a wide margin is a defect.
- `scripts/fabric_logprob.py NEW_DIR REF_DIR` — the quantitative gate. Run
  both builds with `--teacher-file benchmarks/teacher_text.txt` (the app
  scores the text's own tokens instead of generating; `--stage-file` puts the
  text on every rank), and the tool joins the ranks' vocabulary slices into
  log p(token) per position and reports the text's perplexity, the per-token
  deltas with a standard error, and a PASS/FAIL against a mean-NLL bound
  (default 0.02 nat ≈ 2% of perplexity):

  ```
  scripts/fabric_run.sh --stage-file benchmarks/teacher_text.txt -- \
      --model unsloth/GLM-5.3-Flash-FP8 --text "Encyclopedia article." \
      --teacher-file benchmarks/teacher_text.txt --decode-graph
  scripts/fabric_logprob.py /path/to/new_logs /path/to/ref_logs
  ```

  A run of the same binary twice must show a delta of exactly 0 (the decode
  path is deterministic). The logits are fp32 (the head's accumulators,
  unrounded), but the residual stream feeding the head is bf16, so
  single-token deltas of ~0.1 nat are normal between builds that differ in
  reduction order; the mean over the text is the signal. Three
  texts ship: `teacher_text.txt` (556 tokens, a quick look),
  `teacher_text_hard.txt` (7,331 tokens of unmemorized technical prose,
  perplexity ~10.8 — the sensitive one; use it for the verdict) and
  `teacher_text_memorized.txt` (6,549 tokens of Conan Doyle, perplexity
  1.04 — the confident regime). Expect a handful of positions per thousand
  to move by more than 1 nat between any two rounding-level builds: a
  router's top-k boundary flipped an expert there. The tool bounds the
  *rate* of those, not the worst one.

- `scripts/fabric_sampling_profile.py DIR...` — the M6 sampling-width gate.
  Make one teacher-forced run per shipped text with `--sampling-profile` and
  fetched rank logs, then pass all three directories together:

  ```
  scripts/fabric_run.sh --fetch-logs \
      --stage-file benchmarks/teacher_text.txt --log-dir /tmp/mass-quick -- \
      --model unsloth/GLM-5.3-Flash-FP8 --text "Encyclopedia article." \
      --teacher-file benchmarks/teacher_text.txt --sampling-profile \
      --decode-graph
  # Repeat as /tmp/mass-hard and /tmp/mass-memorized with the other texts.
  python3 scripts/fabric_sampling_profile.py \
      /tmp/mass-quick /tmp/mass-hard /tmp/mass-memorized
  ```

  The tool requires identical contiguous measurements on every rank, reports
  top-{32,64,128,256} mass and exact-gather fallback rate at T=1/top_p=0.95,
  and chooses the smallest k at or below 1%. The serving-side sweep is
  `scripts/serve_width_sweep.sh [OUT_DIR] [WIDTHS] [MODES]` (boots the world per
  width and graph mode over `--sampling-candidates`, the same seeded requests
  every time) with `scripts/width_sweep_report.py OUT_DIR/sweep.tsv` fitting
  the cost of one fallback. The extra diagnostic gather is
  intentionally outside the production path, so these runs are not throughput
  measurements.

### Speculative decode with the MTP layer (`--mtp`)

The checkpoint ships a multi-token-prediction layer (`num_nextn_predict_layers
= 1`): one extra DSA + MoE block that, given the main stack's hidden at
position q and the token chosen for q+1, guesses the token at q+2.
`glm_gen_check --mtp` uses it for greedy speculative decoding on the fabric:

```
scripts/fabric_run.sh -- --model unsloth/GLM-5.3-Flash-FP8 \
    --chat "Write a history of the Roman Republic." --steps 300 \
    --decode-graph --mtp
```

Every step is **one CUDA graph replay** that carries its own control flow:
the two-row verify of `[next, draft]`, the pick behind the head (each
rank's argmax gathered through one bus collective and judged on the
device), the commit (a rejected second row is rolled back on the device —
the KDA/DSA kernels snapshot their state after every speculative row, so a
retraction is a predicated copy per state family — and the position
advances), the MTP block over the accepted rows (a fixed two-row batch; the
second row is padding after a miss), its own pick, and the next step's
tokens written on the device. The host launches, waits, reads two small
pinned verdicts and logs. **The transcript is exactly the plain loop's** —
the verify rows are bitwise the single-token rows, and
`scripts/fabric_xcript.py PLAIN_DIR MTP_DIR` must print `IDENTICAL`; MTP
only changes what a token costs. Measured (2026-09-03, TP=4): 88.7% of
drafts accepted on coherent text, 1.89 tokens per step, **22.45 ms/token
effective vs 31.3 plain** (−28%). The step is 42.4 ms: the second verify
row costs ~7 ms because its MoE experts are extra DRAM bytes (only the
experts both rows share are read once — the slots run in expert order so
the second read is an L2 hit), the draft block ~2 ms. Acceptance drops on
incoherent text (63% on the post-EOS rambling `--no-eos` produces), and
below ~45% speculation stops paying; the summary line reports the rate.
Every rank computes the verdict itself from an identical gathered table; a
rank whose table was corrupt is caught at the next pick by a digest every
rank carries (`GlmDevicePicker`). Without `--decode-graph` the same step
runs eagerly (26 ms/token). The draft layer adds ~7.3 GiB per rank to the
resident footprint. Serving uses adaptive scalar and row-batched variants
with up to four MTP requests per replay. At concurrency 1 on the four-node service
(2026-09-03), a replay is 43.6–44.1 ms and carries 1.69–1.88 tokens by the
text's acceptance, 21.8–26.0 ms/token. The original always-eight-row graph
delivered 19.3/35.9/65.8 tok/s at 1/2/4 live requests; adaptive selection
raises that curve to 38.48/38.97/60.27, using the batch only at four. The
corresponding adaptive T=1 curve is 30.97/31.67/41.76/76.18 tok/s at
1/2/4/8 live requests. Every measured world's four op streams were
identical, and the final 256-token transcript was identical across modes,
slots, occupancy, and scalar↔batch transitions. Per user, those curves are
32/63/96/105 ms per token at 1/2/4/8 live T=1 requests and 26/51/66 at 1/2/4
live MTP requests: below the crossover each user pays the live count times
the scalar replay, at the crossover the batch is faster for the user too
(four scalar replays would be ~129 and ~104 ms per token).
