# Issue #36: full-model investigation and engine watchdog

PR #40 fixed automatic prefill chunking, shutdown deadlines, a reducer stream
wait, and misleading collective diagnostics. It still left serving indefinitely
blocked if a CUDA/model call outside a collective never returned. The follow-up
adds an independent 120-second engine progress watchdog. It exits rank 0 with
status 2 without teardown; journal closure terminates its peers. An external
supervisor must restart the failed deployment.

The watchdog reads atomic pass and prefill progress counters. It covers engine
passes, journal broadcasts and stats/log publication, including admission-time
CUDA waits. Idle serving and long prefills that keep advancing are allowed.
Startup is outside its scope. The existing 30-second signal-triggered shutdown
watchdog remains in place.

## Hardware and method

September 23, 2026, DGX Spark GB10, CUDA 13. The existing four-node deployment
was stopped for isolated tests. Qwen ran on one node; GLM ran on two nodes with
`DGPP_ROCE_DEVICES="rocep1s0f0 roceP2p1s0f0"` and GIDs `3 3`.

- GLM: `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`, revision
  `17b8d4d83e33e836e9549d7e1a8d25b5117203c1`; reported settings:
  KV capacity 491520, concurrency 2, FP8 KV, BF12 weights, graph decode, MTP,
  prefix cache 1.5 GiB. The candidate cancellation/deferred-admission run disabled
  EOS so the reservation holder could stay alive. Other GLM runs used model EOS.
- Qwen: `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision
  `fc694b54fb0174e0913e6adf86691ef85a4ead47`; KV capacity 262144, concurrency 2,
  BF12 weights, FP8 dense weights, mmap n-gram table, graph decode, MTP,
  prefix cache 1.5 GiB. This family uses BF16 KV despite the FP8 config knob.
- Long input: 200000 content tokens, 200012 after the chat template; short input:
  19 tokens, with 24 output tokens. Real model tokenizers, direct HTTP, proxies
  bypassed. `curl --max-time 5` disconnected during incomplete long prefill and
  returned 28; a short request followed immediately. Both stream modes tested.
- Legacy comparison: saved deployment binary `a895db96bb17.dirty`, explicit
  prefill budget 0. It is not an exact rebuild of the reporter's `c6ca191863d6`.
  The relevant HTTP, scheduler, GLM decode and collective source files match
  between those commits; GenerationService changes concern thinking parsing.
- Automatic chunking: resolved budget 256. Qwen automatic tests used PR #40;
  GLM automatic tests used the new watchdog build. The new build also ran Qwen
  with explicit budget 0 to test progress beyond the watchdog deadline.

## Results

| Test | Result |
| --- | --- |
| Legacy Qwen, abandoned 200K prefill | Prefill 202.9 s; fresh request waited 199.016 s |
| Automatic Qwen, non-streaming / streaming | Fresh request 0.130 / 0.129 s; one cancellation each |
| Legacy two-node GLM, abandoned 200K prefill | Prefill 748.125 s; fresh request timed out at 600 s; engine recovered unaided after prefill finished |
| Candidate two-node GLM, non-streaming / streaming | Fresh request 0.971 / 0.972 s; unfinished prefill cancelled at a chunk boundary |
| Candidate GLM deferred admission | Queued request promoted after its peer retired, then cancelled during prefill; fresh request 0.738 s |
| Candidate Qwen, explicit unchunked prefill | Healthy prefill completed in 204.686 s without a false 120 s timeout |
| Injected CUDA wait, PR #40, one-rank GPU fixture | Still stuck after 125 s; required test cleanup |
| Same injected wait, candidate, one-rank GPU fixture | Exit 2 without a termination signal; client disconnected at 120.021 s |
| Same injected wait on both ranks, full GLM candidate | Rank 0 exit 2; rank 1 exit 3; client disconnected at 120.022 s; no termination signals |

The deferred-admission test used a short streaming request reserving 400000
output tokens: 3129/3840 blocks were occupied, leaving insufficient space for
the second request's 1563 blocks. Cancelling the holder promoted the queued
200K prompt; cancelling that prompt retired it too. This exercises pool
reservation, deferral, promotion and in-flight cancellation. It does not replay
the reporter's exact cached 283K first prompt.

Both normal GLM runs stopped cleanly with identical operation streams across
ranks. The candidate cancellation tests returned pool usage to its two cached
blocks and left no active, queued or prefilling requests. The legacy runs
counted disconnects promptly too, but could not retire cancelled work until the
whole prefill returned. Snapshot counters remained stale while actual prefill
positions advanced; those counters alone do not establish a permanent hang.

## Failure injection and regression coverage

The test-only preload intercepts `cudaStreamSynchronize`. After a successful
warm-up, an external flag makes the next call wait indefinitely. The PR #40
host stack places the wait in `GraphEngineAdapter::push_spec`, called by
`configure_sampling` during scheduler admission, before prefill progress begins.
No collective completion watchdog covers this call. This reproduces the missing
failure bound without claiming to reproduce a physical GPU/driver failure.

Host regressions passed: `engine_watchdog_test`, `shutdown_watchdog_test`,
`serve_test`, `scheduler_test`, and `fabric_serve_test`. Watchdog tests cover no
signal, a full stderr pipe, idle serving, repeated passes, long progressing
prefill, unchanged token positions, and progress that subsequently stops. The
server and relevant targets built with the CI preset and warnings as errors.

## What this establishes and what it does not

The full models reproduce abandoned-prefill blocking, and automatic chunking
fixes that behavior. The independent watchdog closes the confirmed gap that
allowed a stopped engine to survive beyond all collective deadlines without a
signal. It is a bounded fatal exit, not recovery of a failed CUDA context.

The original permanent hardware stall was not reproduced spontaneously. Its
precise initiating cause and the reported zero cancellation counter remain
unconfirmed; direct curl disconnects were counted on both old and new builds.
Recovering bulk stall dumps occurred during the legacy GLM run, but their
latency fields do not identify a failed bulk posting operation. The exact
original request, connection route, full logs and stacks would be needed to
attribute that separate detail. These results do not claim to eliminate all
transient fabric stalls or diagnose the underlying driver.

`summary.json` records binary hashes, timings and normal operation-stream hashes.
The complete local evidence (configs, metrics samples, logs, request bodies,
operation streams, fault shim and scripts) is retained under this directory in
the investigation workspace, with the large raw artifacts excluded from Git.
