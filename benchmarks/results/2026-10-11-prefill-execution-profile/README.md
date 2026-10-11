# Issue 104: execution profiling, still unresolved

[Issue 104](https://github.com/HawkBearPig/dgpp/issues/104) has been reopened.
PR 107 contained the throughput regression by disabling early admission by
default. It did not deliver the requested combination of prompt admission and
preserved throughput. Closing the issue at that point was premature.

These measurements locate the extra execution cost. They do not validate a
new scheduler or a throughput-neutral implementation. No runtime change was
made during this investigation.

## Matched workload and native timings

One GB10, four request slots, Qwen3.8-Flash-Next NVFP4 routed experts, FP8 dense
weights, BF16 KV, MTP3. The executable is the final PR 107 runtime
(`3426204f065a20e2b8de231153567f1aff60949c`), SHA-256
`fc677569137b13586254c9823c500baac2b98450d78bdb536bb920ecb3f11399`.

Each fresh deployment warms up before collection. The measured workload is
`scripts/serve_admission_bench.py`'s `sustained_idle`, seed 104: a 12,991-token
long prompt, then eight 18-token prompts arriving at 250 ms intervals. The
long request emits one token and each arrival emits 128. Each matched pair
has identical request bodies, response bytes, and usage: 13,135 prompt tokens
and 1,025 output tokens. The arrival is triggered by observing native prefill
progress, as in the previous campaign.

| Busy / idle caps | Early admission | Total wall, s | Native prefill, s | Native decode, s | Decode passes |
|---|---|---:|---:|---:|---:|
| 256 / 256 | Off | 21.103 | 15.041 | 5.959 | 66 |
| 256 / 256 | On | 24.561 | 16.242 | 8.137 | 96 |
| 256 / 2048 | Off | 14.237 | 8.367 | 5.812 | 66 |
| 256 / 2048 | On | 22.110 | 14.028 | 7.901 | 96 |

With equal caps, early admission adds 1.200 seconds of native prefill time and
2.177 seconds of native decode time. With split caps, those increases are
5.661 and 2.089 seconds. Total wall time also includes frontend work and
benchmark drain/publication waits. These are single profiled runs, not a new
performance acceptance campaign or a statistical zero-regression claim.

## What the GPU traces add

Nsight Systems collected CUDA API activity and individual CUDA graph nodes.
Collection started after model loading, graph construction and a 32-token
warmup, then stopped after the measured workload drained. The trace extractor
rejects a capture containing `cudaStreamBeginCapture`. Graph nodes represent
decode replays; kernels outside graphs include prefill and its bookkeeping.
These are execution categories, not per-request NVTX ranges.

Kernel execution occupied 97.1%, 97.1%, 97.2% and 97.8% of the first-to-last
kernel intervals for equal/off, equal/on, split/on and split/off respectively.
This rules out a large period of completely idle GPU time as the explanation.
It does **not** establish full SM occupancy, bandwidth saturation or the
amount of possible overlap between different kinds of work.

For equal caps, summed decode-graph kernel durations increased from 6.897 to
10.121 seconds. The largest increases were:

| Decode kernel family | Increase in summed kernel durations, s |
|---|---:|
| NVFP4 routed-expert gate/up | 1.053 |
| NVFP4 routed-expert down | 0.503 |
| Weight prefetch | 0.398 |
| FP8 streaming matrix/vector product with FP32 output | 0.339 |

Those sums include overlapping CUDA streams and must not be added to native
elapsed timings. They identify where additional device work occurs; the
native decode increase remains 2.177 seconds.

The split-cap prefill trace explains why the binary busy/idle policy costs
much more than the short prompt's token count suggests:

| Eager kernel family | Early off, summed s | Early on, summed s | Instances off / on |
|---|---:|---:|---:|
| FP8-to-BF16 weight conversion | 0.437 | 2.260 | 3,640 / 19,240 |
| NVFP4 grouped gate/up products | 1.397 | 3.112 | 864 / 4,416 |
| NVFP4 grouped down products | 1.090 | 2.041 | 432 / 2,208 |

The same prompt work invokes many more weight-conversion and expert kernels
once an early decoder selects the smaller busy cap. Spending every allocated
token cannot remove that repeated work. The equal-cap runs also retain the
30 extra decode passes, independently of the busy/idle cap difference.

## Consequences for the implementation

A budget-only change has not been demonstrated to meet the acceptance bar.
Keeping the idle cap while decoding could reduce repeated prefill work, but
it also changes the decoder's maximum interruption and needs direct latency
validation. Smaller decode batches still have their own repeated model work.

The execution path must be investigated for opportunities to share work
between prefill and decode. Qwen's existing grouped cold-prefill surface is
not a validated mixed prefill/decode operation: its common matrix operations
use prefill dispatch, while decode uses its own dispatch, speculative row
snapshots, rollback and persistent feeds. Merely concatenating rows would
need numerical and state validation before any performance result counted.
This is an implementation direction, not a proven solution.

Acceptance still requires prompt short-request service, total completion time,
incumbent decoder latency, output correctness, cancellation and reservation
safety under unchanged concurrency, KV capacity, MTP settings and workload.
Issue 104 stays open until that combination is demonstrated.

## Evidence and reproduction

`measurements.json` contains the exact configurations, benchmark arguments,
native deltas, request/output hashes, client update timings, per-kernel counts
and durations, interval unions and SHA-256 hashes of the raw evidence.
`extract.py` regenerates it from the four local profile directories. The raw
Nsight reports and SQLite exports are retained under
`build-pr104-shared/profile-w1-{256,2048}-early{0,1}/`; binary traces are not
checked into Git. Collection order was equal/off, equal/on, split/on,
split/off. This is not a randomized or replicated timing study.

The profiling commands recorded in `receipts.json` use
`nsys launch --trace=cuda --cuda-graph-trace=node` through the existing cluster
launcher's `--head-wrap`. After warming through `Bench.stream`, `nsys start`
enables collection with CPU sampling and context-switch tracing disabled;
`Bench.run('sustained_idle', 0)` runs the measured case; `nsys stop` writes the
report, and `nsys export --type=sqlite` provides the extractor input.

Production was idle before the campaign. The original production binary and
resolved configuration were restored afterward; live executable hashes were
verified on all four ranks and HTTP health was checked. `production.json`
records that verification. There are no benchmark servers or profiling
sessions left running. The four final `nsys shutdown` commands returned 1
because application shutdown had already ended those sessions ("There are no
active sessions"); capture stops, exports and server shutdowns succeeded.
