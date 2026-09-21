# FP8 vocabulary-head end-to-end comparison, 2026-09-20

The FP8 vocabulary-head dispatch raised measured end-to-end throughput from
**129.28 to 143.84 tokens/s (+11.26%)** on two DGX Sparks at concurrency 4
and MTP depth 3. The concurrency-1 control was effectively unchanged:
74.94 to 75.06 tokens/s (+0.16%). Both builds completed the same number of
requests and output tokens. Greedy text varied, including between repeats
of the unchanged baseline: this is a fixed-prompt, fixed-output-budget
throughput result, not an identical-token-path or quality-equivalence claim.

## Workload and controls

- Model: `nvidia/Qwen3.8-Flash-Next-NVFP4`, tensor parallelism across two GB10
  Sparks; FP8 dense weights, mmap n-gram table, BF16 KV, CUDA decode graphs.
- Server capacity 4; MTP depth 3; `graph_batch_min_live=2` (the default for
  this capacity); `DGPP_DENSE_GEMV_ROWS=4`; KV capacity 65536. Prefix caching
  was disabled and every measured request reported zero cached prompt tokens.
- The five fixed classes from `scripts/serve_load.py`: prose, code, JSON,
  math and chat. The corpus check against `fabric_mtp_classes.sh` passed.
  Prompt selection matches the existing C1/C4 sweep: prompt 0 for C1, then
  the four distinct class prompts starting at prompt 1 for C4.
- Greedy, thinking off, `max_tokens=256`. All **300 measured requests**
  completed successfully with exactly 256 output tokens and a length finish.
- Each epoch warmed with four concurrent mixed prompts (256 output tokens
  each), then ran three repeats per class at C1 and C4. Warm-up is excluded.
- Order: baseline A1, candidate B1, candidate B2, baseline A2, with a fresh
  two-rank process restart for each epoch. There are six measurements per
  class/concurrency/build, spanning two process epochs.
- Primary rate: the sum of server-reported completion tokens divided by
  summed client phase wall time, including prefill, HTTP/SSE handling and
  request completion. SSE chunk count is not treated as token count.

The candidate is `8d37f87ba5910814891c020792de9fc0561d652b`. Both variants
use the native `release` preset (CUDA 13, SM121a, production optimization
settings). The baseline was made by restoring only
`src/models/qwen/forward.cpp` from upstream parent
`444441feed0c4d7c256ff7ff7868b587823b737f`, rebuilding, and saving a separate
binary. The only production-code difference is the vocabulary-head dispatch;
the baseline also has a `.dirty` version stamp. The candidate's Release
focused gates passed before measuring.

[Binary identities](2026-09-20-qwen-fp8-head-e2e/raw/binaries.json):

- Baseline SHA256: `eed7c982f7b90fd4c0a29f958b94eb4a086fc4a3256c1a7e2d778216b18b3d59`.
- Candidate SHA256: `15ef71781b8079dc022cd26be919b824be73d8fcd329b3de483cde905192fe76`.

## Results

Pooled across all five classes and both epochs of each build:

| Live requests | Baseline tokens/s | Candidate tokens/s | Change | Tokens per build |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 74.94 | 75.06 | +0.16% | 7680 |
| 4 | 129.28 | 143.84 | **+11.26%** | 30720 |

For C4, the same 30720 output tokens took 237.624 seconds of measured phase
wall time in the baseline and 213.570 seconds in the candidate, a 10.12%
reduction in elapsed time. Throughput improvement and time reduction use
different denominators.

C4 per-class medians, six phases per build:

| Class | Baseline tokens/s | Candidate tokens/s | Change |
| --- | ---: | ---: | ---: |
| Prose | 102.81 | 114.17 | +11.05% |
| Code | 146.03 | 161.04 | +10.28% |
| JSON | 172.07 | 191.86 | +11.50% |
| Math | 146.53 | 162.37 | +10.81% |
| Chat | 106.11 | 120.15 | +13.24% |

The pooled C4 gains were **11.83%** for A1 versus B1 and **10.69%** for A2
versus B2. Both candidate epochs were faster than both baseline epochs.
The final baseline was 0.79% faster than the first; the observed gain did
not disappear when the baseline ran last. These are descriptive results
from two process epochs per build, not a confidence interval or a claim
about other models, concurrency levels, hardware or output lengths.

GPU temperature rose during the campaign. Before/after observations are
retained in [sensors.csv](2026-09-20-qwen-fp8-head-e2e/raw/sensors.csv);
clocks were not locked. These snapshots do not measure energy per request.

## Calibration and transcript limits

Before any A/B timing, six repeated unchanged-baseline C4 mixed-prompt
phases had a throughput coefficient of variation of **1.76%**, but their
greedy texts were not all identical. The initial protocol stopped there and
restored production. A second bounded calibration using the supported
`graph_batch_min_live=1` setting also varied in text (timing CV 2.94%). That
setting was discarded; the final ABBA run uses the original default 2.
The source can switch numerical paths as batch occupancy changes, but this
campaign did not establish the cause of every observed text difference.

Because the baseline itself failed the proposed identical-text gate, the
measurement question was explicitly revised before the A/B run: compare
throughput for the same prompts and fixed output budgets, retain all output
variation, and make no identical-token-path or quality-equivalence claim.
No prompt classes or measured A/B phases were discarded.

- All 30 paired C1 requests had identical output text and token counts.
- At C4, 59 of 120 cross-build request pairs had identical text; all 120
  pairs had identical output-token counts.
- Comparing baseline A1 with baseline A2, 31 of 60 C4 request pairs had
  identical text. Candidate B1 versus B2 also matched in 31 of 60 pairs.
- Across six repeats, 16 of the 20 C4 prompt groups varied within each build.

Text can change routed experts and MTP acceptance. This result therefore
measures the serving implementation on the declared workload; it does not
isolate every millisecond to the head kernel. The unchanged C1 control,
consistent positive C4 class results and the separate
[head-kernel measurement](2026-09-19-qwen-fp8-head.md) support the intended
performance interpretation. Matched real-checkpoint teacher-forced numerical
validation remains outstanding and is not replaced by throughput evidence.

## Reproduction and evidence

[benchmark_epoch.py](2026-09-20-qwen-fp8-head-e2e/benchmark_epoch.py) imports
the repository's corpus, streaming client and phase metrics. With one of
the explicit configurations above serving at `127.0.0.1:30002`, run:

```bash
python3 benchmarks/results/2026-09-20-qwen-fp8-head-e2e/benchmark_epoch.py A1.json
```

Restart with the candidate for B1 and B2, then with the baseline for A2.
Keep the same configuration and warm-up procedure. Run the calibration
with `--calibration` separately; do not include it in the A/B aggregate.
Both binaries must use the same build preset and production code apart from
the head dispatch. Run this only on reserved idle hardware.

The [raw epoch archive](https://github.com/jontaylor/dgpp/releases/download/fp8-head-benchmark-2026-09-20/fp8-head-raw-epochs.tar.gz)
retains request texts, usage counts, receive timestamps, timing gaps,
prompt/text hashes, metric snapshots and both calibration records. It is
hosted separately to keep compressed measurement data out of the source tree.
The downloaded archive was hash-checked and its analyzer output reproduced
`summary.json` exactly. Extract it into a scratch directory and check the
[SHA256SUMS](2026-09-20-qwen-fp8-head-e2e/raw/SHA256SUMS) retained here.
The same analyzer reads either uncompressed `.json` or `.json.gz`:

```bash
python3 benchmarks/results/2026-09-20-qwen-fp8-head-e2e/analyze.py \
  /path/to/extracted-epochs
```

See [summary.json](2026-09-20-qwen-fp8-head-e2e/raw/summary.json),
[protocol.json](2026-09-20-qwen-fp8-head-e2e/raw/protocol.json), and the calibration
and A1/B1/B2/A2 files in the archive. The pooled table uses
all phases; the class table uses per-class medians.

Every epoch was shut down with its explicit launch configuration and both
rank op-stream MD5s agreed; values are retained in
[op-streams.json](2026-09-20-qwen-fp8-head-e2e/raw/op-streams.json). Production
was restored after each preliminary calibration and after the final run.
Both ranks retained the original release binary SHA256
`ab1b88a00de32dedb0de63d285e584651d69c99185c106c4e6d7e549b9585fae`
and original configuration. The service reported no engine failure, the
repeat smoke prompt reused 1128 tokens, and the production cache had 148 slots.

Draft [PR #10](https://github.com/HawkBearPig/dgpp/pull/10) is open;
real-checkpoint teacher-forced numerical validation remains pending.
