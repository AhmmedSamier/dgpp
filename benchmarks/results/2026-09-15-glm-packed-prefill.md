# Full GLM packed int4/int8 prefill

Date: 2026-09-15. Baseline source: `2ea2c29`.

Packed tensor-core prefill reduces matched cold-service time by **2.56× at
2K, 1.97× at 8K and 1.47× at 32K**. Existing numerical limits are unchanged;
GSM8K and extraction retain every baseline grading result. Across two C1
deployment pairs, the equal-class change is −0.0034%, with class changes
from −0.297% to +0.146%; transcripts and speculative work match exactly.

## Implementation

`packq_gemm` adds tensor-core matrix products for full GLM's int4 routed
experts, int8 shared expert and int8 attention projections. A 32 × 64
output tile stages two 64-deep groups of packed weights and BF16 activations.
Integer codes are decoded exactly into BF16 fragments, multiplied on tensor
cores, and scaled per group using FP32 FMAs. There is no per-weight BF16
rounding or full dequantized weight allocation. The stored offset-code,
low-nibble/byte-first and group-64 scale conventions are unchanged.

Grouped gate/up uses the existing device row map directly. Down outputs
remain FP32 until the existing ordered expert accumulation. The kernel
supports empty segments, ragged rows/columns and padded activation/output
strides. No loader, resident weight, KV-cache or checkpoint format changes
are needed.

Automatic dispatch begins at 128 rows per invocation. Grouped MoE prefill
can reach that count by combining shorter prompts. Short-prompt C1 GEMV is retained to
protect the measured C1/MTP behavior, not because matching old outputs is
an accuracy requirement. A 17-row candidate changed MTP acceptance and
reduced code/chat throughput by about 1.2%/2.3% versus the initial baseline;
a fresh baseline reproduced its original transcripts and pass counts.
The cutoff is a conservative bulk-prefill policy, not a demonstrated optimal
crossover or an accuracy boundary. Decode/verification kernels are unchanged.
No existing numerical test threshold has been relaxed.

## Kernel measurements

TP4 production dimensions, synthetic weights and uniform expert routing,
three warmups followed by three timed launches per side, six alternating
A/B pairs. These are medians of CUDA-event times in milliseconds. Clocks
are not locked. Dense/shared weights may stay in L2 across iterations;
these results are not DRAM-bandwidth measurements or service speedups.

At 2,048 prompt tokens:

| Matrix | Format | GEMV ms | GEMM ms | Speedup |
|---|---|---:|---:|---:|
| Routed gate, 512 × 6,144 | Int4 | 30.176 | 4.310 | 7.00× |
| Routed down, 6,144 × 512 | Int4 | 16.292 | 6.746 | 2.42× |
| Fused q/kv-a, 2,624 × 6,144 | Int8 | 56.773 | 2.960 | 19.18× |
| q-b, 4,096 × 2,048 | Int8 | 9.667 | 1.227 | 7.88× |
| Output, 6,144 × 4,096 | Int8 | 83.298 | 6.343 | 13.13× |
| Shared gate, 512 × 6,144 | Int8 | 5.527 | 0.527 | 10.48× |
| Shared down, 6,144 × 512 | Int8 | 4.394 | 0.630 | 6.98× |

At 256 tokens, routed gate improves from 4.280 to 2.866 ms; routed down
is roughly unchanged at 3.121 versus 3.065 ms. At 32 tokens, both routed
projections are roughly unchanged. The 35-case matrix retains those shapes
as well as the large-prefill results, rather than selecting only the
largest speedups.

At 64 rows, routed gate/down take 2.329/2.291 ms with GEMV versus
2.433/2.348 ms with GEMM. At 128 rows they take 2.991/2.757 versus
2.693/2.880 ms. Dense/shared projections improve at both widths, so this
is not evidence that every packed matrix crosses over at 128. The policy
balances whole-prefill gains with the observed short-C1 acceptance effect.

## Validation

The warning-as-error full CI build and all **20 selected suites passed**
(187.58 seconds). The final 160-row arithmetic fixture has zero attention
or expert-selection differences, checks every row, and reaches 0.699%
last-layer hidden relative L2, below the unchanged 1% limit. It has zero
hard top-1 mismatches. Compute Sanitizer memcheck, racecheck and initcheck
all report zero errors/hazards on the final kernel test.

The final validation includes packed GEMV/GEMM, MoE, DSA,
full-GLM forward/decode/TP/engine/loader, Qwen MoE, unit tests and script
reports. Existing decode bitwise checks remain in place. The complete
MoE-chain test spans 16/17/33/127/128/129/257 tokens and compares device
segmentation against the corresponding host chain.

A separate 160-row whole-model fixture exercises automatic packed GEMM
against the same FP64 oracle and unchanged residual/logit budgets. Dense
attention and all-expert routing remove discrete selection boundaries from
this arithmetic check. The original sparse-attention/routing fixture is
unchanged and remains part of the suite; real-checkpoint likelihood and
service tasks separately exercise the production selection rules.

The initial 160-row fixture retained top-2-of-8 routing. **Both baseline and
candidate failed** its 1% per-layer hidden-state budget (last layer 1.04%
and 1.03%, respectively). Near-tied expert choices at earlier positions
changed the inputs attended to at later positions, even when those later
positions selected identical experts. This is why the additional arithmetic
fixture selects every expert; no threshold was raised and no existing
fixture was replaced. The reference generator represents the absence of an
excluded expert with an infinite boundary margin.

The independent FP64 kernel oracle uses exact code × BF16-scale weights.
FP32 outputs require relative L2 error below 5e-6 and bounded absolute
error; BF16 epilogues must round those outputs exactly. Tests cover both
code widths, ragged tiles, zero/NaN scales, mapped and unaligned inputs,
output padding, and cold CUDA graph capture/replay. All four real
checkpoint slices passed the existing two-BF16-ULP/cancellation-floor gate
with zero mismatches. The checkpoint gate has zero mismatches under both lowerings.

## Rounding and the short-sequence coverage failure

The 17-row candidate fails the existing 12-step decode audit's **coverage**
rule: six of thirteen rows retain identical DSA selections, where the rule
requires seven. Those six rows satisfy the unchanged 2% relative-L2 and
near-tie top-1 checks (worst relative L2 0.0141, zero hard top-1 mismatches).
All other selected suites pass. No existing tolerance has been changed.

Extending that diagnostic to 24 decode steps did not repair the comparison:
the retained rows reached relative L2 0.0986 and one hard top-1 mismatch
against the re-forwarded sequence. Matching a row's current selection does
not remove differences in earlier attention histories. That diagnostic
extension was not merged; the original test and limits remain intact.

GEMV scales partial sums of 16 int8 or 32 int4 elements before its lane
reduction. GEMM scales 64-element tensor-core partials and accumulates them
in group order. Both use exact code × BF16-scale weights. The different
FP32 addition order can move a BF16 result across a rounding midpoint;
subsequent norms and the indexer's BF16-to-FP8 quantization can amplify
that small input change enough to alter a top-k boundary.

The synthetic FP64 comparison measures both kernels against the same
unrounded oracle. GEMM's raw FP32 relative errors range up to 3.14e-7;
GEMV's up to 1.31e-7. Long sequential group accumulation has more FP32
roundoff than GEMV's reduction tree. After BF16 rounding, only four of
19,836 synthetic outputs differ, and all four are closer to FP64 with
GEMM. This distinction between accumulator error and final output error
is important; the former is not being hidden by the BF16 comparison.

Four real checkpoint tensors cover **1,867,776 outputs**. Only **179
(0.00958%)** differ: 107 are closer to the unrounded FP64 result with GEMM,
72 farther away. BF16 relative L2 against the unrounded oracle is about
0.00166 under either kernel. Differences between their aggregate errors
are below 5e-12 absolute in each tensor; their direction varies. Both
pass the existing two-BF16-ULP/cancellation-floor gate with zero violations.

To investigate the failing fixture, diagnostic-only model builds captured
the actual FP8 queries, folded indexer weights, cached keys/scales, block
tables, and prefill dot products. Replaying the selector's FP32 operations
and warp reduction reproduced **every captured selection exactly**. The
production server contains none of this instrumentation.

On an identical 35-token sequence (the original 23-token prompt plus twelve
fixed continuation tokens), the first cross-kernel selection change is at
layer 2, position 28. GEMV's boundary gap is 0.0787% of the row's score
scale; the largest score movement is 3.43%. GEMM selects the same set as the
independent FP64 reference there, and also at position 32 where GEMV differs.
At position 34 the direction reverses: GEMV matches the reference and GEMM
differs. Later layers propagate the changed attention histories.

Across this **synthetic fixture**, full-logit relative L2 against the FP64
reference is **0.04697 for GEMV and 0.03700 for GEMM**. Thus reduced
agreement with the decode path does not establish reduced accuracy. This
is a numerical diagnostic on one fixed fixture sequence, not a task-quality
improvement claim. The real-model likelihood results below provide a separate quality check;
task scores and C1 service measurements remain independent gates.

All **1,226 pre-existing CUDA kernel text sections are byte-identical**
between the baseline and both final/17-row candidates; only eight new packed GEMM
instantiations are added. This establishes unchanged device code for the
old decode launchers, while C1 measurements must still check consequences
of changed prefill state and host dispatch.

## Real-model prefill likelihood

The full 78-layer target scores each next token using a full forward, so
all scored positions exercise prefill arithmetic. Baseline diagnostics link
the new scoring app to the exact pre-change projection/MoE objects; the
candidate uses the new packed GEMM. All texts exceed 128 tokens and therefore
exercise the same kernel paths under both candidate cutoffs. Every rank's
local vocabulary log-sum-exp and argmax is merged; completion, position,
target ownership and finite-score checks reject incomplete runs.

The prose text is the complete `benchmarks/teacher_text.txt`. The hard and
memorized texts use their first 1,500 whitespace-delimited words to bound
prompt-wide scratch/logit memory. Text hashes, counts and per-position
log probabilities are retained in the [raw record](2026-09-15-glm-packed-prefill.json).

| Text | Scored tokens | Mean NLL change, nat | Token-level SE | Correct top-1, old → new | Argmax changes | >1 nat moves |
|---|---:|---:|---:|---:|---:|---:|
| Prose | 555 | +0.00775 | 0.00676 | 394 → 393 | 16 | 2 (0.36%) |
| Hard | 2,409 | +0.00377 | 0.00492 | 1,208 → 1,210 | 179 | 17 (0.71%) |
| Memorized | 1,851 | +0.00069 | 0.00267 | 1,833 → 1,832 | 1 | 3 (0.16%) |

All three pass the existing absolute mean-NLL-change limit of 0.02 nat and
1% rate limit for moves above 1 nat. Across 4,815 tokens the correct top-1
count is 3,435 for both paths. Mean NLL rises about 0.00304 nat/token;
the observed direction is reported, rather than treating a passed gate as
proof of identical accuracy. These are limited texts, and token-level SE
is descriptive: adjacent tokens are correlated. Changed argmax tokens are
not equivalent to lost correct predictions.

One initial uncached diagnostic world failed during model-loading skew,
before producing scores. It contributes no numerical samples. The completed
comparisons use the existing resident-image cache on each node.

## Service and C1

Matched cold prompts, three repetitions per length, yield:

| Prompt target | Baseline prefill | Packed GEMM prefill | Speedup | Baseline → new ms/token |
|---|---:|---:|---:|---:|
| 2,048 | 15.628 s | 6.111 s | 2.56× | 7.569 → 2.963 |
| 8,192 | 80.364 s | 40.838 s | 1.97× | 9.759 → 4.959 |
| 32,768 | 512.968 s | 350.069 s | 1.47× | 15.531 → 10.613 |

All nine prompt hashes, actual lengths and computed-token counts match;
all have zero attached prefix-cache tokens. The implementation saves
roughly 4.6–4.9 ms per prompt token across these lengths. Its percentage
benefit decreases as context grows; profile the remaining path before
attributing that cost to a particular attention kernel or collective.

Task quality at concurrency 4 matches the baseline: **GSM8K 59/60 and
extraction 30/30**, with zero truncated answers in either build. Every one
of the 90 per-item correctness outcomes is unchanged. Three GSM8K response
texts differ while retaining the same grading result.

### C1: two deployment pairs

Pair 1 uses the fresh baseline at 19:10–19:14 UTC and candidate at
19:29–19:32. Pair 2 uses the baseline at 19:57–20:01 and candidate at
20:01–20:04. Each is a separate process deployment with the same recipe.
The raw record also retains the initial baseline and rejected 17-row trial.

| Class | Baseline tok/s | Candidate tok/s | Paired change | Difference per 320 tokens | Pair 1 / pair 2 change |
|---|---:|---:|---:|---:|---:|
| Prose | 28.026 | 28.049 | +0.082% | -9.4 ms | +0.070% / +0.095% |
| Code | 29.058 | 29.095 | +0.127% | -13.9 ms | +0.119% / +0.135% |
| JSON | 28.981 | 28.959 | -0.075% | +8.3 ms | +0.070% / -0.220% |
| Math | 28.462 | 28.504 | +0.146% | -16.4 ms | +0.494% / -0.200% |
| Chat | 25.257 | 25.182 | -0.297% | +37.7 ms | -0.104% / -0.490% |

The displayed rates are geometric means of the two per-deployment class
medians. Time differences are `320 / candidate_rate - 320 / baseline_rate`;
they are derived response-time comparisons, not kernel timings. Equal-class
geometric-mean changes are **+0.1295%** in pair 1, **−0.1361%** in pair 2,
and **−0.0034%** across both pairs. The unchanged baseline's between-session
class shifts range from −0.169% to +0.268%, with an equal-class +0.0846%.

The overall result is flat at this campaign's repeatability. Chat remains
slightly slower in both pairs: −0.104% and −0.490%, or a combined **37.7 ms
more per 320-token response**. These two deployment pairs do not rule out a
small class-specific cost; an overall average is not proof that every class
is unchanged. Adjacent prompt repetitions are not independent deployment
experiments, and no causal confidence interval is claimed.

All **30 final-candidate responses** match their references in visible
transcript hash, usage/cache counts and speculative passes. Passes per
320-token response are **171/162/164/166/189** for
prose/code/JSON/math/chat. All final C1 worlds also have the same operation
MD5, `1175fda6ae56e5b40219c0d01c883c23`, on every rank. All 1,226 pre-existing
CUDA kernel bodies remain byte-identical in the final candidate.

For comparison, the rejected 17-row candidate needs **164 instead of 162**
passes for code and **193 instead of 189** for chat. Against the fresh
baseline it is 0.98% and 2.18% slower, respectively (1.19% and 2.33%
against the initial baseline). Its changed prefill state adds measurable
speculative work even though the likelihood checks pass. This is the
throughput evidence for preserving the short-prompt path.

### Fabric and restoration

The baseline cold/quality world shuts down with operation MD5
`5645aa3560e4ba2a26b4ce0efa4e4a90` on all four ranks; the candidate world
with `16ecd705c5c1052786114def65086125` on all four. Cross-build hashes need
not match when generated token choices differ. Rank agreement, arithmetic
accuracy and task quality are separate checks.

The original four-node GLM-Flash service was restored after the campaign.
Every running `/proc/PID/exe` hashes to
`49c0c9dc9fbde3cc04138292cf7059b6cc2e30e1cdb2dbe08aa5c51a39fa8259`.
The model endpoint is healthy and the scheduler is idle. Full-GLM baseline
and final candidate server SHA-256 values are, respectively,
`4afd25fc204985ca40014b6f031bac6093ffe2e5ed28132dc0bae8b13503e167` and
`a0562b01d4b54891fafe75e38c2f2d84abfaccf34dcab24725db07540de95c3e`.


### Procedure and provenance

The service campaign uses four DGX Sparks, the full-GLM eight-slot recipe,
native MTP depth 1, BF16 KV capacity 122,880, a 1 GiB prefix cache and
vocabulary-sharded embeddings. The checkpoint revision is
`147684fbad20c1e283ddff46fd07cb9d4ccbb3da`. The raw record includes the
resolved engine settings and executable SHA-256 values. GPU work runs
serially; clocks are not locked.

The selected suite is reproducible on idle test hardware:

```bash
cmake --build --preset ci -j 4
ctest --test-dir build-ci --output-on-failure -j1 -R \
    '^(packq_gem[mv]_test|glm_moe_test|glm_dsa_.*|dsa_test|qwen_moe_test|unit_tests|script_reports_test)$'
build-ci/packq_gemv_test --checkpoint-dir /path/to/GLM-5.3-Int4-Int8Mix-RTN-g64
```

Each cold-prefill build starts a fresh world and runs this same command,
with the prepared evaluation data directory selected through `DGPP_DATA_DIR`:

```bash
python3 scripts/serve_prefill_probe.py 127.0.0.1 18080 2048 8192 32768 \
    --repeat 3 --seed 7 --tag glm-packq-2026-09-15 --json-out cold.json
```

The probe requires zero attached prefix-cache tokens and verifies that
the scheduler's measured work belongs to exactly that request. It records
actual prompt lengths, hashes, computed tokens, engine prefill time and
visible TTFT. The 529-token calibration precedes the measured prompts in
both fresh worlds. No C1 sweep precedes either cold-prefill campaign.

Separate fresh worlds run `serve_load.py` with five classes, concurrency
1, 320 completion tokens, greedy decoding and three repetitions. Each
class has one prefix miss followed by two hits. Prompt lengths are
37/51/46/68/40 tokens for prose/code/JSON/math/chat, respectively. These
checks protect the established short-prompt C1 workloads. Longer prompts
use the new arithmetic, so their token choices and later MTP acceptance
can differ. The full-GLM template
requires reasoning: the client retries its unsupported thinking-off request
without that option. Usage confirms reasoning is on; the client's legacy
`thinking: false` metadata does not describe the completed requests.

After the cold prompts, each build runs GSM8K 60 and extraction 30 at
concurrency 4, with the existing seed 20260908, 2,048-token completion
limit and low reasoning effort. Scores, item IDs, answers and truncation
status are retained.
