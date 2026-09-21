# Batched sampled-MTP fallback isolation

Date: 2026-09-21
Issue: [#15](https://github.com/HawkBearPig/dgpp/issues/15)
Baseline: `1f357dad7f126c6ea21741c10852f97db1fee71d`.

## Reproduction and cause

Repairing the old depth-one gate to follow scheduler observer events reproduces
the reported failure on both ranks. Request `b`, seed 13, diverges on its first
decode after co-admission with request `a`, seed 11. The synthetic GLM model has
a 96-token vocabulary, a nine-token prompt, two request slots, MTP depth one,
temperature 1, top-p 0.95, frequency penalty 0.2, presence penalty 0.1 and a
24-candidate cap to force exact gather fallbacks. The limits are seven output
tokens for `a` and six for `b`.

Temporary diagnostics confirm that both requests enter the replay with the
correct prompt counts and persistent token feeds. The second request's saved
penalized verifier logits also match the eager oracle (to the diagnostics'
six-decimal output precision). Its fallback nevertheless tests draft 44, while
the verifier actually consumed draft 19.

`collect_verdict(a)` freezes `a`'s drafts and drains the replay for its fallback.
Draining publishes the new draft tokens for every other slot in that batch.
`collect_verdict(b)` then takes its supposed verified drafts from that updated
state. The host accept/residual decision uses the correct logits with the wrong
draft identity; accepted drafts can also contaminate the context mirror. This
is a draft-state lifetime error, not bad penalty-count initialization.

The issue's above-slot-count batch threshold does not provide a scalar control:
the constructor clamps that threshold to the slot count. The existing depth-two
test is a true control because GLM has no batched draft chain at that depth. It
passes before the fix, including co-admission with penalties.

The adapter now freezes every live slot's verified drafts after settling the
older replay and before collecting any current verdict or draining its tail.
The preallocated host buffers are reused; they replace the old per-verdict
vector allocation. The same rule covers scalar execution and disabled
pipelining. The verdict publication is awaited before the optional drain, so
the disabled-pipeline path never reads `inflight_.back()` after clearing it.
No model arithmetic, kernels, captured nodes, sampling draws or penalty rules
change. All temporary diagnostics were removed.

## Permanent regression coverage

The old gate read stored scheduler results while requests were live. Those
results are populated only at retirement, so it compared empty output vectors
and stopped following each request after its first tick. Both sampled-MTP gates
now follow token and retirement events and assert that their final stored
transcripts were checked in full.

- Depth one: a solo request, seed 13 alone, the reported co-admitted pair,
  reused captured slots with a seed that accepts drafts after fallback, and
  staggered arrival. Every replay compares emitted tokens to an independent
  eager speculator; active batched feeds are also checked. Seed 13 must produce
  the same complete transcript in all four admission/reuse scenarios.
- Depth-one acceptance-counter coverage now co-admits two requests rather than
  serializing them into one slot. Depth two retains concurrent scalar coverage.
  Both assert actual replay-family counters, final transcripts and exact eager
  acceptance counters.
- A separate CTest entry, `glm_mtp_sampling_no_pipeline`, runs the expanded
  depth-one gate with `DGPP_PIPELINE=0`.

## Validation

- Full CI and release builds passed.
- All 30 host CTest entries passed in 168.55 seconds.
- All five selected GPU/RDMA CTest entries passed in 66.77 seconds: `glm_tp_test`,
  `glm_mtp_sampling_no_pipeline`, `glm4_engine_test`, `dsv41_engine_test` and
  `qwen_engine_test`. Their harnesses report 46, 1, 4, 5 and 6 cases respectively.
  The opt-in real-checkpoint TP forward parity case in `glm_tp_test` is skipped;
  the real model is exercised separately on the four-node fabric below.
- The expanded depth-one gate passes 28 scheduler ticks and 43 exact fallbacks
  with pipelining both enabled and disabled. The acceptance-counter gates pass
  16 fallbacks at depth one and 15 at depth two, matching the eager tokens and
  per-position acceptance counts.

Reproduction commands (GPU suites run serially with production stopped):

```bash
cmake --preset ci
cmake --build --preset ci -j4
CUDA_VISIBLE_DEVICES='' ctest --test-dir build-ci -L host --output-on-failure -j1
CUDA_DEVICE_MAX_CONNECTIONS=32 ctest --test-dir build-ci \
  -R '^(glm_tp_test|glm_mtp_sampling_no_pipeline|qwen_engine_test|glm4_engine_test|dsv41_engine_test)$' \
  --output-on-failure -j1
cmake --preset release
cmake --build --preset release -j4
```

## Four-node serving comparison

The resident `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8` deployment uses four DGX
Sparks, four request slots, MTP depth one, 128 sampling candidates, BF16 KV,
`bf12+bf16` weight residency and an 8 GiB prefix cache. The site deployment
configuration is unchanged. Baseline and candidate are release builds;
no GPU tests or builds overlap these timings.

```bash
python3 scripts/timed_load.py 127.0.0.1 18080 \
  --concurrency 1,2 --classes prose,json --max-tokens 192 \
  --repeat 3 --warm 1 --json-out RESULTS.json
```

These greedy controls preserve all 18 measured response texts and usage records
exactly. Median engine decode throughput over three repetitions:

| Workload | Concurrency | Baseline tokens/s | Fixed tokens/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Prose | 1 | 60.230 | 60.439 | +0.35% |
| Prose | 2 | 78.984 | 79.038 | +0.07% |
| JSON | 1 | 61.597 | 61.766 | +0.27% |
| JSON | 2 | 79.424 | 79.448 | +0.03% |

These differences are within run-to-run noise; no decode throughput regression
is observed. Exact eager sampling, rather than preserving the baseline's wrong
sampled output, is the correctness criterion for the bug scenario. Forward
arithmetic and weight/cache precision are unchanged, so this change does not
introduce a new numerical approximation requiring a loss-budget tradeoff.

An additional 12 real-model completion requests compare seeds 11 and 13 alone
and concurrently, with temperature 1, top-p 0.95, 64 generated tokens and
`logprobs: 1`. All six comparisons preserve the complete token text and logprob
reports exactly. The three cases are frequency 0.2 plus presence 0.1, repetition
1.15, and no penalties. The paired phases execute 42, 40 and 34 two-slot graph
replays respectively, verified from the decode-batch histogram. They use the
default proposal-draft setting and sampling width, with this plain prompt:

```text
A curious traveler reached the old market. Write a vivid story with varied characters and unexpected events.

```

All nine API checks pass. The four running executables have the same candidate
SHA-256 (`b39701f00c5a0902a136107b31207288626f4f8d409c7264753691c64cd3dfe9`),
and all four operation streams match after the performance, sampling and API
checks (`b0fc446ac439c9ddaeb832a386c4d6919b04a061189dca226f4efa60f84303ed`).
The clean fixing commit is rebuilt for the final production deployment; the
baseline binary is retained only as a local comparison artifact.

Raw build logs, reproductions, diagnostics, responses, metrics and deployment
audits are retained locally under `artifacts/issue15/`.
