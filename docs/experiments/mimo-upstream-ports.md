# MiMo ports onto upstream master

Base: upstream `c6ca191`. Source of the optimization ideas: our saved
`codex/mimo-best-combination` at `808e919`. The branch keeps upstream's
paged KV layout, scaled FP8 representation, BF12 weight companions and
query-tiled attention.

Two independent opt-in experiments:

- `DGPP_MIMO_PREFILL_LAST_HEAD=1`: scalar serving prefill projects only its
  final normalized row through the vocabulary head. All hidden rows remain
  available to MTP. Diagnostic forwards, grouped prefill and decode retain
  every vocabulary row. The GEMM shape changes, so bitwise equality with
  diagnostic forward is not asserted for this option.
- `DGPP_MIMO_MTP_CACHE_ONLY=1`: history-only MTP prefill retains input fusion,
  RMSNorm, QKV projection, RoPE, value scaling and paged KV writes (including
  FP8 scales), then skips attention, output projection, MLP and both TP
  reductions. Decode and chained draft outputs are unchanged.

Both options default off and are sampled during model construction.
`mimo_port_test` compares each switch and their combination with the original
path on BF16/FP8 pools, 23- and 97-token prompts, two slots, repeated opening,
multiple prefill chunks and subsequent draft/decode calls. It requires exact
hidden states and exact cache-only logits; changed head shapes must keep
relative logit L2 below 1e-5 and preserve top-1. This is a synthetic gate,
not broad real-model quality evidence.

The benchmark uses two Sparks, C2, BF16 KV, upstream's shared 131072-token
pool, BF12 weights and MTP1. Upstream does not accept MiMo prefill-budget
controls. These capacity and scheduling settings differ from the saved
256K-per-request, MTP3 deployment. Use identical settings between upstream
and these ports; report comparison with the saved implementation separately.

Raw evidence and deployment restoration records live in
`/mnt/benchmarks/dgpp-mimo-upstream-20260923/`.

## Tool-call compatibility

The pure upstream baseline returned malformed XML as content on the first
actual custom-harness request (0/1 attempted passed; two later probes were
not attempted). The local compact grammar was ported without replacing
upstream's newer schema/reference support or model-opened thinking.
Required argument keys are retained for non-strict MiMo tools too. Generic
newline XML continues to use its existing grammar. Tests cover both forms.

## Reproduce the request panel

On an exclusively reserved server, point `DGPP_MIMO_BENCH_URL` at its `/v1/`
endpoint and `DGPP_MIMO_BENCH_ROOT` at an evidence directory. Run
`python3 benchmarks/mimo_upstream/benchmark.py UNIQUE_TAG`, followed by
`acceptance.py UNIQUE_TAG`. The scripts preserve requests, raw SSE events,
usage, outputs, timings and before/after metrics; they refuse to overwrite
an existing timing panel. `quality.py UNIQUE_TAG 2800 6000` checks exact
retrieval at approximately 31K and 67K tokens. Tools are exercised separately
through the user's actual Python harness; no generated tools are executed.

The subsequent [native MTP experiment](mimo-native-mtp.md) migrates blocks
1/2 behind a separate opt-in. Its depth comparison keeps both prefill flags
off; the original prefill results above remain a separate experiment.
