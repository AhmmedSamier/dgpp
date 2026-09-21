# Prefix reuse for changed questions over a long document

Date: 2026-09-21
Issue: [#17](https://github.com/HawkBearPig/dgpp/issues/17)
Baseline: `b13d06314f241cdedb96125633265718aff66762`.

## Reproduction and implementation

The baseline only saves the deepest reusable prefill cut. A changed question
near the end of the same document therefore misses, even though almost the
entire prompt is unchanged. The new host regression fails against the baseline
when the second prompt saves zero tokens instead of attaching at the earlier
document cut.

For prompts at least four regular chunks long, the scheduler now also saves
`(floor(prompt_tokens / chunk_tokens) - 1) * chunk_tokens`, when it is past the
attached prefix and before the deepest cut. This is an existing prefill cut;
the policy adds a state copy without splitting another GEMM. The deepest cut
retains priority when arena slots are scarce. A changed tail can share the
earlier snapshot's complete KV blocks and allocate only its private suffix.

The model snapshot request supports a second cut in the same walk. Both engine
adapters publish ownership of snapshots taken before a later prefill failure.
Resumable prefill preserves both cuts across budget changes and releases both
slots on cancellation. DeepSeek bounded prefill retains its existing policy:
an extra snapshot would execute another decoder span. Exact prefill can use
the additional cut.

Validation found an existing MTP identity gap: a prefill snapshot at position
`P` includes a draft input containing `prompt[P]`, while the old cache key only
checked `[0, P)`. Editing exactly that token preserves the target state but
changes resumed draft logits in the Qwen fixture. Prefill entries now retain
and check this lookahead token during lookup and deduplication. Rolling/close
entries remain unconstrained, since their lagging draft state catches up on
resume. Miss diagnostics distinguish a changed lookahead token.

## Capacity diagnosis and recipe audit

The original review's misses occurred against only two live entries in a
27-slot arena. During the following decode the KV pool was about 8186/8320
blocks occupied (98%), with only three arena entries. Each cold changed-tail
request allocated another document's worth of KV blocks. This was KV-pool
pressure, not exhaustion of the 1.5 GiB snapshot arena.

All ten checked-in recipe memory plans pass. The [sizing guide](../../docs/prefix-cache.md)
records their snapshot bytes, slot counts, KV capacities and retention rule.
Snapshot size is independent of context length for current families. The
production four-node GLM recipe holds 232 snapshots in 8 GiB; two-node Qwen
holds 27 in 1.5 GiB. Single-node Qwen is the tightest at 13. No recipe budget
was increased: the measured shared-document working set fits the current
budget, and larger historical working sets should be sized explicitly.

Startup and `--memory-plan` now print the requested arena budget, bytes per
snapshot, slot count, actual allocation and separate KV token capacity. The
plan also includes MTP snapshot state for single-node graph deployments.

## Fabric procedure

Two physical Sparks serve
`deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json`, with HTTP
bound to localhost port 30023. The model is
`nvidia/Qwen3.8-Flash-Next-NVFP4`, BF16 KV, FP8 dense weights, MTP depth one,
two request slots, a 532,480-token KV pool and a 1.5 GiB prefix arena per rank.
Production and GPU test worlds are stopped for these measurements. No builds
run alongside timed model requests.

The document is `"Issue17 document {length}.\n"` followed by
`qwen_yarn_release_check.filler(20260921 + length, records)`. Nominal lengths
32,768 and 262,144 use 704 and 5793 records, respectively, yielding actual
prompts of 31,661 and 260,053 tokens. Each question appends
`"\nQuestion {i}: write a numbered list describing the first {30 + i} parcels and their cities."`.
Requests are greedy, streaming and capped at 128 output tokens. Published
metrics are reconciled against completed requests before taking counter
deltas. Six distinct long questions are followed by a repeat of question zero.

Baseline binary SHA-256:
`402b1b68115b0981d8dcadcf6baa7951ce5dc601010480584d8d28c9925a1168`.
Candidate binary SHA-256:
`cc3c8387012b4783ec4df62a6b53aa18d72015cc0599d79f9e366a71ed8b1114`.
The measured candidate includes these source changes and carries the dirty
baseline version; the committed release is rebuilt before production rollout.

## Measured reuse

| Request | Baseline TTFT | Candidate TTFT | Candidate cached tokens |
|---|---:|---:|---:|
| 31,661-token cold prompt | 20.239 s | 20.257 s | 0 |
| Same document, changed question | 20.331 s | 2.174 s | 28,672 |
| Identical question repeated | 84.7 ms | 83.3 ms | 31,656 |
| 260,053-token cold prompt | — | 215.648 s | 0 |
| Five changed questions over that document | — | 4.444–4.499 s | 256,000 each |
| Original long question repeated after those five | — | 164.4 ms | 260,048 |

The six long variants use 13 document entries: one body, six final-prompt and
six completed-answer snapshots. Including earlier calibration and 32K entries,
the arena holds 20/27 entries with **zero evictions**. Physical KV occupancy
is 328,384/532,480 tokens; subtracting the earlier requests, the long document
and its six suffixes occupy 281,728 tokens. Keeping all six as independent
cold KV allocations would exceed this pool.

At 32K, cold prefill changes from 20,221.7 to 20,224.0 ms (0.01%). Changed-tail
prefill drops from 20,306.2 to 2,150.4 ms. Identical-repeat prefill stays at
58.9 ms. The three matching baseline/candidate transcripts are byte-identical;
decode pass counts match, and measured decode time per pass does not regress.

A fresh baseline run of long question one takes 215.497 s to first token,
versus 4.468 s after shared-document attach in the candidate (48.2× faster).
The generated text, token counts and decode-pass counts match exactly.
The original long question's cold and repeated responses also match exactly.
All five two-rank campaigns have matching operation-stream SHA-256 hashes
within each run.

Opting out with `prefix_cache: false` also removes the cache's structural
prefill cuts. Its changed GEMM shapes can produce a different greedy transcript
on both revisions; it is not a bitwise oracle for cache-enabled execution.
The 32K cache-disabled transcript is identical between baseline and candidate.
The native bitwise gates and the 260K cold baseline comparison use matching
cuts, so they directly test correctness of restored state.

The first 128-token long-context samples varied from 30.69 to 32.74 ms per
decode pass, including a candidate sample 5.6% above the fresh baseline.
To separate this variation from a regression, a second comparison makes five
identical 256-token generations on each build. The candidate first seeds the
shared document with question zero, then runs question one via body attach and
four identical repeats; the baseline runs question one cold followed by four
identical repeats. Every paired transcript and its 134 decode passes match.
The four repeated-hit medians are **31.407 ms/pass baseline and 31.035 ms/pass
candidate** (candidate 1.2% lower). The changed-tail candidate itself measures
31.030 ms/pass. These runs show no persistent decode slowdown; the measured
benefit is avoiding prefill, not changing decode arithmetic.

## Native validation

- Full CI and release builds pass; all 30 host CTest entries pass (166.47 s).
- All nine selected GPU/RDMA suites pass (76.16 s): `qwen_decode_test`,
  `glm_tp_test`, `dsv41_decode_test`, `glm_dsa_decode_test`, `glm4_decode_test`,
  and the Qwen, GLM-4, full GLM and DeepSeek engine suites. GLM reports 47
  cases; its opt-in real-checkpoint TP-forward case is skipped. Real-model
  validation is the physical-fabric procedure above.
- Scheduler coverage includes changed-tail and identical hits; one-, two-,
  six- and 27-slot arenas; seven variants under LRU pressure; edits before or
  exactly at the cached cut; short and opted-out prompts; continuation
  publication; cancellation; and failures after the earlier snapshot.
- Qwen and GLM fixtures compare target and MTP logits bitwise after attaching
  either snapshot, through subsequent decode and continuation budget changes.
  Snapshot cleanup returns every KV reference. DeepSeek adapter tests ensure
  bounded mode retains its original snapshot policy.

Local raw evidence is retained in `artifacts/issue17/`: baseline/candidate
campaign JSON, rank logs and operation-stream hashes, recipe plans, reproduction
logs, build logs, host/GPU CTest logs and production rollout checks.

NVMe-backed retention of evicted KV blocks and prefix state is deferred to
[enhancement #26](https://github.com/HawkBearPig/dgpp/issues/26).
