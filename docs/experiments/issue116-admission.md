# Admission and mixed-prefill investigation

Hardware measurements for [issue #116](https://github.com/HawkBearPig/dgpp/issues/116),
2026-10-17. The cache-reclamation implementation is `18392dc1fcd1`; the baseline
is its parent, `08c53ad982d2`. Both server binaries use the release preset. The
benchmark timestamp correlation change is `30bb520e` and does not change the
server.

## Shared scheduling policy

Admission, reservation growth, prefix-cache eviction, youngest-first shedding,
and the adaptive prefill controller belong to the common scheduler. MiMo uses
`SessionModel<MimoModel>` and the common eager/graph engines and `PrefillBatch`.
Its attention kernels, KV layout, draft state, and execution geometry remain
model responsibilities. Engine capabilities provide the shared policy with
alignment and capacity constraints; these decisions do not need model-name
branches.

## Cache reclamation on Qwen

Two GB10 nodes, TP2, Qwen3.8-27B FP8 revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, four request slots, an 8,192-token
KV pool, grow admission with a 256-token window, MTP off, and both latency
targets zero. Serving processes run with `MemorySwapMax=0`; both ranks report
zero process and cgroup swap use.

Three distinct 2,304-token prompts each generate one token and leave an idle
history. These histories pin 108 of the pool's 128 blocks. A new, cache-disabled
256-token prompt requests 2,048 output tokens with greedy sampling and
`ignore_eos: true`. Its full lifetime fits in the pool.

| Server | Output tokens | Cache evictions during growth | Requests shed |
| --- | ---: | ---: | ---: |
| Baseline | 1,024 / 2,048 | 0 | 1 |
| Candidate | 2,048 / 2,048 | 1 | 0 |

The pool reaches 128/128 physical blocks; the candidate reclaims an idle history and
continues. This is a controlled test of real KV ownership on a smaller pool,
not a replay of the reported OpenCode workload.

An additional run attaches to the oldest history while a cache-disabled
newcomer grows. Extend the seed prompt beyond the snapshot and include a
structural turn boundary so the lookup can probe that saved position. The
attached request reuses 2,304 tokens and completes 512 output tokens; the
newcomer completes 2,048. One idle history is evicted and neither live request
is shed. Subsequent requests hit both the protected history and the other
retained history, while the evicted history misses.

For genuine oversubscription, two cache-disabled requests each have a
3,456-token prompt and a 1,280-token output allowance. The older request
completes all 1,280 tokens. After idle entries are exhausted, the younger
request is shed at 512 tokens. The final cache contains no entries.

All nine checks in `serve_api_check.py` pass on the candidate, including prefix
reuse. Candidate, baseline, and the protection/oversubscription run each stop
cleanly with identical operation streams across their participating ranks.
The admission benchmark's metrics sampler also completes a live HTTP/SSE
smoke run; that short run does not establish sustained decode/prefill overlap.

## Cache reclamation on MiMo

The same probe also runs on the reporter's MiMo-V2.6-Flash-MOPD checkpoint,
revision `2479e2d0029eca9a34cc7e7f55a121925f81908e`, on two GB10 nodes with
TP2, four slots, FP8 KV, BF12-packed BF16 weights, decode graphs, and native
MTP depth one. The pool, grow window, latency targets, and request lengths
match the Qwen probe above. The reporter's MTP setting is not specified, so
this validates the model's cache ownership without claiming an exact replay.

| Server | Output tokens | Cache evictions during growth | Requests shed |
| --- | ---: | ---: | ---: |
| Baseline | 1,023 / 2,048 | 0 | 1 |
| Candidate | 2,048 / 2,048 | 2 | 0 |

MiMo retains both an intermediate prompt snapshot and a final prompt snapshot
for each seed. These six entries share physical blocks: they pin 108 unique
blocks, although the per-entry `blocks_pinned` sum is 204. The candidate must
evict both snapshots of the oldest history before its physical blocks become
free. This exercises the shared-snapshot case on the actual model, including
an eviction that alone cannot make room. Both builds stop cleanly with
identical operation streams across their two ranks.

An allocation-only memory plan also fits the reporter's 393,216-token FP8
pool with these settings. The pressure probe deliberately uses 8,192 tokens
to reach the capacity boundary quickly; the larger plan is not a full-pool
workload validation.

## Mixed-prefill controller probe

Use the same MiMo revision, precision, TP2, four slots, and MTP depth one,
with a 262,144-token pool and full admission. Both serving ranks have
`MemorySwapMax=0` and report zero process/cgroup swap use. Each target setting
starts a fresh server. A 32-token decode warmup precedes a solo long prompt;
the mixed scenario then uses the same long prompt on that warmed process.

The benchmark uses seed 104 and 41,356 words, verified as exactly 65,536
prompt tokens. After native prefill becomes visible, eight short requests
arrive 250 ms apart, each requesting 768 output tokens. The long request
requests one token. All requests use greedy sampling, `ignore_eos: true`,
and disabled prefix caching. Metrics are sampled every 250 ms, equally for
each configuration. The resolved automatic busy/idle caps are 256/2,048
tokens; the common engine's mixed execution can use the idle cap when it
advances decoders in the same pass.

This is a synthetic, limited-vocabulary prompt with deterministic counting
responses, not the reporter's chat workload. In particular, its high MTP
acceptance and model routing need not match real agent conversations. The
reporter's probe scripts and MTP settings remain unavailable.
These measurements use the candidate built from the current branch baseline,
not rebuilt copies of the report's `eb802041` or `b5dd4d94` binaries.

This is an exploratory sweep with one timed mixed run per target setting,
not a repeated performance qualification. The run order is fixed budgets,
default targets, a separate fresh-controller default probe, then two- and
three-second decode-gap targets. The first-token target stays at three seconds
except when both targets are disabled. The fresh-controller probe skips the
solo long prompt and retains only the benchmark's short decode warmup.

Run each warmed case against its freshly configured server with:

```bash
python3 scripts/serve_admission_bench.py \
  --url http://127.0.0.1:18080 --label mimo-target-setting \
  --scenarios solo_prefill sustained_idle --seed 104 --prompt-words 41356 \
  --arrival-tokens 768 --arrivals 8 --interval 0.25 --metrics-interval 0.25 \
  --trials 1 --timeout 1800 --out artifacts/mimo-target-setting.json
```

Each of the five mixed runs completes nine requests, 65,672 computed prompt
tokens, and 6,145 output tokens, with identical
request bodies. No requests fail, cancel, or shed, all KV reservations drain,
and both ranks replay identical operation streams. Three short requests stay
decoding throughout the long prefill after initial admission in the warmed
runs; the fresh-controller probe briefly has two before the third starts.

| Decode-gap / first-token targets (ms) | Long TTFT (s) | All requests drained (s) | Aggregate output (tok/s) | Overlapping decoder SSE gap p95 / max (s) |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | 78.1 | 138.2 | 44.45 | 3.516 / 4.730 |
| 1000 / 3000 | 116.5 | 169.4 | 36.28 | 0.727 / 0.753 |
| 2000 / 3000 | 85.3 | 142.4 | 43.14 | 1.656 / 1.708 |
| 3000 / 3000 | 80.1 | 138.5 | 44.37 | 2.132 / 2.284 |

The gap distribution pools short-request SSE update intervals that overlap the
long request's lifetime. These are client update gaps, not individual-token
latencies. Aggregate output throughput includes the time to finish all nine
requests, including queued short requests after the long prompt finishes.
Solo long-prompt TTFT ranges from 74.5 to 75.2 seconds across these settings.

At default targets, 454 of 456 sampled observations with both the long reader
and live decoders have a 256- or 512-token adaptive limit; none has a 64-token
limit. The separate fresh-controller default run takes 111.5 seconds for the
long prompt and 167.4 seconds to drain all requests, with 36.70 aggregate output
tok/s and SSE gap p95/max of 0.826/0.855 seconds. Its sampled mixed limits also
remain at 256 or 512 tokens.

Every enabled-target mixed run records zero decode-gap misses and six
first-token misses: the long prompt and queued requests still exceed the
three-second first-token target. Disabled targets do not count target misses;
the fixed-budget run's zero counter is not evidence of short gaps.

The warmed default is 1.49 times the fixed-budget long TTFT here and buys much
shorter decoder update gaps. Looser targets recover most of the fixed-budget
throughput in this probe. Neither the warmed nor fresh-controller case
reproduces the reported 64-token limit or 4.5-times slowdown. These measurements
do not establish the cause of that report or justify a global default change.
The shared whole-pass cost model remains a possible area to investigate with
the reporter's workload and repeated, alternating-order comparisons. No
MiMo-specific scheduling policy or controller-default change is introduced.
