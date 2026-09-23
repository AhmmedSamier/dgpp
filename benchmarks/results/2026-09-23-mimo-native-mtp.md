# MiMo native MTP3 on upstream — 2026-09-23

Native MTP3 improves code, JSON and arithmetic decode on this bounded two-Spark panel, but prose remains slower than MTP1. Recursive block-0 MTP3 performs worse on most probes. The native fix is available behind `DGPP_MIMO_NATIVE_MTP=1`; it remains opt-in.

## Controlled setup

- Branch base: upstream `c6ca191`, with the repaired compact tool grammar. Initial depth controls use frozen `fb92284`; native and final MTP1 use `b15860f` (manifest below).
- Both prior performance switches, `DGPP_MIMO_PREFILL_LAST_HEAD` and `DGPP_MIMO_MTP_CACHE_ONLY`, were off. Native heads intrinsically update unselected/history-only heads through KV append only, as in the saved implementation.
- Two DGX Sparks, C2, BF16 KV, eligible BF16 matrices stored as BF12, shared 131072-token pool, 1.5 GiB prefix snapshot budget, temperature zero and default thinking. The initial panels set `DGPP_RESIDENT_CACHE=0`, which was ineffective; the final control uses the correct `off` value. Startup and warmup are excluded.
- Nineteen timed requests per panel: two repetitions of four short C1 classes, two concurrent code/JSON pairs, two cold 7K/31K cases, one cold 67K case and two warm 67K repeats. Request-counter deltas and idle starts/ends verify exclusivity. Four additional serial acceptance probes are separate from timing.
- Native mode loads all three checkpoint heads. Head d uses the backbone hidden at p-d, not the preceding head output, with token p+1 and attention position p-d. [Implementation contract](../../docs/experiments/mimo-native-mtp.md).

## Client decode rate

Rate is `(completion_tokens - 1) / seconds_after_first_visible_content_or_reasoning_delta`. Entries are medians of two requests, except the single cold 67K decode. These are client streaming rates, not published engine-counter rates.

| Configuration | Code C1 | JSON C1 | Prose C1 | Maths C1 | 67K cold decode |
|---|---:|---:|---:|---:|---:|
| Initial MTP1 | 47.51 | 47.97 | 42.64 | 49.28 | 39.89 |
| Recursive block-0 MTP3 | 42.55 | 50.77 | 33.02 | 44.30 | 35.18 |
| Native blocks 0/1/2 MTP3 | 54.25 | 58.48 | 37.39 | 60.15 | 45.19 |
| Final same-binary MTP1 | 47.68 | 47.91 | 42.71 | 49.38 | 39.99 |

Native changes against the final MTP1 control: code +13.8%, json +22.0%, prose -12.5%, math +21.8%.

## Cold prefill / first-token time

| Configuration | 7,213 tokens | 30,876 tokens | 67,396 tokens | 67K warm |
|---|---:|---:|---:|---:|
| Initial MTP1 | 10.990s | 50.609s | 127.191s | 0.305s |
| Recursive block-0 MTP3 | 11.199s | 51.320s | 127.676s | 0.353s |
| Native blocks 0/1/2 MTP3 | 10.746s | 49.490s | 124.810s | 0.344s |
| Final same-binary MTP1 | 11.005s | 50.744s | 127.535s | 0.296s |

Native prefill also uses cache-only history for all three heads. This table measures the whole implementation, including that work reduction; it does not isolate head count from history-update cost.

## Acceptance: accepted at position / all draft rounds

| Configuration / probe | Position 1 | Position 2 | Position 3 | Accepted tokens / round |
|---|---:|---:|---:|---:|
| Initial MTP1 / code | 92.5% | — | — | 0.925 |
| Initial MTP1 / json | 98.4% | — | — | 0.984 |
| Initial MTP1 / prose | 69.5% | — | — | 0.695 |
| Initial MTP1 / math | 97.7% | — | — | 0.977 |
| Recursive block-0 MTP3 / code | 93.8% | 51.0% | 21.9% | 1.667 |
| Recursive block-0 MTP3 / json | 97.4% | 76.9% | 52.6% | 2.269 |
| Recursive block-0 MTP3 / prose | 73.2% | 25.2% | 3.1% | 1.016 |
| Recursive block-0 MTP3 / math | 98.9% | 52.7% | 23.7% | 1.753 |
| Native blocks 0/1/2 MTP3 / code | 93.2% | 79.7% | 74.3% | 2.473 |
| Native blocks 0/1/2 MTP3 / json | 98.5% | 95.5% | 92.4% | 2.864 |
| Native blocks 0/1/2 MTP3 / prose | 70.9% | 40.9% | 20.9% | 1.327 |
| Native blocks 0/1/2 MTP3 / math | 98.5% | 92.5% | 89.6% | 2.806 |
| Final same-binary MTP1 / code | 92.5% | — | — | 0.925 |
| Final same-binary MTP1 / json | 98.4% | — | — | 0.984 |
| Final same-binary MTP1 / prose | 69.5% | — | — | 0.695 |
| Final same-binary MTP1 / math | 97.7% | — | — | 0.977 |

The denominator is all draft rounds, including those rejected at an earlier position; these are not conditional acceptance rates. Attempt counts and raw counters are retained in the JSON and acceptance artifacts. Acceptance is strongly workload-dependent.

## Correctness and validation limits

- Both Sparks passed the independent CPU history/indexing oracle plus all 16 native state configurations (checkpoint/BF12 weights, BF16/FP8 KV, 2/23/97/2111-token prompts), including wrap, eager rollback, cross-slot prefix attachment and reopen.
- Both passed all six engine tests, including scalar and batched native depth-3 execution in a two-rank loopback world against plain greedy decode; both passed the native-config decode suite. Both Compute Sanitizer memchecks reported zero errors. Actual two-Spark serving is separately represented by these request panels.
- The first gate attempt failed native graph capture because D2D memcpy nodes violated the kernel-only graph contract. `3becf89` replaced them with the existing device-copy kernel. The second passed the graph suite, then rejected a no-MTP fixture via an overly strict option guard; `b15860f` leaves zero-head configurations unchanged. Neither failed attempt produced native real-model timing results. All attempts and restorations are retained.
- Output comparisons include both content and reasoning. Counts below are against the initial MTP1 panel. A differing concurrent native code response matched the initial MTP1 solo response exactly; concurrency changes scheduling and row shapes. This does not prove universal greedy equivalence.

| Panel | Exact text/reasoning matches | Differing request names |
|---|---:|---|
| Initial MTP1 | 19/19 | None |
| Recursive block-0 MTP3 | 19/19 | None |
| Native blocks 0/1/2 MTP3 | 18/19 | code-c2-r0 |
| Final same-binary MTP1 | 19/19 | None |

Native MTP3 passed all three actual custom-harness tool probes with populated and typed arguments (generated tools were not executed), eight prefix branch/reuse requests, and exact JSON retrieval at 30921 tokens, 67437 tokens. All other timing configurations passed eight prefix branch/reuse requests too. These fixed probes are not broad SWE-bench/HumanEval quality evidence, capacity stress, or a guarantee of MTP3 benefit for every workload.

## Memory and operational state

Native mode adds two paged KV layers and the backbone history/rollback ring. With the same 1.5 GiB prefix budget it allocates 94 snapshots of 16.281 MiB, versus 4096 small metadata slots in the one-head implementation. Native snapshots preserve the full history ring; compacting them is separate future work.

After the native quality checks, Spark-1 ran out of disk while staging the final-control executable. Its automatic restoration also failed before launch. The cause was an ineffective resident-cache switch: this version recognizes `DGPP_RESIDENT_CACHE=off`, not `0`. Native startup had replaced the one-head image with the three-head layout, reached disk capacity at layer 42 and continued loading all remaining weights from checkpoint. Timed requests began only after the full model was ready. The initial MTP1/recursive-MTP3 starts had also fallen back after an image-write warning. These startup failures produced no partial-model timing samples.

With both servers stopped, only the two experiment-generated resident images were removed, reclaiming about 69/79 GiB. The final control uses `off` on both ranks, verified in the environment and startup logs; no model checkpoint or unrelated image was removed. Cleanup manifests and both failed startup logs are retained. The earlier prefill report has been corrected too.

The original native-MTP3 service was restored and verified as an intermediate recovery (`raw/r4-native-restored/`). At the user’s request, the tested upstream-based native MTP3 candidate was then deployed on both Sparks and left running at `http://192.168.0.171:30001/v1`. Final state: source `b15860f`, native blocks 0/1/2, C2, BF16 KV, shared 131072-token pool, 1.5 GiB prefix budget (94 slots), `DGPP_MIMO_NATIVE_MTP=1`, and `DGPP_RESIDENT_CACHE=off`. Both earlier prefill switches remain off, matching the measured configuration. Binary/config/environment checks, idle/healthy metrics, both-rank logs and eight repeated-prefix requests are recorded in `raw/native-mtp3-manual/`. The shared 128K capacity is not interchangeable with the old service’s 256K per request.

Raw logs, all failed attempts, SSE streams, requests, usage, config attestations and restoration records: `/mnt/benchmarks/dgpp-mimo-mtp3-20260923/`. The dirty main checkout was not staged or reset.
