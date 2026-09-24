# Issue #45: watchdog progress inside a slow prefill chunk

The watchdog has a real false-timeout case: successful tensor-parallel reductions
inside an internal prefill chunk did not reset its 120-second deadline. A
controlled two-Spark Qwen test failed on the baseline and completed with the fix.
The fix still terminates a genuine stall at the same deadline.

The full-model investigation also completed: a 261,120-token chat request
succeeded on both the reported source revision and that revision with the fix.
Thus the available full-model setup did **not** reproduce the reporter's natural
failure. Total prefill time greater than 120 seconds is insufficient to trigger
it; a gap of that length between observed progress updates is required.

## Full-model comparison

September 24, 2026; two DGX Spark GB10 nodes, CUDA 13, TP2,
`nvidia/Qwen3.8-Flash-Next-NVFP4`, KV capacity 262144,
`decode_graph: false`, `mtp: false`, `prefill_budget_tokens: 0`.
Additional settings included a 1.5 GiB prefix cache, mmap n-gram table, two
request slots, BF16 KV, FP8 dense weights and `bf16_weights: bf12+bf16`. The request contains one user message
with 261,068 repetitions of ` a`; the chat template adds 52 tokens. The server
confirmed exactly 261,120 prompt tokens and zero cached prompt tokens in both
runs. Generation used temperature zero and `max_tokens: 1`.

| Build | HTTP | Client wall time | Internal chunks | Longest chunk |
| --- | --- | --- | --- | --- |
| Stock `92cfc682` + timing logs | 200 | 275.335 s | 130 | 70.988 s |
| Same + watchdog fix | 200 | 217.956 s | 130 | 5.707 s |

Both returned the same single reasoning token (`We`), and both ranks exited zero
on the subsequent clean shutdown. Neither watchdog fired. The largest internal
chunk was 2,048 tokens; prefix/snapshot boundaries introduced smaller chunks.
Sampled metrics showed token progress advancing throughout the long scheduler
pass. Both runs began with a fresh server and no prior inference request in that
server. Loader/filesystem caches were warmer for the second run, so the wall-time
difference is **not** evidence of a speed improvement from the fix.

The reporter's local loader patch, original request and complete configuration
were unavailable. These results match the reported public settings and prompt
length, not their entire environment. An initial token-count calibration request
contained 261,160 tokens and was manually interrupted; its shutdown-deadline
exit is not counted as a watchdog regression result.

Operation streams matched across both ranks after clean shutdown:

- Baseline: `dcf07fc57ea11de83837c07e0552cf11` (MD5, 247 bytes per rank).
- Fixed: `68ae1cc4759b2bb87580a50fea3290e0` (MD5, 247 bytes per rank).

Request IDs differ between runs; after normalizing those IDs, the streams match
across builds too. Both selected token ID 1596.

## Controlled reproduction

The small NVFP4 Qwen fixture has four layers and one PLE layer. Settings:
`decode_graph: false`, `mtp: false`, `prefill_budget_tokens: 0`, KV capacity 512,
prefix cache disabled. The prompt `Test` consists of four tokens in one chunk.
Its ten reductions execute through the real model, GPU kernels and network.

Identical test-only instrumentation in both binaries waits 15 seconds **after**
each successful `boundary_->reduce` return. This makes a healthy chunk take about
150 seconds while completing a reduction every 15 seconds. Metrics show zero
completed tokens until the chunk finishes. The production patch contains no
injected delay, model instrumentation or timeout increase.

The baseline exited rank 0 with status 2 at about 120 seconds, after eight
successful reductions; its peer exited 3 and curl returned 52. The candidate
completed the chunk in 150.079 seconds and returned one output token. A subsequent
ordinary request also completed (6 ms prefill). After arming an indefinite CUDA
wait in that same candidate server, rank 0 still exited 2 at about 120 seconds,
rank 1 exited 3 and curl returned 52.

The fixture client probe samples once per second, so its elapsed measurements
include up to one second of polling overhead. Full-model client times above come
directly from curl; chunk timings come from server logs.

## Fix and validation

The bus publishes an atomic epoch only for successful collective completions.
The serving watchdog observes it alongside scheduler passes and completed token
positions. Eager, bulk, stream and graph collectives publish completions.
Submission, polling and failed/poisoned work do not publish progress. The watchdog
still calls neither CUDA nor lock-taking bus/metrics operations, and keeps its
120-second deadline and fatal exit behavior.

- Six host watchdog tests passed, covering idle serving, repeated passes,
  advancing/stalled tokens, advancing/stalled collectives inside a chunk, and
  exit with a blocked logging pipe.
- GPU/fabric tests passed: pending submissions do not count; successful eager
  and bulk operations advance once; a missing-peer failure does not count;
  graph and stream operations advance once per generation at worlds two and four.
- The controlled TP2 serving test exercises the application's actual wiring.
  Its subsequent indefinite CUDA wait confirms that genuine stalls still exit.
- The final server linked without model instrumentation also served the small
  fixture and logged a clean shutdown. The full-model candidate above used only
  chunk timing logs, without any delay injection.
- Changed translation units compiled with warnings as errors; `git diff --check`
  passed.

Targeted test commands:

```sh
./build-ci/engine_watchdog_test
DGPP_TEST_FILTER=collective_progress CUDA_MODULE_LOADING=EAGER ./build-ci/bus_test
```

The investigation binaries were linked incrementally against unchanged CI
libraries. For the full-model comparison, the Qwen head/kernel translation units
were rebuilt at `92cfc682`; all other reused production source translation units
are unchanged between that revision and master `76ac4d0`. Both full-model binaries
have identical chunk timing instrumentation; only the candidate adds the fix.
The reused build directory embeds an older version stamp. The tested source revisions and binary SHA-256 hashes identify the artifacts:

- Full-model baseline: `ac779ab2a7127ab7b8a555beeff023b70434452cf9e3ba6640e0d0feefb6661b`.
- Full-model fixed: `0b2020a1215b32d22cec8064d3e7372a697c4188fc4bdc42d30a9e1bc7d7e5b5`.
- Controlled baseline: `88f7f7aa42a140c264522cf995403657b1a5f981d32050135bfa568f104447ca`.
- Controlled fixed: `f3ad92d73235bf216828ce29e86d952cabddccbfbb80cd2850c5fc908e39aeba`.

The controlled-fixture baseline tree `b7f6c71` is identical to master `76ac4d0`.

## Interpretation and remaining limits

Budget zero retains internal Qwen chunks of at most 2,048 tokens. At the full
checkpoint's 48 layers, a chunk performs 96 attention/FFN reductions plus two PLE
reductions. The reported advance to sequence 83 is consistent with continued
work inside the first chunk; it does not imply the entire prompt is one forward.
The full-model results and source inspection confirm this distinction.

The fix covers successful multi-rank collectives. A single-rank model, or an
individual operation with no completed chunks or collectives for 120 seconds,
still reaches the existing deadline. An initial experiment that delayed every
CUDA synchronization reached this limit before its first collective; that
broader injection did not isolate the reported advancing-collective case.
