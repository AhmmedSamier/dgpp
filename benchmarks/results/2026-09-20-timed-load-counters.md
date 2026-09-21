# Reconcile benchmark counters with completed requests

Base: `4f04f16dae597d4665475b2a771bf6b6725267de`.
Issue: [#24](https://github.com/HawkBearPig/dgpp/issues/24).

The timed load client used to accept the first idle metrics response whose
`prompts_prefilled` counter had advanced, then reject a token-count mismatch
as unrelated traffic. In `GenerationService::route_metrics`, the active and
queued gauges come from live request records, while cumulative scheduler
counters come from the last tick's published snapshot. A completed SSE
response and idle live gauges can therefore precede publication of its final
tokens. Two identical stale reads would not establish readiness either.

## Change

The maintained client is now `scripts/timed_load.py`. The dated entry point
delegates to it, resolving the scripts directory from its own path so existing
callers continue to work from any working directory. Imports no longer replace
`serve_load.phase` globally.

The client reads metrics immediately. After known requests complete, it accepts
a snapshot only when both live gauges are idle and the prompt/token deltas
exactly match the requests' usage. Behind counters are polled under a monotonic
five-second deadline. The 10 ms polling interval only limits HTTP traffic;
neither elapsed time nor repeated identical values makes a snapshot acceptable.
HTTP timeouts use the remaining budget, and responses arriving after the
deadline are rejected. Excess counts and counter resets fail immediately.
Errors report the expected and observed deltas and live gauges.

The verified snapshot is carried forward, with a zero-delta check before the
next measured phase. `serve_load.main` accepts optional callbacks so the timed
client can reconcile warmup, the single isolation request and the concurrent
isolation batch as well. This prevents their late counters from entering the
first measured phase. The ordinary load client keeps its existing behavior.
As before, the experiment requires an otherwise idle server at startup.

## Reproduction and tests

Before changing the client, a controlled four-request phase returned 320 tokens
per request. Metrics reported an idle world and four prefills, but only 1278 of
1280 new tokens. The old client raised `scheduler counters include unexpected
requests` after its second metrics read, before reading the correct snapshot.

The new CTest entry, `timed_load_test`, contains 13 tests. Fake-clock tests
cover immediate success without sleeping, repeated stale snapshots, a final
two-token update, lagging prompt counts, busy gauges, deadline expiration,
HTTP timeouts, late HTTP responses, excess counts, resets, traffic between
phases and invalid engine times. They also check that client results pass
through unchanged and failed phases do not append an engine report.

A local HTTP/SSE fixture publishes the final two tokens only on the third
metrics read after each request group. CLI coverage runs warmup, isolation at
concurrency four, two prompt classes, concurrency one/four and two repetitions:
26 requests and eight measured phases per run. The ordinary load client, the
new timed client and the dated entry point send identical request bodies. All
timed phases reconcile, exclude warmup/isolation work and report the fixture's
exact expected engine rates. A separate immediate-publication mixed sweep
covers the CLI path without warmup, isolation or prompt classes.

An additional A/B run executed the base client and the fixed client against
the same HTTP/SSE fixture with `--warm 0 --concurrency 1,4 --repeat 2
--max-tokens 8`. With immediate publication, every phase's engine counters,
engine rate, prompt hashes, token counts, usage and transcript matched exactly.
With delayed publication, the original CLI reproduced the false failure; the
fixed CLI produced the same four correct reports. The fixture deliberately
assigns 2 ms per decode token, so its 500 tokens/s is a test expectation,
not a model-throughput measurement.

## Build and broader validation

```bash
python3 tests/python/timed_load_test.py -v
cmake --preset ci
cmake --build --preset ci -j 4
CUDA_VISIBLE_DEVICES="" ctest --test-dir build-ci -L host -V -j 1
```

The 13 targeted tests and the full native build passed. All 29 host CTest
entries passed in 167.56 seconds, including the nine checkpoint-backed
gates, 61 service cases and the Python script/launcher checks. Each of the
two unit entries ran 221 host cases and skipped the same three CUDA arena
cases because CUDA was deliberately hidden. No checkpoint gate was skipped.
Python 3.10 syntax checks for all changed Python files and `git diff --check`
also passed.

## Performance and accuracy scope

The client request corpus, sampling settings, SSE parser and measured client
intervals are unchanged. Reconciliation occurs after each completed request
group, outside the phase's wall timing. Engine rates retain the formula
`1000 * (tokens_generated - prompts_prefilled) / step_ms`. A ready snapshot
requires one immediate read and no polling delay; delayed publication adds
only the reads needed to satisfy the counter condition.

This change affects Python benchmark accounting and entry points. No server,
scheduler, rank protocol, kernel or model arithmetic changed. The deterministic
HTTP comparisons validate workload and measurement preservation; live model
accuracy and GPU-throughput benchmarks were not rerun, and production was not
restarted. No published model-performance numbers were changed.
