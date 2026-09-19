# MTP acceptance metrics over HTTP

## Scope

Base: upstream `444441feed0c4d7c256ff7ff7868b587823b737f`.

Expose the existing `Scheduler::Meters::mtp` snapshot as
`scheduler.spec_decode` on `/metrics` and `/v1/metrics`. The change adds no
engine reads from the HTTP thread, GPU synchronization, kernels, journal
operations or collectives. It contains no Home Assistant integration,
decode-batch telemetry or deployment changes.

The engine increments attempts for each verified speculative position and
accepts only for the accepted prefix. Variable verification depth makes the
sum of attempts the appropriate draft-token denominator. Position zero's
attempts count request verification rounds with drafts. These counters precede
response stop/length trimming and exclude padding and the non-speculative row.
See the [API contract](../../docs/openai-compatibility.md#speculative-decoding-counters).

## Validation

The native x86-64 host build used an isolated Ubuntu 24.04 container with
GCC 13.3, CMake 3.28.3, CUDA 13.0.88, native cuBLAS development libraries
(`libcublas-dev-13-0` 13.1.1.3-1), libibverbs 50.0 and Python 3.12.3.
No GPU devices were passed to the container. Changed C++ ranges were formatted
with clang-format 18.1.3 and the repository style.

```bash
cmake --preset ci
cmake --build build-ci -j 4 --target unit_tests http_server_test serve_test fabric_serve_test scheduler_test roster_check
ctest --test-dir build-ci -L host -LE checkpoint --output-on-failure
```

- Configuration and all six selected targets passed with warnings as errors.
- The new HTTP regression passed for both URLs at depths 0, 1, 3 and 8,
  including variable-depth totals, counts above 32 bits and a return to zero.
- With only `generation_service.cpp` restored to the upstream base and the
  test rebuilt, the regression failed with `minijson: missing key 'spec_decode'`.
  The change was then restored and all selected targets rebuilt.
- All 16 host CTest entries passed, with zero failures, in 166.81 seconds.
  This includes the service, scheduler, fabric-service fake, unit, HTTP,
  roster and nine Python suites; checkpoint-labelled cases were excluded.

Local raw logs are under the ignored `artifacts/mtp-metrics/` directory.

## Initial Spark hardware validation

Initial tested source commit: `616c3de`, before the sampled-fallback accounting
correction described below.
A separate checkout on a GB10 Spark was configured with `cmake --preset ci`
and fully built with `cmake --build --preset ci -j 4` before testing. Both
production ranks were stopped for GPU/RDMA execution. The serial command was:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=32 ctest --test-dir build-ci --output-on-failure -j 1 --timeout 300
```

The full run completed in 630.38 seconds: **101 passed, zero failed, 10 skipped**.
The skipped tests require unavailable checkpoint data: `tokenizer_test`,
`dsv41_tokenizer_test`, `dsv41_prompt_test`, `glm4_tokenizer_test`,
`glm_dsa_tokenizer_test`, `chat_template_test`, `glm4_chat_template_test`,
`glm_dsa_chat_template_test`, `glm_vision_stream_test` and
`glm_vision_frontend_test`. Optional real-checkpoint cases inside other binaries
also report their own skips; process success does not validate absent models.

The same native binary was then staged to both Sparks for a separate TP2
Qwen3.8-Flash-Next-NVFP4 deployment, with concurrency 4, MTP depth 3, 65,536 KV
tokens and a 1 GiB prefix cache. Three identical counting requests verified
that both metrics routes returned the same `spec_decode` object:

```json
{
  "depth": 3,
  "num_drafts_total": 84,
  "num_draft_tokens_total": 252,
  "num_accepted_tokens_total": 252,
  "num_draft_tokens_per_pos_total": [84, 84, 84],
  "num_accepted_tokens_per_pos_total": [84, 84, 84]
}
```

The second and third requests each reused 1,136 cached prompt tokens.
`scripts/serve_api_check.py` passed its stop, streaming, multiple-choice,
logit-bias and usage checks. After clean shutdown, both ranks' operation
streams had MD5 `203b9e4b26af40bd8224e12d9820f290`.

The original production release `0.1.0+g42b105d-mtpmetrics` and unchanged
configuration were restored on both ranks. Three health-check requests
succeeded; the second and third reused 1,128 prompt tokens. Metrics reported
148 cache slots and no engine failure, and both live rank logs confirmed
successful request retirement. Production deployment settings were not changed.

The first maintenance attempt stopped at a harness self-SSH host-key check
before GPU tests and automatically restored production successfully. The
check was changed to run locally on rank 0 before the successful run. The
harness also uses the older production release's `/v1/metrics` route; the
new branch was checked through both aliases.

Raw build, CTest, live API, metric, shutdown and restoration evidence is saved
locally under `artifacts/mtp-metrics/server-validation/` (ignored by Git).

## Sampled-fallback accounting correction

The isolated review found that `collect_verdict` counted accepts from the
provisional device verdict. When a capped candidate prefix requires exact
host fallback, the host can subsequently accept additional drafts without
those positions reaching the counters. Count attempts and accepts after
fallback resolution, using the final decided prefix; keep the scheduled
verification depth as the attempt denominator. Both engine-lifetime and
per-slot counters use the same accounting loop. Token decisions, RNG draws,
rollback and collective operations are unchanged.

The existing sampled-MTP eager-oracle gates now assert per-position lifetime
and live-slot counters at depths 1 and 2. Seed 814480 exercises accepted drafts
under the capped candidate prefix. Expected counts come from the eager
speculator's committed prefix before response length trimming, across scalar
and batched phases and slot reuse.

Both new regressions were built against the unfixed engine and failed with
`lifetime MTP counters include final fallback accepts` (two tests, two
failures). Production was restored after that short regression window, and
response generation and cache reuse passed. The corrected source then
completed a full native `ci` build successfully.

**Validation pending:** the corrected engine's GPU regression, full suite and
fabric checks have not run. Hardware belongs to another task; no follow-up
maintenance job from this thread is running or queued. The successful full
suite and TP2 results above apply to the initial implementation, not this
correction. Local evidence and harness files are under the ignored
`artifacts/mtp-metrics/fallback-fix/` directory (remote build and regression
logs remain in the matching directory in the isolated Spark checkout).

## Limits

The live counting requests establish real-model counter publication, not a
representative acceptance-rate or throughput benchmark. Variable-depth
arithmetic is covered by the host regression. GPU/RDMA fixtures and a physical
two-Spark serving world were exercised; a physical four-node run and the
unavailable checkpoint cases were not. The correction changes host-side accounting in the shared graph engine.
The relevant engine regressions and fabric checks must be rerun before
claiming the corrected branch is hardware-validated.
