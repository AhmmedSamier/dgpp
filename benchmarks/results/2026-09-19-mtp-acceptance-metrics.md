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

## Limits

The host fixture verifies publication and HTTP serialization of existing
counters, not real-model acceptance or numerical behavior. This branch does
not change paths that tensor-parallel ranks execute together, so it introduces
no new fabric-evidence requirement. The full CUDA/checkpoint/RDMA suite from
`CONTRIBUTING.md` remains a merge gate on reserved idle hardware; production
inference was not stopped, restarted or used for this validation.
