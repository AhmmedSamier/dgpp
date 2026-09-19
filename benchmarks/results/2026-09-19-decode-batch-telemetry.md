# Decode graph batch telemetry

## Scope

Based on upstream `444441feed0c4d7c256ff7ff7868b587823b737f`.
Extracts the decode-batch counters from historical commit `8825c4b`, excluding
its prefill tuning and performance experiments. It does not change graph
capacity, scheduling, MTP acceptance accounting or deployment configuration.

`SchedulerEngine` supplies zero defaults; the graph adapter records successful
launches, capacity, active requests and actual verification width. Scheduler
snapshots publish these values through both metrics aliases. The histogram
includes scalar fallback, and the last-launch fields persist after requests
close. A compile-time check keeps the histogram large enough for the picker.

## Validation

Validation is performed in this isolated worktree with a disposable native
x86-64 build container derived from `dgpp-spark-cross:upstream-pr`, adding native
libibverbs, cuBLAS and clang-format packages. No production processes or
configuration are changed. Logs are retained locally under
`artifacts/decode-batch/`.

Commands and results:

- `cmake --preset ci`: configured with CUDA 13.0.88 and GCC 13.3.
- `cmake --build build-ci -j 4 --target serve_test scheduler_test qwen_engine_test`:
  all three targets built successfully, including CUDA kernels and the graph
  adapter instantiated by the Qwen test. This is native x86-64 compilation,
  not ARM64 execution.
- `DGPP_TEST_FILTER=serve_decodeBatchMetrics build-ci/serve_test`: one test passed.
  It covers zero defaults, publication from the engine, both endpoint aliases,
  every histogram bucket and retained last-launch shape at zero occupancy.
- `ctest --test-dir build-ci -R '^(serve_test|scheduler_test)$' --output-on-failure -j 1`:
  both suites passed (58 service and 57 scheduler cases) in 118.89 seconds.
- Changed C++ ranges formatted with clang-format 18; `git diff --check` passed.

The Qwen assertions exercise scalar MTP depths one and two, last-launch retention
at close, full six/eight-slot batches and padding after an interior slot retires.
These assertions are not claimed as passing until they execute on target hardware.

## Remaining contribution gates

Run the new Qwen scalar/depth-two and wide sparse-batch assertions serially on
idle target hardware. Perform the required fabric serving check and compare
shutdown op-stream digests across participating ranks. A workstation build
cannot establish target runtime behavior; no GPU timing or throughput claim is
made here. The contribution guide's full build and suite remain required before
upstream readiness. PR creation is intentionally held.
