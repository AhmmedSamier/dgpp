# Contributing

DGPP is developed against real hardware: a DGX Spark (GB10) node builds and
runs the single-node suites, and the four-node RoCE fabric runs the
multi-node checks and serving benchmarks. This page describes how to
prepare and validate a change.

## Environment

- A GB10 node with CUDA 13 (the toolkit, cuBLASLt, an SM 12.1 compiler),
  CMake 3.24+, a C++20 compiler, rdma-core with libibverbs, Python 3.10+.
  Without libibverbs, configure with `-DDGPP_ENABLE_IBV=OFF`; the bus and
  the fabric gates are then unavailable.
- The checkpoint in the Hugging Face cache (`~/.cache/huggingface/hub`)
  for the real-model checks; the synthetic fixtures the tests write need
  nothing.
- The fabric for anything under `src/net/`, the graph engine, or the
  service's multi-rank paths: the loopback gates cover the protocol on one
  node; four-node checks exercise the network.

## Build

```bash
cmake --preset ci && cmake --build build-ci -j -- -k    # warnings as errors; -k keeps going past the first error
ctest --test-dir build-ci --output-on-failure
```

`scripts/ci-local.sh` configures the CI preset, builds and runs CTest.
The other presets are `release` (used by `scripts/release.sh`), `debug`,
`asan` and `ubsan`.

- **Run a full build before `ctest`.** The suite runs whatever binaries
  exist; an old binary can pass without testing your changes.
- **One CUDA suite at a time on a node that is also serving.** The loopback
  worlds share the GPU with the service and each owns a port in
  the 299xx range. `DGPP_TEST_FILTER=<substring>` runs a subset of a binary.

## What a change needs

1. **Tests.** New behavior gets a test that would fail without it, in the
   suite whose subject it is (`tests/unit` for pure logic, `tests/host` for
   the scheduler, the service and the journal with their fakes,
   `tests/cuda` for kernels and loopback worlds). Test invalid requests
   and configuration values as well, including whether the error names
   the offending field.
2. **Evidence on the fabric** when the change touches the path the ranks
   execute together: boot the world (`scripts/dgpp-cluster up`), run the
   relevant check (`docs/operations.md` lists them; `scripts/serve_api_check.py`
   for the request contract), stop it (`down`) and keep the four op-stream
   md5s identical. A numerics change is judged as `docs/numerics.md`
   describes, not by eye. A change that moves a published throughput number
   re-runs that number's procedure from `docs/benchmarks.md` §9 and updates the
   row there, with its date.
3. **Engineering records.** Changes with new measurements or debugging
   findings get an
   entry in the engineering record, `benchmarks/results/2026-08-29-bus-m5.md`
   (a dated `##` heading, the change, its validation and the results).
   Preserve the context and names used in historical entries.
4. **Documentation.** Keep `DESIGN.md` aligned with the implementation and
   `PLAN.md` with its status. Update `docs/operations.md` and `README.md`
   when a change affects setup, usage or supported behavior.

## Style

- C++20, two-space indent, Google-derived formatting (`.clang-format`);
  format new and touched code with the `format` target (the tree has not
  been reformatted wholesale, so `format-check` over everything is not
  yet clean).
- Write documentation in direct, connected prose. Explain current behavior
  and its constraints before the details. Comments should explain intent,
  invariants or a non-obvious tradeoff. Put investigation chronology in a
  dated engineering record and link to it when the evidence helps explain
  a constraint; avoid dates and milestone labels in routine comments.
- Reject invalid configuration and protocol state with a useful error.
- Preserve determinism across ranks: anything a rank
  decides that another rank must agree with is a pure function of the
  journaled stream, and the op-stream fold checks it every tick.
- Operations scripts use the Python standard library. Reference generators
  may use optional dependencies and should document them.

## Commits

One change per commit, with a subject that says what changed and why in
plain words; the record entry carries the detail. Commit to `master`
after the full build and the suite pass and the fabric evidence, when the
change needed it, is in the record.

## Reporting a problem

Include the version (`dgpp-serve --version`, or the `dgpp-serve 0.x` line
at the top of a rank's log), the `config:` line every rank logs at
startup, rank 0's log and the peers' (`dgpp-cluster down` fetches them
into the log dir), and the op-stream md5s if the world stopped cleanly.
