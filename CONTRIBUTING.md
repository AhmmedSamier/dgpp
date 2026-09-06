# Contributing

DGPP is developed against real hardware: a DGX Spark (GB10) node builds and
runs the single-node suites, and the four-node RoCE fabric runs the
multi-node gates and the serving evidence. This page is what a change needs
to land.

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
  node, the four-node rituals cover the wire.

## Build

```bash
cmake --preset ci && cmake --build build-ci -j -- -k    # warnings as errors; -k keeps going past the first error
ctest --test-dir build-ci --output-on-failure
```

`scripts/ci-local.sh` does exactly that. The other presets are `release`
(what `scripts/release.sh` packs), `debug`, `asan` and `ubsan`. Two rules
that were learned the hard way:

- **Run a full build before `ctest`.** The suite runs whatever binaries
  exist; a target you did not build by name is a stale binary that passes.
- **One CUDA suite at a time on a node that is also serving.** The loopback
  worlds share the GPU with the service and each owns a port in
  29910–29941. `DGPP_TEST_FILTER=<substring>` runs a subset of a binary.

## What a change needs

1. **A gate.** New behavior gets a test that would fail without it, in the
   suite whose subject it is (`tests/unit` for pure logic, `tests/host` for
   the scheduler, the service and the journal with their fakes,
   `tests/cuda` for kernels and the loopback worlds). Refusals are gated by
   name: a request or a config the code rejects must be rejected with the
   offending field in the message, and a test says so.
2. **Evidence on the fabric** when the change touches the path the ranks
   execute together: boot the world (`scripts/dgpp-cluster up`), run the
   relevant ritual (`docs/operations.md` lists them; `scripts/serve_api_check.py`
   for the request contract), stop it (`down`) and keep the four op-stream
   md5s identical. A numerics change is judged as `docs/numerics.md`
   describes, not by eye.
3. **The record.** Every change that measures or fixes something gets an
   entry in the engineering record, `benchmarks/results/2026-08-29-bus-m5.md`
   (a dated `##` heading, what was built, how it was gated, what the fabric
   showed, what the docs say now). The record is the memory of the project;
   the earlier entries keep the names of their day.
4. **The docs as built.** `DESIGN.md` describes contracts as they are, not
   as planned; `PLAN.md` the status; `docs/operations.md` and `README.md`
   whatever an operator or a new developer would now read differently.

## Style

- C++20, two-space indent, Google-derived formatting (`.clang-format`);
  format new and touched code with the `format` target (the tree has not
  been reformatted wholesale, so `format-check` over everything is not
  yet clean).
- Comments say why, and they name the day and the record entry when a
  line exists because of something that happened. Loud failure beats a
  silent fallback: the refusal ladder, the journal's protocol checks and
  the configuration handshake all follow that rule.
- Determinism across ranks is a contract, not a goal: anything a rank
  decides that another rank must agree with is a pure function of the
  journaled stream, and the op-stream fold checks it every tick.
- Python scripts use the standard library only.

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
