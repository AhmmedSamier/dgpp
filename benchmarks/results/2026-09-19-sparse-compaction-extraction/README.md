# Sparse decode-batch compaction extraction

## Scope

Prepared on `codex/upstream-sparse-compaction` against upstream `444441feed0c4d7c256ff7ff7868b587823b737f`, extracting `42b105dc33d2a7d1558798d3e6be090153debfe8`.
Fixed-depth Qwen selects graph families by active request count while preserving physical request state. Other model families and confidence-scheduled depth retain their existing selection policy.

The extraction retains upstream's 16-row limit. Sampling regression cases use eight physical slots at MTP depth 1. Existing family replay counters check sparse-family selection without depending on the separate decode telemetry change. The two sampling lanes are marked serial GPU/RDMA tests.

## Validation status

Merge conflicts resolved; whitespace checked. An isolated Sol High review found no confirmed actionable defects. Native compilation, 15 selected CTest runs, an isolated sampler memcheck, and two-node real-model API/concurrency checks passed on September 20. Production was stopped for serial GPU/RDMA validation and restored to its exact original configuration and binaries. See the [validation record](../2026-09-20-sparse-compaction-validation/README.md) for scope, hashes, results and limitations.

The [original measurements](../2026-09-18-compact-batch-mappings/README.md) describe an older combined branch, including wider decode support. They are historical evidence, not validation of this upstream extraction.

## Remaining contribution gates

- Format touched C++/CUDA code.
- Complete full-model teacher-forced compaction-on/off comparisons and any missing adversarial mapping/fallback coverage.
- Build all targets before the full suite; record skipped checkpoint cases explicitly.
- Complete four-node fabric validation; the successful two-node check does not replace this gate.

The extraction is prepared for draft review; the remaining gates above are not complete.
