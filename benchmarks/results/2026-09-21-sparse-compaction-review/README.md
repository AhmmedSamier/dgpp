# Sparse compaction review follow-up

The review of `0425604` identified a rank-agreement gap: the environment switch was read independently on each rank. `engine.compact_batches` now defaults to false, is validated as a boolean, travels in the settings journal, and is passed explicitly to graph-engine construction after peers adopt rank 0's settings. The runtime environment switch is removed. Missing fields in older settings records default off; invalid journal flag values are rejected by name. Qwen fixture tests opt in explicitly through a test-only setting, with a dedicated disabled-policy CTest lane.

Mapped commit capture now rejects fewer than two physical slots before deriving a stride and checks the uniform destination layout across all slots in debug builds. A bus-free regression exercises mapped commit, draft rows, chain rows, PLE hashing, PLE context and PLE convolution with an accepted padded group whose request ID is -1. It checks unchanged unmapped state as well as padding outputs. This provides a memcheck target without RDMA deadlines:

```sh
DGPP_TEST_FILTER=compact_mapping compute-sanitizer --tool memcheck --error-exitcode 99 build-ci/glm_pick_test
```

Touched C++/CUDA ranges were formatted with clang-format 19 using the repository style. The batch map is grouped with graph execution state, and the new kernel no longer reopens the enclosing namespace. The validation record uses a neutral node description.

## Validation and limits

- Five configuration tests passed in a standalone native host build, including opt-in/default-off and invalid-type rejection.
- `fabric_serve_test` passed on the workstation: journal settings round-trip, old-record default, invalid-flag rejection, real TCP/HTTP lifecycle, cancellation, shutdown and failure propagation with fake engines.
- ARM64 C++20 syntax checks of the serving application and Qwen/sampler CUDA test sources passed with warnings as errors using CUDA 13 headers.
- The modified speculative CUDA kernel translation unit compiled for SM 12.1 with the ARM64 cross toolchain.
- Production was not stopped or changed. Updated GPU tests, debug capture assertions, and memcheck have not yet run. Earlier GPU evidence applies to `0425604`, not this follow-up.

The maintainer reports a full 113/113 suite with no skipped cases at `0425604`, no measurable dense-occupancy regression at the tested C1-C4 shapes, a 26 percent sparse step-time improvement, and matching rank operation streams. See the PR review for the exact protocols. These are maintainer-reported results and do not validate the later edits.

Snapshotting all physical slots remains unchanged. Measure its C16 cost before optimizing the mapping further. Full-model teacher-forced comparisons and four-node Qwen validation remain pending; the feature stays default off. The maintainer's four-node GLM run supplies shared-kernel evidence only. PR #13 remains separate and its overlapping wide-lane tests must be reconciled when either PR lands.
