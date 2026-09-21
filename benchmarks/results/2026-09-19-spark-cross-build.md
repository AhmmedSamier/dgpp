# Spark cross-build validation

The cross-build adds an Ubuntu 24.04 Docker environment, a CMake toolchain and
release-derived preset, and a small Bash wrapper for building ARM64/GB10
binaries on x86 Linux. It changes no inference code or deployment settings.

Base: upstream `a0708b969650ae1a8a65b8a1877fdcd45ce76590`. The original tooling
was isolated from `d272609`; the release preset retains the base's server-only
target and optimization/stripping settings.

## Reproduction

Run from a fresh checkout on x86 Linux with Docker:

```bash
docker build --no-cache --platform linux/amd64 -t dgpp-spark-cross:upstream-pr -f dev/Dockerfile.spark-cross dev
DGPP_CROSS_IMAGE=dgpp-spark-cross:upstream-pr scripts/spark-cross build
DGPP_CROSS_IMAGE=dgpp-spark-cross:upstream-pr scripts/spark-cross command cmake --install build-spark-cross --prefix "$PWD/install/spark"
python3 tests/python/spark_cross_test.py
python3 -m unittest discover -s tests/python -p '*_test.py'
```

## Dependency handling

CUDA's cross cuBLAS package contains a link stub. Install staging needs the
actual ARM64 cuBLASLt shared library. The Dockerfile extracts that library
from the ARM64 runtime package because installing both packages introduces
overlapping cuBLAS files. PCRE2 uses upstream's pinned static source build.
Package repository versions are not locked; record the installed versions
when reproducing this build.

Docker 28.2.2 built the clean image with CMake 3.28.3, AArch64 GCC/G++ 13.3.0, nvcc 13.0.88,
Python 3.12.3, cudart 13.0.96, cuBLASLt 13.1.1.3 and ARM64 libibverbs
50.0-2ubuntu0.2. Its local image ID was
`sha256:2b4256500b6eca04e1e561b9f460196a59cde5aa39bb143e9fd7fa81e7d42954`.

Linked worktrees need their common Git directory mounted read-only as well
as the source directory; otherwise the normal version target stamps the
binary as unknown. A real container probe resolves the source revision with
that mount. The wrapper regression covers its paths and access mode.

## Validation

- Clean Docker image build without cached layers passed.
- Ten wrapper tests cover CLI validation, argument boundaries, build failure
  propagation, missing images, ownership and linked-worktree metadata mounts.
- The workstation Python suite and the Python suites registered with CTest
  have two failures already present on unchanged upstream `a0708b9`:
  `PortabilityTest.test_scripts_index_and_setup_document_links` (upstream's
  missing `vision_prefix_cache_check.py` index entry) and
  `SiteEnvTest.test_cpp_fixture_matches_resolved_example` (prefill settings
  differ from the checked-in resolved fixture). The new script is indexed.
  Baseline discovery ran 150 tests with two failures and three skips.
  Final candidate discovery ran 160 tests: 155 passed, the same two failed,
  and three were skipped.
- Container CTest ran nine Python suites: seven passed, including all ten
  cross-build tests; two failed for the same baseline issues. These tests
  require a passwd entry for the numeric host UID, which the build image
  intentionally does not create. The test invocation mounted a temporary
  passwd file containing a `builder` entry at `/etc/passwd:ro`; ordinary
  builds need no passwd mapping. Running without that test fixture also
  exposes unrelated `getpwuid()` errors in the existing launcher tests.

- The server built successfully from an empty `build-spark-cross` directory,
  including PCRE2 10.45's C sources with the ARM64 compiler. Rebuilding with
  linked-worktree metadata visible restamped and relinked the server.
- The documented `cmake --install` command passed. `file` and `readelf`
  identify the server, cudart and cuBLASLt as AArch64 ELF files. The server is
  stripped with no `.debug_info`, retains `$ORIGIN/../lib` as its install
  RUNPATH, and needs `libcudart.so.13` and `libcublasLt.so.13`. There is no
  dynamic PCRE2 dependency. PCRE2's license is staged, while standalone
  PCRE2 archives, headers and tools are excluded from the runtime layout.
- Shell syntax, preset JSON syntax and `git diff --check` passed.

Local logs and package checks are retained under the ignored
`artifacts/cross-build-pr/` directory. No ARM64 executable was run on x86.

## Spark hardware validation

The initial workstation preparation above did not contact the inference
nodes. A subsequent authorized maintenance window validated source commit
`53a533930664eed0407cc0997f8f16e51d12d3d2` on GB10 with driver 580.173.02.
Production was stopped through its existing launcher before GPU/RDMA work;
the stopped world's operation-stream digests matched across both ranks.

The full native `ci` build and full Docker `spark-cross` build passed. Both
test sets were freshly built, then run serially on the same idle Spark:

```bash
cmake --preset ci
cmake --build --preset ci --parallel 2
CUDA_DEVICE_MAX_CONNECTIONS=32 ctest --test-dir build-ci --output-on-failure -j 1 --timeout 180
# Build on x86, transfer the cross-built executables and CTest manifest to
# the same source/build paths on Spark, then run there:
scripts/spark-cross command cmake --build --preset spark-cross --parallel 2 --target all
CUDA_DEVICE_MAX_CONNECTIONS=32 ctest --test-dir build-spark-cross --output-on-failure -j 1 --timeout 180
```

| Build | Passed | Failed | Skipped | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| Native `ci` | 97 | 2 | 10 | 602.37 s |
| Cross-built release | 97 | 2 | 10 | 602.98 s |

The two failures are the same upstream Python checks identified above.
Depending on directory enumeration order, the scripts-index assertion first
names `prefill_fairness_check.py` or `vision_prefix_cache_check.py`; both
entries are missing upstream. No C++/GPU test reported failure in either run.
The ten CTest skips require unavailable GLM/DeepSeek checkpoints, including
the GLM vision stream/frontend checks. The optional real-checkpoint subcase
inside `glm_tp_test` also remained disabled; its synthetic checks passed.
Local loopback worlds do not constitute a four-node fabric validation.

The staged cross-built server executed successfully on both Sparks and
resolved its bundled cudart/cuBLASLt libraries. A separate two-rank Qwen
NVFP4 deployment used the packaged release at C4/MTP3, a 65,536-token KV
pool and 1 GiB prefix snapshots. Those settings fit upstream's decode-row
limit; the local production C16/MTP3 configuration depends on later changes
outside this PR. Test ports, logs and staging directories were isolated.

Both-rank preflight and model startup passed. Running
`python3 scripts/serve_api_check.py HOST PORT` against the temporary service
passed all nine checks: streamed and non-streamed stops and multiple choices,
logit-bias forcing/banning/rejection, cached-token usage and reasoning-token
usage. Shutdown completed cleanly; both ranks' operation streams had MD5
`d31c16ee0484151ac304321662f169e6`.

Production was restored to its original release and deployment configuration.
The deployment file's SHA256 was unchanged, and both ranks retained the
original server SHA256
`ab1b88a00de32dedb0de63d285e584651d69c99185c106c4e6d7e549b9585fae`.
Two identical 1,137-token smoke prompts returned `READY`; the repeat reused
1,128 prompt tokens. The restored cache reported 148 slots, both ranks were
running, and service metrics reported zero failed requests and no engine
failure. The temporary test world was stopped before production restarted.

Build logs, both complete CTest logs, the API results and maintenance/restoration
records remain under `artifacts/cross-build-pr/server-validation/`. These
measurements validate this cross-build and two-node configuration; they are
not evidence for missing checkpoints or four-node hardware.

No inference performance or numerical change is claimed. The full suite
has been executed, but its two baseline failures and checkpoint skips remain
explicit limitations rather than a clean full-suite merge-gate result.

## Integration with updated upstream

Upstream `444441feed0c4d7c256ff7ff7868b587823b737f` was merged into the
published PR branch after the hardware run. Conflict resolution retains
upstream's guided setup documentation and bootstrap test registration,
alongside the cross-build instructions and `spark_cross_test`. Relative to
that upstream commit, this PR still changes no inference source or deployment
configuration.

Fresh local validation of the merged tree passed:

- Python discovery: 182 tests, 179 passed and three skipped. Updated upstream
  fixes the two earlier Python failures; their historical results above are
  retained for the previously tested source revision.
- Container CTest: all ten Python suites passed, including bootstrap and
  cross-build tests.
- Full ARM64 cross-build of all targets, including upstream's new BF12 code.
- Install staging and ELF checks: server and CUDA libraries remain AArch64,
  the server retains its install RUNPATH, and PCRE2 remains statically linked.
- `git diff --check` passed.

Logs are under `artifacts/cross-build-pr/merge-*.log`. No inference node was
contacted or restarted during this integration. The hardware results above
apply to `53a5339`; GPU/RDMA and live serving were not rerun for the merged
upstream runtime.

## Issue #21 follow-ups (2026-09-21)

The follow-up branch starts from upstream
`c5a69134410f2646d195b339c8f197043410cb04`. Both configure and build preset
lists now put `spark-cross` after `ci`. The wrapper prints `nvcc --version`
and `aarch64-linux-gnu-g++-13 --version` before configuration, so saved build
logs identify the compilers even when image package versions float. The guide
now explicitly distinguishes the default `dgpp_serve_app` build from
`build --target all`, which also builds the test executables.

Local validation:

- All 12 wrapper tests passed. The two new tests fail against the unchanged
  upstream wrapper and pass with the change: compiler versions precede
  configuration, and a failed compiler probe prevents configuration and
  compilation.
- Python discovery ran 227 tests: 222 passed and five skipped.
- Shell syntax and `git diff --check` passed.
- Real container `cmake --list-presets` puts `spark-cross` after `ci`.
- A clean server cross-build completed all 184 build steps. An incremental
  run through the actual wrapper printed both compiler versions and passed.
- Install staging passed. `file` identifies the stripped server, cudart and
  cuBLASLt as AArch64 ELF files; `readelf -d` confirms `$ORIGIN/../lib` as
  the server's RUNPATH, with no dynamic PCRE2 dependency.

The existing image is
`sha256:845fb41b6b74c3e9a161b3dea641c51968b337bc49b1b2e55ae1a3c71b29cbd2`;
it reports nvcc 13.0.88 and AArch64 G++ 13.3.0
(`13.3.0-6ubuntu2~24.04.1`). The workstation's Docker bridge was unavailable
(`adding interface ... to bridge docker0 failed: Device does not exist`).
Local container validation uses `--network host` as a workstation workaround;
an ignored local `docker` shim injects that flag when checking the actual
wrapper. The repository wrapper's networking behavior is unchanged. Logs are retained
under `artifacts/issue-21/`.

### Follow-up target validation

After the local checks, an authorized idle-hardware window validated the
cross-built server from `196d0d5c33b7` (upstream `c5a6913` plus the tooling,
tests and documentation changes above). Both ranks ran
`0.1.0+g196d0d5c33b7` with server SHA256
`0f47f6600fb857a78e212eb3399ddf9985c4d6cea705e741ee5369dec71bfe86`.
This closes the two-Spark target-validation gap identified in issue #21.

The temporary deployment used the upstream Qwen NVFP4 two-rank recipe at
C4/MTP1, with 65,536 KV tokens, 1.5 GiB prefix snapshots, FP8 dense weights,
`bf12+bf16` BF16 weights and the MMA vocabulary head. HTTP port 30002,
fabric/journal ports 29870/29871, release staging and logs were isolated from
production. Both rank logs reported configuration digest `84a458bdfefa48c0`.
Preflight verified both staged versions, checkpoints, RoCE devices and runtime
library resolution before startup.

`python3 scripts/serve_api_check.py 127.0.0.1 30002` passed all nine checks:
streamed and non-streamed stops and multiple choices, logit-bias forcing,
banning and validation, cached-token usage, and reasoning-token usage. The
test world shut down cleanly, with both operation streams reporting MD5
`0b55ebf21c5fc22f83127cf219e253f7`. Both logs were inspected; neither reported
an ERROR or FATAL. Both reported the checkpoint's missing `min_p` and
`repetition_penalty` generation defaults and their documented fallback.

Production was restored to `0.1.0+g116d3c375d87`, retaining the original C16/MTP3
configuration, 256/512 prefill budgets and 148 prefix slots. The deployment and
site-file hashes were unchanged, and the resolved configuration matched its
pre-maintenance snapshot. Both ranks again ran the original binary SHA256
`126c060ca0eff544dba2379f9a38b91c43663557e698b5803a936d4c2224e57b`;
their resolved configuration files had identical SHA256
`4acb2a06d9c37a059e4c85afbd48c43bb8950b88a3a7ec792b9dab187041c969`.
Two smoke requests returned `READY`; the second reused 808 prompt tokens.
Service metrics reported no engine failure. The maintenance locks were released
after verification.

The commands, both-rank logs, API output, digests and restoration snapshots
are under `artifacts/issue-21/server-validation/`. This run validates the
current cross-built server's two-node request path. It does not rerun the full
native or GPU/RDMA CTest suites, establish numerical equivalence or throughput,
or provide four-node coverage. The subsequent record/status update changes
only Markdown; the executed server revision remains `196d0d5c33b7`.
