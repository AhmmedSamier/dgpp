# Sparse compaction: native and two-node validation

## Revision and scope

Validated the uncommitted extraction on `codex/upstream-sparse-compaction`, based on upstream `444441feed0c4d7c256ff7ff7868b587823b737f`. No implementation changes were made during this validation window. The source snapshot was checked against a SHA-256 manifest after copying to rank 0's node.

- Source patch SHA-256: `24a93c0d4a3f794b2fd40c1400410796a48e51addf96df59eb07909a6b906eb1`.
- Source manifest SHA-256: `2314b1805ab021cb6fe80507220c25983e432dbb959396936db547812c4dada2`.
- Server version: `0.1.0+g444441feed0c.dirty`, CUDA 13.0.
- Server SHA-256, identical on both ranks: `71839f45b3c7abb49aa5000bc1370c21bcbce85a814bf11c12ba223ad252f4d1`.

Native builds used the `ci` preset on a GB10 Spark, with two build jobs. GPU/RDMA tests ran serially after stopping production on both nodes. This is two-node RoCE evidence, not the contribution guide's four-node gate. The worktree was not committed or published.

## Focused checks

All selected binaries were rebuilt before testing:

```sh
cmake --preset ci
cmake --build build-ci -j 2 --target qwen_engine_test glm_pick_test qwen_decode_test dgpp_serve_app
cmake --build build-ci -j 2 --target qwen_forward_test qwen_ple_test
ctest --test-dir build-ci -R '^(glm_pick_test|qwen_ple_test|qwen_forward_fixture|qwen_decode_test|qwen_engine_test|qwen_engine_row_independent|qwen_engine_bf12|qwen_engine_compact_sampling_1|qwen_engine_compact_sampling_32)$' --output-on-failure -j 1 --timeout 240
DGPP_COMPACT_BATCH=0 ctest --test-dir build-ci -R '^qwen_engine_(test|row_independent|bf12)$' --output-on-failure -j 1 --timeout 240
cmake --build build-ci -j 2 --target glm4_engine_test glm_dsa_engine_test dsv41_engine_test
ctest --test-dir build-ci -R '^(glm4_engine_test|glm_dsa_engine_test|dsv41_engine_test)$' --output-on-failure -j 1 --timeout 240
DGPP_TEST_FILTER=compact_mapping compute-sanitizer --tool memcheck --error-exitcode 99 build-ci/glm_pick_test
```

| Check | Result |
|---|---|
| Focused Qwen, PLE and shared sampler lanes | 9/9 passed |
| Compaction disabled: production, row-independent and BF12 lanes | 3/3 passed |
| GLM-DSA, GLM-4 and DeepSeek engine regressions | 3/3 passed |
| Mapped sampler memcheck | 1 test passed; zero errors |

The Qwen decode fixture's wide teacher-forced check covered 224 rows: worst relative L2 `0.009175`, two near ties, mean NLL delta `0.000105993` nat against its `0.02` nat limit. This is the fixture's batched-versus-reference numerical check, not a full-model compaction-on/off comparison. The row-independent and sampled engine lanes check transcript/state isolation under remapping, sparse occupancy, reuse and prefill continuation.

## Real-model service

Started the same candidate binary on both nodes with `nvidia/Qwen3.8-Flash-Next-NVFP4`, eight request slots, MTP depth 1, 16 decode rows, FP8 dense weights, sampling candidate cap 128, and an 8 GiB prefix cache. This retains upstream's decode limit; it does not include the separate C16/MTP3 change.

The launcher initially rejected a resolved configuration containing site fields. The test setup was corrected to use a deployment JSON plus a site environment file before any candidate process started.

- `serve_api_check.py` passed: one-shot/streamed stops, multiple choices, forced and banned token bias, invalid-bias rejection, prefix-cache usage, and reasoning-token accounting.
- Eight concurrent sampled requests with token limits 24 through 192 all completed. Shorter requests retired while longer requests remained active.
- A second eight-request run disabled thinking and checked that every response returned content logprobs.
- Eight concurrent requests with distinct strict JSON schemas each returned its own required key/value, exercising per-request grammar masks after slot reuse.
- Both ranks shut down cleanly. Operation-stream MD5 was `0f82462eb153640fd611a02f880bcd87` on both ranks.
- Rank logs contained no ERROR or FATAL entries. Warnings described neutral generation-config defaults and rank 0 normalizing the batch-minimum setting from 0 to 2 on the peer.

These are functional checks, not performance measurements or a demonstration of full-model transcript equivalence. The live sampled runs logged zero sampled fallbacks; they do not establish coverage of that path.

## Production restoration

Production was stopped at 01:37:58 UTC and was ready again by 01:46:43 UTC. Restored release `0.1.0+g42b105d-mtpmetrics`, C16/MTP3, 8 GiB prefix cache, original HTTP port, and the exact original resolved configuration. Both production binaries retained SHA-256 `ab1b88a00de32dedb0de63d285e584651d69c99185c106c4e6d7e549b9585fae`. Health and a real content request passed; both rank logs showed the request completing.

Local raw logs, source patch/manifest, deployment files, probe scripts and responses are retained under `artifacts/compaction-20260920/` in the preparation worktree.

## Remaining gates

- Full-model teacher-forced compaction-on/off comparisons, with controlled graph shapes and the upstream numerical criteria.
- Explicit sampled-fallback coverage and broader adversarial mapping/state checks where existing tests do not exercise the path.
- Formatting touched code; the workstation/native environment did not provide clang-format.
- Full build and unfiltered suite, with checkpoint skips reported explicitly.
- Four-node fabric validation. Two available Sparks cannot satisfy this gate.

No PR was opened during this validation window.
