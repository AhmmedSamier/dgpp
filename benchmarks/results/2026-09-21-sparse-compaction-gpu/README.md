# Sparse compaction: review-head GPU validation

Validated clean source `ec2ac4139e31d6407b765d8ad91279d3e816e804` natively with CUDA 13 on GB10. Candidate server SHA-256 was `239149c6026c54ea4f98bd642beb83a5780a6ac30d7b7af97c0ff2e4e32e1717`, identical on both ranks. Production was stopped for the window; all GPU/RDMA tests ran serially. No implementation changes were made during this validation.

## Build and tests

The `ci` preset configured and the full native build passed (`cmake --build build-ci -j 4`). An unfiltered `ctest --test-dir build-ci --output-on-failure -j 1 --timeout 180` completed in 604.75 seconds: **113 passed, 10 skipped, zero failed**.

Skipped checkpoint-dependent cases: `tokenizer_test`, `dsv41_tokenizer_test`, `dsv41_prompt_test`, `glm4_tokenizer_test`, `glm_dsa_tokenizer_test`, `chat_template_test`, `glm4_chat_template_test`, `glm_dsa_chat_template_test`, `glm_vision_stream_test`, and `glm_vision_frontend_test`. Their required non-Qwen model files were unavailable. This does not replace the maintainer's zero-skip suite on the earlier revision.

The seven focused Qwen engine/sampler CTest lanes and six additional decode/PLE/shared-family checks also passed before the full run. The disabled-policy lane uses the explicit test-only switch; serving uses the journaled boolean.

Both tests selected by the command below passed with **zero memory errors**, including mapped commit, draft rows, chain rows and all three PLE padding paths:

```sh
DGPP_TEST_FILTER=compact_mapping compute-sanitizer --tool memcheck --error-exitcode 99 build-ci/glm_pick_test
```

A separate Qwen engine test translation unit was rebuilt from its generated compile command with `-UNDEBUG` appended after the normal flags, then linked against the rebuilt libraries. Running its `wide` filter passed both cases, including slot reuse and prefill continuation. Thus the new capture-time uniform-stride checks were enabled and exercised, rather than compiled out by the CI build's `NDEBUG`.

## Snapshot cost at sixteen physical slots

The real checkpoint has `indexer_compress_ratio=4` and `indexer_head_dim=128`, so each draft/chain ring snapshot copies **1,024 bytes per physical request**. The attached probe captures the actual `glm_device_copy` kernel sequence, comparing two and sixteen physical slots at that size. It tests one pass (draft snapshot) and three passes (draft snapshot, chain snapshot, chain restore). Each graph has 100 warmups and 1,000 measured replays, repeated ten times with alternating two/sixteen-slot order. CUDA events measure elapsed replay time, including any submission gaps. The CSV reports every repeat.

| Captured copy sequence | Two slots, median µs | Sixteen slots, median µs | Difference, µs |
|---|---:|---:|---:|
| Draft snapshot | 4.094 | 13.368 | 9.274 |
| Draft + chain snapshot/restore | 6.147 | 35.842 | 29.695 |

This isolates the extra snapshot work at C16; it is not an end-to-end C16/MTP3 engine benchmark and does not measure interactions with surrounding model kernels. The measured tens of microseconds do not currently justify complicating snapshot ownership. Keep the current implementation, with end-to-end validation when combined with the wider-decode change.

To build the probe after building `dgpp_kernels`, use the native CUDA compiler:

```sh
nvcc -std=c++20 -I src benchmarks/results/2026-09-21-sparse-compaction-gpu/snapshot_probe.cpp build-ci/libdgpp_kernels.a -o snapshot_probe
./snapshot_probe
```

## Two-node service

Started the real Qwen NVFP4 checkpoint at eight slots/MTP depth 1, FP8 dense weights, sampling cap 128, 8 GiB prefix cache, and 256/512-token prefill budgets. `engine.compact_batches` was explicitly true. The peer log confirms `compact=1` arriving from rank 0 before graph construction.

- `serve_api_check.py`: all checks passed (stops, streaming, multiple choices, bias, invalid-field rejection and cache/reasoning usage).
- Eight simultaneous sampled content requests, token limits 24–192: all completed and returned logprobs as shorter requests retired.
- Eight simultaneous distinct strict JSON schemas: every response matched its own schema/key/value.
- No ERROR/FATAL log entries. Clean shutdown; both operation streams had MD5 `bd368906a786de0ba5bb32f47c8988da`.

These are functional serving checks. They do not establish full-model teacher-forced numerical parity or four-node Qwen behavior. The feature remains default off.

## Restoration and artifacts

Production was stopped at 13:45:17 UTC and ready again at 14:04:53 UTC. Restored the original `0.1.0+g5d98ea6b7b13` release, C16/MTP3, budgeted prefill, and exact resolved configuration. Both restored binaries retained SHA-256 `3563cebf2c3821e4990d69641827d4acb4376c0c0090d090873c04f97b67967d`. Health and two repeated content requests passed, with both ranks participating.

Raw build/test logs, assertion-build commands, configurations and API responses are retained locally under `artifacts/review-gpu-20260921/`. The checked-in probe and CSV make the isolated copy measurement reproducible. Subsequent changes only add this evidence; full-model teacher-forced on/off comparison and four-node Qwen validation remain pending.
