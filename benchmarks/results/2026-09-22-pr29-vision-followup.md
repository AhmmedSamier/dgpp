# PR #29 Qwen vision follow-up — 2026-09-22

Reviewed `5db266e55b9e52917d99fe1f94d5c0825f1785e5` from
[PR #29](https://github.com/HawkBearPig/dgpp/pull/29).

## Fixes

- Allocate the FP32 attention tile for all four patches per visual token,
  and use the same calculation in the memory plan. At the 1,024-token ceiling
  the tile requires 32 MiB, rather than the original 8 MiB.
- Synchronize shared softmax reductions before reusing their scratch space.
- Allocate the thin-image test's full 16×1,024 RGB input.
- Move the oracle's native output to the selected device and construct RoPE
  frequencies without mixing NumPy arrays and Torch tensors.
- Align projection bias rounding, FP32 rotary arithmetic, LayerNorm's Welford
  reduction and GELU arithmetic with the CUDA reference. Use exact GELU in the
  merger and tanh GELU in blocks. Add operation traces and CUDA regression tests.

The oracle preserves the native encoder's FP32 QK-score/scaling contract,
with BF16 projections and probabilities. It is not a claim of bitwise parity
with Transformers attention backends that round scores to BF16. No comparison
tolerance was relaxed. The documented single-frame and language-model mRoPE
limitations remain; no full serving quality or performance claim is made here.

## Environment

- NVIDIA GB10, CUDA 13, Release build targeting `sm_121a`.
- PyTorch `2.14.0+cu130`, NumPy; TF32 and reduced-precision BF16 reductions off.
- Checkpoint: `Qwen/Qwen3.8-Flash-Next-FP8`, snapshot
  `236dfdf285828023ca3bcd3f37366c58a3469b13`.
- An existing serving process stayed running and was not reconfigured.

## Results

All final BF16 encoder outputs matched the CUDA oracle bit for bit:

| Canvas | Visual tokens | Maximum absolute error | Relative RMS error |
| --- | ---: | ---: | ---: |
| 64×64 | 4 | 0 | 0 |
| 256×256 | 64 | 0 | 0 |
| 512×1,024 | 512 | 0 | 0 |
| 1,024×1,024 | 1,024 | 0 | 0 |

The maximum-canvas native output digest was `7544296647852347379`.

- Six focused CTest targets passed: `unit_tests`, `serve_test`,
  `fabric_serve_test`, `scheduler_test`, `glm_vision_frontend_test`, and
  `qwen_vision_test`.
- AddressSanitizer: all 12 image-input tests passed.
- Compute Sanitizer memcheck: maximum-canvas encoder, **0 errors**.
- Compute Sanitizer racecheck: CUDA LayerNorm/GELU and softmax tests,
  **0 hazards**, including both softmax paths and 4,096 patches per row.
- Full serving executable and encoder harness built successfully.
- Python compilation and `git diff --check` passed.

## Reproduction

Set `CKPT` to the snapshot directory and use a CUDA-enabled Python environment:

```bash
cmake --build build-ci -j 4 --target dgpp_serve_app qwen_vision_check qwen_vision_test
ctest --test-dir build-ci --output-on-failure \
  -R '^(unit_tests|serve_test|fabric_serve_test|scheduler_test|glm_vision_frontend_test|qwen_vision_test)$'
build-ci/qwen_vision_check "$CKPT" 1024 1024 /tmp/qwen-vision.bf16
python3 tools/qwen_vision_reference.py "$CKPT" 1024 1024 /tmp/qwen-vision.bf16
compute-sanitizer --tool memcheck --error-exitcode 42 \
  build-ci/qwen_vision_check "$CKPT" 1024 1024 /tmp/qwen-vision.bf16
compute-sanitizer --tool racecheck --error-exitcode 42 build-ci/qwen_vision_test
```
