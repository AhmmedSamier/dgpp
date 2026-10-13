#pragma once
// The Qwen3-ASR audio tower on the device (docs/audio.md): the BF16 forward
// of transformers' Qwen3ASREncoder + multi_modal_projector at batch size 1,
// mirroring models/qwen/vision.hpp's structure (device BF16 weights loaded
// by checkpoint key, preallocated workspace, one encode call per segment).
//
//   log-mel [128, frames] -> three stride-2 convs (GELU exact between) ->
//   conv_out 7680 -> 1024 + 13-row sinusoidal positions per 100-frame chunk
//   -> packed valid rows (chunk-major) -> 24 x [LN, 16-head bidirectional
//   attention windowed to 104 packed rows, LN, fc1 -> 4096, erf GELU,
//   fc2, +] -> LN -> projector 1024 -> 1024 -> output_dim.
//
// The fp32 host oracle (models/qwen_asr/audio_tower_reference.hpp) is the
// parity gate; this class is the speed-up behind the same numbers. Linears
// run CublasLtGemm::matmul_linear_bf16 (fp32 split-K, one bf16 rounding),
// LayerNorm/GELU/residual/softmax reuse the vision kernels.
#include <cstdint>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "models/qwen_asr/audio_config.hpp"

namespace dgpp {
class QwenAsrTowerEncoder {
 public:
  QwenAsrTowerEncoder(const QwenAsrAudioConfig& config, const std::string& checkpoint,
                      cudaStream_t stream);
  ~QwenAsrTowerEncoder();
  QwenAsrTowerEncoder(const QwenAsrTowerEncoder&) = delete;
  QwenAsrTowerEncoder& operator=(const QwenAsrTowerEncoder&) = delete;

  // One encoder segment: feats [128, frames] fp32 host, mask [frames] int32
  // host (1 for valid mel frames), frames a positive multiple of 100. Writes
  // the projected rows (rows x cfg.output bf16) to dst (device) and returns
  // the row count (the mask's post-conv valid count).
  int encode_piece(const float* feats, const int32_t* mask, int frames, uint16_t* dst);

  static size_t weight_bytes(const QwenAsrAudioConfig& cfg);
  static size_t workspace_bytes(const QwenAsrAudioConfig& cfg);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dgpp
