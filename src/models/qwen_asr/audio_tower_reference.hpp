#pragma once
// CPU fp32 oracle for the Qwen3-ASR audio tower (docs/audio.md): the exact
// forward of transformers' Qwen3ASREncoder + multi_modal_projector at
// batch size 1 - the packing order and windowing included. The CUDA encoder
// is pinned against this (same order of accumulation wherever a kernel
// mirrors it) and the CPU path may ship it verbatim for admission-side
// parity checks.
//
// One segment in, one row block out: features are the frontend's chunk-
// aligned zero-padded [128, frames] log-mel matrix (frames a multiple of
// 100), and the returned rows are the projected embeddings for the
// audio_token_id span, in packed chunk-major order.
#include <cstdint>
#include <string>
#include <vector>

#include "models/qwen_asr/audio_config.hpp"

namespace dgpp::qwen_asr_ref {
struct AudioTowerLayerWeights {
  std::vector<float> ln1_w, ln1_b, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
  std::vector<float> ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b;
};

struct AudioTowerWeights {
  std::vector<float> conv1_w, conv1_b;  // [480,1,3,3], [480]
  std::vector<float> conv2_w, conv2_b;  // [480,480,3,3]
  std::vector<float> conv3_w, conv3_b;
  std::vector<float> conv_out_w;        // [1024, 480*16]
  std::vector<float> pos;               // [13,1024] sinusoidal, computed
  std::vector<AudioTowerLayerWeights> layers;
  std::vector<float> ln_post_w, ln_post_b, proj1_w, proj1_b, proj2_w, proj2_b;
};

// thinker.audio_tower.* (BF16 or F32) from a checkpoint directory.
AudioTowerWeights load_audio_tower(const std::string& checkpoint_dir);

// The computed sinusoidal table (the module's non-persistent buffer):
// rows x (channels/2) angles, sin | cos, max_timescale 10000.
std::vector<float> sinusoidal_positions(int rows, int channels);

// out is resized to rows x cfg.output (rows = the mask's post-conv valid
// count). feats row-major [mel_bins][frames].
void encode_audio(const QwenAsrAudioConfig& cfg, const AudioTowerWeights& w, const float* feats,
                  const int32_t* mask, int frames, std::vector<float>& out);
}  // namespace dgpp::qwen_asr_ref
