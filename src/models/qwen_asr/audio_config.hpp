#pragma once
// The Qwen3-ASR audio tower's shape (docs/audio.md), parsed from the
// checkpoint's thinker_config.audio_config. Same policy as every config
// here: a field is either parsed into a known-supported value or refused
// with a message naming it.
//
// The tower is a Whisper-style encoder with a convolutional frontend:
//   128-bin log-mel -> three (kernel 3, stride 2, pad 1) convs in 2-D
//   (freq x time, GELU between) -> per-time-step concat over the 16
//   surviving freq bins (480 * 16 = 7680) -> d_model 1024 -> + sinusoidal
//   positions (13 rows) -> 24 x [LN, 16-head bidirectional attention
//   windowed to 13 * (n_window_infer / 2 * n_window-chunks) = 104 rows,
//   LN, fc1 -> 4096, erf-GELU, fc2, +] -> LN -> projector 1024 -> 1024 ->
//   output_dim.
// Each 100-frame (10 s) chunk yields 13 rows, so time_steps is 13 by
// construction; the conv geometry is refused, never derived.
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "common/audio_input.hpp"
#include "loaders/minijson.hpp"

namespace dgpp {
struct QwenAsrAudioConfig {
  static constexpr int kConvKernel = 3;
  static constexpr int kFreqSteps = 4;  // 128 -> 64 -> 32 -> 16 (three stride-2 convs)
  static constexpr int kTimeSteps = 13;  // ceil3(100)
  static constexpr float kLayerNormEps = 1e-5f;  // nn.LayerNorm's default

  int d_model = 1024;
  int heads = 16;
  int ffn = 4096;
  int layers = 24;
  int mel_bins = 128;
  int conv_channels = 480;
  int output = 2048;  // must equal the language model's hidden size
  int n_window = kAudioWindowHalf;
  int n_window_infer = 800;
  AudioTokens tokens;

  int head_dim() const { return d_model / heads; }
  int freq_bins() const { return mel_bins >> 3; }  // 16: three stride-2 convs
  int conv_flat() const { return conv_channels * freq_bins(); }  // 7680
  // Attention windows in packed rows: 13 rows per chunk, n_window_infer /
  // (2 * n_window) chunks per window.
  int window_rows() const { return kTimeSteps * (n_window_infer / (2 * n_window)); }

  static QwenAsrAudioConfig parse(const minijson::Value& root) {
    // Accept either the flat audio_config or the Omni-style nesting under
    // thinker_config (the ASR checkpoint ships the latter).
    const minijson::Value* node = root.find("audio_config");
    const minijson::Value* holder = &root;
    if ((!node || node->is_null()) && root.find("thinker_config")) {
      holder = root.find("thinker_config");
      node = holder->find("audio_config");
    }
    if (!node || !node->is_object())
      throw std::invalid_argument("Qwen ASR: config carries no audio_config object");
    const auto& v = *node;
    QwenAsrAudioConfig c;
    const auto field = [&](const char* key, int want) {
      if (const auto* f = v.find(key); f && !f->is_null()) {
        if (!f->is_number() || !std::isfinite(f->as_double()) ||
            f->as_double() != std::floor(f->as_double()) || static_cast<int>(f->as_int()) != want)
          throw std::invalid_argument(std::string("Qwen audio_config.") + key +
                                      ": this tower needs the value " + std::to_string(want));
      }
      return want;
    };
    c.d_model = field("d_model", 1024);
    c.heads = field("encoder_attention_heads", 16);
    c.ffn = field("encoder_ffn_dim", 4096);
    c.layers = field("encoder_layers", 24);
    c.mel_bins = field("num_mel_bins", 128);
    c.conv_channels = field("downsample_hidden_size", 480);
    c.n_window = field("n_window", kAudioWindowHalf);
    c.n_window_infer = field("n_window_infer", 800);
    field("max_position_embeddings", kTimeSteps);
    if (const auto* act = v.find("activation_function"); !act || !act->is_string() ||
        act->as_string() != "gelu")
      throw std::invalid_argument("Qwen audio_config.activation_function: erf GELU only");
    c.output = c.d_model;  // replaced below by the explicit output_dim
    if (const auto* f = v.find("output_dim"); f && f->is_number())
      c.output = static_cast<int>(f->as_int());
    const auto token_id = [&](const char* name) {
      const auto* f = holder->find(name);
      if (!f || !f->is_number() || f->as_int() < 0)
        throw std::invalid_argument(std::string("Qwen ASR: ") + name +
                                    ": must be a non-negative integer token id");
      return f->as_int();
    };
    c.tokens = AudioTokens{token_id("audio_start_token_id"), token_id("audio_token_id"),
                           token_id("audio_end_token_id")};
    if (c.tokens.start == c.tokens.pad || c.tokens.pad == c.tokens.end ||
        c.tokens.start == c.tokens.end)
      throw std::invalid_argument("Qwen ASR: the audio delimiters must be three distinct ids");
    return c;
  }

  // The language model must agree on the projected width; the caller that
  // knows the text config asserts it.
  void require_output(int text_hidden) const {
    if (output != text_hidden)
      throw std::invalid_argument("Qwen audio_config.output_dim must equal the text model's "
                                  "hidden_size");
  }

  size_t weight_bytes() const {
    return 4 * (static_cast<size_t>(mel_bins / 2) * conv_channels * 9 +  // conv1 (input 1ch)
                2 * static_cast<size_t>(conv_channels) * conv_channels * 9 + 3 * conv_channels +
                static_cast<size_t>(conv_flat()) * d_model + 13 * d_model +
                static_cast<size_t>(layers) *
                    (4 * static_cast<size_t>(d_model) * d_model + 4 * d_model + 2 * 2 * d_model +
                     2 * static_cast<size_t>(d_model) * ffn + d_model + ffn) +
                2 * d_model + static_cast<size_t>(d_model) * d_model + d_model +
                static_cast<size_t>(output) * d_model + output);
  }
};
}  // namespace dgpp
