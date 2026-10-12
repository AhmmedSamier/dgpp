// Host gate: the Qwen3-ASR audio tower. Part 1 parses the tower config
// (always runs); part 2 replays tools/qwen3_asr_tower_goldens.py's packed
// embedding rows through the fp32 CPU reference and the checkpoint's own
// BF16->F32 weights (skipped - exit 2 - when the model cache is absent).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/audio_input.hpp"
#include "common/base64.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen_asr/audio_tower_reference.hpp"

namespace {
void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
template <class F>
void rejects(F f, const char* what) {
  bool threw = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error(std::string("expected rejection: ") + what);
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

constexpr const char* kConfig = R"json(
{"thinker_config": {
   "audio_config": {"model_type": "qwen3_asr_encoder", "d_model": 1024,
     "encoder_attention_heads": 16, "encoder_ffn_dim": 4096, "encoder_layers": 24,
     "num_mel_bins": 128, "downsample_hidden_size": 480, "n_window": 50,
     "n_window_infer": 800, "max_position_embeddings": 13, "activation_function": "gelu",
     "output_dim": 2048},
   "audio_start_token_id": 151669, "audio_token_id": 151676, "audio_end_token_id": 151670}}
)json";

void config_test() {
  const auto doc = dgpp::minijson::parse(kConfig);
  const auto c = dgpp::QwenAsrAudioConfig::parse(doc.root);
  require(c.d_model == 1024 && c.output == 2048 && c.window_rows() == 104, "geometry");
  require(c.tokens.start == 151669 && c.tokens.pad == 151676 && c.tokens.end == 151670, "ids");
  require(c.conv_flat() == 7680 && c.head_dim() == 64, "derived");
  {
    auto bad = std::string(kConfig);
    bad.replace(bad.find("\"d_model\": 1024"), 15, "\"d_model\": 1280");
    const auto b = dgpp::minijson::parse(bad);
    rejects([&] { dgpp::QwenAsrAudioConfig::parse(b.root); }, "d_model");
  }
  {
    auto bad = std::string(kConfig);
    bad.replace(bad.find("audio_token_id\": 151676"), 23, "audio_token_id\": 151669");
    const auto b = dgpp::minijson::parse(bad);
    rejects([&] { dgpp::QwenAsrAudioConfig::parse(b.root); }, "duplicate delimiter");
  }
}

void golden_test(const std::string& checkpoint) {
  const auto text = read_file(std::string(DGPP_SOURCE_DIR) +
                              "/tests/data/qwen3_asr_tower_goldens.json");
  const auto doc = dgpp::minijson::parse(text);
  const auto cfg = dgpp::QwenAsrAudioConfig::parse(
      dgpp::minijson::parse(read_file(checkpoint + "/config.json")).root);
  const auto weights = dgpp::qwen_asr_ref::load_audio_tower(checkpoint);
  double worst = 0.0;
  for (const auto& clip : doc.root.at("clips").members()) {
    size_t done = 0;
    std::vector<float> mine_rows;
    for (const auto& seg : clip.value.at("segments").items()) {
      const int bins = static_cast<int>(seg.at("features_shape").items()[0].as_int());
      const int frames = static_cast<int>(seg.at("features_shape").items()[1].as_int());
      const auto feats_b = dgpp::decode_base64(std::string(seg.at("features_b64").as_string()),
                                               1 << 26);
      const auto mask_b = dgpp::decode_base64(std::string(seg.at("mask_b64").as_string()), 1 << 24);
      require(bins == dgpp::kAudioMelBins, "segment bins");
      std::vector<float> feats(feats_b.size() / 4);
      std::memcpy(feats.data(), feats_b.data(), feats_b.size());
      std::vector<int32_t> mask(mask_b.size() / 4);
      std::memcpy(mask.data(), mask_b.data(), mask_b.size());
      require(seg.at("valid_frames").as_int() ==
                  std::accumulate(mask.begin(), mask.end(), static_cast<int64_t>(0)),
              "segment mask sums");
      std::vector<float> rows;
      dgpp::qwen_asr_ref::encode_audio(cfg, weights, feats.data(), mask.data(), frames, rows);
      mine_rows.insert(mine_rows.end(), rows.begin(), rows.end());
      done += rows.size() / cfg.output;
    }
    require(done == static_cast<size_t>(clip.value.at("rows").as_int()), "clip row count");
    const auto want_b =
        dgpp::decode_base64(std::string(clip.value.at("embed_b64").as_string()), 1 << 26);
    std::vector<float> want(want_b.size() / 4);
    std::memcpy(want.data(), want_b.data(), want_b.size());
    require(want.size() == mine_rows.size(), "clip row bytes");
    double clip_worst = 0.0;
    for (size_t i = 0; i < want.size(); ++i)
      clip_worst = std::max(clip_worst, static_cast<double>(std::abs(want[i] - mine_rows[i])));
    std::printf("  %12s rows %6zu  worst |delta| %.3e\n", std::string(clip.key).c_str(), done,
                clip_worst);
    if (!(clip_worst < 5e-4)) throw std::runtime_error("tower rows drifted from the oracle");
    worst = std::max(worst, clip_worst);
  }
  std::printf("tower parity: worst |delta| = %.3e\n", worst);
}
}  // namespace

int main() {
  try {
    config_test();
    std::string error;
    const auto checkpoint = dgpp::hf::model_dir("Qwen/Qwen3-ASR-1.7B", &error);
    if (checkpoint.empty()) {
      std::printf("[SKIP] %s\n", error.c_str());
      return 2;
    }
    golden_test(checkpoint);
  } catch (const std::exception& e) {
    std::printf("[FAIL] %s\n", e.what());
    return 1;
  }
  std::printf("[ OK ] qwen_asr_tower_test\n");
  return 0;
}
