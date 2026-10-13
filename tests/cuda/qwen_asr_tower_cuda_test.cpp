// CUDA gate: the Qwen3-ASR device tower (docs/audio.md). Replays
// tools/qwen3_asr_tower_goldens.py's packed embedding rows through
// QwenAsrTowerEncoder (BF16 on the device) against the fp32 torch oracle.
// The device path rounds once per stage boundary where the oracle does
// not, so the gate is 5e-2 absolute (the oracle's own gate is 5e-4);
// transcription equality on a real clip is the semantic backstop.
// Skipped (exit 2) without the model cache or a CUDA device.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/audio_input.hpp"
#include "common/base64.hpp"
#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen_asr/audio_config.hpp"
#include "models/qwen_asr/tower.hpp"

namespace {
void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

void golden_test(const std::string& checkpoint) {
  DGPP_CUDA_OK(cudaSetDevice(0));
  cudaStream_t stream = nullptr;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  const auto text = read_file(std::string(DGPP_SOURCE_DIR) +
                              "/tests/data/qwen3_asr_tower_goldens.json");
  const auto doc = dgpp::minijson::parse(text);
  const auto cfg = dgpp::QwenAsrAudioConfig::parse(
      dgpp::minijson::parse(read_file(checkpoint + "/config.json")).root);
  dgpp::QwenAsrTowerEncoder tower(cfg, checkpoint, stream);
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
      uint16_t* dst = nullptr;
      DGPP_CUDA_OK(cudaMalloc(&dst, static_cast<size_t>(frames) * cfg.output * 2));
      const int rows = tower.encode_piece(feats.data(), mask.data(), frames, dst);
      std::vector<uint16_t> bits(static_cast<size_t>(rows) * cfg.output);
      DGPP_CUDA_OK(cudaMemcpy(bits.data(), dst, bits.size() * 2, cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaFree(dst));
      for (const auto b : bits) mine_rows.push_back(dgpp::bf16_bits_to_float(b));
      done += static_cast<size_t>(rows);
    }
    require(done == static_cast<size_t>(clip.value.at("rows").as_int()), "clip row count");
    const auto want_b =
        dgpp::decode_base64(std::string(clip.value.at("embed_b64").as_string()), 1 << 26);
    std::vector<float> want(want_b.size() / 4);
    std::memcpy(want.data(), want_b.data(), want_b.size());
    require(want.size() == mine_rows.size(), "clip row bytes");
    double clip_worst = 0.0;
    double clip_sum = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
      const double d = static_cast<double>(std::abs(want[i] - mine_rows[i]));
      clip_worst = std::max(clip_worst, d);
      clip_sum += d;
    }
    std::printf("  %12s rows %6zu  worst |delta| %.3e  mean %.3e\n", std::string(clip.key).c_str(),
                done, clip_worst, clip_sum / want.size());
    if (!(clip_worst < 5e-2)) throw std::runtime_error("device tower rows drifted from the oracle");
    worst = std::max(worst, clip_worst);
  }
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
  std::printf("device tower parity: worst |delta| = %.3e\n", worst);
}
}  // namespace

int main() {
  try {
    std::string error;
    const auto checkpoint = dgpp::hf::model_dir("Qwen/Qwen3-ASR-1.7B", &error);
    if (checkpoint.empty()) {
      std::printf("[SKIP] %s\n", error.c_str());
      return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
      std::printf("[SKIP] no CUDA device\n");
      return 2;
    }
    golden_test(checkpoint);
  } catch (const std::exception& e) {
    std::printf("[FAIL] %s\n", e.what());
    return 1;
  }
  std::printf("[ OK ] qwen_asr_tower_cuda_test\n");
  return 0;
}
