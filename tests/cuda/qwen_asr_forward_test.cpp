// CUDA + checkpoint gate for the Qwen3-ASR language half (docs/audio.md).
// Part 1 (always): the LM config parser and the expected-tensor table.
// Part 2 (skipped, exit 2, without the model cache): the resident loader
// and the row walk against tools/qwen_asr_text_goldens.py - the checkpoint's
// plain Qwen3 decoder replayed fp32 on CPU torch, a fixed 96-token prompt,
// the last row's logits.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/base64.hpp"
#include "common/dtypes.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen_asr/binding.hpp"
#include "models/qwen_asr/model.hpp"

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

constexpr const char* kLmConfig = R"json(
{"thinker_config": {
   "text_config": {"model_type": "qwen3", "hidden_size": 2048, "intermediate_size": 6144,
     "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
     "head_dim": 128, "hidden_act": "silu", "max_position_embeddings": 65536,
     "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "tie_word_embeddings": true,
     "rope_scaling": {"rope_type": "default", "interleaved": true, "mrope_section": [24, 20, 20]},
     "vocab_size": 151936, "attention_bias": false, "use_cache": true},
   "audio_config": {"d_model": 1024},
   "audio_start_token_id": 151669, "audio_token_id": 151676, "audio_end_token_id": 151670}}
)json";

void config_test() {
  const auto doc = dgpp::minijson::parse(kLmConfig);
  const auto c = dgpp::QwenAsrLmConfig::parse(doc.root);
  require(c.hidden_size == 2048 && c.num_hidden_layers == 28 && c.vocab_size == 151936, "shape");
  require(c.q_proj_width() == 2048 && c.kv_proj_width() == 1024, "projections");
  auto g = c.glm_geometry();
  require(g.head_dim == 128 && g.rotary_dim == 128 && !g.attention_bias && g.use_qk_norm &&
              g.first_k_dense_replace == 28 && !g.is_moe_layer(0) && g.mtp_layer() < 0,
          "the layer objects' geometry");
  {
    auto bad = std::string(kLmConfig);
    const std::string from = "\"head_dim\": 128", to = "\"head_dim\": 256";
    bad.replace(bad.find(from), from.size(), to);
    const auto b = dgpp::minijson::parse(bad);
    rejects([&] { dgpp::QwenAsrLmConfig::parse(b.root); }, "head_dim 256");
  }
  {
    auto bad = std::string(kLmConfig);
    const std::string from = "\"rope_type\": \"default\"";
    auto at = bad.find(from) + std::string("\"rope_type\": \"").size();
    bad.replace(at, std::string("default").size(), "yarn");
    const auto b = dgpp::minijson::parse(bad);
    rejects([&] { dgpp::QwenAsrLmConfig::parse(b.root); }, "yarn rope");
  }
  {
    auto bad = std::string(kLmConfig);
    const std::string from = "\"attention_bias\": false";
    auto at = bad.find(from) + std::string("\"attention_bias\": ").size();
    bad.replace(at, std::string("false").size(), "true");
    const auto b = dgpp::minijson::parse(bad);
    rejects([&] { dgpp::QwenAsrLmConfig::parse(b.root); }, "attention bias");
  }
}

void binding_test() {
  const auto doc = dgpp::minijson::parse(kLmConfig);
  const auto c = dgpp::QwenAsrLmConfig::parse(doc.root);
  const auto layer = dgpp::qwen_asr_expected_layer_tensors(c, 7);
  require(layer.size() == 11, "layer table width");
  auto globals = dgpp::qwen_asr_expected_global_tensors(c);
  require(globals.size() == 3, "global table width");
  require(globals[2].unused() && globals[2].name == "thinker.lm_head.weight",
          "head is a tied copy");
  // A synthetic present-map: exact, with the tower as foreign bytes.
  std::unordered_map<std::string, dgpp::QwenAsrTensorDesc> present;
  for (auto& e : globals) present[e.name] = {e.dtype, e.shape};
  for (int l = 0; l < c.num_hidden_layers; ++l)
    for (auto& e : dgpp::qwen_asr_expected_layer_tensors(c, l))
      present[e.name] = {e.dtype, e.shape};
  present["thinker.audio_tower.conv2d1.weight"] = {dgpp::DType::BF16, {480, 1, 3, 3}};
  auto ok = dgpp::qwen_asr_validate_text_binding(c, present);
  require(ok.ok() && ok.tower == 1, "clean binding validates");
  present["thinker.model.layers.3.self_attn.q_proj.weight"].dtype = dgpp::DType::F32;
  present.erase("thinker.model.layers.9.mlp.down_proj.weight");
  present["bogus.weight"] = {dgpp::DType::BF16, {1}};
  const auto bad = dgpp::qwen_asr_validate_text_binding(c, present);
  require(bad.missing == 1 && bad.dtype_mismatch == 1 && bad.unexpected == 1 && !bad.ok(),
          "binding reports faults");
}

void parity_test(const std::string& checkpoint) {
  const auto text = read_file(std::string(DGPP_SOURCE_DIR) +
                              "/tests/data/qwen_asr_text_goldens.json");
  const auto doc = dgpp::minijson::parse(text);
  std::vector<int64_t> tokens;
  for (const auto& t : doc.root.at("tokens").items()) tokens.push_back(t.as_int());
  const int T = static_cast<int>(tokens.size());
  const auto cfg = dgpp::QwenAsrConfig::from_json_file(checkpoint + "/config.json");
  const int V = cfg.lm.vocab_size;
  dgpp::QwenAsrModel model(cfg, checkpoint, 2048, 4096, 1, 0, /*serving_logits=*/false);
  auto out = model.forward(tokens);
  require(out.logits.size() >= static_cast<size_t>(T - 1) * V, "logits rows");
  const auto want_b = dgpp::decode_base64(std::string(doc.root.at("logits_b64").as_string()),
                                          1 << 26);
  std::vector<float> want(want_b.size() / 4);
  std::memcpy(want.data(), want_b.data(), want_b.size());
  require(want.size() == static_cast<size_t>(V), "golden width");
  const float* got = out.logits.data() + static_cast<size_t>(T - 1) * V;
  int argmax = 0;
  double dot = 0, na = 0, nb = 0, mx = 0;
  for (int i = 0; i < V; ++i) {
    const float a = got[i], b = want[i];
    if (a > got[argmax]) argmax = i;
    dot += double(a) * b;
    na += double(a) * a;
    nb += double(b) * b;
    mx = std::max(mx, static_cast<double>(std::fabs(a - b)));
  }
  const double rel = dot / std::sqrt(std::max(na * nb, 1e-30));
  std::printf("last row: argmax %d (want %d), cos %.9f, max |delta| %.4f\n", argmax,
              static_cast<int>(doc.root.at("argmax").as_int()), rel, mx);
  if (argmax != doc.root.at("argmax").as_int()) throw std::runtime_error("argmax moved");
  if (!(rel > 0.9999)) throw std::runtime_error("the logits' direction drifted");
  // The top-8 of the fp32 walk must stay in the engine's top 16.
  const auto& top_ids = doc.root.at("top_ids").items();
  std::vector<std::pair<float, int>> order;
  for (int i = 0; i < V; ++i) order.emplace_back(got[i], i);
  std::partial_sort(order.begin(), order.begin() + 16, order.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
  for (size_t k = 0; k < top_ids.size() && k < 8; ++k) {
    const int64_t id = top_ids[k].as_int();
    bool found = false;
    for (int j = 0; j < 16; ++j) found = found || order[static_cast<size_t>(j)].second == id;
    if (!found) throw std::runtime_error("a golden top-8 id left the engine's top 16");
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
}
}  // namespace

int main(int argc, char** argv) {
  try {
    config_test();
    binding_test();
    std::string error;
    const auto checkpoint = dgpp::hf::model_dir("Qwen/Qwen3-ASR-1.7B", &error);
    if (checkpoint.empty()) {
      std::printf("[SKIP] %s\n", error.c_str());
      return 2;
    }
    if (argc > 1 && std::strcmp(argv[1], "--config-only") == 0) {
      std::printf("[ OK ] qwen_asr config/binding only\n");
      return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
      std::printf("[SKIP] no CUDA device\n");
      return 2;
    }
    parity_test(checkpoint);
  } catch (const std::exception& e) {
    std::printf("[FAIL] %s\n", e.what());
    return 1;
  }
  std::printf("[ OK ] qwen_asr_forward_test\n");
  return 0;
}
