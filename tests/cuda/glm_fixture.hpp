#pragma once
// Shared synthetic mini-checkpoint writer for the GLM forward tests.
// Extracted from glm_forward_test.cpp (M4) so the M5 tensor-parallel parity
// test can generate its own fixture from a different config with IDENTICAL
// value machinery — the weight distributions are the parity surface, not
// the geometry. Everything is deterministic per tensor NAME (same scheme
// as the loader test), so any two fixtures with the same tensor names
// share values regardless of shape.
//
// The writer is config-agnostic: it enumerates the binding table and
// writes one safetensors shard + config.json. The JSON text is passed in
// verbatim because the on-disk config must round-trip through
// GlmTextConfig::parse exactly as the test that generated it.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "glm_rng.hpp"
#include "kernels/latent_format.hpp"
#include "loaders/minijson.hpp"
#include "models/glm/binding.hpp"
#include "models/glm/config.hpp"

namespace fs = std::filesystem;

namespace glmfx {

using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp4_e2m1_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::GlmExpectedTensor;
using dgpp::GlmTensorRole;
using dgpp::GlmTextConfig;
using dgpp::GlmWeightClass;

// RNG + per-name seeding live in glm_rng.hpp (shared with .cu tests,
// which cannot include minijson under nvcc).
using glmrng::Rng;
using glmrng::seed_for;

// Distribution per weight class: magnitudes that keep every downstream
// nonlinearity in its informative range (post-norm activations have rms 1,
// so projection sigma ~0.05 keeps dots O(1); sigmoid/softmax see their
// slopes, not their saturation plateaus).
inline float sample_value(Rng& rng, GlmWeightClass cls) {
  switch (cls) {
    case GlmWeightClass::LayerNorm:
    case GlmWeightClass::FinalNorm:
      return 0.8f + 0.4f * (0.5f * (rng.unit() + 1.0f));  // [0.8, 1.2]
    case GlmWeightClass::Kda:
      return 0.05f * rng.normal3();  // projections; f32 below
    case GlmWeightClass::Dsa:
      return 0.8f + 0.4f * (0.5f * (rng.unit() + 1.0f));  // q_aln/kv_aln
    case GlmWeightClass::DsaIndexer:
      return 0.05f * rng.normal3();
    case GlmWeightClass::Mhc:
      return 0.02f * rng.normal3();  // fn rows; base/scale below
    case GlmWeightClass::Router:
      return 0.05f * rng.normal3();
    case GlmWeightClass::Embed:
    case GlmWeightClass::LmHead:
      return 0.3f * rng.normal3();
    case GlmWeightClass::DenseMlp:
    case GlmWeightClass::SharedExpert:
    case GlmWeightClass::RoutedExpert:
      return rng.normal3();  // fp8 payload codes (scaled at dequant)
    case GlmWeightClass::Mtp:
      return 0.05f * rng.normal3();
  }
  return 0.f;
}

// The NVFP4 triple's values (docs/nvfp4_plan.md §1): every matrix gets a
// global scale g in [0.5, 4) seeded by its base name, block scales
// e4m3((0.01..0.07) x g) so the dequantized weights e2m1 x s / g land in
// the FP8 fixture's magnitude range, and e2m1 codes from a normal sample —
// two per byte, low nibble first, the format's packing.
inline float fp4_fixture_global(const std::string& base) {
  Rng g(seed_for(base) ^ 0x5bd1e995u);
  return 0.5f + 1.75f * (g.unit() + 1.0f);
}
inline bool fp4_fixture_bytes(const GlmExpectedTensor& e,
                              std::vector<uint8_t>& out) {
  std::string base;
  const char* suffix = nullptr;
  switch (e.role) {
    case GlmTensorRole::Fp4Packed: suffix = "_packed"; break;
    case GlmTensorRole::Fp4Scale: suffix = "_scale"; break;
    case GlmTensorRole::Fp4Global: suffix = "_global_scale"; break;
    default: return false;
  }
  base = e.name.substr(0, e.name.size() - std::strlen(suffix));
  const float g = fp4_fixture_global(base);
  Rng rng(seed_for(e.name));
  out.assign(e.nbytes(), 0);
  if (e.role == GlmTensorRole::Fp4Global) {
    std::memcpy(out.data(), &g, 4);
  } else if (e.role == GlmTensorRole::Fp4Scale) {
    for (size_t i = 0; i < out.size(); ++i)
      out[i] = float_to_fp8_e4m3_bits((0.01f + 0.03f * (rng.unit() + 1.0f)) * g);
  } else {
    for (size_t i = 0; i < out.size(); ++i) {
      const uint8_t lo = float_to_fp4_e2m1_bits(1.5f * rng.normal3());
      const uint8_t hi = float_to_fp4_e2m1_bits(1.5f * rng.normal3());
      out[i] = static_cast<uint8_t>(lo | (hi << 4));
    }
  }
  return true;
}

// The quantization_config a fixture's config.json carries for an NVFP4
// profile — the composed hybrid's spelling (GlmTextConfig::parse_expert_format).
inline const char* fp4_fixture_quantization_config() {
  return R"json({"quant_method":"dgpp_mixed",
    "routed_experts":{"format":"nvfp4-pack-quantized","num_bits":4,"group_size":16,
                      "scale_dtype":"torch.float8_e4m3fn","global_scale_dtype":"torch.float32"},
    "fp8":{"quant_method":"fp8","fmt":"e4m3","activation_scheme":"dynamic","weight_block_size":[128,128]},
    "mtp_layer":{"source":"base"},"dsa_attention":{"source":"base"}})json";
}

inline std::vector<uint8_t> tensor_bytes(const GlmExpectedTensor& e) {
  std::vector<uint8_t> fp4;
  if (fp4_fixture_bytes(e, fp4)) return fp4;
  Rng rng(seed_for(e.name));
  std::vector<uint8_t> out(e.nbytes());
  const size_t n = e.numel();
  const std::string name = e.name;
  const bool is_mhc_base = name.find("_hc_") != std::string::npos &&
                           name.find("_base") != std::string::npos;
  const bool is_mhc_scale = name.find("_hc_") != std::string::npos &&
                            name.find("_scale") != std::string::npos;
  const bool is_router_bias = name.find("e_score_correction_bias") !=
                              std::string::npos;
  const bool is_a_log = name.find("A_log") != std::string::npos;
  const bool is_dt_bias = name.find("dt_bias") != std::string::npos;
  const bool is_k_norm_w =
      name.find("indexer.k_norm.weight") != std::string::npos;
  const bool is_k_norm_b =
      name.find("indexer.k_norm.bias") != std::string::npos;
  const bool is_ape =
      name.find("index_kpool_compress_ape") != std::string::npos;
  const bool is_conv = name.find("conv1d") != std::string::npos;
  const bool is_scale_inv = name.find("_scale_inv") != std::string::npos;
  // Class-based sampling above is too coarse for the DSA/KDA members that
  // are not norms: kv_b/o_norm landed near-1 (or tiny) and produced a
  // 45-magnitude attention path — legal, but a badly-conditioned fixture.
  const bool is_kv_b = name.find("kv_b_proj.weight") != std::string::npos;
  const bool is_o_norm = name.find("o_norm.weight") != std::string::npos;

  for (size_t i = 0; i < n; ++i) {
    float v = sample_value(rng, e.cls);
    if (is_scale_inv) v = 0.01f + 0.06f * (0.5f * (rng.unit() + 1.0f));
    else if (is_mhc_base) v = 0.1f * rng.normal3();
    else if (is_mhc_scale) v = 0.7f + 0.6f * (0.5f * (rng.unit() + 1.0f));
    else if (is_router_bias) v = 0.05f * rng.normal3();
    else if (is_a_log) v = 0.3f * rng.normal3();
    else if (is_dt_bias) v = 0.3f * rng.normal3();
    else if (is_k_norm_w) v = 0.8f + 0.4f * (0.5f * (rng.unit() + 1.0f));
    else if (is_o_norm) v = 0.8f + 0.4f * (0.5f * (rng.unit() + 1.0f));
    else if (is_kv_b) v = 0.05f * rng.normal3();
    else if (is_k_norm_b) v = 0.05f * rng.normal3();
    else if (is_ape) v = 0.1f * rng.normal3();
    else if (is_conv) v = 0.15f * rng.normal3();

    if (e.dtype == dgpp::DType::BF16) {
      const uint16_t bits = float_to_bf16_bits(v);
      std::memcpy(&out[i * 2], &bits, 2);
    } else if (e.dtype == dgpp::DType::F32) {
      std::memcpy(&out[i * 4], &v, 4);
    } else if (e.dtype == dgpp::DType::F8_E4M3) {
      out[i] = float_to_fp8_e4m3_bits(v);
    } else {
      throw std::runtime_error("fixture dtype not handled: " + name);
    }
  }
  return out;
}

// Writes `dir` (config.json + one safetensors shard) for `cfg`. `json_text`
// is the raw text_config object (not wrapped) — it is written to disk
// wrapped in {"text_config": ...} and must parse to `cfg`.
inline void write_fixture(const GlmTextConfig& cfg, const char* json_text,
                          const std::string& dir) {
  fs::path root(dir);
  fs::remove_all(root);
  fs::create_directories(root);

  {
    fs::path p = root / "config.json";
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    std::string json = std::string("{\"text_config\":") + json_text;
    if (cfg.routed_expert_format == dgpp::GlmExpertFormat::Nvfp4Group16)
      json += std::string(",\"quantization_config\":") +
              fp4_fixture_quantization_config();
    json += "}";
    std::fwrite(json.data(), 1, json.size(), f);
    std::fclose(f);
  }

  const auto table = dgpp::glm_expected_text_tensors(cfg);
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  for (const auto& e : table) {
    auto b = tensor_bytes(e);
    std::string shape = "[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) shape += ",";
      shape += std::to_string(e.shape[i]);
    }
    shape += "]";
    if (off) header += ",";
    header += "\"" + e.name + "\":{" + "\"dtype\":\"" +
              std::string(dgpp::dtype_name(e.dtype)) +
              "\",\"shape\":" + shape + ",\"data_offsets\":[" +
              std::to_string(off) + "," + std::to_string(off + b.size()) +
              "]}";
    data.insert(data.end(), b.begin(), b.end());
    off += b.size();
  }
  header += "}";

  fs::path shard = root / "model.safetensors";
  std::FILE* f = std::fopen(shard.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write shard");
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  std::printf("fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

}  // namespace glmfx
