// Loader tests for the M4 streaming resident loader (glm_loader.cu). A
// synthetic mini-checkpoint is generated ON DISK from the expected-tensor
// table itself (config.json + one safetensors shard), so fixture and table
// can never disagree. The tests then verify, through the loader's managed
// memory (CPU-readable):
//   * KDA merged in_proj [f_a|g_a|q|k|v|b] and merged conv are byte-exact
//     concatenations of the six separate checkpoint tensors;
//   * DSA fused qkv_a, q_b, o_proj are exact block-scale dequants (bitwise
//     vs the host oracle, including the rounding of the single
//     decode-multiply then BF16 round);
//   * the APE arrives as F32 (converted from checkpoint BF16);
//   * MLP/MoE matrices stay COMPRESSED: payload bytes identical, scale
//     bytes identical, geometry recorded;
//   * globals and MTP head tensors are byte-exact;
//   * layer_bytes() equals actual bump usage for every layer (the
//     no-drift contract), and streaming frees the previous layer.
// The dequant kernel's ragged-tail handling is pinned by a direct
// [1000, 1000] test against the host oracle.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/fp8_dequant.hpp"
#include "loaders/minijson.hpp"
#include "models/glm_binding.hpp"
#include "models/glm_config.hpp"
#include "models/glm_loader.hpp"

namespace {

namespace fs = std::filesystem;
using dgpp::DType;
using dgpp::GlmExpectedTensor;
using dgpp::GlmTextConfig;
using dgpp::GlmTensorDesc;

// Same tiny config as the binding tests, with one ragged edge:
// intermediate_size 1000 exercises non-128-multiple dense-MLP shapes.
const char* kTinyJson = R"json({
  "hidden_size": 512, "vocab_size": 1000, "num_hidden_layers": 6,
  "rms_norm_eps": 1e-6, "tie_word_embeddings": false,
  "hidden_act": "silu", "swiglu_limit": 7.5,
  "layer_types": ["linear_attention", "linear_attention",
                  "deepseek_sparse_attention", "linear_attention",
                  "deepseek_sparse_attention", "linear_attention"],
  "mlp_layer_types": ["dense", "dense", "sparse", "sparse", "sparse", "sparse"],
  "indexer_types": ["full", "full", "full", "full", "full", "full"],
  "first_k_dense_replace": 2,
  "linear_attn_config": {
    "num_heads": 8, "head_dim": 64, "short_conv_kernel_size": 4,
    "gate_lower_bound": -3.5,
    "kda_layers": [0, 1, 3, 5], "full_attn_layers": [2, 4]
  },
  "num_attention_heads": 8, "q_lora_rank": 256, "kv_lora_rank": 128,
  "qk_nope_head_dim": 96, "qk_rope_head_dim": 0, "v_head_dim": 96,
  "mla_use_nope": true,
  "index_n_heads": 4, "index_head_dim": 128, "index_kpool": 4,
  "index_topk": 128, "index_kpool_compress": true,
  "index_kpool_always_select_tail": true, "indexer_rope_interleave": true,
  "intermediate_size": 1000, "moe_intermediate_size": 256,
  "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
  "scoring_func": "sigmoid", "topk_method": "noaux_tc",
  "norm_topk_prob": true, "routed_scaling_factor": 1.5,
  "n_group": 1, "topk_group": 1, "moe_router_dtype": "float32",
  "mhc": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
  "num_nextn_predict_layers": 1
})json";

GlmTextConfig tiny_config() {
  auto parsed = dgpp::minijson::parse(kTinyJson);
  return GlmTextConfig::parse(parsed.root);
}

// ---- deterministic fixture values ------------------------------------
// xorshift64 seeded per tensor name; every tensor is reproducible and
// independent of load order.
uint64_t seed_for(const std::string& name) {
  uint64_t h = 1469598103934665603ull;
  for (char c : name) {
    h ^= static_cast<uint8_t>(c);
    h *= 1099511628211ull;
  }
  return h ^ 0x9e3779b97f4a7c15ull;
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed | 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  float unit() {  // [-1, 1)
    return static_cast<float>(next() >> 11) / static_cast<float>(1ull << 52) -
           1.0f;
  }
};

std::vector<uint8_t> tensor_bytes(const GlmExpectedTensor& e) {
  Rng rng(seed_for(e.name));
  std::vector<uint8_t> out(e.nbytes());
  const size_t n = e.numel();
  const size_t ds = dgpp::dtype_size(e.dtype);
  for (size_t i = 0; i < n; ++i) {
    if (e.dtype == DType::BF16) {
      const float v = rng.unit();
      uint16_t bits = dgpp::float_to_bf16_bits(v);
      std::memcpy(&out[i * 2], &bits, 2);
    } else if (e.dtype == DType::F32) {
      const float v = 0.25f + 1.5f * (rng.unit() + 1.0f);  // [0.25, 2)
      std::memcpy(&out[i * 4], &v, 4);
    } else if (e.dtype == DType::F8_E4M3) {
      const float v = 2.0f * rng.unit();
      out[i] = dgpp::float_to_fp8_e4m3_bits(v);
    } else {
      throw std::runtime_error("fixture dtype not handled");
    }
  }
  (void)ds;
  return out;
}

// Host dequant oracle: decode-multiply, one BF16 round — the kernel's
// exact operation.
std::vector<uint16_t> dequant_oracle(const std::vector<uint8_t>& payload,
                                     const std::vector<uint8_t>& scales,
                                     int64_t rows, int64_t cols) {
  const int64_t sc = (cols + 127) / 128;
  std::vector<uint16_t> out(rows * cols);
  for (int64_t i = 0; i < rows * cols; ++i) {
    const int64_t n = i / cols, k = i % cols;
    float s;
    std::memcpy(&s, &scales[((n / 128) * sc + k / 128) * 4], 4);
    out[i] = dgpp::float_to_bf16_bits(
        dgpp::fp8_e4m3_bits_to_float(payload[i]) * s);
  }
  return out;
}

// ---- synthetic checkpoint on disk ------------------------------------
struct Fixture {
  fs::path dir;
  GlmTextConfig cfg;
  std::unordered_map<std::string, std::vector<uint8_t>> bytes;

  const std::vector<uint8_t>& at(const std::string& name) const {
    auto it = bytes.find(name);
    if (it == bytes.end())
      throw std::runtime_error("fixture missing tensor: " + name);
    return it->second;
  }
};

Fixture write_fixture() {
  Fixture fx;
  fx.dir = fs::temp_directory_path() / "dgpp_glm_loader_test";
  fs::remove_all(fx.dir);
  fs::create_directories(fx.dir);
  fx.cfg = tiny_config();

  // config.json: root object with text_config (the loader's parse shape).
  {
    fs::path p = fx.dir / "config.json";
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    const std::string json =
        std::string("{\"text_config\":") + kTinyJson + "}";
    std::fwrite(json.data(), 1, json.size(), f);
    std::fclose(f);
  }

  // All expected tensors, sequential layout, one shard.
  const auto table = dgpp::glm_expected_text_tensors(fx.cfg);
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  auto append = [&](const GlmExpectedTensor& e) {
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
    fx.bytes.emplace(e.name, std::move(b));
  };
  for (const auto& e : table) append(e);
  header += "}";

  fs::path shard = fx.dir / "model.safetensors";
  std::FILE* f = std::fopen(shard.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write shard");
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  return fx;
}

void require(bool cond, const char* what) {
  if (!cond) throw std::runtime_error(what);
}

template <typename T>
void require_bytes_eq(const T* got, const std::vector<uint8_t>& want,
                      size_t count, const char* what) {
  if (std::memcmp(got, want.data(), count * sizeof(T)) != 0) {
    std::printf("mismatch at %s: first differing offset %zu\n", what,
                [&] {
                  for (size_t i = 0; i < count; ++i)
                    if (std::memcmp(reinterpret_cast<const uint8_t*>(got) +
                                        i * sizeof(T),
                                    want.data() + i * sizeof(T), sizeof(T)))
                      return i;
                  return count;
                }());
    throw std::runtime_error(what);
  }
}

}  // namespace

DGPP_TEST(glm_loader_streams_kda_dense_layer_byte_exact) {
  const Fixture fx = write_fixture();
  dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string());
  const dgpp::GlmLayerResident& r = stream.load_layer(0);
  require(r.kind == dgpp::GlmLayerKind::Kda, "layer 0 is KDA");
  require(r.bytes == dgpp::GlmLayerStream::layer_bytes(fx.cfg, 0),
          "layer bytes match formula");

  const std::string p = "model.language_model.layers.0.self_attn.";
  // Merged in_proj rows [f_a | g_a | q | k | v | b].
  {
    const int64_t head_dim = fx.cfg.kda_head_dim;   // 64
    const int64_t proj = 8 * 64;                    // 512
    const int64_t heads = 8;
    const int64_t hidden = 512;
    const uint16_t* in = static_cast<const uint16_t*>(r.kda.in_proj);
    const struct {
      const char* name;
      int64_t rows;
    } pieces[] = {{"f_a_proj.weight", head_dim},
                  {"g_a_proj.weight", head_dim},
                  {"q_proj.weight", proj},
                  {"k_proj.weight", proj},
                  {"v_proj.weight", proj},
                  {"b_proj.weight", heads}};
    int64_t row = 0;
    for (const auto& piece : pieces) {
      require_bytes_eq(in + row * hidden, fx.at(p + piece.name),
                       static_cast<size_t>(piece.rows * hidden), piece.name);
      row += piece.rows;
    }
  }
  // Merged conv channels [q | k | v].
  {
    const int64_t proj = 512, w = 4;
    const uint16_t* conv = static_cast<const uint16_t*>(r.kda.conv);
    for (int i = 0; i < 3; ++i) {
      static const char* convs[3] = {"q_conv1d.weight", "k_conv1d.weight",
                                     "v_conv1d.weight"};
      require_bytes_eq(conv + i * proj * w, fx.at(p + convs[i]),
                       static_cast<size_t>(proj * w), convs[i]);
    }
  }
  // Direct tensors.
  require_bytes_eq(static_cast<const uint16_t*>(r.kda.f_b),
                   fx.at(p + "f_b_proj.weight"), 512 * 64, "f_b");
  require_bytes_eq(static_cast<const uint16_t*>(r.kda.g_b),
                   fx.at(p + "g_b_proj.weight"), 512 * 64, "g_b");
  require_bytes_eq(r.kda.a_log, fx.at(p + "A_log"), 8, "A_log");
  require_bytes_eq(r.kda.dt_bias, fx.at(p + "dt_bias"), 512, "dt_bias");
  require_bytes_eq(static_cast<const uint16_t*>(r.kda.o_norm),
                   fx.at(p + "o_norm.weight"), 64, "o_norm");
  require_bytes_eq(static_cast<const uint16_t*>(r.kda.o_proj),
                   fx.at(p + "o_proj.weight"), 512 * 512, "o_proj");
  // Dense MLP stays compressed: byte-identical payload and scales.
  static const char* dense[3] = {"gate_proj", "up_proj", "down_proj"};
  for (int i = 0; i < 3; ++i) {
    const std::string name =
        "model.language_model.layers.0.mlp." + std::string(dense[i]) +
        ".weight";
    require_bytes_eq(r.dense[i].payload, fx.at(name), 1000 * 512,
                    dense[i]);
    require_bytes_eq(reinterpret_cast<const uint8_t*>(r.dense[i].scales),
                     fx.at(name + "_scale_inv"), 8 * 4 * 4, "dense scale");
    require(r.dense[i].rows == (i == 2 ? 512 : 1000) &&
                r.dense[i].cols == (i == 2 ? 1000 : 512),
            "dense geometry");
  }
  // Norms + mHC.
  require_bytes_eq(r.ln1, fx.at("model.language_model.layers.0.input_layernorm.weight"),
                   512, "ln1");
  require_bytes_eq(r.ln2,
                   fx.at("model.language_model.layers.0.post_attention_layernorm.weight"),
                   512, "ln2");
  require_bytes_eq(r.mhc.attn_base,
                   fx.at("model.language_model.layers.0.hc_attn_base"), 24,
                   "hc attn base");
  require_bytes_eq(r.mhc.ffn_fn,
                   fx.at("model.language_model.layers.0.hc_ffn_fn"), 24 * 2048,
                   "hc ffn fn");
  // MoE view must be empty on a dense layer.
  require(r.moe.experts.empty() && r.moe.router_gate == nullptr,
          "dense layer has empty moe view");
}

DGPP_TEST(glm_loader_streams_dsa_moe_layer_with_dequant_bridge) {
  const Fixture fx = write_fixture();
  dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string());
  const dgpp::GlmLayerResident& r = stream.load_layer(2);
  require(r.kind == dgpp::GlmLayerKind::Dsa, "layer 2 is DSA");
  require(r.bytes == dgpp::GlmLayerStream::layer_bytes(fx.cfg, 2),
          "layer bytes match formula");

  const std::string p = "model.language_model.layers.2.self_attn.";
  const int64_t hidden = 512, q_lora = 256, kv_lora = 128;

  // Fused qkv_a: q_a rows dequantized, then kv_a rows dequantized.
  {
    const uint16_t* qkv = static_cast<const uint16_t*>(r.dsa.qkv_a);
    const std::vector<uint16_t> qa = dequant_oracle(
        fx.at(p + "q_a_proj.weight"), fx.at(p + "q_a_proj.weight_scale_inv"),
        q_lora, hidden);
    const std::vector<uint16_t> kva = dequant_oracle(
        fx.at(p + "kv_a_proj_with_mqa.weight"),
        fx.at(p + "kv_a_proj_with_mqa.weight_scale_inv"), kv_lora, hidden);
    require_bytes_eq(qkv, {reinterpret_cast<const uint8_t*>(qa.data()),
                           reinterpret_cast<const uint8_t*>(qa.data()) +
                               qa.size() * 2},
                     qa.size(), "qkv_a q_a rows");
    require_bytes_eq(qkv + q_lora * hidden,
                     {reinterpret_cast<const uint8_t*>(kva.data()),
                      reinterpret_cast<const uint8_t*>(kva.data()) +
                          kva.size() * 2},
                     kva.size(), "qkv_a kv_a rows");
  }
  // q_b / o_proj: standalone dequants, bitwise vs oracle.
  {
    const std::vector<uint16_t> qb = dequant_oracle(
        fx.at(p + "q_b_proj.weight"), fx.at(p + "q_b_proj.weight_scale_inv"),
        8 * 96, q_lora);
    require_bytes_eq(static_cast<const uint16_t*>(r.dsa.q_b),
                     {reinterpret_cast<const uint8_t*>(qb.data()),
                      reinterpret_cast<const uint8_t*>(qb.data()) +
                          qb.size() * 2},
                     qb.size(), "q_b dequant");
    const std::vector<uint16_t> op = dequant_oracle(
        fx.at(p + "o_proj.weight"), fx.at(p + "o_proj.weight_scale_inv"),
        hidden, 8 * 96);
    require_bytes_eq(static_cast<const uint16_t*>(r.dsa.o_proj),
                     {reinterpret_cast<const uint8_t*>(op.data()),
                      reinterpret_cast<const uint8_t*>(op.data()) +
                          op.size() * 2},
                     op.size(), "o_proj dequant");
  }
  // kv_b arrives BF16 byte-exact (not quantized in this checkpoint family).
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.kv_b),
                   fx.at(p + "kv_b_proj.weight"), 8 * 192 * kv_lora, "kv_b");
  // APE: BF16 checkpoint tensor converted to F32.
  {
    const auto& src = fx.at(p + "indexer.index_kpool_compress_ape");
    const uint16_t* s16 =
        reinterpret_cast<const uint16_t*>(src.data());
    const float* ape = r.dsa.ape;
    for (size_t i = 0; i < 4 * 128; ++i)
      if (ape[i] != dgpp::bf16_bits_to_float(s16[i]))
        throw std::runtime_error("ape f32 conversion");
  }
  // Indexer tensors byte-exact.
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.wq_b),
                   fx.at(p + "indexer.wq_b.weight"), 512 * q_lora, "wq_b");
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.wk),
                   fx.at(p + "indexer.wk.weight"), 128 * hidden, "wk");
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.wp),
                   fx.at(p + "indexer.weights_proj.weight"), 4 * hidden,
                   "wp");
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.gate),
                   fx.at(p + "indexer.index_kpool_compress_gate"),
                   128 * hidden, "gate");
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.k_norm_w),
                   fx.at(p + "indexer.k_norm.weight"), 128, "k_norm w");
  require_bytes_eq(static_cast<const uint16_t*>(r.dsa.k_norm_b),
                   fx.at(p + "indexer.k_norm.bias"), 128, "k_norm b");
  // Router + experts stay compressed, byte-exact.
  const std::string m = "model.language_model.layers.2.mlp.";
  require_bytes_eq(r.moe.router_gate, fx.at(m + "gate.weight"), 4 * hidden,
                   "router gate");
  require_bytes_eq(r.moe.router_bias, fx.at(m + "gate.e_score_correction_bias"),
                   4, "router bias");
  static const char* mats[3] = {"gate_proj", "up_proj", "down_proj"};
  for (int i = 0; i < 3; ++i) {
    const std::string name =
        m + "shared_experts." + std::string(mats[i]) + ".weight";
    require_bytes_eq(r.moe.shared[i].payload, fx.at(name), 256 * 512,
                    "shared payload");
    require_bytes_eq(reinterpret_cast<const uint8_t*>(r.moe.shared[i].scales),
                     fx.at(name + "_scale_inv"), 2 * 4 * 4, "shared scale");
    const std::string e2 = m + "experts.2." + std::string(mats[i]) +
                           ".weight";
    require_bytes_eq(r.moe.expert(2, i).payload, fx.at(e2), 256 * 512,
                    "expert 2 payload");
    // gate/up are [inter, hidden]; down is [hidden, inter].
    const int64_t want_rows = (i == 2) ? 512 : 256;
    const int64_t want_cols = (i == 2) ? 256 : 512;
    require(r.moe.expert(2, i).rows == want_rows &&
                r.moe.expert(2, i).cols == want_cols,
            "expert geometry");
  }
  require(r.moe.experts.size() == 4 * 3, "expert count");
  // Dense view empty on a MoE layer.
  require(r.dense[0].payload == nullptr, "moe layer has empty dense view");
}

DGPP_TEST(glm_loader_streams_mtp_layer_and_globals) {
  const Fixture fx = write_fixture();
  dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string());
  const dgpp::GlmLayerResident& r = stream.load_layer(6);  // MTP draft
  require(r.kind == dgpp::GlmLayerKind::Dsa, "mtp is DSA");
  require(r.mhc.attn_base == nullptr, "mtp has no mhc");
  require(r.bytes == dgpp::GlmLayerStream::layer_bytes(fx.cfg, 6),
          "mtp bytes match formula");
  const std::string p = "model.language_model.layers.6.";
  require_bytes_eq(r.enorm, fx.at(p + "enorm.weight"), 512, "enorm");
  require_bytes_eq(r.eh_proj, fx.at(p + "eh_proj.weight"), 512 * 1024,
                   "eh_proj");
  require_bytes_eq(r.shared_head_norm, fx.at(p + "shared_head.norm.weight"),
                   512, "shared head norm");
  require(r.moe.experts.size() == 12, "mtp moe present");

  const dgpp::GlmGlobalsResident& g = stream.load_globals();
  require_bytes_eq(g.embed, fx.at("model.language_model.embed_tokens.weight"),
                   1000 * 512, "embed");
  require_bytes_eq(g.lm_head, fx.at("lm_head.weight"), 1000 * 512, "lm_head");
  require_bytes_eq(g.final_norm, fx.at("model.language_model.norm.weight"),
                   512, "final norm");
  require(g.bytes == dgpp::GlmLayerStream::globals_bytes(fx.cfg),
          "globals bytes match formula");
}

DGPP_TEST(glm_loader_streaming_replaces_layers_and_counts_every_kind) {
  const Fixture fx = write_fixture();
  dgpp::GlmLayerStream stream(fx.cfg, fx.dir.string());
  // Every layer (0..6): formula == usage enforced inside load_layer; a
  // throw here IS the failure. Streaming replacement: 0 -> 2 -> 6 -> 1.
  const int seq[] = {0, 2, 6, 1, 3, 4, 5, 0};
  for (int layer : seq) {
    const dgpp::GlmLayerResident& r = stream.load_layer(layer);
    require(r.layer == layer, "resident layer id");
  }
  // Layer capacity = max over layers.
  size_t maxb = 0;
  for (int i = 0; i <= 6; ++i)
    maxb = std::max(maxb, dgpp::GlmLayerStream::layer_bytes(fx.cfg, i));
  require(stream.layer_capacity() == maxb, "capacity is max layer");
  // MTP-only config: no layer 6 when num_nextn_predict_layers == 0.
  bool threw = false;
  try {
    (void)stream.load_layer(7);
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "layer out of range rejected");
}

DGPP_TEST(fp8_dequant_blocks_handles_ragged_tails_bitwise) {
  // [1000, 1000]: 7 full row/col blocks + 104-wide ragged tails on both
  // axes — the geometry the real checkpoint never produces but the kernel
  // must still get right.
  const int64_t rows = 1000, cols = 1000;
  Rng rng(0xC0FFEE);
  std::vector<uint8_t> payload(rows * cols);
  for (auto& b : payload) {
    const float v = 2.0f * rng.unit();
    b = dgpp::float_to_fp8_e4m3_bits(v);
  }
  const int64_t sr = (rows + 127) / 128, sc = (cols + 127) / 128;
  std::vector<uint8_t> scales(sr * sc * 4);
  for (auto& b : scales) b = 0;
  {
    std::vector<float> s(sr * sc);
    for (auto& v : s) v = 0.25f + 1.5f * (rng.unit() + 1.0f);
    std::memcpy(scales.data(), s.data(), scales.size());
  }
  const std::vector<uint16_t> oracle =
      dequant_oracle(payload, scales, rows, cols);

  uint8_t* dev_p = nullptr;
  float* dev_s = nullptr;
  uint16_t* dev_o = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&dev_p, payload.size()));
  DGPP_CUDA_OK(cudaMallocManaged(&dev_s, scales.size()));
  DGPP_CUDA_OK(
      cudaMallocManaged(&dev_o, static_cast<size_t>(rows * cols) * 2));
  std::memcpy(dev_p, payload.data(), payload.size());
  std::memcpy(dev_s, scales.data(), scales.size());
  dgpp::launch_fp8_dequant_blocks(dev_p, dev_s, dev_o, rows, cols, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  require_bytes_eq(dev_o,
                   {reinterpret_cast<const uint8_t*>(oracle.data()),
                    reinterpret_cast<const uint8_t*>(oracle.data()) +
                        oracle.size() * 2},
                   oracle.size(), "ragged dequant");
  DGPP_CUDA_OK(cudaFree(dev_p));
  DGPP_CUDA_OK(cudaFree(dev_s));
  DGPP_CUDA_OK(cudaFree(dev_o));
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
