// Host-only tests for the GLM expected-tensor table and checkpoint binding
// validator (M4): the table reproduces the exact names/dtypes/shapes of the
// GLM-5.3-Flash layout for every weight class, scale pairing is enforced in
// both directions, and every corruption class is caught with the right
// counter.
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "models/glm/binding.hpp"
#include "models/glm/config.hpp"

namespace {

using dgpp::DType;
using dgpp::GlmExpectedTensor;
using dgpp::GlmTextConfig;
using dgpp::GlmTensorDesc;
using dgpp::GlmWeightClass;

// Same tiny fixture as glm_config_test: 6 layers (DSA at 2, 4), dense MLP on
// 0-1, 4 routed experts, MTP present at layer 6.
GlmTextConfig tiny_config() {
  static const char* kJson = R"json({
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
    "intermediate_size": 1024, "moe_intermediate_size": 256,
    "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
    "scoring_func": "sigmoid", "topk_method": "noaux_tc",
    "norm_topk_prob": true, "routed_scaling_factor": 1.5,
    "n_group": 1, "topk_group": 1, "moe_router_dtype": "float32",
    "mhc": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
    "num_nextn_predict_layers": 1
  })json";
  auto parsed = dgpp::minijson::parse(kJson);
  return GlmTextConfig::parse(parsed.root);
}

std::unordered_map<std::string, GlmTensorDesc> table_as_present(
    const std::vector<GlmExpectedTensor>& table) {
  std::unordered_map<std::string, GlmTensorDesc> present;
  for (const auto& e : table)
    present.emplace(e.name, GlmTensorDesc{e.dtype, e.shape});
  return present;
}

void require(bool cond, const char* what) {
  if (!cond) throw std::runtime_error(what);
}

const GlmExpectedTensor* find(const std::vector<GlmExpectedTensor>& t,
                              const std::string& name) {
  for (const auto& e : t)
    if (e.name == name) return &e;
  return nullptr;
}

}  // namespace

DGPP_TEST(glm_binding_table_reproduces_kda_dense_layer) {
  const auto table =
      dgpp::glm_expected_text_tensors(tiny_config());
  // Layer 0: KDA + dense MLP. Representative names, shapes, dtypes.
  const std::string p = "model.language_model.layers.0.self_attn.";
  require(find(table, p + "q_proj.weight") != nullptr, "q_proj present");
  const auto* q = find(table, p + "q_proj.weight");
  require(q->dtype == DType::BF16, "q_proj bf16");
  require(q->shape == (std::vector<int64_t>{8 * 64, 512}), "q_proj shape");
  require(q->cls == GlmWeightClass::Kda, "q_proj class");
  require(find(table, p + "A_log") != nullptr, "A_log present");
  require(find(table, p + "dt_bias")->shape ==
              (std::vector<int64_t>{8 * 64}),
          "dt_bias shape");
  require(find(table, p + "q_conv1d.weight")->shape ==
              (std::vector<int64_t>{8 * 64, 1, 4}),
          "conv1d shape");
  require(find(table, p + "b_proj.weight")->shape ==
              (std::vector<int64_t>{8, 512}),
          "b_proj shape");
  // Dense MLP is quantized with scale partners.
  const auto* gate =
      find(table, "model.language_model.layers.0.mlp.gate_proj.weight");
  require(gate && gate->dtype == DType::F8_E4M3, "dense gate fp8");
  require(gate->shape == (std::vector<int64_t>{1024, 512}), "dense gate shape");
  const auto* gate_scale = find(
      table, "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv");
  require(gate_scale && gate_scale->dtype == DType::F32, "gate scale f32");
  require(gate_scale->shape == (std::vector<int64_t>{1024 / 128, 512 / 128}),
          "gate scale shape");
}

DGPP_TEST(glm_binding_table_reproduces_dsa_moe_layer) {
  const auto table = dgpp::glm_expected_text_tensors(tiny_config());
  const std::string p = "model.language_model.layers.2.self_attn.";
  // MLA core with the real checkpoint's odd sizes: 8 heads x 96 nope.
  require(find(table, p + "q_b_proj.weight")->shape ==
              (std::vector<int64_t>{8 * 96, 256}),
          "q_b shape");
  require(find(table, p + "kv_b_proj.weight")->shape ==
              (std::vector<int64_t>{8 * (96 + 96), 128}),
          "kv_b shape");
  require(find(table, p + "kv_b_proj.weight")->dtype == DType::BF16,
          "kv_b stays bf16");
  require(find(table, p + "o_proj.weight")->shape ==
              (std::vector<int64_t>{512, 8 * 96}),
          "o_proj shape");
  // Non-multiple-of-128 payload: scale grid rounds up.
  const auto* qb_scale =
      find(table, p + "q_b_proj.weight_scale_inv");
  require(qb_scale && qb_scale->shape == (std::vector<int64_t>{6, 2}),
          "q_b scale grid rounds up");
  // Indexer, including the compress pair.
  require(find(table, p + "indexer.wq_b.weight")->shape ==
              (std::vector<int64_t>{4 * 128, 256}),
          "indexer wq_b shape");
  require(find(table, p + "indexer.index_kpool_compress_ape")->shape ==
              (std::vector<int64_t>{4, 128}),
          "compress ape shape");
  // Router + shared + routed experts.
  const std::string m = "model.language_model.layers.2.mlp.";
  require(find(table, m + "gate.weight")->shape ==
              (std::vector<int64_t>{4, 512}),
          "router gate shape");
  require(find(table, m + "gate.e_score_correction_bias")->dtype == DType::F32,
          "router bias f32");
  require(find(table, m + "shared_experts.gate_proj.weight") != nullptr,
          "shared expert un-indexed");
  const auto* e2 = find(table, m + "experts.2.down_proj.weight");
  require(e2 && e2->cls == GlmWeightClass::RoutedExpert, "expert class");
  require(e2->expert == 2, "expert id");
  require(e2->shape == (std::vector<int64_t>{512, 256}), "expert down shape");
  // mHC present on main layers.
  require(find(table, "model.language_model.layers.2.hc_attn_fn")->shape ==
              (std::vector<int64_t>{24, 4 * 512}),
          "hc fn shape");
}

DGPP_TEST(glm_binding_table_counts_every_class) {
  const auto table = dgpp::glm_expected_text_tensors(tiny_config());
  // Per-layer block sizes for the tiny fixture:
  const size_t mhc = 6;   // base, fn, scale x attn/ffn
  const size_t norms = 2;
  const size_t kda = 15;  // q,k,v,f_a,g_a,f_b,g_b,b (8) + A_log,dt_bias (2)
                          // + 3 conv1d + o_norm + o_proj (5)
  const size_t dsa = 18;  // q_a(+scale), q_a_ln, q_b(+scale), kv_a(+scale),
                          // kv_a_ln, kv_b, o_proj(+scale) (11) + indexer: wq_b,
                          // wk, weights_proj, k_norm.w/b, compress gate/ape (7)
  const size_t dense = 6;    // gate/up/down x (payload+scale)
  const size_t moe = 2 + 6 + 4 * 6;  // router + shared + 4 experts x 6
  const size_t mtp_head = 4;         // enorm, hnorm, eh_proj, shared_head.norm

  // Layers 0,1: KDA+dense; 2,4: DSA+MoE; 3,5: KDA+MoE; then MTP + globals.
  const size_t expect = 2 * (mhc + norms + kda + dense) +
                        2 * (mhc + norms + dsa + moe) +
                        2 * (mhc + norms + kda + moe) +
                        (dsa + moe + norms + mtp_head) + 3;
  require(table.size() == expect, "table total");
}

DGPP_TEST(glm_binding_validates_clean_table_and_catches_corruption) {
  const GlmTextConfig cfg = tiny_config();
  const auto table = dgpp::glm_expected_text_tensors(cfg);
  auto present = table_as_present(table);

  // Clean table binds exactly.
  auto rep = dgpp::glm_validate_text_binding(cfg, present, 8);
  require(rep.ok(), "clean table validates");
  require(rep.matched == rep.expected, "all matched");
  require(rep.quantized_matrices == rep.scales_bound, "all scales bound");

  // Vision tensors are tolerated (counted, not validated).
  present.emplace("model.visual.blocks.0.attn.qkv.weight",
                  GlmTensorDesc{DType::BF16, {3072, 1024}});
  rep = dgpp::glm_validate_text_binding(cfg, present, 8);
  require(rep.ok() && rep.vision == 1, "vision tolerated");
  present.erase("model.visual.blocks.0.attn.qkv.weight");

  // Missing payload tensor.
  auto missing = present;
  missing.erase("model.language_model.layers.0.mlp.gate_proj.weight");
  rep = dgpp::glm_validate_text_binding(cfg, missing, 8);
  require(!rep.ok() && rep.missing == 1, "missing caught");

  // Missing scale partner.
  auto no_scale = present;
  no_scale.erase("model.language_model.layers.0.mlp.gate_proj.weight_scale_inv");
  rep = dgpp::glm_validate_text_binding(cfg, no_scale, 8);
  require(!rep.ok() && rep.scales_bad == 1, "missing scale caught");

  // Mis-shaped scale (wrong block grid silently dequantizes wrong blocks).
  auto bad_grid = present;
  bad_grid["model.language_model.layers.0.mlp.gate_proj.weight_scale_inv"] =
      GlmTensorDesc{DType::F32, {7, 4}};
  rep = dgpp::glm_validate_text_binding(cfg, bad_grid, 8);
  require(!rep.ok() && rep.scales_bad == 1, "bad scale grid caught");

  // Wrong scale dtype.
  auto bad_dt = present;
  bad_dt["model.language_model.layers.0.mlp.gate_proj.weight_scale_inv"] =
      GlmTensorDesc{DType::BF16, {8, 4}};
  rep = dgpp::glm_validate_text_binding(cfg, bad_dt, 8);
  require(!rep.ok() && rep.scales_bad == 1, "bad scale dtype caught");

  // Payload dtype mismatch.
  auto bad_payload = present;
  bad_payload["model.language_model.layers.0.mlp.gate_proj.weight"] =
      GlmTensorDesc{DType::BF16, {1024, 512}};
  rep = dgpp::glm_validate_text_binding(cfg, bad_payload, 8);
  require(!rep.ok() && rep.dtype_mismatch == 1, "dtype mismatch caught");

  // Shape mismatch.
  auto bad_shape = present;
  bad_shape["model.language_model.layers.0.self_attn.q_proj.weight"] =
      GlmTensorDesc{DType::BF16, {8 * 64, 513}};
  rep = dgpp::glm_validate_text_binding(cfg, bad_shape, 8);
  require(!rep.ok() && rep.shape_mismatch == 1, "shape mismatch caught");

  // Scale of a removed payload: the scale is itself an expected tensor, so it
  // matches independently — the report still fails on the missing payload.
  auto orphan = missing;
  rep = dgpp::glm_validate_text_binding(cfg, orphan, 8);
  require(!rep.ok() && rep.missing == 1 && rep.unexpected == 0,
          "orphan scale matched, payload missing fails");

  // Fully unexpected tensor.
  auto extra = present;
  extra.emplace("model.language_model.layers.0.mlp.magic_proj.weight",
                GlmTensorDesc{DType::F32, {}});
  rep = dgpp::glm_validate_text_binding(cfg, extra, 8);
  require(!rep.ok() && rep.unexpected == 1, "unexpected caught");

  // Error capping keeps the report bounded on garbage checkpoints.
  auto garbage = present;
  for (int i = 0; i < 50; ++i)
    garbage.emplace("model.language_model.layers.0.junk." + std::to_string(i),
                    GlmTensorDesc{DType::F32, {}});
  rep = dgpp::glm_validate_text_binding(cfg, garbage, 8);
  require(rep.errors.size() == 8 && rep.unexpected == 50,
          "error list capped");
}

DGPP_TEST(glm_binding_scale_shape_rounds_up_blocks) {
  using dgpp::glm_scale_shape;
  require(glm_scale_shape({256, 512}) ==
              (std::vector<int64_t>{2, 4}),
          "exact blocks");
  require(glm_scale_shape({257, 512}) ==
              (std::vector<int64_t>{3, 4}),
          "rows round up");
  require(glm_scale_shape({1536, 4096}) ==
              (std::vector<int64_t>{12, 32}),
          "real q_a grid");
  require(glm_scale_shape({4096, 16384}) ==
              (std::vector<int64_t>{32, 128}),
          "real o_proj grid");
  bool threw = false;
  try {
    (void)glm_scale_shape({128});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "1-D payload rejected");
}

// ---- the NVFP4 profile (docs/nvfp4_plan.md §2.2, gate 5) -------------------
namespace {
GlmTextConfig tiny_config_nvfp4() {
  GlmTextConfig c = tiny_config();
  c.routed_expert_format = dgpp::GlmExpertFormat::Nvfp4Group16;
  return c;
}
}  // namespace

DGPP_TEST(glm_binding_nvfp4_profile_emits_expert_triples_only_on_main_layers) {
  const GlmTextConfig cfg = tiny_config_nvfp4();
  const auto table = dgpp::glm_expected_text_tensors(cfg);
  const std::string m = "model.language_model.layers.2.mlp.experts.2.";
  // gate/up: [inter=256, hidden=512] -> packed [256, 256] U8, scales
  // [256, 32] e4m3, global [1] F32; down: [512, 256] -> [512, 128] / [512, 16].
  const auto* pk = find(table, m + "gate_proj.weight_packed");
  require(pk && pk->dtype == DType::U8 && pk->role == dgpp::GlmTensorRole::Fp4Packed,
          "gate packed present, U8, role");
  require(pk->shape == (std::vector<int64_t>{256, 256}), "gate packed shape");
  require(pk->cls == GlmWeightClass::RoutedExpert && pk->expert == 2 && pk->layer == 2,
          "gate packed class/expert/layer");
  const auto* sc = find(table, m + "gate_proj.weight_scale");
  require(sc && sc->dtype == DType::F8_E4M3 && sc->role == dgpp::GlmTensorRole::Fp4Scale,
          "gate scale present, e4m3, role");
  require(sc->shape == (std::vector<int64_t>{256, 32}), "gate scale shape");
  require(!sc->quantized(), "an e4m3 block scale is NOT an FP8 payload");
  const auto* gs = find(table, m + "gate_proj.weight_global_scale");
  require(gs && gs->dtype == DType::F32 && gs->shape == (std::vector<int64_t>{1}) &&
              gs->role == dgpp::GlmTensorRole::Fp4Global,
          "gate global scale");
  const auto* dn = find(table, m + "down_proj.weight_packed");
  require(dn && dn->shape == (std::vector<int64_t>{512, 128}), "down packed shape");
  require(find(table, m + "down_proj.weight_scale")->shape ==
              (std::vector<int64_t>{512, 16}),
          "down scale shape");
  // The FP8 pair is gone for main-layer experts ...
  require(find(table, m + "gate_proj.weight") == nullptr &&
              find(table, m + "gate_proj.weight_scale_inv") == nullptr,
          "no FP8 pair for an NVFP4 expert");
  // ... and stays for the MTP layer's experts, the shared expert and the
  // dense MLPs (the composed checkpoint keeps the FP8 release there).
  const std::string mtp = "model.language_model.layers.6.mlp.experts.1.";
  require(find(table, mtp + "gate_proj.weight")->quantized() &&
              find(table, mtp + "gate_proj.weight_scale_inv") != nullptr &&
              find(table, mtp + "gate_proj.weight_packed") == nullptr,
          "MTP experts stay FP8");
  require(find(table, "model.language_model.layers.2.mlp.shared_experts.up_proj.weight")
              ->quantized(),
          "shared expert stays FP8");
  require(find(table, "model.language_model.layers.0.mlp.down_proj.weight")->quantized(),
          "dense MLP stays FP8");
  // Count: one extra tensor per routed matrix (triple vs pair) on the four
  // MoE main layers x 4 experts x 3 matrices; the FP8 profile's table is
  // untouched.
  const auto fp8 = dgpp::glm_expected_text_tensors(tiny_config());
  require(table.size() == fp8.size() + 4 * 4 * 3, "nvfp4 table size");
  for (const auto& e : fp8) require(e.dtype != DType::U8, "FP8 table has no U8");
}

DGPP_TEST(glm_binding_nvfp4_validator_binds_the_triple_as_a_unit) {
  const GlmTextConfig cfg = tiny_config_nvfp4();
  const auto table = dgpp::glm_expected_text_tensors(cfg);
  auto present = table_as_present(table);
  auto rep = dgpp::glm_validate_text_binding(cfg, present, 8);
  require(rep.ok(), "clean nvfp4 table validates");
  require(rep.fp4_matrices == 4 * 4 * 3 && rep.fp4_bound == rep.fp4_matrices,
          "every triple bound");
  require(rep.quantized_matrices == rep.scales_bound && rep.quantized_matrices > 0,
          "the FP8 classes still bind");
  const std::string m = "model.language_model.layers.4.mlp.experts.0.up_proj.weight";
  // A missing global scale: the packed payload is present (matched) but the
  // triple is not bound, and the missing tensor is missing in its own right.
  auto no_global = present;
  no_global.erase(m + "_global_scale");
  rep = dgpp::glm_validate_text_binding(cfg, no_global, 8);
  require(!rep.ok() && rep.scales_bad == 1 && rep.missing == 1 &&
              rep.fp4_bound == rep.fp4_matrices - 1,
          "missing global scale caught");
  // A scale grid of the wrong shape (an FP8-style 128x128 grid, say).
  auto bad_scale = present;
  bad_scale[m + "_scale"] = GlmTensorDesc{DType::F8_E4M3, {2, 4}};
  rep = dgpp::glm_validate_text_binding(cfg, bad_scale, 8);
  require(!rep.ok() && rep.scales_bad == 1 && rep.shape_mismatch == 1,
          "wrong scale grid caught");
  // A scale of the wrong dtype (F32 where e4m3 is the format).
  auto bad_dtype = present;
  bad_dtype[m + "_scale"] = GlmTensorDesc{DType::F32, {256, 32}};
  rep = dgpp::glm_validate_text_binding(cfg, bad_dtype, 8);
  require(!rep.ok() && rep.scales_bad == 1 && rep.dtype_mismatch == 1,
          "wrong scale dtype caught");
  // The packed payload of the wrong shape (an unpacked [N, K] would be a
  // different format entirely).
  auto bad_packed = present;
  bad_packed[m + "_packed"] = GlmTensorDesc{DType::U8, {256, 512}};
  rep = dgpp::glm_validate_text_binding(cfg, bad_packed, 8);
  require(!rep.ok() && rep.shape_mismatch == 1, "unpacked payload caught");
  // The FP8 release's tensors against the NVFP4 profile: the experts show
  // as missing triples and unexpected pairs — exactly what glm_bind_check
  // reported on the hybrid before this profile existed, in reverse.
  auto fp8_present = table_as_present(dgpp::glm_expected_text_tensors(tiny_config()));
  rep = dgpp::glm_validate_text_binding(cfg, fp8_present, 8);
  require(!rep.ok() && rep.missing == 4 * 4 * 3 * 3 && rep.unexpected == 4 * 4 * 3 * 2,
          "FP8 experts under the NVFP4 profile: missing triples, unexpected pairs");
}

DGPP_TEST(glm_binding_nvfp4_shapes_require_k_multiple_of_16) {
  require(dgpp::glm_fp4_packed_shape(2048, 4096) == (std::vector<int64_t>{2048, 2048}),
          "real gate packed shape");
  require(dgpp::glm_fp4_scale_shape(2048, 4096) == (std::vector<int64_t>{2048, 256}),
          "real gate scale shape");
  require(dgpp::glm_fp4_packed_shape(4096, 2048) == (std::vector<int64_t>{4096, 1024}),
          "real down packed shape");
  bool threw = false;
  try {
    (void)dgpp::glm_fp4_packed_shape(64, 40);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "K not a multiple of 16 rejected");
}

DGPP_TEST(glm_config_quantization_profile_parses_fp8_and_dgpp_mixed_only) {
  using dgpp::GlmExpertFormat;
  const GlmTextConfig base = tiny_config();
  auto parse_profile = [&](const char* qjson) {
    auto q = dgpp::minijson::parse(qjson);
    return GlmTextConfig::parse_expert_format(&q.root, base);
  };
  require(GlmTextConfig::parse_expert_format(nullptr, base) == GlmExpertFormat::Fp8Block128,
          "absent quantization_config is FP8");
  require(parse_profile(R"({"quant_method":"fp8","fmt":"e4m3","weight_block_size":[128,128]})") ==
              GlmExpertFormat::Fp8Block128,
          "fp8 quant_method is FP8");
  require(parse_profile(R"({"quant_method":"dgpp_mixed",
      "routed_experts":{"format":"nvfp4-pack-quantized","num_bits":4,"group_size":16,"layers":[2,3,4,5]},
      "mtp_layer":{"source":"base"},"dsa_attention":{"source":"base"}})") ==
              GlmExpertFormat::Nvfp4Group16,
          "dgpp_mixed with the main MoE layers is NVFP4");
  auto rejects = [&](const char* qjson) {
    try {
      (void)parse_profile(qjson);
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };
  require(rejects(R"({"quant_method":"compressed-tensors","format":"nvfp4-pack-quantized"})"),
          "a BF16-non-expert NVFP4 checkpoint is rejected (no BF16 expert path)");
  require(rejects(R"({"quant_method":"dgpp_mixed","routed_experts":{"format":"nvfp4-pack-quantized","group_size":32}})"),
          "group 32 rejected");
  require(rejects(R"({"quant_method":"dgpp_mixed","routed_experts":{"format":"nvfp4-pack-quantized","group_size":16,"layers":[2,3,4,5,6]}})"),
          "an NVFP4 MTP layer is rejected");
  require(rejects(R"({"quant_method":"dgpp_mixed","routed_experts":{"format":"nvfp4-pack-quantized","group_size":16},"mtp_layer":{"source":"experts"}})"),
          "a BF16-repo MTP layer is rejected");
  // The whole file: text_config + quantization_config through from_json_file's
  // shape (parse with the root's quantization_config).
  const std::string full = std::string("{\"text_config\":") +
      R"({"hidden_size": 512, "vocab_size": 1000, "num_hidden_layers": 6,
    "rms_norm_eps": 1e-6, "tie_word_embeddings": false,
    "hidden_act": "silu", "swiglu_limit": 7.5,
    "layer_types": ["linear_attention", "linear_attention",
                    "deepseek_sparse_attention", "linear_attention",
                    "deepseek_sparse_attention", "linear_attention"],
    "mlp_layer_types": ["dense", "dense", "sparse", "sparse", "sparse", "sparse"],
    "indexer_types": ["full", "full", "full", "full", "full", "full"],
    "first_k_dense_replace": 2,
    "linear_attn_config": {"num_heads": 8, "head_dim": 64, "short_conv_kernel_size": 4,
      "gate_lower_bound": -3.5, "kda_layers": [0, 1, 3, 5], "full_attn_layers": [2, 4]},
    "num_attention_heads": 8, "q_lora_rank": 256, "kv_lora_rank": 128,
    "qk_nope_head_dim": 96, "qk_rope_head_dim": 0, "v_head_dim": 96, "mla_use_nope": true,
    "index_n_heads": 4, "index_head_dim": 128, "index_kpool": 4, "index_topk": 128,
    "index_kpool_compress": true, "index_kpool_always_select_tail": true,
    "indexer_rope_interleave": true, "intermediate_size": 1024, "moe_intermediate_size": 256,
    "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
    "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": true,
    "routed_scaling_factor": 1.5, "n_group": 1, "topk_group": 1, "moe_router_dtype": "float32",
    "mhc": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
    "num_nextn_predict_layers": 1},
    "quantization_config":{"quant_method":"dgpp_mixed","routed_experts":{"format":"nvfp4-pack-quantized","group_size":16}}})";
  auto root = dgpp::minijson::parse(full);
  const GlmTextConfig c = GlmTextConfig::parse(*root.root.find("text_config"),
                                               root.root.find("quantization_config"));
  require(c.routed_expert_format == GlmExpertFormat::Nvfp4Group16, "root parse sets the profile");
  require(c.expert_format(2) == GlmExpertFormat::Nvfp4Group16 &&
              c.expert_format(6) == GlmExpertFormat::Fp8Block128,
          "expert_format: main MoE layers NVFP4, the MTP layer FP8");
}
