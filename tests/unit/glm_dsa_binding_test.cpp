// The full GLM-5.3 expected-tensor table: its shape on the release's
// config (the per-layer counts the shard headers carry), the TP geometry
// acceptance at the deployment worlds, and — when every shard of the
// checkpoint is in the hub cache — the full binding against every shard's
// header (78 layer shards and passthrough.safetensors: every expected
// tensor present with its dtype and shape, nothing unexpected).
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "glm_dsa_config_json.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/glm_dsa/binding.hpp"
#include "models/glm_dsa/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::GlmDsaTextConfig release_config() {
  // minijson references the text it parses: keep it alive across parse().
  const std::string text = glm_dsa_test::config_json();
  const auto t = dgpp::minijson::parse(text);
  return dgpp::GlmDsaTextConfig::parse(t.root);
}

// The snapshot directory, only when config.json, the index and EVERY file
// the index names are present (a download in progress links the finished
// shards only; a partial snapshot would fail the binding, not skip it).
std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root =
      fs::path(home) / ".cache/huggingface/hub/models--HawkBearPig--GLM-5.3-Int4-Int8Mix-RTN-g64/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path index = snap.path() / "model.safetensors.index.json";
    if (!fs::exists(snap.path() / "config.json") || !fs::exists(index)) continue;
    std::ifstream in(index);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto parsed = dgpp::minijson::parse(text);
    const dgpp::minijson::Value* wm = parsed.root.find("weight_map");
    if (!wm || !wm->is_object()) continue;
    bool complete = true;
    for (const auto& m : wm->members())
      if (!m.value.is_string() || !fs::exists(snap.path() / std::string(m.value.as_string()))) {
        complete = false;
        break;
      }
    if (complete) return snap.path();
  }
  return {};
}

}  // namespace

DGPP_TEST(glm_dsa_binding_table_has_the_release_shape) {
  const dgpp::GlmDsaTextConfig cfg = release_config();
  // A dense layer: 2 norms + 2 latent norms + 5 BF16 projections + 5 indexer + 3 MLP = 17.
  const auto dense = dgpp::glm_dsa_expected_layer_tensors(cfg, 0);
  require(dense.size() == 17, "dense layer " + std::to_string(dense.size()));
  // A MoE layer without an indexer: 4 norms + 5 x 3 attention + router 2 + 257 x 3 x 3 = 2334.
  const auto moe = dgpp::glm_dsa_expected_layer_tensors(cfg, 3);
  require(moe.size() == 2334, "moe layer " + std::to_string(moe.size()));
  // One that owns an indexer: 2334 + 5.
  const auto moe_idx = dgpp::glm_dsa_expected_layer_tensors(cfg, 6);
  require(moe_idx.size() == 2339, "indexer moe layer " + std::to_string(moe_idx.size()));
  // The draft: 4 head + 4 norms + 5 attention + 5 indexer + router 2 + 257 x 3 = 791.
  const auto draft = dgpp::glm_dsa_expected_layer_tensors(cfg, cfg.mtp_layer());
  require(draft.size() == 791, "draft layer " + std::to_string(draft.size()));
  require(draft[0].name == "model.layers.78.enorm.weight", "draft prefix");
  const auto all = dgpp::glm_dsa_expected_text_tensors(cfg);
  require(all.size() == 175985, "table size " + std::to_string(all.size()));
  size_t int4 = 0, int8 = 0, bf16_experts = 0, shapes = 0;
  for (const auto& e : all) {
    if (e.role == dgpp::GlmDsaTensorRole::IntPacked) (e.bits == 4 ? int4 : int8)++;
    if (e.role == dgpp::GlmDsaTensorRole::Bf16Expert) ++bf16_experts;
    if (e.role == dgpp::GlmDsaTensorRole::IntShape) ++shapes;
  }
  require(int4 == 75 * 256 * 3, "int4 matrices " + std::to_string(int4));
  require(int8 == 75 * 8, "int8 matrices " + std::to_string(int8));
  require(bf16_experts == 257 * 3, "draft expert matrices " + std::to_string(bf16_experts));
  require(shapes == int4 + int8, "shape records");
  // The packed shapes: gate [2048, 6144] int4 -> packed [2048, 768], scale [2048, 96];
  // o_proj [6144, 16384] int8 -> packed [6144, 4096], scale [6144, 256].
  for (const auto& e : moe) {
    if (e.name == "model.layers.3.mlp.experts.0.gate_proj.weight_packed")
      require(e.dtype == dgpp::DType::I32 && e.shape == std::vector<int64_t>{2048, 768}, "gate packed shape");
    if (e.name == "model.layers.3.mlp.experts.0.gate_proj.weight_scale")
      require(e.dtype == dgpp::DType::BF16 && e.shape == std::vector<int64_t>{2048, 96}, "gate scale shape");
    if (e.name == "model.layers.3.self_attn.o_proj.weight_packed")
      require(e.shape == std::vector<int64_t>{6144, 4096} && e.bits == 8, "o_proj packed shape");
    if (e.name == "model.layers.3.self_attn.kv_a_proj_with_mqa.weight_packed")
      require(e.shape == std::vector<int64_t>{576, 1536}, "kv_a packed shape (512 + 64 rows)");
    if (e.name == "model.layers.3.self_attn.kv_b_proj.weight_shape")
      require(e.dtype == dgpp::DType::I64 && e.shape == std::vector<int64_t>{2}, "shape record");
  }
}

DGPP_TEST(glm_dsa_tp_geometry_accepts_the_deployment_worlds) {
  const dgpp::GlmDsaTextConfig cfg = release_config();
  for (const int w : {1, 2, 4, 8, 16, 32})
    for (int r = 0; r < w; ++r) dgpp::glm_dsa_tp_validate_geometry(cfg, r, w);
  auto refused = [&](int world) {
    try {
      dgpp::glm_dsa_tp_validate_geometry(cfg, 0, world);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(refused(64), "a 32-wide expert slice is refused");  // 2048 / 64 = 32 < the group
  require(refused(3), "a world that does not divide the heads is refused");
}

DGPP_TEST(glm_dsa_binding_matches_the_landed_checkpoint) {
  const auto snap = landed_snapshot();
  if (snap.empty()) return;
  namespace fs = std::filesystem;
  const dgpp::GlmDsaTextConfig cfg = dgpp::GlmDsaTextConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::GlmDsaTensorDesc> present;
  for (const auto& entry : fs::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      present.emplace(t.name, dgpp::GlmDsaTensorDesc{t.dtype, t.shape});
    });
  }
  const dgpp::GlmDsaBindReport rep = dgpp::glm_dsa_validate_text_binding(cfg, present);
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + ", dtype " +
                        std::to_string(rep.dtype_mismatch) + ", shape " + std::to_string(rep.shape_mismatch) +
                        ", unexpected " + std::to_string(rep.unexpected) + errs);
  require(rep.expected == 175985 && rep.matched == 175985, "every tensor bound");
  require(rep.packed_int4_matrices == 75 * 256 * 3, "int4 matrices " + std::to_string(rep.packed_int4_matrices));
  require(rep.packed_int8_matrices == 75 * 8, "int8 matrices " + std::to_string(rep.packed_int8_matrices));
  require(rep.bf16_expert_matrices == 257 * 3, "draft experts");
}
