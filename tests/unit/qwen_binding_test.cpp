// The Qwen3.8-Flash-Next expected-tensor table: its size
// and the n-gram shard geometry on the release's config, the TP geometry
// acceptance at the deployment worlds, and — when the checkpoint is in the
// hub cache — the full binding against every shard's header (152 089
// tensors: every expected one present with its dtype and shape, nothing
// unexpected but the vision tower).
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/safetensors.hpp"
#include "models/qwen/binding.hpp"
#include "models/qwen/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json") &&
        fs::exists(snap.path() / "model.safetensors.index.json"))
      return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(qwen_binding_table_has_the_release_shape) {
  const auto snap = landed_snapshot();
  if (snap.empty()) return;
  const dgpp::QwenTextConfig cfg = dgpp::QwenTextConfig::from_json_file((snap / "config.json").string());
  const auto table = dgpp::qwen_expected_text_tensors(cfg);
  // 152 089 tensors in the file less the 333 of the vision tower.
  require(table.size() == 151756, "table size " + std::to_string(table.size()));
  require(dgpp::qwen_ngram_shard_capacity(cfg) == 2500012, "shard capacity");
  int64_t rows = 0;
  for (int s = 0; s < cfg.split_ngram_parts; ++s) rows += dgpp::qwen_ngram_shard_rows(cfg, s);
  require(rows == 320001536, "shard rows cover the padded table");
  const auto layer1 = dgpp::qwen_expected_layer_tensors(cfg, 1);
  const auto layer0 = dgpp::qwen_expected_layer_tensors(cfg, 0);
  require(layer1.size() == layer0.size() + 10 + 128, "the PLE layer adds its 10 tensors and 128 shards");
  const auto draft = dgpp::qwen_expected_layer_tensors(cfg, cfg.mtp_layer());
  require(!draft.empty() && draft[0].name.rfind("mtp.layers.0.", 0) == 0, "the draft layer's prefix");
}

DGPP_TEST(qwen_tp_geometry_accepts_the_deployment_worlds) {
  const auto snap = landed_snapshot();
  if (snap.empty()) return;
  const dgpp::QwenTextConfig cfg = dgpp::QwenTextConfig::from_json_file((snap / "config.json").string());
  for (const int w : {1, 2, 4, 8})
    for (int r = 0; r < w; ++r) dgpp::qwen_tp_validate_geometry(cfg, r, w);
  bool refused = false;
  try {
    dgpp::qwen_tp_validate_geometry(cfg, 0, 3);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "world 3 refused");
}

DGPP_TEST(qwen_binding_validates_the_landed_checkpoint_when_present) {
  namespace fs = std::filesystem;
  const auto snap = landed_snapshot();
  if (snap.empty()) return;
  const dgpp::QwenTextConfig cfg = dgpp::QwenTextConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::QwenTensorDesc> present;
  std::vector<fs::path> shards;
  for (const auto& entry : fs::directory_iterator(snap))
    if (entry.path().extension() == ".safetensors") shards.push_back(entry.path());
  require(shards.size() == 131, "131 shards");
  for (const auto& path : shards) {
    auto f = dgpp::SafetensorsFile::open(path.string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      present.emplace(t.name, dgpp::QwenTensorDesc{t.dtype, t.shape});
    });
  }
  require(present.size() == 152089, "152 089 tensors in the headers");
  const dgpp::QwenBindReport rep = dgpp::qwen_validate_text_binding(cfg, present);
  std::string first = rep.errors.empty() ? "" : rep.errors[0];
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + " dtype " +
                        std::to_string(rep.dtype_mismatch) + " shape " +
                        std::to_string(rep.shape_mismatch) + " unexpected " +
                        std::to_string(rep.unexpected) + " first: " + first);
  require(rep.vision == 333 && rep.quantized_matrices == 73728 + 1536 && rep.ngram_shards == 128,
          "the census");
}
