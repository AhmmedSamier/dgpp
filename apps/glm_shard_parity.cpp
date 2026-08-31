// M5 d4 at REAL scale: the sharded-load parity gate as a deployment
// command. Runs the exact CI shard-parity surface (glm_tp_parity — the
// same code glm_tp_test pins at the fixture) against a real checkpoint
// resolved from the canonical HF hub cache: per layer, per rank, the
// sharded GlmLayerStream's bind_sharded output vs full-load + bind,
// BITWISE, plus the replicated-digest agreement and the byte reconcile.
//
// A mismatch prints the layer and surface that differ and exits 1 — this
// is the instrument that answers "is the loader the slicer, at the real
// 76k-tensor geometry, not just the 463-tensor fixture?"
//
// No bus, no fabric: one node, models library only.
//
//   glm_shard_parity --model ORG/NAME [--worlds 2,4] [--layers 0,3,45]
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/log.hpp"
#include "loaders/hf_cache.hpp"
#include "models/glm_config.hpp"
#include "models/glm_tp_parity.hpp"

namespace fs = std::filesystem;

namespace {

std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    out.push_back(std::stoi(item));
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");

  std::string model_id, ckpt;
  std::vector<int> worlds{2};
  std::unique_ptr<std::vector<int>> layers;  // null = all
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        DGPP_LOG_ERROR("missing value for {}", a);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--model") model_id = next();
    else if (a == "--checkpoint-dir") ckpt = next();
    else if (a == "--worlds") worlds = parse_ints(next());
    else if (a == "--layers")
      layers = std::make_unique<std::vector<int>>(parse_ints(next()));
    else {
      std::fprintf(stderr,
                    "usage: glm_shard_parity --model ORG/NAME | "
                    "--checkpoint-dir DIR [--worlds 2,4] [--layers 0,3,45]\n");
      return a == "--help" ? 0 : 1;
    }
  }
  if (!model_id.empty() && !ckpt.empty()) {
    DGPP_LOG_ERROR("--model and --checkpoint-dir are mutually exclusive");
    return 1;
  }
  if (model_id.empty() && ckpt.empty()) {
    std::fprintf(stderr,
                  "usage: glm_shard_parity --model ORG/NAME | "
                  "--checkpoint-dir DIR [--worlds 2,4] [--layers 0,3,45]\n");
    return 1;
  }
  if (model_id.empty()) {
    model_id = "[checkpoint-dir " + ckpt + "]";
  } else {
    std::string err;
    const std::string snapshot = dgpp::hf::model_dir(model_id, &err);
    if (snapshot.empty()) {
      DGPP_LOG_ERROR("--model {}: {}", model_id, err);
      return 1;
    }
    ckpt = snapshot;
    DGPP_LOG_INFO("model {} -> {}", model_id, snapshot);
  }

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    DGPP_LOG_ERROR("no CUDA device visible");
    return 1;
  }
  const dgpp::GlmTextConfig cfg =
      dgpp::GlmTextConfig::from_json_file((fs::path(ckpt) / "config.json").string());

  try {
    for (int world : worlds) {
      const auto t0 = std::chrono::steady_clock::now();
      const dgpp::GlmShardParityReport rep =
          dgpp::glm_shard_parity_check(cfg, ckpt, world, layers.get());
      const double s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();
      DGPP_LOG_INFO(
          "shard parity world={} PASSED: {} layers x {} ranks, {} bound "
          "surfaces bitwise-equal; byte reconcile exact — rank reads "
          "{}/{} source bytes ({:.1f}%, verbatim {} MB); digest {} tensors "
          "/ {} MB; {:.0f}s",
          rep.world, rep.layers_checked, world, rep.surfaces_checked,
          rep.shard_source_bytes >> 20, rep.full_source_bytes >> 20,
          100.0 * static_cast<double>(rep.shard_source_bytes) /
              static_cast<double>(rep.full_source_bytes),
          rep.verbatim_bytes >> 20, rep.digest_tensors,
          rep.digest_bytes >> 20, s);
    }
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("shard parity FAILED: {}", e.what());
    return 1;
  }
  return 0;
}
