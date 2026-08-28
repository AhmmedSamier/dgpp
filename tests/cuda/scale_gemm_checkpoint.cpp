// Real-checkpoint slice parity for the scale-aware GEMM (M4 deliverable 3's
// "real checkpoint slices"). Host-only TU (minijson does not mix with
// nvcc): scans the shard headers, mmaps three representative quantized
// matrices of the real GLM-5.3 revision — one MLA projection with the
// largest K, the o_proj with the largest N, and one routed expert — and
// checks launch_scale_gemm_bf16 against both oracles at M=48.
//
// Manual deployment test where the checkpoint lives; CI runs the synthetic
// suite only (scale_gemm_test without --checkpoint-dir).
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "kernels/scale_gemm.hpp"
#include "loaders/safetensors.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;

struct Slice {
  const char* name;
  int64_t rows;  // N
  int64_t cols;  // K
};

// Representative coverage: q_b has the largest K (1536) and 12 scale
// columns; o_proj the largest N (16384) and 128 scale rows; the expert
// down_proj is the shape every MoE layer repeats 288 times over.
const Slice kSlices[] = {
    {"model.language_model.layers.3.self_attn.q_b_proj.weight", 16384, 1536},
    {"model.language_model.layers.3.self_attn.o_proj.weight", 4096, 16384},
    {"model.language_model.layers.4.mlp.experts.0.down_proj.weight", 4096,
     2048},
};

// Short name for reports: the tensor's last two dotted components.
std::string short_name(const std::string& name) {
  const size_t p = name.rfind('.');
  const size_t q = name.rfind('.', p - 1);
  return name.substr(q + 1);
}

bool run_slice(const dgpp::SafetensorsFile& shard, const Slice& slice) {
  const dgpp::TensorInfo* payload = shard.find(slice.name);
  const dgpp::TensorInfo* scales =
      shard.find(std::string(slice.name) + "_scale_inv");
  if (!payload || !scales)
    throw std::runtime_error(std::string("slice tensors missing: ") +
                             slice.name);
  if (payload->dtype != dgpp::DType::F8_E4M3 ||
      payload->shape[0] != slice.rows || payload->shape[1] != slice.cols)
    throw std::runtime_error(std::string("slice payload geometry: ") +
                             slice.name);
  if (scales->dtype != dgpp::DType::F32)
    throw std::runtime_error(std::string("slice scale dtype: ") +
                             slice.name);

  const int m = 48;
  const int n = static_cast<int>(slice.rows);
  const int k = static_cast<int>(slice.cols);
  Rng rng(0x5EED ^ slice.rows ^ slice.cols);
  std::vector<uint16_t> act(static_cast<size_t>(m) * k);
  fill_act(rng, act);
  std::vector<float> scales_v(scales->numel());
  std::memcpy(scales_v.data(), scales->data, scales_v.size() * 4);

  uint8_t* dev_w = nullptr;
  float* dev_s = nullptr;
  uint16_t* dev_act = nullptr;
  uint16_t* dev_out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&dev_w, payload->nbytes()));
  DGPP_CUDA_OK(cudaMallocManaged(&dev_s, scales_v.size() * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&dev_act, act.size() * 2));
  DGPP_CUDA_OK(
      cudaMallocManaged(&dev_out, static_cast<size_t>(m) * n * 2));
  std::memcpy(dev_w, payload->data, payload->nbytes());
  std::memcpy(dev_s, scales_v.data(), scales_v.size() * 4);
  std::memcpy(dev_act, act.data(), act.size() * 2);

  dgpp::launch_scale_gemm_bf16(dev_act, k, dev_w, dev_s, dev_out, m, n, k,
                               nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(m) * n);
  std::memcpy(got.data(), dev_out, got.size() * 2);
  DGPP_CUDA_OK(cudaFree(dev_w));
  DGPP_CUDA_OK(cudaFree(dev_s));
  DGPP_CUDA_OK(cudaFree(dev_act));
  DGPP_CUDA_OK(cudaFree(dev_out));

  // Oracles read the payload straight from the mmap.
  const std::vector<uint8_t> payload_v(
      static_cast<const uint8_t*>(payload->data),
      static_cast<const uint8_t*>(payload->data) + payload->nbytes());
  bool ok = true;
  for (int strict_i = 0; strict_i < 2; ++strict_i) {
    const bool strict = strict_i == 1;
    const auto oracle =
        gemm_oracle(act, k, payload_v, scales_v, m, n, k, strict);
    const auto rep = compare_bf16_vs_oracle(
        got.data(), oracle, strict ? 2.0 : 8.0, strict ? 1e-3 : 2e-2);
    const double l2_budget = strict ? 1e-3 : 8e-3;
    const long mm_budget = strict ? 0 : static_cast<long>(rep.total) / 20;
    const bool pass = rep.nan_pattern_errors == 0 &&
                      rep.mismatches <= mm_budget && rep.l2_rel <= l2_budget;
    ok &= pass;
    std::printf("[%s] %s %s: l2_rel=%.3g mismatches=%ld/%zu\n",
                pass ? " OK " : "FAIL", short_name(slice.name).c_str(),
                strict ? "strict" : "semantic", rep.l2_rel, rep.mismatches,
                rep.total);
  }
  return ok;
}

}  // namespace

int run_scale_gemm_checkpoint_parity(const char* checkpoint_dir) {
  namespace fs = std::filesystem;
  std::vector<fs::path> shard_paths;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors")
      shard_paths.push_back(entry.path());
  if (shard_paths.empty())
    throw std::runtime_error("no .safetensors shards in checkpoint dir");
  std::sort(shard_paths.begin(), shard_paths.end());

  // Open every shard once; the mmaps must outlive the slice runs.
  std::vector<std::unique_ptr<dgpp::SafetensorsFile>> shards;
  for (const auto& p : shard_paths)
    shards.push_back(dgpp::SafetensorsFile::open(p.string()));

  bool ok = true;
  size_t found = 0;
  for (const auto& shard : shards) {
    for (const auto& slice : kSlices) {
      if (shard->find(slice.name)) {
        ++found;
        ok &= run_slice(*shard, slice);
      }
    }
  }
  if (found != std::size(kSlices))
    throw std::runtime_error("checkpoint lacks one or more parity slices");
  if (!ok) {
    std::printf("checkpoint slice parity FAILED\n");
    return 1;
  }
  std::printf("checkpoint slice parity OK (%zu slices x strict+semantic)\n",
              found);
  return 0;
}
