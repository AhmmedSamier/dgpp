// Matched FP8 prefill launches at sparse expert occupancy. The row-count
// bound is the prompt length, as in serving; no CPU maximum is supplied.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "moe_prefill_variants.hpp"

using namespace dgpp;

namespace {
template <typename T> struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { DGPP_CUDA_OK(cudaMalloc(&p, n * sizeof(T))); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { cudaFree(p); }
  void upload(const std::vector<T>& v) {
    DGPP_CUDA_OK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
};

double milliseconds(int iters, const std::function<void()>& launch) {
  for (int i = 0; i < 3; ++i) launch();
  cudaEvent_t begin, end;
  DGPP_CUDA_OK(cudaEventCreate(&begin));
  DGPP_CUDA_OK(cudaEventCreate(&end));
  DGPP_CUDA_OK(cudaEventRecord(begin));
  for (int i = 0; i < iters; ++i) launch();
  DGPP_CUDA_OK(cudaEventRecord(end));
  DGPP_CUDA_OK(cudaEventSynchronize(end));
  float ms;
  DGPP_CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return ms / iters;
}

template <typename OutT>
void projection(const char* label, int n, int k, int rs, int cs, int tokens,
                const std::vector<MoeSegment>& segs, const std::vector<int32_t>& rows,
                bool mapped, int iters, bench::Variant variant, std::mt19937& rng) {
  const int experts = static_cast<int>(segs.size());
  const size_t weight_count = static_cast<size_t>(n) * k;
  const size_t scale_count = static_cast<size_t>((n + (1 << rs) - 1) >> rs) * ((k + (1 << cs) - 1) >> cs);
  std::vector<uint8_t> codes(weight_count * experts);
  std::vector<float> scales(scale_count * experts);
  for (auto& v : codes) {
    const uint32_t magnitude = rng() % 127, sign = rng() % 2;
    v = static_cast<uint8_t>(magnitude + sign * 128);
  }
  for (auto& v : scales) v = 0.002f + (rng() % 1000) * 0.00001f;
  Buffer<uint8_t> weights(codes.size()); weights.upload(codes);
  Buffer<float> factors(scales.size()); factors.upload(scales);
  std::vector<MoeExpertView> views(experts * 3);
  for (int e = 0; e < experts; ++e) {
    views[e * 3].payload = weights.p + e * weight_count;
    views[e * 3].scales = factors.p + e * scale_count;
    views[e * 3].scale_shift_rows = rs;
    views[e * 3].scale_shift_cols = cs;
  }
  Buffer<MoeExpertView> device_views(views.size()); device_views.upload(views);
  Buffer<MoeSegment> device_segs(segs.size()); device_segs.upload(segs);
  Buffer<int32_t> device_rows(rows.size()); device_rows.upload(rows);
  std::vector<uint16_t> hidden(static_cast<size_t>(mapped ? tokens : rows.size()) * k);
  for (auto& v : hidden) v = float_to_bf16_bits((int(rng() % 2001) - 1000) * 0.001f);
  Buffer<uint16_t> act(hidden.size()); act.upload(hidden);
  const size_t output_count = rows.size() * n;
  Buffer<OutT> baseline(output_count), candidate(output_count);
  const bool selected = tokens <= 256 && experts > 1 &&
      (variant == bench::Variant::kSmallTiles ||
       (std::is_same_v<OutT, float> && tokens > 64 && n >= 1024 && k <= 640));
  std::printf("%s candidate_selected=%s\n", label, selected ? "yes" : "no");
  auto launch = [&](OutT* out, bool candidate) {
    if (candidate && selected) {
      if constexpr (std::is_same_v<OutT, uint16_t>)
        bench::launch_variant_bf16(variant, act.p, k, mapped ? device_rows.p : nullptr,
            device_segs.p, experts, device_views.p, 0, out, n, n, k, nullptr);
      else
        bench::launch_variant_f32(variant, act.p, k, mapped ? device_rows.p : nullptr,
            device_segs.p, experts, device_views.p, 0, out, n, n, k, nullptr);
      return;
    }
    if constexpr (std::is_same_v<OutT, uint16_t>)
      launch_moe_grouped_mma_bf16(act.p, k, device_segs.p, experts, tokens, 0,
          device_views.p, 0, out, n, n, k, nullptr, mapped ? device_rows.p : nullptr);
    else
      launch_moe_grouped_mma_f32(act.p, k, device_segs.p, experts, tokens, 0,
          device_views.p, 0, out, n, n, k, nullptr, mapped ? device_rows.p : nullptr);
  };
  std::vector<OutT> ref(output_count), got(output_count);
  for (int trial = 0; trial < 6; ++trial) {
    double ms[2];
    for (const int mode : {trial % 2, 1 - trial % 2}) {
      ms[mode] = milliseconds(iters, [&] { launch(mode ? candidate.p : baseline.p, mode != 0); });
    }
    DGPP_CUDA_OK(cudaMemcpy(ref.data(), baseline.p, output_count * sizeof(OutT), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(got.data(), candidate.p, output_count * sizeof(OutT), cudaMemcpyDeviceToHost));
    if (std::memcmp(ref.data(), got.data(), output_count * sizeof(OutT)) != 0)
      throw std::runtime_error(std::string(label) + ": candidate differs from baseline");
    std::printf("%s trial=%d baseline_ms=%.6f candidate_ms=%.6f change_percent=%+.3f bitwise=yes\n",
                label, trial, ms[0], ms[1], 100 * (ms[1] / ms[0] - 1));
    std::fflush(stdout);
  }
}
}  // namespace

int main(int argc, char** argv) try {
  int tokens = 256, experts = 512, top_k = 10, hidden = 2560, inter = 320, scale_block = 64, iters = 20;
  std::string distribution = "uniform", variant_name = "compact";
  std::string segments_file;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 == argc) throw std::invalid_argument("missing option value");
    const std::string key = argv[i], value = argv[i + 1];
    if (key == "--distribution") distribution = value;
    else if (key == "--variant") variant_name = value;
    else if (key == "--segments-file") segments_file = value;
    else if (key == "--tokens") tokens = std::stoi(value);
    else if (key == "--experts") experts = std::stoi(value);
    else if (key == "--top-k") top_k = std::stoi(value);
    else if (key == "--hidden") hidden = std::stoi(value);
    else if (key == "--inter") inter = std::stoi(value);
    else if (key == "--scale-block") scale_block = std::stoi(value);
    else if (key == "--iters") iters = std::stoi(value);
    else throw std::invalid_argument("unknown option: " + key);
  }
  if (tokens <= 0 || experts < top_k || top_k <= 0 || iters <= 0 || hidden <= 0 || inter <= 0 ||
      hidden % 32 || inter % 32 || (scale_block != 32 && scale_block != 64 && scale_block != 128) ||
      (distribution != "uniform" && distribution != "skewed" && distribution != "hot"))
    throw std::invalid_argument("invalid shape, iteration count or distribution");
  if (variant_name != "small" && variant_name != "compact" && variant_name != "persistent")
    throw std::invalid_argument("variant must be small, compact or persistent");
  const bench::Variant variant = variant_name == "small" ? bench::Variant::kSmallTiles :
      variant_name == "compact" ? bench::Variant::kCompact : bench::Variant::kPersistent;
  std::printf("experimental_variant=%s (benchmark only)\n", variant_name.c_str());
  std::mt19937 rng(0x51A11);
  std::vector<std::vector<int32_t>> routes(experts);
  if (!segments_file.empty()) {
    std::ifstream input(segments_file);
    int count, total = 0;
    for (int e = 0; e < experts; ++e) {
      if (!(input >> count) || count < 0 || count > tokens)
        throw std::invalid_argument("segments file needs one row count per expert within the prompt length");
      // Replay measured segment sizes with deterministic synthetic rows.
      // This reproduces the work geometry, not the model's activations.
      for (int j = 0; j < count; ++j) routes[e].push_back((e * 31 + j) % tokens);
      total += count;
    }
    if (input >> count || total != tokens * top_k)
      throw std::invalid_argument("segments file must contain exactly tokens * top_k assignments");
    distribution = "recorded";
  } else for (int t = 0; t < tokens; ++t) {
    std::vector<int> selected;
    while (static_cast<int>(selected.size()) < top_k) {
      const int pool = distribution == "hot" ? top_k :
          distribution == "skewed" && rng() % 2 ? std::max(top_k, experts / 8) : experts;
      const int expert = rng() % pool;
      if (std::find(selected.begin(), selected.end(), expert) != selected.end()) continue;
      selected.push_back(expert);
      routes[expert].push_back(t);
    }
  }
  std::vector<int32_t> rows;
  std::vector<MoeSegment> segs;
  int nonempty = 0, maximum = 0;
  for (int e = 0; e < experts; ++e) {
    const int count = static_cast<int>(routes[e].size());
    segs.push_back({static_cast<int>(rows.size()), count, e});
    rows.insert(rows.end(), routes[e].begin(), routes[e].end());
    nonempty += count > 0;
    maximum = std::max(maximum, count);
  }
  int shift = 0;
  while ((1 << shift) < scale_block) ++shift;
  std::printf("shape tokens=%d experts=%d top_k=%d hidden=%d inter=%d scale_block=%d distribution=%s "
              "nonempty=%d max_rows=%d launch_bound=%d\n", tokens, experts, top_k, hidden, inter,
              scale_block, distribution.c_str(), nonempty, maximum, tokens);
  projection<uint16_t>("gate", inter, hidden, shift, 7, tokens, segs, rows, true, iters, variant, rng);
  projection<float>("down", hidden, inter, 7, shift, tokens, segs, rows, false, iters, variant, rng);
} catch (const std::exception& error) {
  std::fprintf(stderr, "moe_prefill_bench: %s\n", error.what());
  return 1;
}
