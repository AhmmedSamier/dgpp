// gpt_doll: M1 validation harness.
//   --selftest            bitwise replay determinism + eager/graph parity
//   --bench [N]           decode roofline benchmark (sync path)
//   --bench-rollout N     monolithic self-feeding rollout graph benchmark
//   --soak-minutes M      steady-state allocation audit over long loop
//   --gen TOKENS...       print a sampled rollout transcript
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "core/arena.hpp"
#include "core/graph.hpp"
#include "core/streams.hpp"
#include "kernels/gemm.hpp"
#include "models/gpt_doll.hpp"

namespace {

using dgpp::Arena;
using dgpp::CublasLtGemm;
using dgpp::DollConfig;
using dgpp::DollModel;
using dgpp::GraphCache;
using dgpp::StreamPool;

struct Options {
  std::string mode = "selftest";
  DollConfig cfg;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next_int = [&]() -> int {
      if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
      return std::atoi(argv[++i]);
    };
    if (a == "--selftest") o.mode = "selftest";
    else if (a == "--bench") { o.mode = "bench"; }
    else if (a == "--bench-rollout") { o.mode = "bench-rollout"; }
    else if (a == "--gen") o.mode = "gen";
    else if (a == "--soak-minutes") o.mode = "soak";
    else if (a == "--hidden") o.cfg.hidden = next_int();
    else if (a == "--layers") o.cfg.layers = next_int();
    else if (a == "--vocab") o.cfg.vocab = next_int();
    else if (a == "--inter") o.cfg.inter = next_int();
    else if (a == "--max-seq") o.cfg.max_seq = next_int();
    else if (a == "--seed") o.cfg.seed = static_cast<uint64_t>(next_int());
    else if (a.rfind("--", 0) == 0) {
      // skip-value flags below are consumed only by main loop bodies
      continue;
    } else {
      // positional values handled per-mode in main()
    }
  }
  return o;
}

int collect_int_args_after_flags(int argc, char** argv,
                                 const std::string& mode_flag, int dflt) {
  for (int i = 1; i < argc; ++i) {
    if (mode_flag == argv[i] && i + 1 < argc && argv[i + 1][0] != '-')
      return std::atoi(argv[i + 1]);
  }
  return dflt;
}

struct Rig {
  Arena arena;
  StreamPool streams;
  GraphCache graphs;
  CublasLtGemm gemm;
  explicit Rig(const DollConfig& cfg)
      : streams(), graphs(), gemm() {
    // Exact accounting from the model's own allocator footprint plus a
    // 4 MB alignment-margin so bump rounding never OOMs.
    const size_t hot = dgpp::DollModel::persistent_hot_bytes(
                           cfg, dgpp::DollModel::kMaxRolloutSteps) +
                       (4ull << 20);
    arena.init({.persistent_hot = hot,
                .scratch_hot = 16ull << 20,
                .host_pinned = 1ull << 20});
    streams.init();
    DGPP_LOG_INFO("arena hot reserve {:.1f} MB",
                  static_cast<double>(hot) / 1048576.0);
  }
};

std::vector<int64_t> make_prompt(DollConfig cfg, int n) {
  std::mt19937_64 rng(cfg.seed ^ 0x5EED);
  std::vector<int64_t> t(n);
  for (auto& v : t) v = static_cast<int64_t>(rng() % static_cast<uint64_t>(cfg.vocab));
  return t;
}

int run_selftest(const Options& o) {
  Rig rig(o.cfg);
  DollModel model(rig.arena, rig.streams, rig.graphs, rig.gemm, o.cfg);
  model.init_weights();
  std::printf("STAGE weights-done\n");

  const int kSteps = 48;
  const auto prompt = make_prompt(o.cfg, 32);

  // ---- run B first: eager (no graphs), isolates launch vs capture issues --
  std::vector<int64_t> ids_b;
  model.reset_kv();
  rig.streams.sync_all();
  int64_t t = model.eager_prefill_result(prompt);
  ids_b.push_back(t);
  for (int i = 1; i < kSteps; ++i) {
    t = model.eager_decode_step(t);
    ids_b.push_back(t);
  }
  std::printf("STAGE eager-done head=%lld\n", static_cast<long long>(ids_b[0]));

  // ---- run A: captured graphs ------------------------------------------
  std::vector<int64_t> ids_a;
  model.reset_kv();
  rig.streams.sync_all();
  model.enqueue_prefill(prompt);
  t = model.sample_prefill_result();
  ids_a.push_back(t);
  for (int i = 1; i < kSteps; ++i) {
    model.enqueue_decode_step(t);
    t = model.poll_decode_sample();
    ids_a.push_back(t);
  }
  std::printf("STAGE captured-done\n");

  size_t mismatches = 0;
  for (int i = 0; i < kSteps; ++i) {
    if (ids_a[i] != ids_b[i]) {
      DGPP_LOG_ERROR("step {} id mismatch captured={} eager={}", i, ids_a[i],
                     ids_b[i]);
      ++mismatches;
    }
  }
  if (mismatches == 0)
    std::printf("PASS graph-vs-eager bitwise transcript (%d steps)\n", kSteps);

  // ---- graph replay repeatability (same inputs twice) -------------------
  model.reset_kv();
  rig.streams.sync_all();
  std::vector<int64_t> ids_c;
  model.enqueue_prefill(prompt);
  t = model.sample_prefill_result();
  ids_c.push_back(t);
  for (int i = 1; i < kSteps / 2; ++i) {
    model.enqueue_decode_step(t);
    t = model.poll_decode_sample();
    ids_c.push_back(t);
  }
  bool repeat_ok = true;
  for (size_t i = 0; i < ids_c.size(); ++i)
    if (ids_c[i] != ids_a[i]) repeat_ok = false;
  std::printf(repeat_ok ? "PASS graph replay repeatability\n"
                        : "FAIL graph replay drifted\n");
  mismatches += repeat_ok ? 0 : 1;

  // ---- rollout mega-graph -----------------------------------------------
  model.capture_rollout(kSteps);
  model.reset_kv();
  model.run_rollout(prompt.front());
  const std::vector<int64_t> first_run = model.rollout_ids();
  bool roll_ok = true;
  for (int64_t id : first_run)
    if (id < 0 || id >= o.cfg.vocab) roll_ok = false;
  model.reset_kv();
  model.run_rollout(prompt.front());
  for (int i = 0; i < kSteps; ++i) {
    if (first_run[i] != model.rollout_ids()[i]) roll_ok = false;
  }
  std::printf(roll_ok ? "PASS rollout graph replay determinism\n"
                      : "FAIL rollout graph replay\n");
  mismatches += roll_ok ? 0 : 1;

  std::printf("graph stats: captures=%llu hits=%llu\n",
              static_cast<unsigned long long>(rig.graphs.captures()),
              static_cast<unsigned long long>(rig.graphs.hits()));
  return mismatches == 0 ? 0 : 1;
}

int run_bench(const Options& o, bool use_rollout, int steps) {
  Rig rig(o.cfg);
  DollModel model(rig.arena, rig.streams, rig.graphs, rig.gemm, o.cfg);
  model.init_weights();

  cudaEvent_t beg, end;
  DGPP_CUDA_OK(cudaEventCreate(&beg));
  DGPP_CUDA_OK(cudaEventCreate(&end));

  double wall_ns = 0;
  std::vector<int64_t> ids;
  if (use_rollout) {
    model.capture_rollout(steps);
    // warmup: clocks & heuristics
    model.reset_kv();
    model.run_rollout(7);
    model.reset_kv();
    DGPP_CUDA_OK(cudaEventRecord(beg, rig.streams.stream(StreamPool::Compute)));
    model.run_rollout(7);
    DGPP_CUDA_OK(cudaEventRecord(end, rig.streams.stream(StreamPool::Compute)));
    DGPP_CUDA_OK(cudaEventSynchronize(end));
    float ms = 0;
    DGPP_CUDA_OK(cudaEventElapsedTime(&ms, beg, end));
    wall_ns = static_cast<double>(ms) * 1e6;
    ids.assign(model.rollout_ids().begin(), model.rollout_ids().end());
    model.reset_kv();
  } else {
    constexpr int kWarmup = 24;
    auto prompt = make_prompt(o.cfg, 32);
    model.enqueue_prefill(prompt);
    int64_t tok = model.sample_prefill_result();
    std::vector<int64_t> warm;
    for (int i = 0; i < kWarmup; ++i) {
      model.enqueue_decode_step(tok);
      tok = model.poll_decode_sample();
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) {
      model.enqueue_decode_step(tok);
      tok = model.poll_decode_sample();
      ids.push_back(tok);
    }
    const auto t1 = std::chrono::steady_clock::now();
    wall_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                t1 - t0)
                                .count());
    model.reset_kv();
  }

  // Average KV length across the timed window: prompt rows + half the steps.
  const int avg_seq = 32 + steps / 2 + 1;
  const uint64_t bytes =
      model.decode_step_bytes(avg_seq) * static_cast<uint64_t>(steps);
  const double sec = wall_ns / 1e9;
  const double tok_per_s = steps / sec;
  const double achieved = bytes / sec;
  const double roofline_bw = 231.8e9;  // measured M0 read bandwidth
  const double ideal_sec =
      bytes /
      roofline_bw;  // naive traffic model at measured streaming BW
  const double ratio = ideal_sec / sec;

  std::printf("BENCH mode=%s steps=%d tok/s=%.2f step_ms=%.3f "
              "traffic=%.2fGB eff_bw=%.0fGB/s roofline_ratio=%.1f%%\n",
              use_rollout ? "rollout" : "sync", steps, tok_per_s,
              sec * 1000.0 / steps, bytes / 1e9, achieved / 1e9, ratio * 100.0);
  if (!ids.empty())
    DGPP_LOG_INFO("sample head: {} {} {} {}", ids[0],
                  ids.size() > 1 ? ids[1] : -1, ids.size() > 2 ? ids[2] : -1,
                  ids.size() > 3 ? ids[3] : -1);
  return ratio >= 0.90 ? 0 : 1;
}

int run_soak(const Options& o, int minutes) {
  Rig rig(o.cfg);
  DollModel model(rig.arena, rig.streams, rig.graphs, rig.gemm, o.cfg);
  model.init_weights();

  const auto prompt = make_prompt(o.cfg, 64);
  model.enqueue_prefill(prompt);
  int64_t tok = model.sample_prefill_result();

  auto base = rig.arena.stats();
  uint64_t alloc_calls_0 = 0, slab_mallocs_0 = 0, hw0 = 0;
  for (auto& [n, stp] : base) {
    alloc_calls_0 += stp->alloc_calls;
    slab_mallocs_0 += stp->slab_mallocs;
    hw0 += stp->high_water;
  }
  DGPP_LOG_INFO("soak baseline alloc_calls={} slabs={} high_water={}",
                alloc_calls_0, slab_mallocs_0, hw0);

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::minutes(minutes);
  uint64_t iters = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    for (int i = 0; i < 256; ++i) {
      model.enqueue_decode_step(tok);
      tok = model.poll_decode_sample();
      ++iters;
    }
    model.reset_kv();
    rig.arena.reset_scratch();
  }
  auto after = rig.arena.stats();
  uint64_t alloc_calls_1 = 0, slab_mallocs_1 = 0;
  for (auto& [name, stp] : after) {
    (void)name;
    alloc_calls_1 += stp->alloc_calls;
    slab_mallocs_1 += stp->slab_mallocs;
  }
  const bool stable_slabs = slab_mallocs_1 == slab_mallocs_0;
  std::printf("SOAK %dmin iters=%llu steady_alloc_growth=%llu slabs_stable=%s "
              "-> %s\n",
              minutes, static_cast<unsigned long long>(iters),
              static_cast<unsigned long long>(alloc_calls_1 - alloc_calls_0),
              stable_slabs ? "yes" : "no",
              stable_slabs ? "PASS" : "FAIL");
  return stable_slabs ? 0 : 1;
}

int run_gen(const Options& o, int argc, char** argv) {
  std::vector<int64_t> seed_tokens;
  for (int i = 1; i < argc; ++i) {
    char* endp = nullptr;
    long v = std::strtol(argv[i], &endp, 10);
    if (endp && *endp == '\0' && v >= 0) seed_tokens.push_back(v);
  }
  if (seed_tokens.empty()) seed_tokens = make_prompt(o.cfg, 4);
  Rig rig(o.cfg);
  DollModel model(rig.arena, rig.streams, rig.graphs, rig.gemm, o.cfg);
  model.init_weights();
  model.enqueue_prefill(seed_tokens);
  int64_t tok = model.sample_prefill_result();

  // Diagnostic: pull raw fp32 logit row plus upstream activation stats.
  {
    std::vector<float> lg(o.cfg.vocab);
    DGPP_CUDA_OK(cudaMemcpy(lg.data(), model.debug_logits_f32(),
                            sizeof(float) * o.cfg.vocab,
                            cudaMemcpyDeviceToHost));
    int best = 0;
    double sum = 0;
    float mx = -1e30f;
    for (int i = 0; i < o.cfg.vocab; ++i) {
      sum += lg[i];
      if (lg[i] > mx) { mx = lg[i]; best = i; }
    }
    std::printf("LOGITSProbe mean=%.6f max=%.6f argmax=%d l[alt]=%.6f\n",
                sum / o.cfg.vocab, mx, best, lg[best * 3 % o.cfg.vocab]);

    size_t stride = 0;
    const uint16_t* xd =
        static_cast<const uint16_t*>(model.debug_x_rows(&stride));
    std::vector<uint16_t> xr(o.cfg.hidden);
    DGPP_CUDA_OK(cudaMemcpy(xr.data(), xd, sizeof(uint16_t) * o.cfg.hidden,
                            cudaMemcpyDeviceToHost));
    double acc = 0;
    int nzx = 0, nnx = 0;
    for (int i = 0; i < o.cfg.hidden; ++i) {
      float v = dgpp::bf16_bits_to_float(xr[i]);
      acc += v * v;
      if (v != 0.f) ++nzx;
      if (std::isnan(v)) ++nnx;
    }
    std::printf(
        "XPROBE row0 rms=%.8f nz=%d/%d nan=%d\n", std::sqrt(acc / o.cfg.hidden),
        nzx, o.cfg.hidden, nnx);

    const uint16_t* nd =
        static_cast<const uint16_t*>(model.debug_normed_row());
    DGPP_CUDA_OK(cudaMemcpy(xr.data(), nd, sizeof(uint16_t) * o.cfg.hidden,
                            cudaMemcpyDeviceToHost));
    acc = 0;
    double acc8 = 0;
    int nnnan = 0;
    const uint8_t* a8d = static_cast<const uint8_t*>(model.debug_act8_row());
    std::vector<uint8_t> ar(o.cfg.hidden);
    DGPP_CUDA_OK(cudaMemcpy(ar.data(), a8d, o.cfg.hidden,
                            cudaMemcpyDeviceToHost));
    for (int i = 0; i < o.cfg.hidden; ++i) {
      float v = dgpp::bf16_bits_to_float(xr[i]);
      acc += v * v;
      float fv = dgpp::fp8_e4m3_bits_to_float(ar[i]);
      acc8 += fv * fv;
      if (std::isnan(fv)) ++nnnan;
    }
    std::printf("NORMPROBE rms=%.6f | ACT8 rms=%.6f nan=%d/%d\n",
                std::sqrt(acc / o.cfg.hidden), std::sqrt(acc8 / o.cfg.hidden),
                nnnan, o.cfg.hidden);

    const uint16_t* ed = static_cast<const uint16_t*>(model.debug_embed());
    DGPP_CUDA_OK(cudaMemcpy(xr.data(), ed + static_cast<size_t>(seed_tokens.front()) *
                                                     o.cfg.hidden,
                            sizeof(uint16_t) * o.cfg.hidden,
                            cudaMemcpyDeviceToHost));
    acc = 0;
    nzx = 0;
    for (int i = 0; i < o.cfg.hidden; ++i) {
      float v = dgpp::bf16_bits_to_float(xr[i]);
      acc += v * v;
      if (v != 0.f) ++nzx;
    }
    std::printf("EMBEDPROBE tok%lld rms=%.8f nz=%d\n",
                static_cast<long long>(seed_tokens.front()),
                std::sqrt(acc / o.cfg.hidden), nzx);
  }

  std::vector<int64_t> out{seed_tokens.begin(), seed_tokens.end()};
  out.push_back(tok);
  for (int i = 0; i < 96; ++i) {
    model.enqueue_decode_step(tok);
    tok = model.poll_decode_sample();
    out.push_back(tok);
  }
  std::printf("GEN");
  for (auto t : out) std::printf(" %lld", static_cast<long long>(t));
  std::printf("\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  try {
    Options o = parse(argc, argv);
    int device_count = 0;
    DGPP_CUDA_OK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
      DGPP_LOG_ERROR("no CUDA device present");
      return 2;
    }
    if (o.mode == "selftest") return run_selftest(o);
    if (o.mode == "bench")
      return run_bench(o, /*use_rollout=*/false,
                       collect_int_args_after_flags(argc, argv, "--bench", 400));
    if (o.mode == "bench-rollout")
      return run_bench(
          o, true,
          collect_int_args_after_flags(argc, argv, "--bench-rollout", 2000));
    if (o.mode == "gen") return run_gen(o, argc, argv);
    if (o.mode == "soak")
      return run_soak(
          o, collect_int_args_after_flags(argc, argv, "--soak-minutes", 10));
    DGPP_LOG_ERROR("unknown mode {}", o.mode);
    return 2;
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("fatal: {}", e.what());
    return 1;
  }
}