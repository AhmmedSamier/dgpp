// The Llama TP fold math without fabric (docs/zeta_plan.md): two
// world-2 ranks in one process over the tiny fixture, folded through a
// local rank-ordered fp32 sum (the fabric kernel's canonical chain for
// two ranks: bf16 loads, fp32 rank-0-first accumulation, one bf16
// rounding) — against the world-1 model as the oracle. Asserted:
//   * cross-rank: every layer's residual and the final read bitwise
//     identical on both ranks (both receive the same folded bytes);
//   * vs world 1: per-layer residuals within the numerics budget (the
//     sliced GEMMs reduce in a different order than the full widths),
//     the final read likewise, the top-1 per token equal or a certified
//     near tie.
// The verbs loopback (the transport the folds ride) is the fabric's own
// gate on a multi-Spark box; the loader's byte-exact slices at worlds
// 1/2/4/8 are pinned in llama_loader_test.
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "engine/boundary_reducer.hpp"
#include "models/llama/config.hpp"
#include "models/llama/forward.hpp"
#include "llama_fixture.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::LlamaModel;
using dgpp::LlamaTextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Two-participant fold exchange: each fold takes a monotonic turn. The
// first arrival publishes its partial's host copy and device pointer and
// waits until ITS turn is recorded as paired; the second arrival folds in
// canonical rank order and writes the sum back to both device buffers,
// then records the paired turn. The release is turn-based, not a boolean
// slot flag: the pairer's NEXT publish used to re-set the slot flag
// before the released rank's predicate re-check, sending it back to
// sleep on a foreign fold — both ranks then starved. Folds are globally
// sequential (each rank blocks in reduce until its fold completes), so
// one slot still suffices.
struct FoldExchange {
  std::mutex m;
  std::condition_variable cv;
  bool pending = false;
  uint64_t turn_counter = 0, pending_turn = 0, paired_turn = 0;
  int first_rank = -1;
  std::vector<uint16_t> first_host;
  uint16_t* first_dev = nullptr;
  int rows = 0, hidden = 0;
};

class LocalRankReducer final : public dgpp::BoundaryReducer {
 public:
  LocalRankReducer(FoldExchange* ex, int rank) : ex_(ex), rank_(rank) {}
  void reduce(uint16_t* partial, int rows, int hidden) override {
    const size_t n = static_cast<size_t>(rows) * hidden;
    std::vector<uint16_t> mine(n);
    DGPP_CUDA_OK(cudaMemcpy(mine.data(), partial, n * 2, cudaMemcpyDeviceToHost));
    std::unique_lock<std::mutex> lk(ex_->m);
    const uint64_t turn = ++ex_->turn_counter;
    if (!ex_->pending) {
      ex_->pending = true;
      ex_->pending_turn = turn;
      ex_->first_rank = rank_;
      ex_->first_host = std::move(mine);
      ex_->first_dev = partial;
      ex_->rows = rows;
      ex_->hidden = hidden;
      ex_->cv.wait(lk, [&] { return ex_->paired_turn >= turn; });
      return;
    }
    require(rows == ex_->rows && hidden == ex_->hidden, "fold shape drift between ranks");
    const std::vector<uint16_t>& a = (ex_->first_rank == 0) ? ex_->first_host : mine;
    const std::vector<uint16_t>& b = (ex_->first_rank == 0) ? mine : ex_->first_host;
    std::vector<uint16_t> sum(n);
    for (size_t i = 0; i < n; ++i)
      sum[i] = float_to_bf16_bits(bf16_bits_to_float(a[i]) + bf16_bits_to_float(b[i]));
    DGPP_CUDA_OK(cudaMemcpy(ex_->first_dev, sum.data(), n * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(partial, sum.data(), n * 2, cudaMemcpyHostToDevice));
    ex_->paired_turn = ex_->pending_turn;
    ex_->pending_turn = 0;
    ex_->pending = false;
    ex_->first_dev = nullptr;
    ex_->first_host.clear();
    lk.unlock();
    ex_->cv.notify_all();
  }

 private:
  FoldExchange* ex_;
  int rank_;
};

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

struct Stats {
  double l2 = 0, max_ulps = 0;
  long hard = 0, total = 0;
};

Stats compare_bf16(const uint16_t* got, const uint16_t* want, size_t n, int hard) {
  Stats s;
  s.total = static_cast<long>(n);
  double rms = 0;
  for (size_t i = 0; i < n; ++i) rms += std::pow(bf16_bits_to_float(want[i]), 2);
  rms = std::sqrt(rms / std::max<size_t>(n, 1));
  const double floor_abs = 0.02 * rms;
  double sum_d2 = 0, sum_o2 = 0;
  for (size_t i = 0; i < n; ++i) {
    const double g = bf16_bits_to_float(got[i]), w = bf16_bits_to_float(want[i]);
    int u = bf16_ulps(got[i], want[i]);
    if (std::fabs(g - w) <= floor_abs) u = 0;
    s.max_ulps = std::max(s.max_ulps, static_cast<double>(u));
    sum_d2 += (g - w) * (g - w);
    sum_o2 += w * w;
    if (u > hard) ++s.hard;
  }
  s.l2 = std::sqrt(sum_d2) / std::sqrt(sum_o2 + 1e-30);
  return s;
}

std::vector<int64_t> test_tokens(const LlamaTextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0x9E3779B9D7C11A5ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  return t;
}

struct RankOut {
  LlamaModel::Outputs out;
  std::string error;
};

void rank_forward(int rank, const LlamaTextConfig& cfg, const std::string& dir,
                  const std::vector<int64_t>& tokens, int T, FoldExchange* ex, RankOut* out) {
  try {
    LocalRankReducer reducer(ex, rank);
    LlamaModel model(cfg, dir, T, 256, dgpp::LlamaResidency::Streaming, &reducer, rank, 2, 1);
    out->out = model.forward(tokens, true);
  } catch (const std::exception& e) {
    out->error = "rank " + std::to_string(rank) + ": " + e.what();
  }
}

}  // namespace

DGPP_TEST(llama_tp_world_2_folds_match_world_1) {
  const std::string dir = (fs::current_path() / "llama_tp_fixture").string();
  const LlamaTextConfig cfg = llamafx::tiny_config();
  llamafx::write_fixture(cfg, dir);
  const int T = 40;
  const std::vector<int64_t> tokens = test_tokens(cfg, T);
  const int H = cfg.hidden_size, V = cfg.vocab_size;

  LlamaModel oracle(cfg, dir, T, 256);
  const LlamaModel::Outputs ref = oracle.forward(tokens, true);

  FoldExchange ex;
  RankOut outs[2];
  std::thread workers[2];
  for (int r = 0; r < 2; ++r)
    workers[r] = std::thread(rank_forward, r, std::cref(cfg), std::cref(dir), std::cref(tokens), T, &ex,
                             &outs[r]);
  for (int r = 0; r < 2; ++r) workers[r].join();
  for (int r = 0; r < 2; ++r) require(outs[r].error.empty(), outs[r].error);

  // Cross-rank: bitwise identical residuals and reads (both ranks
  // receive the same folded bytes). The logits are vocab-sharded: each
  // rank's slice must match the oracle's corresponding columns.
  require(outs[0].out.layer_states.size() == outs[1].out.layer_states.size(), "layer capture count");
  for (size_t l = 0; l < outs[0].out.layer_states.size(); ++l)
    require(outs[0].out.layer_states[l] == outs[1].out.layer_states[l], "rank residuals differ at layer " + std::to_string(l));
  require(outs[0].out.final_hidden_bits == outs[1].out.final_hidden_bits, "rank final reads differ");
  bool ok = true, ok_rank_head = true;
  for (int r = 0; r < 2; ++r) {
    const auto& o = outs[r].out;
    require(o.lm_vocab_count == V / 2 && o.lm_vocab_begin == r * (V / 2), "head slice geometry");
    // The head slice: the same K-width dot as the oracle's columns, so
    // near-bitwise; gate it on a tight relative band, not bits.
    double d2 = 0, o2 = 0;
    for (int t = 0; t < T; ++t) {
      const float* got = o.logits.data() + static_cast<size_t>(t) * o.lm_vocab_count;
      const float* want = ref.logits.data() + static_cast<size_t>(t) * V + o.lm_vocab_begin;
      for (int i = 0; i < o.lm_vocab_count; ++i) {
        d2 += std::pow(got[i] - want[i], 2);
        o2 += std::pow(want[i], 2);
      }
    }
    const double l2 = std::sqrt(d2) / std::sqrt(o2 + 1e-30);
    std::printf("[ .. ] rank %d head slice vs w1 columns: l2 %.3g\n", r, l2);
    if (l2 > 0.01) ok_rank_head = false;
  }

  // Vs world 1: the numerics budget and the top-1 rule. The top-1 runs
  // on the gathered full distribution (both rank slices concatenated).
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const Stats s = compare_bf16(outs[0].out.layer_states[l].data(), ref.layer_states[l].data(),
                                 static_cast<size_t>(T) * H, 128);
    std::printf("[ .. ] layer %d vs w1: l2 %.3g max %g ulps, hard %ld of %ld\n", l, s.l2, s.max_ulps,
                s.hard, s.total);
    if (s.l2 > 0.01 || static_cast<double>(s.hard) / s.total > 0.005) ok = false;
  }
  const Stats fs_ = compare_bf16(outs[0].out.final_hidden_bits.data(), ref.final_hidden_bits.data(),
                                 static_cast<size_t>(T) * H, 128);
  std::printf("[ .. ] final read vs w1: l2 %.3g max %g ulps, hard %ld of %ld\n", fs_.l2, fs_.max_ulps,
              fs_.hard, fs_.total);
  if (fs_.l2 > 0.01 || static_cast<double>(fs_.hard) / fs_.total > 0.005) ok = false;
  std::vector<float> gathered(static_cast<size_t>(T) * V);
  for (int t = 0; t < T; ++t)
    for (int r = 0; r < 2; ++r) {
      const auto& o = outs[r].out;
      std::memcpy(gathered.data() + static_cast<size_t>(t) * V + o.lm_vocab_begin,
                  o.logits.data() + static_cast<size_t>(t) * o.lm_vocab_count,
                  static_cast<size_t>(o.lm_vocab_count) * 4);
    }
  const auto got = LlamaModel::topk(gathered, T, V, 4);
  const auto want = LlamaModel::topk(ref.logits, T, V, 4);
  int hard = 0, soft = 0;
  for (int t = 0; t < T; ++t) {
    if (got[t][0].first == want[t][0].first) continue;
    const double margin = std::fabs(want[t][0].second - want[t][1].second) /
                          (std::fabs(want[t][0].second) + 1e-30);
    if (margin < 0.02) {
      ++soft;
      std::printf("[ .. ]   t%d near-tie: w2 %d vs w1 %d (margin %.4g)\n", t, got[t][0].first,
                  want[t][0].first, margin);
    } else {
      ++hard;
      std::printf("[ .. ]   t%d HARD: w2 %d (%g) vs w1 %d (%g), margin %.4g\n", t, got[t][0].first,
                  got[t][0].second, want[t][0].first, want[t][0].second, margin);
    }
  }
  std::printf("[ .. ] top-1 vs w1: %d hard, %d near-tie of %d rows\n", hard, soft, T);
  if (hard != 0) ok = false;
  if (!ok_rank_head) ok = false;
  require(ok, "world-2 folds differ from world 1 beyond the numerics budget");
}

int main() { return dgpp::test::run_all(); }
