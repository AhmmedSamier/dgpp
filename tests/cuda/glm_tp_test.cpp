// M5 deliverable 3, forward integration: tensor-parallel block-boundary
// parity, loopback (one process, real verbs QPs over 127.0.0.1 — the same
// transport the bus scenarios use). The TP fixture is geometry the TP=4
// world can shard: 4 KDA heads, 8 DSA heads, 8 routed experts, dense
// intermediate 200 (a partial 128-block on every scale grid — the slicing
// offset math's worst case), MoE intermediate 64. Values come from the
// shared fixture writer (glm_fixture.hpp, via glm_tp_fixture.cpp — g++,
// because nvcc cannot compile minijson's vector-of-incomplete Member).
//
// Oracles and assertions:
//   * world=1 engine forward (the M4 path, byte-identical code) is the
//     oracle: free-run final hidden + logits under tolerance, per-layer
//     ISOLATED parity (synthetic stream states replayed into every rank
//     and the oracle — the curated-suite discipline adapted: both sides
//     compute each layer from the SAME entering state, so drift is
//     bounded by one layer's floor), and router flips certified as
//     measured near ties (glm_route_audit).
//   * Cross-rank: final hidden, per-layer captures, route ids/weights,
//     and logits are BITWISE identical on every rank — the canonical
//     rank-order fold's guarantee (a violation means a rank's replicated
//     state diverged: the silent-corruption class).
//   * Chunking: 21 tokens over hidden 256 folds two collectives per
//     boundary (16-row slot + 5-row tail), exercising the row loop.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "common/test.hpp"
#include "models/glm_forward.hpp"
#include "models/glm_route_audit.hpp"
#include "models/glm_tp_bus.hpp"
#include "net/collective_bus.hpp"

#include "glm_rng.hpp"

// Fixture side (g++ TU): config + on-disk checkpoint generation.
dgpp::GlmTextConfig glm_tp_test_config();
void glm_tp_write_fixture(const std::string& dir);

using dgpp::GlmBusBoundaryReducer;
using dgpp::GlmDiagnosticModel;
using dgpp::GlmTextConfig;
using dgpp::glm_route::RouteFlipAudit;
using dgpp::glm_route::audit_route_flips;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

BusOptions loop_options(int rank, int world, uint16_t port) {
  BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.lane_devices = {"rocep1s0f0", "roceP2p1s0f0"};
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : "127.0.0.1";
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = 8;
  o.lat_slot_bytes = 8192;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = 5000;
  // Kernel deadline: a wedged peer must fail the collective in seconds,
  // not wedge the whole process for minutes (the 600s consumer default
  // is for long forwards under a healthy fabric).
  o.consumer_deadline_s = 20.0;
  o.launch_consumers = false;     // per-collective kernels own doorbells
  return o;
}

// Starts a loopback world of buses (rank 0 listens), or returns empty.
std::vector<std::unique_ptr<CollectiveBus>> start_world(int world,
                                                        uint16_t port) {
  std::vector<std::unique_ptr<CollectiveBus>> out;
  for (int r = 0; r < world; ++r)
    out.push_back(std::make_unique<CollectiveBus>(
        loop_options(r, world, port)));
  std::vector<std::string> errors(static_cast<size_t>(world));
  std::thread listener([&] {
    if (!out[0]->start(&errors[0]))
      DGPP_LOG_ERROR("tp world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(
              &errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("tp world rank {}: {}", r,
                       errors[static_cast<size_t>(r)]);
    });
  listener.join();
  for (auto& t : connectors) t.join();
  for (int r = 0; r < world; ++r)
    if (!errors[static_cast<size_t>(r)].empty()) return {};
  return out;
}

// One rank's work: construct the TP model over its bus, wait at the
// construction barrier (NO rank's forward may start until every rank's
// allocation-phase device syncs are done — a first collective spinning
// on doorbells while a peer constructs is the loopback deadlock the
// bring-up measured), then the free-run and isolated forwards.
// Everything returns home; the main thread asserts.
struct RankOutcome {
  std::string error;
  GlmDiagnosticModel::Outputs free_out;
  GlmDiagnosticModel::Outputs iso_out;
  std::vector<std::vector<uint16_t>> captures;
};

struct ConstructBarrier {
  std::mutex mu;
  std::condition_variable cv;
  int left = 0;  // ranks still constructing
  explicit ConstructBarrier(int world) : left(world) {}
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mu);
    if (--left == 0)
      cv.notify_all();
    else
      cv.wait(lock, [this] { return left == 0; });
  }
};

void rank_work(int rank, int world, const std::string& dir,
               const GlmTextConfig& cfg,
               const std::vector<int64_t>& tokens,
               const std::vector<const uint16_t*>& inputs,
               int64_t cache_tokens, CollectiveBus* bus,
               ConstructBarrier* barrier, RankOutcome* out) {
  // Arrive at the construction barrier exactly once, from whatever path
  // exits the construction phase — a second arrive_and_wait() (the naive
  // catch) decrements `left` below zero and strands every rank at the
  // barrier forever, swallowing the error with it.
  bool arrived = false;
  auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  try {
    const auto t0 = std::chrono::steady_clock::now();
    DGPP_LOG_INFO("rank {}: model ctor begin", rank);
    GlmBusBoundaryReducer reducer(*bus);
    GlmDiagnosticModel model(cfg, dir, static_cast<int>(tokens.size()),
                             cache_tokens, &reducer, rank, world);
    DGPP_LOG_INFO(
        "rank {}: model ctor done in {:.0f}ms", rank,
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0)
            .count());
    arrive_once();
    out->free_out = model.forward(tokens);
    out->iso_out = model.forward_isolated(tokens, inputs, out->captures);
  } catch (const std::exception& e) {
    // Loud: a swallowed rank error presents as an unexplained hang (the
    // peers block on a collective that will never be submitted).
    DGPP_LOG_ERROR("rank {} failed: {}", rank, e.what());
    out->error = std::string("rank ") + std::to_string(rank) + ": " + e.what();
    arrive_once();  // never strand peers at the barrier
  }
}

std::vector<int64_t> make_tokens(int n, int vocab) {
  glmrng::Rng rng(20260829);
  std::vector<int64_t> t(static_cast<size_t>(n));
  for (auto& id : t)
    id = static_cast<int64_t>(rng.next() % static_cast<uint64_t>(vocab));
  return t;
}

// Isolated-mode stream states: [num_layers+1][T, 4, hidden] bf16,
// normal-ish magnitudes (post-embedding + mHC compute scale).
std::vector<std::vector<uint16_t>> make_layer_states(int layers, int tokens,
                                                     int hidden) {
  std::vector<std::vector<uint16_t>> v(
      static_cast<size_t>(layers) + 1,
      std::vector<uint16_t>(static_cast<size_t>(tokens) * 4 * hidden));
  for (size_t l = 0; l < v.size(); ++l) {
    glmrng::Rng rng(0xC0FFEE + l);
    for (auto& b : v[l]) b = dgpp::float_to_bf16_bits(rng.normal3());
  }
  return v;
}

double l2_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  double acc = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = dgpp::bf16_bits_to_float(a[i]) -
                     dgpp::bf16_bits_to_float(b[i]);
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<double>(a.size()));
}

bool bits_equal(const std::vector<uint16_t>& a,
                const std::vector<uint16_t>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
}

// Runs one loopback TP world end to end and asserts the full parity
// surface against the supplied oracle references: cross-rank bitwise at
// every observable, free-run l2 + top-1 near-tie certification, per-layer
// isolated kept-row l2, and route-flip certification.
void check_world(int world, uint16_t port, const std::string& dir,
                 const GlmTextConfig& cfg,
                 const std::vector<int64_t>& tokens,
                 const std::vector<const uint16_t*>& state_ptrs,
                 int64_t cache, const GlmDiagnosticModel::Outputs& ref_free,
                 const std::vector<std::vector<uint16_t>>& ref_captures,
                 const char* what) {
  DGPP_LOG_INFO("TP loopback [{}] world={} tokens={}", what, world,
                tokens.size());
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(world, port);
  require(!buses.empty(), "tp bus world failed to start");

  std::vector<RankOutcome> ranks(static_cast<size_t>(world));
  ConstructBarrier barrier(world);
  std::vector<std::thread> workers;
  for (int r = 0; r < world; ++r)
    workers.emplace_back(rank_work, r, world, dir, std::cref(cfg),
                         std::cref(tokens), std::cref(state_ptrs), cache,
                         buses[static_cast<size_t>(r)].get(), &barrier,
                         &ranks[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < world; ++r)
    require(ranks[static_cast<size_t>(r)].error.empty(),
            ranks[static_cast<size_t>(r)].error);

  // ---- cross-rank: bitwise at every observable -------------------
  for (int r = 1; r < world; ++r) {
    const RankOutcome& a = ranks[0];
    const RankOutcome& b = ranks[static_cast<size_t>(r)];
    require(bits_equal(a.free_out.final_hidden_bits,
                       b.free_out.final_hidden_bits),
            "free-run final hidden differs bitwise across ranks");
    require(bits_equal(a.free_out.logits_bits, b.free_out.logits_bits),
            "logits differ bitwise across ranks");
    require(a.captures.size() == b.captures.size() &&
                a.captures.size() == ref_captures.size(),
            "capture count mismatch");
    for (size_t l = 0; l < a.captures.size(); ++l)
      require(bits_equal(a.captures[l], b.captures[l]),
              "per-layer stream state differs bitwise across ranks");
    require(a.free_out.routes.size() == b.free_out.routes.size(),
            "route layer count mismatch");
    for (size_t l = 0; l < a.free_out.routes.size(); ++l) {
      require(a.free_out.routes[l].ids == b.free_out.routes[l].ids &&
                  a.free_out.routes[l].weights ==
                      b.free_out.routes[l].weights,
              "routing decisions differ across ranks");
    }
  }

  // ---- vs oracle: free-run end-to-end -----------------------------
  // Free-run divergence is REPORTED, not asserted (the M4 discipline:
  // cross-implementation bf16 at depth compounds legitimately — the
  // per-layer ISOLATED bound below is the assertion surface). The head
  // keeps a hard gate: every top-1 disagreement must certify as a
  // boundary near tie (oracle top-2 margin within 32x the measured
  // TP-vs-oracle logit noise).
  const double free_l2 = l2_bf16(ranks[0].free_out.final_hidden_bits,
                                 ref_free.final_hidden_bits);
  int top1_mismatch = 0, top1_uncertified = 0;
  double worst_margin_ratio = 0;
  {
    const int V = cfg.vocab_size;
    const size_t T = tokens.size();
    const uint16_t* tp = ranks[0].free_out.logits_bits.data();
    const uint16_t* rf = ref_free.logits_bits.data();
    for (size_t t = 0; t < T; ++t) {
      const uint16_t* tr = tp + t * V;
      const uint16_t* rr = rf + t * V;
      int tp_top = 0, rf_top = 0, rf_second = -1;
      double noise = 0;
      for (int c = 0; c < V; ++c) {
        const double d = std::abs(dgpp::bf16_bits_to_float(tr[c]) -
                                  dgpp::bf16_bits_to_float(rr[c]));
        noise = std::max(noise, d);
        if (dgpp::bf16_bits_to_float(tr[c]) >
            dgpp::bf16_bits_to_float(tr[tp_top]))
          tp_top = c;
        if (dgpp::bf16_bits_to_float(rr[c]) >
            dgpp::bf16_bits_to_float(rr[rf_top])) {
          rf_second = rf_top;
          rf_top = c;
        } else if (rf_second < 0 ||
                   dgpp::bf16_bits_to_float(rr[c]) >
                       dgpp::bf16_bits_to_float(rr[rf_second])) {
          if (c != rf_top) rf_second = c;
        }
      }
      if (tp_top == rf_top) continue;
      ++top1_mismatch;
      const double margin = std::abs(
          dgpp::bf16_bits_to_float(rr[rf_top]) -
          dgpp::bf16_bits_to_float(rr[rf_second]));
      worst_margin_ratio = std::max(worst_margin_ratio, margin / noise);
      if (noise <= 0 || margin > 32.0 * noise) ++top1_uncertified;
    }
  }
  require(top1_uncertified == 0,
          "free-run top-1 disagreement is not a certified near tie "
          "(oracle top-2 margin > 32x TP-vs-oracle logit noise)");

  // ---- vs oracle: per-layer isolated parity (the assertion surface) --
  // Kept-row discipline: a token whose routing flipped vs the oracle
  // legitimately moves O(1) (the noaux bias ties scores at the selection
  // boundary); its rows are excluded from the l2 and the flips are
  // certified separately below. Kept rows must sit under 2e-2 — the
  // M4 curated-suite tier — and measured an order lower.
  double worst_layer_l2 = 0;
  {
    const size_t T = tokens.size();
    const size_t row_elems =
        static_cast<size_t>(cfg.hidden_size) * 4;  // [4, hidden] per token
    for (size_t l = 0; l < ranks[0].captures.size(); ++l) {
      // Which route belongs to this layer (MoE layers only)?
      const dgpp::GlmRouteTraceLayer* eng_route = nullptr;
      const dgpp::GlmRouteTraceLayer* ref_route = nullptr;
      for (size_t ri = 0; ri < ranks[0].free_out.routes.size(); ++ri)
        if (ranks[0].free_out.routes[ri].layer_idx == l) {
          eng_route = &ranks[0].free_out.routes[ri];
          ref_route = &ref_free.routes[ri];
          break;
        }
      int kept_rows = 0, flipped_rows = 0;
      double acc = 0;
      for (size_t t = 0; t < T; ++t) {
        const bool flipped =
            eng_route &&
            !std::equal(eng_route->ids.begin() + t * eng_route->top_k,
                        eng_route->ids.begin() + (t + 1) * eng_route->top_k,
                        ref_route->ids.begin() + t * ref_route->top_k);
        if (flipped) {
          ++flipped_rows;
          continue;
        }
        ++kept_rows;
        const uint16_t* a =
            ranks[0].captures[l].data() + t * row_elems;
        const uint16_t* b = ref_captures[l].data() + t * row_elems;
        for (size_t i = 0; i < row_elems; ++i) {
          const double d = dgpp::bf16_bits_to_float(a[i]) -
                            dgpp::bf16_bits_to_float(b[i]);
          acc += d * d;
        }
      }
      const double l2 = kept_rows
                            ? std::sqrt(acc / (double(kept_rows) * row_elems))
                            : 0.0;
      worst_layer_l2 = std::max(worst_layer_l2, l2);
      require(l2 < 0.02,
              "isolated kept-row l2 vs oracle (layer " + std::to_string(l) +
                  ")");
      if (flipped_rows)
        DGPP_LOG_INFO("TP [{}] world={} layer {} isolated: {} flipped rows "
                      "excluded (certified below)",
                      what, world, l, flipped_rows);
    }
  }

  // ---- vs oracle: routing (flips must be certified near ties) ------
  int64_t flips = 0, swaps = 0;
  double worst_noise_mult = 0;
  size_t oi = 0;  // oracle route index (MoE layers only, ascending)
  for (size_t l = 0; l < ranks[0].free_out.routes.size(); ++l) {
    const auto& eng = ranks[0].free_out.routes[l];
    const auto& ref_route = ref_free.routes[oi];
    require(ref_route.layer_idx == eng.layer_idx,
            "route layer alignment vs oracle");
    const std::vector<float>& eng_biased =
        ranks[0].free_out.route_biased[l];
    const std::vector<float>& ref_biased = ref_free.route_biased[oi];
    RouteFlipAudit audit;
    audit_route_flips(eng.ids.data(), ref_route.ids.data(),
                      eng_biased.data(), ref_biased.data(),
                      static_cast<int64_t>(eng.tokens),
                      cfg.moe_config().top_k, cfg.moe_config().n_experts,
                      audit, eng.layer_idx);
    flips += audit.tokens_flipped;
    swaps += audit.swaps_certified;
    worst_noise_mult = std::max(worst_noise_mult, audit.max_noise_multiple);
    ++oi;
  }

  DGPP_LOG_INFO(
      "TP [{}] world={} free l2={:.4f} worst layer l2={:.4f} top1 misses={} "
      "(uncertified {}, worst margin {:.1f}x noise) route flips={} "
      "(certified swaps {}, worst {:.1f}x noise)",
      what, world, free_l2, worst_layer_l2, top1_mismatch, top1_uncertified,
      worst_margin_ratio, flips, swaps, worst_noise_mult);

  for (auto& b : buses) b->stop();
}

// Oracle references for one token shape: the world=1 model's free-run and
// isolated outputs (with per-layer captures), determinism double-checked.
struct OracleRef {
  GlmDiagnosticModel::Outputs free_out;
  std::vector<std::vector<uint16_t>> captures;
};

OracleRef run_oracle(const GlmTextConfig& cfg, const std::string& dir,
                     const std::vector<int64_t>& tokens,
                     const std::vector<const uint16_t*>& state_ptrs,
                     int64_t cache) {
  OracleRef ref;
  GlmDiagnosticModel oracle(cfg, dir, static_cast<int>(tokens.size()), cache);
  ref.free_out = oracle.forward(tokens);
  const GlmDiagnosticModel::Outputs again = oracle.forward(tokens);
  require(ref.free_out.final_hidden_bits == again.final_hidden_bits &&
              ref.free_out.logits_bits == again.logits_bits,
          "oracle forward is not deterministic across calls");
  oracle.forward_isolated(tokens, state_ptrs, ref.captures);
  return ref;
}

DGPP_TEST(glm_tp_forward_parity_loopback) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);

  // ---- decode-shaped case (T=8): every boundary takes the pre-staged
  // seam — stage() hands the pinned send source to the producing GEMM
  // (8 x 256 = 2048 elems, one latency slot), world 2 exercises the
  // zero-copy path and the GEMM-writes-pinned contract.
  {
    const std::vector<int64_t> tokens = make_tokens(8, cfg.vocab_size);
    const std::vector<std::vector<uint16_t>> states = make_layer_states(
        cfg.num_hidden_layers, static_cast<int>(tokens.size()),
        cfg.hidden_size);
    std::vector<const uint16_t*> state_ptrs;
    for (const auto& s : states) state_ptrs.push_back(s.data());
    const OracleRef ref =
        run_oracle(cfg, dir, tokens, state_ptrs, 128);
    check_world(2, 29903, dir, cfg, tokens, state_ptrs, 128, ref.free_out,
                ref.captures, "staged");
  }

  // ---- the 21-token case: 5376 elems per boundary — above one latency
  // slot, so boundaries stay on the chunked device path at worlds 2 and 4
  // (the committed baseline).
  {
    const std::vector<int64_t> tokens = make_tokens(21, cfg.vocab_size);
    const std::vector<std::vector<uint16_t>> states = make_layer_states(
        cfg.num_hidden_layers, static_cast<int>(tokens.size()),
        cfg.hidden_size);
    std::vector<const uint16_t*> state_ptrs;
    for (const auto& s : states) state_ptrs.push_back(s.data());
    const OracleRef ref =
        run_oracle(cfg, dir, tokens, state_ptrs, 128);
    check_world(2, 29899, dir, cfg, tokens, state_ptrs, 128, ref.free_out,
                ref.captures, "chunked");
    check_world(4, 29900, dir, cfg, tokens, state_ptrs, 128, ref.free_out,
                ref.captures, "chunked");
  }
}

}  // namespace

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    DGPP_LOG_INFO("no CUDA device visible; skipping glm_tp_test");
    return 2;
  }
  return dgpp::test::run_all();
}
