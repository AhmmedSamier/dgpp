// M5 deliverable 3, forward integration: tensor-parallel block-boundary
// parity, loopback (one process, real verbs QPs over 127.0.0.1 — the same
// transport the bus scenarios use). The TP fixture is geometry the TP=4
// world can shard: 4 KDA heads, 8 DSA heads, 8 routed experts, inter dims
// 512 (128-multiple quotients at worlds 2 and 4 — the quantized
// scale-grid slice contract; misaligned starts throw, and the negative
// test below pins that). Values come from the shared fixture writer
// (glm_fixture.hpp, via glm_tp_fixture.cpp — g++, because nvcc cannot
// compile minijson's vector-of-incomplete Member).
//
// Oracles and assertions:
//   * world=1 engine forward (the M4 path, byte-identical code) is the
//     oracle: free-run final hidden + logits under tolerance, per-layer
//     ISOLATED parity (synthetic stream states replayed into every rank
//     and the oracle — the curated-suite discipline adapted: both sides
//     compute each layer from the SAME entering state, so drift is
//     bounded by one layer's floor), and router flips certified as
//     measured near ties (glm_route_audit).
//   * ISOLATED parity asserts BOTH surfaces per layer: the mHC stream
//     snapshots AND the raw block-boundary folds (attn + FFN). The
//     stream snapshots pass through the mHC stream update, whose mixing
//     coefficients attenuate boundary errors ~300x — the dense
//     scale-grid slice bug measured 0.30 l2-wrong at the fold while the
//     stream-state reading stayed at 0.001, silently under the 0.02
//     budget. The folds are the unattenuated surface; both are asserted.
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
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "common/test.hpp"
#include "loaders/hf_cache.hpp"
#include "models/dsa_geometry.hpp"
#include "models/glm_forward.hpp"
#include "models/glm_route_audit.hpp"
#include "models/glm_sampler.hpp"
#include "models/glm_tp.hpp"
#include "models/glm_tp_parity.hpp"
#include "models/glm_tp_bus.hpp"
#include "models/kda_geometry.hpp"
#include "net/collective_bus.hpp"

#include "glm_rng.hpp"

// Fixture side (g++ TU): config + on-disk checkpoint generation.
dgpp::GlmTextConfig glm_tp_test_config();
void glm_tp_write_fixture(const std::string& dir);

using dgpp::GlmBusBoundaryReducer;
using dgpp::GlmDiagnosticModel;
using dgpp::GlmHeadSharding;
using dgpp::GlmResidency;
using dgpp::GlmLayerBound;
using dgpp::bf16_bits_to_float;
using dgpp::bus_greedy_pick;
using dgpp::glm_sample::Candidate;
using dgpp::glm_sample::local_max;
using dgpp::GlmLayerResident;
using dgpp::GlmLayerStream;
using dgpp::GlmMlpKind;
using dgpp::GlmShardParityReport;
using dgpp::GlmTextConfig;
using dgpp::glm_route::RouteFlipAudit;
using dgpp::glm_route::Top1AuditSummary;
using dgpp::glm_route::audit_route_flips;
using dgpp::glm_route::audit_routes_cascade;
using dgpp::glm_route::audit_top1_near_ties;
using dgpp::glm_route::certify_top1_near_ties;
using dgpp::glm_route::l2_bf16;
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
  // Kernel deadline: a wedged peer must fail the collective in seconds,
  // not wedge the whole process for minutes (the 600s consumer default
  // is for long forwards under a healthy fabric). Overridable because
  // sanitizer builds slow the CPU submission paths 10-50x — a release-
  // tuned 5s no-progress budget flakes at world=4 under UBSan (measured:
  // 1 fail in 3 runs, a rank falling behind, not a bus defect).
  o.completion_timeout_ms = [] {
    const char* ms = std::getenv("DGPP_TEST_BUS_TIMEOUT_MS");
    return ms ? std::atoi(ms) : 5000;
  }();
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
  // 2*num_layers post-fold boundary outputs (attn, FFN per layer) from the
  // isolated forward — the unattenuated assertion surface.
  std::vector<std::vector<uint16_t>> boundary;
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
    out->iso_out =
        model.forward_isolated(tokens, inputs, out->captures, &out->boundary);
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

bool bits_equal(const std::vector<uint16_t>& a,
                const std::vector<uint16_t>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
}

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
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
                  const GlmDiagnosticModel::Outputs& ref_iso,
                  const std::vector<std::vector<uint16_t>>& ref_captures,
                  const std::vector<std::vector<uint16_t>>& ref_boundary,
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
    require(bits_equal(a.free_out.logits, b.free_out.logits),
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

  // Optional output dump (DGPP_TP_DUMP_DIR): the free-run observables as
  // files, exactly the fabric runner's {out}.* names — the offline
  // instruments (bitwise compares vs fabric runs, rel_l2 tables) work on
  // one artifact format then. When the boundary capture is present, both
  // sides' folds go out too ({what}.fold.{attn|ffn}.L{N}.{tp|ref}.bf16):
  // the fold l2 tables above summarize what happened, the dumps are what
  // the ulp-level forensics (|d| vs |oracle| per element) run on.
  if (const char* dump_dir = std::getenv("DGPP_TP_DUMP_DIR")) {
    const std::string p = std::string(dump_dir) + "/" + what;
    const auto dump = [&](const std::string& suffix,
                          const std::vector<uint16_t>& bits) {
      std::FILE* f = std::fopen((p + suffix).c_str(), "wb");
      if (!f) throw std::runtime_error("cannot write " + p + suffix);
      if (std::fwrite(bits.data(), 2, bits.size(), f) != bits.size())
        throw std::runtime_error("short write to " + p + suffix);
      std::fclose(f);
    };
    dump(".final_hidden.bf16", ranks[0].free_out.final_hidden_bits);
    {
      const std::vector<float>& lg = ranks[0].free_out.logits;
      std::FILE* f = std::fopen((p + ".logits.f32").c_str(), "wb");
      if (!f) throw std::runtime_error("cannot write " + p + ".logits.f32");
      if (std::fwrite(lg.data(), sizeof(float), lg.size(), f) != lg.size())
        throw std::runtime_error("short write to " + p + ".logits.f32");
      std::fclose(f);
    }
    const bool have_folds = ranks[0].boundary.size() ==
                                2 * static_cast<size_t>(cfg.num_hidden_layers) &&
                            ref_boundary.size() == ranks[0].boundary.size();
    if (have_folds)
      for (int l = 0; l < cfg.num_hidden_layers; ++l)
        for (int site = 0; site < 2; ++site) {
          const char* sname = site == 0 ? "attn" : "ffn";
          dump(".fold." + std::string(sname) + ".L" + std::to_string(l) +
                   ".tp.bf16",
               ranks[0].boundary[2 * l + site]);
          dump(".fold." + std::string(sname) + ".L" + std::to_string(l) +
                   ".ref.bf16",
               ref_boundary[2 * l + site]);
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
  const Top1AuditSummary top1 = audit_top1_near_ties(
      ranks[0].free_out.logits.data(), ref_free.logits.data(),
      cfg.vocab_size, tokens.size());
  certify_top1_near_ties(top1, tokens.size(), what);
  // Reported IMMEDIATELY (the parity sections below can throw before the
  // end-of-run summary, and the free-run numbers are the hunt's context:
  // flip compounding is the designated non-assertion surface, so its size
  // must survive every failure path).
  DGPP_LOG_INFO(
      "TP [{}] world={} FREE-RUN: final_hidden l2={:.6f}, top-1 misses={}/{} "
      "(all certified, worst margin {:.1f}x noise)",
      what, world, free_l2, top1.misses, tokens.size(),
      top1.worst_margin_ratio);

  // ---- vs oracle: per-layer isolated parity (the assertion surface) --
  // Kept-row discipline: a token whose routing flipped vs the oracle
  // legitimately moves O(1) (the noaux bias ties scores at the selection
  // boundary); its rows are excluded from the l2 and the flips are
  // certified separately below. The routes compared are the ISOLATED
  // forward's OWN (TP-isolated vs oracle-isolated) — the routing the
  // compared layers actually executed — NOT the free-run's: the isolated
  // streams pass through each layer's attention fold first, whose
  // cross-implementation noise flips dense near-ties inside the isolated
  // forward at real dims, and the free-run filter misread those flips as
  // kept rows (the 0.57 hunt: layers 3/39 "folds" were one-expert
  // reroutes on two tokens, uniform ~0.08 additive — not compute
  // divergence).
  // Failures COLLECT (with values) and assert at the end: one run must
  // yield the whole per-layer table — a first-throw here cost the hunt
  // the numbers behind every layer after the first miss.
  const auto find_route =
      [](const std::vector<dgpp::GlmRouteTraceLayer>& routes,
         uint32_t layer) -> const dgpp::GlmRouteTraceLayer* {
    for (const auto& r : routes)
      if (r.layer_idx == layer) return &r;
    return nullptr;
  };
  double worst_layer_l2 = 0;
  std::vector<std::string> parity_failures;
  {
    const size_t T = tokens.size();
    const size_t row_elems =
        static_cast<size_t>(cfg.hidden_size) * 4;  // [4, hidden] per token
    for (size_t l = 0; l < ranks[0].captures.size(); ++l) {
      // captures[l] is the stream state AFTER layer l-1 (index 0 is the
      // initial state); the row filter must therefore pair it with
      // layer (l-1)'s routing — the route that produced it. The loop
      // originally paired captures[l] with route[l], one layer off:
      // invisible at fixture (isolated flips never happened there), but
      // at real dims it kept flip-affected rows at the capture following
      // a flipping layer (layers 40/43/44's "failures" were layers
      // 39/42/43's certified flips, mis-paired).
      const uint32_t src_layer =
          l == 0 ? 0 : static_cast<uint32_t>(l) - 1;
      // Which route did the ISOLATED forward execute for this layer
      // (MoE layers only)? l == 0 is the initial state — definitionally
      // identical, no route pairs with it at all.
      const dgpp::GlmRouteTraceLayer* eng_route =
          l == 0 ? nullptr
                 : find_route(ranks[0].iso_out.routes, src_layer);
      const dgpp::GlmRouteTraceLayer* ref_route =
          l == 0 ? nullptr : find_route(ref_iso.routes, src_layer);
      // A MoE layer must carry its route on BOTH sides or neither — a
      // one-sided route is a bookkeeping bug (and used to be a null
      // deref: the short-circuit below only guards eng_route).
      require(!eng_route || ref_route,
              "isolated route present on TP side only (layer " +
                  std::to_string(l) + ")");
      require(!ref_route || eng_route,
              "isolated route present on oracle side only (layer " +
                  std::to_string(l) + ")");
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
      DGPP_LOG_INFO(
          "TP [{}] world={} capture {} (post-layer {}): kept-row l2={:.6f} "
          "({} kept, {} flipped rows excluded, certified below)",
          what, world, l, src_layer, l2, kept_rows, flipped_rows);
      if (l2 >= 0.02)
        parity_failures.push_back(
            "isolated kept-row l2 vs oracle (capture after layer " +
            std::to_string(src_layer) + ") = " + std::to_string(l2));
    }
  }

  // ---- vs oracle: the raw boundary folds (the unattenuated surface) --
  // The stream snapshots above pass through the mHC stream update, whose
  // mixing coefficients compress boundary errors below the budget (the
  // dense scale-grid slice bug measured 0.30 at the fold, 0.001 in the
  // stream state). The folds themselves carry no such attenuation: the
  // attention fold compares all rows (routing-independent), the FFN fold
  // keeps the kept-row discipline (a flipped route legitimately moves the
  // FFN output O(1) for that token).
  double worst_fold_l2 = 0;
  {
    const size_t T = tokens.size();
    const size_t row_elems = static_cast<size_t>(cfg.hidden_size);
    require(ranks[0].boundary.size() ==
                    2 * static_cast<size_t>(cfg.num_hidden_layers) &&
                ref_boundary.size() ==
                    2 * static_cast<size_t>(cfg.num_hidden_layers),
            "boundary fold capture count mismatch");
    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
      // The ISOLATED forward's routing (same discipline as the capture
      // loop above — the free-run's is a different forward).
      const dgpp::GlmRouteTraceLayer* eng_route =
          find_route(ranks[0].iso_out.routes, static_cast<uint32_t>(l));
      const dgpp::GlmRouteTraceLayer* ref_route =
          find_route(ref_iso.routes, static_cast<uint32_t>(l));
      for (int site = 0; site < 2; ++site) {
        const bool is_ffn = site == 1;
        const uint16_t* a = ranks[0].boundary[2 * l + site].data();
        const uint16_t* b = ref_boundary[2 * l + site].data();
        // Same presence discipline as the capture loop: both sides or
        // neither — a one-sided route must fail loudly, never deref.
        require(!eng_route || ref_route,
                "fold route present on TP side only (layer " +
                    std::to_string(l) + ")");
        require(!ref_route || eng_route,
                "fold route present on oracle side only (layer " +
                    std::to_string(l) + ")");
        int kept = 0;
        double acc = 0;
        for (size_t t = 0; t < T; ++t) {
          const bool flipped =
              is_ffn && eng_route &&
              !std::equal(eng_route->ids.begin() + t * eng_route->top_k,
                          eng_route->ids.begin() + (t + 1) * eng_route->top_k,
                          ref_route->ids.begin() + t * ref_route->top_k);
          if (flipped) continue;
          ++kept;
          for (size_t i = 0; i < row_elems; ++i) {
            const double d = dgpp::bf16_bits_to_float(a[t * row_elems + i]) -
                             dgpp::bf16_bits_to_float(b[t * row_elems + i]);
            acc += d * d;
          }
        }
        const double l2 =
            kept ? std::sqrt(acc / (double(kept) * row_elems)) : 0.0;
        worst_fold_l2 = std::max(worst_fold_l2, l2);
        // Per-token distribution: cross-implementation bf16 regrouping
        // noise concentrates on cancellation-heavy tokens (output much
        // smaller than its contributions), while a systematic slicing/
        // kernel bug is uniform. The tail (max token) is the noise
        // signature; the median says whether the whole fold moved.
        std::vector<double> per_token;
        for (size_t t = 0; t < T; ++t) {
          const bool flipped =
              is_ffn && eng_route &&
              !std::equal(eng_route->ids.begin() + t * eng_route->top_k,
                          eng_route->ids.begin() + (t + 1) * eng_route->top_k,
                          ref_route->ids.begin() + t * ref_route->top_k);
          if (flipped) continue;
          double ta = 0;
          for (size_t i = 0; i < row_elems; ++i) {
            const double d = dgpp::bf16_bits_to_float(a[t * row_elems + i]) -
                             dgpp::bf16_bits_to_float(b[t * row_elems + i]);
            ta += d * d;
          }
          per_token.push_back(std::sqrt(ta / row_elems));
        }
        std::sort(per_token.begin(), per_token.end());
        DGPP_LOG_INFO(
            "TP [{}] world={} layer {} {} fold: kept-row l2={:.6f} "
            "(token l2 min/med/max = {:.6f}/{:.6f}/{:.6f}, n={})",
            what, world, l, is_ffn ? "ffn" : "attn", l2,
            per_token.empty() ? 0.0 : per_token.front(),
            per_token.empty()
                ? 0.0
                : per_token[per_token.size() / 2],
            per_token.empty() ? 0.0 : per_token.back(), per_token.size());
        if (l2 >= 0.02)
          parity_failures.push_back(
              std::string("boundary fold l2 vs oracle layer ") +
              std::to_string(l) + (is_ffn ? " (ffn)" : " (attn)") + " = " +
              std::to_string(l2));
      }
    }
  }
  require(parity_failures.empty(), [&] {
    std::string joined = "parity surface vs oracle (" +
                         std::to_string(parity_failures.size()) +
                         " failing layers):";
    for (const auto& f : parity_failures) joined += "\n  " + f;
    return joined;
  }());

  // ---- vs oracle: routing (the cascade discipline) ------------------
  // Both surfaces: the FREE-RUN routes (whole-trajectory view) and the
  // ISOLATED routes (the kept-row exclusions' own certification). Every
  // token's FIRST divergence must certify; later divergences of an
  // already-flipped token are consequences (reported). The isolated
  // audit is the parity gate's routing assertion; the free-run audit is
  // the same discipline on the compounded trajectory.
  audit_routes_cascade(ranks[0].free_out.routes,
                       ranks[0].free_out.route_biased, ref_free.routes,
                       ref_free.route_biased, cfg.moe_config().top_k,
                       cfg.moe_config().n_experts, what);
  {
    const std::string iso_label =
        std::string(what) + "/isolated";
    audit_routes_cascade(ranks[0].iso_out.routes,
                         ranks[0].iso_out.route_biased, ref_iso.routes,
                         ref_iso.route_biased, cfg.moe_config().top_k,
                         cfg.moe_config().n_experts, iso_label.c_str());
  }

  DGPP_LOG_INFO(
      "TP [{}] world={} free l2={:.4f} worst layer l2={:.4f} worst fold "
      "l2={:.4f} top1 misses={} (uncertified {}, worst margin {:.1f}x "
      "noise) — route audits above (cascade discipline)",
      what, world, free_l2, worst_layer_l2, worst_fold_l2, top1.misses,
      top1.uncertified, top1.worst_margin_ratio);

  for (auto& b : buses) b->stop();
}

// Oracle references for one token shape: the world=1 model's free-run and
// isolated outputs (with per-layer captures and boundary folds),
// determinism double-checked. iso_out carries the ISOLATED forward's own
// routing — the kept-row discipline's flip filter must compare the
// routing the isolated layers ACTUALLY EXECUTED (TP-isolated vs
// oracle-isolated), not the free-run's: the isolated streams enter each
// layer from the replayed reference state but pass through that layer's
// own attention FIRST, and the attention fold's ~1e-3-rms cross-
// implementation noise flips DENSE router near-ties inside the isolated
// forward at real dims (288 sigmoid-scored experts, top-8). The
// free-run-route filter misread those flips as kept rows — the 0.57
// hunt's layers 3/39 "failures" were one-expert reroutes, not compute
// divergence (the fold error vectors measured a uniform ~0.08 additive
// per element: exactly one expert's weighted contribution, and
// uncorrelated across the two affected tokens).
struct OracleRef {
  GlmDiagnosticModel::Outputs free_out;
  GlmDiagnosticModel::Outputs iso_out;
  std::vector<std::vector<uint16_t>> captures;
  std::vector<std::vector<uint16_t>> boundary;
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
              ref.free_out.logits == again.logits,
          "oracle forward is not deterministic across calls");
  ref.iso_out =
      oracle.forward_isolated(tokens, state_ptrs, ref.captures, &ref.boundary);
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
                ref.iso_out, ref.captures, ref.boundary, "staged");
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
                ref.iso_out, ref.captures, ref.boundary, "chunked");
    check_world(4, 29900, dir, cfg, tokens, state_ptrs, 128, ref.free_out,
                ref.iso_out, ref.captures, ref.boundary, "chunked");
  }
}

// ---- M6 d3: the vocabulary-sharded lm head ---------------------------------
// The serving seam: a VocabSharded model computes ONLY its slice of the
// logits, and that slice must equal the replicated head's columns
// BITWISE — same weight rows enter the same K-reduction, so the
// sampling merge inherits exactness instead of inheriting drift. Pinned
// at world 1 (the degenerate slice = the full vocab) and at world 4 over
// the loopback bus (the deployment geometry, where the slice is 1/4 of
// the vocab and its bounds are rank-dependent).
DGPP_TEST(glm_tp_head_shard_parity) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> tokens = make_tokens(8, cfg.vocab_size);

  // ---- world 1: sharded degenerates to the full vocab, bitwise.
  {
    const GlmDiagnosticModel::Outputs full =
        GlmDiagnosticModel(cfg, dir, 8, 128).forward(tokens);
    const GlmDiagnosticModel::Outputs shard =
        GlmDiagnosticModel(cfg, dir, 8, 128, nullptr, 0, 1,
                           GlmResidency::Streaming,
                           GlmHeadSharding::VocabSharded)
            .forward(tokens);
    require(full.lm_vocab_begin == 0 &&
                full.lm_vocab_count == cfg.vocab_size,
            "full head must report [0, vocab)");
    require(shard.lm_vocab_begin == 0 &&
                shard.lm_vocab_count == cfg.vocab_size,
            "world-1 sharded slice must be the full vocab");
    require(bits_equal(full.logits, shard.logits),
            "world-1 sharded head drifted from the full head");
  }

  // ---- world 4: each rank's slice == the full-head model's columns.
  constexpr int kWorld = 4;
  constexpr uint16_t kPort = 29914;
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort);
  require(!buses.empty(), "tp bus world failed to start");

  std::vector<std::string> errors(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  workers.reserve(kWorld);
  for (int r = 0; r < kWorld; ++r) {
    workers.emplace_back([&, r] {
      bool arrived = false;
      const auto arrive_once = [&] {
        if (arrived) return;
        arrived = true;
        barrier.arrive_and_wait();
      };
      try {
        GlmBusBoundaryReducer reducer(*buses[static_cast<size_t>(r)]);
        // Two models per rank, constructed back to back so the barrier
        // still guards every allocation phase; forwards run after (the
        // same order on every rank, so collectives stay aligned).
        GlmDiagnosticModel full(cfg, dir, 8, 128, &reducer, r, kWorld);
        GlmDiagnosticModel shard(cfg, dir, 8, 128, &reducer, r, kWorld,
                                 GlmResidency::Streaming,
                                 GlmHeadSharding::VocabSharded);
        arrive_once();
        const GlmDiagnosticModel::Outputs a = full.forward(tokens);
        const GlmDiagnosticModel::Outputs b = shard.forward(tokens);
        // Head placement cannot touch the pre-head computation.
        require(bits_equal(a.final_hidden_bits, b.final_hidden_bits),
                "final hidden differs between head modes");
        const int begin =
            static_cast<int>(static_cast<int64_t>(cfg.vocab_size) * r /
                             kWorld);
        const int count = static_cast<int>(
            static_cast<int64_t>(cfg.vocab_size) * (r + 1) / kWorld - begin);
        require(b.lm_vocab_begin == begin && b.lm_vocab_count == count,
                "sharded slice bounds wrong at rank " + std::to_string(r));
        // The bitwise claim, row by row: b's row must equal a's row
        // restricted to columns [begin, begin+count).
        for (int t = 0; t < static_cast<int>(tokens.size()); ++t) {
          const float* full_row = a.logits.data() +
              static_cast<size_t>(t) * cfg.vocab_size + begin;
          const float* shard_row =
              b.logits.data() + static_cast<size_t>(t) * b.lm_vocab_count;
          if (std::memcmp(full_row, shard_row,
                          static_cast<size_t>(count) * sizeof(float)) != 0)
            throw std::runtime_error(
                "head shard logits row " + std::to_string(t) +
                " differs from the replicated head's columns (rank " +
                std::to_string(r) + ")");
        }
        arrive_once();  // release peers even on success paths
      } catch (const std::exception& e) {
        errors[static_cast<size_t>(r)] =
            "rank " + std::to_string(r) + ": " + e.what();
        arrive_once();
      }
    });
  }
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r)
    require(errors[static_cast<size_t>(r)].empty(), errors[static_cast<size_t>(r)]);
}

// ---- M6 d3: end-to-end greedy generation ----------------------------------
// The d3 criterion on the fixture, stated as the invariant that actually
// holds: the DISTRIBUTED pick (per-rank slice argmax + the bus
// gather/broadcast) must equal the CENTRALIZED argmax over the SAME
// geometry's full logits, token for token, step after step — plus
// rank-consistency of the whole loop. The world-1 single-rank loop is
// run only as a LOGGED reference: its sequence may legitimately differ
// from the world-4 one (the boundary folds' ulp-class noise flips
// near-tie argmaxes — the same regime the route-audit discipline
// certifies; asserting cross-geometry token equality would be asserting
// the noise away). (T^2 re-forward per step — the incremental decode
// engine with persistent state is M6 stage 2; this pins the SEMANTICS
// the stateful engine must then preserve.)
DGPP_TEST(glm_tp_greedy_gen_loopback) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt = make_tokens(8, cfg.vocab_size);
  constexpr int kSteps = 6;
  constexpr int kWorld = 4;
  const int max_tokens = static_cast<int>(prompt.size()) + kSteps + 1;

  // ---- reference: the centralized world-1 greedy loop (logged) ---------
  std::vector<int64_t> w1_seq;
  {
    GlmDiagnosticModel model(cfg, dir, max_tokens, 128);
    std::vector<int64_t> toks = prompt;
    for (int s = 0; s < kSteps; ++s) {
      const GlmDiagnosticModel::Outputs out = model.forward(toks);
      const int T = static_cast<int>(toks.size());
      const float* row =
          out.logits.data() + static_cast<size_t>(T - 1) * cfg.vocab_size;
      const Candidate c = local_max(row, cfg.vocab_size, 0);
      toks.push_back(c.id);
      w1_seq.push_back(c.id);
    }
  }
  require(w1_seq.size() == kSteps, "w1 reference produced wrong length");

  // ---- the distributed loop: world 4, BOTH heads, bus pick ----------
  // Full-head model = the centralized pick's source (its logits are the
  // bitwise union of the shards — pinned by glm_tp_head_shard_parity);
  // sharded model = the serving path. Every rank forwards BOTH each
  // step (same order on every rank, collectives stay aligned).
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29916);
  require(!buses.empty(), "tp bus world failed to start");
  // Small-collective canary: 16 elems as the FIRST collective on a fresh
  // bus must pass (pinned standalone). The OPEN ENGINE BUG is the
  // staged->small-plain TRANSITION (see bus_greedy_pick's note): 16 elems
  // after a run of 2048-elem staged collectives stalls. This canary guards
  // the passing path while that hunt stays open.
  {
    std::vector<std::string> probe(kWorld);
    ConstructBarrier probe_barrier(kWorld);
    std::vector<std::thread> probe_workers;
    for (int r = 0; r < kWorld; ++r)
      probe_workers.emplace_back([&, r] {
        try {
          uint16_t* s = nullptr;
          DGPP_CUDA_OK(cudaMallocManaged(&s, 32));
          std::memset(s, 0, 32);
          s[0] = static_cast<uint16_t>(r);
          probe_barrier.arrive_and_wait();
          std::string err;
          const uint64_t id =
              buses[static_cast<size_t>(r)]->allreduce(s, s, 16, &err);
          if (id == 0) throw std::runtime_error("probe submit: " + err);
          const auto res = buses[static_cast<size_t>(r)]->wait_allreduce(
              id, 60000);
          if (!res.ok) throw std::runtime_error("probe wait: " + res.error);
          cudaFree(s);
        } catch (const std::exception& e) {
          probe[static_cast<size_t>(r)] = e.what();
          probe_barrier.arrive_and_wait();
        }
      });
    for (auto& t : probe_workers) t.join();
    for (int r = 0; r < kWorld; ++r)
      require(probe[static_cast<size_t>(r)].empty(),
              "probe rank " + std::to_string(r) + ": " +
                  probe[static_cast<size_t>(r)]);
    DGPP_LOG_INFO("small-collective probe: 16 elems standalone PASSED");
  }
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<int64_t>> rank_seqs(
      kWorld, std::vector<int64_t>(kSteps, -1));
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  workers.reserve(kWorld);
  for (int r = 0; r < kWorld; ++r) {
    workers.emplace_back([&, r] {
      bool arrived = false;
      const auto arrive_once = [&] {
        if (arrived) return;
        arrived = true;
        barrier.arrive_and_wait();
      };
      uint16_t* scratch = nullptr;  // gather-scratch (2048 bf16)
      try {
        GlmBusBoundaryReducer reducer(*buses[static_cast<size_t>(r)]);
        GlmDiagnosticModel full(cfg, dir, max_tokens, 128, &reducer, r,
                               kWorld);
        GlmDiagnosticModel shard(cfg, dir, max_tokens, 128, &reducer, r,
                                 kWorld, GlmResidency::Streaming,
                                 GlmHeadSharding::VocabSharded);
        DGPP_CUDA_OK(cudaMallocManaged(&scratch, sizeof(uint16_t) * 2048));
        arrive_once();
        std::vector<int64_t> toks = prompt;
        for (int s = 0; s < kSteps; ++s) {
          const GlmDiagnosticModel::Outputs full_out = full.forward(toks);
          const GlmDiagnosticModel::Outputs shard_out = shard.forward(toks);
          const int T = static_cast<int>(toks.size());
          // centralized pick: the full head's last row, whole vocab
          const float* full_row =
              full_out.logits.data() +
              static_cast<size_t>(T - 1) * cfg.vocab_size;
          const Candidate central = local_max(full_row, cfg.vocab_size, 0);
          // distributed pick: this rank's slice through the bus
          const float* slice_row =
              shard_out.logits.data() +
              static_cast<size_t>(T - 1) * shard_out.lm_vocab_count;
          const Candidate local =
              local_max(slice_row, shard_out.lm_vocab_count,
                        shard_out.lm_vocab_begin);
          const int32_t token = bus_greedy_pick(
              *buses[static_cast<size_t>(r)], r, kWorld, local, scratch,
              60000);
          if (token < 0 || token >= cfg.vocab_size)
            throw std::runtime_error("generated id out of range");
          if (token != central.id)
            throw std::runtime_error(
                "distributed pick != centralized pick at step " +
                std::to_string(s) + ": " + std::to_string(token) +
                " vs " + std::to_string(central.id));
          toks.push_back(token);
          rank_seqs[static_cast<size_t>(r)][static_cast<size_t>(s)] = token;
        }
        cudaFree(scratch);
      } catch (const std::exception& e) {
        if (scratch) cudaFree(scratch);
        errors[static_cast<size_t>(r)] =
            "rank " + std::to_string(r) + ": " + e.what();
        arrive_once();
      }
    });
  }
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r)
    require(errors[static_cast<size_t>(r)].empty(), errors[static_cast<size_t>(r)]);

  // ---- rank consistency: identical sequences everywhere -------------
  for (int r = 1; r < kWorld; ++r) {
    require(rank_seqs[static_cast<size_t>(r)] == rank_seqs[0],
            "greedy sequence differs across ranks — rank-consistency "
            "broken at rank " +
                std::to_string(r));
  }
  std::string w1_txt, w4_txt;
  for (int64_t t : w1_seq) w1_txt += std::to_string(t) + " ";
  for (int64_t t : rank_seqs[0]) w4_txt += std::to_string(t) + " ";
  DGPP_LOG_INFO("greedy w1 reference: {}", w1_txt);
  DGPP_LOG_INFO("greedy w4 dist      : {}", w4_txt);
}

// M6 Stage 2: the stateful decode session vs the stateless re-forward.
// Tiers (the parity discipline — CERTIFY near ties, never assume them):
//   * single-chunk prefill (prompt <= 2048): BITWISE on the last row — the
//     session runs run_stack's exact op sequence on fresh state;
//   * steps and multi-chunk prefill: free-run l2 REPORTED, top-1
//     divergence CERTIFIED (the re-forward's batched GEMMs vs the step's
//     M=1 / the chunk's M=2048 ulps; the recurrence itself is exact —
//     DESIGN §7.1's one-shared-recurrence claim is what keeps the noise
//     at ulps instead of compounding drift);
//   * world 4: the same tiering through the bus (the step's T=1 staged
//     folds), plus rank-consistent transcripts (the fold's bitwise
//     guarantee — every rank's full-head row must be identical).
// Route flips per step are REPORTED only: the router's input row is the
// certified logits surface; routing certification is the re-forward
// gates' tier (glm_route_audit's cascade over full forwards).
DGPP_TEST(glm_tp_decode_session_parity) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt = make_tokens(9, cfg.vocab_size);
  constexpr int kSteps = 6;
  const int max_tokens =
      static_cast<int>(prompt.size()) + kSteps + 1;
  const size_t V = static_cast<size_t>(cfg.vocab_size);
  const size_t H = static_cast<size_t>(cfg.hidden_size);

  // The engine transcript drives BOTH sides (identical tokens in), so the
  // reference rows and the session rows are comparable position by
  // position. Engine and reference are SEPARATE model instances: a plain
  // forward mid-session would clobber the session's state pools (and now
  // throws — the hazard is pinned in the negative test below).
  const auto bits_argmax = [&](const std::vector<float>& row, int count,
                               int begin) {
    return local_max(row.data(), count, begin);
  };
  const auto engine_transcript = [&](GlmDiagnosticModel& eng,
                                     std::vector<int64_t>* gen_out,
                                     std::vector<std::vector<float>>* rows) {
    const GlmDiagnosticModel::Outputs pre = eng.session_prefill(prompt);
    std::vector<int64_t> gen;
    int32_t token = bits_argmax(pre.logits, pre.lm_vocab_count,
                                pre.lm_vocab_begin).id;
    for (int s = 0; s < kSteps; ++s) {
      const auto t0 = std::chrono::steady_clock::now();
      const GlmDiagnosticModel::Outputs out = eng.session_step(token);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      gen.push_back(token);
      if (rows) rows->push_back(out.logits);
      DGPP_LOG_DEBUG("session step {} ({}ms): token {}", s, ms, token);
      token = bits_argmax(out.logits, out.lm_vocab_count,
                        out.lm_vocab_begin)
                  .id;
    }
    *gen_out = gen;
  };

  // ---- phase A: world 1 ------------------------------------------------
  std::vector<int64_t> w1_gen;
  std::vector<std::vector<float>> w1_rows;
  {
    GlmDiagnosticModel eng(cfg, dir, max_tokens, 128);
    GlmDiagnosticModel ref(cfg, dir, max_tokens, 128);

    // BITWISE tier: single-chunk prefill == the re-forward's last row.
    const GlmDiagnosticModel::Outputs pre = eng.session_prefill(prompt);
    const GlmDiagnosticModel::Outputs ref_pre = ref.forward(prompt);
    const size_t rowV = (prompt.size() - 1) * V;
    const size_t rowH = (prompt.size() - 1) * H;
    require(pre.logits.size() == V, "prefill row shape");
    require(pre.final_hidden_bits.size() == H, "prefill hidden shape");
    require(std::equal(pre.logits.begin(), pre.logits.end(),
                       ref_pre.logits.begin() + rowV),
            "single-chunk prefill logits must be BITWISE the re-forward's "
            "last row (same op sequence on fresh state)");
    require(std::equal(pre.final_hidden_bits.begin(),
                       pre.final_hidden_bits.end(),
                       ref_pre.final_hidden_bits.begin() + rowH),
            "single-chunk prefill final_hidden must be BITWISE the "
            "re-forward's last row");

    // The engine transcript, then the reference over the SAME tokens.
    engine_transcript(eng, &w1_gen, &w1_rows);
    std::vector<int64_t> seq = prompt;
    seq.insert(seq.end(), w1_gen.begin(), w1_gen.end());
    const GlmDiagnosticModel::Outputs ref_all = ref.forward(seq);
    require(ref_all.logits.size() == seq.size() * V,
            "reference forward row count");

    // Steps: row s is position P+s (the step consumed gen[s] there).
    for (int s = 0; s < kSteps; ++s) {
      const size_t pos = prompt.size() + static_cast<size_t>(s);
      std::vector<float> rr(ref_all.logits.begin() + pos * V,
                               ref_all.logits.begin() + (pos + 1) * V);
      const double rel = dgpp::glm_route::l2_rel(w1_rows[s], rr);
      const Top1AuditSummary t1 =
          audit_top1_near_ties(w1_rows[s].data(), rr.data(), cfg.vocab_size, 1);
      certify_top1_near_ties(t1, 1, "decode step (w1)");
      DGPP_LOG_INFO("step {} vs re-forward: rel_l2 {:.6f}, top1 misses {}",
                    s, rel, t1.misses);
    }

    // ---- multi-chunk prefill: 2050 = 2048 + 2 (a mid-pool chunk end), --
    // then steps at pool-interior positions (the tail path).
    const std::vector<int64_t> big = make_tokens(2050, cfg.vocab_size);
    constexpr int kBigSteps = 3;
    const int big_max = 2050 + kBigSteps + 1;
    GlmDiagnosticModel eng2(cfg, dir, big_max, big_max);
    GlmDiagnosticModel ref2(cfg, dir, big_max, big_max);
    const GlmDiagnosticModel::Outputs pre2 = eng2.session_prefill(big);
    std::vector<int64_t> gen2;
    int32_t token2 =
        bits_argmax(pre2.logits, pre2.lm_vocab_count,
                  pre2.lm_vocab_begin)
            .id;
    std::vector<std::vector<float>> rows2;
    for (int s = 0; s < kBigSteps; ++s) {
      const GlmDiagnosticModel::Outputs out = eng2.session_step(token2);
      rows2.push_back(out.logits);
      gen2.push_back(token2);
      token2 = bits_argmax(out.logits, out.lm_vocab_count,
                         out.lm_vocab_begin)
                   .id;
    }
    std::vector<int64_t> seq2 = big;
    seq2.insert(seq2.end(), gen2.begin(), gen2.end());
    const GlmDiagnosticModel::Outputs ref2_all = ref2.forward(seq2);

    // Prefill chunking tier: last row, chunked (2048+2) vs single-shot.
    {
      const size_t pos = 2049;
      std::vector<float> rr(ref2_all.logits.begin() + pos * V,
                               ref2_all.logits.begin() + (pos + 1) * V);
      const double rel = dgpp::glm_route::l2_rel(pre2.logits, rr);
      const Top1AuditSummary t1 = audit_top1_near_ties(
          pre2.logits.data(), rr.data(), cfg.vocab_size, 1);
      certify_top1_near_ties(t1, 1, "multi-chunk prefill (w1)");
      DGPP_LOG_INFO("multi-chunk prefill (2048+2) vs re-forward: rel_l2 "
                    "{:.6f}, top1 misses {} — the KDA recurrence crossed "
                    "the chunk boundary; noise must be GEMM-batch ulps, "
                    "not state drift",
                    rel, t1.misses);
    }
    for (int s = 0; s < kBigSteps; ++s) {
      const size_t pos = 2050 + static_cast<size_t>(s);
      std::vector<float> rr(ref2_all.logits.begin() + pos * V,
                               ref2_all.logits.begin() + (pos + 1) * V);
      const double rel = dgpp::glm_route::l2_rel(rows2[s], rr);
      const Top1AuditSummary t1 =
          audit_top1_near_ties(rows2[s].data(), rr.data(), cfg.vocab_size, 1);
      certify_top1_near_ties(t1, 1, "decode step after chunk boundary (w1)");
      DGPP_LOG_INFO("post-chunk step {} vs re-forward: rel_l2 {:.6f}, "
                    "top1 misses {}",
                    s, rel, t1.misses);
    }
  }

  // ---- phase B: world 4 through the bus --------------------------------
  const int kWorld = 4;
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29918);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<int64_t>> rank_seqs(
      kWorld, std::vector<int64_t>(kSteps, -1));
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r) {
    workers.emplace_back([&, r] {
      bool arrived = false;
      const auto arrive_once = [&] {
        if (arrived) return;
        arrived = true;
        barrier.arrive_and_wait();
      };
      try {
        GlmBusBoundaryReducer reducer(*buses[static_cast<size_t>(r)]);
        GlmDiagnosticModel eng(cfg, dir, max_tokens, 128, &reducer, r, kWorld);
        GlmDiagnosticModel ref(cfg, dir, max_tokens, 128, &reducer, r, kWorld);
        // Construction barrier (the same one every other world-4 section
        // has): the constructors cudaDeviceSynchronize (the loader's
        // sync_load_boundary), which waits for EVERY kernel on the device
        // — including a faster peer's first collective kernel spinning on
        // this rank's doorbell. Without it: 1 start in ~8 deadlocked until
        // the 5 s watchdog (the 2026-09-02 "seq-1 wedge" hunt, caught by a
        // backtrace: two ranks in cudaDeviceSynchronize, two in
        // spin_then_wait).
        arrive_once();

        // BITWISE prefill tier through the bus (identical collective
        // sequences on both sides).
        const GlmDiagnosticModel::Outputs pre = eng.session_prefill(prompt);
        const GlmDiagnosticModel::Outputs ref_pre = ref.forward(prompt);
        const size_t rowV = (prompt.size() - 1) * V;
        require(std::equal(pre.logits.begin(), pre.logits.end(),
                           ref_pre.logits.begin() + rowV),
                "w4 single-chunk prefill must be BITWISE the re-forward's "
                "last row");

        std::vector<int64_t> gen;
        std::vector<std::vector<float>> rows;
        engine_transcript(eng, &gen, &rows);
        for (int s = 0; s < kSteps; ++s)
          rank_seqs[static_cast<size_t>(r)][static_cast<size_t>(s)] =
              gen[static_cast<size_t>(s)];

        std::vector<int64_t> seq = prompt;
        seq.insert(seq.end(), gen.begin(), gen.end());
        const GlmDiagnosticModel::Outputs ref_all = ref.forward(seq);
        for (int s = 0; s < kSteps; ++s) {
          const size_t pos = prompt.size() + static_cast<size_t>(s);
          std::vector<float> rr(ref_all.logits.begin() + pos * V,
                                   ref_all.logits.begin() + (pos + 1) * V);
          const double rel = dgpp::glm_route::l2_rel(rows[s], rr);
          const Top1AuditSummary t1 = audit_top1_near_ties(
              rows[s].data(), rr.data(), cfg.vocab_size, 1);
          certify_top1_near_ties(t1, 1, "decode step (w4)");
          DGPP_LOG_INFO("w4 step {} vs re-forward: rel_l2 {:.6f}, top1 "
                        "misses {}",
                        s, rel, t1.misses);
        }
        arrive_once();
      } catch (const std::exception& e) {
        errors[static_cast<size_t>(r)] =
            "rank " + std::to_string(r) + ": " + e.what();
        arrive_once();
      }
    });
  }
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r)
    require(errors[static_cast<size_t>(r)].empty(), errors[static_cast<size_t>(r)]);
  for (int r = 1; r < kWorld; ++r)
    require(rank_seqs[static_cast<size_t>(r)] == rank_seqs[0],
            "w4 session transcript differs across ranks at rank " +
                std::to_string(r));

  std::string w1_txt, w4_txt;
  for (int64_t t : w1_gen) w1_txt += std::to_string(t) + " ";
  for (int64_t t : rank_seqs[0]) w4_txt += std::to_string(t) + " ";
  DGPP_LOG_INFO("session w1 transcript: {}", w1_txt);
  DGPP_LOG_INFO("session w4 transcript: {}", w4_txt);
}

// The session/forward state-sharing hazard, pinned: forward() on a model
// instance with an open decode session must THROW (it would silently
// clobber the session's KDA/DSA state pools — the exact corruption this
// gate's phase-A design would otherwise have absorbed as noise).
DGPP_TEST(glm_tp_decode_session_hazard) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt = make_tokens(8, cfg.vocab_size);
  GlmDiagnosticModel model(cfg, dir, 16, 16);
  model.session_prefill(prompt);
  bool threw = false;
  std::string msg;
  try {
    (void)model.forward(prompt);
  } catch (const std::exception& e) {
    threw = true;
    msg = e.what();
  }
  require(threw && msg.find("decode session is open") != std::string::npos,
          "forward mid-session must throw the session-open guard (got: " +
              msg + ")");
}

// The quantized scale-grid slice contract, pinned: a rank's slice of a
// block-scaled matrix must start 128-aligned in the sliced dimension.
// The pre-fix fixture geometry (dense inter 200 / MoE inter 64) is the
// counterexample — it must be REJECTED loudly, never silently mis-scaled
// (the hunt: rank 1's dense slice started mid-block, read the wrong
// scale rows for 72 of its 100 inter dims, and the fold carried ~0.30
// l2 error past a 0.02-budget assertion). Both consumers must reject it:
// GlmTpViews AND the sharded GlmLayerStream (before a single shard is
// opened — a bad config fails in milliseconds, not after 328 GB of mmap).
DGPP_TEST(glm_tp_slice_alignment_contract) {
  const GlmTextConfig base = glm_tp_test_config();
  cudaStream_t st;
  DGPP_CUDA_OK(cudaStreamCreate(&st));
  const auto expect_throw = [&](GlmTextConfig cfg, const char* needle) {
    bool threw = false;
    std::string msg;
    try {
      dgpp::GlmTpViews tp(cfg, 1, 2, st);
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    require(threw && msg.find(needle) != std::string::npos,
            std::string("misaligned inter slice must throw with '") +
                needle + "' (got: " + msg + ")");
    threw = false;
    msg.clear();
    try {
      dgpp::GlmLayerStream shard(cfg, "glm_tp_fixture", 1, 2);
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    require(threw && msg.find(needle) != std::string::npos,
            std::string("sharded loader must reject misaligned geometry "
                        "with '") +
                needle + "' (got: " + msg + ")");
  };
  {
    GlmTextConfig bad = base;
    bad.intermediate_size = 200;  // divides by 2; quotient 100 % 128 != 0
    expect_throw(bad, "128");
  }
  {
    GlmTextConfig bad = base;
    bad.moe_intermediate_size = 64;  // quotient 32 % 128 != 0
    expect_throw(bad, "128");
  }
  DGPP_CUDA_OK(cudaStreamDestroy(st));
}

// ---------------------------------------------------------------------------
// M5 d4: sharded load vs full-load+views, pinned BITWISE. The comparator
// and driver live in src/models/glm_tp_parity — shared with the
// glm_shard_parity app so the fixture CI and the real-checkpoint gate
// run ONE code path (the fixture is geometry the real model's 76k-tensor
// binding is the generalization of; the app is how that generalization
// is actually tested at scale). This test pins: every bound surface of
// every layer (all 6 fixture layers + the MTP draft layer, KDA and DSA,
// dense and MoE) matches byte-for-byte at worlds 2 and 4, the
// replicated digest is rank-invariant, and the byte reconcile identity
// holds — sum_r(source) == world1 total + (world-1)*verbatim.
// ---------------------------------------------------------------------------
DGPP_TEST(glm_tp_shard_parity) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);

  for (int world : {2, 4}) {
    const GlmShardParityReport rep =
        glm_shard_parity_check(cfg, dir, world);
    require(rep.layers_checked ==
                cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0),
            "shard parity: layer count mismatch");
    DGPP_LOG_INFO(
        "shard parity world={}: {} layers x {} ranks, {} bound surfaces "
        "bitwise-equal; byte reconcile exact — rank reads {}/{} source "
        "bytes ({:.0f}%)",
        world, rep.layers_checked, world, rep.surfaces_checked,
        rep.shard_source_bytes, rep.full_source_bytes,
        100.0 * static_cast<double>(rep.shard_source_bytes) /
            static_cast<double>(rep.full_source_bytes));

    // RESIDENT mode: the sharded streams materialize every layer once
    // and the same parity must hold bitwise (resident bytes are
    // streaming bytes by construction — this is what keeps the two
    // build paths from ever drifting), plus the residency contract's
    // proof: every layer re-served from cache, same addresses, zero
    // storage reads.
    const GlmShardParityReport rrep =
        glm_shard_parity_check(cfg, dir, world, nullptr, /*resident=*/true);
    require(rrep.surfaces_checked == rep.surfaces_checked,
            "shard parity: resident mode checked a different surface set");
    require(rrep.cache_hits == rep.layers_checked * world,
            "shard parity: resident cache-hit count mismatch");
    require(rrep.shard_source_bytes == rep.shard_source_bytes,
            "shard parity: resident mode read a different byte set");
    DGPP_LOG_INFO(
        "shard parity world={} [RESIDENT]: {} surfaces bitwise-equal; "
        "{} cache hits across {} ranks with 0 storage reads — residency "
        "contract holds",
        world, rrep.surfaces_checked, rrep.cache_hits, world);
  }
}

// ---------------------------------------------------------------------------
// The 0.57 hunt (2026-08-31): the SAME forward-parity machinery against
// the REAL checkpoint, loopback. The fabric gate's first real TP=2 run
// failed its tolerance tier (final_hidden rel_l2 0.5677, logits 0.5159,
// uniform on every token from token 0 — systematic, not drift) while the
// cross-rank BITWISE surface passed; the CI fixture cannot reproduce a
// geometry-class bug its own dims never reach (fixture DSA q_lora 64 /
// kv_lora 64 / nope 32 / v 32 / 8 heads vs real 1536 / 512 / 256 / 256 /
// 64; KDA head_dim 64 vs 128). This test is the hunt's second instrument:
// per-layer isolated parity and raw boundary folds vs the world=1 oracle
// AT real dims, with two token counts chosen to bisect the FOLD path at
// real hidden 4096 (GlmBusBoundaryReducer routes on rows*hidden):
//   T=21  -> 86016 elems > 2 latency slots -> the BULK RS+AG machine
//            (exactly the fabric run's every-boundary path; the CI
//            forward at fixture hidden 256 NEVER exercised bulk folds)
//   T=2   -> 8192 elems <= 2 slots -> the CHUNKED latency path
// T=21 failing while T=2 passes convicts the bulk path in the forward
// context; both failing convicts the sharded compute at real local
// geometry; both passing points at the fabric context itself.
// Env-gated so CI stays fixture-only: DGPP_TP_REAL_MODEL=ORG/NAME.
// ---------------------------------------------------------------------------
DGPP_TEST(glm_tp_forward_parity_real) {
  const char* model_env = std::getenv("DGPP_TP_REAL_MODEL");
  if (!model_env || !*model_env) {
    DGPP_LOG_INFO("glm_tp_forward_parity_real: skipped (set "
                  "DGPP_TP_REAL_MODEL=ORG/NAME to run against the real "
                  "checkpoint)");
    return;
  }
  const std::string model_id = model_env;
  std::string err;
  const std::string dir = dgpp::hf::model_dir(model_id, &err);
  require(!dir.empty(), "real model resolve failed: " + err);
  DGPP_LOG_INFO("real-dims forward parity: {} -> {}", model_id, dir);
  const GlmTextConfig cfg =
      GlmTextConfig::from_json_file(
          (std::filesystem::path(dir) / "config.json").string());
  const int64_t cache = 128;

  std::vector<std::string> case_failures;
  // T=21 at real hidden 4096 rides the bulk RS+AG machine at every
  // boundary — the only fold path reachable at real dims: the chunked
  // latency path needs rows*4096 <= 8192 (T <= 2), and the DSA layer
  // rejects max_tokens below its index-derived minimum, so no legal T
  // takes it. Worlds 2 AND 4: the gate's TP=2/TP=4 parity tier in one
  // command.
  for (const auto& [tokens_n, world, port, what] :
       std::vector<std::tuple<int, int, uint16_t, const char*>>{
           {21, 2, 29910, "bulk-real-w2"},
           {21, 4, 29912, "bulk-real-w4"},
       }) {
    // A failing case must not strand the other: each case's evidence is
    // collected (logs + dumps land regardless), the exception is
    // re-raised only after every case ran — the hunt gets the whole
    // table in one process.
    try {
      const std::vector<int64_t> tokens =
          make_tokens(tokens_n, cfg.vocab_size);
      const std::vector<std::vector<uint16_t>> states = make_layer_states(
          cfg.num_hidden_layers, tokens_n, cfg.hidden_size);
      std::vector<const uint16_t*> state_ptrs;
      for (const auto& s : states) state_ptrs.push_back(s.data());
      // Oracle: single pass (the fixture test's determinism double-run is
      // halved away here — M4 and the fabric runs established the world=1
      // chain's determinism, and each real-dims pass costs minutes).
      OracleRef ref;
      {
        GlmDiagnosticModel oracle(cfg, dir, tokens_n, cache);
        ref.free_out = oracle.forward(tokens);
        ref.iso_out = oracle.forward_isolated(tokens, state_ptrs,
                                               ref.captures, &ref.boundary);
      }
      check_world(world, port, dir, cfg, tokens, state_ptrs, cache,
                  ref.free_out, ref.iso_out, ref.captures, ref.boundary,
                  what);
    } catch (const std::exception& e) {
      case_failures.push_back(std::string("[") + what + "] " + e.what());
      DGPP_LOG_ERROR("real-dims case {} failed: {}", what, e.what());
    }
  }
  require(case_failures.empty(), [&] {
    std::string joined;
    for (const auto& f : case_failures) joined += f + "\n";
    return joined;
  }());
}

}  // namespace

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  // One process, several ranks, each with kernels that spin on a peer's
  // doorbell: CUDA's default LAZY module loading deadlocks that shape (a
  // first launch waits for an idle device — see bus_kernel.hpp,
  // bus_preload_kernels). The bus preloads its own kernels; the model's
  // compute kernels launched beside a live collective are loaded eagerly
  // here, before the first CUDA call. Production is one rank per box.
  setenv("CUDA_MODULE_LOADING", "EAGER", /*overwrite=*/0);

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    DGPP_LOG_INFO("no CUDA device visible; skipping glm_tp_test");
    return 2;
  }
  return dgpp::test::run_all();
}
