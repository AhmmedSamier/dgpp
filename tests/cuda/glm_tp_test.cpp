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
using dgpp::GlmLayerBound;
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
    dump(".logits.bf16", ranks[0].free_out.logits_bits);
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
      ranks[0].free_out.logits_bits.data(), ref_free.logits_bits.data(),
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
              ref.free_out.logits_bits == again.logits_bits,
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

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    DGPP_LOG_INFO("no CUDA device visible; skipping glm_tp_test");
    return 2;
  }
  return dgpp::test::run_all();
}
