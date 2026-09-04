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
#include <array>
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
#include "models/glm_fabric_engine.hpp"
#include "models/glm_forward.hpp"
#include "models/glm_graph_check.hpp"
#include "models/glm_route_audit.hpp"
#include "models/glm_sampler.hpp"
#include "models/glm_speculative.hpp"
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

// Every host-side wait budget in these worlds (boundary reducers, picks,
// replay finishes, the adapter's pick timeout) — 60 s in release, lifted
// under compute-sanitizer with the bus timeout and the kernel deadline:
//   DGPP_TEST_BUS_TIMEOUT_MS=120000 DGPP_TEST_CONSUMER_DEADLINE_S=120
//   DGPP_TEST_WAIT_TIMEOUT_MS=test_wait_timeout_ms()0
int test_wait_timeout_ms() {
  static const int ms = [] {
    const char* v = std::getenv("DGPP_TEST_WAIT_TIMEOUT_MS");
    return v ? std::atoi(v) : 60000;
  }();
  return ms;
}

BusOptions loop_options(int rank, int world, uint16_t port,
                        size_t lat_slot_bytes = 8192) {
  BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.lane_devices = {"rocep1s0f0", "roceP2p1s0f0"};
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : "127.0.0.1";
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = 8;
  o.lat_slot_bytes = lat_slot_bytes;
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
  // The kernel-side deadline follows the same rule (compute-sanitizer
  // memcheck slows every instrumented kernel enough that a peer's prefill
  // outlasts 20 s of a spinning collective).
  o.consumer_deadline_s = [] {
    const char* s = std::getenv("DGPP_TEST_CONSUMER_DEADLINE_S");
    return s ? std::atof(s) : 20.0;
  }();
  o.launch_consumers = false;     // per-collective kernels own doorbells
  return o;
}

// Starts a loopback world of buses (rank 0 listens), or returns empty.
std::vector<std::unique_ptr<CollectiveBus>> start_world(int world,
                                                        uint16_t port,
                                                        size_t lat_slot_bytes =
                                                            8192) {
  std::vector<std::unique_ptr<CollectiveBus>> out;
  for (int r = 0; r < world; ++r)
    out.push_back(std::make_unique<CollectiveBus>(
        loop_options(r, world, port, lat_slot_bytes)));
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
    GlmBusBoundaryReducer reducer(*bus, test_wait_timeout_ms());
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
        GlmBusBoundaryReducer reducer(*buses[static_cast<size_t>(r)], test_wait_timeout_ms());
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

// M6 6b's sizing probe over the real collective: local top-256 candidates and
// fp64 slice normalizers survive the bf16 digit wire exactly, every rank
// reconstructs the same global table, and its masses match a centralized
// full-vocabulary oracle. The fixture includes ties across shards so the token
// id half of canonical ordering is part of the check.
DGPP_TEST(glm_sampling_profile_mass_gather_matches_centralized_loopback) {
  constexpr int kWorld = 2;
  constexpr int kSlice = 300;
  constexpr uint16_t kPort = 29925;
  std::vector<float> logits(kWorld * kSlice);
  for (int id = 0; id < static_cast<int>(logits.size()); ++id) {
    logits[static_cast<size_t>(id)] =
        static_cast<float>((id * 37) % 211 - 105) / 16.0f;
  }

  const double full_lse =
      dgpp::glm_sample::slice_logsumexp(logits.data(), logits.size());
  const std::vector<Candidate> full_top = dgpp::glm_sample::local_topk(
      logits.data(), logits.size(), 0, dgpp::kSamplingProfileMaxK);
  const std::vector<double> expected =
      dgpp::glm_sample::topk_probability_masses(
          full_top, full_lse,
          std::vector<int>(dgpp::kSamplingProfileTopKs.begin(),
                           dgpp::kSamplingProfileTopKs.end()));

  std::vector<std::unique_ptr<CollectiveBus>> buses =
      start_world(kWorld, kPort, 64 * 1024);
  require(!buses.empty(), "sampling-profile bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::array<double, dgpp::kSamplingProfileTopKs.size()>> got(
      kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int rank = 0; rank < kWorld; ++rank) {
    workers.emplace_back([&, rank] {
      uint16_t* scratch = nullptr;
      bool arrived = false;
      const auto arrive_once = [&] {
        if (arrived) return;
        arrived = true;
        barrier.arrive_and_wait();
      };
      try {
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&scratch),
            sizeof(uint16_t) *
                dgpp::sampling_profile_scratch_elems(kWorld),
            cudaHostAllocDefault));
        const float* slice = logits.data() + rank * kSlice;
        const double local_lse =
            dgpp::glm_sample::slice_logsumexp(slice, kSlice);
        arrive_once();
        got[static_cast<size_t>(rank)] = dgpp::bus_sampling_topk_masses(
            *buses[static_cast<size_t>(rank)], rank, kWorld, slice, kSlice,
            rank * kSlice, local_lse, scratch, test_wait_timeout_ms());
        cudaFreeHost(scratch);
        scratch = nullptr;
      } catch (const std::exception& error) {
        if (scratch != nullptr) cudaFreeHost(scratch);
        errors[static_cast<size_t>(rank)] = error.what();
        arrive_once();
      }
    });
  }
  for (auto& worker : workers) worker.join();
  for (int rank = 0; rank < kWorld; ++rank)
    require(errors[static_cast<size_t>(rank)].empty(),
            "sampling-profile rank " + std::to_string(rank) + ": " +
                errors[static_cast<size_t>(rank)]);
  require(got[0] == got[1], "sampling-profile masses differ across ranks");
  for (size_t i = 0; i < expected.size(); ++i)
    require(std::abs(got[0][i] - expected[i]) < 2e-14,
            "sampling-profile mass differs from centralized oracle at k=" +
                std::to_string(dgpp::kSamplingProfileTopKs[i]));
}

// M6.6b's width-independent correctness seam over the real bus, in
// GLM-5.3-Flash-FP8's actual default regime (temperature=1, top_p=.95, no
// semantic top_k — the checkpoint's generation_config.json carries exactly
// those two sampling fields). A peaked distribution resolves inside the
// candidate prefix over a run of draws and must equal the sharded reference
// bitwise on every rank; a deliberately flat distribution requests the exact
// full-logit fallback without consuming its RNG draw. Penalties are applied
// before the per-rank top-k on context ids owned by both shards, and the
// cross-shard tie SURVIVES them (equal counts on both ids), so the id half
// of the canonical order is on the wire and decides draws. Every call also
// carries rank 0's decision digest back to every rank.
DGPP_TEST(glm_sampling_prefix_decision_matches_centralized_loopback) {
  constexpr int kWorld = 2;
  constexpr int kSlice = 300;
  constexpr int kVocab = kWorld * kSlice;
  constexpr int kCandidates = 32;
  constexpr uint16_t kPort = 29926;
  constexpr uint64_t kSeed = 0x53f1a5ull;
  const std::vector<uint64_t> counters{4, 5, 6, 7, 8, 9, 10, 11};
  constexpr uint64_t kFallbackCounter = 9;

  std::vector<float> peaked(kVocab, -10.0f);
  peaked[11] = 9.0f;
  peaked[350] = 9.0f;  // cross-shard tie; the penalties below keep it one
  peaked[207] = 8.5f;
  const std::vector<float> flat(kVocab, 0.0f);
  const std::vector<int32_t> context{11, 350, 401};
  const std::vector<dgpp::glm_sample::VocabSlice> layout{{0, kSlice},
                                                         {kSlice, kSlice}};
  dgpp::glm_sample::Params params;
  params.temperature = 1.0f;
  params.top_p = 0.95f;
  params.top_k = 0;
  params.frequency_penalty = 0.05f;
  params.presence_penalty = 0.10f;
  params.logprobs = 3;

  const auto identical = [](const dgpp::glm_sample::Result& a,
                            const dgpp::glm_sample::Result& b) {
    return a.token == b.token &&
           std::memcmp(&a.logprob, &b.logprob, sizeof(float)) == 0 &&
           a.top_logprobs == b.top_logprobs;
  };

  // Fixture sanity: the tie survives the penalties and the canonical order
  // breaks it by id, so a draw can land on either shard's member.
  {
    std::vector<float> adjusted = peaked;
    dgpp::glm_sample::apply_penalties(
        adjusted.data(), kVocab, 0, params,
        dgpp::glm_sample::count_context(context));
    require(adjusted[11] == adjusted[350],
            "fixture: the cross-shard tie must survive the penalties");
    const std::vector<Candidate> top =
        dgpp::glm_sample::local_topk(adjusted.data(), kVocab, 0, 2);
    require(top[0].id == 11 && top[1].id == 350,
            "fixture: the canonical order breaks the tie by id");
  }

  // Expectations: the sharded reference per counter. On this unambiguous
  // nucleus it also equals sample_reference() bitwise (same three survivors,
  // the shared selector's final arithmetic).
  std::vector<dgpp::glm_sample::Result> expected;
  bool both_tied_ids_drawn_11 = false;
  bool both_tied_ids_drawn_350 = false;
  for (uint64_t counter : counters) {
    dgpp::glm_sample::Rng rng{kSeed, counter};
    expected.push_back(dgpp::glm_sample::sample_reference_sharded(
        peaked.data(), kVocab, layout, params, rng, context));
    require(rng.counter == counter + 1, "the reference draws exactly once");
    dgpp::glm_sample::Rng plain{kSeed, counter};
    const dgpp::glm_sample::Result reference =
        dgpp::glm_sample::sample_reference(peaked.data(), kVocab, params,
                                           plain, context);
    require(identical(expected.back(), reference),
            "sharded reference differs from sample_reference on an "
            "unambiguous nucleus at counter " + std::to_string(counter));
    both_tied_ids_drawn_11 |= expected.back().token == 11;
    both_tied_ids_drawn_350 |= expected.back().token == 350;
  }
  require(both_tied_ids_drawn_11 && both_tied_ids_drawn_350,
          "fixture: the run of draws must land on both tied ids");
  double expected_fallback_mass = 0.0;
  {
    // The flat fixture's transported mass: 32 of 600 equal tokens.
    std::vector<float> adjusted = flat;
    dgpp::glm_sample::apply_penalties(
        adjusted.data(), kVocab, 0, params,
        dgpp::glm_sample::count_context(context));
    const std::vector<Candidate> prefix = dgpp::glm_sample::local_topk(
        adjusted.data(), kVocab, 0, kCandidates);
    dgpp::glm_sample::Rng rng{77, kFallbackCounter};
    const dgpp::glm_sample::PrefixDecision d =
        dgpp::glm_sample::sample_from_prefix(
            prefix, kVocab,
            dgpp::glm_sample::sharded_scaled_logsumexp(adjusted.data(), layout,
                                                       params.temperature),
            params, rng);
    require(!d.resolved && rng.counter == kFallbackCounter,
            "fixture: the flat prefix must be a fallback");
    expected_fallback_mass = d.covered_mass;
    // ... and the fallback's own answer exists: the complete list resolves
    // with the draw the prefix left behind.
    dgpp::glm_sample::Rng again{77, kFallbackCounter};
    (void)dgpp::glm_sample::sample_reference_sharded(
        flat.data(), kVocab, layout, params, again, context);
    require(again.counter == kFallbackCounter + 1,
            "fixture: the complete list consumes the reserved draw");
  }

  std::vector<std::unique_ptr<CollectiveBus>> buses =
      start_world(kWorld, kPort, 64 * 1024);
  require(!buses.empty(), "sampling-prefix bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<dgpp::glm_sample::PrefixDecision>> resolved(kWorld);
  std::vector<std::vector<uint64_t>> resolved_counters(kWorld);
  std::vector<dgpp::glm_sample::PrefixDecision> fallback(kWorld);
  std::vector<uint64_t> fallback_counters(kWorld, 0);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int rank = 0; rank < kWorld; ++rank) {
    workers.emplace_back([&, rank] {
      uint16_t* scratch = nullptr;
      bool arrived = false;
      const auto arrive_once = [&] {
        if (arrived) return;
        arrived = true;
        barrier.arrive_and_wait();
      };
      try {
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&scratch),
            sizeof(uint16_t) * dgpp::sampling_prefix_scratch_elems(
                                   kWorld, kCandidates),
            cudaHostAllocDefault));
        arrive_once();
        for (uint64_t counter : counters) {
          dgpp::glm_sample::Rng rng{kSeed, counter};
          resolved[static_cast<size_t>(rank)].push_back(
              dgpp::bus_sampling_prefix(
                  *buses[static_cast<size_t>(rank)], rank, kWorld,
                  peaked.data() + rank * kSlice, kSlice, rank * kSlice,
                  kVocab, params, rng, context, kCandidates, scratch,
                  test_wait_timeout_ms()));
          resolved_counters[static_cast<size_t>(rank)].push_back(rng.counter);
        }
        dgpp::glm_sample::Rng fallback_rng{77, kFallbackCounter};
        fallback[static_cast<size_t>(rank)] = dgpp::bus_sampling_prefix(
            *buses[static_cast<size_t>(rank)], rank, kWorld,
            flat.data() + rank * kSlice, kSlice, rank * kSlice, kVocab,
            params, fallback_rng, context, kCandidates, scratch,
            test_wait_timeout_ms());
        fallback_counters[static_cast<size_t>(rank)] = fallback_rng.counter;
        cudaFreeHost(scratch);
        scratch = nullptr;
      } catch (const std::exception& error) {
        if (scratch != nullptr) cudaFreeHost(scratch);
        errors[static_cast<size_t>(rank)] = error.what();
        arrive_once();
      }
    });
  }
  for (auto& worker : workers) worker.join();
  for (int rank = 0; rank < kWorld; ++rank)
    require(errors[static_cast<size_t>(rank)].empty(),
            "sampling-prefix rank " + std::to_string(rank) + ": " +
                errors[static_cast<size_t>(rank)]);
  for (int rank = 0; rank < kWorld; ++rank) {
    const auto& got = resolved[static_cast<size_t>(rank)];
    require(got.size() == counters.size(), "every draw returned");
    for (size_t i = 0; i < counters.size(); ++i) {
      const std::string where = "rank " + std::to_string(rank) +
                                " counter " + std::to_string(counters[i]);
      require(got[i].resolved, "peaked bus prefix must resolve: " + where);
      require(identical(got[i].result, expected[i]),
              "bus prefix result differs from the sharded reference: " +
                  where);
      require(resolved_counters[static_cast<size_t>(rank)][i] ==
                  counters[i] + 1,
              "resolved bus prefix consumed the wrong RNG count: " + where);
    }
    require(!fallback[static_cast<size_t>(rank)].resolved,
            "flat bus prefix must request full-logit fallback");
    require(fallback_counters[static_cast<size_t>(rank)] == kFallbackCounter,
            "bus fallback consumed the reserved RNG draw");
    require(fallback[static_cast<size_t>(rank)].covered_mass ==
                expected_fallback_mass,
            "bus fallback mass differs from the centralized prefix");
  }
}

// M6 6b's full-logit fallback transport: every rank's fp32 slice reaches
// every rank bit-for-bit through the bulk collective's 8-bit-digit
// encoding, including the payloads a raw bf16 wire would canonicalize or
// flush (NaN payloads, infinities, denormals, negative zero). The vocab
// spans two bulk stripes at the loopback slot size, so the striped path
// (not just a single-stripe corner) is the one under test.
DGPP_TEST(glm_logit_gather_is_lossless_loopback) {
  constexpr int kWorld = 2;
  constexpr int kVocab = 40000;  // 160000 words: two 131072-word stripes
  constexpr uint16_t kPort = 29927;
  const std::vector<dgpp::glm_sample::VocabSlice> layout =
      dgpp::glm_sample::vocab_layout(kVocab, kWorld);
  std::vector<float> full(kVocab);
  {
    uint64_t x = 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < kVocab; ++i) {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      uint32_t bits = static_cast<uint32_t>(x);
      switch (i % 11) {
        case 0: bits = 0x7fc00001u; break;  // NaN with a payload
        case 1: bits = 0xffbfffffu; break;  // negative signalling-ish NaN
        case 2: bits = 0x7f800000u; break;  // +inf
        case 3: bits = 0xff800000u; break;  // -inf
        case 4: bits = 0x80000000u; break;  // -0
        case 5: bits = 0x00000001u; break;  // the smallest denormal
        case 6: bits = 0x807fffffu; break;  // the largest negative denormal
        default: break;                     // random bits
      }
      std::memcpy(&full[static_cast<size_t>(i)], &bits, sizeof(bits));
    }
  }

  std::vector<std::unique_ptr<CollectiveBus>> buses =
      start_world(kWorld, kPort, 64 * 1024);
  require(!buses.empty(), "logit-gather bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<float>> got(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int rank = 0; rank < kWorld; ++rank) {
    workers.emplace_back([&, rank] {
      uint16_t* scratch = nullptr;
      bool arrived = false;
      const auto arrive_once = [&] {
        if (arrived) return;
        arrived = true;
        barrier.arrive_and_wait();
      };
      try {
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&scratch),
            sizeof(uint16_t) * dgpp::sampling_gather_scratch_elems(kVocab),
            cudaHostAllocDefault));
        arrive_once();
        const auto& slice = layout[static_cast<size_t>(rank)];
        dgpp::bus_gather_logits(*buses[static_cast<size_t>(rank)], rank,
                                kWorld, full.data() + slice.begin,
                                slice.count, slice.begin, kVocab, scratch,
                                test_wait_timeout_ms(),
                                &got[static_cast<size_t>(rank)]);
        cudaFreeHost(scratch);
        scratch = nullptr;
      } catch (const std::exception& error) {
        if (scratch != nullptr) cudaFreeHost(scratch);
        errors[static_cast<size_t>(rank)] = error.what();
        arrive_once();
      }
    });
  }
  for (auto& worker : workers) worker.join();
  for (int rank = 0; rank < kWorld; ++rank)
    require(errors[static_cast<size_t>(rank)].empty(),
            "logit-gather rank " + std::to_string(rank) + ": " +
                errors[static_cast<size_t>(rank)]);
  for (int rank = 0; rank < kWorld; ++rank) {
    require(got[static_cast<size_t>(rank)].size() == full.size(),
            "gathered vocabulary size");
    require(std::memcmp(got[static_cast<size_t>(rank)].data(), full.data(),
                        sizeof(float) * full.size()) == 0,
            "gathered logits differ bitwise on rank " + std::to_string(rank));
  }
}

// The eager fabric sampler end to end over the real bus — the closure
// glm_serve's eager engine and glm_gen_check run — against the sharded
// reference at the loader's layout, bitwise, across a run of steps whose
// context grows (penalties) and whose distributions alternate between a
// peaked shape the prefix resolves and a flat shape that takes the exact
// gather fallback. The counter advances once per step either way.
DGPP_TEST(glm_fabric_sample_matches_sharded_reference_loopback) {
  constexpr int kWorld = 2;
  constexpr int kVocab = 700;
  constexpr uint16_t kPort = 29928;
  constexpr int kSteps = 6;
  const std::vector<dgpp::glm_sample::VocabSlice> layout =
      dgpp::glm_sample::vocab_layout(kVocab, kWorld);
  dgpp::glm_sample::Params params;
  params.temperature = 1.0f;
  params.top_p = 0.95f;
  params.presence_penalty = 0.2f;
  params.frequency_penalty = 0.1f;
  params.logprobs = 2;
  const std::vector<int64_t> prompt{5, 9, 9, 400, 699};
  // Step s's full distribution: even steps peaked (a handful of tokens
  // carry the nucleus), odd steps flat (the 128-wide prefix holds ~37% of
  // the mass: fallback).
  const auto logits_for = [&](int step) {
    std::vector<float> v(kVocab, step % 2 == 0 ? -8.0f : 0.0f);
    if (step % 2 == 0) {
      v[static_cast<size_t>((37 * step + 3) % kVocab)] = 6.0f;
      v[static_cast<size_t>((37 * step + 351) % kVocab)] = 5.5f;
      v[static_cast<size_t>((37 * step + 690) % kVocab)] = 5.0f;
    } else {
      for (int i = 0; i < kVocab; ++i)
        v[static_cast<size_t>(i)] += 0.001f * static_cast<float>(i % 7);
    }
    return v;
  };

  // The centralized expectation: the sharded reference, one draw per step,
  // the context growing with each sampled token.
  std::vector<dgpp::glm_sample::Result> expected;
  {
    dgpp::glm_sample::Rng rng{0xabcdefull, 0};
    std::vector<int32_t> context(prompt.begin(), prompt.end());
    for (int step = 0; step < kSteps; ++step) {
      const std::vector<float> full = logits_for(step);
      expected.push_back(dgpp::glm_sample::sample_reference_sharded(
          full.data(), kVocab, layout, params, rng, context));
      context.push_back(expected.back().token);
    }
    require(rng.counter == kSteps, "one draw per step");
  }

  std::vector<std::unique_ptr<CollectiveBus>> buses =
      start_world(kWorld, kPort, 64 * 1024);
  require(!buses.empty(), "fabric-sample bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<dgpp::glm_sample::Result>> got(kWorld);
  std::vector<uint64_t> counters(kWorld, 0);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int rank = 0; rank < kWorld; ++rank) {
    workers.emplace_back([&, rank] {
      uint16_t* prefix_scratch = nullptr;
      uint16_t* gather_scratch = nullptr;
      bool arrived = false;
      const auto arrive_once = [&] {
        if (arrived) return;
        arrived = true;
        barrier.arrive_and_wait();
      };
      const auto release = [&] {
        if (prefix_scratch != nullptr) cudaFreeHost(prefix_scratch);
        if (gather_scratch != nullptr) cudaFreeHost(gather_scratch);
        prefix_scratch = gather_scratch = nullptr;
      };
      try {
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&prefix_scratch),
            sizeof(uint16_t) * dgpp::fabric_sampling_prefix_scratch_elems(kWorld),
            cudaHostAllocDefault));
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&gather_scratch),
            sizeof(uint16_t) * dgpp::sampling_gather_scratch_elems(kVocab),
            cudaHostAllocDefault));
        arrive_once();
        const dgpp::GenEngineAdapter::Sample sample = dgpp::make_fabric_sample(
            buses[static_cast<size_t>(rank)].get(), rank, kWorld,
            prefix_scratch, gather_scratch, kVocab, test_wait_timeout_ms());
        const auto& slice = layout[static_cast<size_t>(rank)];
        dgpp::glm_sample::Rng rng{0xabcdefull, 0};
        std::vector<int32_t> context(prompt.begin(), prompt.end());
        for (int step = 0; step < kSteps; ++step) {
          const std::vector<float> full = logits_for(step);
          dgpp::GlmDiagnosticModel::Outputs out;
          out.logits.assign(full.begin() + slice.begin,
                            full.begin() + slice.begin + slice.count);
          out.lm_vocab_begin = slice.begin;
          out.lm_vocab_count = slice.count;
          got[static_cast<size_t>(rank)].push_back(
              sample(out, params, rng, context));
          context.push_back(got[static_cast<size_t>(rank)].back().token);
        }
        counters[static_cast<size_t>(rank)] = rng.counter;
        release();
      } catch (const std::exception& error) {
        release();
        errors[static_cast<size_t>(rank)] = error.what();
        arrive_once();
      }
    });
  }
  for (auto& worker : workers) worker.join();
  for (int rank = 0; rank < kWorld; ++rank)
    require(errors[static_cast<size_t>(rank)].empty(),
            "fabric-sample rank " + std::to_string(rank) + ": " +
                errors[static_cast<size_t>(rank)]);
  for (int rank = 0; rank < kWorld; ++rank) {
    require(got[static_cast<size_t>(rank)].size() == expected.size(),
            "every step sampled");
    for (int step = 0; step < kSteps; ++step) {
      const auto& g = got[static_cast<size_t>(rank)][static_cast<size_t>(step)];
      const auto& e = expected[static_cast<size_t>(step)];
      require(g.token == e.token &&
                  std::memcmp(&g.logprob, &e.logprob, sizeof(float)) == 0 &&
                  g.top_logprobs == e.top_logprobs,
              "fabric sample differs from the sharded reference at step " +
                  std::to_string(step) + " on rank " + std::to_string(rank));
    }
    require(counters[static_cast<size_t>(rank)] == kSteps,
            "the fabric sampler consumed one draw per step");
  }
}

// The speculative step's row deciders over the real bus — bus_spec_accept
// (the accept test and residual, with the gather fallback) and bus_sample_row
// (row 1) — against the speculative reference at the loader's layout,
// bitwise, over a run of steps whose drafts are sometimes the likely token
// and sometimes an unlikely one, with a flat row that forces the fallback.
// Two draws per step on every rank.
DGPP_TEST(glm_spec_accept_matches_reference_loopback) {
  constexpr int kWorld = 2;
  constexpr int kVocab = 700;
  constexpr uint16_t kPort = 29930;
  constexpr int kSteps = 8;
  const std::vector<dgpp::glm_sample::VocabSlice> layout =
      dgpp::glm_sample::vocab_layout(kVocab, kWorld);
  dgpp::glm_sample::Params params;
  params.temperature = 1.0f;
  params.top_p = 0.95f;
  params.presence_penalty = 0.15f;
  const std::vector<int64_t> prompt{3, 3, 250, 699};
  const auto row_for = [&](int step, int row) {
    // Even steps peaked (resolves in the 128-wide prefix), odd steps flat
    // (the fallback); row 1 differs from row 0 by a shift.
    std::vector<float> v(kVocab, step % 2 == 0 ? -7.0f : 0.0f);
    if (step % 2 == 0) {
      // The argmax holds ~98% of the mass: a draft equal to it is accepted
      // unless the accept draw lands in the top two percent.
      v[static_cast<size_t>((41 * step + 5 + 100 * row) % kVocab)] = 6.0f;
      v[static_cast<size_t>((41 * step + 360 + 100 * row) % kVocab)] = 2.0f;
      v[static_cast<size_t>((41 * step + 690 + 100 * row) % kVocab)] = 1.5f;
    } else {
      for (int i = 0; i < kVocab; ++i)
        v[static_cast<size_t>(i)] += 0.002f * static_cast<float>((i + row) % 5);
    }
    return v;
  };
  // Drafts: the row-0 argmax on steps 0,1 mod 4 (likely), a tail token else.
  const auto draft_for = [&](int step) -> int32_t {
    const std::vector<float> v = row_for(step, 0);
    if (step % 4 < 2)
      return dgpp::glm_sample::local_max(v.data(), kVocab, 0).id;
    return static_cast<int32_t>((step * 97 + 13) % kVocab);
  };

  // The centralized expectation.
  struct Expected {
    dgpp::glm_sample::SpecStepReference ref;
    uint64_t counter_after;
  };
  std::vector<Expected> expected;
  {
    dgpp::glm_sample::Rng rng{0x5eedull, 0};
    std::vector<int32_t> context(prompt.begin(), prompt.end());
    for (int step = 0; step < kSteps; ++step) {
      const std::vector<float> r0 = row_for(step, 0);
      const std::vector<float> r1 = row_for(step, 1);
      const int32_t draft = draft_for(step);
      Expected e;
      e.ref = dgpp::glm_sample::spec_reference_sharded(
          r0.data(), r1.data(), kVocab, layout, draft, params, rng, context);
      e.counter_after = rng.counter;
      expected.push_back(e);
      if (e.ref.accepted) context.push_back(draft);
      context.push_back(e.ref.accepted ? e.ref.winners[1] : e.ref.winners[0]);
    }
  }

  std::vector<std::unique_ptr<CollectiveBus>> buses =
      start_world(kWorld, kPort, 64 * 1024);
  require(!buses.empty(), "spec-accept bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<dgpp::glm_sample::SpecStepReference>> got(kWorld);
  std::vector<std::vector<uint64_t>> counters(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int rank = 0; rank < kWorld; ++rank) {
    workers.emplace_back([&, rank] {
      uint16_t* prefix_scratch = nullptr;
      uint16_t* gather_scratch = nullptr;
      bool arrived = false;
      const auto arrive_once = [&] {
        if (arrived) return;
        arrived = true;
        barrier.arrive_and_wait();
      };
      const auto release = [&] {
        if (prefix_scratch) cudaFreeHost(prefix_scratch);
        if (gather_scratch) cudaFreeHost(gather_scratch);
        prefix_scratch = gather_scratch = nullptr;
      };
      try {
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&prefix_scratch),
            sizeof(uint16_t) * dgpp::fabric_sampling_prefix_scratch_elems(kWorld),
            cudaHostAllocDefault));
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&gather_scratch),
            sizeof(uint16_t) * dgpp::sampling_gather_scratch_elems(kVocab),
            cudaHostAllocDefault));
        arrive_once();
        CollectiveBus& bus = *buses[static_cast<size_t>(rank)];
        const auto& slice = layout[static_cast<size_t>(rank)];
        std::vector<float> gather_buffer;
        dgpp::glm_sample::Rng rng{0x5eedull, 0};
        std::vector<int32_t> context(prompt.begin(), prompt.end());
        for (int step = 0; step < kSteps; ++step) {
          const std::vector<float> r0 = row_for(step, 0);
          const std::vector<float> r1 = row_for(step, 1);
          const int32_t draft = draft_for(step);
          dgpp::glm_sample::SpecStepReference s;
          const dgpp::glm_sample::SpecPrefixDecision d0 = dgpp::bus_spec_accept(
              bus, rank, kWorld, r0.data() + slice.begin, slice.count,
              slice.begin, kVocab, draft, params, rng, context,
              dgpp::kSamplingCandidates, prefix_scratch, gather_scratch,
              test_wait_timeout_ms(), &gather_buffer);
          s.accepted = d0.accepted;
          s.row0 = d0.result;
          s.winners[0] = d0.result.token;
          if (d0.accepted) {
            context.push_back(draft);
            s.row1 = dgpp::bus_sample_row(
                bus, rank, kWorld, r1.data() + slice.begin, slice.count,
                slice.begin, kVocab, params, rng, context,
                dgpp::kSamplingCandidates, prefix_scratch, gather_scratch,
                test_wait_timeout_ms(), &gather_buffer);
            s.winners[1] = s.row1.token;
          }
          context.push_back(s.accepted ? s.winners[1] : s.winners[0]);
          got[static_cast<size_t>(rank)].push_back(s);
          counters[static_cast<size_t>(rank)].push_back(rng.counter);
        }
        release();
      } catch (const std::exception& error) {
        release();
        errors[static_cast<size_t>(rank)] = error.what();
        arrive_once();
      }
    });
  }
  for (auto& worker : workers) worker.join();
  for (int rank = 0; rank < kWorld; ++rank)
    require(errors[static_cast<size_t>(rank)].empty(),
            "spec-accept rank " + std::to_string(rank) + ": " +
                errors[static_cast<size_t>(rank)]);
  int accepts = 0;
  for (int rank = 0; rank < kWorld; ++rank) {
    require(got[static_cast<size_t>(rank)].size() == expected.size(), "every step ran");
    for (int step = 0; step < kSteps; ++step) {
      const auto& g = got[static_cast<size_t>(rank)][static_cast<size_t>(step)];
      const auto& e = expected[static_cast<size_t>(step)].ref;
      const auto same = [](const dgpp::glm_sample::Result& a,
                           const dgpp::glm_sample::Result& b) {
        return a.token == b.token &&
               std::memcmp(&a.logprob, &b.logprob, sizeof(float)) == 0;
      };
      require(g.accepted == e.accepted && g.winners[0] == e.winners[0] &&
                  g.winners[1] == e.winners[1] && same(g.row0, e.row0) &&
                  (!e.accepted || same(g.row1, e.row1)),
              "bus speculative step " + std::to_string(step) + " on rank " +
                  std::to_string(rank) + " differs from the reference");
      require(counters[static_cast<size_t>(rank)][static_cast<size_t>(step)] ==
                  expected[static_cast<size_t>(step)].counter_after,
              "counter differs at step " + std::to_string(step));
      if (rank == 0 && e.accepted) ++accepts;
    }
  }
  require(accepts > 0 && accepts < kSteps, "the run must accept and reject");
}

// The eager sampled speculator on the MTP fixture over the real bus: every
// rank produces the same transcript (the accept test, the residual, the
// row-1 sample and the greedy draft are collectives or rank-identical
// decisions) and the RNG consumes one draw for the prefill and two per
// step. The fixture's untrained draft block rarely proposes a token the
// sampler keeps, so acceptance is reported, not required; the exactness of
// the decisions is the synthetic gate above's.
DGPP_TEST(glm_tp_sampled_speculator_loopback_rank_identical) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt = make_tokens(9, cfg.vocab_size);
  constexpr int kTokens = 12;
  constexpr int kWorld = 2;
  const int max_tokens = static_cast<int>(prompt.size()) + kTokens + 4;
  dgpp::glm_sample::Params params;
  params.temperature = 0.6f;  // peaked enough for some drafts to stand
  params.top_p = 0.95f;
  params.presence_penalty = 0.1f;

  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29931);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<int32_t>> seqs(kWorld);
  std::vector<int> steps(kWorld, 0), accepted(kWorld, 0);
  std::vector<uint64_t> counters(kWorld, 0);
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
      uint16_t* scratch = nullptr;
      uint16_t* prefix_scratch = nullptr;
      uint16_t* gather_scratch = nullptr;
      const auto release = [&] {
        if (scratch) cudaFreeHost(scratch);
        if (prefix_scratch) cudaFreeHost(prefix_scratch);
        if (gather_scratch) cudaFreeHost(gather_scratch);
        scratch = prefix_scratch = gather_scratch = nullptr;
      };
      try {
        CollectiveBus& bus = *buses[static_cast<size_t>(r)];
        GlmBusBoundaryReducer reducer(bus, test_wait_timeout_ms());
        GlmDiagnosticModel shard(cfg, dir, max_tokens, 128, &reducer, r,
                                 kWorld, GlmResidency::Streaming,
                                 GlmHeadSharding::VocabSharded,
                                 /*max_requests=*/1, /*mtp=*/true);
        DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                                   sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld),
                                   cudaHostAllocDefault));
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&prefix_scratch),
            sizeof(uint16_t) * dgpp::fabric_sampling_prefix_scratch_elems(kWorld),
            cudaHostAllocDefault));
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&gather_scratch),
            sizeof(uint16_t) * dgpp::sampling_gather_scratch_elems(cfg.vocab_size),
            cudaHostAllocDefault));
        arrive_once();
        const auto pick_rows = [&](const std::vector<Candidate>& locals) {
          return dgpp::bus_greedy_pick_rows(bus, r, kWorld, locals, scratch,
                                            test_wait_timeout_ms());
        };
        const dgpp::GenEngineAdapter::Sample row1 = dgpp::make_fabric_sample(
            &bus, r, kWorld, prefix_scratch, gather_scratch, cfg.vocab_size,
            test_wait_timeout_ms());
        dgpp::glm_sample::Rng rng{0x77ull, 0};
        const GlmDiagnosticModel::Outputs pre = shard.session_prefill(prompt);
        const std::vector<int32_t> prompt_context(prompt.begin(), prompt.end());
        const int32_t first = row1(pre, params, rng, prompt_context).token;
        dgpp::SampledSpeculator spec(
            shard, 0, pick_rows,
            dgpp::make_fabric_spec_row0(&bus, r, kWorld, prefix_scratch,
                                        gather_scratch, cfg.vocab_size,
                                        test_wait_timeout_ms()),
            row1, params, rng, prompt);
        spec.start(first);
        std::vector<int32_t>& got = seqs[static_cast<size_t>(r)];
        while (static_cast<int>(got.size()) < kTokens) {
          const std::vector<int32_t> committed = spec.step();
          got.insert(got.end(), committed.begin(), committed.end());
        }
        got.resize(kTokens);
        steps[static_cast<size_t>(r)] = spec.steps();
        accepted[static_cast<size_t>(r)] = spec.accepted_drafts();
        counters[static_cast<size_t>(r)] = spec.rng().counter;
        release();
      } catch (const std::exception& e) {
        release();
        errors[static_cast<size_t>(r)] =
            "rank " + std::to_string(r) + ": " + e.what();
        DGPP_LOG_ERROR("sampled speculator rank {} failed: {}", r, e.what());
        arrive_once();
      }
    });
  }
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r)
    require(errors[static_cast<size_t>(r)].empty(), errors[static_cast<size_t>(r)]);
  for (int r = 1; r < kWorld; ++r) {
    require(seqs[static_cast<size_t>(r)] == seqs[0],
            "sampled speculator transcript differs across ranks");
    require(steps[static_cast<size_t>(r)] == steps[0] &&
                accepted[static_cast<size_t>(r)] == accepted[0] &&
                counters[static_cast<size_t>(r)] == counters[0],
            "sampled speculator step/acceptance/counter differ across ranks");
  }
  require(counters[0] == 1 + 2 * static_cast<uint64_t>(steps[0]),
          "one draw for the prefill and two per step");
  for (const int32_t t : seqs[0])
    require(t >= 0 && t < cfg.vocab_size, "token in range");
  DGPP_LOG_INFO("sampled speculator w2: {} tokens in {} steps, {} drafts "
                "accepted; transcripts identical across ranks",
                kTokens, steps[0], accepted[0]);
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
              id, test_wait_timeout_ms());
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
        GlmBusBoundaryReducer reducer(*buses[static_cast<size_t>(r)], test_wait_timeout_ms());
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
              test_wait_timeout_ms());
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
        GlmBusBoundaryReducer reducer(*buses[static_cast<size_t>(r)], test_wait_timeout_ms());
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
// Speculative verify + rollback (DESIGN §9), world 1. A T-row verify's
// rows must be BITWISE the single-token steps over the same tokens, and a
// rollback to `a` rows must leave the state bitwise where `a` steps would
// have — pinned by stepping ON from the rolled-back state and comparing
// the continuation's logits with the sequential transcript's. Covers:
// accept-all (T=2, T=4), reject to 1 of 2, reject to 2 of 4 (mid-batch),
// and a verify whose rows straddle a kpool boundary (the DSA ring/pool
// completion path). Every comparison is bitwise: same kernels, same op
// order per row, fp32 state.
DGPP_TEST(glm_tp_session_verify_rollback_matches_sequential_bitwise) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  // GIVEN a prompt whose length puts the first verify rows mid-pool
  // (9 % 4 == 1), a tokens stream, and the sequential reference transcript:
  const std::vector<int64_t> prompt = make_tokens(9, cfg.vocab_size);
  const std::vector<int64_t> toks = make_tokens(12, cfg.vocab_size);
  const int max_tokens = static_cast<int>(prompt.size() + toks.size()) + 2;
  const size_t V = static_cast<size_t>(cfg.vocab_size);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  GlmDiagnosticModel model(cfg, dir, max_tokens, 128);

  std::vector<std::vector<float>> seq_logits;
  std::vector<std::vector<uint16_t>> seq_hidden;
  model.session_prefill(prompt);
  for (int64_t t : toks) {
    const GlmDiagnosticModel::Outputs o = model.session_step(t);
    seq_logits.push_back(o.logits);
    seq_hidden.push_back(o.final_hidden_bits);
  }
  const auto row_of = [&](const std::vector<float>& m, size_t r) {
    return std::vector<float>(m.begin() + r * V, m.begin() + (r + 1) * V);
  };
  const auto hrow_of = [&](const std::vector<uint16_t>& m, size_t r) {
    return std::vector<uint16_t>(m.begin() + r * H, m.begin() + (r + 1) * H);
  };
  // Checks the first `good` rows of a T-row verify of toks[i..) against
  // the sequential rows (rows >= good carried wrong tokens).
  const auto check_rows = [&](const GlmDiagnosticModel::Outputs& o, size_t i,
                              size_t T, size_t good, const char* what) {
    require(o.logits.size() == T * V && o.final_hidden_bits.size() == T * H,
            std::string(what) + ": verify output shape");
    for (size_t r = 0; r < good; ++r) {
      require(row_of(o.logits, r) == seq_logits[i + r],
              std::string(what) + ": verify row " + std::to_string(r) +
                  " logits must be BITWISE the sequential step's");
      require(hrow_of(o.final_hidden_bits, r) == seq_hidden[i + r],
              std::string(what) + ": verify row " + std::to_string(r) +
                  " hidden must be BITWISE the sequential step's");
    }
  };
  const auto slice = [&](size_t i, size_t n) {
    return std::vector<int64_t>(toks.begin() + i, toks.begin() + i + n);
  };

  // WHEN the same tokens go through verify/rollback in a mixed schedule:
  model.session_prefill(prompt);
  size_t i = 0;
  // T=2, accept both (positions 9,10).
  check_rows(model.session_verify(0, slice(i, 2)), i, 2, 2, "accept-all T=2");
  i += 2;
  // T=2 with a WRONG second token, reject to 1: the retracted row must
  // leave no trace — the next step over the right token matches.
  {
    std::vector<int64_t> ids = slice(i, 2);
    ids[1] = (ids[1] + 7) % cfg.vocab_size;  // the rejected draft
    const GlmDiagnosticModel::Outputs o = model.session_verify(0, ids);
    check_rows(o, i, 2, 1, "reject-to-1 row 0");
    model.session_rollback(0, 1);
    require(model.session_position(0) == static_cast<int64_t>(prompt.size() + i + 1),
            "rollback rewinds the position by the rejected rows");
    i += 1;
  }
  // T=4 straddling the pool boundary at position 12 (i=3: positions
  // 12..15), accept all.
  check_rows(model.session_verify(0, slice(i, 4)), i, 4, 4, "accept-all T=4");
  i += 4;
  // T=4 with wrong rows 2,3: reject to 2 (a mid-batch snapshot).
  {
    std::vector<int64_t> ids = slice(i, 4);
    ids[2] = (ids[2] + 3) % cfg.vocab_size;
    ids[3] = (ids[3] + 5) % cfg.vocab_size;
    const GlmDiagnosticModel::Outputs o = model.session_verify(0, ids);
    check_rows(o, i, 4, 2, "reject-to-2 rows 0,1");
    model.session_rollback(0, 2);
    i += 2;
  }
  // THEN plain steps from the rolled-back state continue the sequential
  // transcript bitwise:
  for (; i < toks.size(); ++i) {
    const GlmDiagnosticModel::Outputs o = model.session_step(toks[i]);
    require(o.logits == seq_logits[i],
            "post-rollback step " + std::to_string(i) +
                " logits must be BITWISE the sequential step's");
  }
  // rollback bounds
  bool threw = false;
  try { model.session_rollback(0, 0); } catch (const std::invalid_argument&) { threw = true; }
  require(threw, "rollback to 0 rows must be rejected");
}

// The MTP draft block (DESIGN §9), world 1: deterministic across model
// instances; a two-row draft's last row is BITWISE the same row drafted
// alone after a one-row draft (the block's rows are causal + per-token,
// exactly as the main stack's verify rows are); the prefill filled the
// block's cache (drafting after a prompt works at all); and the row-
// accounting invariant (draft exactly the rows since the last draft) is
// enforced. The block's numerics against the reference are not pinnable
// here (no host reference); the real model's acceptance rate is that gate.
DGPP_TEST(glm_tp_mtp_draft_batched_matches_sequential_bitwise) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  // GIVEN a prompt and the tokens the "picks" will feed:
  const std::vector<int64_t> prompt = make_tokens(9, cfg.vocab_size);
  const std::vector<int64_t> toks = make_tokens(6, cfg.vocab_size);
  const int max_tokens = static_cast<int>(prompt.size() + toks.size()) + 4;
  const size_t V = static_cast<size_t>(cfg.vocab_size);
  GlmDiagnosticModel a(cfg, dir, max_tokens, 128, nullptr, 0, 1,
                       GlmResidency::Streaming, GlmHeadSharding::Full, 1,
                       /*mtp=*/true);
  GlmDiagnosticModel b(cfg, dir, max_tokens, 128, nullptr, 0, 1,
                       GlmResidency::Streaming, GlmHeadSharding::Full, 1,
                       /*mtp=*/true);
  require(a.mtp_enabled(), "mtp enabled");

  // WHEN both prefill (the block runs over rows 0..P-2) and draft row P-1:
  a.session_prefill(prompt);
  b.session_prefill(prompt);
  const GlmDiagnosticModel::Outputs da = a.session_draft(0, {toks[0]});
  const GlmDiagnosticModel::Outputs db = b.session_draft(0, {toks[0]});
  require(da.logits.size() == V, "draft logits are one vocab row");
  require(da.logits == db.logits, "draft must be deterministic across instances");

  // A verifies two rows and accepts both, then drafts the two rows in ONE
  // call; B takes the same rows one at a time (verify 1, draft 1, twice).
  a.session_verify(0, {toks[0], toks[1]});
  const GlmDiagnosticModel::Outputs two = a.session_draft(0, {toks[1], toks[2]});
  b.session_verify(0, {toks[0]});
  (void)b.session_draft(0, {toks[1]});
  b.session_verify(0, {toks[1]});
  const GlmDiagnosticModel::Outputs one = b.session_draft(0, {toks[2]});
  // THEN the drafted distributions agree bitwise:
  require(two.logits == one.logits,
          "a two-row draft's last row must be BITWISE the same row drafted "
          "alone (causal DSA with self-include, per-token MoE, row-independent "
          "GEMV)");

  // A rejected verify row, rolled back, then the one accepted row drafted:
  a.session_verify(0, {toks[2], toks[3]});
  a.session_rollback(0, 1);
  (void)a.session_draft(0, {toks[3]});
  b.session_verify(0, {toks[2]});
  (void)b.session_draft(0, {toks[3]});
  a.session_verify(0, {toks[3]});
  b.session_verify(0, {toks[3]});
  require(a.session_draft(0, {toks[4]}).logits == b.session_draft(0, {toks[4]}).logits,
          "drafts after a rollback must match the sequential path bitwise");

  // The invariant: the draft must cover exactly the rows since the last one.
  a.session_verify(0, {toks[4], toks[5]});
  bool threw = false;
  try { (void)a.session_draft(0, {toks[5]}); } catch (const std::invalid_argument&) { threw = true; }
  require(threw, "a draft that does not reach the session position must throw");
  (void)a.session_draft(0, {toks[5], toks[0]});
}

// The exit criterion at fixture scale (DESIGN §9, PLAN M8): the greedy
// speculative loop — sharded heads, bus folds of T=2 verify rows, the
// two-row bus pick, rollbacks, drafts — must produce the plain greedy
// session's transcript EXACTLY, on every rank. The fixture's random
// draft layer accepts what it accepts (reported); the equality holds
// regardless, which is the point: MTP changes the cost, never the output.
DGPP_TEST(glm_tp_speculative_loopback_matches_plain_greedy) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt = make_tokens(9, cfg.vocab_size);
  constexpr int kTokens = 10;  // generated tokens to compare
  constexpr int kWorld = 4;
  // Verify rows may run one past the compared length; the block trails.
  const int max_tokens = static_cast<int>(prompt.size()) + kTokens + 4;

  // ---- reference: the plain greedy session, world 1, no MTP ------------
  std::vector<int32_t> plain;
  {
    GlmDiagnosticModel model(cfg, dir, max_tokens, 128);
    GlmDiagnosticModel::Outputs out = model.session_prefill(prompt);
    int32_t tok = local_max(out.logits.data(), cfg.vocab_size, 0).id;
    for (int s = 0; s < kTokens; ++s) {
      plain.push_back(tok);
      out = model.session_step(tok);
      tok = local_max(out.logits.data(), cfg.vocab_size, 0).id;
    }
  }

  // ---- world 1 speculative (the eager driver, identity pick) ------------
  {
    GlmDiagnosticModel model(cfg, dir, max_tokens, 128, nullptr, 0, 1,
                             GlmResidency::Streaming, GlmHeadSharding::Full, 1,
                             /*mtp=*/true);
    dgpp::GreedySpeculator spec(model, 0, [](const std::vector<Candidate>& l) {
      std::vector<int32_t> w;
      for (const Candidate& c : l) w.push_back(c.id);
      return w;
    });
    const GlmDiagnosticModel::Outputs out = model.session_prefill(prompt);
    spec.start(local_max(out.logits.data(), cfg.vocab_size, 0).id);
    std::vector<int32_t> got;
    while (static_cast<int>(got.size()) < kTokens)
      for (int32_t t : spec.step()) got.push_back(t);
    got.resize(kTokens);
    require(got == plain, "world-1 speculative transcript != plain greedy");
    DGPP_LOG_INFO("speculative w1: {} steps for {} tokens, {} drafts accepted",
                  spec.steps(), kTokens, spec.accepted_drafts());
  }

  // ---- world 4: sharded heads, bus folds, two-row bus pick --------------
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29917);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<int32_t>> rank_seqs(kWorld);
  std::vector<int> rank_steps(kWorld, 0);
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
      uint16_t* scratch = nullptr;
      try {
        CollectiveBus& bus = *buses[static_cast<size_t>(r)];
        GlmBusBoundaryReducer reducer(bus, test_wait_timeout_ms());
        // The reference at world 4 is the PLAIN sharded session on the
        // same bus (the fold order differs from world 1 by rounding, and
        // the random fixture's vocab of 96 has near ties everywhere).
        GlmDiagnosticModel plain_shard(cfg, dir, max_tokens, 128, &reducer,
                                       r, kWorld, GlmResidency::Streaming,
                                       GlmHeadSharding::VocabSharded);
        GlmDiagnosticModel shard(cfg, dir, max_tokens, 128, &reducer, r,
                                 kWorld, GlmResidency::Streaming,
                                 GlmHeadSharding::VocabSharded, 1,
                                 /*mtp=*/true);
        DGPP_CUDA_OK(cudaMallocManaged(
            &scratch, sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld)));
        arrive_once();
        const auto pick1 = [&](const GlmDiagnosticModel::Outputs& o) {
          return dgpp::bus_greedy_pick(
              bus, r, kWorld,
              local_max(o.logits.data(), o.lm_vocab_count, o.lm_vocab_begin),
              scratch, test_wait_timeout_ms());
        };
        std::vector<int32_t> plain4;
        {
          GlmDiagnosticModel::Outputs o = plain_shard.session_prefill(prompt);
          int32_t tok = pick1(o);
          for (int s = 0; s < kTokens; ++s) {
            plain4.push_back(tok);
            o = plain_shard.session_step(tok);
            tok = pick1(o);
          }
        }
        dgpp::GreedySpeculator spec(
            shard, 0, [&](const std::vector<Candidate>& locals) {
              return dgpp::bus_greedy_pick_rows(bus, r, kWorld, locals,
                                                scratch, test_wait_timeout_ms());
            });
        const GlmDiagnosticModel::Outputs out = shard.session_prefill(prompt);
        spec.start(pick1(out));
        std::vector<int32_t>& got = rank_seqs[static_cast<size_t>(r)];
        while (static_cast<int>(got.size()) < kTokens)
          for (int32_t t : spec.step()) got.push_back(t);
        got.resize(kTokens);
        rank_steps[static_cast<size_t>(r)] = spec.steps();
        if (got != plain4)
          throw std::runtime_error(
              "speculative transcript over the bus != the plain sharded "
              "session's on the same bus");
        // The ACCEPT path over the bus (the random draft layer never
        // takes it): feed the plain transcript's next token as the
        // "draft" so every verify accepts both rows, then the two-row
        // draft runs the block's T=2 folds and its two-row DSA/MoE.
        {
          GlmDiagnosticModel::Outputs o = shard.session_prefill(prompt);
          int32_t next = pick1(o);
          std::vector<int32_t> committed;
          size_t i = 0;
          (void)shard.session_draft(0, {next});  // the block's row P-1
          while (i + 1 < plain4.size()) {
            const std::vector<int64_t> fed{next, plain4[i + 1]};
            o = shard.session_verify(0, fed);
            const std::vector<int32_t> winners = dgpp::bus_greedy_pick_rows(
                bus, r, kWorld, dgpp::local_row_maxes(o, 2), scratch, test_wait_timeout_ms());
            const dgpp::SpecVerdict v = dgpp::judge_verify(fed, winners);
            if (v.accepted != 2)
              throw std::runtime_error(
                  "a verify fed the true next token must accept both rows "
                  "(step " + std::to_string(i) + ")");
            for (int32_t t : v.committed) committed.push_back(t);
            next = v.next;
            (void)shard.session_draft(0, v.draft_rows);  // T=2 draft
            i += 2;
          }
          committed.push_back(next);
          committed.resize(plain4.size());
          if (committed != plain4)
            throw std::runtime_error(
                "accept-all verify transcript over the bus != plain");
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
  for (int r = 1; r < kWorld; ++r)
    require(rank_seqs[static_cast<size_t>(r)] == rank_seqs[0],
            "rank " + std::to_string(r) +
                ": speculative transcript differs across ranks");
  DGPP_LOG_INFO("speculative w4: {} steps for {} tokens; transcript == the "
                "plain sharded session's on every rank",
                rank_steps[0], kTokens);
}

// ---------------------------------------------------------------------------
// The on-device step (DESIGN §9, phases A+B): the T=2 verify recorded as a
// graph WITH the pick behind it — glm_pick_local, the gather as a recorded
// collective node, glm_pick_verdict — and the COMMIT behind the pick (the
// conditional rollback + the device position advance), the rows' positions
// off the device position, replayed over the loopback bus at world 4. The
// host never rolls back, never stages a position, never admits blocks per
// step. Pins, per step: the device verdict (winners, accepted, next) == the
// host pick + judge over the SAME replayed logits; the draft's eager device
// pick == the host pick; the host position mirror == the device's; and the
// whole transcript == the plain sharded session's, identical on every rank
// (the random fixture's draft is rejected nearly every step, so the device
// rollback runs nearly every step — a wrong copy would derail the KDA
// state and the transcript with it). Also the first in-process capture of
// the model's decode graph (the fabric app was the only exerciser before).
// ---------------------------------------------------------------------------
DGPP_TEST(glm_tp_device_pick_graph_loopback_matches_host_pick) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt = make_tokens(9, cfg.vocab_size);
  constexpr int kTokens = 12;
  constexpr int kWorld = 4;
  const int max_tokens = static_cast<int>(prompt.size()) + kTokens + 4;

  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29919);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<int32_t>> rank_seqs(kWorld);
  std::vector<int> rank_steps(kWorld, 0);
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
      uint16_t* scratch = nullptr;
      cudaGraphExec_t graph_exec = nullptr;
      try {
        CollectiveBus& bus = *buses[static_cast<size_t>(r)];
        GlmBusBoundaryReducer reducer(bus, test_wait_timeout_ms());
        GlmDiagnosticModel plain_shard(cfg, dir, max_tokens, 128, &reducer,
                                       r, kWorld, GlmResidency::Streaming,
                                       GlmHeadSharding::VocabSharded);
        // Resident: the graph era's precondition (session_graph_prepare).
        GlmDiagnosticModel shard(cfg, dir, max_tokens, 128, &reducer, r,
                                 kWorld, GlmResidency::Resident,
                                 GlmHeadSharding::VocabSharded, 1,
                                 /*mtp=*/true);
        shard.set_decode_route_traces(false);
        // Kernels-only graph (glm_check_decode_graph): the tail's D2H mirror
        // nodes deadlocked this world at one hardware connection (10/10) —
        // session_graph_outputs copies the tail eagerly after each replay.
        shard.set_decode_tail_mirrors(false);
        dgpp::GlmDevicePicker picker(bus, r, kWorld, test_wait_timeout_ms());
        dgpp::GlmGraphRecordReducer recorder(bus, shard.stream());
        DGPP_CUDA_OK(cudaMallocManaged(
            &scratch, sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld)));
        arrive_once();
        const auto host_pick = [&](const GlmDiagnosticModel::Outputs& o,
                                   int rows) {
          return dgpp::bus_greedy_pick_rows(bus, r, kWorld,
                                            dgpp::local_row_maxes(o, rows),
                                            scratch, test_wait_timeout_ms());
        };
        // ---- reference: the plain sharded session on the same bus ------
        std::vector<int32_t> plain4;
        {
          GlmDiagnosticModel::Outputs o = plain_shard.session_prefill(prompt);
          int32_t tok = host_pick(o, 1)[0];
          for (int s = 0; s < kTokens; ++s) {
            plain4.push_back(tok);
            o = plain_shard.session_step(tok);
            tok = host_pick(o, 1)[0];
          }
        }
        // ---- the speculative session: prefill, first pick, first draft --
        GlmDiagnosticModel::Outputs pre = shard.session_prefill(prompt);
        int32_t next = host_pick(pre, 1)[0];
        const auto device_inputs = [&](int rows) {
          dgpp::GlmDevicePicker::Inputs in;
          in.logits = shard.device_logits();
          in.rows = rows;
          in.vocab_count = shard.lm_vocab_count();
          in.vocab_begin = shard.lm_vocab_begin();
          in.fed = shard.device_tokens();
          return in;
        };
        // The draft's pick, eager on the device; checked against the host.
        const auto draft_after = [&](const std::vector<int64_t>& rows) {
          const GlmDiagnosticModel::Outputs d = shard.session_draft(0, rows);
          const int32_t host = host_pick(d, 1)[0];
          const dgpp::GlmPickVerdict& v =
              picker.run(shard.stream(), device_inputs(1));
          if (v.accepted != 1 || v.next != host || v.winners[0] != host)
            throw std::runtime_error("draft: device pick " +
                                     std::to_string(v.next) + " != host " +
                                     std::to_string(host));
          return v.next;
        };
        int32_t draft = draft_after({next});

        // ---- capture: the T=2 verify + the recorded pick + the commit --
        shard.session_graph_prepare();
        shard.session_reserve_blocks(0, max_tokens);
        dgpp::GlmBoundaryReducer* eager = shard.set_boundary(&recorder);
        std::string gerr;
        require(bus.graph_record_begin(&gerr), "graph_record_begin: " + gerr);
        cudaGraph_t graph = nullptr;
        DGPP_CUDA_OK(cudaStreamBeginCapture(shard.stream(),
                                            cudaStreamCaptureModeThreadLocal));
        shard.session_graph_capture_step(0, std::vector<int64_t>{next, next},
                                         /*device_positions=*/true);
        picker.record(shard.stream(), device_inputs(2));
        shard.session_graph_capture_commit(0, picker.device_verdict());
        DGPP_CUDA_OK(cudaStreamEndCapture(shard.stream(), &graph));
        require(graph != nullptr, "capture produced no graph");
        require(bus.graph_record_end(&gerr), "graph_record_end: " + gerr);
        shard.set_boundary(eager);
        dgpp::glm_check_decode_graph(graph, r, "device-pick graph");
        DGPP_CUDA_OK(cudaGraphInstantiate(&graph_exec, graph, nullptr,
                                          nullptr, 0));
        cudaGraphDestroy(graph);

        // ---- the replay loop -------------------------------------------
        std::vector<int32_t>& got = rank_seqs[static_cast<size_t>(r)];
        int steps = 0;
        while (static_cast<int>(got.size()) < kTokens) {
          const std::vector<int64_t> fed{next, draft};
          shard.session_graph_stage(0, fed);
          require(bus.graph_replay_arm(&gerr), "graph_replay_arm: " + gerr);
          DGPP_CUDA_OK(cudaGraphLaunch(graph_exec, shard.stream()));
          DGPP_CUDA_OK(cudaStreamSynchronize(shard.stream()));
          require(bus.graph_replay_finish(test_wait_timeout_ms(), &gerr),
                  "graph_replay_finish: " + gerr);
          const GlmDiagnosticModel::Outputs out = shard.session_graph_outputs(0);
          const dgpp::GlmPickVerdict& v = picker.verdict();
          // The host judge over the same logits the graph produced.
          const std::vector<int32_t> winners = host_pick(out, 2);
          const dgpp::SpecVerdict want = dgpp::judge_verify(fed, winners);
          const std::vector<Candidate> locals = dgpp::local_row_maxes(out, 2);
          for (int row = 0; row < 2; ++row) {
            if (v.winners[row] != winners[static_cast<size_t>(row)])
              throw std::runtime_error(
                  "step " + std::to_string(steps) + " row " +
                  std::to_string(row) + ": device winner " +
                  std::to_string(v.winners[row]) + " != host " +
                  std::to_string(winners[static_cast<size_t>(row)]));
            const Candidate& local = locals[static_cast<size_t>(row)];
            if (picker.local(row).best_id != local.id ||
                std::memcmp(&picker.local(row).best_logit, &local.logit, 4) != 0)
              throw std::runtime_error("device local argmax != host's");
          }
          if (v.accepted != want.accepted || v.next != want.next)
            throw std::runtime_error("device verdict != judge_verify");
          // The device rolled back and advanced; the host mirror follows.
          const int64_t before = shard.session_position(0);
          shard.session_graph_settle(0, v.accepted);
          if (shard.session_position(0) != before + v.accepted)
            throw std::runtime_error("host position mirror did not settle");
          ++steps;
          for (int row = 0; row < v.accepted; ++row)
            got.push_back(static_cast<int32_t>(fed[static_cast<size_t>(row)]));
          next = v.next;
          draft = draft_after(
              std::vector<int64_t>(v.winners, v.winners + v.accepted));
        }
        got.resize(kTokens);
        rank_steps[static_cast<size_t>(r)] = steps;
        if (got != plain4)
          throw std::runtime_error(
              "graph verify + device pick transcript != the plain sharded "
              "session's on the same bus");
        cudaGraphExecDestroy(graph_exec);
        graph_exec = nullptr;
        cudaFree(scratch);
        scratch = nullptr;
      } catch (const std::exception& e) {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
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
  for (int r = 1; r < kWorld; ++r)
    require(rank_seqs[static_cast<size_t>(r)] == rank_seqs[0],
            "rank " + std::to_string(r) + ": transcript differs across ranks");
  DGPP_LOG_INFO("device pick graph w4: {} steps for {} tokens; device "
                "verdicts == host picks every step, transcript == plain",
                rank_steps[0], kTokens);
}

// ---------------------------------------------------------------------------
// The on-device step, whole (DESIGN §9, phases C+D): ONE graph per step —
// the T=2 verify, the recorded pick, the commit, the draft block's fixed
// two rows off the device verdict (the second a padding row after a
// miss), the draft's head on both rows, the recorded draft pick on the
// last accepted row, and the next replay's fed tokens written on the
// device. The host launches, syncs, finishes the bus window, reads two
// pinned verdicts and settles its mirrors. Reference: the eager
// GreedySpeculator (host picks, eager draft) on a SECOND model over the
// same bus, stepped in lockstep — every step's (accepted, next, draft)
// must match, so the in-graph draft with its padding row proposes exactly
// what the eager one-row draft proposes; and the transcript must equal
// the plain sharded session's on every rank.
// ---------------------------------------------------------------------------
DGPP_TEST(glm_tp_full_graph_step_loopback_matches_eager_speculator) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt = make_tokens(9, cfg.vocab_size);
  constexpr int kTokens = 12;
  constexpr int kWorld = 4;
  const int max_tokens = static_cast<int>(prompt.size()) + kTokens + 4;

  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29921);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<int32_t>> rank_seqs(kWorld);
  std::vector<int> rank_steps(kWorld, 0);
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
      uint16_t* scratch = nullptr;
      cudaGraphExec_t graph_exec = nullptr;
      try {
        CollectiveBus& bus = *buses[static_cast<size_t>(r)];
        GlmBusBoundaryReducer reducer(bus, test_wait_timeout_ms());
        GlmDiagnosticModel plain_shard(cfg, dir, max_tokens, 128, &reducer,
                                       r, kWorld, GlmResidency::Streaming,
                                       GlmHeadSharding::VocabSharded);
        GlmDiagnosticModel eager_shard(cfg, dir, max_tokens, 128, &reducer,
                                       r, kWorld, GlmResidency::Streaming,
                                       GlmHeadSharding::VocabSharded, 1,
                                       /*mtp=*/true);
        GlmDiagnosticModel shard(cfg, dir, max_tokens, 128, &reducer, r,
                                 kWorld, GlmResidency::Resident,
                                 GlmHeadSharding::VocabSharded, 1,
                                 /*mtp=*/true);
        shard.set_decode_route_traces(false);
        shard.set_decode_tail_mirrors(false);
        dgpp::GlmDevicePicker picker(bus, r, kWorld, test_wait_timeout_ms());
        dgpp::GlmGraphRecordReducer recorder(bus, shard.stream());
        DGPP_CUDA_OK(cudaMallocManaged(
            &scratch, sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld)));
        arrive_once();
        const auto host_pick = [&](const GlmDiagnosticModel::Outputs& o,
                                   int rows) {
          return dgpp::bus_greedy_pick_rows(bus, r, kWorld,
                                            dgpp::local_row_maxes(o, rows),
                                            scratch, test_wait_timeout_ms());
        };
        // ---- reference 1: the plain sharded session ---------------------
        std::vector<int32_t> plain4;
        {
          GlmDiagnosticModel::Outputs o = plain_shard.session_prefill(prompt);
          int32_t tok = host_pick(o, 1)[0];
          for (int s = 0; s < kTokens; ++s) {
            plain4.push_back(tok);
            o = plain_shard.session_step(tok);
            tok = host_pick(o, 1)[0];
          }
        }
        // ---- reference 2: the eager speculator, stepped in lockstep -----
        dgpp::GreedySpeculator eager(eager_shard, 0,
                                     [&](const std::vector<Candidate>& l) {
                                       return dgpp::bus_greedy_pick_rows(
                                           bus, r, kWorld, l, scratch, test_wait_timeout_ms());
                                     });
        // Both speculators start with the TRUE next token as the first
        // proposal (plain4[1]): the first step accepts both rows, so the
        // in-graph draft runs two REAL rows (no padding) at least once and
        // its two-row bits are compared against the eager two-row draft.
        eager.start(host_pick(eager_shard.session_prefill(prompt), 1)[0],
                    plain4[1]);

        // ---- the graph model: prefill, first pick, first draft (eager) --
        const GlmDiagnosticModel::Outputs pre = shard.session_prefill(prompt);
        int32_t next = host_pick(pre, 1)[0];
        const auto device_inputs = [&](int rows, int slot,
                                       const dgpp::GlmPickVerdict* select) {
          dgpp::GlmDevicePicker::Inputs in;
          in.logits = shard.device_logits();
          in.rows = rows;
          in.vocab_count = shard.lm_vocab_count();
          in.vocab_begin = shard.lm_vocab_begin();
          in.fed = shard.device_tokens();
          in.slot = slot;
          in.row_select = select;
          return in;
        };
        (void)shard.session_draft(0, {next});
        (void)picker.run(shard.stream(), device_inputs(1, 1, nullptr));
        int32_t draft = plain4[1];
        if (next != eager.next() || draft != eager.draft())
          throw std::runtime_error("first next/draft differ from the eager "
                                   "speculator's");

        // ---- capture the whole step -------------------------------------
        shard.session_graph_prepare();
        shard.session_reserve_blocks(0, max_tokens);
        dgpp::GlmBoundaryReducer* eager_reducer = shard.set_boundary(&recorder);
        std::string gerr;
        require(bus.graph_record_begin(&gerr), "graph_record_begin: " + gerr);
        cudaGraph_t graph = nullptr;
        DGPP_CUDA_OK(cudaStreamBeginCapture(shard.stream(),
                                            cudaStreamCaptureModeThreadLocal));
        shard.session_graph_capture_step(0, std::vector<int64_t>{next, draft},
                                         /*device_positions=*/true,
                                         /*device_tokens=*/true);
        picker.record(shard.stream(), device_inputs(2, 0, nullptr));
        shard.session_graph_capture_commit(0, picker.device_verdict(0));
        shard.session_graph_capture_draft(0, picker.device_verdict(0));
        picker.record(shard.stream(),
                      device_inputs(1, 1, picker.device_verdict(0)));
        shard.session_graph_capture_next_tokens(0, picker.device_verdict(1));
        DGPP_CUDA_OK(cudaStreamEndCapture(shard.stream(), &graph));
        require(graph != nullptr, "capture produced no graph");
        require(bus.graph_record_end(&gerr), "graph_record_end: " + gerr);
        shard.set_boundary(eager_reducer);
        dgpp::glm_check_decode_graph(graph, r, "full-step graph");
        DGPP_CUDA_OK(cudaGraphInstantiate(&graph_exec, graph, nullptr,
                                          nullptr, 0));
        cudaGraphDestroy(graph);
        shard.session_graph_seed_tokens(0, {next, draft});

        // ---- the loop: launch, sync, finish, read, settle ----------------
        std::vector<int32_t>& got = rank_seqs[static_cast<size_t>(r)];
        int steps = 0;
        while (static_cast<int>(got.size()) < kTokens) {
          require(bus.graph_replay_arm(&gerr), "graph_replay_arm: " + gerr);
          DGPP_CUDA_OK(cudaGraphLaunch(graph_exec, shard.stream()));
          DGPP_CUDA_OK(cudaStreamSynchronize(shard.stream()));
          require(bus.graph_replay_finish(test_wait_timeout_ms(), &gerr),
                  "graph_replay_finish: " + gerr);
          const dgpp::GlmPickVerdict v0 = picker.verdict(0);
          const dgpp::GlmPickVerdict v1 = picker.verdict(1);
          shard.session_graph_settle(0, v0.accepted);
          // The eager reference takes the same step (its own collectives,
          // between windows) and must agree on everything observable.
          const std::vector<int32_t> committed = eager.step();
          if (static_cast<int>(committed.size()) != v0.accepted)
            throw std::runtime_error(
                "step " + std::to_string(steps) + ": graph accepted " +
                std::to_string(v0.accepted) + " rows, eager " +
                std::to_string(committed.size()));
          if (v0.next != eager.next())
            throw std::runtime_error("step " + std::to_string(steps) +
                                     ": next differs from the eager speculator");
          if (v1.next != eager.draft())
            throw std::runtime_error(
                "step " + std::to_string(steps) + ": in-graph draft " +
                std::to_string(v1.next) + " != eager draft " +
                std::to_string(eager.draft()));
          const int64_t fed[2] = {next, draft};
          for (int row = 0; row < v0.accepted; ++row) {
            if (fed[row] != committed[static_cast<size_t>(row)])
              throw std::runtime_error("committed tokens differ");
            got.push_back(static_cast<int32_t>(fed[row]));
          }
          next = v0.next;
          draft = v1.next;
          ++steps;
        }
        got.resize(kTokens);
        rank_steps[static_cast<size_t>(r)] = steps;
        if (got != plain4)
          throw std::runtime_error(
              "one-graph speculative transcript != the plain sharded "
              "session's on the same bus");
        if (steps >= kTokens)
          throw std::runtime_error("the forced first accept never happened");
        cudaGraphExecDestroy(graph_exec);
        graph_exec = nullptr;
        cudaFree(scratch);
        scratch = nullptr;
      } catch (const std::exception& e) {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
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
  for (int r = 1; r < kWorld; ++r)
    require(rank_seqs[static_cast<size_t>(r)] == rank_seqs[0],
            "rank " + std::to_string(r) + ": transcript differs across ranks");
  DGPP_LOG_INFO("one-graph step w4: {} steps for {} tokens; every step's "
                "(accepted, next, draft) == the eager speculator's, "
                "transcript == plain on every rank",
                rank_steps[0], kTokens);
}

// ---------------------------------------------------------------------------
// M6.6a phase 1 as the service drives it: Scheduler -> graph engine -> the
// whole on-device MTP replay. Unlike the lower-level gate above, this pins the
// public token timing: prefill emits token 0; a replay returns the verify
// winners after it (one on a miss, two on a hit); the scheduler truncates at
// the request cap. It then closes and reuses slot 0 with the SAME graph.
//
// An eager GreedySpeculator on a second MTP model is stepped in lockstep, one
// step per scheduler tick, and after every tick the graph's device token feed
// [next, draft] (written by the replay's last node) must equal the eager
// speculator's (next, draft). The transcript alone cannot see this: a draft
// block left stale by the slot reuse (its cache, its device row counter) only
// lowers acceptance, while the verify rows still decide the plain transcript
// — so this is the check that the reuse reseeds the draft side too.
// ---------------------------------------------------------------------------
DGPP_TEST(glm_tp_serving_graph_adapter_matches_plain_and_reuses_slot) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt = make_tokens(9, cfg.vocab_size);
  constexpr int kTokens = 8;
  constexpr int kReuseTokens = 5;
  constexpr int kWorld = 2;
  const int max_tokens = static_cast<int>(prompt.size()) + kTokens + 2;

  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29922);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<int64_t>> first_seqs(kWorld);
  std::vector<std::vector<int64_t>> reuse_seqs(kWorld);
  std::vector<int> first_steps(kWorld, 0);
  std::vector<int> reuse_steps(kWorld, 0);
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
      uint16_t* scratch = nullptr;
      try {
        CollectiveBus& bus = *buses[static_cast<size_t>(r)];
        GlmBusBoundaryReducer reducer(bus, test_wait_timeout_ms());
        GlmDiagnosticModel plain(cfg, dir, max_tokens, 128, &reducer, r,
                                 kWorld, GlmResidency::Streaming,
                                 GlmHeadSharding::VocabSharded);
        GlmDiagnosticModel eager(cfg, dir, max_tokens, 128, &reducer, r,
                                 kWorld, GlmResidency::Streaming,
                                 GlmHeadSharding::VocabSharded,
                                 /*max_requests=*/1, /*mtp=*/true);
        GlmDiagnosticModel graph(cfg, dir, max_tokens, 128, &reducer, r,
                                 kWorld, GlmResidency::Resident,
                                 GlmHeadSharding::VocabSharded,
                                 /*max_requests=*/1, /*mtp=*/true);
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&scratch),
            sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld),
            cudaHostAllocDefault));
        arrive_once();

        const auto pick_rows = [&](const std::vector<Candidate>& locals) {
          return dgpp::bus_greedy_pick_rows(bus, r, kWorld, locals, scratch,
                                            test_wait_timeout_ms());
        };
        const auto host_pick = [&](const GlmDiagnosticModel::Outputs& out) {
          return pick_rows(dgpp::local_row_maxes(out, 1))[0];
        };
        std::vector<int64_t> want;
        GlmDiagnosticModel::Outputs out = plain.session_prefill(prompt);
        int32_t token = host_pick(out);
        for (int i = 0; i < kTokens; ++i) {
          want.push_back(token);
          if (i + 1 < kTokens) {
            out = plain.session_step(token);
            token = host_pick(out);
          }
        }
        plain.session_close(0);

        {
          dgpp::GlmGraphEngineAdapter engine(
              &graph, &bus, r, kWorld, scratch, cfg.vocab_size);
          dgpp::glm::Scheduler sched(&engine, /*eos_token_ids=*/{});
          // One request through the scheduler, tick by tick. With no EOS
          // set and max_steps > 1 no request retires at admission, so
          // every tick that returns true ran exactly one replay; the eager
          // speculator takes its step after each and the two must agree on
          // what the device will feed next. Returns the replay count.
          const auto run_request = [&](const std::string& id,
                                       int max_steps) {
            dgpp::GreedySpeculator spec(eager, 0, pick_rows);
            spec.start(host_pick(eager.session_prefill(prompt)));
            dgpp::glm::SchedulerRequest req;
            req.id = id;
            req.prompt = prompt;
            req.max_steps = max_steps;
            sched.submit(std::move(req));
            int steps = 0;
            while (sched.tick()) {
              ++steps;
              (void)spec.step();
              int64_t fed[2] = {-1, -1};
              // The model's stream, never the legacy stream: the peer
              // rank's thread may be mid-capture (the note on
              // session_graph_seed_tokens).
              DGPP_CUDA_OK(cudaMemcpyAsync(fed, graph.device_tokens(),
                                           sizeof(fed),
                                           cudaMemcpyDeviceToHost,
                                           graph.stream()));
              DGPP_CUDA_OK(cudaStreamSynchronize(graph.stream()));
              if (fed[0] != spec.next() || fed[1] != spec.draft())
                throw std::runtime_error(
                    "request '" + id + "' replay " + std::to_string(steps) +
                    ": the graph feeds [" + std::to_string(fed[0]) + ", " +
                    std::to_string(fed[1]) +
                    "] but the eager speculator has next " +
                    std::to_string(spec.next()) + ", draft " +
                    std::to_string(spec.draft()));
            }
            eager.session_close(0);
            return steps;
          };

          first_steps[static_cast<size_t>(r)] = run_request("first", kTokens);
          first_seqs[static_cast<size_t>(r)] = sched.results()[0].generated;
          if (first_seqs[static_cast<size_t>(r)] != want)
            throw std::runtime_error(
                "serving graph transcript differs from plain decode");

          // A second admission must use the graph era already recorded: its
          // eager prefill/draft happen between windows, then slot 0 is seeded
          // and replayed without another graph_record_begin.
          reuse_steps[static_cast<size_t>(r)] =
              run_request("reuse", kReuseTokens);
          reuse_seqs[static_cast<size_t>(r)] = sched.results()[1].generated;
          if (reuse_seqs[static_cast<size_t>(r)].size() !=
                  static_cast<size_t>(kReuseTokens) ||
              !std::equal(reuse_seqs[static_cast<size_t>(r)].begin(),
                          reuse_seqs[static_cast<size_t>(r)].end(),
                          want.begin()))
            throw std::runtime_error(
                "reused serving graph transcript differs from plain prefix");
        }
        cudaFreeHost(scratch);
        scratch = nullptr;
      } catch (const std::exception& e) {
        if (scratch != nullptr) cudaFreeHost(scratch);
        errors[static_cast<size_t>(r)] =
            "rank " + std::to_string(r) + ": " + e.what();
        DGPP_LOG_ERROR("serving graph adapter rank {} failed: {}", r,
                       e.what());
        arrive_once();
      }
    });
  }
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r)
    require(errors[static_cast<size_t>(r)].empty(),
            errors[static_cast<size_t>(r)]);
  for (int r = 1; r < kWorld; ++r) {
    require(first_seqs[static_cast<size_t>(r)] == first_seqs[0],
            "serving graph first transcript differs across ranks");
    require(reuse_seqs[static_cast<size_t>(r)] == reuse_seqs[0],
            "serving graph reused transcript differs across ranks");
    require(first_steps[static_cast<size_t>(r)] == first_steps[0] &&
                reuse_steps[static_cast<size_t>(r)] == reuse_steps[0],
            "serving graph replay counts differ across ranks");
  }
  DGPP_LOG_INFO("serving graph adapter w2: {} replays for {} tokens, then {} "
                "replays for {} tokens on the reused slot; every replay's "
                "device token feed == the eager speculator's (next, draft)",
                first_steps[0], kTokens, reuse_steps[0], kReuseTokens);
}

// M6 6b, the device path: the plain graphs with the ON-DEVICE sampling pick
// (scalar variant, then the two-slot batch with a greedy request beside a
// sampled one, then a slot reused) in lockstep with the eager sampling engine
// on the same bus. Every request's transcript must equal the eager engine's
// and agree across ranks; with the candidate width capped at six per rank on
// this 96-token vocabulary, the model-default request must fall back on some
// steps (served between windows through the bulk gather) and a narrow
// top-p request must resolve on the device on others.
DGPP_TEST(glm_tp_serving_graph_sampling_matches_eager_engine) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt = make_tokens(9, cfg.vocab_size);
  constexpr int kWorld = 2;
  constexpr int kSlots = 2;
  // 24 of 48 ids per rank: the model-default nucleus (95%) cannot fit the
  // 48-candidate prefix of this near-flat fixture, a 10% one can.
  constexpr int kCap = 24;
  const int max_tokens = static_cast<int>(prompt.size()) + 12;

  struct Spec {
    const char* id;
    int max_steps;
    dgpp::glm_sample::Params params;
    uint64_t seed;
  };
  const auto model_default = [] {
    dgpp::glm_sample::Params p;
    p.temperature = 1.0f;
    p.top_p = 0.95f;
    p.frequency_penalty = 0.2f;
    p.presence_penalty = 0.1f;
    return p;
  }();
  const auto narrow = [] {
    dgpp::glm_sample::Params p;
    p.temperature = 0.8f;
    p.top_p = 0.1f;
    return p;
  }();
  // Phase 1: one sampled request alone (the scalar variant). Phase 2: a
  // sampled request and a greedy one together (the batch), then the greedy
  // one's slot reused by a narrow-top-p request (device resolutions).
  const std::vector<std::vector<Spec>> phases{
      {{"solo", 8, model_default, 7}},
      {{"a", 8, model_default, 11},
       {"g", 5, dgpp::glm_sample::greedy_params(), 0}},
      {{"n", 6, narrow, 23}},
  };

  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29929);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<std::vector<int64_t>>> graph_seqs(kWorld);
  std::vector<uint64_t> fallbacks(kWorld, 0);
  std::vector<int> sampled_steps(kWorld, 0);
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
      uint16_t* scratch = nullptr;
      uint16_t* prefix_scratch = nullptr;
      uint16_t* gather_scratch = nullptr;
      const auto release = [&] {
        if (scratch) cudaFreeHost(scratch);
        if (prefix_scratch) cudaFreeHost(prefix_scratch);
        if (gather_scratch) cudaFreeHost(gather_scratch);
        scratch = prefix_scratch = gather_scratch = nullptr;
      };
      try {
        CollectiveBus& bus = *buses[static_cast<size_t>(r)];
        GlmBusBoundaryReducer reducer(bus, test_wait_timeout_ms());
        GlmDiagnosticModel eager(cfg, dir, max_tokens, 128, &reducer, r,
                                 kWorld, GlmResidency::Streaming,
                                 GlmHeadSharding::VocabSharded, kSlots);
        GlmDiagnosticModel graph(cfg, dir, max_tokens, 128, &reducer, r,
                                 kWorld, GlmResidency::Resident,
                                 GlmHeadSharding::VocabSharded, kSlots);
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&scratch),
            sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld),
            cudaHostAllocDefault));
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&prefix_scratch),
            sizeof(uint16_t) * dgpp::fabric_sampling_prefix_scratch_elems(kWorld),
            cudaHostAllocDefault));
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&gather_scratch),
            sizeof(uint16_t) * dgpp::sampling_gather_scratch_elems(cfg.vocab_size),
            cudaHostAllocDefault));
        arrive_once();

        dgpp::GenEngineAdapter eager_engine(
            &eager, kSlots,
            dgpp::make_fabric_pick(&bus, r, kWorld, scratch, cfg.vocab_size,
                                   test_wait_timeout_ms()),
            dgpp::make_fabric_sample(&bus, r, kWorld, prefix_scratch,
                                     gather_scratch, cfg.vocab_size,
                                     test_wait_timeout_ms()));
        dgpp::GlmGraphEngineAdapter graph_engine(
            &graph, &bus, r, kWorld, scratch, cfg.vocab_size,
            test_wait_timeout_ms(), /*batch_min_live=*/kSlots, prefix_scratch,
            gather_scratch, kCap);
        if (!graph_engine.supports_sampling() ||
            graph_engine.sampling_candidates() != kCap)
          throw std::runtime_error("graph engine did not arm the device "
                                   "sampler at the capped width");
        dgpp::glm::Scheduler eager_sched(&eager_engine, /*eos=*/{});
        dgpp::glm::Scheduler graph_sched(&graph_engine, /*eos=*/{});

        size_t result_index = 0;
        for (const std::vector<Spec>& phase : phases) {
          for (const Spec& spec : phase) {
            dgpp::glm::SchedulerRequest req;
            req.id = spec.id;
            req.prompt = prompt;
            req.max_steps = spec.max_steps;
            req.sampling = spec.params;
            req.seed = spec.seed;
            eager_sched.submit(req);
            graph_sched.submit(req);
          }
          // Lockstep ticks: the eager engine's collectives, then the graph's
          // (its replay window plus the eager fallback/prefill collectives
          // between windows), in the same order on every rank.
          bool more = true;
          while (more) {
            const bool e = eager_sched.tick();
            const bool g = graph_sched.tick();
            if (e != g)
              throw std::runtime_error("the two schedulers disagree on "
                                       "pending work");
            more = e;
          }
          for (const Spec& spec : phase) {
            const auto& want = eager_sched.results()[result_index].generated;
            const auto& got = graph_sched.results()[result_index].generated;
            ++result_index;
            if (want.size() != static_cast<size_t>(spec.max_steps))
              throw std::runtime_error(std::string("eager transcript length "
                                                   "for ") + spec.id);
            if (got != want)
              throw std::runtime_error(
                  std::string("graph sampling transcript for '") + spec.id +
                  "' differs from the eager sampling engine's");
            graph_seqs[static_cast<size_t>(r)].push_back(got);
            if (spec.params.temperature > 0.0f)
              sampled_steps[static_cast<size_t>(r)] += spec.max_steps;
          }
        }
        fallbacks[static_cast<size_t>(r)] = graph_engine.fallbacks();
        release();
      } catch (const std::exception& e) {
        release();
        errors[static_cast<size_t>(r)] =
            "rank " + std::to_string(r) + ": " + e.what();
        DGPP_LOG_ERROR("graph sampling rank {} failed: {}", r, e.what());
        arrive_once();
      }
    });
  }
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r)
    require(errors[static_cast<size_t>(r)].empty(), errors[static_cast<size_t>(r)]);
  for (int r = 1; r < kWorld; ++r) {
    require(graph_seqs[static_cast<size_t>(r)] == graph_seqs[0],
            "graph sampling transcripts differ across ranks");
    require(fallbacks[static_cast<size_t>(r)] == fallbacks[0],
            "fallback counts differ across ranks");
  }
  // The prefill picks are host decisions; the device decided every other
  // sampled step, some inside the prefix and some through the fallback.
  const int device_steps = sampled_steps[0] - 3;  // three sampled prefills
  require(fallbacks[0] > 0, "the capped width must force some fallbacks");
  require(static_cast<int>(fallbacks[0]) < device_steps,
          "some sampled steps must resolve on the device");
  DGPP_LOG_INFO("graph sampling w2: {} sampled device steps, {} served by the "
                "fallback; transcripts == the eager sampling engine's on "
                "every rank",
                device_steps, fallbacks[0]);
}

// M6 6b, the device path under MTP: the one-graph T=2 step with the
// on-device speculative verdict (the accept test, the residual, row 1's
// sample) in lockstep with the eager SampledSpeculator on a second model
// over the same bus — one request alone (the scalar variant), then two
// together (the batch). The graph engine's transcript must equal the
// speculator's committed stream, and after every replay the graph's
// [next, draft] feed must equal the speculator's (next, draft) — which is
// where a wrong draft rollback would show. The candidate cap forces
// fallbacks of both rows, served by rolling the in-graph draft back and
// re-drafting eagerly.
DGPP_TEST(glm_tp_serving_mtp_graph_sampling_matches_eager_speculator) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt = make_tokens(9, cfg.vocab_size);
  constexpr int kWorld = 2;
  constexpr int kSlots = 2;
  constexpr int kCap = 24;
  const int max_tokens = static_cast<int>(prompt.size()) + 16;
  dgpp::glm_sample::Params params;
  params.temperature = 1.0f;
  params.top_p = 0.95f;
  params.frequency_penalty = 0.2f;
  params.presence_penalty = 0.1f;
  struct Spec {
    const char* id;
    int max_steps;
    uint64_t seed;
  };
  const std::vector<std::vector<Spec>> phases{{{"solo", 7, 7}},
                                              {{"a", 7, 11}, {"b", 6, 13}}};

  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29932);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<std::vector<int64_t>>> graph_seqs(kWorld);
  std::vector<uint64_t> fallbacks(kWorld, 0);
  std::vector<int> replays(kWorld, 0);
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
      uint16_t* scratch = nullptr;
      uint16_t* prefix_scratch = nullptr;
      uint16_t* gather_scratch = nullptr;
      const auto release = [&] {
        if (scratch) cudaFreeHost(scratch);
        if (prefix_scratch) cudaFreeHost(prefix_scratch);
        if (gather_scratch) cudaFreeHost(gather_scratch);
        scratch = prefix_scratch = gather_scratch = nullptr;
      };
      try {
        CollectiveBus& bus = *buses[static_cast<size_t>(r)];
        GlmBusBoundaryReducer reducer(bus, test_wait_timeout_ms());
        // Two slots with the draft layer's cache: the pool the batched MTP
        // gate uses.
        GlmDiagnosticModel eager(cfg, dir, max_tokens, 256, &reducer, r,
                                 kWorld, GlmResidency::Streaming,
                                 GlmHeadSharding::VocabSharded, kSlots,
                                 /*mtp=*/true);
        GlmDiagnosticModel graph(cfg, dir, max_tokens, 256, &reducer, r,
                                 kWorld, GlmResidency::Resident,
                                 GlmHeadSharding::VocabSharded, kSlots,
                                 /*mtp=*/true);
        DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                                   sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld),
                                   cudaHostAllocDefault));
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&prefix_scratch),
            sizeof(uint16_t) * dgpp::fabric_sampling_prefix_scratch_elems(kWorld),
            cudaHostAllocDefault));
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&gather_scratch),
            sizeof(uint16_t) * dgpp::sampling_gather_scratch_elems(cfg.vocab_size),
            cudaHostAllocDefault));
        arrive_once();

        const auto pick_rows = [&](const std::vector<Candidate>& locals) {
          return dgpp::bus_greedy_pick_rows(bus, r, kWorld, locals, scratch,
                                            test_wait_timeout_ms());
        };
        const dgpp::GenEngineAdapter::Sample row1 = dgpp::make_fabric_sample(
            &bus, r, kWorld, prefix_scratch, gather_scratch, cfg.vocab_size,
            test_wait_timeout_ms());
        const dgpp::SampledSpeculator::Row0 row0 = dgpp::make_fabric_spec_row0(
            &bus, r, kWorld, prefix_scratch, gather_scratch, cfg.vocab_size,
            test_wait_timeout_ms());
        dgpp::GlmGraphEngineAdapter engine(
            &graph, &bus, r, kWorld, scratch, cfg.vocab_size,
            test_wait_timeout_ms(), /*batch_min_live=*/kSlots, prefix_scratch,
            gather_scratch, kCap);
        if (!engine.supports_sampling() || engine.sampling_candidates() != kCap)
          throw std::runtime_error("the MTP graph engine did not arm the "
                                   "device sampler at the capped width");
        dgpp::glm::Scheduler sched(&engine, /*eos=*/{});

        size_t result_index = 0;
        for (const std::vector<Spec>& phase : phases) {
          const size_t n = phase.size();
          std::vector<std::unique_ptr<dgpp::SampledSpeculator>> specs;
          std::vector<std::vector<int32_t>> streams(n);  // committed so far
          // The speculators prefill BEFORE the scheduler admits (the same
          // collectives the graph engine's prefill issues at admission ride
          // the bus in the same order on every rank either way).
          for (size_t i = 0; i < n; ++i) {
            dgpp::glm_sample::Rng rng{phase[i].seed, 0};
            const GlmDiagnosticModel::Outputs pre =
                eager.session_prefill(static_cast<int>(i), prompt);
            const std::vector<int32_t> prompt_context(prompt.begin(), prompt.end());
            const int32_t first = row1(pre, params, rng, prompt_context).token;
            specs.push_back(std::make_unique<dgpp::SampledSpeculator>(
                eager, static_cast<int>(i), pick_rows, row0, row1, params, rng,
                prompt));
            specs.back()->start(first);
          }
          for (size_t i = 0; i < n; ++i) {
            dgpp::glm::SchedulerRequest req;
            req.id = phase[i].id;
            req.prompt = prompt;
            req.max_steps = phase[i].max_steps;
            req.sampling = params;
            req.seed = phase[i].seed;
            sched.submit(std::move(req));
          }
          // Lockstep: one scheduler tick (at most one admission, one replay
          // for every live slot), then one eager step per request that was
          // live in that replay, then the transcript and feed checks.
          std::vector<bool> live(n, false);
          std::vector<bool> finished(n, false);
          while (sched.tick()) {
            ++replays[static_cast<size_t>(r)];
            for (size_t i = 0; i < n; ++i)
              if (!live[i]) {
                live[i] = true;  // the tick's one admission, in order
                break;
              }
            for (size_t i = 0; i < n; ++i) {
              if (!live[i] || finished[i]) continue;
              const std::vector<int32_t> committed = specs[i]->step();
              streams[i].insert(streams[i].end(), committed.begin(),
                                committed.end());
              const dgpp::glm::Scheduler::Result* res = sched.find(phase[i].id);
              if (res == nullptr) throw std::runtime_error("result lookup");
              // The graph's tokens so far vs the speculator's committed
              // stream plus its pending next, over the same length.
              std::vector<int64_t> want(streams[i].begin(), streams[i].end());
              want.push_back(specs[i]->next());
              if (res->generated.size() > want.size())
                throw std::runtime_error("the graph engine ran ahead of the "
                                         "speculator");
              if (!std::equal(res->generated.begin(), res->generated.end(),
                              want.begin()))
                throw std::runtime_error(
                    std::string("request '") + phase[i].id +
                    "' graph transcript differs from the eager sampled "
                    "speculator's after replay " +
                    std::to_string(replays[static_cast<size_t>(r)]));
              if (res->status != dgpp::glm::Scheduler::Result::Status::kActive) {
                finished[i] = true;
                continue;
              }
              int64_t fed[2] = {-1, -1};
              DGPP_CUDA_OK(cudaMemcpyAsync(fed, graph.device_tokens() + 2 * res->slot,
                                           sizeof(fed), cudaMemcpyDeviceToHost,
                                           graph.stream()));
              DGPP_CUDA_OK(cudaStreamSynchronize(graph.stream()));
              // The scalar variant stages its feed at the next replay from
              // the adapter's pending/draft; the batch keeps it on the
              // device. Both must equal the speculator's (next, draft).
              const bool batch_live =
                  static_cast<int>(std::count(live.begin(), live.end(), true)) -
                      static_cast<int>(std::count(finished.begin(), finished.end(), true)) >= kSlots;
              if (batch_live && (fed[0] != specs[i]->next() || fed[1] != specs[i]->draft()))
                throw std::runtime_error(
                    std::string("request '") + phase[i].id + "': the graph feeds [" +
                    std::to_string(fed[0]) + ", " + std::to_string(fed[1]) +
                    "] but the speculator has next " +
                    std::to_string(specs[i]->next()) + ", draft " +
                    std::to_string(specs[i]->draft()));
            }
          }
          for (size_t i = 0; i < n; ++i) {
            const auto& got = sched.results()[result_index++].generated;
            if (got.size() != static_cast<size_t>(phase[i].max_steps))
              throw std::runtime_error(std::string("transcript length for ") +
                                       phase[i].id);
            graph_seqs[static_cast<size_t>(r)].push_back(got);
            eager.session_close(static_cast<int>(i));
          }
        }
        fallbacks[static_cast<size_t>(r)] = engine.fallbacks();
        release();
      } catch (const std::exception& e) {
        release();
        errors[static_cast<size_t>(r)] =
            "rank " + std::to_string(r) + ": " + e.what();
        DGPP_LOG_ERROR("MTP graph sampling rank {} failed: {}", r, e.what());
        arrive_once();
      }
    });
  }
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r)
    require(errors[static_cast<size_t>(r)].empty(), errors[static_cast<size_t>(r)]);
  for (int r = 1; r < kWorld; ++r) {
    require(graph_seqs[static_cast<size_t>(r)] == graph_seqs[0],
            "MTP graph sampling transcripts differ across ranks");
    require(fallbacks[static_cast<size_t>(r)] == fallbacks[0],
            "fallback counts differ across ranks");
  }
  require(fallbacks[0] > 0, "the capped width must force some fallbacks");
  DGPP_LOG_INFO("MTP graph sampling w2: {} replays, {} fallbacks; transcripts "
                "== the eager sampled speculator's on every rank",
                replays[0], fallbacks[0]);
}

// The full Phase-2 MTP shape: four slots x two speculative rows make one
// eight-row replay. Only noncontiguous slots 0 and 3 are live; the middle
// groups are padding, then slot 3 closes and is reused without recording
// another era. An eager speculator per live slot is the
// oracle for both the newly decided tokens and the graph's persistent
// [next,draft] feed; this catches cross-slot rollback, hidden-cache, and draft
// row-selection aliases even when the public transcript happens to survive.
DGPP_TEST(glm_tp_serving_mtp_batched_graph_matches_independent_speculators) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt_a = make_tokens(9, cfg.vocab_size);
  const std::vector<int64_t> prompt_b = make_tokens(7, cfg.vocab_size);
  const std::vector<int64_t> prompt_c = make_tokens(8, cfg.vocab_size);
  constexpr int kWorld = 2;
  constexpr int kRounds = 3;
  const int max_tokens = 24;

  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29924);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<int64_t>> rank_evidence(kWorld);
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
      uint16_t* scratch = nullptr;
      try {
        CollectiveBus& bus = *buses[static_cast<size_t>(r)];
        GlmBusBoundaryReducer reducer(bus, test_wait_timeout_ms());
        GlmDiagnosticModel eager(cfg, dir, max_tokens, 256, &reducer, r,
                                 kWorld, GlmResidency::Streaming,
                                 GlmHeadSharding::VocabSharded,
                                 /*max_requests=*/4, /*mtp=*/true);
        GlmDiagnosticModel graph(cfg, dir, max_tokens, 256, &reducer, r,
                                 kWorld, GlmResidency::Resident,
                                 GlmHeadSharding::VocabSharded,
                                 /*max_requests=*/4, /*mtp=*/true);
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&scratch),
            sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld),
            cudaHostAllocDefault));
        arrive_once();

        const auto pick_rows = [&](const std::vector<Candidate>& locals) {
          return dgpp::bus_greedy_pick_rows(bus, r, kWorld, locals, scratch,
                                            test_wait_timeout_ms());
        };
        const auto first_pick = [&](GlmDiagnosticModel& model, int req,
                                    const std::vector<int64_t>& prompt) {
          return pick_rows(dgpp::local_row_maxes(
              model.session_prefill(req, prompt), 1))[0];
        };
        const auto newly_decided = [](dgpp::GreedySpeculator& spec) {
          const int32_t old_draft = spec.draft();
          const std::vector<int32_t> consumed = spec.step();
          std::vector<int32_t> out;
          if (consumed.size() == 2) out.push_back(old_draft);
          out.push_back(spec.next());
          return out;
        };

        dgpp::GlmGraphEngineAdapter engine(
            &graph, &bus, r, kWorld, scratch, cfg.vocab_size,
            /*pick_timeout_ms=*/test_wait_timeout_ms(), /*batch_min_live=*/2);
        require(engine.decode_batch_capacity() == 4,
                "MTP graph did not advertise its four fixed slots");
        // The service records every variant at startup; the warm sessions
        // must leave no residue in any slot's state or token feed.
        engine.warm_captures(make_tokens(4, cfg.vocab_size));

        const int32_t eager_first_a = first_pick(eager, 0, prompt_a);
        dgpp::GreedySpeculator spec_a(eager, 0, pick_rows);
        spec_a.start(eager_first_a);
        const int32_t graph_first_a = engine.prefill(0, prompt_a);
        require(graph_first_a == eager_first_a, "slot 0 prefill pick differs");
        engine.reserve(0, static_cast<int64_t>(prompt_a.size()) + 10);

        // Deliberately occupy slot 3 so slots 1 and 2 are all-padding spans.
        const int32_t eager_first_b = first_pick(eager, 3, prompt_b);
        dgpp::GreedySpeculator spec_b(eager, 3, pick_rows);
        spec_b.start(eager_first_b);
        const int32_t graph_first_b = engine.prefill(3, prompt_b);
        require(graph_first_b == eager_first_b, "slot 3 prefill pick differs");
        engine.reserve(3, static_cast<int64_t>(prompt_b.size()) + 10);

        for (int round = 0; round < kRounds; ++round) {
          const std::vector<int32_t> want_a = newly_decided(spec_a);
          const std::vector<int32_t> want_b = newly_decided(spec_b);
          const std::vector<int> order = round & 1 ? std::vector<int>{3, 0}
                                                   : std::vector<int>{0, 3};
          const auto got = engine.step_batch(order);
          for (size_t i = 0; i < order.size(); ++i) {
            const std::vector<int32_t>& want = order[i] == 0 ? want_a : want_b;
            require(got[i] == want,
                    "batched MTP newly-decided tokens differ at round " +
                        std::to_string(round) + " slot " +
                        std::to_string(order[i]));
          }
          int64_t feed[8] = {};
          DGPP_CUDA_OK(cudaMemcpyAsync(feed, graph.device_tokens(), sizeof(feed),
                                       cudaMemcpyDeviceToHost, graph.stream()));
          DGPP_CUDA_OK(cudaStreamSynchronize(graph.stream()));
          require(feed[0] == spec_a.next() && feed[1] == spec_a.draft() &&
                      feed[2] == 0 && feed[3] == 0 && feed[4] == 0 &&
                      feed[5] == 0 && feed[6] == spec_b.next() &&
                      feed[7] == spec_b.draft(),
                  "batched MTP next/draft feed is not slot-local");
          rank_evidence[static_cast<size_t>(r)].insert(
              rank_evidence[static_cast<size_t>(r)].end(), std::begin(feed),
              std::end(feed));
        }

        // Retire slot 3: adaptive execution drops to slot 0's compact scalar
        // variant. The live slot continues exactly; fixed-batch token rows
        // are allowed to stay stale until the next transition seeds them.
        engine.close(3);
        eager.session_close(3);
        const std::vector<int32_t> want_a = newly_decided(spec_a);
        const auto one = engine.step_batch({0});
        require(one.size() == 1 && one[0] == want_a,
                "live slot changed while its peer was padding");
        int64_t padded_feed[8] = {};
        DGPP_CUDA_OK(cudaMemcpyAsync(padded_feed, graph.device_tokens(),
                                     sizeof(padded_feed), cudaMemcpyDeviceToHost,
                                     graph.stream()));
        DGPP_CUDA_OK(cudaStreamSynchronize(graph.stream()));
        require(padded_feed[0] == spec_a.next() &&
                    padded_feed[1] == spec_a.draft(),
                "slot 0 scalar graph did not leave its compact feed");

        const int32_t eager_first_c = first_pick(eager, 3, prompt_c);
        dgpp::GreedySpeculator spec_c(eager, 3, pick_rows);
        spec_c.start(eager_first_c);
        const int32_t graph_first_c = engine.prefill(3, prompt_c);
        require(graph_first_c == eager_first_c, "reused slot prefill differs");
        engine.reserve(3, static_cast<int64_t>(prompt_c.size()) + 10);
        const std::vector<int32_t> want_a2 = newly_decided(spec_a);
        const std::vector<int32_t> want_c = newly_decided(spec_c);
        const auto reused = engine.step_batch({3, 0});
        require(reused.size() == 2 && reused[0] == want_c &&
                    reused[1] == want_a2,
                "reused MTP slot differs from independent eager state");

        // Drop back below the crossover with only nonzero slot 3 alive. This
        // forces its slot-specific scalar graph to capture and proves a batch
        // -> scalar transition does not accidentally bind slot 0's state.
        engine.close(0);
        eager.session_close(0);
        const std::vector<int32_t> want_c2 = newly_decided(spec_c);
        const auto scalar_three = engine.step_batch({3});
        require(scalar_three.size() == 1 && scalar_three[0] == want_c2,
                "slot 3 scalar graph differs after leaving batch mode");
        engine.close(3);
        eager.session_close(3);
        cudaFreeHost(scratch);
        scratch = nullptr;
      } catch (const std::exception& e) {
        if (scratch != nullptr) cudaFreeHost(scratch);
        errors[static_cast<size_t>(r)] =
            "rank " + std::to_string(r) + ": " + e.what();
        arrive_once();
      }
    });
  }
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r)
    require(errors[static_cast<size_t>(r)].empty(),
            errors[static_cast<size_t>(r)]);
  require(rank_evidence[0] == rank_evidence[1],
          "batched MTP token feeds differ across ranks");
}

// Phase 2's T=1 form: two request slots occupy two fixed graph rows. Strict
// admission brings the second request in one tick after the first, then every
// tick advances both with one replay. Different prompt lengths make a state
// alias visible; each transcript must equal an independently decoded scalar
// session, and both ranks must publish the same pair.
DGPP_TEST(glm_tp_serving_plain_batched_graph_matches_independent_sessions) {
  const GlmTextConfig cfg = glm_tp_test_config();
  const std::string dir = "glm_tp_fixture";
  glm_tp_write_fixture(dir);
  const std::vector<int64_t> prompt_a = make_tokens(9, cfg.vocab_size);
  const std::vector<int64_t> prompt_b = make_tokens(7, cfg.vocab_size);
  constexpr int kTokensA = 6;
  constexpr int kTokensB = 5;
  constexpr int kWorld = 2;
  const int max_tokens = static_cast<int>(prompt_a.size()) + kTokensA + 1;

  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29923);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<std::string> errors(kWorld);
  std::vector<std::vector<int64_t>> rank_seqs(kWorld);
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
      uint16_t* scratch = nullptr;
      try {
        CollectiveBus& bus = *buses[static_cast<size_t>(r)];
        GlmBusBoundaryReducer reducer(bus, test_wait_timeout_ms());
        GlmDiagnosticModel plain(cfg, dir, max_tokens, 256, &reducer, r,
                                 kWorld, GlmResidency::Streaming,
                                 GlmHeadSharding::VocabSharded);
        GlmDiagnosticModel graph(cfg, dir, max_tokens, 256, &reducer, r,
                                 kWorld, GlmResidency::Resident,
                                 GlmHeadSharding::VocabSharded,
                                 /*max_requests=*/2, /*mtp=*/false);
        DGPP_CUDA_OK(cudaHostAlloc(
            reinterpret_cast<void**>(&scratch),
            sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld),
            cudaHostAllocDefault));
        arrive_once();
        const auto pick = [&](const GlmDiagnosticModel::Outputs& out) {
          return dgpp::bus_greedy_pick_rows(
              bus, r, kWorld, dgpp::local_row_maxes(out, 1), scratch,
              test_wait_timeout_ms())[0];
        };

        const auto scalar_generate = [&](const std::vector<int64_t>& prompt,
                                         int count) {
          std::vector<int64_t> got;
          GlmDiagnosticModel::Outputs out = plain.session_prefill(prompt);
          int32_t token = pick(out);
          for (int i = 0; i < count; ++i) {
            got.push_back(token);
            if (i + 1 < count) {
              out = plain.session_step(token);
              token = pick(out);
            }
          }
          plain.session_close(0);
          return got;
        };
        const std::vector<int64_t> want_a =
            scalar_generate(prompt_a, kTokensA);
        const std::vector<int64_t> want_b =
            scalar_generate(prompt_b, kTokensB);

        {
          dgpp::GlmGraphEngineAdapter engine(
              &graph, &bus, r, kWorld, scratch, cfg.vocab_size);
          require(engine.decode_batch_capacity() == 2,
                  "Phase-2 graph did not advertise both fixed slots");
          engine.warm_captures(make_tokens(4, cfg.vocab_size));
          dgpp::glm::Scheduler sched(&engine, /*eos_token_ids=*/{});
          dgpp::glm::SchedulerRequest a;
          a.id = "batch-a";
          a.prompt = prompt_a;
          a.max_steps = kTokensA;
          sched.submit(std::move(a));
          dgpp::glm::SchedulerRequest b;
          b.id = "batch-b";
          b.prompt = prompt_b;
          b.max_steps = kTokensB;
          sched.submit(std::move(b));
          sched.run_to_completion();
          if (sched.results()[0].generated != want_a ||
              sched.results()[1].generated != want_b)
            throw std::runtime_error(
                "T=1 row-batched graph transcript differs from independent "
                "scalar decode");
          rank_seqs[static_cast<size_t>(r)] = sched.results()[0].generated;
          rank_seqs[static_cast<size_t>(r)].push_back(-1);
          rank_seqs[static_cast<size_t>(r)].insert(
              rank_seqs[static_cast<size_t>(r)].end(),
              sched.results()[1].generated.begin(),
              sched.results()[1].generated.end());
        }
        cudaFreeHost(scratch);
        scratch = nullptr;
      } catch (const std::exception& e) {
        if (scratch != nullptr) cudaFreeHost(scratch);
        errors[static_cast<size_t>(r)] =
            "rank " + std::to_string(r) + ": " + e.what();
        arrive_once();
      }
    });
  }
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r)
    require(errors[static_cast<size_t>(r)].empty(),
            errors[static_cast<size_t>(r)]);
  require(rank_seqs[0] == rank_seqs[1],
          "T=1 row-batched graph transcripts differ across ranks");
}

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
