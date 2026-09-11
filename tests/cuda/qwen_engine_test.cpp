// The Qwen decode engines over the engine core (Q6 stage B, 2026-09-09):
// EagerEngineAdapter<QwenModel> and GraphEngineAdapter<QwenModel> on the
// fixture, a loopback world of 2 (real verbs QPs over 127.0.0.1) against
// the world-1 eager engine.
//
// Gates: the graph engine's SCALAR replays (the device-driven T=1 graph:
// device positions, the recorded commit, the pinned token upload) and its
// ROW-BATCHED replays (the fixed slot-major batch off the persistent
// feeds, closed slots padding at -1, the 2-slot and full families) produce
// the eager engine's transcripts at the same world exactly (the recorded
// kernels are the eager kernels); the world-2 eager transcripts follow the
// world-1 ones (reported; the folds reassociate, so a near tie may flip a
// late token — the first tokens must agree).
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "common/test.hpp"
#include "engine/tp_bus.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/forward.hpp"
#include "models/qwen/loader.hpp"
#include "net/collective_bus.hpp"
#include "qwen_fixture.hpp"
#include "engine/eager_engine.hpp"
#include "engine/graph_engine.hpp"
#include "engine/tp_bus.hpp"

namespace fs = std::filesystem;
using dgpp::BusBoundaryReducer;
using dgpp::EagerEngineAdapter;
using dgpp::GraphEngineAdapter;
using dgpp::QwenModel;
using dgpp::QwenResidency;
using dgpp::QwenTextConfig;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

int wait_timeout_ms() {
  const char* v = std::getenv("DGPP_TEST_WAIT_TIMEOUT_MS");
  return v ? std::atoi(v) : 60000;
}

BusOptions loop_options(int rank, int world, uint16_t port) {
  BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : "127.0.0.1";
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = 8;
  o.lat_slot_bytes = 262144;  // every fold on the latency path (qwen_tp_test's note)
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = [] {
    const char* ms = std::getenv("DGPP_TEST_BUS_TIMEOUT_MS");
    return ms ? std::atoi(ms) : 5000;
  }();
  o.consumer_deadline_s = [] {
    const char* s = std::getenv("DGPP_TEST_CONSUMER_DEADLINE_S");
    return s ? std::atof(s) : 20.0;
  }();
  o.launch_consumers = false;
  return o;
}

std::vector<std::unique_ptr<CollectiveBus>> start_world(int world, uint16_t port) {
  std::vector<std::unique_ptr<CollectiveBus>> out;
  for (int r = 0; r < world; ++r) out.push_back(std::make_unique<CollectiveBus>(loop_options(r, world, port)));
  std::vector<std::string> errors(static_cast<size_t>(world));
  std::thread listener([&] {
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("qwen engine world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("qwen engine world rank {}: {}", r, errors[static_cast<size_t>(r)]);
    });
  listener.join();
  for (auto& t : connectors) t.join();
  for (int r = 0; r < world; ++r)
    if (!errors[static_cast<size_t>(r)].empty()) return {};
  return out;
}

struct ConstructBarrier {
  std::mutex mu;
  std::condition_variable cv;
  int left;
  explicit ConstructBarrier(int world) : left(world) {}
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mu);
    if (--left == 0) cv.notify_all();
    else cv.wait(lock, [this] { return left == 0; });
  }
};

std::vector<int64_t> smoke_tokens(const QwenTextConfig& cfg, int n, uint64_t seed) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = seed;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  return t;
}

std::string ids_text(const std::vector<int32_t>& ids) {
  std::string s;
  for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
  return s;
}

// The engine's greedy transcript for one prompt in slot `req`: the
// prefill's pick then `steps` scalar steps.
template <class Engine>
std::vector<int32_t> solo(Engine& eng, int req, const std::vector<int64_t>& prompt, int steps) {
  std::vector<int32_t> out;
  out.push_back(eng.prefill(req, prompt));
  eng.reserve(req, static_cast<int64_t>(prompt.size()) + steps + 1);
  for (int s = 0; s < steps; ++s) {
    const std::vector<int32_t> t = eng.step(req);
    require(t.size() == 1, "a scalar step decides one token");
    out.push_back(t[0]);
  }
  eng.close(req);
  return out;
}

size_t agreeing_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  size_t n = 0;
  while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
  return n;
}

constexpr int kSlots = 3;
constexpr int kSteps = 12;
constexpr int kMaxTokens = 64;
constexpr int64_t kCache = 512;
constexpr int kWorld = 2;
constexpr uint16_t kPort = 29944;

struct Ref {
  std::vector<int32_t> a, b, c;
};

Ref world1_reference(const QwenTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
                     const std::vector<int64_t>& B, const std::vector<int64_t>& C) {
  QwenModel m(cfg, dir, kMaxTokens, kCache, QwenResidency::Resident, nullptr, 0, 1, kSlots);
  EagerEngineAdapter<QwenModel> eng(&m, kSlots, dgpp::make_w1_pick(cfg.vocab_size));
  Ref r;
  r.a = solo(eng, 0, A, kSteps);
  r.b = solo(eng, 1, B, kSteps);
  r.c = solo(eng, 2, C, kSteps);
  return r;
}

struct RankOutcome {
  std::string error;
  std::vector<int32_t> ea, eb, ec;   // the eager engine, world 2
  std::vector<int32_t> ga;           // the graph engine, scalar
  std::vector<int32_t> ba, bb, bc;   // the graph engine, batched
  std::vector<int32_t> ma, mb, mc;   // the MTP graph engine: scalar A, batched B and C
  int mtp_steps_a = 0;               // scalar MTP steps A took (< kSteps: drafts stood)
};

void rank_work(int r, const QwenTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
               const std::vector<int64_t>& B, const std::vector<int64_t>& C, CollectiveBus* bus,
               ConstructBarrier* barrier, RankOutcome* out) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    QwenModel eager(cfg, dir, kMaxTokens, kCache, QwenResidency::Resident, &reducer, r, kWorld, kSlots);
    QwenModel graph(cfg, dir, kMaxTokens, kCache, QwenResidency::Resident, &reducer, r, kWorld, kSlots);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<QwenModel> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    GraphEngineAdapter<QwenModel> graph_engine(&graph, bus, r, kWorld, scratch, cfg.vocab_size,
                                               wait_timeout_ms(), /*batch_min_live=*/2);

    // 1. The eager engine at world 2.
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);

    // 2. The graph engine, scalar replays (one live slot).
    out->ga = solo(graph_engine, 1, A, kSteps);
    graph_engine.drain();

    // 3. The row-batched replays: A and B live (the 2-slot family), then
    //    C joins (the full family), then B closes (slot 1 pads).
    out->ba.push_back(graph_engine.prefill(0, A));
    graph_engine.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 1);
    out->bb.push_back(graph_engine.prefill(1, B));
    graph_engine.reserve(1, static_cast<int64_t>(B.size()) + kSteps + 1);
    for (int s = 0; s < 5; ++s) {
      const auto t = graph_engine.step_batch({0, 1});
      require(t.size() == 2 && t[0].size() == 1 && t[1].size() == 1, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bb.push_back(t[1][0]);
    }
    graph_engine.drain();
    out->bc.push_back(graph_engine.prefill(2, C));
    graph_engine.reserve(2, static_cast<int64_t>(C.size()) + kSteps + 1);
    for (int s = 0; s < 4; ++s) {
      const auto t = graph_engine.step_batch({0, 1, 2});
      require(t.size() == 3, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bb.push_back(t[1][0]);
      out->bc.push_back(t[2][0]);
    }
    graph_engine.close(1);
    for (int s = 0; s < 3; ++s) {
      const auto t = graph_engine.step_batch({0, 2});
      require(t.size() == 2, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bc.push_back(t[1][0]);
    }
    graph_engine.close(0);
    graph_engine.close(2);
    graph_engine.drain();
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

// The MTP graph engine (the one-graph draft) in its own bus world (one
// graph era per bus): the T=2 verify with the recorded commit, the in-graph
// draft and its pick, the [next, draft] feed — scalar for A, then the
// batched families for B and C — against the plain eager engine.
void rank_work_mtp(int r, const QwenTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
                   const std::vector<int64_t>& B, const std::vector<int64_t>& C, CollectiveBus* bus,
                   ConstructBarrier* barrier, RankOutcome* out, int depth) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    QwenModel eager(cfg, dir, kMaxTokens, kCache, QwenResidency::Resident, &reducer, r, kWorld, kSlots);
    QwenModel mtp(cfg, dir, kMaxTokens, kCache, QwenResidency::Resident, &reducer, r, kWorld, kSlots, /*mtp=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<QwenModel> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);
    {
      GraphEngineAdapter<QwenModel> mtp_engine(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                               /*batch_min_live=*/2, /*prefix_scratch=*/nullptr,
                                               /*gather_scratch=*/nullptr, /*candidates=*/0, /*grammar=*/nullptr,
                                               /*prefix_slots=*/0, /*mtp_depth=*/depth);
      out->ma.push_back(mtp_engine.prefill(0, A));
      mtp_engine.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 2 + depth);
      while (out->ma.size() < static_cast<size_t>(kSteps) + 1) {
        const std::vector<int32_t> t = mtp_engine.step(0);
        require(!t.empty() && t.size() <= static_cast<size_t>(1 + depth), "mtp step shape");
        out->ma.insert(out->ma.end(), t.begin(), t.end());
        ++out->mtp_steps_a;
      }
      out->ma.resize(static_cast<size_t>(kSteps) + 1);
      mtp_engine.close(0);
      out->mb.push_back(mtp_engine.prefill(1, B));
      mtp_engine.reserve(1, static_cast<int64_t>(B.size()) + kSteps + 2 + depth);
      out->mc.push_back(mtp_engine.prefill(2, C));
      mtp_engine.reserve(2, static_cast<int64_t>(C.size()) + kSteps + 2 + depth);
      while (out->mb.size() < static_cast<size_t>(kSteps) + 1 || out->mc.size() < static_cast<size_t>(kSteps) + 1) {
        const auto t = mtp_engine.step_batch({1, 2});
        require(t.size() == 2, "mtp batch step shape");
        out->mb.insert(out->mb.end(), t[0].begin(), t[0].end());
        out->mc.insert(out->mc.end(), t[1].begin(), t[1].end());
      }
      out->mb.resize(static_cast<size_t>(kSteps) + 1);
      out->mc.resize(static_cast<size_t>(kSteps) + 1);
      mtp_engine.close(1);
      mtp_engine.close(2);
      mtp_engine.drain();
    }
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

}  // namespace

DGPP_TEST(qwen_engines_loopback_world_2_mtp_graph_matches_plain_decode) {
  const QwenTextConfig cfg = qwenfx::tiny_config();
  const std::string dir = "qwen_engine_fixture";
  qwenfx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort + 1);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<RankOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_mtp, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B),
                         std::cref(C), buses[static_cast<size_t>(r)].get(), &barrier,
                         &outs[static_cast<size_t>(r)], /*depth=*/1);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].ma == outs[0].ma && outs[static_cast<size_t>(r)].mb == outs[0].mb,
            "the ranks' MTP transcripts differ");
  const RankOutcome& o = outs[0];
  DGPP_LOG_INFO("world 2 MTP graph: A {} ({} steps) | B {} | C {}", ids_text(o.ma), o.mtp_steps_a, ids_text(o.mb),
                ids_text(o.mc));
  require(o.ma == o.ea, "the MTP scalar transcript differs from the plain eager engine's");
  require(o.mb == o.eb, "the MTP batched transcript of B differs from the plain eager engine's");
  require(o.mc == o.ec, "the MTP batched transcript of C differs from the plain eager engine's");
  // A random-weight fixture drafts by chance only (the acceptance rate is
  // the real checkpoint's measurement, scripts/fabric_mtp_classes.sh).
  DGPP_LOG_INFO("world 2 MTP graph: A took {} steps for {} tokens", o.mtp_steps_a, kSteps);
}

// The mmap'ed n-gram table: the same world, the same MTP
// graph engine (scalar and batched replays, the eager prefills and the
// fallbacks' rows) with the table left in the checkpoint's shard and each
// walk's rows gathered by a host node forked inside the walk — the
// transcripts bitwise the resident table's, on every rank.
DGPP_TEST(qwen_engines_loopback_world_2_mtp_graph_over_the_mmap_table_matches_resident) {
  const QwenTextConfig cfg = qwenfx::tiny_config();
  const std::string dir = "qwen_engine_fixture";
  qwenfx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  std::vector<RankOutcome> resident(kWorld), mapped(kWorld);
  for (int pass = 0; pass < 2; ++pass) {
    dgpp::QwenLayerStream::set_ngram_table_mmap(pass == 1);
    std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort + 6 + 2 * pass);
    require(!buses.empty(), "the loopback bus world failed to start");
    std::vector<RankOutcome>& outs = pass == 0 ? resident : mapped;
    ConstructBarrier barrier(kWorld);
    std::vector<std::thread> workers;
    for (int r = 0; r < kWorld; ++r)
      workers.emplace_back(rank_work_mtp, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B),
                           std::cref(C), buses[static_cast<size_t>(r)].get(), &barrier,
                           &outs[static_cast<size_t>(r)], /*depth=*/1);
    for (auto& t : workers) t.join();
    dgpp::QwenLayerStream::set_ngram_table_mmap(false);
    for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  }
  for (int r = 0; r < kWorld; ++r) {
    const RankOutcome& a = resident[static_cast<size_t>(r)];
    const RankOutcome& b = mapped[static_cast<size_t>(r)];
    require(b.ea == a.ea && b.eb == a.eb && b.ec == a.ec, "the eager transcripts differ over the mmap'ed table");
    require(b.ma == a.ma && b.mb == a.mb && b.mc == a.mc, "the MTP graph transcripts differ over the mmap'ed table");
    require(b.mtp_steps_a == a.mtp_steps_a, "the MTP step count differs over the mmap'ed table");
  }
  DGPP_LOG_INFO("world 2 MTP graph over the mmap'ed table: A {} ({} steps) | B {} | C {} — the resident table's",
                ids_text(mapped[0].ma), mapped[0].mtp_steps_a, ids_text(mapped[0].mb), ids_text(mapped[0].mc));
}

// Depth 2 (2026-09-10, kDraftChain): the chained draft rows through the
// block, the ring copied around them — the greedy transcripts still the
// plain engine's, scalar and batched, on every rank.
DGPP_TEST(qwen_engines_loopback_world_2_mtp_depth2_graph_matches_plain_decode) {
  const QwenTextConfig cfg = qwenfx::tiny_config();
  const std::string dir = "qwen_engine_fixture";
  qwenfx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort + 2);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<RankOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_mtp, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B),
                         std::cref(C), buses[static_cast<size_t>(r)].get(), &barrier,
                         &outs[static_cast<size_t>(r)], /*depth=*/2);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].ma == outs[0].ma && outs[static_cast<size_t>(r)].mb == outs[0].mb,
            "the ranks' depth-2 MTP transcripts differ");
  const RankOutcome& o = outs[0];
  require(o.ma == o.ea, "the depth-2 MTP scalar transcript differs from the plain eager engine's");
  require(o.mb == o.eb, "the depth-2 MTP batched transcript of B differs from the plain eager engine's");
  require(o.mc == o.ec, "the depth-2 MTP batched transcript of C differs from the plain eager engine's");
  DGPP_LOG_INFO("world 2 MTP depth 2: A took {} steps for {} tokens", o.mtp_steps_a, kSteps);
}

DGPP_TEST(qwen_engines_loopback_world_2_graph_matches_eager) {
  const QwenTextConfig cfg = qwenfx::tiny_config();
  const std::string dir = "qwen_engine_fixture";
  qwenfx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  DGPP_LOG_INFO("world 1 eager: A {} | B {} | C {}", ids_text(ref.a), ids_text(ref.b), ids_text(ref.c));

  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<RankOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B), std::cref(C),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r) {
    require(outs[static_cast<size_t>(r)].ea == outs[0].ea && outs[static_cast<size_t>(r)].ga == outs[0].ga &&
                outs[static_cast<size_t>(r)].ba == outs[0].ba,
            "the ranks' transcripts differ");
  }
  const RankOutcome& o = outs[0];
  DGPP_LOG_INFO("world 2 eager: A {} | B {} | C {}", ids_text(o.ea), ids_text(o.eb), ids_text(o.ec));
  DGPP_LOG_INFO("world 2 graph scalar A {}", ids_text(o.ga));
  DGPP_LOG_INFO("world 2 graph batched: A {} | B {} | C {}", ids_text(o.ba), ids_text(o.bb), ids_text(o.bc));
  // The world-2 eager engine against world 1: reported, the first tokens
  // must agree (the folds reassociate; a near tie may flip a late token).
  const size_t pa = agreeing_prefix(o.ea, ref.a), pb = agreeing_prefix(o.eb, ref.b), pc = agreeing_prefix(o.ec, ref.c);
  DGPP_LOG_INFO("world 2 vs world 1 agreeing prefixes: A {}/{} B {}/{} C {}/{}", pa, ref.a.size(), pb, ref.b.size(),
                pc, ref.c.size());
  require(pa >= 3 && pb >= 3 && pc >= 3, "the world-2 eager engine diverges from world 1 at the start");
  // The graph engine against the eager engine at the same world: exact.
  require(o.ga == o.ea, "the scalar graph transcript differs from the eager engine's");
  require(o.ba == o.ea, "the batched graph transcript of A differs from the eager engine's");
  require(std::equal(o.bb.begin(), o.bb.end(), o.eb.begin()), "the batched graph transcript of B differs");
  require(std::equal(o.bc.begin(), o.bc.end(), o.ec.begin()), "the batched graph transcript of C differs");
  require(o.bb.size() == 10 && o.bc.size() == 8, "the batched transcripts' lengths");
}

int main() { return dgpp::test::run_all(); }
