#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "common/test.hpp"
#include "engine/eager_engine.hpp"
#ifdef DGPP_TEST_HAS_BUS
#include "engine/graph_engine.hpp"
#include "serve/prefill_policy.hpp"
#endif
#include "glm4_fixture.hpp"
#include "dsv4_fixture.hpp"
#include "dsv41_fixture.hpp"
#include "models/dsv41/model.hpp"
#include "qwen35_fixture.hpp"
#include "models/dsv4/model.hpp"
#include "models/qwen/model35.hpp"
#include "glm_dsa_fixture.hpp"
#include "mimo_fixture.hpp"
#include "models/glm4/forward.hpp"
#include "models/glm/forward.hpp"
#include "models/glm/route_audit.hpp"
#include "models/glm_dsa/model.hpp"
#include "models/mimo/forward.hpp"

using namespace dgpp;
GlmTextConfig glm_tp_test_config();
void glm_tp_write_fixture(const std::string& dir);
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::vector<int64_t> tokens(int count, int vocab, int salt) {
  std::vector<int64_t> out(count);
  for (int i = 0; i < count; ++i) out[i] = (i * 17 + salt) % vocab;
  return out;
}
template <class Output> void exact(const Output& a, const Output& b, const char* message) {
  require(a.logits.size() == b.logits.size() && !a.logits.empty(), "output shape");
  require(std::memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float)) == 0, message);
}
template <class Output> void close(const Output& a, const Output& b, const char* message) {
  require(a.logits.size() == b.logits.size() && !a.logits.empty(), "output shape");
  double error = 0, norm = 0;
  for (size_t i = 0; i < a.logits.size(); ++i) {
    require(std::isfinite(a.logits[i]) && std::isfinite(b.logits[i]), "finite logits");
    error += std::pow(a.logits[i] - b.logits[i], 2);
    norm += std::pow(b.logits[i], 2);
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-30));
  std::printf("  %s: relative L2 %.8g\n", message, relative);
  // Different row shapes can choose different GEMM reductions. Use the
  // existing session/group numerical tier and certify any changed argmax;
  // same-geometry isolation and snapshot checks above remain bitwise.
  require(relative < 0.01, message);
  const int vocab = b.lm_vocab_count;
  require(vocab > 0 && b.logits.size() % vocab == 0, "numeric comparison vocabulary");
  const auto audit = glm_route::audit_top1_near_ties(a.logits.data(), b.logits.data(), vocab, b.logits.size() / vocab);
  glm_route::certify_top1_near_ties(audit, b.logits.size() / vocab, message);
}
#ifdef DGPP_TEST_HAS_BUS
template <class Model> void check_graph(Model& model, Model& reference, int vocab,
    const std::vector<int64_t>& a, const std::vector<int64_t>& b, int unit, bool compact) {
  const int decoder = model.max_session_requests() - 1;
  net::BusOptions options;
  options.world_size = 1;
  options.my_rank = 0;
  options.rendezvous_port = 30994;
  options.lat_slots = options.bulk_slots = 8;
  options.lat_slot_bytes = options.bulk_slot_bytes = 262144;
  options.completion_timeout_ms = 60000;
  net::CollectiveBus bus(options);
  std::string error;
  require(bus.start(&error), error.c_str());
  uint16_t* scratch = nullptr;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                             2 * kPickScratchElems(1), cudaHostAllocDefault));
  try {
    {
      GraphEngineAdapter<Model> engine(&model, &bus, 0, 1, scratch, vocab, 60000,
          2, nullptr, nullptr, 1, nullptr, 0, model.mtp_enabled() && Model::kBatchedDraftChain ? 3 : 1, compact);
      sched::AdmissionPolicy policy;
      policy.prefill_budget_tokens = 2 * unit;
      policy = serve::resolve_prefill_policy(policy, engine, std::nullopt);
      require(policy.prefill_early_admission, "generic mixed executor defaults to early admission");
      int64_t pending = engine.prefill(decoder, b);
      const auto first = reference.session_prefill(decoder, b);
      require(pending == std::max_element(first.logits.begin(), first.logits.end()) - first.logits.begin(),
              "graph decoder starts at oracle token");
      engine.reserve(decoder, 768);
      (void)engine.prefill(0, b);
      engine.reserve(0, 768);
      for (int i = 0; i < 12; ++i) {
        const auto baseline = engine.step_batch({0, decoder});
        if (compact)
          require(engine.decode_batch_stats().slots == 2, "compact graph must skip holes between live requests");
        for (const auto token : baseline.back()) {
          const auto next = reference.session_step(decoder, pending);
          const auto wanted = std::max_element(next.logits.begin(), next.logits.end()) - next.logits.begin();
          if (token != wanted)
            std::printf("ordinary graph mismatch: mtp=%d compact=%d slot=%d token=%d wanted=%ld margin=%.9g\n",
                model.mtp_enabled(), compact, decoder, token, wanted,
                next.logits[wanted] - next.logits[token]);
          require(token == wanted, "ordinary graph speculative verdict differs from plain oracle");
          pending = token;
        }
      }
      engine.close(0);
      engine.begin_prefill(0, a, 768, 2 * unit, {});
      require(engine.can_mix_prefill(0, {decoder}), "graph accepts reader and padded decoder family");
      int64_t progress = 0;
      std::vector<int64_t> cuts;
      for (;;) {
        const auto result = engine.advance_prefill_and_step(0, 2 * unit, {decoder});
        if (compact)
          require(engine.decode_batch_stats().slots == 2, "mixed graph must map the high decoder slot into the smallest bucket");
        require(result.decoded.size() == 1 && !result.decoded[0].empty(), "mixed graph decoder progresses");
        for (const auto token : result.decoded[0]) {
          const auto next = reference.session_step(decoder, pending);
          const auto wanted = std::max_element(next.logits.begin(), next.logits.end()) - next.logits.begin();
          if (token != wanted)
            std::printf("graph mismatch: mtp=%d compact=%d reader_pos=%ld token=%d wanted=%ld margin=%.9g\n",
                model.mtp_enabled(), compact, progress, token, wanted,
                next.logits[wanted] - next.logits[token]);
          require(token == wanted, "mixed graph speculative verdict differs from plain oracle");
          pending = token;
        }
        progress += result.prefill.computed_tokens;
        if (result.prefill.first_token >= 0) {
          const auto reader = reference.session_prefill(0, a, cuts);
          require(result.prefill.first_token ==
                      std::max_element(reader.logits.begin(), reader.logits.end()) - reader.logits.begin(),
                  "mixed graph reader differs from matching cuts");
          break;
        }
        cuts.push_back(progress);
      }
      require(progress == static_cast<int64_t>(a.size()), "graph counts every reader token once");
      engine.reserve(0, 768);
      require(!engine.step_batch({0, decoder}).empty(), "completed mixed reader has valid graph draft state");
      engine.close(0); engine.close(decoder);
      reference.session_close(0); reference.session_close(decoder);
    }
    require(model.kv_blocks_in_use() == 0, "graph releases every reservation");
  } catch (...) { cudaFreeHost(scratch); throw; }
  DGPP_CUDA_OK(cudaFreeHost(scratch));
}
#endif
template <class Model, class Factory> void check(Factory make, int vocab, int alignment = 1, bool compact_supported = true, bool bounded = false) {
  const int unit = std::max(8, alignment), reserve = 768;
  const auto a = tokens(std::max(77, 2 * alignment + 13), vocab, 3);
  const auto changed = tokens(a.size(), vocab, 9), b = tokens(std::max(53, alignment + 5), vocab, 5);
  for (bool mtp : {false, true}) for (bool compact : {false, true}) {
    if (compact && !compact_supported) continue;
    auto model = make(mtp, compact);
    auto reference = make(mtp, compact);
    // Same shape and same second request, a different first request. The
    // second request's target AND draft must be independent of its neighbor.
    const auto first = model->session_prefill_group({0, 1}, {&a, &b});
    typename Model::Outputs draft;
    if (mtp) draft = model->session_draft(1, {7});
    model->session_close(0); model->session_close(1);
    const auto second = model->session_prefill_group({0, 1}, {&changed, &b});
    exact(first[1], second[1], "grouped target reads another request");
    if (mtp) exact(draft, model->session_draft(1, {7}), "grouped draft reads another request");
    model->session_close(0); model->session_close(1);

    // A yielded prefill uses identical cuts to its uninterrupted oracle.
    std::vector<int64_t> cuts;
    for (int cut = unit; cut < int(a.size()); cut += unit) cuts.push_back(cut);
    const auto expected = reference->session_prefill(0, a, cuts);
    (void)model->session_prefill(1, b);
    auto cursor = model->session_prefill_begin(0, a, reserve, unit);
    while (!model->session_prefill_advance(cursor)) (void)model->session_step(1, 11);
    exact(cursor.output, expected, "interleaved prefill target differs at identical cuts");
    if (mtp) exact(model->session_draft(0, {7}), reference->session_draft(0, {7}),
                   "interleaved prefill draft differs at identical cuts");
    model->session_close(0); model->session_close(1); reference->session_close(0);

    // Each reader chunk shares one walk with a four-row speculative prefix.
    // Accept different prefix lengths and compare the next forced decode.
    (void)model->session_prefill(1, b); (void)reference->session_prefill(1, b);
    cursor = model->session_prefill_begin(0, a, reserve, 2 * unit);
    auto oracle = reference->session_prefill_begin(0, a, reserve, unit);
    oracle.relative_budget = true;  // follow the mixed walk's actual cuts
    int iteration = 0;
    while (cursor.next < cursor.end) {
      const auto outputs = model->session_prefill_verify(cursor, 2 * unit, {1}, {{7, 8, 9, 10}});
      (void)reference->session_prefill_advance(oracle,
          (cursor.next - oracle.next + alignment - 1) / alignment * alignment);
      const auto verification = reference->session_verify(1, {7, 8, 9, 10});
      require(cursor.next == oracle.next, "shared physical row budget");
      if (!bounded || cursor.next == cursor.end) close(cursor.output, oracle.output, "mixed reader");
      close(outputs[0], verification, "mixed verifier");
      const int accepted = 1 + iteration++ % 4;
      model->session_rollback(1, accepted, 4); reference->session_rollback(1, accepted, 4);
      close(model->session_step(1, 13), reference->session_step(1, 13), "mixed rollback continuation");
    }
    if (mtp) close(model->session_draft(0, {7}), reference->session_draft(0, {7}), "mixed draft history");
    model->session_close(0); model->session_close(1);
    reference->session_close(0); reference->session_close(1);
    require(model->kv_blocks_in_use() == 0 && reference->kv_blocks_in_use() == 0, "released all KV blocks");

    // Mutate one reader while retaining identical row geometry. The other
    // reader AND decoder must be independent, through yields and rollback.
    // Fixed geometry isolates state leaks from GEMM-shape rounding.
    auto ca = model->session_prefill_begin(0, a, reserve, 2 * unit);
    auto cb = model->session_prefill_begin(1, b, reserve, 2 * unit);
    auto ra = reference->session_prefill_begin(0, changed, reserve, 2 * unit);
    auto rb = reference->session_prefill_begin(1, b, reserve, 2 * unit);
    (void)model->session_prefill(2, b); (void)reference->session_prefill(2, b);
    iteration = 0;
    while (ca.next < ca.end || cb.next < cb.end) {
      std::vector<typename Model::PrefillCursor*> readers, oracles;
      if (ca.next < ca.end) { readers.push_back(&ca); oracles.push_back(&ra); }
      if (cb.next < cb.end) { readers.push_back(&cb); oracles.push_back(&rb); }
      const std::vector<int64_t> budgets(readers.size(), 2 * unit);
      if (iteration++ == 0) {
        (void)model->session_prefill_advance_group(readers, budgets);
        (void)reference->session_prefill_advance_group(oracles, budgets);
      } else {
        const auto out = model->session_prefill_verify_group(readers, budgets, {2}, {{7, 8, 9, 10}});
        const auto wanted = reference->session_prefill_verify_group(oracles, budgets, {2}, {{7, 8, 9, 10}});
        exact(out[0], wanted[0], "decoder reads another request's prefill");
        model->session_rollback(2, 2, 4); reference->session_rollback(2, 2, 4);
      }
      for (size_t i = 0; i < readers.size(); ++i) {
        require(readers[i]->next == oracles[i]->next, "group continuation cut");
        if (readers[i]->req == 1 && (!bounded || readers[i]->next == readers[i]->end))
          exact(readers[i]->output, oracles[i]->output, "group continuation reads another request");
      }
    }
    if (mtp) exact(model->session_draft(1, {7}), reference->session_draft(1, {7}),
                    "group continuation draft reads another request");
    for (int req = 0; req < 3; ++req) { model->session_close(req); reference->session_close(req); }

    // The same request lifetime is available to the eager scheduler adapter,
    // including cancellation before completion and immediate slot reuse.
    const auto pick = [](const DecodeOutputs& out) -> int32_t {
      return std::max_element(out.logits.begin(), out.logits.end()) - out.logits.begin();
    };
    EagerEngineAdapter<Model> engine(model.get(), 3, pick);
    require(engine.prefill_chunk_alignment() > 0, "eager resumable execution");
    engine.begin_prefill(0, a, reserve, unit, {});
    require(engine.advance_prefill(0).first_token < 0, "prefill yielded");
    engine.close(0);
    require(model->kv_blocks_in_use() == 0, "cancelled cursor releases reservations");
    engine.begin_prefill(0, b, reserve, unit, {});
    sched::SchedulerEngine::PrefillProgress progress;
    do { progress = engine.advance_prefill(0); } while (progress.first_token < 0);
    require(engine.step(0).size() == 1, "completed eager reader decodes");
    engine.close(0);
    require(model->kv_blocks_in_use() == 0, "reused slot releases reservations");
    require(engine.supports_mixed_prefill(), "eager shares the mixed executor");
    int64_t pending = engine.prefill(1, b);
    (void)reference->session_prefill(1, b);
    engine.begin_prefill(0, a, reserve, 2 * unit, {});
    while (true) {
      const auto result = engine.advance_prefill_and_step(0, 2 * unit, {1});
      const auto next = reference->session_step(1, pending);
      require(result.decoded[0].size() == 1 && result.decoded[0][0] == pick(next), "eager mixed decoder token");
      pending = result.decoded[0][0];
      if (result.prefill.first_token >= 0) break;
    }
    engine.close(0); engine.close(1); reference->session_close(1);

    auto stale = model->session_prefill_begin(0, a, reserve, unit);
    model->session_close(0);
    auto replacement = model->session_prefill_begin(0, b, reserve, unit);
    bool rejected = false;
    try { (void)model->session_prefill_advance(stale); } catch (const std::logic_error&) { rejected = true; }
    require(rejected && replacement.next == 0 && model->session_position(0) == 0,
            "a cancelled cursor cannot advance a reused slot at the same position");
    model->session_close(0);
    uint8_t* snapshot_bytes = nullptr;
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&snapshot_bytes), model->session_snapshot_bytes()));
    typename Model::SessionSnapshotMeta meta;
    typename Model::SnapshotRequest snapshot{unit, snapshot_bytes, &meta};
    cursor = model->session_prefill_begin(0, a, reserve, unit, {unit}, &snapshot);
    while (!model->session_prefill_advance(cursor)) {}
    require(snapshot.taken, "reader takes its requested snapshot");
    const auto full = cursor.output;
    if (mtp) draft = model->session_draft(0, {7});
    model->session_close(0);
    model->session_attach(2, snapshot_bytes, meta);
    auto restored = model->session_prefill_begin(2, a, reserve, unit, {unit}, nullptr, unit);
    while (!model->session_prefill_advance(restored)) {}
    exact(restored.output, full, "snapshot restored into a different request changes target");
    if (mtp) exact(model->session_draft(2, {7}), draft, "snapshot restored into a different request changes draft");
    model->session_close(2);
    model->session_release_snapshot(meta);
    DGPP_CUDA_OK(cudaFree(snapshot_bytes));
    require(model->kv_blocks_in_use() == 0, "reader snapshots release all references");
#ifdef DGPP_TEST_HAS_BUS
    check_graph(*model, *reference, vocab, a, b, unit, compact);
#endif
  }
}
template <class Model>
void check_decoder_mhc_recipe(Model& model, Model& reference, int vocab, int hidden, const std::string& prefix) {
  const auto decode_env = prefix + "_CAPTURE_DECODE";
  const auto prefill_env = prefix + "_CAPTURE_PREFILL";
  const auto peer = tokens(17, vocab, 3);
  const auto prompt = tokens(259, vocab, 13);
  const auto feeds = tokens(4, vocab, 7);
  (void)model.session_prefill(2, peer);
  (void)reference.session_prefill(2, peer);
  auto reader = model.session_prefill_begin(0, prompt, 768, 128);
  // Compare the first sublayer input, before attention or expert routing
  // can introduce their own numerical differences. Layer zero has no
  // Engram, so identical embeddings must keep the decoder's mHC recipe.
  setenv(decode_env.c_str(), "1", 1);
  setenv(prefill_env.c_str(), "1", 1);
  try {
    (void)reference.session_verify(2, feeds);
    (void)model.session_prefill_verify(reader, 128, {2}, {feeds});
  } catch (...) {
    unsetenv(decode_env.c_str());
    unsetenv(prefill_env.c_str());
    throw;
  }
  unsetenv(decode_env.c_str());
  unsetenv(prefill_env.c_str());
  require(!model.debug_sites().empty() && !reference.debug_sites().empty(), "mHC site capture exists");
  if (std::getenv("DGPP_PREFILL_COMPARE_SITES")) {
    using Site = typename Model::SiteCapture;
    for (size_t layer = 0; layer < reference.debug_sites().size(); ++layer) {
      require(layer < model.debug_sites().size(), "mixed site capture layer count");
      for (const auto& [name, member] : {
          std::pair{"attention input", &Site::x_attn}, std::pair{"attention result", &Site::attn_out},
          std::pair{"attention streams", &Site::streams_after_attn}, std::pair{"expert input", &Site::x_ffn},
          std::pair{"expert result", &Site::ffn_out}}) {
        const auto& a = model.debug_sites()[layer].*member;
        const auto& b = reference.debug_sites()[layer].*member;
        require(a.size() >= b.size(), "mixed site capture prefix shape");
        size_t changed = 0;
        for (size_t i = 0; i < b.size(); ++i) changed += a[i] != b[i];
        std::printf("mixed site trace %s layer %zu %s: %zu/%zu changed\n",
                    prefix.c_str(), layer, name, changed, b.size());
      }
    }
  }
  for (size_t layer = 0; layer < reference.debug_sites().size(); ++layer) {
    const auto& a = model.debug_sites().at(layer);
    const auto& b = reference.debug_sites()[layer];
    require(a.x_ffn.size() >= b.x_ffn.size() && a.ffn_out.size() >= b.ffn_out.size(),
            "mixed expert decoder prefix shape");
    if (std::equal(b.x_ffn.begin(), b.x_ffn.end(), a.x_ffn.begin()))
      require(std::equal(b.ffn_out.begin(), b.ffn_out.end(), a.ffn_out.begin()),
              "identical decoder expert inputs must retain the native expert output");
  }
  const auto& actual = model.debug_sites().front().x_attn;
  const auto& expected = reference.debug_sites().front().x_attn;
  require(expected.size() == feeds.size() * hidden && actual.size() >= expected.size(),
          "mHC decoder prefix shape");
  require(std::equal(expected.begin(), expected.end(), actual.begin()),
          "mixed decoder mHC input must equal its isolated decode recipe");
  if constexpr (requires { model.debug_sites().front().post_attn; }) {
    const auto& actual_post = model.debug_sites().front().post_attn;
    const auto& expected_post = reference.debug_sites().front().post_attn;
    require(expected_post.size() == feeds.size() * 4 && actual_post.size() >= expected_post.size(),
            "mHC decoder post coefficient shape");
    require(std::equal(expected_post.begin(), expected_post.end(), actual_post.begin()),
            "mixed decoder mHC post coefficients must equal its isolated decode recipe");
  }
  if constexpr (requires { model.debug_layer_pre(); }) {
    // V4.1 collapses with the preceding site's coefficients, so its first
    // attention input alone cannot expose a changed coefficient recipe.
    // The first layer's FFN coefficients are captured before MoE output
    // changes the stream; compare their fp32 values without BF16 masking.
    require(!model.debug_layer_pre().empty() && !reference.debug_layer_pre().empty(), "mHC coefficient capture exists");
    const auto& actual_pre = model.debug_layer_pre().front();
    const auto& expected_pre = reference.debug_layer_pre().front();
    require(expected_pre.size() == feeds.size() * 4 && actual_pre.size() >= expected_pre.size(),
            "mHC decoder coefficient shape");
    require(std::equal(expected_pre.begin(), expected_pre.end(), actual_pre.begin()),
            "mixed decoder mHC coefficients must equal its isolated decode recipe");
  }
}
}  // namespace

DGPP_TEST(shared_prefill_glm) {
  const auto cfg = glm_tp_test_config();
  const std::string dir = "shared_glm_fixture";
  glm_tp_write_fixture(dir);
  check<GlmDiagnosticModel>([&](bool mtp, bool compact) {
    return std::make_unique<GlmDiagnosticModel>(cfg, dir, 2048, 4096, nullptr, 0, 1,
        GlmResidency::Resident, GlmHeadSharding::Full, 4, mtp, LatentFormat::kBf16, compact);
  }, cfg.vocab_size, cfg.dsa_config().index_kpool);
}
DGPP_TEST(shared_prefill_glm4) {
  const auto cfg = glm4fx::tiny_config();
  const std::string dir = "shared_glm4_fixture";
  glm4fx::write_fixture(cfg, dir);
  check<Glm4Model>([&](bool mtp, bool compact) {
    return std::make_unique<Glm4Model>(cfg, dir, 256, 2048, Glm4Residency::Resident,
                                      nullptr, 0, 1, 4, mtp, 32, compact);
  }, cfg.vocab_size);
}
DGPP_TEST(shared_prefill_glm_dsa) {
  const auto cfg = glmdsafx::tiny_config();
  const std::string dir = "shared_glm_dsa_fixture";
  glmdsafx::write_fixture(cfg, dir);
  check<GlmDsaModel>([&](bool mtp, bool compact) {
    return std::make_unique<GlmDsaModel>(cfg, dir, 256, 2048, GlmDsaResidency::Resident,
                                        nullptr, 0, 1, 4, mtp, 16, LatentFormat::kBf16, compact);
  }, cfg.vocab_size);
}
DGPP_TEST(shared_prefill_mimo) {
  const auto cfg = mimofx::tiny_config();
  const std::string dir = "shared_mimo_fixture";
  mimofx::write_fixture(cfg, dir);
  check<MimoModel>([&](bool mtp, bool compact) {
    return std::make_unique<MimoModel>(cfg, dir, 256, 2048, MimoResidency::Resident,
                                      nullptr, 0, 1, 4, mtp, 16, LatentFormat::kBf16, compact);
  }, cfg.vocab_size);
}
DGPP_TEST(shared_prefill_qwen35) {
  for (bool mixed_quant : {false, true}) {
    const auto cfg = qwen35fx::tiny_config(mixed_quant);
    const std::string dir = mixed_quant ? "shared_qwen35_mixed_fixture" : "shared_qwen35_fixture";
    qwen35fx::write_fixture(dir, mixed_quant);
    check<Qwen35Model>([&](bool mtp, bool) {
      return std::make_unique<Qwen35Model>(cfg, dir, 256, 2048, LoaderResidency::Resident,
                                          nullptr, 0, 1, 4, 16, mtp);
    }, cfg.vocab_size);
  }
}
DGPP_TEST(shared_prefill_qwen35_per_tensor) {
  const auto cfg = qwen35fx::tiny_config();
  const std::string dir = "shared_qwen35_per_tensor_fixture";
  qwen35fx::write_fixture(dir);
  struct RestoreRecipe {
    ~RestoreRecipe() { Qwen35Model::set_prefill_fp8_per_tensor(false); }
  } restore;
  Qwen35Model::set_prefill_fp8_per_tensor(true);
  check<Qwen35Model>([&](bool mtp, bool) {
    return std::make_unique<Qwen35Model>(cfg, dir, 256, 2048, LoaderResidency::Resident,
                                        nullptr, 0, 1, 4, 16, mtp);
  }, cfg.vocab_size);
}

DGPP_TEST(shared_prefill_qwen35_large_projection_groups) {
  for (bool mixed_quant : {false, true}) for (bool mtp : {false, true}) {
    const auto cfg = qwen35fx::tiny_config(mixed_quant);
    const std::string dir = mixed_quant ? "shared_qwen35_large_mixed_fixture" : "shared_qwen35_large_fixture";
    qwen35fx::write_fixture(dir, mixed_quant);
    Qwen35Model model(cfg, dir, 512, 2048, LoaderResidency::Resident, nullptr, 0, 1, 4, 16, mtp);
    Qwen35Model reference(cfg, dir, 512, 2048, LoaderResidency::Resident, nullptr, 0, 1, 4, 16, mtp);
    const auto a = tokens(301, cfg.vocab_size, 3), b = tokens(293, cfg.vocab_size, 5);
    const auto changed = tokens(a.size(), cfg.vocab_size, 9), decoder = tokens(53, cfg.vocab_size, 7);
    auto run = [&](const std::vector<int64_t>& first) {
      (void)model.session_prefill(2, decoder);
      (void)reference.session_prefill(2, decoder);
      auto ca = model.session_prefill_begin(0, first, 768, 192);
      auto cb = model.session_prefill_begin(1, b, 768, 192);
      const std::vector<Qwen35Model::PrefillCursor*> readers{&ca, &cb};
      std::vector<int64_t> cuts_a, cuts_b;
      while (ca.next != ca.end || cb.next != cb.end) {
        const auto out = model.session_prefill_verify_group(readers, {192, 192}, {2}, {{7, 8, 9, 10}});
        const auto wanted = reference.session_verify(2, {7, 8, 9, 10});
        close(out[0], wanted, "large reader projections preserve verifier");
        model.session_rollback(2, 4, 4);
        reference.session_rollback(2, 4, 4);
        if (ca.next != ca.end) cuts_a.push_back(ca.next);
        if (cb.next != cb.end) cuts_b.push_back(cb.next);
      }
      close(ca.output, reference.session_prefill(0, first, cuts_a), "large first reader at matching cuts");
      close(cb.output, reference.session_prefill(1, b, cuts_b), "large second reader at matching cuts");
      if (mtp) close(model.session_draft(1, {7}), reference.session_draft(1, {7}),
                     "large shared projections preserve draft history");
      auto output = cb.output;
      for (int req : {0, 1, 2}) { model.session_close(req); reference.session_close(req); }
      return output;
    };
    const auto before = run(a);
    const auto after = run(changed);
    exact(before, after, "large shared projections read another request");
    require(model.kv_blocks_in_use() == 0 && reference.kv_blocks_in_use() == 0,
            "large grouped prefills release reservations");
  }
}
DGPP_TEST(shared_prefill_dsv4) {
  const auto cfg = dsv4fx::tiny_config();
  const std::string dir = "shared_dsv4_fixture";
  dsv4fx::write_fixture(cfg, dir);
  check<Dsv4Model>([&](bool mtp, bool compact) {
    return std::make_unique<Dsv4Model>(cfg, dir, 512, 2048, Dsv4Residency::Resident,
                                      nullptr, 0, 1, 4, mtp, 32, compact);
  }, cfg.vocab_size, 128);
}
DGPP_TEST(shared_prefill_dsv41) {
  const auto cfg = dsv41fx::tiny_config();
  const std::string dir = "shared_dsv41_mixed_fixture";
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  for (bool bounded : {false, true})
    check<Dsv41Model>([&](bool mtp, bool compact) {
      auto model = std::make_unique<Dsv41Model>(cfg, dir, 512, 4096, Dsv41Residency::Resident,
                                               nullptr, 0, 1, 4, mtp, 32, compact);
      model->set_prefill_bounded(bounded);
      return model;
    }, cfg.vocab_size, 128, true, bounded);
}
DGPP_TEST(shared_prefill_dsv41_continuations) {
  const auto cfg = dsv41fx::tiny_config();
  const std::string dir = "shared_dsv41_fixture";
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  for (bool bounded : {false, true}) {
    Dsv41Model model(cfg, dir, 512, 4096, Dsv41Residency::Resident, nullptr, 0, 1, 3, true, 16);
    Dsv41Model reference(cfg, dir, 512, 4096, Dsv41Residency::Resident, nullptr, 0, 1, 3, true, 16);
    model.set_prefill_bounded(bounded); reference.set_prefill_bounded(bounded);
    const auto a = tokens(385, cfg.vocab_size, 7), b = tokens(259, cfg.vocab_size, 13);
    const auto peer = tokens(17, cfg.vocab_size, 3);
    auto ca = model.session_prefill_begin(0, a, 768, 128);
    auto cb = model.session_prefill_begin(1, b, 768, 128);
    auto oracle = reference.session_prefill_begin(0, a, 768, 128);
    (void)model.session_prefill(2, peer);
    while (ca.next < ca.end) {
      (void)model.session_prefill_advance(ca);
      (void)reference.session_prefill_advance(oracle);
      if (cb.next < cb.end) (void)model.session_prefill_advance(cb);
      (void)model.session_step(2, 11);
    }
    exact(ca.output, oracle.output, "bounded continuation shares tail scratch with a peer");
    exact(model.session_draft(0, {7}), reference.session_draft(0, {7}), "bounded continuation draft history");
    for (int req = 0; req < 3; ++req) model.session_close(req);
    reference.session_close(0);
    // The grouped draft must read its own packed decoder segment too.
    const auto short_a = tokens(53, cfg.vocab_size, 7), short_b = tokens(37, cfg.vocab_size, 3);
    const auto other = tokens(53, cfg.vocab_size, 11);
    (void)model.session_prefill_group({0, 1}, {&short_a, &short_b});
    const auto expected = model.session_draft(1, {7});
    model.session_close(0); model.session_close(1);
    (void)model.session_prefill_group({0, 1}, {&other, &short_b});
    exact(model.session_draft(1, {7}), expected, "DSpark grouped draft reads another request");
    model.session_close(0); model.session_close(1);
    require(model.kv_blocks_in_use() == 0, "bounded continuations release all KV");
  }
}
DGPP_TEST(shared_prefill_dsv4_decoder_mhc_recipe) {
  // Wider reductions expose rounding differences hidden by the 256-wide
  // smoke fixture; the layer schedule and cache geometry stay small.
  std::string config_json = dsv4fx::tiny_config_json();
  const std::string hidden = "\"hidden_size\": 256";
  config_json.replace(config_json.find(hidden), hidden.size(), "\"hidden_size\": 2048");
  const auto cfg = Dsv4Config::parse(minijson::parse(config_json).root);
  const std::string dir = "shared_dsv4_mhc_fixture";
  dsv4fx::write_fixture(cfg, dir, config_json.c_str());
  Dsv4Model model(cfg, dir, 512, 4096, Dsv4Residency::Resident, nullptr, 0, 1, 3, false, 16);
  Dsv4Model reference(cfg, dir, 512, 4096, Dsv4Residency::Resident, nullptr, 0, 1, 3, false, 16);
  check_decoder_mhc_recipe(model, reference, cfg.vocab_size, cfg.hidden_size, "DGPP_DSV4");
}
DGPP_TEST(shared_prefill_dsv41_decoder_mhc_recipe) {
  // Wider reductions expose rounding differences hidden by the 256-wide
  // smoke fixture; the layer schedule and cache geometry stay small.
  std::string config_json = dsv41fx::tiny_config_json();
  const std::string hidden = "\"hidden_size\": 256";
  config_json.replace(config_json.find(hidden), hidden.size(), "\"hidden_size\": 2048");
  const auto cfg = Dsv41TextConfig::parse(minijson::parse(config_json).root);
  const std::string dir = "shared_dsv41_mhc_fixture";
  dsv41fx::write_fixture(cfg, dir, config_json.c_str());
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  for (bool bounded : {false, true}) {
    Dsv41Model model(cfg, dir, 512, 4096, Dsv41Residency::Resident, nullptr, 0, 1, 3, false, 16);
    Dsv41Model reference(cfg, dir, 512, 4096, Dsv41Residency::Resident, nullptr, 0, 1, 3, false, 16);
    model.set_prefill_bounded(bounded); reference.set_prefill_bounded(bounded);
    check_decoder_mhc_recipe(model, reference, cfg.vocab_size, cfg.hidden_size, "DGPP_DSV41");
  }
}
DGPP_TEST(shared_prefill_dsv41_workspace) {
  auto cfg = dsv41fx::tiny_config();
  cfg.sliding_window = 384;
  const std::string dir = "shared_dsv41_workspace_fixture";
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  Dsv41Model model(cfg, dir, 512, 4096, Dsv41Residency::Resident, nullptr, 0, 1, 3, true, 16);
  Dsv41Model reference(cfg, dir, 512, 4096, Dsv41Residency::Resident, nullptr, 0, 1, 3, true, 16);
  model.set_prefill_bounded(true); reference.set_prefill_bounded(true);
  require(model.max_mixed_prefill_readers(4) == 1, "bounded mixed workspace accounts for retained tails");
  const auto a = tokens(385, cfg.vocab_size, 7), b = tokens(385, cfg.vocab_size, 13);
  auto ca = model.session_prefill_begin(0, a, 768, 128);
  auto cb = model.session_prefill_begin(1, b, 768, 128);
  auto ra = reference.session_prefill_begin(0, a, 768, 128);
  auto rb = reference.session_prefill_begin(1, b, 768, 128);
  while (ca.next < ca.end) {
    // The last call's two 384-row tails exceed the 512-row workspace.
    // Group advancement must fall back before walking either reader.
    (void)model.session_prefill_advance_group({&ca, &cb}, {128, 128});
    (void)reference.session_prefill_advance(ra);
    (void)reference.session_prefill_advance(rb);
    require(ca.next == ra.next && cb.next == rb.next, "bounded fallback preserves cursor progress");
  }
  close(ca.output, ra.output, "bounded group fallback first reader");
  close(cb.output, rb.output, "bounded group fallback second reader");
  close(model.session_draft(1, {7}), reference.session_draft(1, {7}), "bounded fallback draft");
  for (int req = 0; req < 2; ++req) { model.session_close(req); reference.session_close(req); }
  require(model.kv_blocks_in_use() == 0 && reference.kv_blocks_in_use() == 0, "bounded fallback releases reservations");
}
int main() { return test::run_all(); }
