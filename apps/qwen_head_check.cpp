// Matched teacher-forced Qwen head scoring. The manifest contains named texts
// (or explicit token ids for fixtures). Each mode uses the production model
// dispatch, captured verification batches, and cold short-prefill forwards.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "engine/graph_engine.hpp"
#include "engine/tp_bus.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/forward.hpp"
#include "text/tokenizer.hpp"

namespace {
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

template <class T>
uint64_t digest(const T* values, size_t count) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < count; ++i) {
    if constexpr (std::is_same_v<T, float>) h ^= std::bit_cast<uint32_t>(values[i]);
    else h ^= static_cast<uint64_t>(values[i]);
    h *= 1099511628211ull;
  }
  return h;
}

struct Corpus {
  std::string name;
  std::vector<int64_t> ids;
};

// Local log-sum-exp and two best logits suffice to merge the vocabulary
// shards. Exact logit/hidden hashes additionally gate unchanged-mode repeats.
void score(const dgpp::QwenModel::Outputs& out, int row, int hidden, int64_t target,
           int rank, int repeat, const std::string& name, const char* lane, int width,
           int index) {
  const int n = out.lm_vocab_count, begin = out.lm_vocab_begin;
  const float* logits = out.logits.data() + static_cast<size_t>(row) * n;
  int best = -1, second = -1;
  for (int col = 0; col < n; ++col) {
    require(std::isfinite(logits[col]), "nonfinite vocabulary logit");
    if (best < 0 || logits[col] > logits[best]) { second = best; best = col; }
    else if (second < 0 || logits[col] > logits[second]) second = col;
  }
  require(second >= 0, "vocabulary shard needs at least two tokens");
  double sum = 0;
  for (int col = 0; col < n; ++col) sum += std::exp(double(logits[col]) - logits[best]);
  const double lse = logits[best] + std::log(sum);
  const int64_t local = target - begin;
  char target_logit[64];
  if (local >= 0 && local < n) std::snprintf(target_logit, sizeof(target_logit), "%.9g", logits[local]);
  else std::snprintf(target_logit, sizeof(target_logit), "null");
  const auto* h = out.final_hidden_bits.data() + static_cast<size_t>(row) * hidden;
  std::printf("[head_score] {\"rank\":%d,\"repeat\":%d,\"corpus\":\"%s\",\"lane\":\"%s\","
              "\"width\":%d,\"index\":%d,\"target\":%lld,\"lse\":%.17g,\"target_logit\":%s,"
              "\"top\":[[%d,%.9g],[%d,%.9g]],\"hidden\":\"%016llx\",\"logits\":\"%016llx\"}\n",
              rank, repeat, name.c_str(), lane, width, index, static_cast<long long>(target), lse,
              target_logit, best + begin, logits[best], second + begin, logits[second],
              static_cast<unsigned long long>(digest(h, static_cast<size_t>(hidden))),
              static_cast<unsigned long long>(digest(logits, static_cast<size_t>(n))));
}

struct Graph {
  cudaGraphExec_t exec = nullptr;
  dgpp::PickVerdict* verdicts = nullptr;
  int32_t* request_map = nullptr;
  ~Graph() {
    if (exec) cudaGraphExecDestroy(exec);
    if (verdicts) cudaFree(verdicts);
    if (request_map) cudaFreeHost(request_map);
  }
};

void capture(Graph& graph, dgpp::QwenModel& model, dgpp::net::CollectiveBus* bus,
             dgpp::GraphRecordReducer* recorder, int variant, int requests, int rows,
             const std::vector<int32_t>& mapping = {}, const std::vector<int>& active = {}) {
  std::vector<dgpp::PickVerdict> verdicts(static_cast<size_t>(requests));
  for (int q = 0; q < requests; ++q) {
    verdicts[q].rows = rows;
    verdicts[q].accepted =
        active.empty() || std::find(active.begin(), active.end(), q) != active.end() ? rows : 0;
  }
  if (!mapping.empty()) {
    require(mapping.size() == static_cast<size_t>(requests), "map shape");
    DGPP_CUDA_OK(
        cudaHostAlloc(&graph.request_map, requests * sizeof(int32_t), cudaHostAllocDefault));
    std::copy(mapping.begin(), mapping.end(), graph.request_map);
  }
  model.session_graph_batch_map_source(graph.request_map);
  DGPP_CUDA_OK(cudaMalloc(&graph.verdicts, verdicts.size() * sizeof(verdicts[0])));
  DGPP_CUDA_OK(cudaMemcpy(graph.verdicts, verdicts.data(), verdicts.size() * sizeof(verdicts[0]),
                          cudaMemcpyHostToDevice));
  auto* eager = bus ? model.set_boundary(recorder) : nullptr;
  std::string error;
  if (bus) require(bus->graph_record_begin(&error, variant), error);
  cudaGraph_t recorded = nullptr;
  DGPP_CUDA_OK(cudaStreamBeginCapture(model.stream(), cudaStreamCaptureModeThreadLocal));
  model.session_graph_capture_batch(rows, requests);
  model.session_graph_capture_commit_batch(graph.verdicts);
  DGPP_CUDA_OK(cudaStreamEndCapture(model.stream(), &recorded));
  if (bus) {
    require(bus->graph_record_end(&error), error);
    model.set_boundary(eager);
  }
  size_t count = 0;
  DGPP_CUDA_OK(cudaGraphGetNodes(recorded, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count);
  DGPP_CUDA_OK(cudaGraphGetNodes(recorded, nodes.data(), &count));
  int heads = 0;
  for (auto node : nodes) {
    cudaGraphNodeType type;
    DGPP_CUDA_OK(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeParams params{};
    const auto status = cudaGraphKernelNodeGetParams(node, &params);
    if (status == cudaErrorInvalidDeviceFunction) { (void)cudaGetLastError(); continue; }
    DGPP_CUDA_OK(status);
    const char* name = nullptr;
    DGPP_CUDA_OK(cudaFuncGetName(&name, params.func));
    if (std::string(name).find("mma_gemv_kernel") != std::string::npos &&
        *static_cast<void**>(params.kernelParams[4]) == model.device_logits()) ++heads;
  }
  const bool mma = model.fp8_head_mma() && requests * rows > dgpp::dense_gemv_rows();
  require(heads == int(mma), "captured vocabulary head does not match requested dispatch");
  DGPP_CUDA_OK(cudaGraphInstantiate(&graph.exec, recorded, nullptr, nullptr, 0));
  DGPP_CUDA_OK(cudaGraphDestroy(recorded));
  model.session_graph_batch_map_source(nullptr);
}

// Keep the teacher tokens and physical slots identical while changing the
// recorded family. Full-depth cases score each entire corpus; the T=1/T=2
// boundaries use its configured prefix. Dense cases are exact controls.
void compaction_scores(dgpp::QwenModel& model, dgpp::net::CollectiveBus* bus,
                       dgpp::GraphRecordReducer* recorder, const std::vector<Corpus>& corpora,
                       const std::string& contents, bool compact, int rank, int world, int repeats,
                       int boundary_tokens, int hidden, int vocab) {
  struct Shape {
    const char* lane;
    int rows;
    std::vector<int> physical;
  };
  const std::vector<Shape> shapes{
      {"dense4", 2, {0, 1, 2, 3}},
      {"sparse2", 1, {15, 0}},
      {"sparse2", 2, {15, 0}},
      {"sparse2", 4, {15, 0}},
      {"padded5", 4, {15, 0, 7, 3, 12}},
      {"dense16", 4, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
  std::vector<std::unique_ptr<Graph>> graphs(shapes.size());
  std::printf(
      "[head_begin] {\"rank\":%d,\"world\":%d,\"mode\":\"gemv\",\"repeats\":%d,"
      "\"capacity\":64,\"boundary_tokens\":%d,\"prefill_trials\":0,\"yarn\":0,\"corpora\":%zu,"
      "\"compaction\":\"%s\",\"manifest\":\"%016llx\",\"vocab_begin\":%d,"
      "\"vocab_count\":%d,\"vocab\":%d}\n",
      rank, world, repeats, boundary_tokens, corpora.size(), compact ? "on" : "off",
      static_cast<unsigned long long>(digest(contents.data(), contents.size())),
      model.lm_vocab_begin(), model.lm_vocab_count(), vocab);
  size_t total = 0;
  for (int repeat = 0; repeat < repeats; ++repeat)
    for (const auto& c : corpora) {
      for (size_t si = 0; si < shapes.size(); ++si) {
        const auto& shape = shapes[si];
        const int live = static_cast<int>(shape.physical.size()), rows = shape.rows;
        const int width = live * rows;
        const int physical = *std::max_element(shape.physical.begin(), shape.physical.end()) + 1;
        int requests = physical;
        if (compact) {
          for (int bucket : {2, 3, 4, 6, 8, 12, 16}) {
            if (bucket >= live &&
                dgpp::QwenModel::compact_batch_compatible(physical * rows, bucket * rows)) {
              requests = bucket;
              break;
            }
          }
        }
        const size_t n = rows == 4 ? c.ids.size() : std::min(c.ids.size(), size_t(boundary_tokens));
        const int length = static_cast<int>(n / live), steps = (length - 17) / rows;
        require(steps > 0, "compaction corpus is too short for its physical layout");
        for (int q = 0; q < live; ++q) {
          const auto begin = c.ids.begin() + q * length;
          (void)model.session_prefill(shape.physical[q], std::vector<int64_t>(begin, begin + 16));
          model.session_reserve_blocks(shape.physical[q], length + 64);
        }
        for (int step = 0; step < steps; ++step) {
          for (int q = 0; q < live; ++q) {
            const auto begin = c.ids.begin() + q * length + 16 + step * rows;
            model.session_graph_seed_feed(shape.physical[q],
                                          std::vector<int64_t>(begin, begin + rows));
          }
          if (!graphs[si]) {
            graphs[si] = std::make_unique<Graph>();
            std::vector<int32_t> mapping;
            std::vector<int> active = shape.physical;
            if (compact) {
              mapping.assign(requests, -1);
              std::copy(shape.physical.begin(), shape.physical.end(), mapping.begin());
              for (int q = 0; q < live; ++q) active[q] = q;
            }
            capture(*graphs[si], model, bus, recorder, static_cast<int>(si), requests, rows,
                    mapping, active);
          }
          model.session_graph_use_batch_contract(rows, requests);
          model.session_graph_stage_batch();
          std::string error;
          if (bus) require(bus->graph_replay_arm(&error, static_cast<int>(si)), error);
          DGPP_CUDA_OK(cudaGraphLaunch(graphs[si]->exec, model.stream()));
          DGPP_CUDA_OK(cudaStreamSynchronize(model.stream()));
          if (bus) require(bus->graph_replay_finish(120000, &error), error);
          const auto out = model.session_graph_outputs(shape.physical[0]);
          for (int q = 0; q < live; ++q) {
            for (int row = 0; row < rows; ++row) {
              const int pos = q * length + 16 + step * rows + row + 1;
              score(out, (compact ? q : shape.physical[q]) * rows + row, hidden, c.ids[pos], rank,
                    repeat, c.name, shape.lane, width, step * width + q * rows + row);
              ++total;
            }
            model.session_graph_settle(shape.physical[q], rows);
          }
        }
        for (int req : shape.physical) model.session_close(req);
        std::printf(
            "[head_case] {\"rank\":%d,\"repeat\":%d,\"corpus\":\"%s\",\"lane\":\"%s\","
            "\"width\":%d,\"count\":%d,\"tokens\":%zu,\"token_hash\":\"%016llx\"}\n",
            rank, repeat, c.name.c_str(), shape.lane, width, steps * width, c.ids.size(),
            static_cast<unsigned long long>(digest(c.ids.data(), c.ids.size())));
        std::fflush(stdout);
      }
    }
  std::printf("[head_end] {\"rank\":%d,\"count\":%zu}\n", rank, total);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string checkpoint, model_id, manifest, peer, image_dir, mode = "gemv", compaction;
    int rank = 0, world = 1, port = 29950, repeats = 2, boundary_tokens = 1024;
    int prefill_trials = 32, capacity = 16;
    bool yarn = false;
    auto next = [&](int& i) {
      require(i + 1 < argc, "missing argument value");
      return std::string(argv[++i]);
    };
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--checkpoint-dir") checkpoint = next(i);
      else if (a == "--model") model_id = next(i);
      else if (a == "--requests") manifest = next(i);
      else if (a == "--fp8-head") mode = next(i);
      else if (a == "--world") world = std::stoi(next(i));
      else if (a == "--rank") rank = std::stoi(next(i));
      else if (a == "--port") port = std::stoi(next(i));
      else if (a == "--peer") peer = next(i);
      else if (a == "--image-dir") image_dir = next(i);
      else if (a == "--repeats") repeats = std::stoi(next(i));
      else if (a == "--boundary-tokens") boundary_tokens = std::stoi(next(i));
      else if (a == "--prefill-trials") prefill_trials = std::stoi(next(i));
      else if (a == "--decode-capacity") capacity = std::stoi(next(i));
      else if (a == "--compaction")
        compaction = next(i);
      else if (a == "--yarn") yarn = true;
      else throw std::runtime_error("unknown argument: " + a);
    }
    require(mode == "gemv" || mode == "mma", "--fp8-head must be gemv or mma");
    require(repeats >= 2 && boundary_tokens >= 64 && prefill_trials > 0,
            "need >=2 repeats, >=64 boundary tokens and positive prefill trials");
    require(capacity == 8 || capacity == 16, "--decode-capacity must be 8 or 16");
    require(compaction.empty() || compaction == "off" || compaction == "on",
            "--compaction must be off or on");
    require(compaction.empty() || (mode == "gemv" && !yarn),
            "compaction comparison holds the default head and rope settings fixed");
    if (!compaction.empty()) capacity = 64;
    require(world >= 1 && rank >= 0 && rank < world, "invalid rank/world");
    require(dgpp::dense_gemv_rows() == 4, "this gate requires DGPP_DENSE_GEMV_ROWS=4");
    if (checkpoint.empty()) {
      std::string error;
      checkpoint = dgpp::hf::model_dir(model_id, &error);
      require(!checkpoint.empty(), "cannot resolve checkpoint: " + error);
    }
    std::ifstream input(manifest, std::ios::binary);
    require(bool(input), "cannot open --requests manifest");
    const std::string contents((std::istreambuf_iterator<char>(input)), {});
    const auto document = dgpp::minijson::parse(contents);
    require(document.root.is_array(), "manifest must be a JSON array");
    std::unique_ptr<dgpp::text::Tokenizer> tokenizer;
    std::vector<Corpus> corpora;
    std::set<std::string> names;
    for (const auto& entry : document.root.items()) {
      const auto* name = entry.find("name");
      require(name && name->is_string(), "corpus needs a name");
      Corpus c{std::string(name->as_string()), {}};
      require(!c.name.empty() && c.name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") ==
                                    std::string::npos && names.insert(c.name).second,
              "corpus names must be unique lowercase identifiers");
      if (const auto* ids = entry.find("ids")) {
        require(ids->is_array(), "ids must be an array");
        for (const auto& id : ids->items()) {
          require(id.kind() == dgpp::minijson::Value::Kind::Int, "token id must be an integer");
          c.ids.push_back(id.as_int());
        }
      } else {
        const auto* text = entry.find("text");
        require(text && text->is_string(), "corpus needs text or ids");
        if (!tokenizer) tokenizer = std::make_unique<dgpp::text::Tokenizer>(
            dgpp::text::Tokenizer::load((std::filesystem::path(checkpoint) / "tokenizer.json").string()));
        c.ids = tokenizer->encode(text->as_string());
      }
      require(c.ids.size() >= 128, "each corpus needs at least 128 tokens");
      corpora.push_back(std::move(c));
    }
    require(!corpora.empty(), "empty corpus manifest");
    auto cfg = dgpp::QwenTextConfig::from_json_file((std::filesystem::path(checkpoint) / "config.json").string());
    if (yarn) cfg.rope_scaling = dgpp::RopeScaling{2.0, 262144};
    size_t max_tokens = 0;
    for (const auto& c : corpora) {
      max_tokens = std::max(max_tokens, c.ids.size());
      for (auto id : c.ids) require(id >= 0 && id < cfg.vocab_size, "token outside vocabulary");
    }
    dgpp::QwenLayerStream::set_dense_weights_fp8(true);
    dgpp::QwenLayerStream::set_ngram_table_mmap(true);
    dgpp::QwenLayerStream::set_resident_image_dir(image_dir);
    std::unique_ptr<dgpp::net::CollectiveBus> bus;
    std::unique_ptr<dgpp::BusBoundaryReducer> reducer;
    if (world > 1) {
      require(rank == 0 || !peer.empty(), "peer required for nonzero rank");
      bus = std::make_unique<dgpp::net::CollectiveBus>(dgpp::fabric_bus_options(
          rank, world, static_cast<uint16_t>(port), peer, 120000,
          static_cast<size_t>(64) * cfg.hyper_width() * 2 + 4096));
      std::string error;
      require(bus->start(&error), error);
      reducer = std::make_unique<dgpp::BusBoundaryReducer>(*bus, 120000);
    }
    const int request_slots = compaction.empty() ? 4 : 16;
    // Each request reserves 64 extra tokens and rounds up to a complete KV
    // block. Account for that slack per request, including the dense C16 case.
    const int64_t cache_slack = std::max<int64_t>(
        1024, request_slots * (64 + dgpp::QwenModel::kv_block_tokens_static() - 1));
    dgpp::QwenModel model(cfg, checkpoint, 64, static_cast<int64_t>(max_tokens) + cache_slack,
                          dgpp::QwenResidency::Resident, reducer.get(), rank, world, request_slots,
                          false, capacity, mode == "mma");
    model.set_decode_route_traces(false);
    model.set_decode_tail_mirrors(false);
    model.session_graph_prepare();
    std::unique_ptr<dgpp::GraphRecordReducer> recorder;
    if (bus) recorder = std::make_unique<dgpp::GraphRecordReducer>(*bus, model.stream());
    if (!compaction.empty()) {
      compaction_scores(model, bus.get(), recorder.get(), corpora, contents, compaction == "on",
                        rank, world, repeats, boundary_tokens, cfg.hidden_size, cfg.vocab_size);
      return 0;
    }
    const std::vector<std::pair<int, int>> shapes{{2, 2}, {3, 2}, {4, 2}, {4, 3}, {4, 4}};
    std::vector<std::unique_ptr<Graph>> graphs(shapes.size());
    std::printf("[head_begin] {\"rank\":%d,\"world\":%d,\"mode\":\"%s\",\"repeats\":%d,"
                "\"capacity\":%d,\"boundary_tokens\":%d,\"prefill_trials\":%d,\"yarn\":%d,\"corpora\":%zu,"
                "\"manifest\":\"%016llx\",\"vocab_begin\":%d,\"vocab_count\":%d,\"vocab\":%d}\n",
                rank, world, mode.c_str(), repeats, capacity, boundary_tokens, prefill_trials, int(yarn), corpora.size(),
                static_cast<unsigned long long>(digest(contents.data(), contents.size())),
                model.lm_vocab_begin(), model.lm_vocab_count(), cfg.vocab_size);
    size_t total = 0;
    for (int repeat = 0; repeat < repeats; ++repeat) for (const auto& c : corpora) {
      for (size_t shape = 0; shape < shapes.size(); ++shape) {
        const auto [requests, rows] = shapes[shape];
        const int width = requests * rows;
        if (width > capacity) continue;
        const size_t n = width == capacity ? c.ids.size() : std::min(c.ids.size(), size_t(boundary_tokens));
        const int length = static_cast<int>(n / requests);
        const int steps = (length - 17) / rows;
        require(steps > 0, "empty verification case");
        for (int req = 0; req < requests; ++req) {
          const auto begin = c.ids.begin() + req * length;
          (void)model.session_prefill(req, std::vector<int64_t>(begin, begin + 16));
          model.session_reserve_blocks(req, length + 64);
        }
        for (int step = 0; step < steps; ++step) {
          for (int req = 0; req < requests; ++req) {
            const auto begin = c.ids.begin() + req * length + 16 + step * rows;
            model.session_graph_seed_feed(req, std::vector<int64_t>(begin, begin + rows));
          }
          if (!graphs[shape]) {
            graphs[shape] = std::make_unique<Graph>();
            capture(*graphs[shape], model, bus.get(), recorder.get(), static_cast<int>(shape), requests, rows);
          }
          model.session_graph_use_batch_contract(rows, requests);
          model.session_graph_stage_batch();
          std::string error;
          if (bus) require(bus->graph_replay_arm(&error, static_cast<int>(shape)), error);
          DGPP_CUDA_OK(cudaGraphLaunch(graphs[shape]->exec, model.stream()));
          DGPP_CUDA_OK(cudaStreamSynchronize(model.stream()));
          if (bus) require(bus->graph_replay_finish(120000, &error), error);
          const auto out = model.session_graph_outputs(0);
          for (int req = 0; req < requests; ++req) {
            for (int row = 0; row < rows; ++row) {
              const int pos = req * length + 16 + step * rows + row + 1;
              score(out, req * rows + row, cfg.hidden_size, c.ids[pos], rank, repeat,
                    c.name, "verify", width, step * width + req * rows + row);
              ++total;
            }
            model.session_graph_settle(req, rows);
          }
        }
        for (int req = 0; req < requests; ++req) model.session_close(req);
        std::printf("[head_case] {\"rank\":%d,\"repeat\":%d,\"corpus\":\"%s\",\"lane\":\"verify\","
                    "\"width\":%d,\"count\":%d,\"tokens\":%zu,\"token_hash\":\"%016llx\"}\n",
                    rank, repeat, c.name.c_str(), width, steps * width, c.ids.size(),
                    static_cast<unsigned long long>(digest(c.ids.data(), c.ids.size())));
        std::fflush(stdout);
      }
      for (int width : {1, 4, 5, 8, 9, 16, 17}) {
        for (int trial = 0; trial < prefill_trials; ++trial) {
          const size_t start = static_cast<size_t>(trial) * (c.ids.size() - width - 1) / prefill_trials;
          const std::vector<int64_t> prompt(c.ids.begin() + start, c.ids.begin() + start + width);
          const auto out = model.forward(prompt);
          for (int row = 0; row < width; ++row) {
            score(out, row, cfg.hidden_size, c.ids[start + row + 1], rank, repeat,
                  c.name, "prefill", width, trial * width + row);
            ++total;
          }
        }
        std::printf("[head_case] {\"rank\":%d,\"repeat\":%d,\"corpus\":\"%s\",\"lane\":\"prefill\","
                    "\"width\":%d,\"count\":%d,\"tokens\":%zu,\"token_hash\":\"%016llx\"}\n",
                    rank, repeat, c.name.c_str(), width, prefill_trials * width, c.ids.size(),
                    static_cast<unsigned long long>(digest(c.ids.data(), c.ids.size())));
        std::fflush(stdout);
      }
    }
    std::printf("[head_end] {\"rank\":%d,\"count\":%zu}\n", rank, total);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qwen_head_check: %s\n", e.what());
    return 1;
  }
}
