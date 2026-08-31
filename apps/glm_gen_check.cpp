// M6 Stage 1d: glm_gen_check — greedy generation end-to-end at real dims.
// The "it speaks" instrument: prompt token-ids in, generated token-ids out,
// through the exact serving path (resident TP forward + vocab-sharded head
// + the distributed greedy pick over the bus).
//
// MODES
//   world=1 (no --world): the local reference — full head, streaming
//     residency, plain argmax. No bus, no peers; the loop logic and the
//     w1 sequence land in the record before the fabric run.
//   world>1: the fabric TP shape — every rank a process, VocabSharded
//     head, Resident residency by default (the M6 serving contract;
//     --streaming keeps the M4 diagnostic loader), the per-step winner
//     through bus_greedy_pick (the same exact pick the CI gate pins:
//     glm_tp_greedy_gen_loopback). Every rank writes its tokens file;
//     the run record cross-checks them (md5) — the pick's rank
//     consistency is by construction (canonical order + broadcast),
//     the files are the audit trail.
//
// PROMPT: either raw text (--text "...", tokenized by the Stage 3 exact
// tokenizer, whose ids match HF tokenizers 0.23.1 byte-for-byte) or
// comma-separated token ids (--prompt "1,2,3"). Generated tokens decode
// through the same tokenizer; EOS stops the loop (config's eos_token_ids,
// --no-eos disables).
//
// T² IS DIAGNOSTIC HERE: every step re-forwards the whole sequence
// (fresh KDA/DSA state per call — GlmDiagnosticModel's contract). The
// incremental decode engine is Stage 2; do not read this app's per-step
// time as serving latency. EOS-stop is likewise Stage 2 policy (the
// sampler/service seam); this loop runs exactly --steps steps.
//
// ALLOCATION DISCIPLINE (the burst-wedge lesson, now a rule): every
// device allocation — the pick scratch included — happens BEFORE the
// world forms. Nothing allocates between collectives; the per-step
// host buffers are hoisted out of the loop.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "loaders/hf_cache.hpp"
#include "models/glm_forward.hpp"
#include "models/glm_sampler.hpp"
#include "models/glm_tokenizer.hpp"
#include "models/glm_tp_bus.hpp"
#include "net/collective_bus.hpp"

namespace fs = std::filesystem;
using dgpp::GlmDiagnosticModel;
using dgpp::GlmTextConfig;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Real-mesh budgets, not loopback budgets (found by the first fabric gate
// run, 2026-08-30: a cold peer's first boundary waits behind seconds of
// cold NVMe weight streaming). The lane watchdog arms from each POST,
// so these bounds measure genuine in-flight stalls only.
BusOptions bus_options(int rank, int world, uint16_t port,
                       const std::string& peer, int rendezvous_timeout_ms) {
  BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.lane_devices = {"rocep1s0f0", "roceP2p1s0f0"};
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : peer;
  o.rendezvous_timeout_ms = rendezvous_timeout_ms;
  o.lat_slots = 8;
  o.lat_slot_bytes = 8192;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = 120000;
  o.consumer_deadline_s = 60.0;
  o.launch_consumers = false;
  return o;
}

std::vector<int64_t> parse_prompt_ids(const std::string& text,
                                      int64_t vocab_size) {
  std::vector<int64_t> ids;
  std::string num;
  const auto flush = [&] {
    require(!num.empty(), "empty token id in --prompt");
    char* end = nullptr;
    const long long v = std::strtoll(num.c_str(), &end, 10);
    require(end && *end == '\0', "bad token id in --prompt: " + num);
    require(v >= 0 && v < vocab_size,
            "token id out of range [0, vocab): " + num);
    ids.push_back(static_cast<int64_t>(v));
    num.clear();
  };
  for (const char c : text) {
    if (c == ',' || c == ' ') flush();
    else num.push_back(c);
  }
  flush();
  require(!ids.empty(), "--prompt produced no token ids");
  return ids;
}

void write_tokens_file(const std::string& path, int rank, int world,
                       const std::vector<int64_t>& prompt,
                       const std::vector<int64_t>& generated) {
  std::string txt = "rank " + std::to_string(rank) + " world " +
                    std::to_string(world) + "\nprompt";
  for (int64_t t : prompt) txt += " " + std::to_string(t);
  txt += "\ngenerated";
  for (int64_t t : generated) txt += " " + std::to_string(t);
  txt += "\n";
  std::FILE* f = std::fopen(path.c_str(), "wb");
  require(f != nullptr, "cannot write " + path);
  const size_t n = std::fwrite(txt.data(), 1, txt.size(), f);
  std::fclose(f);
  require(n == txt.size(), "short write to " + path);
}

std::string ids_line(const std::vector<int64_t>& ids) {
  std::string s;
  for (int64_t t : ids) {
    if (!s.empty()) s += ",";
    s += std::to_string(t);
  }
  return s;
}

int run(const GlmTextConfig& cfg, const std::string& ckpt, int world,
        int rank, uint16_t port, const std::string& peer,
        const std::vector<int64_t>& prompt, int steps, bool resident,
        bool incremental, bool no_eos, const dgpp::GlmTokenizer& tok,
        const std::string& out_prefix, int rendezvous_timeout_ms) {
  const int max_tokens = static_cast<int>(prompt.size()) + steps + 1;
  const int64_t cache = std::max<int64_t>(128, max_tokens);

  // ---- world 1: no bus, full head, plain argmax ----------------------
  if (world == 1) {
    const auto t_construct = std::chrono::steady_clock::now();
    GlmDiagnosticModel model(cfg, ckpt, max_tokens, cache);
    const double construct_s = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() -
                                   t_construct)
                                   .count();
    DGPP_LOG_INFO("w1 model constructed in {:.1f}s (streaming)", construct_s);
    std::vector<int64_t> generated;
    std::string generated_text;  // decoded via the exact tokenizer
    const auto is_eos = [&](int64_t id) {
      return !no_eos && std::find(cfg.eos_token_ids.begin(),
                                 cfg.eos_token_ids.end(), id) !=
                           cfg.eos_token_ids.end();
    };
    std::vector<float> frow(static_cast<size_t>(cfg.vocab_size));
    if (incremental) {
      // The serving path (Stage 2): prefill once, then one stateful
      // step per token — constant work per step, no T^2 re-forward.
      const auto t0 = std::chrono::steady_clock::now();
      const GlmDiagnosticModel::Outputs out = model.session_prefill(prompt);
      const double prefill_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count();
      for (int i = 0; i < out.lm_vocab_count; ++i)
        frow[static_cast<size_t>(i)] =
            dgpp::bf16_bits_to_float(out.logits_bits[static_cast<size_t>(i)]);
      dgpp::glm_sample::Candidate best =
          dgpp::glm_sample::local_max(frow.data(), cfg.vocab_size, 0);
      DGPP_LOG_INFO("[gen] prefill: {} tokens in {:.0f}ms", prompt.size(),
                    prefill_ms);
      for (int s = 0; s < steps; ++s) {
        generated.push_back(best.id);
        generated_text += tok.decode(best.id, /*skip_special_tokens=*/false);
        if (is_eos(best.id)) {
          DGPP_LOG_INFO("[gen] eos stop at step {} (token {})", s, best.id);
          break;
        }
        if (s + 1 == steps) break;
        const auto t0s = std::chrono::steady_clock::now();
        const GlmDiagnosticModel::Outputs step =
            model.session_step(generated.back());
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0s)
                              .count();
        for (int i = 0; i < step.lm_vocab_count; ++i)
          frow[static_cast<size_t>(i)] =
              dgpp::bf16_bits_to_float(step.logits_bits[static_cast<size_t>(i)]);
        best = dgpp::glm_sample::local_max(frow.data(), cfg.vocab_size, 0);
        DGPP_LOG_INFO("[gen] step {}: token {} (logit {:.4f}) [{:.0f}ms]",
                      s + 1, best.id, best.logit, ms);
      }
    } else {
      // The re-forward reference (the T^2 diagnostic loop).
      std::vector<int64_t> toks = prompt;
      for (int s = 0; s < steps; ++s) {
        const auto t0 = std::chrono::steady_clock::now();
        const GlmDiagnosticModel::Outputs out = model.forward(toks);
        const int T = static_cast<int>(toks.size());
        require(out.lm_vocab_count == cfg.vocab_size,
                "w1 head must be full-vocab");
        const uint16_t* row = out.logits_bits.data() +
                              static_cast<size_t>(T - 1) * cfg.vocab_size;
        for (int i = 0; i < cfg.vocab_size; ++i)
          frow[static_cast<size_t>(i)] = dgpp::bf16_bits_to_float(row[i]);
        const dgpp::glm_sample::Candidate best =
            dgpp::glm_sample::local_max(frow.data(), cfg.vocab_size, 0);
        toks.push_back(best.id);
        generated.push_back(best.id);
        generated_text += tok.decode(best.id, /*skip_special_tokens=*/false);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        DGPP_LOG_INFO("[gen] step {}: token {} (logit {:.4f}) [{:.0f}ms]",
                      s, best.id, best.logit, ms);
      }
    }
    write_tokens_file(out_prefix + ".tokens.txt", rank, world, prompt,
                      generated);
    DGPP_LOG_INFO("w1 generated ids: {}", ids_line(generated));
    DGPP_LOG_INFO("w1 generated text: {}", generated_text);
    return 0;
  }

  // ---- fabric TP: bus, sharded head, resident by default --------------
  // The pick scratch (bus_greedy_pick's caller-owned gather table) is
  // allocated BEFORE the world forms — the discipline at the top of this
  // file. Model construction happens after bus.start like glm_tp_check:
  // separate processes put a spinning peer kernel on the PEER's device;
  // it cannot block this rank's construction-phase device syncs.
  uint16_t* pick_scratch = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&pick_scratch,
                                 sizeof(uint16_t) * 4 * world));
  std::unique_ptr<CollectiveBus> bus;
  try {
    bus = std::make_unique<CollectiveBus>(
        bus_options(rank, world, port, peer, rendezvous_timeout_ms));
    std::string err;
    if (!bus->start(&err))
      throw std::runtime_error("rank " + std::to_string(rank) +
                               " bus start: " + err);

    dgpp::GlmBusBoundaryReducer reducer(*bus);
    const auto t_construct = std::chrono::steady_clock::now();
    GlmDiagnosticModel model(
        cfg, ckpt, max_tokens, cache, &reducer, rank, world,
        resident ? dgpp::GlmResidency::Resident
                 : dgpp::GlmResidency::Streaming,
        dgpp::GlmHeadSharding::VocabSharded);
    const double construct_s = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() -
                                   t_construct)
                                   .count();
    DGPP_LOG_INFO("rank {} model constructed in {:.1f}s ({})", rank,
                  construct_s,
                  resident ? "resident" : "streaming");

    std::vector<int64_t> toks = prompt;
    std::vector<int64_t> generated;
    std::vector<float> fslice;  // hoisted: no per-step device-adjacent work
    double forward_ms_total = 0.0;
    const auto is_eos = [&](int64_t id) {
      return !no_eos && std::find(cfg.eos_token_ids.begin(),
                                  cfg.eos_token_ids.end(), id) !=
                            cfg.eos_token_ids.end();
    };
    // The step body: run one row (forward or stateful step), pick the
    // winner through the bus, record it. Returns the picked token.
    const auto run_step = [&](const GlmDiagnosticModel::Outputs& out,
                              const char* what, int s) -> int32_t {
      const uint16_t* row = out.logits_bits.data();
      fslice.resize(static_cast<size_t>(out.lm_vocab_count));
      for (int i = 0; i < out.lm_vocab_count; ++i)
        fslice[static_cast<size_t>(i)] =
            dgpp::bf16_bits_to_float(row[static_cast<size_t>(i)]);
      const dgpp::glm_sample::Candidate local = dgpp::glm_sample::local_max(
          fslice.data(), out.lm_vocab_count, out.lm_vocab_begin);
      const int32_t token =
          dgpp::bus_greedy_pick(*bus, rank, world, local, pick_scratch, 60000);
      require(token >= 0 && token < cfg.vocab_size,
              "generated id out of range: " + std::to_string(token));
      DGPP_LOG_INFO(
          "[gen] rank {} {} {}: token {} (local slice [{},{}) best "
          "{} logit {:.4f})",
          rank, what, s, token, out.lm_vocab_begin,
          out.lm_vocab_begin + out.lm_vocab_count, local.id, local.logit);
      return token;
    };
    std::string generated_text;
    if (incremental) {
      // The serving path (Stage 2): prefill once, one stateful step per
      // token, distributed pick over the sharded head's slices.
      const auto t0 = std::chrono::steady_clock::now();
      const GlmDiagnosticModel::Outputs pre = model.session_prefill(prompt);
      const double prefill_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count();
      DGPP_LOG_INFO("rank {} prefill: {} tokens in {:.0f}ms", rank,
                    prompt.size(), prefill_ms);
      const auto tp0 = std::chrono::steady_clock::now();
      int32_t token = run_step(pre, "prefill+pick", 0);
      forward_ms_total =
          prefill_ms + std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - tp0)
                           .count();
      for (int s = 0; s < steps; ++s) {
        generated.push_back(token);
        toks.push_back(token);
        generated_text += tok.decode(token, /*skip_special_tokens=*/false);
        if (is_eos(token)) {
          DGPP_LOG_INFO("rank {} eos stop at step {} (token {})", rank, s,
                        token);
          break;
        }
        if (s + 1 == steps) break;
        const auto t0s = std::chrono::steady_clock::now();
        const GlmDiagnosticModel::Outputs step =
            model.session_step(generated.back());
        token = run_step(step, "step", s + 1);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0s)
                              .count();
        forward_ms_total += ms;
        DGPP_LOG_INFO("rank {} step {:.0f}ms (stateful step + pick)", rank,
                      ms);
      }
    } else {
      // The re-forward reference (the T^2 loop).
      for (int s = 0; s < steps; ++s) {
        const auto t0 = std::chrono::steady_clock::now();
        const GlmDiagnosticModel::Outputs out = model.forward(toks);
        const int32_t token = run_step(out, "step", s);
        generated_text += tok.decode(token, /*skip_special_tokens=*/false);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        forward_ms_total += ms;
        toks.push_back(token);
        generated.push_back(token);
      }
    }

    write_tokens_file(out_prefix + ".tokens.txt", rank, world, prompt,
                      generated);
    DGPP_LOG_INFO("rank {} generated ids: {}", rank, ids_line(generated));
    DGPP_LOG_INFO("rank {} generated text: {}", rank, generated_text);

    const auto stats = bus->stats();
    const auto summarize = [](const std::vector<double>& us) {
      std::vector<double> sorted = us;
      std::sort(sorted.begin(), sorted.end());
      const auto pct = [&](double f) {
        return sorted.empty()
                   ? 0.0
                   : sorted[std::min(sorted.size() - 1,
                                     static_cast<size_t>(f * sorted.size()))];
      };
      return std::make_pair(sorted.size(),
                            std::make_pair(pct(0.5), pct(0.99)));
    };
    const auto [lat_n, lat_tails] = summarize(stats.latency.latency_us);
    const auto [bulk_n, bulk_tails] = summarize(stats.bulk.latency_us);
    DGPP_LOG_INFO(
        "rank {}: {} steps in {:.1f}ms total ({:.0f}ms/step avg incl. "
        "prefill; {}), lat {} "
        "(p50={:.1f}us p99={:.1f}us), bulk {} (p50={:.1f}us p99={:.1f}us)",
        rank, steps, forward_ms_total, forward_ms_total / steps,
        incremental ? "incremental engine — constant per-step, the serving "
                      "path (steady-state from the per-step logs)"
                    : "re-forward T^2 diagnostic",
        lat_n, lat_tails.first, lat_tails.second, bulk_n, bulk_tails.first,
        bulk_tails.second);
    bus->stop();
  } catch (...) {
    if (pick_scratch) cudaFree(pick_scratch);
    throw;
  }
  cudaFree(pick_scratch);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");

  std::string config_path, ckpt, peer, prompt_text, text_prompt, out_prefix =
                                                   "glm_gen",
              model_id;
  int world = 1, rank = 0, steps = 8, rendezvous_timeout_ms = 120000;
  uint16_t port = 29970;
  bool resident = true, incremental = true, no_eos = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      require(i + 1 < argc, "missing value for " + a);
      return argv[++i];
    };
    if (a == "--model") model_id = next();
    else if (a == "--checkpoint-dir") ckpt = next();
    else if (a == "--world") world = std::stoi(next());
    else if (a == "--rank") rank = std::stoi(next());
    else if (a == "--peer") peer = next();
    else if (a == "--port") port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--prompt") prompt_text = next();
    else if (a == "--text") text_prompt = next();
    else if (a == "--steps") steps = std::stoi(next());
    else if (a == "--streaming") resident = false;
    else if (a == "--engine") {
      const std::string v = next();
      if (v == "incremental") incremental = true;
      else if (v == "reforward") incremental = false;
      else {
        DGPP_LOG_ERROR("--engine must be incremental|reforward");
        return 1;
      }
    }
    else if (a == "--no-eos") no_eos = true;
    else if (a == "--rendezvous-timeout-ms") rendezvous_timeout_ms = std::stoi(next());
    else if (a == "--out") out_prefix = next();
    else {
      std::fprintf(stderr,
                   "usage: glm_gen_check --model ORG/NAME | --checkpoint-dir "
                   "DIR --prompt ID,ID,... [--steps N] [--world N --rank R "
                   "--peer HOST --port N] [--streaming] [--engine "
                   "incremental|reforward] [--no-eos] [--out PREFIX]\n");
      return a == "--help" ? 0 : 1;
    }
  }
  if (world > 1) {
    require(rank >= 0 && rank < world, "--rank outside --world");
    require(!peer.empty() || rank == 0,
            "ranks > 0 need --peer (rank 0's fabric IP)");
  }
  if (steps < 1) {
    DGPP_LOG_ERROR("--steps must be >= 1");
    return 1;
  }
  if (!model_id.empty()) {
    if (!ckpt.empty()) {
      DGPP_LOG_ERROR("--model and --checkpoint-dir are mutually exclusive");
      return 1;
    }
    std::string err;
    const std::string snapshot = dgpp::hf::model_dir(model_id, &err);
    if (snapshot.empty()) {
      DGPP_LOG_ERROR("--model {}: {}", model_id, err);
      return 1;
    }
    ckpt = snapshot;
    DGPP_LOG_INFO("model {} -> {}", model_id, snapshot);
  }
  if (ckpt.empty()) {
    std::fprintf(stderr,
                 "usage: glm_gen_check --model ORG/NAME | --checkpoint-dir "
                 "DIR --prompt ID,ID,... [--steps N] [--world N --rank R "
                 "--peer HOST --port N] [--streaming] [--out PREFIX]\n");
    return 1;
  }

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    DGPP_LOG_ERROR("no CUDA device visible");
    return 1;
  }
  try {
    const GlmTextConfig cfg =
        GlmTextConfig::from_json_file((fs::path(ckpt) / "config.json").string());
    if (cfg.vocab_size > (1 << 18)) {
      DGPP_LOG_ERROR(
          "vocab {} exceeds the 6-bit-triplet pick encoding (2^18) — this "
          "app needs a wider id path before it can serve this checkpoint",
          cfg.vocab_size);
      return 1;
    }
    // --text goes through the exact tokenizer (Stage 3): the ids match
    // HF tokenizers 0.23.1 byte-for-byte (glm_tokenizer_test pins it).
    // --prompt stays for raw ids (parity with the earlier records).
    if (!text_prompt.empty() && !prompt_text.empty()) {
      DGPP_LOG_ERROR("--text and --prompt are mutually exclusive");
      return 1;
    }
    std::vector<int64_t> prompt;
    const dgpp::GlmTokenizer tok = dgpp::GlmTokenizer::load(
        (fs::path(ckpt) / "tokenizer.json").string());
    if (!text_prompt.empty()) {
      prompt = tok.encode(text_prompt);
      DGPP_LOG_INFO("prompt encoded to {} ids by the exact tokenizer",
                    prompt.size());
      require(!prompt.empty(), "--text produced no tokens");
    } else {
      prompt = parse_prompt_ids(prompt_text, cfg.vocab_size);
    }
    return run(cfg, ckpt, world, rank, port, peer, prompt, steps, resident, incremental, no_eos, tok,
               out_prefix, rendezvous_timeout_ms);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("rank {}: {}", rank, e.what());
    return 1;
  }
}
