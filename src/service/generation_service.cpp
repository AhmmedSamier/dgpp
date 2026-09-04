#include "service/generation_service.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <utility>

#include "common/log.hpp"
#include "service/json_out.hpp"

namespace dgpp::service {

namespace {

using dgpp::glm::Scheduler;
using dgpp::glm::SchedulerRequest;

// OpenAI finish_reason from the scheduler's retire reason.
const char* finish_reason(Scheduler::Result::Reason r) {
  switch (r) {
    case Scheduler::Result::Reason::kEos: return "stop";
    case Scheduler::Result::Reason::kSteps: return "length";
    // A cancelled stream has no reader; a cancelled non-stream request
    // cannot exist (cancellation only follows a client disconnect).
    case Scheduler::Result::Reason::kCancelled: return "stop";
    default: return "stop";
  }
}

void append_usage(std::string* out, int prompt_tokens,
                  int completion_tokens) {
  out->append("\"usage\":{\"prompt_tokens\":");
  append_json_int(out, prompt_tokens);
  out->append(",\"completion_tokens\":");
  append_json_int(out, completion_tokens);
  out->append(",\"total_tokens\":");
  append_json_int(out, prompt_tokens + completion_tokens);
  out->push_back('}');
}

// The chat.completion.chunk preamble: id/object/created/model.
void append_chunk_preamble(std::string* out, const std::string& id,
                           int64_t created, const std::string& model) {
  out->append("{\"id\":");
  append_json_string(out, id);
  out->append(",\"object\":\"chat.completion.chunk\",\"created\":");
  append_json_int(out, created);
  out->append(",\"model\":");
  append_json_string(out, model);
  out->append(",\"system_fingerprint\":null,\"choices\":[");
}

std::string chat_chunk_first(const std::string& id, int64_t created,
                             const std::string& model) {
  std::string out;
  append_chunk_preamble(&out, id, created, model);
  out.append(
      "{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
      "\"logprobs\":null,\"finish_reason\":null}]}");
  return out;
}

std::string chat_chunk_content(const std::string& id, int64_t created,
                               const std::string& model,
                               const std::string& delta,
                               const std::string& logprobs_json = "") {
  std::string out;
  append_chunk_preamble(&out, id, created, model);
  out.append("{\"index\":0,\"delta\":{\"content\":");
  append_json_string(&out, delta);
  out.append("},\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":null}]}");
  return out;
}

std::string chat_chunk_final(const std::string& id, int64_t created,
                             const std::string& model,
                             Scheduler::Result::Reason reason) {
  std::string out;
  append_chunk_preamble(&out, id, created, model);
  out.append("{\"index\":0,\"delta\":{},\"logprobs\":null,"
              "\"finish_reason\":");
  append_json_string(&out, finish_reason(reason));
  out.append("}]}");
  return out;
}

std::string chat_chunk_usage(const std::string& id, int64_t created,
                             const std::string& model, int prompt_tokens,
                             int completion_tokens) {
  std::string out;
  out.append("{\"id\":");
  append_json_string(&out, id);
  out.append(",\"object\":\"chat.completion.chunk\",\"created\":");
  append_json_int(&out, created);
  out.append(",\"model\":");
  append_json_string(&out, model);
  out.append(",\"system_fingerprint\":null,\"choices\":[],");
  append_usage(&out, prompt_tokens, completion_tokens);
  out.push_back('}');
  return out;
}

// The one-shot chat.completion object.
std::string chat_completion_body(const std::string& id, int64_t created,
                                 const std::string& model,
                                 const std::string& text,
                                 Scheduler::Result::Reason reason,
                                 int prompt_tokens, int completion_tokens,
                                 const std::string& logprobs_json = "") {
  std::string out;
  out.append("{\"id\":");
  append_json_string(&out, id);
  out.append(",\"object\":\"chat.completion\",\"created\":");
  append_json_int(&out, created);
  out.append(",\"model\":");
  append_json_string(&out, model);
  out.append(",\"system_fingerprint\":null,\"choices\":[{\"index\":0,"
              "\"message\":{\"role\":\"assistant\",\"content\":");
  append_json_string(&out, text);
  out.append("},\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":");
  append_json_string(&out, finish_reason(reason));
  out.append("}],");
  append_usage(&out, prompt_tokens, completion_tokens);
  out.push_back('}');
  return out;
}

// The legacy text_completion chunk shapes (POST /v1/completions).
std::string text_chunk_preamble_fields(const std::string& id, int64_t created,
                                       const std::string& model) {
  std::string out;
  out.append("{\"id\":");
  append_json_string(&out, id);
  out.append(",\"object\":\"text_completion.chunk\",\"created\":");
  append_json_int(&out, created);
  out.append(",\"model\":");
  append_json_string(&out, model);
  out.append(",\"system_fingerprint\":null,\"choices\":[");
  return out;
}

std::string text_chunk_first(const std::string& id, int64_t created,
                             const std::string& model) {
  std::string out = text_chunk_preamble_fields(id, created, model);
  out.append(
      "{\"index\":0,\"text\":\"\",\"logprobs\":null,\"finish_reason\":null}]}");
  return out;
}

std::string text_chunk_delta(const std::string& id, int64_t created,
                             const std::string& model,
                             const std::string& delta,
                             const std::string& logprobs_json = "") {
  std::string out = text_chunk_preamble_fields(id, created, model);
  out.append("{\"index\":0,\"text\":");
  append_json_string(&out, delta);
  out.append(",\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":null}]}");
  return out;
}

std::string text_chunk_final(const std::string& id, int64_t created,
                             const std::string& model,
                             Scheduler::Result::Reason reason) {
  std::string out = text_chunk_preamble_fields(id, created, model);
  out.append("{\"index\":0,\"text\":\"\",\"logprobs\":null,"
              "\"finish_reason\":");
  append_json_string(&out, finish_reason(reason));
  out.append("}]}");
  return out;
}

std::string text_completion_body(const std::string& id, int64_t created,
                                 const std::string& model,
                                 const std::string& text,
                                 Scheduler::Result::Reason reason,
                                 int prompt_tokens, int completion_tokens,
                                 const std::string& logprobs_json = "") {
  std::string out;
  out.append("{\"id\":");
  append_json_string(&out, id);
  out.append(",\"object\":\"text_completion\",\"created\":");
  append_json_int(&out, created);
  out.append(",\"model\":");
  append_json_string(&out, model);
  out.append(",\"system_fingerprint\":null,\"choices\":[{\"index\":0,"
              "\"text\":");
  append_json_string(&out, text);
  out.append(",\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":");
  append_json_string(&out, finish_reason(reason));
  out.append("}],");
  append_usage(&out, prompt_tokens, completion_tokens);
  out.push_back('}');
  return out;
}

// The model object, with the served sampling surface as an extension: the
// effective defaults (what an omitted field becomes) and whether stochastic
// requests are executable on this engine at all.
std::string model_object(const std::string& model_id, int64_t created,
                         bool sampling_available,
                         const glm_sample::Params& d) {
  std::string out;
  out.append("{\"id\":");
  append_json_string(&out, model_id);
  out.append(",\"object\":\"model\",\"created\":");
  append_json_int(&out, created);
  out.append(",\"owned_by\":\"dgpp\",\"sampling\":{\"available\":");
  out.append(sampling_available ? "true" : "false");
  out.append(",\"defaults\":{\"temperature\":");
  append_json_float(&out, d.temperature);
  out.append(",\"top_p\":");
  append_json_float(&out, d.top_p);
  out.append(",\"top_k\":");
  append_json_int(&out, d.top_k);
  out.append(",\"min_p\":");
  append_json_float(&out, d.min_p);
  out.append(",\"repetition_penalty\":");
  append_json_float(&out, d.repetition_penalty);
  out.append("}}}");
  return out;
}

}  // namespace

GenerationService::GenerationService(const ServiceConfig& cfg,
                                     dgpp::glm::SchedulerEngine* engine,
                                     const ModelFrontend* frontend,
                                     std::vector<int64_t> eos_token_ids)
    : cfg_(cfg),
      engine_(engine),
      frontend_(frontend),
      sched_(engine, std::move(eos_token_ids), cfg.queue_limit) {
  if (engine_ == nullptr)
    throw std::invalid_argument("GenerationService: engine must not be null");
  if (frontend_ == nullptr)
    throw std::invalid_argument("GenerationService: frontend required");
  if (cfg_.model_id.empty())
    throw std::invalid_argument("GenerationService: model_id required");
  // The sampling surface: the defaults must be a valid spec (an operator
  // error otherwise), and an engine that cannot draw serves GREEDY defaults
  // — loudly — rather than a stochastic mode it would silently ignore.
  glm_sample::validate_params(cfg_.sampling_defaults);
  sampling_available_ = engine_->supports_sampling();
  if (!sampling_available_ && cfg_.sampling_defaults.temperature > 0.0f) {
    DGPP_LOG_WARN(
        "serve: the bound engine cannot sample yet — the model's default "
        "(temperature {} top_p {}) is NOT served; omitted fields default to "
        "greedy and temperature > 0 is refused (400 sampling_unsupported)",
        cfg_.sampling_defaults.temperature, cfg_.sampling_defaults.top_p);
    cfg_.sampling_defaults = glm_sample::greedy_params();
  }
  seed_rng_.seed(cfg_.fixed_seed.has_value()
                     ? *cfg_.fixed_seed
                     : static_cast<uint64_t>(std::random_device{}()) << 32 ^
                           static_cast<uint64_t>(std::random_device{}()));
  DGPP_LOG_INFO(
      "serve: sampling {} — defaults temperature {} top_p {} top_k {} min_p "
      "{} repetition_penalty {}; seedless requests {}",
      sampling_available_ ? "available" : "unavailable (greedy engine)",
      cfg_.sampling_defaults.temperature, cfg_.sampling_defaults.top_p,
      cfg_.sampling_defaults.top_k, cfg_.sampling_defaults.min_p,
      cfg_.sampling_defaults.repetition_penalty,
      cfg_.fixed_seed.has_value()
          ? "use the fixed seed " + std::to_string(*cfg_.fixed_seed)
          : std::string("draw a fresh seed each"));
  // The SSE tap: tokens and retires ride the scheduler's observer
  // callbacks straight into the request records.
  sched_.set_observer(this);
}

// ---------------------------------------------------------------------------
// The sampling spec (both completion routes)
// ---------------------------------------------------------------------------

bool GenerationService::parse_sampling(const dgpp::minijson::Value& body,
                                       HttpResponseWriter& w,
                                       glm_sample::Params* sampling,
                                       uint64_t* seed) {
  glm_sample::Params p = cfg_.sampling_defaults;
  // One numeric field: present → a finite number inside [lo, hi] (an
  // integer when asked), else the 400 names it.
  const auto field = [&](const char* name, double lo, double hi,
                         bool integer, const auto& apply) -> bool {
    const dgpp::minijson::Value* v = body.find(name);
    if (v == nullptr) return true;
    const double x = v->is_number() ? v->as_double(0.0) : 0.0;
    if (!v->is_number() || !std::isfinite(x) ||
        (integer && x != std::floor(x))) {
      respond_error(w, 400,
                    std::string(name) + " must be " +
                        (integer ? "an integer" : "a finite number"),
                    "invalid_request_error", name);
      return false;
    }
    if (x < lo || x > hi) {
      std::string range = "[";
      append_json_float(&range, lo);
      range += ", ";
      append_json_float(&range, hi);
      range += "]";
      respond_error(w, 400, std::string(name) + " must be in " + range,
                    "invalid_request_error", name);
      return false;
    }
    apply(x);
    return true;
  };
  bool has_seed = false;
  if (!field("temperature", 0.0, 2.0, false,
             [&](double x) { p.temperature = static_cast<float>(x); }) ||
      !field("top_p", 0.0, 1.0, false,
             [&](double x) { p.top_p = static_cast<float>(x); }) ||
      !field("top_k", 0.0, 2147483647.0, true,
             [&](double x) { p.top_k = static_cast<int>(x); }) ||
      !field("min_p", 0.0, 1.0, false,
             [&](double x) { p.min_p = static_cast<float>(x); }) ||
      !field("repetition_penalty", 0.0, 100.0, false,
             [&](double x) { p.repetition_penalty = static_cast<float>(x); }) ||
      !field("presence_penalty", -2.0, 2.0, false,
             [&](double x) { p.presence_penalty = static_cast<float>(x); }) ||
      !field("frequency_penalty", -2.0, 2.0, false,
             [&](double x) { p.frequency_penalty = static_cast<float>(x); }) ||
      !field("seed", -9223372036854775808.0, 9223372036854775807.0, true,
             [&](double) {
               has_seed = true;
               *seed = static_cast<uint64_t>(body.find("seed")->as_int(0));
             }))
    return false;
  try {
    glm_sample::validate_params(p);
  } catch (const std::invalid_argument& e) {
    // The message starts with the field's name ("top_p must be in (0, 1]").
    const std::string what = e.what();
    respond_error(w, 400, what, "invalid_request_error",
                  what.substr(0, what.find(' ')));
    return false;
  }
  if (p.temperature > 0.0f && !sampling_available_) {
    respond_error(w, 400,
                  "temperature > 0 asks for stochastic sampling, which this "
                  "engine cannot execute yet (the graph engine's sampler is "
                  "the next slice); omit temperature or set it to 0",
                  "invalid_request_error", "temperature",
                  "sampling_unsupported");
    return false;
  }
  if (!has_seed)
    *seed = cfg_.fixed_seed.has_value() ? *cfg_.fixed_seed : seed_rng_();
  *sampling = p;
  return true;
}

// ---------------------------------------------------------------------------
// The error contract: the OpenAI error object, never a bare string.
// ---------------------------------------------------------------------------

void GenerationService::respond_error(HttpResponseWriter& w, int status,
                                      const std::string& message,
                                      const std::string& type,
                                      const std::string& param,
                                      const std::string& code) const {
  std::string body = "{\"error\":{\"message\":";
  append_json_string(&body, message);
  body.append(",\"type\":");
  append_json_string(&body, type);
  body.append(",\"param\":");
  if (param.empty())
    body.append("null");
  else
    append_json_string(&body, param);
  body.append(",\"code\":");
  if (code.empty())
    body.append("null");
  else
    append_json_string(&body, code);
  body.append("}}");
  w.respond(status, "application/json", std::move(body));
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

void GenerationService::handle(const HttpRequest& req,
                               HttpResponseWriter& w) {
  const std::string& p = req.path;
  if (p == "/v1/chat/completions" || p == "/v1/completions") {
    if (req.method != "POST") {
      respond_error(w, 405, req.method + " is not allowed here; use POST",
                    "invalid_request_error");
      return;
    }
    if (p == "/v1/chat/completions")
      route_chat_completions(req, w);
    else
      route_completions(req, w);
    return;
  }
  if (p == "/v1/models" || p.rfind("/v1/models/", 0) == 0) {
    if (req.method != "GET") {
      respond_error(w, 405, req.method + " is not allowed here; use GET",
                    "invalid_request_error");
      return;
    }
    route_models(req, w);
    return;
  }
  if (p == "/health") {
    route_health(w);
    return;
  }
  if (p == "/v1/metrics") {
    route_metrics(w);
    return;
  }
  respond_error(w, 404, "unknown route: " + req.method + " " + p,
                "invalid_request_error");
}

void GenerationService::route_health(HttpResponseWriter& w) const {
  w.respond(200, "application/json",
            "{\"status\":\"ok\",\"model\":" + json_string(cfg_.model_id) +
                "}");
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

void GenerationService::route_chat_completions(const HttpRequest& req,
                                              HttpResponseWriter& w) {
  dgpp::minijson::ParseResult parsed;
  try {
    parsed = dgpp::minijson::parse(req.body);
  } catch (const std::exception& e) {
    respond_error(w, 400, std::string("invalid JSON body: ") + e.what(),
                  "invalid_request_error", "body");
    return;
  }
  const dgpp::minijson::Value& body = parsed.root;
  if (!body.is_object()) {
    respond_error(w, 400, "the request body must be a JSON object",
                  "invalid_request_error", "body");
    return;
  }

  // model — must name the served model.
  const dgpp::minijson::Value* model = body.find("model");
  if (model == nullptr || !model->is_string()) {
    respond_error(w, 400, "model is required and must be a string",
                  "invalid_request_error", "model");
    return;
  }
  if (model->as_string() != cfg_.model_id) {
    respond_error(w, 404, "the model '" + std::string(model->as_string()) +
                              "' does not exist on this server (serving '" +
                              cfg_.model_id + "')",
                  "invalid_request_error", "model", "model_not_found");
    return;
  }

  // messages — [{role, content: string}].
  const dgpp::minijson::Value* messages = body.find("messages");
  if (messages == nullptr || !messages->is_array() ||
      messages->items().empty()) {
    respond_error(w, 400,
                  "messages is required and must be a non-empty array",
                  "invalid_request_error", "messages");
    return;
  }
  for (const auto& msg : messages->items()) {
    if (!msg.is_object() || msg.find("role") == nullptr ||
        !msg.find("role")->is_string() ||
        msg.find("content") == nullptr) {
      respond_error(w, 400,
                    "every message must be an object with role and content",
                    "invalid_request_error", "messages");
      return;
    }
    if (!msg.find("content")->is_string()) {
      respond_error(w, 400,
                    "content must be a string in v1 (content-part arrays "
                    "arrive with the multimodal stage)",
                    "invalid_request_error", "messages.content");
      return;
    }
  }

  // max_completion_tokens (preferred) / max_tokens (legacy) — >= 1.
  const dgpp::minijson::Value* max_new =
      body.find("max_completion_tokens");
  const dgpp::minijson::Value* max_old = body.find("max_tokens");
  const dgpp::minijson::Value* max_tokens = max_new != nullptr ? max_new : max_old;
  if (max_new != nullptr && max_old != nullptr && max_new != max_old) {
    respond_error(w, 400,
                  "max_tokens and max_completion_tokens are aliases; send one",
                  "invalid_request_error", "max_tokens");
    return;
  }
  int steps = cfg_.default_max_tokens;
  if (max_tokens != nullptr) {
    if (!max_tokens->is_number() || max_tokens->as_int(0) < 1) {
      respond_error(w, 400, "max_tokens must be a number >= 1",
                    "invalid_request_error", "max_tokens");
      return;
    }
    steps = static_cast<int>(max_tokens->as_int(0));
  }

  // stream + stream_options.include_usage.
  bool stream = false;
  const dgpp::minijson::Value* sv = body.find("stream");
  if (sv != nullptr) {
    if (!sv->is_bool()) {
      respond_error(w, 400, "stream must be a boolean",
                    "invalid_request_error", "stream");
      return;
    }
    stream = sv->as_bool(false);
  }
  bool include_usage = false;
  const dgpp::minijson::Value* so = body.find("stream_options");
  if (so != nullptr) {
    if (!so->is_object()) {
      respond_error(w, 400, "stream_options must be an object",
                    "invalid_request_error", "stream_options");
      return;
    }
    const dgpp::minijson::Value* iu = so->find("include_usage");
    if (iu != nullptr && !iu->is_bool()) {
      respond_error(w, 400, "stream_options.include_usage must be a boolean",
                    "invalid_request_error", "stream_options.include_usage");
      return;
    }
    include_usage = iu != nullptr && iu->as_bool(false);
  }

  // Sampling: the spec over the model's defaults (M6 6b). temperature 0
  // is the exact greedy path; anything the engine cannot execute is a 400.
  glm_sample::Params sampling;
  uint64_t seed = 0;
  if (!parse_sampling(body, w, &sampling, &seed)) return;

  // logprobs (bool) + top_logprobs (0..20): exact from the same sampler.
  int logprobs = -1;
  {
    const dgpp::minijson::Value* lp = body.find("logprobs");
    const dgpp::minijson::Value* top = body.find("top_logprobs");
    if (lp != nullptr && !lp->is_bool()) {
      respond_error(w, 400, "logprobs must be a boolean",
                    "invalid_request_error", "logprobs");
      return;
    }
    const bool want = lp != nullptr && lp->as_bool(false);
    if (top != nullptr) {
      const double x = top->is_number() ? top->as_double(-1.0) : -1.0;
      if (!top->is_number() || x != std::floor(x) || x < 0.0 || x > 20.0) {
        respond_error(w, 400, "top_logprobs must be an integer in [0, 20]",
                      "invalid_request_error", "top_logprobs");
        return;
      }
      if (!want) {
        respond_error(w, 400, "top_logprobs requires logprobs: true",
                      "invalid_request_error", "top_logprobs");
        return;
      }
    }
    if (want) {
      if (!engine_->supports_logprobs()) {
        respond_error(w, 400,
                      "logprobs are not available on this engine",
                      "invalid_request_error", "logprobs",
                      "logprobs_unsupported");
        return;
      }
      logprobs = top != nullptr ? static_cast<int>(top->as_double(0.0)) : 0;
      sampling.logprobs = logprobs;
    }
  }

  // The loud-refusal ladder for everything not implemented in v1.
  const char* unsupported[] = {
      "stop",           "n",                 "logit_bias",
      "user",           "tools",             "tool_choice",
      "response_format", "parallel_tool_calls", "store",
      "metadata",       "reasoning_effort",  "service_tier",
      "prediction",     "audio",
  };
  for (const char* param : unsupported) {
    if (body.find(param) != nullptr) {
      respond_error(
          w, 400,
          std::string("'") + param +
              "' is not supported in this server version — accepted "
              "parameters behave exactly as specified, and unimplemented "
              "ones are refused rather than ignored",
          "invalid_request_error", param, "unsupported_parameter");
      return;
    }
  }

  // The prompt: render the chat template over the messages, encode.
  std::vector<int64_t> prompt;
  try {
    prompt = frontend_->encode_text(frontend_->render_chat(*messages));
  } catch (const std::exception& e) {
    respond_error(w, 400,
                  "the chat template rejected these messages: " +
                      std::string(e.what()),
                  "invalid_request_error", "messages");
    return;
  }
  if (prompt.empty()) {
    respond_error(w, 400, "the rendered prompt produced no tokens",
                  "invalid_request_error", "messages");
    return;
  }

  // Full-reserve admission arithmetic — a request that can never fit is
  // a 400, never a scheduler deadlock.
  const int64_t reserve =
      engine_->blocks_for_tokens(static_cast<int64_t>(prompt.size()) + steps);
  if (reserve > engine_->pool_blocks_total()) {
    respond_error(
        w, 400,
        "the request's token budget (prompt " +
            std::to_string(prompt.size()) + " + max_tokens " +
            std::to_string(steps) + ") exceeds the model's KV capacity",
        "invalid_request_error", "max_tokens", "context_length_exceeded");
    return;
  }

  auto record = std::make_shared<StreamRecord>();
  record->tag = next_tag_.fetch_add(1, std::memory_order_relaxed);
  {
    char suffix[17];
    std::snprintf(suffix, sizeof(suffix), "%016llx",
                  static_cast<unsigned long long>(record->tag));
    record->id = "chatcmpl-" + std::string(suffix);
  }
  record->model = cfg_.model_id;
  record->created_unix = std::time(nullptr);
  record->stream = stream;
  record->include_usage = include_usage;
  record->prompt_tokens = static_cast<int>(prompt.size());
  record->writer = &w;
  w.set_stream_tag(record->tag);
  if (stream) w.begin_stream();

  SchedulerRequest sr;
  sr.id = record->id;
  sr.prompt = std::move(prompt);
  sr.max_steps = steps;
  sr.sampling = sampling;
  sr.seed = seed;
  sr.logprobs = logprobs;
  record->logprobs = logprobs;
  enqueue_admission(std::move(record), std::move(sr));
}

// ---------------------------------------------------------------------------
// POST /v1/completions (the legacy prompt API)
// ---------------------------------------------------------------------------

void GenerationService::route_completions(const HttpRequest& req,
                                          HttpResponseWriter& w) {
  dgpp::minijson::ParseResult parsed;
  try {
    parsed = dgpp::minijson::parse(req.body);
  } catch (const std::exception& e) {
    respond_error(w, 400, std::string("invalid JSON body: ") + e.what(),
                  "invalid_request_error", "body");
    return;
  }
  const dgpp::minijson::Value& body = parsed.root;
  if (!body.is_object()) {
    respond_error(w, 400, "the request body must be a JSON object",
                  "invalid_request_error", "body");
    return;
  }
  const dgpp::minijson::Value* model = body.find("model");
  if (model == nullptr || !model->is_string()) {
    respond_error(w, 400, "model is required and must be a string",
                  "invalid_request_error", "model");
    return;
  }
  if (model->as_string() != cfg_.model_id) {
    respond_error(w, 404, "the model '" + std::string(model->as_string()) +
                              "' does not exist on this server",
                  "invalid_request_error", "model", "model_not_found");
    return;
  }
  const dgpp::minijson::Value* prompt = body.find("prompt");
  if (prompt == nullptr || !prompt->is_string()) {
    respond_error(w, 400,
                  "prompt is required and must be a string in v1 (prompt "
                  "arrays arrive with the batching stage)",
                  "invalid_request_error", "prompt");
    return;
  }

  int steps = cfg_.default_max_tokens;
  const dgpp::minijson::Value* max_tokens = body.find("max_tokens");
  if (max_tokens != nullptr) {
    if (!max_tokens->is_number() || max_tokens->as_int(0) < 1) {
      respond_error(w, 400, "max_tokens must be a number >= 1",
                    "invalid_request_error", "max_tokens");
      return;
    }
    steps = static_cast<int>(max_tokens->as_int(0));
  }
  bool stream = false;
  const dgpp::minijson::Value* sv = body.find("stream");
  if (sv != nullptr) {
    if (!sv->is_bool()) {
      respond_error(w, 400, "stream must be a boolean",
                    "invalid_request_error", "stream");
      return;
    }
    stream = sv->as_bool(false);
  }

  glm_sample::Params sampling;
  uint64_t seed = 0;
  if (!parse_sampling(body, w, &sampling, &seed)) return;
  // The legacy logprobs: an integer 0..5 (null = none).
  int logprobs = -1;
  if (const dgpp::minijson::Value* lp = body.find("logprobs")) {
    const double x = lp->is_number() ? lp->as_double(-1.0) : -1.0;
    if (!lp->is_number() || x != std::floor(x) || x < 0.0 || x > 5.0) {
      respond_error(w, 400, "logprobs must be an integer in [0, 5]",
                    "invalid_request_error", "logprobs");
      return;
    }
    if (!engine_->supports_logprobs()) {
      respond_error(w, 400, "logprobs are not available on this engine",
                    "invalid_request_error", "logprobs",
                    "logprobs_unsupported");
      return;
    }
    logprobs = static_cast<int>(x);
    sampling.logprobs = logprobs;
  }

  const char* unsupported[] = {"echo",        "suffix",        "top_logprobs",
                              "n",           "best_of",       "stop",
                              "logit_bias",  "user"};
  for (const char* param : unsupported) {
    if (body.find(param) != nullptr) {
      respond_error(w, 400,
                    std::string("'") + param +
                        "' is not supported in this server version",
                    "invalid_request_error", param, "unsupported_parameter");
      return;
    }
  }

  std::vector<int64_t> ids = frontend_->encode_text(prompt->as_string());
  if (ids.empty()) {
    respond_error(w, 400, "the prompt produced no tokens",
                  "invalid_request_error", "prompt");
    return;
  }
  const int64_t reserve =
      engine_->blocks_for_tokens(static_cast<int64_t>(ids.size()) + steps);
  if (reserve > engine_->pool_blocks_total()) {
    respond_error(w, 400,
                  "the request's token budget exceeds the model's KV capacity",
                  "invalid_request_error", "max_tokens",
                  "context_length_exceeded");
    return;
  }

  auto record = std::make_shared<StreamRecord>();
  record->tag = next_tag_.fetch_add(1, std::memory_order_relaxed);
  {
    char suffix[17];
    std::snprintf(suffix, sizeof(suffix), "%016llx",
                  static_cast<unsigned long long>(record->tag));
    record->id = "cmpl-" + std::string(suffix);
  }
  record->model = cfg_.model_id;
  record->created_unix = std::time(nullptr);
  record->stream = stream;
  record->prompt_tokens = static_cast<int>(ids.size());
  record->writer = &w;
  w.set_stream_tag(record->tag);
  if (stream) w.begin_stream();

  SchedulerRequest sr;
  sr.id = record->id;
  sr.prompt = std::move(ids);
  sr.max_steps = steps;
  sr.sampling = sampling;
  sr.seed = seed;
  sr.logprobs = logprobs;
  record->logprobs = logprobs;
  enqueue_admission(std::move(record), std::move(sr));
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

void GenerationService::route_models(const HttpRequest& req,
                                     HttpResponseWriter& w) {
  const int64_t created = std::time(nullptr);
  if (req.path == "/v1/models") {
    w.respond(200, "application/json",
              "{\"object\":\"list\",\"data\":[" +
                  model_object(cfg_.model_id, created, sampling_available_,
                               cfg_.sampling_defaults) +
                  "]}");
    return;
  }
  const std::string id = req.path.substr(std::string("/v1/models/").size());
  if (id == cfg_.model_id) {
    w.respond(200, "application/json",
              model_object(id, created, sampling_available_,
                           cfg_.sampling_defaults));
    return;
  }
  respond_error(w, 404, "the model '" + id + "' does not exist",
                "invalid_request_error", "model", "model_not_found");
}

// ---------------------------------------------------------------------------
// GET /v1/metrics (ours — the scheduler meters, published by the engine)
// ---------------------------------------------------------------------------

void GenerationService::route_metrics(HttpResponseWriter& w) {
  Scheduler::Meters m;
  Stats st;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    m = meters_;
    st = stats_;
  }
  std::string out = "{\"scheduler\":{\"active\":";
  append_json_int(&out, m.active);
  out.append(",\"queued\":");
  append_json_int(&out, m.queued);
  out.append(",\"terminal\":");
  append_json_int(&out, m.terminal);
  out.append(",\"pool_blocks_total\":");
  append_json_int(&out, m.pool_blocks_total);
  out.append(",\"pool_blocks_in_use\":");
  append_json_int(&out, m.pool_blocks_in_use);
  out.append(",\"tokens_generated\":");
  append_json_int(&out, m.tokens_generated);
  out.append("},\"service\":{\"requests_total\":");
  append_json_int(&out, static_cast<int64_t>(st.requests_total));
  out.append(",\"requests_shed\":");
  append_json_int(&out, static_cast<int64_t>(st.requests_shed));
  out.append(",\"requests_cancelled\":");
  append_json_int(&out, static_cast<int64_t>(st.requests_cancelled));
  out.append(",\"tokens_out\":");
  append_json_int(&out, static_cast<int64_t>(st.tokens_out));
  out.append(",\"rejects_bad\":");
  append_json_int(&out, static_cast<int64_t>(st.rejects_bad));
  out.append("}}");
  w.respond(200, "application/json", std::move(out));
}

// ---------------------------------------------------------------------------
// The admission seam
// ---------------------------------------------------------------------------

void GenerationService::enqueue_admission(std::shared_ptr<StreamRecord> record,
                                           SchedulerRequest request) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.requests_total++;
    // EVERY record enters the lifecycle list at enqueue — the observer,
    // the disconnect hook, and the pump must see a request that is
    // still waiting for the engine thread (a disconnect can race the
    // engine pass, and the cancel has to land either way).
    records_.push_back(record);
    if (shutdown_ ||
        pending_admissions_.size() >=
            static_cast<size_t>(cfg_.queue_limit)) {
      record->reject_overloaded = true;
      stats_.requests_shed++;
      return;
    }
    pending_admissions_.push_back(
        PendingAdmission{std::move(record), std::move(request)});
  }
}

void GenerationService::on_disconnect(uint64_t tag) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& r : records_) {
    if (r->tag != tag) continue;
    r->writer_dead = true;
    r->writer = nullptr;
    if (!r->done && !r->cancel_armed) {
      // Stage 4's contract: a client disconnect maps onto the same
      // deterministic retire path as scripted cancellation.
      r->cancel_armed = true;
      pending_cancels_.push_back(PendingCancel{r->id});
      stats_.requests_cancelled++;
      DGPP_LOG_INFO("serve: request {} cancelled by client disconnect",
                    r->id);
    }
    return;
  }
  // Unknown tag: a long-finished record's connection finally closed.
}

// ---------------------------------------------------------------------------
// The observer (engine thread — the scheduler's inline callbacks)
// ---------------------------------------------------------------------------

void GenerationService::on_token(const std::string& id, int64_t token,
                                 int steps_done) {
  (void)steps_done;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& r : records_) {
      if (r->id != id || r->done) continue;
      r->ids.push_back(token);
      // Exact incremental text: the suffix diff of successive full
      // decodes — UTF-8 splits and special tokens come out right by
      // construction (the tokenizer's own decode gates, pinned by its
      // differential goldens).
      const std::string full = frontend_->decode_ids(r->ids);
      if (full.size() > r->text.size())
        r->delta.append(full, r->text.size(), std::string::npos);
      r->text = full;
      stats_.tokens_out++;
      break;
    }
  }
  // The audit tap sees every engine event, even ones this record list
  // no longer knows (post-shutdown ticks) — cross-rank comparability
  // is the tap's entire job.
  if (audit_) audit_->on_token(id, token, steps_done);
}

void GenerationService::on_token_logprobs(const std::string& id,
                                          int steps_done,
                                          const glm_sample::Result& logprobs) {
  (void)steps_done;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& r : records_) {
    if (r->id != id || r->done) continue;
    if (r->logprobs >= 0) r->lps.push_back(logprobs);
    break;
  }
}

namespace {

void append_bytes_array(std::string* out, const std::string& s) {
  out->push_back('[');
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) out->push_back(',');
    append_json_int(out, static_cast<unsigned char>(s[i]));
  }
  out->push_back(']');
}

}  // namespace

std::string GenerationService::logprobs_content(const StreamRecord& r,
                                                size_t from, size_t to) const {
  // {"content":[{"token","logprob","bytes","top_logprobs":[...]}, ...]}
  std::string out = "{\"content\":[";
  for (size_t i = from; i < to && i < r.lps.size(); ++i) {
    if (i != from) out.push_back(',');
    const glm_sample::Result& lp = r.lps[i];
    const std::string tok = frontend_->decode_ids({r.ids[i]});
    out.append("{\"token\":");
    append_json_string(&out, tok);
    out.append(",\"logprob\":");
    append_json_float(&out, lp.logprob);
    out.append(",\"bytes\":");
    append_bytes_array(&out, tok);
    out.append(",\"top_logprobs\":[");
    for (size_t j = 0; j < lp.top_logprobs.size(); ++j) {
      if (j) out.push_back(',');
      const std::string alt = frontend_->decode_ids({lp.top_logprobs[j].first});
      out.append("{\"token\":");
      append_json_string(&out, alt);
      out.append(",\"logprob\":");
      append_json_float(&out, lp.top_logprobs[j].second);
      out.append(",\"bytes\":");
      append_bytes_array(&out, alt);
      out.push_back('}');
    }
    out.append("]}");
  }
  out.append("]}");
  return out;
}

std::string GenerationService::legacy_logprobs(const StreamRecord& r) const {
  // {"tokens":[...],"token_logprobs":[...],"top_logprobs":[{tok: lp}],
  //  "text_offset":[...]}
  std::string tokens = "[", lps = "[", tops = "[", offsets = "[";
  size_t offset = 0;
  for (size_t i = 0; i < r.lps.size(); ++i) {
    if (i) {
      tokens.push_back(',');
      lps.push_back(',');
      tops.push_back(',');
      offsets.push_back(',');
    }
    const std::string tok = frontend_->decode_ids({r.ids[i]});
    append_json_string(&tokens, tok);
    append_json_float(&lps, r.lps[i].logprob);
    tops.push_back('{');
    for (size_t j = 0; j < r.lps[i].top_logprobs.size(); ++j) {
      if (j) tops.push_back(',');
      append_json_string(&tops, frontend_->decode_ids({r.lps[i].top_logprobs[j].first}));
      tops.push_back(':');
      append_json_float(&tops, r.lps[i].top_logprobs[j].second);
    }
    tops.push_back('}');
    append_json_int(&offsets, static_cast<int64_t>(offset));
    offset += tok.size();
  }
  return "{\"tokens\":" + tokens + "],\"token_logprobs\":" + lps +
         "],\"top_logprobs\":" + tops + "],\"text_offset\":" + offsets + "]}";
}

void GenerationService::on_retire(const std::string& id,
                                 const Scheduler::Result& result) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& r : records_) {
      if (r->id != id) continue;
      r->done = true;
      r->reason = result.reason;
      r->completion_tokens = result.steps_done;
      break;
    }
  }
  if (audit_) audit_->on_retire(id, result);
}

// ---------------------------------------------------------------------------
// idle(): the record pump (HTTP thread)
// ---------------------------------------------------------------------------

void GenerationService::idle() { pump_records(); }

void GenerationService::pump_records() {
  // Snapshot the drainable records under the lock; writers/formatting
  // happen outside it.
  std::vector<std::shared_ptr<StreamRecord>> live;
  std::vector<std::shared_ptr<StreamRecord>> finished;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& r : records_) {
      if (r->done || r->reject_overloaded)
        finished.push_back(r);
      else
        live.push_back(r);
    }
  }

  for (auto& r : live) {
    if (r->writer == nullptr || r->writer_dead) continue;
    if (!r->stream) continue;  // one-shots answer once, at finish
    if (!r->first_chunk_sent) {
      r->first_chunk_sent = true;
      r->writer->write_event(r->id.rfind("cmpl-", 0) == 0
                                 ? text_chunk_first(r->id, r->created_unix,
                                                    r->model)
                                 : chat_chunk_first(r->id, r->created_unix,
                                                    r->model));
    }
    if (!r->delta.empty()) {
      std::string lp_json;
      if (r->logprobs >= 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t to = r->lps.size();
        if (to > r->lps_flushed) {
          lp_json = logprobs_content(*r, r->lps_flushed, to);
          r->lps_flushed = to;
        }
      }
      r->writer->write_event(
          r->id.rfind("cmpl-", 0) == 0
              ? text_chunk_delta(r->id, r->created_unix, r->model, r->delta,
                                 lp_json)
              : chat_chunk_content(r->id, r->created_unix, r->model, r->delta,
                                   lp_json));
      r->delta.clear();
    }
  }

  for (auto& r : finished) {
    if (r->writer != nullptr && !r->writer_dead) {
      if (r->reject_overloaded) {
        // A shed request: one-shot gets the 503 object; a stream that
        // already began gets the error event + [DONE] (the only shape
        // an SSE client can see).
        const std::string msg =
            "the server is overloaded — the admission queue is full; "
            "retry after backing off";
        if (r->stream) {
          // Headers were already sent in handle(); an SSE client can
          // only see an error event + [DONE] (OpenAI's stream-error
          // shape), whether or not a role chunk preceded it.
          std::string ev = "{\"error\":{\"message\":";
          append_json_string(&ev, msg);
          ev.append(",\"type\":\"server_error\",\"param\":null,"
                    "\"code\":\"overloaded\"}}");
          r->writer->write_event(ev);
          r->writer->write_event("[DONE]");
          r->writer->end_stream();
          r->first_chunk_sent = true;
        } else {
          respond_error(*r->writer, 503, msg, "server_error", "",
                        "overloaded");
        }
      } else if (r->stream) {
        // Flush any straggler delta, then the terminal sequence.
        if (!r->first_chunk_sent) {
          r->writer->write_event(
              r->id.rfind("cmpl-", 0) == 0
                  ? text_chunk_first(r->id, r->created_unix, r->model)
                  : chat_chunk_first(r->id, r->created_unix, r->model));
          r->first_chunk_sent = true;
        }
        if (!r->delta.empty()) {
          std::string lp_json;
          if (r->logprobs >= 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            const size_t to = r->lps.size();
            if (to > r->lps_flushed) {
              lp_json = logprobs_content(*r, r->lps_flushed, to);
              r->lps_flushed = to;
            }
          }
          r->writer->write_event(
              r->id.rfind("cmpl-", 0) == 0
                  ? text_chunk_delta(r->id, r->created_unix, r->model,
                                     r->delta, lp_json)
                  : chat_chunk_content(r->id, r->created_unix, r->model,
                                       r->delta, lp_json));
          r->delta.clear();
        }
        r->writer->write_event(
            r->id.rfind("cmpl-", 0) == 0
                ? text_chunk_final(r->id, r->created_unix, r->model,
                                   r->reason)
                : chat_chunk_final(r->id, r->created_unix, r->model,
                                   r->reason));
        if (r->include_usage) {
          r->writer->write_event(
              r->id.rfind("cmpl-", 0) == 0
                  ? chat_chunk_usage(r->id, r->created_unix, r->model,
                                     r->prompt_tokens, r->completion_tokens)
                  : chat_chunk_usage(r->id, r->created_unix, r->model,
                                     r->prompt_tokens, r->completion_tokens));
        }
        r->writer->write_event("[DONE]");
        r->writer->end_stream();
      } else {
        // The one-shot completion object.
        std::string lp_json;
        if (r->logprobs >= 0) {
          std::lock_guard<std::mutex> lock(mutex_);
          lp_json = r->id.rfind("cmpl-", 0) == 0
                        ? legacy_logprobs(*r)
                        : logprobs_content(*r, 0, r->lps.size());
        }
        r->writer->respond(
            200, "application/json",
            r->id.rfind("cmpl-", 0) == 0
                ? text_completion_body(r->id, r->created_unix, r->model,
                                       r->text, r->reason,
                                       r->prompt_tokens,
                                       r->completion_tokens, lp_json)
                : chat_completion_body(r->id, r->created_unix, r->model,
                                        r->text, r->reason,
                                        r->prompt_tokens, r->completion_tokens,
                                        lp_json));
      }
    }
    // Records leave the list only here — done and (writer drained or
    // dead). A cancelled-but-still-generating record keeps its entry
    // until on_retire lands.
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& r : finished) {
    auto it = std::find(records_.begin(), records_.end(), r);
    if (it != records_.end()) records_.erase(it);
  }
}

// ---------------------------------------------------------------------------
// engine_pass() — the app's engine loop body
// ---------------------------------------------------------------------------

bool GenerationService::engine_pass(const PreTickHook& pre_tick) {
  std::vector<PendingAdmission> admissions;
  std::vector<PendingCancel> cancels;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    admissions.swap(pending_admissions_);
    cancels.swap(pending_cancels_);
  }
  PassEvents events;
  for (auto& a : admissions) {
    bool admitted = false;
    bool journaled = false;
    try {
      if (pre_tick) {
        // The journal serializes the EXACT request the scheduler takes
        // (a copy — std::move would leave a husk for the record pump).
        events.submits.push_back(a.request);
        journaled = true;
        admitted = sched_.try_submit(events.submits.back());
      } else {
        admitted = sched_.try_submit(std::move(a.request));
      }
    } catch (const std::exception& e) {
      // Unreachable by construction (the route validated everything
      // the scheduler re-checks) — but a bug here must not wedge the
      // waiting client.
      DGPP_LOG_ERROR("serve: admission rejected: {}", e.what());
    }
    if (!admitted) {
      // A shed (queue full) or a rejected request died HERE, on rank
      // 0 — peers must never learn it existed, or their identical
      // queues would diverge from rank 0's.
      if (journaled) events.submits.pop_back();
      std::lock_guard<std::mutex> lock(mutex_);
      a.record->reject_overloaded = true;
      stats_.requests_shed++;
    }
  }
  for (const auto& c : cancels) {
    // Only cancels that HIT ride the journal (rank 0's scheduler state
    // changed). A late cancel is a no-op everywhere — peers' state is
    // identical, so replaying it would no-op there too; silence is
    // cheaper than noise.
    if (sched_.cancel(c.scheduler_id) && pre_tick)
      events.cancels.push_back(c.scheduler_id);
  }

  // The fixed journal position: this record and the tick below are one
  // atomic unit — rank 0 never ticks without broadcasting, a peer
  // never ticks without a record (see fabric_serve.hpp).
  if (pre_tick) pre_tick(events);

  const bool more = sched_.tick();  // may throw on scheduler contract
                                       // violations — the app treats
                                       // that as fatal (operator class)

  {
    std::lock_guard<std::mutex> lock(mutex_);
    meters_ = sched_.meters();
    return more || !pending_admissions_.empty();
  }
}

void GenerationService::begin_shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  shutdown_ = true;
  for (auto& a : pending_admissions_) {
    a.record->reject_overloaded = true;
    stats_.requests_shed++;
  }
  pending_admissions_.clear();
  for (auto& r : records_) {
    if (!r->done && !r->reject_overloaded) {
      r->reject_overloaded = true;
      stats_.requests_shed++;
      pending_cancels_.push_back(PendingCancel{r->id});
    }
  }
}

GenerationService::Stats GenerationService::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

}  // namespace dgpp::service
