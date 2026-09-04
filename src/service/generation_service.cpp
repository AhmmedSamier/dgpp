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
using dgpp::glm::ToolCallParser;

// OpenAI finish_reason from the scheduler's retire reason. A turn that
// ended naturally after at least one parsed tool call is "tool_calls";
// the steps cap is "length" whatever was parsed.
const char* finish_reason(Scheduler::Result::Reason r, bool tool_calls) {
  switch (r) {
    case Scheduler::Result::Reason::kEos:
      return tool_calls ? "tool_calls" : "stop";
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

// One chunk carrying `delta_json` (a complete JSON object: the delta).
std::string chat_chunk_delta(const std::string& id, int64_t created,
                             const std::string& model,
                             const std::string& delta_json,
                             const std::string& logprobs_json = "") {
  std::string out;
  append_chunk_preamble(&out, id, created, model);
  out.append("{\"index\":0,\"delta\":");
  out.append(delta_json);
  out.append(",\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":null}]}");
  return out;
}

std::string delta_text(const char* field, const std::string& text) {
  std::string out = "{\"";
  out.append(field);
  out.append("\":");
  append_json_string(&out, text);
  out.push_back('}');
  return out;
}

// The tool-call id: call_<16 hex>, a pure function of the record and the
// call's index (reproducible under the gates' fixed seeds, distinct per
// request and per call).
std::string tool_call_id(uint64_t tag, int index) {
  uint64_t z = tag * 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(index) +
               0x632BE59BD9B4E019ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  char buf[24];
  std::snprintf(buf, sizeof(buf), "call_%016llx",
                static_cast<unsigned long long>(z));
  return buf;
}

// The streamed tool-call deltas: one announcing the call (index, id, type,
// name, empty arguments) and one carrying the complete arguments string
// — a client that concatenates argument fragments sees one fragment.
std::string delta_tool_call_start(int index, const std::string& call_id,
                                  const std::string& name) {
  std::string out = "{\"tool_calls\":[{\"index\":";
  append_json_int(&out, index);
  out.append(",\"id\":");
  append_json_string(&out, call_id);
  out.append(",\"type\":\"function\",\"function\":{\"name\":");
  append_json_string(&out, name);
  out.append(",\"arguments\":\"\"}}]}");
  return out;
}

std::string delta_tool_call_arguments(int index, const std::string& args) {
  std::string out = "{\"tool_calls\":[{\"index\":";
  append_json_int(&out, index);
  out.append(",\"function\":{\"arguments\":");
  append_json_string(&out, args);
  out.append("}}]}");
  return out;
}

std::string chat_chunk_final(const std::string& id, int64_t created,
                             const std::string& model, const char* finish,
                             const std::string& logprobs_json = "") {
  std::string out;
  append_chunk_preamble(&out, id, created, model);
  out.append("{\"index\":0,\"delta\":{},\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":");
  append_json_string(&out, finish);
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

// The one-shot chat.completion object around a complete message object.
std::string chat_completion_body(const std::string& id, int64_t created,
                                 const std::string& model,
                                 const std::string& message_json,
                                 const char* finish, int prompt_tokens,
                                 int completion_tokens,
                                 const std::string& logprobs_json = "") {
  std::string out;
  out.append("{\"id\":");
  append_json_string(&out, id);
  out.append(",\"object\":\"chat.completion\",\"created\":");
  append_json_int(&out, created);
  out.append(",\"model\":");
  append_json_string(&out, model);
  out.append(",\"system_fingerprint\":null,\"choices\":[{\"index\":0,"
              "\"message\":");
  out.append(message_json);
  out.append(",\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":");
  append_json_string(&out, finish);
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
                             const std::string& model, const char* finish,
                             const std::string& logprobs_json = "") {
  std::string out = text_chunk_preamble_fields(id, created, model);
  out.append("{\"index\":0,\"text\":\"\",\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":");
  append_json_string(&out, finish);
  out.append("}]}");
  return out;
}

std::string text_completion_body(const std::string& id, int64_t created,
                                 const std::string& model,
                                 const std::string& text, const char* finish,
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
  append_json_string(&out, finish);
  out.append("}],");
  append_usage(&out, prompt_tokens, completion_tokens);
  out.push_back('}');
  return out;
}

// The model object, with the served surfaces as extensions: the sampling
// defaults (what an omitted field becomes), whether stochastic requests
// are executable on this engine at all, and the tool/reasoning surface.
std::string model_object(const std::string& model_id, int64_t created,
                         bool sampling_available,
                         const glm_sample::Params& d, bool tools_available,
                         bool constraints_available,
                         bool reasoning_in_content) {
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
  out.append("}},\"tools\":{\"available\":");
  out.append(tools_available ? "true" : "false");
  out.append(",\"constrained\":");
  out.append(constraints_available ? "true" : "false");
  out.append("},\"response_format\":{\"json_object\":");
  out.append(constraints_available ? "true" : "false");
  out.append(",\"json_schema\":");
  out.append(constraints_available ? "true" : "false");
  out.append("},\"reasoning\":{\"in_content\":");
  out.append(reasoning_in_content ? "true" : "false");
  out.append("}}");
  return out;
}

// A minijson value is a string or an array whose items are strings or
// objects — the two content forms the template's visible_text renders
// (content parts: {type: "text", text} and the media kinds it answers
// with a reminder).
bool content_form_ok(const dgpp::minijson::Value& v) {
  if (v.is_string()) return true;
  if (!v.is_array()) return false;
  for (const auto& item : v.items()) {
    if (item.is_string()) continue;
    if (!item.is_object()) return false;
    const dgpp::minijson::Value* type = item.find("type");
    if (type == nullptr || !type->is_string()) return false;
  }
  return true;
}

}  // namespace

GenerationService::GenerationService(const ServiceConfig& cfg,
                                     dgpp::glm::SchedulerEngine* engine,
                                     const ModelFrontend* frontend,
                                     std::vector<int64_t> eos_token_ids)
    : cfg_(cfg),
      engine_(engine),
      frontend_(frontend),
      markers_(frontend != nullptr ? frontend->markers()
                                   : dgpp::glm::ChatMarkers{}),
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
  DGPP_LOG_INFO(
      "serve: tool calls {}; tool_choice/parallel_tool_calls {}; reasoning {}",
      markers_.tool_calls_available()
          ? "available (the template's markers are in the tokenizer)"
          : "unavailable (no tool-call markers in this tokenizer)",
      constraints_available() ? "enforced by constrained decoding"
                              : "not enforceable (no masks on this engine)",
      !markers_.reasoning_available()
          ? "not split (no </think> marker)"
          : cfg_.reasoning_in_content ? "folded into content"
                                      : "on reasoning_content");
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
// The chat request's conversation and tools (M6 6f)
// ---------------------------------------------------------------------------

bool GenerationService::parse_chat(const dgpp::minijson::Value& body,
                                   HttpResponseWriter& w, ChatPlan* plan) {
  using dgpp::minijson::Member;
  using dgpp::minijson::Value;
  const auto refuse = [&](const std::string& message, const std::string& param,
                          const std::string& code = "") {
    respond_error(w, 400, message, "invalid_request_error", param, code);
    return false;
  };

  const Value* messages = body.find("messages");
  if (messages == nullptr || !messages->is_array() ||
      messages->items().empty())
    return refuse("messages is required and must be a non-empty array",
                  "messages");

  // ---- tools --------------------------------------------------------------
  const Value* tools = body.find("tools");
  std::vector<std::string> tool_names;
  if (tools != nullptr) {
    if (!tools->is_array()) return refuse("tools must be an array", "tools");
    for (size_t i = 0; i < tools->items().size(); ++i) {
      const Value& t = tools->items()[i];
      const std::string where = "tools[" + std::to_string(i) + "]";
      if (!t.is_object()) return refuse(where + " must be an object", where);
      const Value* type = t.find("type");
      if (type != nullptr &&
          (!type->is_string() || type->as_string() != "function"))
        return refuse(where + ".type must be \"function\"", where + ".type");
      const Value* fn = t.find("function");
      if (fn != nullptr && !fn->is_object())
        return refuse(where + ".function must be an object",
                      where + ".function");
      const Value& def = fn != nullptr ? *fn : t;
      const Value* name = def.find("name");
      if (name == nullptr || !name->is_string() || name->as_string().empty())
        return refuse(where + ".function.name is required and must be a "
                      "non-empty string",
                      where + ".function.name");
      const Value* params = def.find("parameters");
      if (params != nullptr && !params->is_object())
        return refuse(where + ".function.parameters must be an object (a "
                      "JSON schema)",
                      where + ".function.parameters");
      tool_names.emplace_back(name->as_string());
    }
  }
  const bool have_tools = tools != nullptr && !tools->items().empty();
  if (have_tools && !tool_calls_available())
    return refuse(
        "tools cannot be served: this model's tokenizer has no tool-call "
        "markers, so the service could not parse the model's calls",
        "tools", "tools_unsupported");

  // ---- tool_choice --------------------------------------------------------
  enum class Choice { kAuto, kNone, kRequired, kNamed };
  Choice choice = Choice::kAuto;
  std::string named;
  if (const Value* tc = body.find("tool_choice")) {
    if (!have_tools)
      return refuse("tool_choice requires a non-empty tools array",
                    "tool_choice");
    if (tc->is_string()) {
      const std::string_view s = tc->as_string();
      if (s == "auto")
        choice = Choice::kAuto;
      else if (s == "none")
        choice = Choice::kNone;
      else if (s == "required")
        choice = Choice::kRequired;
      else
        return refuse("tool_choice must be \"auto\", \"none\", \"required\" "
                      "or {\"type\": \"function\", \"function\": {\"name\": "
                      "…}}",
                      "tool_choice");
    } else if (tc->is_object()) {
      const Value* type = tc->find("type");
      if (type != nullptr &&
          (!type->is_string() || type->as_string() != "function"))
        return refuse("tool_choice.type must be \"function\"",
                      "tool_choice.type");
      const Value* fn = tc->find("function");
      const Value* name = fn != nullptr && fn->is_object() ? fn->find("name")
                                                            : nullptr;
      if (name == nullptr || !name->is_string())
        return refuse("tool_choice.function.name must be a string",
                      "tool_choice.function.name");
      named = std::string(name->as_string());
      if (std::find(tool_names.begin(), tool_names.end(), named) ==
          tool_names.end())
        return refuse("tool_choice names '" + named +
                          "', which is not one of the request's tools",
                      "tool_choice.function.name");
      choice = Choice::kNamed;
    } else {
      return refuse("tool_choice must be a string or an object", "tool_choice");
    }
  }

  // ---- parallel_tool_calls ----------------------------------------------
  bool parallel = true;
  if (const Value* ptc = body.find("parallel_tool_calls")) {
    if (!have_tools)
      return refuse("parallel_tool_calls requires tools", "parallel_tool_calls");
    if (!ptc->is_bool())
      return refuse("parallel_tool_calls must be a boolean",
                    "parallel_tool_calls");
    parallel = ptc->as_bool(true);
  }

  // ---- the grammar (M6 6g) ---------------------------------------------
  // tool_choice none / required / named and parallel_tool_calls false are
  // enforced by constrained decoding: the pick's mask on every rank.
  {
    using dgpp::glm::GrammarSpec;
    GrammarSpec& g = plan->grammar;
    switch (choice) {
      case Choice::kAuto:
        // auto arms the grammar too (M6 6i): calls at will, but every call
        // well-formed — a known name, the closed keys, typed values.
        g.mode = have_tools ? GrammarSpec::Mode::kAuto : GrammarSpec::Mode::kNone;
        break;
      case Choice::kNone: g.mode = GrammarSpec::Mode::kForbidCalls; break;
      case Choice::kRequired: g.mode = GrammarSpec::Mode::kRequired; break;
      case Choice::kNamed:
        g.mode = GrammarSpec::Mode::kNamed;
        g.named = named;
        break;
    }
    g.parallel = parallel;
    if (g.active() && !constraints_available()) {
      // tool_choice none is served by leaving the tools out of the render
      // on any engine (the grammar is belt and braces), and auto with
      // parallel calls asked for no guarantee; required, named and
      // parallel_tool_calls false are guarantees only a masked pick gives.
      if (choice == Choice::kNone || (choice == Choice::kAuto && parallel)) {
        g = GrammarSpec{};
      } else {
        const char* field = choice == Choice::kAuto ? "parallel_tool_calls"
                                                    : "tool_choice";
        return refuse(
            std::string(field) +
                " cannot be enforced on this engine (it has no constrained "
                "decoding); omit it — tool_choice auto with "
                "parallel_tool_calls true is served",
            field, "constrained_decoding_unsupported");
      }
    }
    if (g.active()) {
      for (size_t i = 0; i < tools->items().size(); ++i) {
        const Value& t = tools->items()[i];
        const Value* fn = t.find("function");
        const Value& def = fn != nullptr && fn->is_object() ? *fn : t;
        // The closed key set and the typed arguments (M6 6g / 6i); a
        // strict function outside the enforceable subset is a 400 naming
        // the keyword path, a non-strict one leaves that value free.
        std::vector<std::string> warnings;
        try {
          g.tools.push_back(dgpp::glm::grammar_tool_from_function(def, &warnings));
        } catch (const std::invalid_argument& e) {
          const std::string prefix =
              "tools[" + std::to_string(i) + "]" + (fn != nullptr ? ".function." : ".");
          const std::string what = e.what();
          const size_t colon = what.find(':');
          return refuse(prefix + what,
                        prefix + (colon == std::string::npos ? what : what.substr(0, colon)),
                        "unsupported_schema");
        }
        for (const std::string& w : warnings)
          DGPP_LOG_WARN("serve: tools[{}] {}", i, w);
      }
    }
  }

  // ---- response_format (M6 6h) -------------------------------------------
  // json_object and json_schema are guarantees of the same masked pick: the
  // content is one JSON text (an object in json_object mode; conforming to
  // the schema in json_schema mode), then the turn ends. The prompt is
  // untouched; the model reasons first when the template opens thinking.
  if (const Value* rf = body.find("response_format")) {
    if (!rf->is_object())
      return refuse("response_format must be an object", "response_format");
    const Value* type = rf->find("type");
    if (type == nullptr || !type->is_string())
      return refuse("response_format.type must be a string", "response_format.type");
    const std::string_view kind = type->as_string();
    if (kind != "text" && kind != "json_object" && kind != "json_schema")
      return refuse("response_format.type must be \"text\", \"json_object\" or "
                    "\"json_schema\"",
                    "response_format.type");
    if (kind != "text") {
      if (have_tools)
        return refuse(
            "response_format json_object / json_schema cannot be combined "
            "with tools in this server version (a turn is either JSON or "
            "tool calls)",
            "response_format", "unsupported_parameter");
      if (!constraints_available())
        return refuse(
            "response_format json_object / json_schema cannot be enforced "
            "on this engine (it has no constrained decoding)",
            "response_format", "constrained_decoding_unsupported");
      using dgpp::glm::GrammarSpec;
      GrammarSpec& g = plan->grammar;
      g = GrammarSpec{};
      g.mode = GrammarSpec::Mode::kJson;
      if (kind == "json_schema") {
        const Value* js = rf->find("json_schema");
        if (js == nullptr || !js->is_object())
          return refuse("response_format.json_schema must be an object",
                        "response_format.json_schema");
        const Value* name = js->find("name");
        if (name == nullptr || !name->is_string() || name->as_string().empty())
          return refuse("response_format.json_schema.name must be a non-empty "
                        "string",
                        "response_format.json_schema.name");
        const Value* strict_v = js->find("strict");
        if (strict_v != nullptr && !strict_v->is_bool() && !strict_v->is_null())
          return refuse("response_format.json_schema.strict must be a boolean",
                        "response_format.json_schema.strict");
        const bool strict = strict_v != nullptr && strict_v->as_bool(false);
        const Value* schema = js->find("schema");
        if (schema != nullptr && !schema->is_object())
          return refuse("response_format.json_schema.schema must be an object "
                        "(a JSON schema)",
                        "response_format.json_schema.schema");
        if (schema != nullptr) {
          // Compile here to refuse loudly: the subset is type, properties,
          // required, additionalProperties, items, minItems, maxItems,
          // enum, const, anyOf. strict: anything outside it is a 400 naming
          // the keyword; otherwise the schema falls back to free JSON with a
          // warning (OpenAI's non-strict mode promises no conformance).
          try {
            dgpp::glm::compile_json_schema(*schema);
            g.json_schema = dgpp::glm::json_text_of(*schema);
          } catch (const std::invalid_argument& e) {
            const std::string what = e.what();  // "schema.<path>: reason"
            const size_t colon = what.find(':');
            const std::string where =
                "response_format.json_schema." +
                (colon == std::string::npos ? std::string("schema") : what.substr(0, colon));
            if (strict)
              return refuse("response_format.json_schema.schema cannot be enforced: " +
                                what,
                            where, "unsupported_schema");
            DGPP_LOG_WARN(
                "serve: response_format json_schema '{}' is not strict and "
                "its schema is outside the enforceable subset ({}); serving "
                "it as json_object",
                std::string(name->as_string()), what);
            g.json_schema.clear();
          }
        }
      }
    }
  }

  // ---- reasoning_effort / chat_template_kwargs --------------------------
  std::optional<std::string> effort;
  const auto effort_ok = [](std::string_view s) {
    return s == "minimal" || s == "low" || s == "medium" || s == "high";
  };
  if (const Value* re = body.find("reasoning_effort")) {
    if (!re->is_string() || !effort_ok(re->as_string()))
      return refuse("reasoning_effort must be one of \"minimal\", \"low\", "
                    "\"medium\", \"high\" (this template renders low and "
                    "high; anything else is its maximum)",
                    "reasoning_effort");
    effort = std::string(re->as_string());
  }
  std::vector<Member> extra;
  if (const Value* kw = body.find("chat_template_kwargs")) {
    if (!kw->is_object())
      return refuse("chat_template_kwargs must be an object",
                    "chat_template_kwargs");
    for (const Member& m : kw->members()) {
      const std::string where = "chat_template_kwargs." + m.key;
      if (m.key == "clear_thinking") {
        if (!m.value.is_bool()) return refuse(where + " must be a boolean", where);
        extra.push_back(m);
      } else if (m.key == "reasoning_effort") {
        if (!m.value.is_string() || !effort_ok(m.value.as_string()))
          return refuse(where + " must be one of \"minimal\", \"low\", "
                        "\"medium\", \"high\"",
                        where);
        if (effort.has_value() && *effort != m.value.as_string())
          return refuse("reasoning_effort and chat_template_kwargs."
                        "reasoning_effort disagree; send one",
                        where);
        effort = std::string(m.value.as_string());
      } else if (m.key == "enable_thinking") {
        return refuse(
            "this template has no enable_thinking knob — thinking is always "
            "on (the generation prompt opens <think>); use reasoning_effort",
            where, "unsupported_parameter");
      } else {
        return refuse(where + " is not a knob of this template (it reads "
                      "clear_thinking and reasoning_effort)",
                      where, "unsupported_parameter");
      }
    }
  }

  // ---- messages: validate and normalize ----------------------------------
  std::vector<Value> msgs;
  msgs.reserve(messages->items().size());
  for (size_t i = 0; i < messages->items().size(); ++i) {
    const Value& msg = messages->items()[i];
    const std::string where = "messages[" + std::to_string(i) + "]";
    if (!msg.is_object()) return refuse(where + " must be an object", where);
    const Value* role = msg.find("role");
    if (role == nullptr || !role->is_string())
      return refuse(where + ".role is required and must be a string",
                    where + ".role");
    const std::string_view r = role->as_string();
    if (r != "system" && r != "user" && r != "assistant" && r != "tool")
      return refuse(where + ".role '" + std::string(r) +
                        "' is not one this template renders (system, user, "
                        "assistant, tool)",
                    where + ".role");
    const Value* content = msg.find("content");
    std::vector<Member> members;
    members.reserve(msg.members().size() + 1);
    bool content_written = false;
    if (r == "assistant") {
      const Value* tcs = msg.find("tool_calls");
      std::vector<Value> calls;
      if (tcs != nullptr) {
        if (!tcs->is_array())
          return refuse(where + ".tool_calls must be an array",
                        where + ".tool_calls");
        for (size_t j = 0; j < tcs->items().size(); ++j) {
          const Value& tc = tcs->items()[j];
          const std::string at = where + ".tool_calls[" + std::to_string(j) + "]";
          if (!tc.is_object()) return refuse(at + " must be an object", at);
          const Value* type = tc.find("type");
          if (type != nullptr &&
              (!type->is_string() || type->as_string() != "function"))
            return refuse(at + ".type must be \"function\"", at + ".type");
          const Value* id = tc.find("id");
          if (id != nullptr && !id->is_string())
            return refuse(at + ".id must be a string", at + ".id");
          const Value* fn = tc.find("function");
          if (fn == nullptr || !fn->is_object())
            return refuse(at + ".function must be an object", at + ".function");
          const Value* name = fn->find("name");
          if (name == nullptr || !name->is_string())
            return refuse(at + ".function.name must be a string",
                          at + ".function.name");
          // arguments: the OpenAI wire form is a JSON-object STRING; the
          // template iterates a mapping — parse it here (vLLM's
          // postprocess does the same); an object passes through.
          Value args_value;
          if (const Value* args = fn->find("arguments")) {
            if (args->is_object()) {
              args_value = *args;
            } else if (args->is_string()) {
              try {
                dgpp::minijson::ParseResult pr =
                    dgpp::minijson::parse(args->as_string());
                if (!pr.root.is_object()) throw std::runtime_error("not an object");
                args_value = std::move(pr.root);
              } catch (const std::exception& e) {
                return refuse(at + ".function.arguments must be a JSON object "
                              "or the string of one (" + e.what() + ")",
                              at + ".function.arguments");
              }
            } else {
              return refuse(at + ".function.arguments must be a JSON object "
                            "or the string of one",
                            at + ".function.arguments");
            }
          } else {
            args_value = Value::make_object({});
          }
          std::vector<Member> fn_members;
          for (const Member& fm : fn->members())
            if (fm.key != "arguments") fn_members.push_back(fm);
          fn_members.push_back(Member{"arguments", std::move(args_value)});
          std::vector<Member> tc_members;
          for (const Member& tm : tc.members())
            if (tm.key != "function") tc_members.push_back(tm);
          tc_members.push_back(
              Member{"function", Value::make_object(std::move(fn_members))});
          calls.push_back(Value::make_object(std::move(tc_members)));
        }
      }
      const bool has_calls = !calls.empty();
      if (content == nullptr || content->is_null()) {
        if (!has_calls)
          return refuse(where + ".content is required for an assistant "
                        "message without tool_calls",
                        where + ".content");
      } else if (!content_form_ok(*content)) {
        return refuse(where + ".content must be a string or an array of "
                      "content parts",
                      where + ".content");
      }
      if (const Value* rc = msg.find("reasoning_content"))
        if (!rc->is_string() && !rc->is_null())
          return refuse(where + ".reasoning_content must be a string",
                        where + ".reasoning_content");
      for (const Member& m : msg.members()) {
        if (m.key == "content") {
          members.push_back(Member{"content", (content->is_null()
                                                   ? Value::make_string("")
                                                   : *content)});
          content_written = true;
        } else if (m.key == "tool_calls") {
          if (has_calls)
            members.push_back(Member{"tool_calls", Value::make_array(calls)});
        } else if (m.key == "reasoning_content" && m.value.is_null()) {
          // a null reasoning_content is "none" — the template tests
          // `is string`, so drop it
        } else {
          members.push_back(m);
        }
      }
      if (!content_written)
        members.push_back(Member{"content", Value::make_string("")});
    } else if (r == "tool") {
      if (content == nullptr || !(content->is_string() || content->is_array()))
        return refuse(where + ".content is required for a tool message: a "
                      "string, or the template's list of outputs / tool "
                      "references",
                      where + ".content");
      if (const Value* id = msg.find("tool_call_id"))
        if (!id->is_string())
          return refuse(where + ".tool_call_id must be a string",
                        where + ".tool_call_id");
      for (const Member& m : msg.members()) members.push_back(m);
    } else {
      if (content == nullptr || !content_form_ok(*content))
        return refuse(where + ".content is required and must be a string or "
                      "an array of content parts",
                      where + ".content");
      for (const Member& m : msg.members()) members.push_back(m);
    }
    msgs.push_back(Value::make_object(std::move(members)));
  }

  // ---- the template globals ----------------------------------------------
  std::vector<Member> globals;
  globals.push_back(Member{"messages", Value::make_array(std::move(msgs))});
  if (have_tools && choice != Choice::kNone)
    globals.push_back(Member{"tools", *tools});
  if (effort.has_value())
    globals.push_back(
        Member{"reasoning_effort", Value::make_owned_string(*effort)});
  for (Member& m : extra) globals.push_back(std::move(m));
  plan->globals = Value::make_object(std::move(globals));
  plan->tools_requested = have_tools;
  if (have_tools) plan->schemas = dgpp::glm::ToolSchemas(*tools);
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
      "stop",       "n",           "logit_bias", "user",
      "store",      "metadata",    "service_tier", "prediction",
      "audio",      "modalities",  "web_search_options", "functions",
      "function_call",
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

  // The conversation, the tools and the template knobs (M6 6f).
  ChatPlan plan;
  if (!parse_chat(body, w, &plan)) return;

  // The prompt: render the chat template over the globals, encode. A
  // constrained tool_choice changes nothing here — the grammar rides the
  // scheduler request and masks the pick on every rank (M6 6g).
  std::vector<int64_t> prompt;
  std::string rendered;
  try {
    rendered = frontend_->render_chat(plan.globals);
    prompt = frontend_->encode_text(rendered);
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
  const bool opens_thinking = markers_.think_open.available() &&
                              prompt.back() == markers_.think_open.id;
  ToolCallParser::Options popts;
  popts.start_in_reasoning = opens_thinking;

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
  record->chat = true;
  record->stream = stream;
  record->include_usage = include_usage;
  record->prompt_tokens = static_cast<int>(prompt.size());
  record->parser = std::make_unique<ToolCallParser>(
      markers_,
      [this](const std::vector<int64_t>& ids) {
        return frontend_->decode_ids(ids);
      },
      std::move(plan.schemas), std::move(popts));
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
  sr.grammar = std::move(plan.grammar);
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
  record->chat = false;
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
                               cfg_.sampling_defaults, tool_calls_available(),
                               constraints_available(),
                               cfg_.reasoning_in_content) +
                  "]}");
    return;
  }
  const std::string id = req.path.substr(std::string("/v1/models/").size());
  if (id == cfg_.model_id) {
    w.respond(200, "application/json",
              model_object(id, created, sampling_available_,
                           cfg_.sampling_defaults, tool_calls_available(),
                           constraints_available(), cfg_.reasoning_in_content));
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
  out.append(",\"tool_calls_out\":");
  append_json_int(&out, static_cast<int64_t>(st.tool_calls_out));
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

void GenerationService::absorb(StreamRecord& r, ParserEvent ev) {
  // The fold knob: reasoning rides as content text, the model's own
  // "</think>" included where it produced it — exactly the decode a
  // client of a server without a reasoning parser would see.
  if (cfg_.reasoning_in_content) {
    if (ev.kind == ParserEvent::Kind::kReasoning) {
      ev.kind = ParserEvent::Kind::kContent;
    } else if (ev.kind == ParserEvent::Kind::kReasoningClosed) {
      ev.kind = ParserEvent::Kind::kContent;
      ev.text = markers_.think_close.text;
    }
  }
  switch (ev.kind) {
    case ParserEvent::Kind::kReasoning:
      r.reasoning += ev.text;
      break;
    case ParserEvent::Kind::kContent:
      if (ev.text.empty()) return;
      r.content += ev.text;
      break;
    case ParserEvent::Kind::kToolCall:
      r.calls.push_back(ev.call);
      stats_.tool_calls_out++;
      break;
    case ParserEvent::Kind::kReasoningClosed:
      return;  // structural only
  }
  if (r.stream) r.pending.push_back(std::move(ev));
}

void GenerationService::on_token(const std::string& id, int64_t token,
                                 int steps_done) {
  (void)steps_done;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& r : records_) {
      if (r->id != id || r->done) continue;
      r->ids.push_back(token);
      if (r->chat) {
        // The parser routes the id by state (reasoning / content / a tool
        // call block) and yields the exact text deltas of each run.
        std::vector<ParserEvent> events;
        r->parser->feed(token, &events);
        for (ParserEvent& ev : events) absorb(*r, std::move(ev));
      } else {
        // Exact incremental text: the suffix diff of successive full
        // decodes — UTF-8 splits and special tokens come out right by
        // construction (the tokenizer's own decode gates, pinned by its
        // differential goldens).
        const std::string full = frontend_->decode_ids(r->ids);
        if (full.size() > r->text.size())
          r->delta.append(full, r->text.size(), std::string::npos);
        r->text = full;
      }
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
      if (r->chat && r->parser && !r->done) {
        // An open block at the end flushes as content.
        std::vector<ParserEvent> events;
        r->parser->finish(&events);
        for (ParserEvent& ev : events) absorb(*r, std::move(ev));
      }
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

void GenerationService::flush_chat_stream(StreamRecord& r) {
  // Take the unflushed events (and, when a chunk will carry them, the
  // logprobs entries since the last flush) under the lock; format and
  // write outside it.
  std::vector<ParserEvent> events;
  std::string lp_json;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    events.swap(r.pending);
    if (!events.empty() && r.logprobs >= 0 && r.lps.size() > r.lps_flushed) {
      lp_json = logprobs_content(r, r.lps_flushed, r.lps.size());
      r.lps_flushed = r.lps.size();
    }
  }
  if (!r.first_chunk_sent) {
    r.first_chunk_sent = true;
    r.writer->write_event(chat_chunk_first(r.id, r.created_unix, r.model));
  }
  for (const ParserEvent& ev : events) {
    std::string delta;
    switch (ev.kind) {
      case ParserEvent::Kind::kReasoning:
        delta = delta_text("reasoning_content", ev.text);
        break;
      case ParserEvent::Kind::kContent:
        delta = delta_text("content", ev.text);
        break;
      case ParserEvent::Kind::kToolCall: {
        const int index = r.calls_announced++;
        const std::string call_id = tool_call_id(r.tag, index);
        r.writer->write_event(chat_chunk_delta(
            r.id, r.created_unix, r.model,
            delta_tool_call_start(index, call_id, ev.call.name), lp_json));
        lp_json.clear();
        delta = delta_tool_call_arguments(index, ev.call.arguments);
        break;
      }
      case ParserEvent::Kind::kReasoningClosed:
        continue;
    }
    r.writer->write_event(
        chat_chunk_delta(r.id, r.created_unix, r.model, delta, lp_json));
    lp_json.clear();
  }
}

void GenerationService::flush_legacy_stream(StreamRecord& r) {
  std::string delta;
  std::string lp_json;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    delta.swap(r.delta);
    if (!delta.empty() && r.logprobs >= 0 && r.lps.size() > r.lps_flushed) {
      lp_json = logprobs_content(r, r.lps_flushed, r.lps.size());
      r.lps_flushed = r.lps.size();
    }
  }
  if (!r.first_chunk_sent) {
    r.first_chunk_sent = true;
    r.writer->write_event(text_chunk_first(r.id, r.created_unix, r.model));
  }
  if (!delta.empty())
    r.writer->write_event(
        text_chunk_delta(r.id, r.created_unix, r.model, delta, lp_json));
}

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
    if (r->chat)
      flush_chat_stream(*r);
    else
      flush_legacy_stream(*r);
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
        // Flush any straggler events, then the terminal sequence. The
        // final chunk carries the logprobs entries no content chunk took
        // (the EOS pick decodes to nothing, so its entry lands here).
        if (r->chat)
          flush_chat_stream(*r);
        else
          flush_legacy_stream(*r);
        std::string lp_json;
        if (r->logprobs >= 0) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (r->lps.size() > r->lps_flushed) {
            lp_json = logprobs_content(*r, r->lps_flushed, r->lps.size());
            r->lps_flushed = r->lps.size();
          }
        }
        const char* finish = finish_reason(r->reason, !r->calls.empty());
        r->writer->write_event(
            r->chat ? chat_chunk_final(r->id, r->created_unix, r->model,
                                       finish, lp_json)
                    : text_chunk_final(r->id, r->created_unix, r->model,
                                       finish, lp_json));
        if (r->include_usage) {
          r->writer->write_event(chat_chunk_usage(r->id, r->created_unix,
                                                  r->model, r->prompt_tokens,
                                                  r->completion_tokens));
        }
        r->writer->write_event("[DONE]");
        r->writer->end_stream();
      } else {
        // The one-shot completion object.
        std::string lp_json;
        if (r->logprobs >= 0) {
          std::lock_guard<std::mutex> lock(mutex_);
          lp_json = r->chat ? logprobs_content(*r, 0, r->lps.size())
                            : legacy_logprobs(*r);
        }
        const char* finish = finish_reason(r->reason, !r->calls.empty());
        if (r->chat) {
          // {"role","content"(null when only calls),"reasoning_content"?,
          //  "tool_calls"?}
          std::string msg = "{\"role\":\"assistant\",\"content\":";
          if (r->content.empty() && !r->calls.empty())
            msg.append("null");
          else
            append_json_string(&msg, r->content);
          if (!r->reasoning.empty()) {
            msg.append(",\"reasoning_content\":");
            append_json_string(&msg, r->reasoning);
          }
          if (!r->calls.empty()) {
            msg.append(",\"tool_calls\":[");
            for (size_t i = 0; i < r->calls.size(); ++i) {
              if (i) msg.push_back(',');
              msg.append("{\"id\":");
              append_json_string(&msg, tool_call_id(r->tag, static_cast<int>(i)));
              msg.append(",\"type\":\"function\",\"function\":{\"name\":");
              append_json_string(&msg, r->calls[i].name);
              msg.append(",\"arguments\":");
              append_json_string(&msg, r->calls[i].arguments);
              msg.append("}}");
            }
            msg.push_back(']');
          }
          msg.push_back('}');
          r->writer->respond(
              200, "application/json",
              chat_completion_body(r->id, r->created_unix, r->model, msg,
                                   finish, r->prompt_tokens,
                                   r->completion_tokens, lp_json));
        } else {
          r->writer->respond(
              200, "application/json",
              text_completion_body(r->id, r->created_unix, r->model, r->text,
                                   finish, r->prompt_tokens,
                                   r->completion_tokens, lp_json));
        }
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
