#include "models/glm_tool_grammar.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "loaders/minijson.hpp"
#include "models/glm_tokenizer.hpp"

namespace dgpp::glm {

struct GrammarVocab::JsonHolder {
  std::once_flag once;
  std::unique_ptr<JsonTables> tables;
};

// ---------------------------------------------------------------------------
// GrammarVocab
// ---------------------------------------------------------------------------

GrammarVocab::GrammarVocab(std::vector<std::string> texts, ChatMarkers markers,
                           std::vector<int64_t> eos_ids, int vocab_size,
                           int64_t call_turn_eos)
    : json_(std::make_shared<JsonHolder>()),
      texts_(std::move(texts)),
      markers_(std::move(markers)),
      eos_(std::move(eos_ids)),
      call_eos_(call_turn_eos),
      vocab_size_(vocab_size) {
  if (vocab_size_ < 1)
    throw std::invalid_argument("GrammarVocab: vocab_size must be positive");
  for (size_t id = 0; id < texts_.size(); ++id) {
    if (texts_[id].empty() || static_cast<int64_t>(id) >= vocab_size_) continue;
    by_first_[static_cast<unsigned char>(texts_[id][0])].push_back(
        static_cast<int32_t>(id));
  }
  for (const ChatMarker* m :
       {&markers_.tool_call_open, &markers_.tool_call_close,
        &markers_.arg_key_open, &markers_.arg_key_close,
        &markers_.arg_value_open, &markers_.arg_value_close})
    if (m->available()) marker_ids_.push_back(m->id);
  if (call_eos_ < 0 && !eos_.empty()) call_eos_ = eos_[0];
}

GrammarVocab GrammarVocab::from_tokenizer(const GlmTokenizer& tok,
                                          const std::vector<int64_t>& eos_ids,
                                          int vocab_size) {
  const int64_t max_id = tok.max_id();
  std::vector<std::string> texts(
      static_cast<size_t>(std::max<int64_t>(max_id + 1, 0)));
  for (int64_t id = 0; id <= max_id; ++id)
    texts[static_cast<size_t>(id)] = tok.decode(id, /*skip_special_tokens=*/true);
  // The turn-ending id after the last call: <|observation|> when it is an
  // EOS id of this checkpoint, else the first EOS id.
  int64_t call_eos = -1;
  for (const auto& added : tok.added_tokens())
    if (added.content == "<|observation|>")
      for (const int64_t e : eos_ids)
        if (e == added.id) call_eos = e;
  return GrammarVocab(std::move(texts), ChatMarkers::from_tokenizer(tok),
                      eos_ids, vocab_size, call_eos);
}

void GrammarVocab::prepare_json() const {
  std::call_once(json_->once,
                 [&] { json_->tables = std::make_unique<JsonTables>(*this); });
}

const JsonTables& GrammarVocab::json_tables() const {
  prepare_json();
  return *json_->tables;
}

bool GrammarVocab::is_eos(int64_t id) const {
  for (const int64_t e : eos_)
    if (e == id) return true;
  return false;
}

// ---------------------------------------------------------------------------
// The tool entry from a function definition (M6 6g keys, 6i typed values)
// ---------------------------------------------------------------------------

namespace {

std::string rendered_enum_text(const minijson::Value& v) {
  // What the template writes for the value: a string raw, anything else
  // through tojson.
  return v.is_string() ? std::string(v.as_string()) : json_text_of(v);
}

bool names_string(const minijson::Value& type) {
  if (type.is_string()) return type.as_string() == "string";
  if (type.is_array())
    for (const minijson::Value& e : type.items())
      if (e.is_string() && e.as_string() == "string") return true;
  return false;
}

// The property admits a raw-text value (string-typed, untyped, or a type
// list / anyOf with such an alternative): the grammar cannot type it.
bool string_possible(const minijson::Value& prop) {
  if (const minijson::Value* any = prop.find("anyOf")) {
    if (!any->is_array()) return true;
    for (const minijson::Value& alt : any->items())
      if (!alt.is_object() || string_possible(alt)) return true;
    return false;
  }
  const minijson::Value* type = prop.find("type");
  if (type == nullptr) return true;
  if (type->is_string()) return type->as_string() == "string";
  if (type->is_array()) {
    if (type->items().empty()) return true;
    return names_string(*type);
  }
  return true;  // an unrecognised type value: leave it free
}

}  // namespace

GrammarTool grammar_tool_from_function(const minijson::Value& def,
                                       std::vector<std::string>* warnings) {
  GrammarTool tool;
  if (const minijson::Value* name = def.find("name"))
    tool.name = std::string(name->as_string());
  const minijson::Value* strict_v = def.find("strict");
  const bool strict = strict_v != nullptr && strict_v->is_bool() && strict_v->as_bool();
  tool.strict = strict;
  const minijson::Value* params = def.find("parameters");
  if (params == nullptr || !params->is_object()) return tool;
  const minijson::Value* props = params->find("properties");
  const minijson::Value* extra = params->find("additionalProperties");
  const bool closed = extra != nullptr && extra->is_bool() && !extra->as_bool(true);
  if (props == nullptr || !props->is_object()) return tool;
  // Keys close only when the schema says so (additionalProperties false —
  // JSON Schema's default is open) and declares properties.
  if (closed) {
    tool.constrain_keys = true;
    for (const minijson::Member& pm : props->members()) tool.keys.push_back(pm.key);
  }
  // `required`: the names among the declared properties (a name outside
  // them can never be satisfied — dropped, with a warning under strict).
  if (const minijson::Value* req = params->find("required");
      req != nullptr && req->is_array()) {
    for (const minijson::Value& e : req->items()) {
      if (!e.is_string()) continue;
      const std::string key(e.as_string());
      if (props->find(key) != nullptr) {
        if (std::find(tool.required_keys.begin(), tool.required_keys.end(),
                      key) == tool.required_keys.end())
          tool.required_keys.push_back(key);
      } else if (strict && warnings != nullptr) {
        warnings->push_back("required key '" + key + "' of '" + tool.name +
                            "' is not a declared property; not enforced");
      }
    }
  }
  for (const minijson::Member& pm : props->members()) {
    GrammarArg arg;
    arg.key = pm.key;
    const minijson::Value& prop = pm.value;
    const std::string path = "parameters.properties." + pm.key;
    if (!prop.is_object()) {
      if (strict) throw std::invalid_argument(path + ": must be a schema object");
      tool.args.push_back(std::move(arg));
      continue;
    }
    if (strict) {
      // Every property must lie inside the enforceable subset.
      try {
        compile_json_schema(prop);
      } catch (const std::invalid_argument& e) {
        const std::string what = e.what();  // "schema.<path>: reason"
        throw std::invalid_argument(
            path + (what.rfind("schema", 0) == 0 ? what.substr(6) : ": " + what));
      }
    }
    const minijson::Value* en = prop.find("enum");
    const minijson::Value* cs = prop.find("const");
    if (string_possible(prop)) {
      // Raw text — typable only through an enum's exact texts.
      if (en != nullptr && en->is_array() && !en->items().empty() &&
          prop.find("anyOf") == nullptr) {
        arg.kind = GrammarArg::Kind::kText;
        for (const minijson::Value& e : en->items())
          arg.texts.push_back(rendered_enum_text(e));
      } else if (cs != nullptr && prop.find("anyOf") == nullptr) {
        arg.kind = GrammarArg::Kind::kText;
        arg.texts.push_back(rendered_enum_text(*cs));
      }
      tool.args.push_back(std::move(arg));
      continue;
    }
    // A JSON-typed property: the machine under its own schema.
    try {
      compile_json_schema(prop);
      arg.kind = GrammarArg::Kind::kJson;
      arg.schema = json_text_of(prop);
    } catch (const std::invalid_argument& e) {
      if (warnings != nullptr)
        warnings->push_back("argument '" + pm.key + "' of '" + tool.name +
                            "' is outside the constrained subset (" + e.what() +
                            "); its value stays free text");
    }
    tool.args.push_back(std::move(arg));
  }
  return tool;
}

// ---------------------------------------------------------------------------
// GrammarState
// ---------------------------------------------------------------------------

GrammarState::GrammarState(const GrammarVocab* vocab, GrammarSpec spec,
                           bool prompt_opens_thinking)
    : vocab_(vocab), spec_(std::move(spec)) {
  if (!spec_.active()) return;
  if (spec_.mode == GrammarSpec::Mode::kJson) {
    if (vocab_ == nullptr || vocab_->eos_ids().empty())
      throw std::invalid_argument(
          "GrammarState: a JSON grammar needs a vocabulary with an EOS id");
    std::shared_ptr<const JsonSchema> schema;
    if (spec_.json_schema.empty()) {
      schema = std::make_shared<const JsonSchema>(json_object_schema());
    } else {
      minijson::ParseResult parsed;
      try {
        parsed = minijson::parse(spec_.json_schema);
      } catch (const std::exception& e) {
        throw std::invalid_argument(
            std::string("GrammarState: the JSON schema text does not parse: ") +
            e.what());
      }
      schema = std::make_shared<const JsonSchema>(compile_json_schema(parsed.root));
    }
    json_ = JsonMachine(std::move(schema), &vocab_->json_tables());
    state_ = prompt_opens_thinking && vocab_->markers().think_close.available()
                 ? State::kThink
                 : State::kJsonBody;
    return;
  }
  if (vocab_ == nullptr || !vocab_->usable())
    throw std::invalid_argument(
        "GrammarState: an active grammar needs a vocabulary with the "
        "tool-call markers and an EOS id");
  if (spec_.mode == GrammarSpec::Mode::kNamed) {
    bool found = false;
    for (const GrammarTool& t : spec_.tools) found = found || t.name == spec_.named;
    if (!found)
      throw std::invalid_argument(
          "GrammarState: the named function is not among the tools");
  }
  if ((spec_.mode == GrammarSpec::Mode::kRequired ||
       spec_.mode == GrammarSpec::Mode::kNamed) &&
      spec_.tools.empty())
    throw std::invalid_argument("GrammarState: a required call needs tools");
  // The typed arguments' schemas, compiled once per request.
  arg_schemas_.resize(spec_.tools.size());
  for (size_t t = 0; t < spec_.tools.size(); ++t) {
    const GrammarTool& tool = spec_.tools[t];
    arg_schemas_[t].resize(tool.args.size());
    for (size_t a = 0; a < tool.args.size(); ++a) {
      const GrammarArg& arg = tool.args[a];
      if (arg.kind != GrammarArg::Kind::kJson) continue;
      minijson::ParseResult parsed;
      try {
        parsed = minijson::parse(arg.schema);
      } catch (const std::exception& e) {
        throw std::invalid_argument("GrammarState: the schema text of argument '" +
                                    arg.key + "' of '" + tool.name +
                                    "' does not parse: " + e.what());
      }
      arg_schemas_[t][a] =
          std::make_shared<const JsonSchema>(compile_json_schema(parsed.root));
    }
  }
  state_ = prompt_opens_thinking && vocab_->markers().think_close.available()
               ? State::kThink
               : State::kTop;
}

const GrammarArg* GrammarState::current_arg() const {
  const GrammarTool* t = current_tool();
  if (t == nullptr || arg_ < 0 || arg_ >= static_cast<int>(t->args.size()))
    return nullptr;
  return &t->args[static_cast<size_t>(arg_)];
}

const char* GrammarState::state_name() const {
  if (!active()) return dead_ ? "dead" : "inactive";
  switch (state_) {
    case State::kThink: return "think";
    case State::kTop: return "top";
    case State::kName: return "name";
    case State::kKey: return "key";
    case State::kAfterKey: return "after-key";
    case State::kValue: return "value";
    case State::kAfterValue: return "after-value";
    case State::kEnd: return "end";
    case State::kDone: return "done";
    case State::kJsonBody: return json_.state_name();
  }
  return "?";
}

bool GrammarState::obligation_open() const {
  if (spec_.mode == GrammarSpec::Mode::kJson) return !json_.done();
  return (spec_.mode == GrammarSpec::Mode::kRequired ||
          spec_.mode == GrammarSpec::Mode::kNamed) &&
         calls_ == 0;
}

bool GrammarState::calls_remaining() const {
  switch (spec_.mode) {
    case GrammarSpec::Mode::kNone: return true;
    case GrammarSpec::Mode::kForbidCalls: return false;
    case GrammarSpec::Mode::kAuto: return spec_.parallel || calls_ == 0;
    case GrammarSpec::Mode::kRequired: return spec_.parallel || calls_ == 0;
    case GrammarSpec::Mode::kNamed: return calls_ == 0;
    case GrammarSpec::Mode::kJson: return false;
  }
  return false;
}

const GrammarTool* GrammarState::current_tool() const {
  return tool_ >= 0 && tool_ < static_cast<int>(spec_.tools.size())
             ? &spec_.tools[static_cast<size_t>(tool_)]
             : nullptr;
}

bool GrammarState::keys_possible() const {
  const GrammarTool* t = current_tool();
  if (t == nullptr) return true;
  if (!t->constrain_keys) return true;
  for (const std::string& k : t->keys)
    if (!key_used(k)) return true;
  return false;
}

bool GrammarState::key_used(const std::string& key) const {
  return std::find(used_keys_.begin(), used_keys_.end(), key) != used_keys_.end();
}

bool GrammarState::call_closable() const {
  const GrammarTool* t = current_tool();
  if (t == nullptr || !t->strict) return true;
  for (const std::string& k : t->required_keys)
    if (!key_used(k)) return false;
  return true;
}

bool GrammarState::call_closable_for(const std::string& name) const {
  // At the name, before any key: closable unless a strict tool requires one.
  for (const GrammarTool& t : spec_.tools)
    if (t.name == name) return !t.strict || t.required_keys.empty();
  return true;
}

void GrammarState::enter(State s) {
  state_ = s;
  match_ = TextMatch{};
  if (s == State::kName) {
    if (spec_.mode == GrammarSpec::Mode::kNamed) {
      match_.targets.push_back(spec_.named);
    } else {
      for (const GrammarTool& t : spec_.tools) match_.targets.push_back(t.name);
    }
    tool_ = -1;
  } else if (s == State::kKey) {
    // A closed key set, less the keys this call has already used.
    const GrammarTool* t = current_tool();
    if (t != nullptr && t->constrain_keys)
      for (const std::string& k : t->keys)
        if (!key_used(k)) match_.targets.push_back(k);
  } else if (s == State::kValue) {
    // The value's constraint is the key's declared argument, if any.
    arg_ = -1;
    const GrammarTool* t = current_tool();
    if (t != nullptr)
      for (size_t i = 0; i < t->args.size(); ++i)
        if (t->args[i].key == key_) arg_ = static_cast<int>(i);
    const GrammarArg* a = current_arg();
    if (a != nullptr && a->kind == GrammarArg::Kind::kText) {
      match_.targets = a->texts;
    } else if (a != nullptr && a->kind == GrammarArg::Kind::kJson) {
      value_json_ = JsonMachine(
          arg_schemas_[static_cast<size_t>(tool_)][static_cast<size_t>(arg_)],
          &vocab_->json_tables());
    }
  } else if (s == State::kTop) {
    tool_ = -1;
  }
}

std::vector<int64_t> GrammarState::match_ids(const TextMatch& m,
                                             int64_t closer) const {
  std::vector<int64_t> out;
  for (const std::string& target : m.targets) {
    if (target.size() <= m.emitted.size()) continue;
    if (target.compare(0, m.emitted.size(), m.emitted) != 0) continue;
    const size_t remaining = target.size() - m.emitted.size();
    const unsigned char b = static_cast<unsigned char>(target[m.emitted.size()]);
    for (const int32_t id : vocab_->ids_starting_with(b)) {
      const std::string& text = vocab_->text(id);
      if (text.size() <= remaining &&
          target.compare(m.emitted.size(), text.size(), text) == 0)
        out.push_back(id);
    }
  }
  if (m.complete() && closer >= 0) out.push_back(closer);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

void GrammarState::free_mask(TokenMask* out, int64_t extra_allowed,
                             bool forbid_markers) const {
  const int vocab = vocab_->vocab_size();
  out->vocab = vocab;
  out->words.assign(static_cast<size_t>(TokenMask::words_for(vocab)), 0xffffffffu);
  int allowed = vocab;
  // The pad bits past the vocabulary are cleared for tidiness (never read).
  if (vocab % 32 != 0)
    out->words.back() &= (1u << (vocab % 32)) - 1u;
  const auto clear = [&](int64_t id) {
    if (id < 0 || id >= vocab || id == extra_allowed) return;
    uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
    const uint32_t bit = 1u << (id & 31);
    if (w & bit) {
      w &= ~bit;
      --allowed;
    }
  };
  if (forbid_markers)
    for (const int64_t m : vocab_->marker_ids()) clear(m);
  if (obligation_open())
    for (const int64_t e : vocab_->eos_ids()) clear(e);
  out->allowed = allowed;
}

void GrammarState::list_mask(TokenMask* out,
                             const std::vector<int64_t>& ids) const {
  const int vocab = vocab_->vocab_size();
  out->vocab = vocab;
  out->words.assign(static_cast<size_t>(TokenMask::words_for(vocab)), 0u);
  int allowed = 0;
  for (const int64_t id : ids) {
    if (id < 0 || id >= vocab) continue;
    uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
    const uint32_t bit = 1u << (id & 31);
    if (!(w & bit)) {
      w |= bit;
      ++allowed;
    }
  }
  out->allowed = allowed;
  if (allowed == 0)
    throw std::logic_error("GrammarState: a position with no allowed id");
}

void GrammarState::mask(TokenMask* out) const {
  out->allowed = 0;
  if (!active()) return;
  const ChatMarkers& m = vocab_->markers();
  switch (state_) {
    case State::kThink:
      // Free (the markers included — reasoning is the model's), but the
      // turn may not end while a call is owed.
      if (!obligation_open()) return;  // unconstrained
      free_mask(out, /*extra_allowed=*/-1, /*forbid_markers=*/false);
      return;
    case State::kTop: {
      const bool free_top = spec_.mode == GrammarSpec::Mode::kAuto ||
                            spec_.mode == GrammarSpec::Mode::kForbidCalls;
      if (free_top) {
        free_mask(out, calls_remaining() ? m.tool_call_open.id : -1);
        return;
      }
      std::vector<int64_t> ids;
      if (calls_remaining()) ids.push_back(m.tool_call_open.id);
      if (!obligation_open()) ids.push_back(vocab_->call_turn_eos());
      list_mask(out, ids);
      return;
    }
    case State::kName: {
      std::vector<int64_t> ids = match_ids(match_, -1);
      if (match_.complete()) {
        if (keys_possible_for(match_.emitted)) ids.push_back(m.arg_key_open.id);
        if (call_closable_for(match_.emitted)) ids.push_back(m.tool_call_close.id);
      }
      list_mask(out, ids);
      return;
    }
    case State::kKey: {
      const GrammarTool* t = current_tool();
      if (t != nullptr && t->constrain_keys) {
        list_mask(out, match_ids(match_, m.arg_key_close.id));
        return;
      }
      free_mask(out, m.arg_key_close.id);
      return;
    }
    case State::kAfterKey:
      list_mask(out, {m.arg_value_open.id});
      return;
    case State::kValue: {
      const GrammarArg* a = current_arg();
      if (a == nullptr || a->kind == GrammarArg::Kind::kFree) {
        free_mask(out, m.arg_value_close.id);
      } else if (a->kind == GrammarArg::Kind::kText) {
        list_mask(out, match_ids(match_, m.arg_value_close.id));
      } else {
        json_mask(value_json_, m.arg_value_close.id, out);
      }
      return;
    }
    case State::kAfterValue: {
      // A closed set with every key used has every required key used, so
      // the two conditions are never both false: no dead end.
      std::vector<int64_t> ids;
      if (keys_possible()) ids.push_back(m.arg_key_open.id);
      if (call_closable()) ids.push_back(m.tool_call_close.id);
      list_mask(out, ids);
      return;
    }
    case State::kEnd:
    case State::kDone:
      list_mask(out, {vocab_->call_turn_eos()});
      return;
    case State::kJsonBody:
      json_mask(json_, /*closer=*/-1, out);
      return;
  }
}

// A JSON machine's position: its mask over token texts, never a marker (a
// "<think>" would be legal string content), and the closer — EOS for the
// body, </arg_value> for a typed argument — once the text is complete.
void GrammarState::json_mask(const JsonMachine& machine, int64_t closer,
                             TokenMask* out) const {
  machine.mask(*vocab_, out);
  const int vocab = vocab_->vocab_size();
  const auto set = [&](int64_t id, bool on) {
    if (id < 0 || id >= vocab) return;
    uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
    const uint32_t bit = 1u << (id & 31);
    if (on && !(w & bit)) {
      w |= bit;
      ++out->allowed;
    } else if (!on && (w & bit)) {
      w &= ~bit;
      --out->allowed;
    }
  };
  for (const int64_t m : vocab_->marker_ids()) set(m, false);
  set(vocab_->markers().think_open.id, false);
  set(vocab_->markers().think_close.id, false);
  if (machine.done()) {
    if (closer < 0)
      for (const int64_t e : vocab_->eos_ids()) set(e, true);
    else
      set(closer, true);
  }
  if (out->allowed == 0)
    throw std::logic_error("GrammarState: a JSON position with no allowed id");
}

bool GrammarState::json_allows(const JsonMachine& machine, int64_t closer,
                               int64_t id) const {
  if (closer < 0 ? vocab_->is_eos(id) : id == closer) return machine.done();
  if (vocab_->is_eos(id)) return false;
  for (const int64_t m : vocab_->marker_ids())
    if (m == id) return false;
  if (id == vocab_->markers().think_open.id || id == vocab_->markers().think_close.id)
    return false;
  return machine.allows(*vocab_, id);
}

bool GrammarState::keys_possible_for(const std::string& name) const {
  for (const GrammarTool& t : spec_.tools)
    if (t.name == name) return !t.constrain_keys || !t.keys.empty();
  return true;
}

bool GrammarState::allows(int64_t id) const {
  if (!active()) return true;
  if (state_ == State::kJsonBody) return json_allows(json_, -1, id);
  if (state_ == State::kValue) {
    const GrammarArg* a = current_arg();
    if (a != nullptr && a->kind == GrammarArg::Kind::kJson)
      return json_allows(value_json_, vocab_->markers().arg_value_close.id, id);
  }
  TokenMask m;
  mask(&m);
  return m.allows(id);
}

void GrammarState::advance(int64_t id) {
  if (!active()) return;
  if (!allows(id)) {
    dead_ = true;
    return;
  }
  const ChatMarkers& m = vocab_->markers();
  const auto after_call = [&] {
    ++calls_;
    if (calls_remaining())
      enter(State::kTop);
    else
      enter(State::kEnd);
  };
  switch (state_) {
    case State::kThink:
      if (id == m.think_close.id)
        enter(spec_.mode == GrammarSpec::Mode::kJson ? State::kJsonBody
                                                     : State::kTop);
      return;
    case State::kJsonBody:
      if (vocab_->is_eos(id)) {
        state_ = State::kDone;
        return;
      }
      for (const char c : vocab_->text(id))
        if (!json_.feed(static_cast<uint8_t>(c))) {
          dead_ = true;
          return;
        }
      return;
    case State::kTop:
      if (id == m.tool_call_open.id) {
        enter(State::kName);
      } else if (vocab_->is_eos(id)) {
        state_ = State::kDone;
      }
      return;
    case State::kName:
      if (id == m.arg_key_open.id || id == m.tool_call_close.id) {
        // The name is complete: bind the tool; the call's key ledger opens.
        tool_ = -1;
        used_keys_.clear();
        for (size_t i = 0; i < spec_.tools.size(); ++i)
          if (spec_.tools[i].name == match_.emitted) tool_ = static_cast<int>(i);
        if (id == m.arg_key_open.id)
          enter(State::kKey);
        else
          after_call();
        return;
      }
      match_.emitted += vocab_->text(id);
      return;
    case State::kKey:
      if (id == m.arg_key_close.id) {
        key_ = match_.emitted;
        used_keys_.push_back(key_);
        enter(State::kAfterKey);
        return;
      }
      match_.emitted += vocab_->text(id);
      return;
    case State::kAfterKey:
      enter(State::kValue);
      return;
    case State::kValue: {
      if (id == m.arg_value_close.id) {
        enter(State::kAfterValue);
        return;
      }
      const GrammarArg* a = current_arg();
      if (a != nullptr && a->kind == GrammarArg::Kind::kText) {
        match_.emitted += vocab_->text(id);
      } else if (a != nullptr && a->kind == GrammarArg::Kind::kJson) {
        for (const char c : vocab_->text(id))
          if (!value_json_.feed(static_cast<uint8_t>(c))) {
            dead_ = true;
            return;
          }
      }
      return;
    }
    case State::kAfterValue:
      if (id == m.arg_key_open.id)
        enter(State::kKey);
      else
        after_call();
      return;
    case State::kEnd:
      state_ = State::kDone;
      return;
    case State::kDone:
      return;
  }
}

}  // namespace dgpp::glm
