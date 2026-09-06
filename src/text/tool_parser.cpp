#include "text/tool_parser.hpp"

#include <utility>

#include "text/chat_template.hpp"
#include "text/tokenizer.hpp"

namespace dgpp::text {

// ---------------------------------------------------------------------------
// Markers
// ---------------------------------------------------------------------------

ChatMarkers ChatMarkers::from_tokenizer(const Tokenizer& tok) {
  ChatMarkers m;
  const auto lookup = [&](const char* text) {
    ChatMarker out;
    out.text = text;
    for (const auto& added : tok.added_tokens())
      if (added.content == text) {
        out.id = added.id;
        break;
      }
    return out;
  };
  m.think_open = lookup("<think>");
  m.think_close = lookup("</think>");
  m.tool_call_open = lookup("<tool_call>");
  m.tool_call_close = lookup("</tool_call>");
  m.arg_key_open = lookup("<arg_key>");
  m.arg_key_close = lookup("</arg_key>");
  m.arg_value_open = lookup("<arg_value>");
  m.arg_value_close = lookup("</arg_value>");
  for (const char* role : {"<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"}) {
    const ChatMarker r = lookup(role);
    if (r.available()) m.role_markers.push_back(r);
  }
  return m;
}

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

ToolSchemas::ToolSchemas(const minijson::Value& tools) {
  for (const minijson::Value& entry : tools.items()) {
    if (!entry.is_object()) continue;
    const minijson::Value* fn = entry.find("function");
    const minijson::Value& tool = fn != nullptr && fn->is_object() ? *fn : entry;
    const minijson::Value* name = tool.find("name");
    if (name == nullptr || !name->is_string()) continue;
    std::map<std::string, Type>& params = types_[std::string(name->as_string())];
    const minijson::Value* parameters = tool.find("parameters");
    if (parameters == nullptr || !parameters->is_object()) continue;
    const minijson::Value* properties = parameters->find("properties");
    if (properties == nullptr || !properties->is_object()) continue;
    for (const minijson::Member& prop : properties->members()) {
      Type t = Type::kUnknown;
      if (prop.value.is_object()) {
        const minijson::Value* type = prop.value.find("type");
        if (type != nullptr && type->is_string()) {
          const std::string_view ty = type->as_string();
          if (ty == "string")
            t = Type::kString;
          else if (ty == "integer" || ty == "number" || ty == "boolean" ||
                   ty == "object" || ty == "array" || ty == "null")
            t = Type::kJson;
        }
      }
      params[prop.key] = t;
    }
  }
}

ToolSchemas::Type ToolSchemas::type_of(const std::string& function,
                                     const std::string& key) const {
  const auto fn = types_.find(function);
  if (fn == types_.end()) return Type::kUnknown;
  const auto param = fn->second.find(key);
  return param == fn->second.end() ? Type::kUnknown : param->second;
}

// ---------------------------------------------------------------------------
// The parser
// ---------------------------------------------------------------------------

ToolCallParser::ToolCallParser(ChatMarkers markers, Decode decode,
                               ToolSchemas schemas, Options options)
    : markers_(std::move(markers)),
      decode_(std::move(decode)),
      schemas_(std::move(schemas)),
      options_(std::move(options)) {
  if (options_.start_in_tool_call && markers_.tool_calls_available()) {
    state_ = State::kToolCall;
    sub_ = Sub::kName;
    seeded_name_ = options_.seeded_name;
    raw_has_prefix_ = false;
  } else if (options_.start_in_reasoning && markers_.reasoning_available()) {
    state_ = State::kReasoning;
  } else {
    state_ = State::kContent;
  }
}

void ToolCallParser::run_append(Run* run, int64_t id, Event::Kind kind,
                                std::vector<Event>* out) {
  run->ids.push_back(id);
  // Exact incremental text: the suffix diff of successive full decodes of
  // the run — UTF-8 splits and skipped special tokens come out right by
  // construction (the tokenizer's own decode, pinned by its goldens).
  const std::string full = decode_(run->ids);
  if (full.size() > run->text.size()) {
    Event ev;
    ev.kind = kind;
    ev.text.assign(full, run->text.size(), std::string::npos);
    out->push_back(std::move(ev));
  }
  run->text = full;
}

void ToolCallParser::enter_tool_call(int64_t opening_id) {
  state_ = State::kToolCall;
  sub_ = Sub::kName;
  raw_.clear();
  raw_.push_back(opening_id);
  raw_has_prefix_ = true;
  seeded_name_.clear();
  name_ids_.clear();
  key_ids_.clear();
  value_ids_.clear();
  name_.clear();
  key_.clear();
  args_.clear();
  run_ = Run{};
}

void ToolCallParser::abort_block(std::vector<Event>* out) {
  // The block's literal text becomes content: markers decode to their
  // text (they are not special tokens), so a client sees exactly what the
  // model wrote.
  std::string text;
  if (!raw_has_prefix_) text = options_.forced_prefix_text;
  text += decode_(raw_);
  if (!text.empty()) {
    Event ev;
    ev.kind = Event::Kind::kContent;
    ev.text = std::move(text);
    out->push_back(std::move(ev));
  }
  state_ = State::kContent;
  run_ = Run{};
  raw_.clear();
}

std::string ToolCallParser::typed_value(const std::string& function,
                                        const std::string& key,
                                        const std::string& text) const {
  const ToolSchemas::Type type = schemas_.type_of(function, key);
  if (type != ToolSchemas::Type::kString) {
    // JSON when the whole text parses as one value (the template's tojson
    // of a non-string); the text itself otherwise.
    try {
      const minijson::ParseResult parsed = minijson::parse(text);
      size_t end = parsed.consumed;
      while (end < text.size() &&
             (text[end] == ' ' || text[end] == '\n' || text[end] == '\t' ||
              text[end] == '\r'))
        ++end;
      if (end == text.size())
        return Value::from_minijson(parsed.root).to_json(/*ensure_ascii=*/false);
    } catch (const std::exception&) {
      // not JSON — a string it is
    }
  }
  return Value::string_value(text).to_json(/*ensure_ascii=*/false);
}

void ToolCallParser::complete_block(std::vector<Event>* out) {
  Event ev;
  ev.kind = Event::Kind::kToolCall;
  ev.call.name = name_;
  std::string args = "{";
  for (size_t i = 0; i < args_.size(); ++i) {
    if (i) args += ", ";
    args += Value::string_value(args_[i].first).to_json(false);
    args += ": ";
    args += typed_value(name_, args_[i].first, args_[i].second);
  }
  args += "}";
  ev.call.arguments = std::move(args);
  out->push_back(std::move(ev));
  ++calls_;
  state_ = State::kContent;
  run_ = Run{};
  raw_.clear();
}

void ToolCallParser::feed(int64_t id, std::vector<Event>* out) {
  switch (state_) {
    case State::kReasoning:
      if (is_marker(id, markers_.think_close)) {
        state_ = State::kContent;
        run_ = Run{};
        Event ev;
        ev.kind = Event::Kind::kReasoningClosed;
        out->push_back(std::move(ev));
        return;
      }
      // The prompt already opened the block; a repeated opener is noise.
      if (is_marker(id, markers_.think_open)) return;
      run_append(&run_, id, Event::Kind::kReasoning, out);
      return;

    case State::kContent:
      if (is_marker(id, markers_.tool_call_open) &&
          markers_.tool_calls_available()) {
        enter_tool_call(id);
        return;
      }
      run_append(&run_, id, Event::Kind::kContent, out);
      return;

    case State::kToolCall:
      break;
  }

  // Inside a block. Every marker is structural; anything else is text of
  // the current segment. A wrong marker aborts the block (its id included
  // in the flushed text); a nested <tool_call> aborts and starts over.
  raw_.push_back(id);
  const bool open = is_marker(id, markers_.tool_call_open);
  const bool close = is_marker(id, markers_.tool_call_close);
  const bool key_open = is_marker(id, markers_.arg_key_open);
  const bool key_close = is_marker(id, markers_.arg_key_close);
  const bool value_open = is_marker(id, markers_.arg_value_open);
  const bool value_close = is_marker(id, markers_.arg_value_close);
  const bool structural =
      open || close || key_open || key_close || value_open || value_close;

  if (open) {
    raw_.pop_back();  // the nested opener belongs to the next block
    abort_block(out);
    enter_tool_call(id);
    return;
  }

  switch (sub_) {
    case Sub::kName:
      if (!structural) {
        name_ids_.push_back(id);
        return;
      }
      name_ = seeded_name_ + decode_(name_ids_);
      if (key_open) {
        sub_ = Sub::kKey;
        key_ids_.clear();
        return;
      }
      if (close) {
        complete_block(out);
        return;
      }
      abort_block(out);
      return;

    case Sub::kKey:
      if (!structural) {
        key_ids_.push_back(id);
        return;
      }
      if (key_close) {
        key_ = decode_(key_ids_);
        sub_ = Sub::kAfterKey;
        return;
      }
      abort_block(out);
      return;

    case Sub::kAfterKey:
      if (value_open) {
        sub_ = Sub::kValue;
        value_ids_.clear();
        return;
      }
      abort_block(out);
      return;

    case Sub::kValue:
      if (!structural) {
        value_ids_.push_back(id);
        return;
      }
      if (value_close) {
        args_.emplace_back(key_, decode_(value_ids_));
        sub_ = Sub::kAfterValue;
        return;
      }
      abort_block(out);
      return;

    case Sub::kAfterValue:
      if (key_open) {
        sub_ = Sub::kKey;
        key_ids_.clear();
        return;
      }
      if (close) {
        complete_block(out);
        return;
      }
      abort_block(out);
      return;
  }
}

void ToolCallParser::finish(std::vector<Event>* out) {
  if (state_ == State::kToolCall) abort_block(out);
}

}  // namespace dgpp::text
