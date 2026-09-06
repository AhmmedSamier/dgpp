#include "text/json_grammar.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>
#include <utility>

#include "text/chat_template.hpp"
#include "text/tool_grammar.hpp"

namespace dgpp::text {

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

std::string json_text_of(const minijson::Value& v) {
  return Value::from_minijson(v).to_json(/*ensure_ascii=*/false);
}

namespace {

uint32_t class_of_text(const std::string& text) {
  if (text.empty()) return 0;
  switch (text[0]) {
    case '"': return JsonSchemaNode::kString;
    case '{': return JsonSchemaNode::kObject;
    case '[': return JsonSchemaNode::kArray;
    case 't': case 'f': return JsonSchemaNode::kBoolean;
    case 'n': return JsonSchemaNode::kNull;
    default: {
      const bool integral = text.find_first_of(".eE") == std::string::npos;
      return integral ? (JsonSchemaNode::kNumber | JsonSchemaNode::kInteger)
                      : JsonSchemaNode::kNumber;
    }
  }
}

uint32_t type_bit(std::string_view t, const std::string& path) {
  if (t == "object") return JsonSchemaNode::kObject;
  if (t == "array") return JsonSchemaNode::kArray;
  if (t == "string") return JsonSchemaNode::kString;
  if (t == "number") return JsonSchemaNode::kNumber | JsonSchemaNode::kInteger;
  if (t == "integer") return JsonSchemaNode::kInteger;
  if (t == "boolean") return JsonSchemaNode::kBoolean;
  if (t == "null") return JsonSchemaNode::kNull;
  throw std::invalid_argument(path + ".type: unknown type '" + std::string(t) + "'");
}

struct Compiler {
  JsonSchema out;
  std::vector<std::string>* unenforced = nullptr;  // null: the strict compile

  int compile(const minijson::Value& v, const std::string& path) {
    if (!v.is_object())
      throw std::invalid_argument(path + ": a schema must be an object");
    const int index = static_cast<int>(out.nodes.size());
    out.nodes.emplace_back();
    JsonSchemaNode n;
    static const char* kIgnored[] = {"title", "description", "default",
                                     "examples", "$schema", "$comment",
                                     "deprecated", "readOnly", "writeOnly",
                                     "$id"};
    static const char* kSupported[] = {"type", "properties", "required",
                                       "additionalProperties", "items",
                                       "minItems", "maxItems", "enum",
                                       "const", "anyOf"};
    // Keywords that narrow a typed value without changing its shape: a
    // tool argument's schema tolerates them (recorded, never applied — the
    // integer stays an integer, its range is the model's to respect); a
    // response_format schema refuses them like any other unsupported one.
    static const char* kNarrowing[] = {
        "minimum",       "maximum",       "exclusiveMinimum", "exclusiveMaximum",
        "multipleOf",    "minLength",     "maxLength",        "pattern",
        "format",        "minProperties", "maxProperties",    "uniqueItems",
        "minContains",   "maxContains",   "contentEncoding",  "contentMediaType"};
    const auto narrowing = [&](const std::string& key) {
      for (const char* k : kNarrowing)
        if (key == k) return true;
      return false;
    };
    const auto ignored = [&](const std::string& key) {
      for (const char* k : kIgnored)
        if (key == k) return true;
      return unenforced != nullptr && narrowing(key);
    };
    for (const minijson::Member& m : v.members()) {
      bool known = ignored(m.key);
      for (const char* k : kSupported) known = known || m.key == k;
      if (!known)
        throw std::invalid_argument(path + "." + m.key +
                                    ": unsupported keyword (the constrained "
                                    "subset is type, properties, required, "
                                    "additionalProperties, items, minItems, "
                                    "maxItems, enum, const, anyOf)");
      if (unenforced != nullptr && narrowing(m.key))
        unenforced->push_back(path + "." + m.key);
    }
    if (const minijson::Value* any = v.find("anyOf")) {
      if (!any->is_array() || any->items().empty())
        throw std::invalid_argument(path + ".anyOf: must be a non-empty array");
      for (const minijson::Member& m : v.members())
        if (m.key != "anyOf" && !ignored(m.key))
          throw std::invalid_argument(path + "." + m.key +
                                      ": cannot be combined with anyOf here");
      for (size_t i = 0; i < any->items().size(); ++i)
        n.any_of.push_back(
            compile(any->items()[i], path + ".anyOf[" + std::to_string(i) + "]"));
      out.nodes[static_cast<size_t>(index)] = std::move(n);
      return index;
    }
    bool typed = false;
    if (const minijson::Value* t = v.find("type")) {
      typed = true;
      n.types = 0;
      if (t->is_string()) {
        n.types = type_bit(t->as_string(), path);
      } else if (t->is_array()) {
        for (const minijson::Value& e : t->items()) {
          if (!e.is_string())
            throw std::invalid_argument(path + ".type: entries must be strings");
          n.types |= type_bit(e.as_string(), path);
        }
        if (n.types == 0)
          throw std::invalid_argument(path + ".type: empty type list");
      } else {
        throw std::invalid_argument(path + ".type: must be a string or a list");
      }
    }
    if (const minijson::Value* props = v.find("properties")) {
      if (!props->is_object())
        throw std::invalid_argument(path + ".properties: must be an object");
      for (const minijson::Member& pm : props->members()) {
        n.property_names.push_back(pm.key);
        n.property_nodes.push_back(compile(pm.value, path + ".properties." + pm.key));
        n.property_required.push_back(false);
      }
      if (!typed) n.types = JsonSchemaNode::kObject;
    }
    if (const minijson::Value* req = v.find("required")) {
      if (!req->is_array())
        throw std::invalid_argument(path + ".required: must be an array");
      for (const minijson::Value& r : req->items()) {
        if (!r.is_string())
          throw std::invalid_argument(path + ".required: entries must be strings");
        bool found = false;
        for (size_t i = 0; i < n.property_names.size(); ++i)
          if (n.property_names[i] == r.as_string()) {
            n.property_required[i] = true;
            found = true;
          }
        if (!found)
          throw std::invalid_argument(path + ".required: '" +
                                      std::string(r.as_string()) +
                                      "' is not a declared property");
      }
      if (!typed) n.types = JsonSchemaNode::kObject;
    }
    if (const minijson::Value* add = v.find("additionalProperties")) {
      if (add->is_bool())
        n.additional = add->as_bool() ? JsonSchemaNode::kAny : JsonSchemaNode::kClosed;
      else if (add->is_object())
        n.additional = compile(*add, path + ".additionalProperties");
      else
        throw std::invalid_argument(path + ".additionalProperties: must be a "
                                    "boolean or a schema");
      if (!typed) n.types = JsonSchemaNode::kObject;
    }
    if (const minijson::Value* items = v.find("items")) {
      if (!items->is_object())
        throw std::invalid_argument(path + ".items: must be a schema");
      n.items = compile(*items, path + ".items");
      if (!typed) n.types = JsonSchemaNode::kArray;
    }
    const auto bound = [&](const char* key, int* into) {
      if (const minijson::Value* b = v.find(key)) {
        if (!b->is_number() || b->as_int() < 0 || b->as_double() != b->as_int())
          throw std::invalid_argument(path + "." + key +
                                      ": must be a non-negative integer");
        *into = static_cast<int>(b->as_int());
        if (!typed) n.types = JsonSchemaNode::kArray;
      }
    };
    bound("minItems", &n.min_items);
    bound("maxItems", &n.max_items);
    if (n.max_items >= 0 && n.max_items < n.min_items)
      throw std::invalid_argument(path + ".maxItems: below minItems");
    const minijson::Value* en = v.find("enum");
    const minijson::Value* cs = v.find("const");
    if (en != nullptr && cs != nullptr)
      throw std::invalid_argument(path + ".const: cannot be combined with enum");
    if (en != nullptr || cs != nullptr) {
      n.has_enum = true;
      if (en != nullptr) {
        if (!en->is_array() || en->items().empty())
          throw std::invalid_argument(path + ".enum: must be a non-empty array");
        for (const minijson::Value& e : en->items())
          n.enum_texts.push_back(json_text_of(e));
      } else {
        n.enum_texts.push_back(json_text_of(*cs));
      }
      uint32_t classes = 0;
      for (const std::string& t : n.enum_texts) {
        const uint32_t c = class_of_text(t);
        if (c == JsonSchemaNode::kObject || c == JsonSchemaNode::kArray)
          throw std::invalid_argument(path + "." + (en ? "enum" : "const") +
                                      ": object and array values are not "
                                      "supported");
        classes |= c;
      }
      if (typed) {
        for (const std::string& t : n.enum_texts)
          if ((class_of_text(t) & n.types) == 0)
            throw std::invalid_argument(path + "." + (en ? "enum" : "const") +
                                        ": the value " + t +
                                        " is outside the declared type");
        n.types &= classes;
      } else {
        n.types = classes;
      }
    }
    out.nodes[static_cast<size_t>(index)] = std::move(n);
    return index;
  }
};

}  // namespace

JsonSchema compile_json_schema(const minijson::Value& schema,
                               std::vector<std::string>* unenforced) {
  Compiler c;
  c.unenforced = unenforced;
  c.out.root = c.compile(schema, "schema");
  return std::move(c.out);
}

JsonSchema json_object_schema() {
  JsonSchema s;
  JsonSchemaNode root;
  root.types = JsonSchemaNode::kObject;
  root.additional = JsonSchemaNode::kAny;
  s.nodes.push_back(std::move(root));
  s.root = 0;
  return s;
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

namespace {

const char* literal_text(uint8_t lit) {
  switch (lit) {
    case 1: return "true";
    case 2: return "false";
    default: return "null";
  }
}

bool is_digit(uint8_t b) { return b >= '0' && b <= '9'; }
bool is_hex(uint8_t b) {
  return is_digit(b) || (b >= 'a' && b <= 'f') || (b >= 'A' && b <= 'F');
}

void push_event(JsonLexer::Step* s, JsonLexer::Event e) {
  for (JsonLexer::Event& slot : s->events)
    if (slot == JsonLexer::Event::kNone) {
      slot = e;
      return;
    }
}

}  // namespace

std::string JsonLexer::literal_rest() const {
  if (state_ != State::kLiteral) return "";
  return std::string(literal_text(lit_) + lit_pos_);
}

void JsonLexer::end_value(Step* s) {
  push_event(s, Event::kValueEnd);
  state_ = stack_.empty() ? State::kDone : State::kValueEnd;
  integer_only_ = false;
}

bool JsonLexer::string_byte(uint8_t b, Step* s, bool key) {
  const Event content = key ? Event::kKeyByte : Event::kValueByte;
  if (esc_ == 0) {
    if (b == '"') {
      if (key) {
        state_ = State::kKeyEnd;
        push_event(s, Event::kKeyEnd);
      } else {
        push_event(s, Event::kValueByte);
        end_value(s);
      }
      return true;
    }
    if (b == '\\') {
      esc_ = 1;
      push_event(s, content);
      return true;
    }
    if (b < 0x20) return false;  // an unescaped control character
    push_event(s, content);
    return true;
  }
  if (esc_ == 1) {
    switch (b) {
      case '"': case '\\': case '/': case 'b': case 'f': case 'n': case 'r':
      case 't':
        esc_ = 0;
        break;
      case 'u':
        esc_ = 2;  // four hex digits to come
        break;
      default:
        return false;
    }
    push_event(s, content);
    return true;
  }
  if (!is_hex(b)) return false;
  esc_ = esc_ == 5 ? 0 : static_cast<uint8_t>(esc_ + 1);
  push_event(s, content);
  return true;
}

JsonLexer::Step JsonLexer::feed(uint8_t b) {
  Step s;
  const auto reject = [&] {
    s.ok = false;
    return s;
  };
  // The whitespace-run cap: structural whitespace counts, anything else
  // (a string's content included) resets.
  if (in_string()) {
    ws_run_ = 0;
  } else if (is_ws(b)) {
    if (ws_run_ >= kMaxWsRun) return reject();
    ++ws_run_;
  } else {
    ws_run_ = 0;
  }
  for (int pass = 0; pass < 2; ++pass) {
    switch (state_) {
      case State::kValue:
      case State::kArrayOpen: {
        if (is_ws(b)) return s;
        if (b == ']' && state_ == State::kArrayOpen) {
          stack_.pop_back();
          push_event(&s, Event::kClose);
          end_value(&s);
          return s;
        }
        uint32_t cls = 0;
        switch (b) {
          case '{': cls = JsonSchemaNode::kObject; break;
          case '[': cls = JsonSchemaNode::kArray; break;
          case '"': cls = JsonSchemaNode::kString; break;
          case '-': cls = JsonSchemaNode::kNumber; break;
          case 't': case 'f': cls = JsonSchemaNode::kBoolean; break;
          case 'n': cls = JsonSchemaNode::kNull; break;
          default:
            if (is_digit(b)) cls = JsonSchemaNode::kNumber;
            else return reject();
        }
        s.cls = cls;
        push_event(&s, Event::kValueStart);
        switch (b) {
          case '{': stack_.push_back('{'); state_ = State::kObjectOpen; break;
          case '[': stack_.push_back('['); state_ = State::kArrayOpen; break;
          case '"': state_ = State::kInString; push_event(&s, Event::kValueByte); break;
          case '-': state_ = State::kNumMinus; push_event(&s, Event::kValueByte); break;
          case '0': state_ = State::kNumZero; push_event(&s, Event::kValueByte); break;
          case 't': case 'f': case 'n':
            lit_ = b == 't' ? 1 : b == 'f' ? 2 : 3;
            lit_pos_ = 1;
            state_ = State::kLiteral;
            push_event(&s, Event::kValueByte);
            break;
          default:
            state_ = State::kNumInt;
            push_event(&s, Event::kValueByte);
            break;
        }
        return s;
      }
      case State::kObjectOpen:
        if (is_ws(b)) return s;
        if (b == '"') {
          state_ = State::kInKey;
          push_event(&s, Event::kKeyStart);
          return s;
        }
        if (b == '}') {
          stack_.pop_back();
          push_event(&s, Event::kClose);
          end_value(&s);
          return s;
        }
        return reject();
      case State::kKeyStart:
        if (is_ws(b)) return s;
        if (b == '"') {
          state_ = State::kInKey;
          push_event(&s, Event::kKeyStart);
          return s;
        }
        return reject();
      case State::kInKey:
        if (!string_byte(b, &s, /*key=*/true)) return reject();
        return s;
      case State::kKeyEnd:
        if (is_ws(b)) return s;
        if (b == ':') {
          state_ = State::kValue;
          return s;
        }
        return reject();
      case State::kInString:
        if (!string_byte(b, &s, /*key=*/false)) return reject();
        return s;
      case State::kNumMinus:
        if (b == '0') state_ = State::kNumZero;
        else if (is_digit(b)) state_ = State::kNumInt;
        else return reject();
        push_event(&s, Event::kValueByte);
        return s;
      case State::kNumZero:
      case State::kNumInt:
      case State::kNumFrac:
      case State::kNumExp: {
        if (is_digit(b) && state_ != State::kNumZero) {
          push_event(&s, Event::kValueByte);
          return s;
        }
        if (b == '.' && (state_ == State::kNumZero || state_ == State::kNumInt)) {
          if (integer_only_) return reject();
          state_ = State::kNumDot;
          push_event(&s, Event::kValueByte);
          return s;
        }
        if ((b == 'e' || b == 'E') && state_ != State::kNumExp) {
          if (integer_only_) return reject();
          state_ = State::kNumE;
          push_event(&s, Event::kValueByte);
          return s;
        }
        if (is_digit(b)) return reject();  // a leading zero
        // The number ends here; the byte belongs to the enclosing state.
        end_value(&s);
        continue;
      }
      case State::kNumDot:
        if (!is_digit(b)) return reject();
        state_ = State::kNumFrac;
        push_event(&s, Event::kValueByte);
        return s;
      case State::kNumE:
        if (b == '+' || b == '-') {
          state_ = State::kNumESign;
          push_event(&s, Event::kValueByte);
          return s;
        }
        if (!is_digit(b)) return reject();
        state_ = State::kNumExp;
        push_event(&s, Event::kValueByte);
        return s;
      case State::kNumESign:
        if (!is_digit(b)) return reject();
        state_ = State::kNumExp;
        push_event(&s, Event::kValueByte);
        return s;
      case State::kLiteral: {
        const char* text = literal_text(lit_);
        if (b != static_cast<uint8_t>(text[lit_pos_])) return reject();
        ++lit_pos_;
        push_event(&s, Event::kValueByte);
        if (text[lit_pos_] == '\0') end_value(&s);
        return s;
      }
      case State::kValueEnd: {
        if (is_ws(b)) return s;
        const char t = top();
        if (b == ',') {
          push_event(&s, Event::kComma);
          state_ = t == '{' ? State::kKeyStart : State::kValue;
          return s;
        }
        if ((b == '}' && t == '{') || (b == ']' && t == '[')) {
          stack_.pop_back();
          push_event(&s, Event::kClose);
          end_value(&s);
          return s;
        }
        return reject();
      }
      case State::kDone:
        if (is_ws(b)) return s;
        return reject();
      case State::kCount:
        return reject();
    }
  }
  return reject();
}

bool JsonLexer::make_base(State state, char ctx, bool integer_only,
                          JsonLexer* out) {
  // A canonical prefix that lands the lexer in `state` with `ctx` as its
  // one open container (none at the top level). Keys exist only in
  // objects; a closer exists only inside a container.
  const std::string in_obj = "{\"k\":";
  const std::string in_arr = "[";
  std::string prefix;
  const std::string ctxp = ctx == '{' ? in_obj : ctx == '[' ? in_arr : "";
  switch (state) {
    case State::kValue: prefix = ctx == '[' ? "[1," : ctxp; break;
    case State::kArrayOpen: if (ctx != '[') return false; prefix = "["; break;
    case State::kObjectOpen: if (ctx != '{') return false; prefix = "{"; break;
    case State::kKeyStart: if (ctx != '{') return false; prefix = "{\"k\":1,"; break;
    case State::kInKey: if (ctx != '{') return false; prefix = "{\""; break;
    case State::kKeyEnd: if (ctx != '{') return false; prefix = "{\"k\""; break;
    case State::kInString: prefix = ctxp + "\""; break;
    case State::kNumMinus: prefix = ctxp + "-"; break;
    case State::kNumZero: prefix = ctxp + "0"; break;
    case State::kNumInt: prefix = ctxp + "1"; break;
    case State::kNumDot: prefix = ctxp + "1."; break;
    case State::kNumFrac: prefix = ctxp + "1.5"; break;
    case State::kNumE: prefix = ctxp + "1e"; break;
    case State::kNumESign: prefix = ctxp + "1e+"; break;
    case State::kNumExp: prefix = ctxp + "1e5"; break;
    case State::kValueEnd: if (ctx == 0) return false; prefix = ctxp + "1 "; break;
    case State::kDone: if (ctx != 0) return false; prefix = "1 "; break;
    default: return false;
  }
  JsonLexer lx;
  for (const char c : prefix)
    if (!lx.feed(static_cast<uint8_t>(c)).ok) return false;
  if (lx.state() != state || lx.top() != ctx) return false;
  lx.integer_only_ = integer_only;
  lx.ws_run_ = 0;  // a canonical base: the run is the machine's, applied by mask()
  *out = lx;
  return true;
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

namespace {

constexpr JsonLexer::State kTabled[] = {
    JsonLexer::State::kValue,      JsonLexer::State::kArrayOpen,
    JsonLexer::State::kObjectOpen, JsonLexer::State::kKeyStart,
    JsonLexer::State::kInKey,      JsonLexer::State::kKeyEnd,
    JsonLexer::State::kInString,   JsonLexer::State::kNumMinus,
    JsonLexer::State::kNumZero,    JsonLexer::State::kNumInt,
    JsonLexer::State::kNumDot,     JsonLexer::State::kNumFrac,
    JsonLexer::State::kNumE,       JsonLexer::State::kNumESign,
    JsonLexer::State::kNumExp,     JsonLexer::State::kValueEnd,
    JsonLexer::State::kDone,
};
constexpr int kTabledCount = static_cast<int>(sizeof(kTabled) / sizeof(kTabled[0]));
constexpr int kCtxCount = 3;  // 0, '{', '['
int ctx_index(char ctx) { return ctx == '{' ? 1 : ctx == '[' ? 2 : 0; }

void set_bit(std::vector<uint32_t>& w, int id) {
  w[static_cast<size_t>(id >> 5)] |= 1u << (id & 31);
}
void clear_bit(std::vector<uint32_t>& w, int id) {
  w[static_cast<size_t>(id >> 5)] &= ~(1u << (id & 31));
}
bool get_bit(const std::vector<uint32_t>& w, int id) {
  return (w[static_cast<size_t>(id >> 5)] >> (id & 31)) & 1u;
}
int popcount(const std::vector<uint32_t>& w) {
  int n = 0;
  for (const uint32_t x : w) n += __builtin_popcount(x);
  return n;
}

// The static simulation of a token from a base: 0 rejected, 1 allowed,
// 2 stack-dependent (the token popped the base's frame and went on).
int simulate_static(const JsonLexer& base, const std::string& text) {
  JsonLexer lx = base;
  const int depth0 = lx.depth();
  for (const char c : text) {
    const uint8_t b = static_cast<uint8_t>(c);
    if (lx.depth() < depth0 && !JsonLexer::is_ws(b)) return 2;
    if (!lx.feed(b).ok) return 0;
  }
  return 1;
}

}  // namespace

int JsonTables::index(JsonLexer::State s, char ctx, bool integer_only) {
  int si = -1;
  for (int i = 0; i < kTabledCount; ++i)
    if (kTabled[i] == s) si = i;
  if (si < 0) return -1;
  return (si * kCtxCount + ctx_index(ctx)) * 2 + (integer_only ? 1 : 0);
}

JsonTables::JsonTables(const GrammarVocab& vocab)
    : vocab_(vocab.vocab_size()), words_((vocab.vocab_size() + 31) / 32) {
  entries_.resize(static_cast<size_t>(kTabledCount * kCtxCount * 2));
  for (auto& c : class_) c.assign(static_cast<size_t>(words_), 0u);
  ws_.assign(static_cast<size_t>(words_), 0u);
  shapes_.resize(static_cast<size_t>(vocab_));
  lead_ws_ids_.assign(static_cast<size_t>(JsonLexer::kMaxWsRun) + 1, {});
  std::map<std::string, int32_t> prefix_ids, tail_ids, scalar_tail_ids;
  // Shapes and class masks.
  for (int id = 0; id < vocab_; ++id) {
    const std::string& text = vocab.text(id);
    Shape& sh = shapes_[static_cast<size_t>(id)];
    if (text.empty()) continue;
    size_t i = 0;
    while (i < text.size() && JsonLexer::is_ws(static_cast<uint8_t>(text[i]))) ++i;
    sh.lead_ws = static_cast<int32_t>(i);
    if (i > 0)
      lead_ws_ids_[static_cast<size_t>(std::min<size_t>(i, JsonLexer::kMaxWsRun))]
          .push_back(id);
    if (i == text.size()) {
      set_bit(ws_, id);
    } else {
      const uint8_t b = static_cast<uint8_t>(text[i]);
      int cls = -1;
      switch (b) {
        case '{': cls = 0; break;
        case '[': cls = 1; break;
        case '"': cls = 2; break;
        case '-': cls = 3; break;
        case 't': case 'f': cls = 5; break;
        case 'n': cls = 6; break;
        default: if (is_digit(b)) cls = 3; break;
      }
      if (cls >= 0) set_bit(class_[static_cast<size_t>(cls)], id);
      if (cls == 3) set_bit(class_[4], id);
    }
    // Unescaped quotes from esc = 0.
    int quotes = 0;
    size_t first_quote = std::string::npos;
    bool esc = false;
    for (size_t j = 0; j < text.size(); ++j) {
      const char c = text[j];
      if (esc) {
        esc = false;
        continue;
      }
      if (c == '\\') {
        esc = true;
        continue;
      }
      if (c == '"') {
        if (quotes == 0) first_quote = j;
        ++quotes;
      }
    }
    sh.quotes = quotes;
    if (quotes == 1) {
      const std::string prefix = text.substr(0, first_quote + 1);
      const std::string tail = text.substr(first_quote + 1);
      auto p = prefix_ids.emplace(prefix, static_cast<int32_t>(prefixes_.size()));
      if (p.second) prefixes_.push_back(prefix);
      sh.prefix_group = p.first->second;
      auto t = tail_ids.emplace(tail, static_cast<int32_t>(tails_.size()));
      if (t.second) tails_.push_back(tail);
      sh.tail_group = t.first->second;
      single_.push_back(id);
    } else if (quotes >= 2) {
      multi_.push_back(id);
    } else {
      bool structural = true;
      for (const char c : text)
        structural = structural && (std::strchr(" \t\n\r,:{}[]", c) != nullptr && c != '\0');
      sh.structural_only = structural;
      if (structural) {
        structural_.push_back(id);
      } else {
        size_t lead = 0;
        while (lead < text.size() && JsonLexer::is_ws(static_cast<uint8_t>(text[lead]))) ++lead;
        const size_t cut = text.find_first_of(",}] \t\n\r:{[", lead);
        if (cut != std::string::npos) {
          const std::string tail = text.substr(cut);
          auto t = scalar_tail_ids.emplace(tail, static_cast<int32_t>(scalar_tails_.size()));
          if (t.second) scalar_tails_.push_back(tail);
          sh.scalar_tail = t.first->second;
          scalar_tail_ids_.push_back(id);
        }
      }
    }
  }
  // The static entries: every existing base, every token, in parallel
  // over the bases (one-time startup work, ~1 s single-threaded).
  struct Job {
    int entry;
    JsonLexer base;
  };
  std::vector<Job> jobs;
  for (int si = 0; si < kTabledCount; ++si)
    for (int ci = 0; ci < kCtxCount; ++ci)
      for (int io = 0; io < 2; ++io) {
        const char ctx = ci == 1 ? '{' : ci == 2 ? '[' : 0;
        JsonLexer base;
        if (!JsonLexer::make_base(kTabled[si], ctx, io != 0, &base)) continue;
        jobs.push_back(Job{(si * kCtxCount + ci) * 2 + io, base});
      }
  const auto run = [&](size_t from, size_t to) {
    for (size_t j = from; j < to; ++j) {
      Entry& e = entries_[static_cast<size_t>(jobs[j].entry)];
      e.words.assign(static_cast<size_t>(words_), 0u);
      for (int id = 0; id < vocab_; ++id) {
        const std::string& text = vocab.text(id);
        if (text.empty()) continue;
        const int verdict = simulate_static(jobs[j].base, text);
        if (verdict == 1) set_bit(e.words, id);
        else if (verdict == 2) e.dynamic.push_back(id);
      }
    }
  };
  const unsigned hw = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  std::vector<std::thread> workers;
  const size_t per = (jobs.size() + hw - 1) / hw;
  for (unsigned t = 0; t < hw; ++t) {
    const size_t from = t * per;
    const size_t to = std::min(jobs.size(), from + per);
    if (from >= to) break;
    workers.emplace_back(run, from, to);
  }
  for (std::thread& w : workers) w.join();
}

const JsonTables::Entry& JsonTables::entry(JsonLexer::State s, char ctx,
                                            bool integer_only) const {
  const int i = index(s, ctx, integer_only);
  if (i < 0 || entries_[static_cast<size_t>(i)].words.empty())
    throw std::logic_error("JsonTables: state has no table");
  return entries_[static_cast<size_t>(i)];
}

const std::vector<uint32_t>& JsonTables::class_mask(uint32_t cls) const {
  switch (cls) {
    case JsonSchemaNode::kObject: return class_[0];
    case JsonSchemaNode::kArray: return class_[1];
    case JsonSchemaNode::kString: return class_[2];
    case JsonSchemaNode::kNumber: return class_[3];
    case JsonSchemaNode::kInteger: return class_[4];
    case JsonSchemaNode::kBoolean: return class_[5];
    default: return class_[6];
  }
}

// ---------------------------------------------------------------------------
// Machine
// ---------------------------------------------------------------------------

JsonMachine::JsonMachine(std::shared_ptr<const JsonSchema> schema,
                         const JsonTables* tables)
    : schema_(std::move(schema)), tables_(tables) {
  if (!schema_ || tables_ == nullptr) return;
  Cursor c;
  c.value_node = schema_->root;
  cursors_.push_back(std::move(c));
  alive_ = true;
}

const char* JsonMachine::state_name() const {
  if (!alive_) return "json-dead";
  switch (lexer_.state()) {
    case JsonLexer::State::kValue: return "json-value";
    case JsonLexer::State::kArrayOpen: return "json-array-open";
    case JsonLexer::State::kObjectOpen: return "json-object-open";
    case JsonLexer::State::kKeyStart: return "json-key-start";
    case JsonLexer::State::kInKey: return "json-key";
    case JsonLexer::State::kKeyEnd: return "json-key-end";
    case JsonLexer::State::kInString: return "json-string";
    case JsonLexer::State::kLiteral: return "json-literal";
    case JsonLexer::State::kValueEnd: return "json-value-end";
    case JsonLexer::State::kDone: return "json-done";
    default: return "json-number";
  }
}

int JsonMachine::expected_node(const Cursor& c) const {
  return c.stack.empty() ? c.value_node : c.stack.back().value_node;
}

void JsonMachine::leaves(int node, std::vector<int>* out) const {
  if (node == JsonSchemaNode::kAny) {
    out->push_back(JsonSchemaNode::kAny);
    return;
  }
  const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(node)];
  if (n.any_of.empty()) {
    out->push_back(node);
    return;
  }
  for (const int alt : n.any_of) leaves(alt, out);
}

namespace {

// A container frame imposes nothing further on its contents (its type and
// enum were checked at its value start).
bool frame_trivial(const JsonSchemaNode& n) {
  return n.property_names.empty() && n.additional == JsonSchemaNode::kAny &&
         n.items == JsonSchemaNode::kAny && n.min_items == 0 && n.max_items < 0;
}

}  // namespace

bool JsonMachine::sweep() {
  const size_t before = cursors_.size();
  cursors_.erase(std::remove_if(cursors_.begin(), cursors_.end(),
                                [](const Cursor& c) { return c.dead; }),
                 cursors_.end());
  if (cursors_.size() != before) ++epoch_;  // the frontier changed
  return !cursors_.empty();
}

bool JsonMachine::done() const {
  if (!alive_ || !lexer_.done()) return false;
  if (lexer_.state() == JsonLexer::State::kDone) return true;
  // A top-level number: complete only where every cursor's target is.
  for (const Cursor& c : cursors_) {
    if (c.target_node < 0) continue;
    const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(c.target_node)];
    bool complete = false;
    for (const int t : c.targets)
      complete = complete || n.enum_texts[static_cast<size_t>(t)].size() == c.target_pos;
    if (!complete) return false;
  }
  return true;
}

bool JsonMachine::on_value_start(uint32_t cls, uint8_t b) {
  ++epoch_;
  std::vector<Cursor> next;
  for (const Cursor& c : cursors_) {
    if (!c.stack.empty() && c.stack.back().kind == '[' &&
        c.stack.back().node != JsonSchemaNode::kAny) {
      const JsonSchemaNode& arr = schema_->nodes[static_cast<size_t>(c.stack.back().node)];
      if (arr.max_items >= 0 && c.stack.back().count >= arr.max_items) continue;
    }
    std::vector<int> options;
    leaves(expected_node(c), &options);
    for (const int leaf : options) {
      Cursor nc = c;
      nc.targets.clear();
      nc.target_node = -1;
      nc.target_pos = 0;
      nc.integer_only = false;
      if (leaf == JsonSchemaNode::kAny) {
        if (cls == JsonSchemaNode::kObject || cls == JsonSchemaNode::kArray) {
          Frame f;
          f.kind = cls == JsonSchemaNode::kObject ? '{' : '[';
          nc.stack.push_back(std::move(f));
        }
        next.push_back(std::move(nc));
        continue;
      }
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(leaf)];
      const uint32_t want = cls == JsonSchemaNode::kNumber
                                ? (JsonSchemaNode::kNumber | JsonSchemaNode::kInteger)
                                : cls;
      if ((n.types & want) == 0) continue;
      if (n.has_enum) {
        for (size_t t = 0; t < n.enum_texts.size(); ++t)
          if (!n.enum_texts[t].empty() &&
              static_cast<uint8_t>(n.enum_texts[t][0]) == b)
            nc.targets.push_back(static_cast<int>(t));
        if (nc.targets.empty()) continue;
        nc.target_node = leaf;
        nc.target_pos = 0;  // the kValueByte of this same byte advances it
      }
      if (cls == JsonSchemaNode::kObject || cls == JsonSchemaNode::kArray) {
        Frame f;
        f.node = leaf;
        f.kind = cls == JsonSchemaNode::kObject ? '{' : '[';
        f.used.assign(n.property_names.size(), false);
        f.value_node = f.kind == '[' ? n.items : JsonSchemaNode::kAny;
        nc.stack.push_back(std::move(f));
      } else if (cls == JsonSchemaNode::kNumber) {
        nc.integer_only = (n.types & JsonSchemaNode::kNumber) == 0;
      }
      next.push_back(std::move(nc));
    }
  }
  cursors_.swap(next);
  return !cursors_.empty();
}

bool JsonMachine::on_value_byte(uint8_t b) {
  for (Cursor& c : cursors_) {
    if (c.integer_only && (b == '.' || b == 'e' || b == 'E')) {
      c.dead = true;
      continue;
    }
    if (c.target_node < 0) continue;
    const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(c.target_node)];
    size_t w = 0;
    for (const int t : c.targets) {
      const std::string& text = n.enum_texts[static_cast<size_t>(t)];
      if (c.target_pos < text.size() &&
          static_cast<uint8_t>(text[c.target_pos]) == b)
        c.targets[w++] = t;
    }
    c.targets.resize(w);
    ++c.target_pos;
    if (w == 0) c.dead = true;
  }
  return sweep();
}

bool JsonMachine::on_value_end() {
  ++epoch_;
  for (Cursor& c : cursors_) {
    if (c.target_node >= 0) {
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(c.target_node)];
      bool complete = false;
      for (const int t : c.targets)
        complete = complete || n.enum_texts[static_cast<size_t>(t)].size() == c.target_pos;
      if (!complete) {
        c.dead = true;
        continue;
      }
      c.targets.clear();
      c.target_node = -1;
      c.target_pos = 0;
    }
    c.integer_only = false;
    if (!c.stack.empty() && c.stack.back().kind == '[') ++c.stack.back().count;
  }
  return sweep();
}

bool JsonMachine::on_key_start() {
  ++epoch_;
  for (Cursor& c : cursors_) {
    Frame& f = c.stack.back();
    f.key.clear();
    if (f.node == JsonSchemaNode::kAny) continue;
    const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
    if (n.additional == JsonSchemaNode::kClosed) {
      bool any_unused = false;
      for (size_t i = 0; i < f.used.size(); ++i) any_unused = any_unused || !f.used[i];
      if (!any_unused) c.dead = true;
    }
  }
  return sweep();
}

bool JsonMachine::on_key_byte(uint8_t b) {
  for (Cursor& c : cursors_) {
    Frame& f = c.stack.back();
    f.key.push_back(static_cast<char>(b));
    if (f.node == JsonSchemaNode::kAny) continue;
    const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
    if (n.additional != JsonSchemaNode::kClosed) continue;
    bool prefix = false;
    for (size_t i = 0; i < n.property_names.size() && !prefix; ++i)
      prefix = !f.used[i] && n.property_names[i].size() >= f.key.size() &&
               n.property_names[i].compare(0, f.key.size(), f.key) == 0;
    if (!prefix) c.dead = true;
  }
  return sweep();
}

bool JsonMachine::on_key_end() {
  ++epoch_;
  for (Cursor& c : cursors_) {
    Frame& f = c.stack.back();
    if (f.node == JsonSchemaNode::kAny) {
      f.value_node = JsonSchemaNode::kAny;
      continue;
    }
    const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
    int found = -1;
    for (size_t i = 0; i < n.property_names.size(); ++i)
      if (n.property_names[i] == f.key) found = static_cast<int>(i);
    if (found >= 0) {
      if (f.used[static_cast<size_t>(found)]) {
        c.dead = true;  // a duplicate key
        continue;
      }
      f.used[static_cast<size_t>(found)] = true;
      f.value_node = n.property_nodes[static_cast<size_t>(found)];
    } else if (n.additional == JsonSchemaNode::kClosed) {
      c.dead = true;
    } else {
      f.value_node = n.additional;
    }
  }
  return sweep();
}

bool JsonMachine::on_comma() {
  ++epoch_;
  for (Cursor& c : cursors_) {
    const Frame& f = c.stack.back();
    if (f.node == JsonSchemaNode::kAny) continue;
    const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
    if (f.kind == '{') {
      if (n.additional == JsonSchemaNode::kClosed) {
        bool any_unused = false;
        for (size_t i = 0; i < f.used.size(); ++i) any_unused = any_unused || !f.used[i];
        if (!any_unused) c.dead = true;
      }
    } else if (n.max_items >= 0 && f.count >= n.max_items) {
      c.dead = true;
    }
  }
  return sweep();
}

bool JsonMachine::on_close() {
  ++epoch_;
  for (Cursor& c : cursors_) {
    const Frame& f = c.stack.back();
    if (f.node != JsonSchemaNode::kAny) {
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
      if (f.kind == '{') {
        for (size_t i = 0; i < f.used.size(); ++i)
          if (n.property_required[i] && !f.used[i]) c.dead = true;
      } else if (f.count < n.min_items) {
        c.dead = true;
      }
    }
    if (!c.dead) c.stack.pop_back();
  }
  return sweep();
}

bool JsonMachine::apply(const JsonLexer::Step& s, uint8_t b) {
  for (const JsonLexer::Event e : s.events) {
    bool ok = true;
    switch (e) {
      case JsonLexer::Event::kNone: return true;
      case JsonLexer::Event::kValueStart: ok = on_value_start(s.cls, b); break;
      case JsonLexer::Event::kValueByte: ok = on_value_byte(b); break;
      case JsonLexer::Event::kValueEnd: ok = on_value_end(); break;
      case JsonLexer::Event::kKeyStart: ok = on_key_start(); break;
      case JsonLexer::Event::kKeyByte: ok = on_key_byte(b); break;
      case JsonLexer::Event::kKeyEnd: ok = on_key_end(); break;
      case JsonLexer::Event::kComma: ok = on_comma(); break;
      case JsonLexer::Event::kClose: ok = on_close(); break;
    }
    if (!ok) return false;
  }
  return true;
}

bool JsonMachine::feed(uint8_t b) {
  if (!alive_) return false;
  const JsonLexer::Step s = lexer_.feed(b);
  if (!s.ok || !apply(s, b)) {
    alive_ = false;
    return false;
  }
  return true;
}

bool JsonMachine::simulate(const std::string& text) const {
  if (text.empty()) return false;
  JsonMachine copy = *this;
  copy.cache_ = Cache{};
  for (const char c : text)
    if (!copy.feed(static_cast<uint8_t>(c))) return false;
  return true;
}

bool JsonMachine::allows(const GrammarVocab& vocab, int64_t id) const {
  if (!alive_) return false;
  return simulate(vocab.text(id));
}

bool JsonMachine::content_constrained() const {
  const JsonLexer::State st = lexer_.state();
  if (st == JsonLexer::State::kInKey) {
    for (const Cursor& c : cursors_) {
      const Frame& f = c.stack.back();
      if (f.node == JsonSchemaNode::kAny) return false;
      if (schema_->nodes[static_cast<size_t>(f.node)].additional !=
          JsonSchemaNode::kClosed)
        return false;
    }
    return !cursors_.empty();
  }
  if (st == JsonLexer::State::kInString || lexer_.in_number() || lexer_.in_literal()) {
    for (const Cursor& c : cursors_)
      if (c.target_node < 0) return false;
    return !cursors_.empty();
  }
  return false;
}

void JsonMachine::content_continuations(std::vector<std::string>* rem) const {
  const JsonLexer::State st = lexer_.state();
  for (const Cursor& c : cursors_) {
    if (st == JsonLexer::State::kInKey) {
      const Frame& f = c.stack.back();
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
      for (size_t i = 0; i < n.property_names.size(); ++i) {
        if (f.used[i]) continue;
        const std::string& p = n.property_names[i];
        if (p.size() < f.key.size() || p.compare(0, f.key.size(), f.key) != 0) continue;
        rem->push_back(p.substr(f.key.size()) + "\"");  // the rest, then the close
      }
    } else if (c.target_node >= 0) {
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(c.target_node)];
      for (const int t : c.targets) {
        const std::string& text = n.enum_texts[static_cast<size_t>(t)];
        if (c.target_pos <= text.size()) rem->push_back(text.substr(c.target_pos));
      }
    }
  }
  std::sort(rem->begin(), rem->end());
  rem->erase(std::unique(rem->begin(), rem->end()), rem->end());
}

bool JsonMachine::schema_constrains_here() const {
  for (const Cursor& c : cursors_) {
    if (c.stack.empty()) return true;  // the root value's class
    for (const Frame& f : c.stack) {
      if (f.node == JsonSchemaNode::kAny) continue;
      if (!frame_trivial(schema_->nodes[static_cast<size_t>(f.node)])) return true;
    }
    if (c.target_node >= 0 || c.integer_only) return true;
  }
  return false;
}

uint32_t JsonMachine::allowed_classes_here() const {
  uint32_t cls = 0;
  for (const Cursor& c : cursors_) {
    if (!c.stack.empty() && c.stack.back().kind == '[' &&
        c.stack.back().node != JsonSchemaNode::kAny) {
      const JsonSchemaNode& arr = schema_->nodes[static_cast<size_t>(c.stack.back().node)];
      if (arr.max_items >= 0 && c.stack.back().count >= arr.max_items) continue;
    }
    std::vector<int> options;
    leaves(expected_node(c), &options);
    for (const int leaf : options) {
      if (leaf == JsonSchemaNode::kAny) return JsonSchemaNode::kAnyClass;
      cls |= schema_->nodes[static_cast<size_t>(leaf)].types;
    }
  }
  return cls;
}

void JsonMachine::mask_brute_force(const GrammarVocab& vocab, TokenMask* out) const {
  const int words = tables_ != nullptr ? tables_->words() : (vocab.vocab_size() + 31) / 32;
  out->vocab = vocab.vocab_size();
  out->words.assign(static_cast<size_t>(words), 0u);
  out->allowed = 0;
  if (!alive_) return;
  for (int id = 0; id < vocab.vocab_size(); ++id)
    if (simulate(vocab.text(id))) set_bit(out->words, id);
  out->allowed = popcount(out->words);
}

void JsonMachine::mask(const GrammarVocab& vocab, TokenMask* out) const {
  const int words = tables_->words();
  out->vocab = vocab.vocab_size();
  out->words.assign(static_cast<size_t>(words), 0u);
  out->allowed = 0;
  if (!alive_) return;
  const JsonLexer::State st = lexer_.state();
  // The whitespace budget left at this position: a token whose leading
  // whitespace run exceeds it would be rejected by the lexer's cap.
  const auto apply_ws_budget = [&] {
    const int run = lexer_.ws_run();
    if (run <= 0 || lexer_.in_string()) return;
    const int budget = JsonLexer::kMaxWsRun - run;
    for (int n = budget + 1; n <= JsonLexer::kMaxWsRun; ++n)
      for (const int32_t id : tables_->lead_ws_ids(n)) clear_bit(out->words, id);
  };
  const auto finish = [&] {
    apply_ws_budget();
    out->allowed = popcount(out->words);
  };
  if (st == JsonLexer::State::kDone) {
    out->words = tables_->whitespace_mask();
    finish();
    return;
  }

  // Content the schema constrains byte by byte: a closed object's key, an
  // enum text (in any escape state). The remainders name the bytes that
  // may come; a token running past a remainder is simulated.
  if (content_constrained()) {
    std::vector<std::string> rems;
    content_continuations(&rems);
    bool complete = false;
    for (const std::string& r : rems) {
      if (r.empty()) {
        complete = true;
        continue;
      }
      for (const int32_t id : vocab.ids_starting_with(static_cast<unsigned char>(r[0]))) {
        const std::string& text = vocab.text(id);
        if (get_bit(out->words, id)) continue;
        if (text.size() <= r.size()) {
          if (r.compare(0, text.size(), text) == 0) set_bit(out->words, id);
        } else if (text.compare(0, r.size(), r) == 0 && simulate(text)) {
          set_bit(out->words, id);
        }
      }
    }
    // A number or literal target already complete: the delimiters decide.
    if (complete)
      for (const int32_t id : tables_->structural_ids())
        if (simulate(vocab.text(id))) set_bit(out->words, id);
    finish();
    return;
  }

  // Positions the tables do not cover: the middle of a literal, an escape.
  if (lexer_.in_literal()) {
    const std::string rest = lexer_.literal_rest();
    for (const int32_t id : vocab.ids_starting_with(static_cast<unsigned char>(rest[0]))) {
      const std::string& text = vocab.text(id);
      if (text.size() <= rest.size()) {
        if (rest.compare(0, text.size(), text) == 0) set_bit(out->words, id);
      } else if (text.compare(0, rest.size(), rest) == 0 && simulate(text)) {
        set_bit(out->words, id);
      }
    }
    finish();
    return;
  }
  if (lexer_.in_escape()) {
    // The escape's own bytes, then free string content; a token that
    // closes the string is simulated for what follows.
    const std::string firsts =
        lexer_.escape() == 1 ? std::string("\"\\/bfnrtu")
                             : std::string("0123456789abcdefABCDEF");
    for (const char f : firsts)
      for (const int32_t id : vocab.ids_starting_with(static_cast<unsigned char>(f))) {
        const std::string& text = vocab.text(id);
        uint8_t esc = lexer_.escape();
        bool ok = true, closes = false;
        for (const char ch : text) {
          const uint8_t b = static_cast<uint8_t>(ch);
          if (esc == 1) {
            if (std::strchr("\"\\/bfnrtu", ch) == nullptr || ch == '\0') { ok = false; break; }
            esc = ch == 'u' ? 2 : 0;
          } else if (esc >= 2) {
            if (!is_hex(b)) { ok = false; break; }
            esc = esc == 5 ? 0 : static_cast<uint8_t>(esc + 1);
          } else if (b == '"') {
            closes = true;
            break;
          } else if (b == '\\') {
            esc = 1;
          } else if (b < 0x20) {
            ok = false;
            break;
          }
        }
        if (!ok) continue;
        if (!closes || simulate(text)) set_bit(out->words, id);
      }
    finish();
    return;
  }

  const bool value_start = st == JsonLexer::State::kValue ||
                           st == JsonLexer::State::kArrayOpen;
  const bool in_string = lexer_.in_string();
  const bool constrained = schema_constrains_here();
  const uint32_t cls = value_start && constrained ? allowed_classes_here()
                                                  : JsonSchemaNode::kAnyClass;
  // The integer table: inside a number every cursor admits digits only;
  // at a value start the admitted classes hold kInteger without kNumber.
  bool all_integer = !cursors_.empty();
  for (const Cursor& c : cursors_) all_integer = all_integer && c.integer_only;
  if (value_start)
    all_integer = (cls & JsonSchemaNode::kInteger) != 0 && (cls & JsonSchemaNode::kNumber) == 0;
  const JsonTables::Entry& e = tables_->entry(st, lexer_.top(), all_integer);
  out->words = e.words;
  if (!constrained) {
    for (const int32_t id : e.dynamic)
      if (simulate(vocab.text(id))) set_bit(out->words, id);
    finish();
    return;
  }

  // Schema mode. Inside a string the answer only changes at the next
  // structural event (or when the frontier shrinks): reuse it.
  if (in_string && cache_.epoch == epoch_ && cache_.state == st &&
      cache_.top == lexer_.top()) {
    for (size_t i = 0; i < tables_->single_quote_ids().size(); ++i) {
      const int32_t id = tables_->single_quote_ids()[i];
      if (get_bit(out->words, id) && !cache_.tail_ok[i]) clear_bit(out->words, id);
    }
    for (size_t i = 0; i < tables_->multi_quote_ids().size(); ++i) {
      const int32_t id = tables_->multi_quote_ids()[i];
      if (get_bit(out->words, id) && !cache_.multi_ok[i]) clear_bit(out->words, id);
    }
    for (const int32_t id : e.dynamic)
      if (simulate(vocab.text(id))) set_bit(out->words, id);
    finish();
    return;
  }

  // 1. A value start under a type filter: only the admitted classes (and
  //    whitespace, and the structural tokens simulated below).
  if (value_start) {
    if (cls != JsonSchemaNode::kAnyClass) {
      std::vector<uint32_t> keep = tables_->whitespace_mask();
      for (const uint32_t bit : {JsonSchemaNode::kObject, JsonSchemaNode::kArray,
                                 JsonSchemaNode::kString, JsonSchemaNode::kNumber,
                                 JsonSchemaNode::kInteger, JsonSchemaNode::kBoolean,
                                 JsonSchemaNode::kNull}) {
        if ((cls & bit) == 0) continue;
        const std::vector<uint32_t>& m = tables_->class_mask(bit);
        for (size_t w = 0; w < keep.size(); ++w) keep[w] |= m[w];
      }
      for (const int32_t id : tables_->structural_ids()) set_bit(keep, id);
      for (size_t w = 0; w < keep.size(); ++w) out->words[w] &= keep[w];
    }
  }
  // 2. Every token whose structure can reach a schema check: the pure
  //    structure and scalar-tail tokens individually, the single-quote
  //    tokens by representative (their free content is the tables'), the
  //    multi-quote tokens individually.
  //    Inside a string these are content; only a quote matters there.
  if (!in_string) {
    for (const int32_t id : tables_->structural_ids())
      if (get_bit(out->words, id) && !simulate(vocab.text(id))) clear_bit(out->words, id);
    for (const int32_t id : tables_->scalar_tail_ids())
      if (get_bit(out->words, id) && !simulate(vocab.text(id))) clear_bit(out->words, id);
  }
  // An enum expected at a value start constrains the scalar content too:
  // a token's leading value bytes must be a prefix of a target.
  bool enum_ahead = false;
  if (value_start)
    for (const Cursor& c : cursors_) {
      std::vector<int> options;
      leaves(expected_node(c), &options);
      for (const int leaf : options)
        enum_ahead = enum_ahead || (leaf != JsonSchemaNode::kAny &&
                                    schema_->nodes[static_cast<size_t>(leaf)].has_enum);
    }
  if (enum_ahead) {
    // Simulate every admitted non-structural token (the class filter
    // already narrowed them to value starts).
    for (int w = 0; w < words; ++w) {
      uint32_t bits = out->words[static_cast<size_t>(w)];
      while (bits != 0u) {
        const int bit = __builtin_ctz(bits);
        bits &= bits - 1;
        const int id = w * 32 + bit;
        if (get_bit(tables_->whitespace_mask(), id)) continue;
        if (!simulate(vocab.text(id))) clear_bit(out->words, id);
      }
    }
  } else {
    std::vector<int8_t> prefix_ok(tables_->prefixes().size(), -1);  // -1 unknown, 0 no, 1 yes, 2 members
    std::vector<int8_t> tail_ok(tables_->tails().size(), -1);
    const std::vector<int32_t>& singles = tables_->single_quote_ids();
    std::vector<uint8_t> single_answers(singles.size(), 0);
    for (size_t i = 0; i < singles.size(); ++i) {
      const int32_t id = singles[i];
      if (!get_bit(out->words, id)) continue;
      const JsonTables::Shape& sh = tables_->shape(id);
      bool ok = false;
      if (in_string) {
        int8_t& t = tail_ok[static_cast<size_t>(sh.tail_group)];
        if (t < 0) t = simulate("\"" + tables_->tails()[static_cast<size_t>(sh.tail_group)]) ? 1 : 0;
        ok = t == 1;
      } else {
        int8_t& p = prefix_ok[static_cast<size_t>(sh.prefix_group)];
        if (p < 0) {
          const std::string& prefix = tables_->prefixes()[static_cast<size_t>(sh.prefix_group)];
          JsonMachine after = *this;
          after.cache_ = Cache{};
          bool alive = true;
          for (const char ch : prefix)
            if (!after.feed(static_cast<uint8_t>(ch))) {
              alive = false;
              break;
            }
          p = !alive ? 0 : after.content_constrained() ? 2 : 1;
        }
        ok = p == 1 || (p == 2 && simulate(vocab.text(id)));
      }
      if (!ok) clear_bit(out->words, id);
      single_answers[i] = ok ? 1 : 0;
    }
    std::vector<uint8_t> multi_answers(tables_->multi_quote_ids().size(), 0);
    for (size_t i = 0; i < tables_->multi_quote_ids().size(); ++i) {
      const int32_t id = tables_->multi_quote_ids()[i];
      if (!get_bit(out->words, id)) continue;
      const bool ok = simulate(vocab.text(id));
      if (!ok) clear_bit(out->words, id);
      multi_answers[i] = ok ? 1 : 0;
    }
    if (in_string) {
      cache_.epoch = epoch_;
      cache_.state = st;
      cache_.top = lexer_.top();
      cache_.tail_ok = std::move(single_answers);
      cache_.multi_ok = std::move(multi_answers);
    }
  }
  for (const int32_t id : e.dynamic)
    if (simulate(vocab.text(id))) set_bit(out->words, id);
  finish();
}

}  // namespace dgpp::text
