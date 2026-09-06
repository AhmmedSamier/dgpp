// M6 6h: the JSON grammar over a fake vocabulary (host-only, always runs).
// Pins: the lexer on a corpus (every valid text accepted and done, every
// invalid one rejected at a byte), the schema compiler's subset and its
// refusals naming the keyword path, and — the exactness gate — the
// representative-based mask() equal to the brute-force answer (every
// token simulated) at every position of random walks driven by the mask
// itself, over the free machine and over schemas exercising every node
// kind; every finished walk parses as JSON and conforms.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/json_grammar.hpp"
#include "text/tool_grammar.hpp"

namespace {

using dgpp::text::ChatMarker;
using dgpp::text::ChatMarkers;
using dgpp::text::GrammarVocab;
using dgpp::text::JsonLexer;
using dgpp::text::JsonMachine;
using dgpp::text::JsonSchema;
using dgpp::text::JsonTables;
using dgpp::text::TokenMask;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Ids 0..255 are single bytes; then the multi-byte tokens a BPE vocabulary
// holds around JSON: quote-bearing pieces with structural prefixes and
// tails, scalars with tails, whole literals and their pieces, whitespace
// runs, escapes, a complete object; markers and EOS ids; vocab_size pads.
const char* kWords[] = {
    "{\"", "\":", "\": \"", "\",", "\"}", ",\"", "\"]", "[\"", ": \"", "\"\n",
    "\"},", "\"}]", "true", "false", "null", "tr", "ue", "fal", "se", "nu",
    "ll", "12", "3.5", "e5", "-1", "0.", "1,", "2}", "5]", "10", "red",
    "blue", "gree", "n\"", "re", "d\"", "city", "days", "unit", "name",
    "abc", " \"", "\n  ", "\n", "  ", "},", "]}", "}}", ",\n", "\\\"",
    "\\n", "\\u00", "1f", "ab", "{\"a\":1}", "[1,2]", "{}", "[]", "\"\"",
    "\"a\"", " ", ":", "\": ", "\": [", "\": {", "\"c", "ity\"", "\"unit\":",
    "celsius", "fahrenheit", "cel", "sius\"", "\"celsius\"", "true,",
    "false}", "null]", "e", "E+", "-", "9", ".", "0", "\"\\", "x\"",
};
constexpr int kWordBase = 256;
constexpr int kWordCount = static_cast<int>(sizeof(kWords) / sizeof(kWords[0]));
constexpr int64_t kThinkOpen = 400, kThinkClose = 401, kToolOpen = 402,
                  kToolClose = 403, kKeyOpen = 404, kKeyClose = 405,
                  kValueOpen = 406, kValueClose = 407;
constexpr int64_t kEosText = 410, kEosUser = 411, kEosObs = 412;
constexpr int kVocab = 416;

GrammarVocab fake_vocab() {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b)
    texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  for (int i = 0; i < kWordCount; ++i)
    texts[static_cast<size_t>(kWordBase + i)] = kWords[i];
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  m.arg_key_open = ChatMarker{kKeyOpen, "<arg_key>"};
  m.arg_key_close = ChatMarker{kKeyClose, "</arg_key>"};
  m.arg_value_open = ChatMarker{kValueOpen, "<arg_value>"};
  m.arg_value_close = ChatMarker{kValueClose, "</arg_value>"};
  return GrammarVocab(std::move(texts), m, {kEosText, kEosUser, kEosObs}, kVocab,
                      kEosObs);
}

const GrammarVocab& vocab() {
  static const GrammarVocab v = fake_vocab();
  return v;
}
const JsonTables& tables() {
  static const JsonTables t(vocab());
  return t;
}

std::shared_ptr<const JsonSchema> compile(const std::string& text) {
  const dgpp::minijson::ParseResult p = dgpp::minijson::parse(text);
  return std::make_shared<const JsonSchema>(dgpp::text::compile_json_schema(p.root));
}

bool lex_all(const std::string& text, JsonLexer* lx) {
  for (const char c : text)
    if (!lx->feed(static_cast<uint8_t>(c)).ok) return false;
  return true;
}

DGPP_TEST(json_grammar_lexerAcceptsTheCorpusAndRejectsTheRest) {
  const char* valid[] = {
      "{}", "[]", "1", "-0", "0.5", "1e5", "1E+5", "-1.25e-3", "\"\"",
      "\"a\\\"b\\\\c\\n\\u00e9\"", "true", "false", "null",
      "{\"a\":1,\"b\":[1,2,{\"c\":null}],\"d\":\"x\"}", " [ 1 , 2 ] ",
      "\n{\n  \"k\" : \"v\"\n}\n", "[[[]]]", "{\"\":{}}",
  };
  for (const char* t : valid) {
    JsonLexer lx;
    require(lex_all(t, &lx), std::string("rejected valid: ") + t);
    require(lx.done(), std::string("not done after: ") + t);
    require(lx.depth() == 0, "stack not empty");
  }
  const char* invalid[] = {
      "01", "1.", "1e", "1e+", ".5", "+1", "-", "[1,]", "{\"a\":1,}",
      "{a:1}", "{\"a\" 1}", "[1 2]", "tru", "truee", "nul", "\"\n\"",
      "\"\\x\"", "\"\\u12g4\"", "]", "}", "{]", "[}", "1 2", "{} {}",
      "\"a\" \"b\"", "{\"a\":}", "[,]", "{,}",
  };
  for (const char* t : invalid) {
    JsonLexer lx;
    const bool ok = lex_all(t, &lx) && lx.done();
    require(!ok, std::string("accepted invalid: ") + t);
  }
  // A number ends at its delimiter with the delimiter re-dispatched: the
  // events of "1}" inside an object end the value, close, end the object.
  JsonLexer lx;
  require(lex_all("{\"k\":1", &lx), "prefix");
  const JsonLexer::Step s = lx.feed('}');
  require(s.ok && s.events[0] == JsonLexer::Event::kValueEnd &&
              s.events[1] == JsonLexer::Event::kClose &&
              s.events[2] == JsonLexer::Event::kValueEnd &&
              s.events[3] == JsonLexer::Event::kNone,
          "number-end events");
  require(lx.done(), "done after 1}");
  // The bases exist where the states exist.
  JsonLexer base;
  require(JsonLexer::make_base(JsonLexer::State::kValue, 0, false, &base) &&
              base.state() == JsonLexer::State::kValue && base.top() == 0,
          "base value/0");
  require(JsonLexer::make_base(JsonLexer::State::kKeyStart, '{', false, &base) &&
              base.top() == '{',
          "base keystart/{");
  require(!JsonLexer::make_base(JsonLexer::State::kKeyStart, '[', false, &base),
          "no keystart in an array");
  require(!JsonLexer::make_base(JsonLexer::State::kValueEnd, 0, false, &base),
          "no value-end at the top");
  require(JsonLexer::make_base(JsonLexer::State::kNumInt, '[', true, &base) &&
              base.integer_only(),
          "integer base");
  require(lex_all("tr", &(lx = JsonLexer())) && lx.literal_rest() == "ue",
          "literal rest");
}

DGPP_TEST(json_grammar_schemaCompilesTheSubsetAndRefusesNamingTheKeyword) {
  // A tool argument's compile tolerates the keywords that only narrow a
  // value, naming each by path; the strict compile refuses them (2026-09-06).
  {
    const dgpp::minijson::ParseResult p = dgpp::minijson::parse(
        R"({"type":"object","properties":{"n":{"type":"integer","minimum":0,"maximum":9},)"
        R"("s":{"type":"string","pattern":"^a","format":"date"}},"required":["n"]})");
    std::vector<std::string> unenforced;
    const dgpp::text::JsonSchema lax = dgpp::text::compile_json_schema(p.root, &unenforced);
    const dgpp::text::JsonSchemaNode& r = lax.nodes[static_cast<size_t>(lax.root)];
    require(r.property_names.size() == 2 &&
                lax.nodes[static_cast<size_t>(r.property_nodes[0])].types ==
                    dgpp::text::JsonSchemaNode::kInteger &&
                lax.nodes[static_cast<size_t>(r.property_nodes[1])].types ==
                    dgpp::text::JsonSchemaNode::kString,
            "tolerated keywords leave the types");
    require(unenforced == std::vector<std::string>{
                              "schema.properties.n.minimum", "schema.properties.n.maximum",
                              "schema.properties.s.pattern", "schema.properties.s.format"},
            "each tolerated keyword named by path");
    bool threw = false;
    try {
      dgpp::text::compile_json_schema(p.root);
    } catch (const std::invalid_argument& e) {
      threw = std::string(e.what()).rfind("schema.properties.n.minimum", 0) == 0;
    }
    require(threw, "the strict compile still refuses by path");
  }
  const auto s = compile(R"({
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "x"},
      "days": {"type": "integer"},
      "unit": {"enum": ["celsius", "fahrenheit"]},
      "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
      "score": {"type": ["number", "null"]},
      "mode": {"const": "fast"},
      "nested": {"anyOf": [{"type": "string"}, {"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["a"], "additionalProperties": false}]}
    },
    "required": ["city", "unit"],
    "additionalProperties": false
  })");
  const dgpp::text::JsonSchemaNode& root = s->nodes[static_cast<size_t>(s->root)];
  require(root.types == dgpp::text::JsonSchemaNode::kObject, "root object");
  require(root.property_names.size() == 7 && root.additional == dgpp::text::JsonSchemaNode::kClosed,
          "root properties closed");
  require(root.property_required[0] && !root.property_required[1] && root.property_required[2],
          "required flags");
  const auto& days = s->nodes[static_cast<size_t>(root.property_nodes[1])];
  require(days.types == dgpp::text::JsonSchemaNode::kInteger, "integer type");
  const auto& unit = s->nodes[static_cast<size_t>(root.property_nodes[2])];
  require(unit.has_enum && unit.enum_texts.size() == 2 && unit.enum_texts[0] == "\"celsius\"" &&
              unit.types == dgpp::text::JsonSchemaNode::kString,
          "enum texts and inferred class");
  const auto& tags = s->nodes[static_cast<size_t>(root.property_nodes[3])];
  require(tags.min_items == 1 && tags.max_items == 3 && tags.items >= 0, "array bounds");
  const auto& score = s->nodes[static_cast<size_t>(root.property_nodes[4])];
  require(score.types == (dgpp::text::JsonSchemaNode::kNumber | dgpp::text::JsonSchemaNode::kInteger |
                          dgpp::text::JsonSchemaNode::kNull),
          "type list");
  const auto& mode = s->nodes[static_cast<size_t>(root.property_nodes[5])];
  require(mode.has_enum && mode.enum_texts == std::vector<std::string>{"\"fast\""}, "const");
  const auto& nested = s->nodes[static_cast<size_t>(root.property_nodes[6])];
  require(nested.any_of.size() == 2, "anyOf");

  struct Refusal {
    const char* schema;
    const char* path;
  };
  const Refusal refusals[] = {
      {R"({"type":"object","properties":{"city":{"type":"string","pattern":"^a"}}})",
       "schema.properties.city.pattern"},
      {R"({"type":"string","minLength":1})", "schema.minLength"},
      {R"({"type":"object","required":["zzz"]})", "schema.required"},
      {R"({"type":"whatever"})", "schema.type"},
      {R"({"enum":[{"a":1}]})", "schema.enum"},
      {R"({"anyOf":[{"type":"string"}],"type":"string"})", "schema.type"},
      {R"({"type":"array","items":{"type":"string","format":"date"}})",
       "schema.items.format"},
      {R"({"type":"string","enum":[1]})", "schema.enum"},
      {R"({"const":1,"enum":[1]})", "schema.const"},
      {R"({"type":"array","minItems":3,"maxItems":2})", "schema.maxItems"},
  };
  for (const Refusal& r : refusals) {
    bool threw = false;
    std::string msg;
    try {
      compile(r.schema);
    } catch (const std::invalid_argument& e) {
      threw = true;
      msg = e.what();
    }
    require(threw, std::string("compiled: ") + r.schema);
    require(msg.rfind(r.path, 0) == 0,
            std::string("refusal '") + msg + "' does not start with " + r.path);
  }
  // Free JSON mode: a root object holding anything.
  const JsonSchema free = dgpp::text::json_object_schema();
  require(free.nodes.size() == 1 && free.nodes[0].types == dgpp::text::JsonSchemaNode::kObject &&
              free.nodes[0].additional == dgpp::text::JsonSchemaNode::kAny,
          "json_object schema");
}

// A light conformance check of a finished document against the schemas
// below (each walk's text must parse and satisfy the pinned facts).
void check_conforms(const std::string& which, const std::string& text) {
  dgpp::minijson::ParseResult p;
  try {
    p = dgpp::minijson::parse(text);
  } catch (const std::exception& e) {
    throw std::runtime_error(which + ": unparsable '" + text + "': " + e.what());
  }
  const dgpp::minijson::Value& v = p.root;
  if (which == "free" || which == "closed") require(v.is_object(), which + ": root not an object: " + text);
  if (which == "closed") {
    require(v.find("city") != nullptr && v.find("city")->is_string(), "closed: city missing: " + text);
    require(v.find("unit") != nullptr, "closed: unit missing: " + text);
    const std::string_view u = v.find("unit")->as_string();
    require(u == "celsius" || u == "fahrenheit", "closed: unit value: " + text);
    for (const dgpp::minijson::Member& m : v.members())
      require(m.key == "city" || m.key == "unit" || m.key == "days" || m.key == "tags" ||
                  m.key == "score" || m.key == "mode" || m.key == "nested",
              "closed: foreign key '" + m.key + "' in " + text);
    if (const dgpp::minijson::Value* d = v.find("days"))
      require(d->is_number() && d->as_double() == d->as_int(), "closed: days not integral: " + text);
    if (const dgpp::minijson::Value* t = v.find("tags")) {
      require(t->is_array() && !t->items().empty() && t->items().size() <= 3, "closed: tags bounds: " + text);
      for (const dgpp::minijson::Value& e : t->items()) require(e.is_string(), "closed: tag type: " + text);
    }
    if (const dgpp::minijson::Value* m = v.find("mode"))
      require(m->is_string() && m->as_string() == "fast", "closed: mode const: " + text);
  }
  if (which == "list") {
    require(v.is_array() && v.items().size() >= 2 && v.items().size() <= 4, "list: bounds: " + text);
    for (const dgpp::minijson::Value& e : v.items()) {
      require(e.is_object(), "list: item not an object: " + text);
      require(e.find("id") != nullptr && e.find("id")->is_number(), "list: id: " + text);
      for (const dgpp::minijson::Member& m : e.members())
        require(m.key == "id" || m.key == "kind", "list: foreign key '" + m.key + "' in " + text);
      if (const dgpp::minijson::Value* k = e.find("kind")) {
        require(k->is_string() || k->is_null(), "list: kind: " + text);
        if (k->is_string()) require(k->as_string() == "a" || k->as_string() == "b", "list: kind enum: " + text);
      }
    }
  }
  if (which == "scalar") {
    if (v.is_number()) require(v.as_double() == 3 || v.as_double() == 10, "scalar: number enum: " + text);
    else require(v.is_bool() && v.as_bool(), "scalar: not 3/10/true: " + text);
  }
}

// Random walks driven by mask(): at every position the representative mask
// equals brute force; the walk feeds a random allowed token.
void oracle_walks(const std::string& which, std::shared_ptr<const JsonSchema> schema,
                  int walks, uint32_t seed, int* finished_out) {
  std::mt19937 rng(seed);
  int finished = 0;
  long positions = 0;
  std::string last_text, last_state;
  for (int w = 0; w < walks; ++w) {
    JsonMachine m(schema, &tables());
    std::string text;
    for (int step = 0; step < 120 && m.alive(); ++step) {
      TokenMask fast, slow;
      m.mask(vocab(), &fast);
      m.mask_brute_force(vocab(), &slow);
      ++positions;
      if (fast.words != slow.words || fast.allowed != slow.allowed) {
        std::string diff;
        for (int id = 0; id < kVocab; ++id)
          if (fast.allows(id) != slow.allows(id))
            diff += " " + std::to_string(id) + "('" + vocab().text(id) + "')" +
                    (fast.allows(id) ? "+" : "-");
        throw std::runtime_error(which + ": mask differs from brute force after '" + text +
                                 "' (" + m.state_name() + "):" + diff);
      }
      // allows() agrees with the mask on a few ids.
      for (int k = 0; k < 4; ++k) {
        const int id = static_cast<int>(rng() % kVocab);
        require(m.allows(vocab(), id) == fast.allows(id),
                which + ": allows() disagrees at id " + std::to_string(id) + " after '" + text + "'");
      }
      if (m.done()) {
        ++finished;
        check_conforms(which, text);
        break;
      }
      require(fast.allowed > 0, which + ": no token allowed after '" + text + "'");
      // Pick a random allowed id; most of the time steer toward tokens
      // carrying structure (a quote, a closer, a delimiter) so walks
      // finish. Empty-text ids are never allowed.
      std::vector<int> allowed, steering;
      for (int id = 0; id < kVocab; ++id)
        if (fast.allows(id)) {
          allowed.push_back(id);
          if (vocab().text(id).find_first_of("\"}],:") != std::string::npos)
            steering.push_back(id);
        }
      const bool steer = step > 3 && !steering.empty() && rng() % 100 < 70;
      const std::vector<int>& pool = steer ? steering : allowed;
      const int pick = pool[rng() % pool.size()];
      const std::string& t = vocab().text(pick);
      require(!t.empty(), which + ": an empty-text id allowed");
      for (const char c : t)
        require(m.feed(static_cast<uint8_t>(c)),
                which + ": an allowed token rejected: '" + t + "' after '" + text + "'");
      text += t;
      last_text = text;
      last_state = m.state_name();
    }
  }
  std::printf("      %s: %d walks, %ld positions, %d finished\n", which.c_str(), walks,
              positions, finished);
  if (std::getenv("DGPP_JSON_WALK_DEBUG") != nullptr)
    std::printf("      last walk: '%s' (%s)\n", last_text.c_str(), last_state.c_str());
  *finished_out = finished;
}

DGPP_TEST(json_grammar_maskEqualsBruteForceOnRandomWalks) {
  int finished = 0;
  oracle_walks("free", std::make_shared<const JsonSchema>(dgpp::text::json_object_schema()),
               120, 20260904u, &finished);
  require(finished > 20, "free walks rarely finish");
  const auto closed = compile(R"({
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "days": {"type": "integer"},
      "unit": {"enum": ["celsius", "fahrenheit"]},
      "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
      "score": {"type": ["number", "null"]},
      "mode": {"const": "fast"},
      "nested": {"anyOf": [{"type": "string"}, {"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["a"], "additionalProperties": false}]}
    },
    "required": ["city", "unit"],
    "additionalProperties": false
  })");
  oracle_walks("closed", closed, 150, 7u, &finished);
  require(finished > 20, "closed walks rarely finish");
  const auto list = compile(R"({
    "type": "array", "minItems": 2, "maxItems": 4,
    "items": {"type": "object", "properties": {"id": {"type": "number"}, "kind": {"enum": ["a", "b", null]}},
              "required": ["id"], "additionalProperties": false}
  })");
  oracle_walks("list", list, 120, 99u, &finished);
  require(finished > 10, "list walks rarely finish");
  const auto scalar = compile(R"({"anyOf": [{"enum": [3, 10]}, {"const": true}]})");
  oracle_walks("scalar", scalar, 40, 3u, &finished);
  require(finished > 20, "scalar walks rarely finish");
}

DGPP_TEST(json_grammar_machineFactsAtTheEdges) {
  // JSON mode: the root must be an object — the first token is '{' or
  // whitespace (or a whitespace-prefixed '{'), nothing else.
  JsonMachine m(std::make_shared<const JsonSchema>(dgpp::text::json_object_schema()), &tables());
  TokenMask mask;
  m.mask(vocab(), &mask);
  for (int id = 0; id < kVocab; ++id) {
    const std::string& t = vocab().text(id);
    if (t.empty()) {
      require(!mask.allows(id), "empty id allowed");
      continue;
    }
    size_t i = 0;
    while (i < t.size() && JsonLexer::is_ws(static_cast<uint8_t>(t[i]))) ++i;
    const bool expect = i == t.size() || t[i] == '{';
    if (expect) {
      // ... provided the rest is a valid object prefix.
      require(mask.allows(id) == m.allows(vocab(), id), "root allows");
    } else {
      require(!mask.allows(id), "root allowed '" + t + "'");
    }
  }
  require(mask.allows('{') && mask.allows(' ') && !mask.allows('[') && !mask.allows('"') &&
              !mask.allows('1'),
          "root object starts");
  // A disallowed byte kills the machine; the mask is then empty.
  require(m.feed('{'), "open");
  require(!m.feed(']'), "] after {");
  require(!m.alive(), "dead");
  m.mask(vocab(), &mask);
  require(mask.allowed == 0, "dead mask empty");
  // Done: whitespace only, and done() holds.
  JsonMachine d(std::make_shared<const JsonSchema>(dgpp::text::json_object_schema()), &tables());
  for (const char c : std::string("{\"a\": [1, 2]}")) require(d.feed(static_cast<uint8_t>(c)), "doc");
  require(d.done(), "done");
  d.mask(vocab(), &mask);
  require(mask.allows(' ') && mask.allows('\n') && !mask.allows('}') && !mask.allows(','),
          "done mask");
  require(d.feed(' ') && d.done(), "ws after done");
  // A closed object's keys are spelled from the declared names only, in
  // any tokenization; a foreign key dies at its first byte.
  const auto closed = compile(R"({"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"],"additionalProperties":false})");
  JsonMachine c(closed, &tables());
  for (const char ch : std::string("{\"")) require(c.feed(static_cast<uint8_t>(ch)), "open key");
  c.mask(vocab(), &mask);
  require(mask.allows('c') && mask.allows('d') && !mask.allows('x') && !mask.allows('"'),
          "closed key first bytes");
  require(mask.allows(kWordBase + 36) /* city */ && mask.allows(kWordBase + 37) /* days */,
          "whole key tokens");
  require(!mask.allows(kWordBase + 40) /* abc */, "foreign key token");
  for (const char ch : std::string("city\":")) require(c.feed(static_cast<uint8_t>(ch)), "city key");
  c.mask(vocab(), &mask);
  require(mask.allows('"') && mask.allows(' ') && !mask.allows('1') && !mask.allows('{'),
          "string value expected");
  for (const char ch : std::string("\"Oslo\",\"")) require(c.feed(static_cast<uint8_t>(ch)), "value, next key");
  c.mask(vocab(), &mask);
  require(mask.allows('d') && !mask.allows('c'), "city used, days left");
  for (const char ch : std::string("days\":")) require(c.feed(static_cast<uint8_t>(ch)), "days key");
  c.mask(vocab(), &mask);
  require(mask.allows('1') && mask.allows('-') && !mask.allows('"') && !mask.allows('t'),
          "integer expected");
  require(c.feed('3'), "3");
  c.mask(vocab(), &mask);
  require(!mask.allows('.') && !mask.allows('e') && mask.allows('}') && !mask.allows(','),
          "integer-only, no keys left");
  require(c.feed('}') && c.done(), "closed done");
  // The whitespace-run cap: sixteen structural whitespace bytes pass, the
  // seventeenth is refused, a structural byte is still allowed; the mask
  // drops the tokens whose leading run would overflow the budget; and a
  // complete text with the cap reached admits nothing (the grammar layer
  // adds EOS).
  JsonMachine w(std::make_shared<const JsonSchema>(dgpp::text::json_object_schema()), &tables());
  for (int i = 0; i < 15; ++i) require(w.feed('\n'), "whitespace within the cap");
  w.mask(vocab(), &mask);
  require(mask.allows(' ') && mask.allows('{') && !mask.allows(kWordBase + 44) /* "  " */ &&
              !mask.allows(kWordBase + 42) /* "\n  " */,
          "one byte of budget left: single whitespace and structure only");
  require(w.feed('\n'), "the sixteenth whitespace byte");
  w.mask(vocab(), &mask);
  require(!mask.allows(' ') && !mask.allows('\n') && mask.allows('{') && !mask.allows(kWordBase + 43),
          "budget spent: structure only");
  {
    JsonMachine over = w;
    require(!over.feed('\n'), "the seventeenth whitespace byte is refused");
  }
  require(w.feed('{') && w.feed('}') && w.done(), "the brace resets the run");
  for (int i = 0; i < 16; ++i) require(w.feed(' '), "trailing whitespace within the cap");
  w.mask(vocab(), &mask);
  require(mask.allowed == 0 && w.done(), "complete and capped: nothing but the end");
  // Content whitespace inside a string never counts.
  JsonMachine sw(std::make_shared<const JsonSchema>(dgpp::text::json_object_schema()), &tables());
  for (const char ch : std::string("{\"a\": \"")) require(sw.feed(static_cast<uint8_t>(ch)), "open string");
  for (int i = 0; i < 40; ++i) require(sw.feed(' '), "spaces are string content");
  require(sw.feed('"'), "the string closes");
  require(sw.lexer().ws_run() == 0, "the run is zero after content");
  // Required keys missing: the closer is refused.
  JsonMachine r(closed, &tables());
  for (const char ch : std::string("{\"days\":1")) require(r.feed(static_cast<uint8_t>(ch)), "days only");
  r.mask(vocab(), &mask);
  require(!mask.allows('}') && mask.allows(','), "city still owed");
}

}  // namespace
