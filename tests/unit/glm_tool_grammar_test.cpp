// M6 6g: the tool-call grammar over a fake vocabulary (host-only, always
// runs). Pins: the masks of every state (which ids a position allows and
// the allowed count the sampler treats as the vocabulary), the name/key
// automaton over token texts (any tokenization of a name, nothing else,
// prefix-sharing names), the modes (required / named / auto-single /
// forbid, parallel or not), the EOS discipline while a call is owed, a
// disallowed id killing the grammar, and a complete valid turn being
// accepted position by position.
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "models/glm_tool_grammar.hpp"

namespace {

using dgpp::glm::ChatMarker;
using dgpp::glm::ChatMarkers;
using dgpp::glm::GrammarSpec;
using dgpp::glm::GrammarState;
using dgpp::glm::GrammarTool;
using dgpp::glm::GrammarVocab;
using dgpp::glm::TokenMask;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The fake vocabulary: ids 0..255 are single bytes, then a few multi-byte
// word tokens, the eight markers, and three EOS ids; vocab_size pads it.
constexpr int64_t kGet = 256, kWeather = 257, kGetWeather = 258,
                  kUnderscore = 259, kWea = 260, kTher = 261, kCity = 262,
                  kGetT = 263, kIme = 264;
constexpr int64_t kThinkOpen = 300, kThinkClose = 301, kToolOpen = 302,
                  kToolClose = 303, kKeyOpen = 304, kKeyClose = 305,
                  kValueOpen = 306, kValueClose = 307;
constexpr int64_t kEosText = 310, kEosUser = 311, kEosObs = 312;
constexpr int kVocab = 320;

GrammarVocab fake_vocab() {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b) texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  texts[kGet] = "get";
  texts[kWeather] = "_weather";
  texts[kGetWeather] = "get_weather";
  texts[kUnderscore] = "_";
  texts[kWea] = "wea";
  texts[kTher] = "ther";
  texts[kCity] = "city";
  texts[kGetT] = "get_t";
  texts[kIme] = "ime";
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

GrammarSpec spec_of(GrammarSpec::Mode mode, bool parallel = true,
                    const std::string& named = "") {
  GrammarSpec s;
  s.mode = mode;
  s.parallel = parallel;
  s.named = named;
  GrammarTool weather;
  weather.name = "get_weather";
  weather.constrain_keys = true;
  weather.keys = {"city", "days"};
  GrammarTool time;
  time.name = "get_time";  // shares the "get_" prefix
  time.constrain_keys = false;
  GrammarTool bare;
  bare.name = "ping";
  bare.constrain_keys = true;  // closed and empty: no arguments at all
  s.tools = {weather, time, bare};
  return s;
}

std::vector<int64_t> allowed_ids(const GrammarState& g) {
  TokenMask m;
  g.mask(&m);
  std::vector<int64_t> out;
  if (!m.constrained()) return out;
  for (int64_t id = 0; id < kVocab; ++id)
    if (m.allows(id)) out.push_back(id);
  require(static_cast<int>(out.size()) == m.allowed,
          "allowed count equals the set bits: " + std::to_string(out.size()) +
              " vs " + std::to_string(m.allowed));
  return out;
}

bool same(std::vector<int64_t> a, std::vector<int64_t> b) {
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return a == b;
}

std::string show(const std::vector<int64_t>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size() && i < 12; ++i)
    s += (i ? "," : "") + std::to_string(v[i]);
  if (v.size() > 12) s += ",…";
  return s + "]";
}

void feed(GrammarState& g, const std::vector<int64_t>& ids) {
  for (const int64_t id : ids) {
    require(g.allows(id), "id " + std::to_string(id) + " allowed in state " +
                              g.state_name());
    g.advance(id);
  }
}

DGPP_TEST(tool_grammar_requiredOwesACallAndForcesItsShape) {
  const GrammarVocab v = fake_vocab();
  GrammarState g(&v, spec_of(GrammarSpec::Mode::kRequired), /*thinking=*/true);
  // Thinking: free except the three EOS ids (the turn may not end).
  {
    const std::vector<int64_t> a = allowed_ids(g);
    require(a.size() == static_cast<size_t>(kVocab - 3), "think allows all but EOS: " +
                                                             std::to_string(a.size()));
    require(g.allows('a') && g.allows(kThinkClose) && g.allows(kToolOpen) &&
                !g.allows(kEosObs) && !g.allows(kEosUser),
            "think mask membership");
  }
  feed(g, {'h', 'm', kThinkClose});
  // After </think> under required: only <tool_call>.
  require(same(allowed_ids(g), {kToolOpen}), "top owes a call: " + show(allowed_ids(g)));
  feed(g, {kToolOpen});
  // The name: any token that continues a tool name from its start.
  require(same(allowed_ids(g), {'g', 'p', kGet, kGetWeather, kGetT}),
          "name starts: " + show(allowed_ids(g)));
  feed(g, {kGet});
  // "get" emitted: "_" (either name), "_weather", "_t" is not a token here.
  require(same(allowed_ids(g), {'_', kUnderscore, kWeather}),
          "after 'get': " + show(allowed_ids(g)));
  feed(g, {kUnderscore, kWea});  // "get_wea"
  require(same(allowed_ids(g), {'t', kTher}), "after 'get_wea': " + show(allowed_ids(g)));
  feed(g, {kTher});  // "get_weather" complete, no longer name extends it
  require(same(allowed_ids(g), {kKeyOpen, kToolClose}),
          "complete name: <arg_key> or </tool_call>: " + show(allowed_ids(g)));
  feed(g, {kKeyOpen});
  // Closed keys: "city" or "days" — by their tokens.
  require(same(allowed_ids(g), {'c', 'd', kCity}), "keys: " + show(allowed_ids(g)));
  feed(g, {kCity});
  require(same(allowed_ids(g), {kKeyClose}), "complete key");
  feed(g, {kKeyClose});
  require(same(allowed_ids(g), {kValueOpen}), "after key");
  feed(g, {kValueOpen});
  // Values are free text: everything but the markers (and EOS: a call is
  // still owed until it closes), plus the closer.
  {
    const std::vector<int64_t> a = allowed_ids(g);
    require(g.allows('P') && g.allows(kValueClose) && g.allows(kThinkOpen) &&
                !g.allows(kKeyOpen) && !g.allows(kToolOpen) && !g.allows(kEosObs),
            "value mask membership");
    require(a.size() == static_cast<size_t>(kVocab - 5 - 3), "value allows all but 5 "
            "markers and 3 EOS: " + std::to_string(a.size()));
  }
  feed(g, {'P', 'a', kValueClose});
  require(same(allowed_ids(g), {kKeyOpen, kToolClose}), "after value");
  feed(g, {kToolClose});
  // One call closed, parallel: another call or the turn end (only the
  // call-turn EOS).
  require(same(allowed_ids(g), {kToolOpen, kEosObs}),
          "after a call (parallel): " + show(allowed_ids(g)));
  feed(g, {kEosObs});
  require(same(allowed_ids(g), {kEosObs}), "done: EOS again");
  require(g.active(), "still active (never died)");
}

DGPP_TEST(tool_grammar_namedSingleCallThenEos) {
  const GrammarVocab v = fake_vocab();
  GrammarState g(&v, spec_of(GrammarSpec::Mode::kNamed, true, "get_time"),
                 /*thinking=*/false);
  require(same(allowed_ids(g), {kToolOpen}), "no think: a call right away");
  feed(g, {kToolOpen});
  require(same(allowed_ids(g), {'g', kGet, kGetT}),
          "only get_time's starts: " + show(allowed_ids(g)));
  feed(g, {kGetT, kIme});
  // get_time's keys are open: free text or the closers.
  require(same(allowed_ids(g), {kKeyOpen, kToolClose}), "name complete");
  feed(g, {kKeyOpen});
  require(g.allows('z') && g.allows(kKeyClose) && !g.allows(kValueOpen),
          "open keys are free text");
  feed(g, {'z', kKeyClose, kValueOpen, '1', kValueClose, kToolClose});
  // Named: exactly one call, then the turn ends.
  require(same(allowed_ids(g), {kEosObs}), "named: EOS after the one call");
  // A tool without arguments closes right after its name.
  GrammarState p(&v, spec_of(GrammarSpec::Mode::kNamed, true, "ping"), false);
  feed(p, {kToolOpen, 'p', 'i', 'n', 'g'});
  require(same(allowed_ids(p), {kToolClose}), "closed empty keys: no <arg_key>");
}

DGPP_TEST(tool_grammar_autoSingleAndForbidModes) {
  const GrammarVocab v = fake_vocab();
  // auto + parallel false: free until a call opens, then that call, then EOS.
  GrammarState a(&v, spec_of(GrammarSpec::Mode::kAuto, false), true);
  require(!a.active() || allowed_ids(a).empty(), "think is unconstrained (no call owed)");
  feed(a, {'x', kThinkClose});
  require(a.allows('H') && a.allows(kEosUser) && a.allows(kToolOpen) &&
              !a.allows(kKeyOpen),
          "top: free, calls may open, stray markers may not");
  feed(a, {'H', 'i', kToolOpen, kGetWeather, kToolClose});
  require(same(allowed_ids(a), {kEosObs}), "auto single: EOS after the one call");
  // forbid: everything but <tool_call> (and the other markers).
  GrammarState f(&v, spec_of(GrammarSpec::Mode::kForbidCalls), true);
  feed(f, {kThinkClose});
  require(f.allows('H') && f.allows(kEosUser) && !f.allows(kToolOpen) &&
              !f.allows(kToolClose),
          "forbid: no call may open");
  {
    TokenMask m;
    f.mask(&m);
    require(m.allowed == kVocab - 6, "forbid allows all but the six markers");
  }
  // required + parallel false: exactly one call.
  GrammarState r(&v, spec_of(GrammarSpec::Mode::kRequired, false), false);
  feed(r, {kToolOpen, kGetWeather, kToolClose});
  require(same(allowed_ids(r), {kEosObs}), "required single: EOS");
}

DGPP_TEST(tool_grammar_disallowedIdKillsTheGrammar) {
  const GrammarVocab v = fake_vocab();
  GrammarState g(&v, spec_of(GrammarSpec::Mode::kRequired), false);
  require(!g.allows('H'), "text before the call is not allowed");
  g.advance('H');  // the MTP draft may propose it; the verify rejects it
  require(!g.active(), "dead after a disallowed id");
  TokenMask m;
  g.mask(&m);
  require(!m.constrained(), "a dead grammar constrains nothing");
  // An inactive spec is never active.
  GrammarState none(&v, GrammarSpec{}, true);
  require(!none.active(), "kNone is inactive");
  none.mask(&m);
  require(!m.constrained(), "inactive: no mask");
  // A named function outside the tools refuses.
  bool threw = false;
  try {
    GrammarState bad(&v, spec_of(GrammarSpec::Mode::kNamed, true, "nope"), false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "named outside tools refuses");
}

DGPP_TEST(tool_grammar_jsonModeSpellsOneTextThenEos) {
  // response_format (M6 6h) as a grammar: thinking stays free but the
  // turn cannot end; </think> opens the JSON body, where the machine's
  // mask rules, the markers never appear, and EOS comes only once the
  // text is complete.
  const GrammarVocab v = fake_vocab();
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kJson;  // json_schema "" = json_object
  GrammarState g(&v, spec, /*prompt_opens_thinking=*/true);
  require(g.active() && std::string(g.state_name()) == "think", "starts thinking");
  require(g.allows('x') && g.allows(kThinkClose) && g.allows(kToolOpen) &&
              !g.allows(kEosText) && !g.allows(kEosObs),
          "thinking is free, EOS withheld");
  g.advance(kThinkClose);
  require(std::string(g.state_name()) == "json-value", "the body opens");
  require(g.allows('{') && g.allows(' ') && g.allows('\n') && !g.allows('"') &&
              !g.allows('[') && !g.allows('1') && !g.allows(kEosText) &&
              !g.allows(kThinkOpen) && !g.allows(kThinkClose) && !g.allows(kToolOpen),
          "json_object: an object opens, nothing else");
  TokenMask m;
  g.mask(&m);
  require(m.constrained() && m.allows('{') && !m.allows(kEosText) && m.allowed >= 5,
          "the body mask is a real constraint");
  for (const char c : std::string("{\"a\":1")) g.advance(static_cast<unsigned char>(c));
  require(g.active() && !g.allows(kEosText) && g.allows(',') && g.allows('}'),
          "inside the object: no EOS yet");
  g.advance('}');
  require(g.allows(kEosText) && g.allows(kEosObs) && g.allows(' ') && !g.allows(',') &&
              !g.allows('{'),
          "complete: EOS or whitespace only");
  g.mask(&m);
  require(m.allows(kEosText) && m.allows(kEosUser) && !m.allows('}'), "done mask");
  g.advance(kEosText);
  require(std::string(g.state_name()) == "done", "EOS ends the turn");

  // A schema: closed keys spelled from the declared names, an integer
  // value, the closer only once the required key is in.
  GrammarSpec typed;
  typed.mode = GrammarSpec::Mode::kJson;
  typed.json_schema =
      "{\"type\":\"object\",\"properties\":{\"k\":{\"type\":\"integer\"},"
      "\"s\":{\"enum\":[\"on\",\"off\"]}},\"required\":[\"k\"],"
      "\"additionalProperties\":false}";
  GrammarState t(&v, typed, /*prompt_opens_thinking=*/false);
  require(std::string(t.state_name()) == "json-value", "no think block: body at once");
  t.advance('{');
  require(t.allows('"') && !t.allows('}'), "the required key is owed");
  t.advance('"');
  require(t.allows('k') && t.allows('s') && !t.allows('z') && !t.allows('"'),
          "keys from the declared names");
  for (const char c : std::string("k\":")) t.advance(static_cast<unsigned char>(c));
  require(t.allows('-') && t.allows('7') && !t.allows('"') && !t.allows('t'),
          "an integer value");
  t.advance('4');
  require(!t.allows('.') && !t.allows('e') && t.allows('}') && t.allows(','),
          "integer: no fraction; the object may close");
  for (const char c : std::string(",\"s\":\"o")) t.advance(static_cast<unsigned char>(c));
  require(t.allows('n') && t.allows('f') && !t.allows('x') && !t.allows('"'),
          "an enum string is spelled from its targets");
  for (const char c : std::string("ff\"}")) t.advance(static_cast<unsigned char>(c));
  require(t.allows(kEosText), "done under the schema");
  // A disallowed id kills the grammar (the sampler never produces one).
  GrammarState k(&v, typed, false);
  k.advance('[');
  require(!k.active(), "a disallowed id kills the JSON grammar");
  // A schema text that does not parse, or is outside the subset, refuses
  // at construction.
  GrammarSpec bad = typed;
  bad.json_schema = "{not json";
  bool threw = false;
  try {
    GrammarState b(&v, bad, false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "unparsable schema text refused");
  bad.json_schema = "{\"type\":\"string\",\"pattern\":\"^a\"}";
  threw = false;
  try {
    GrammarState b(&v, bad, false);
  } catch (const std::invalid_argument& e) {
    threw = std::string(e.what()).rfind("schema.pattern", 0) == 0;
  }
  require(threw, "an unsupported keyword refused by name");
  // The spec's equality covers the schema text.
  require(!(spec == typed), "specs differ by schema");
  GrammarSpec same = typed;
  require(same == typed, "equal specs");
}

}  // namespace
