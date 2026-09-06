// M6 6f: the tool-call / reasoning parser over a fake decoder (host-only,
// always runs). The real tokenizer's round trip — render(assistant
// tool_calls) → encode → parse — lives in glm_chat_template_test; here
// the state machine's contract is pinned on synthetic ids: the reasoning
// split, exact streamed deltas, schema-typed and inferred values, nested
// JSON, several calls per turn, and every malformed shape falling back to
// literal content with no half-parsed call.
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/tool_parser.hpp"

namespace {

using dgpp::text::ChatMarker;
using dgpp::text::ChatMarkers;
using dgpp::text::ToolCallParser;
using dgpp::text::ToolSchemas;
using Event = ToolCallParser::Event;
using Kind = ToolCallParser::Event::Kind;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The fake tokenizer: ids below 256 are bytes; the markers are 1001..1008
// and decode to their literal text (they are not special tokens, exactly
// like the real ones); 999 is a special EOS that decodes to nothing.
constexpr int64_t kEos = 999;
constexpr int64_t kThinkOpen = 1001, kThinkClose = 1002, kToolOpen = 1003,
                  kToolClose = 1004, kKeyOpen = 1005, kKeyClose = 1006,
                  kValueOpen = 1007, kValueClose = 1008;

ChatMarkers fake_markers() {
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  m.arg_key_open = ChatMarker{kKeyOpen, "<arg_key>"};
  m.arg_key_close = ChatMarker{kKeyClose, "</arg_key>"};
  m.arg_value_open = ChatMarker{kValueOpen, "<arg_value>"};
  m.arg_value_close = ChatMarker{kValueClose, "</arg_value>"};
  return m;
}

std::string fake_decode(const std::vector<int64_t>& ids) {
  static const std::map<int64_t, std::string> markers = {
      {kThinkOpen, "<think>"},   {kThinkClose, "</think>"},
      {kToolOpen, "<tool_call>"}, {kToolClose, "</tool_call>"},
      {kKeyOpen, "<arg_key>"},   {kKeyClose, "</arg_key>"},
      {kValueOpen, "<arg_value>"}, {kValueClose, "</arg_value>"},
  };
  std::string out;
  for (const int64_t id : ids) {
    if (id == kEos) continue;
    const auto m = markers.find(id);
    if (m != markers.end())
      out += m->second;
    else if (id >= 0 && id < 256)
      out.push_back(static_cast<char>(id));
  }
  return out;
}

// A token stream from text: bytes, with the marker strings mapped to
// their ids (leftmost-longest, like the real added-token scan).
std::vector<int64_t> ids_of(const std::string& text) {
  static const std::vector<std::pair<std::string, int64_t>> table = {
      {"</tool_call>", kToolClose}, {"<tool_call>", kToolOpen},
      {"</arg_value>", kValueClose}, {"<arg_value>", kValueOpen},
      {"</arg_key>", kKeyClose},   {"<arg_key>", kKeyOpen},
      {"</think>", kThinkClose},   {"<think>", kThinkOpen},
  };
  std::vector<int64_t> out;
  for (size_t i = 0; i < text.size();) {
    bool matched = false;
    for (const auto& [s, id] : table) {
      if (text.compare(i, s.size(), s) == 0) {
        out.push_back(id);
        i += s.size();
        matched = true;
        break;
      }
    }
    if (!matched) out.push_back(static_cast<unsigned char>(text[i++]));
  }
  return out;
}

ToolSchemas weather_schemas() {
  static const std::string tools =
      R"([{"type":"function","function":{"name":"get_weather",)"
      R"("parameters":{"type":"object","properties":{)"
      R"("city":{"type":"string"},"days":{"type":"integer"},)"
      R"("code":{"type":"string"},"opts":{"type":"object"}}}}},)"
      R"({"name":"flat_tool","parameters":{"type":"object",)"
      R"("properties":{"x":{"type":"number"}}}}])";
  static const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(tools);
  return ToolSchemas(parsed.root);
}

struct Run {
  std::string reasoning, content;
  int reasoning_closed = 0;
  std::vector<ToolCallParser::Call> calls;
  std::vector<Kind> order;
};

Run drive(const std::vector<int64_t>& ids, ToolCallParser::Options opts = {},
          ToolSchemas schemas = weather_schemas()) {
  ToolCallParser parser(fake_markers(), fake_decode, std::move(schemas), opts);
  std::vector<Event> events;
  for (const int64_t id : ids) parser.feed(id, &events);
  parser.finish(&events);
  Run run;
  for (const Event& ev : events) {
    run.order.push_back(ev.kind);
    switch (ev.kind) {
      case Kind::kReasoning: run.reasoning += ev.text; break;
      case Kind::kReasoningClosed: ++run.reasoning_closed; break;
      case Kind::kContent: run.content += ev.text; break;
      case Kind::kToolCall: run.calls.push_back(ev.call); break;
    }
  }
  require(static_cast<int>(run.calls.size()) == parser.calls(),
          "calls() counts the emitted calls");
  return run;
}

DGPP_TEST(tool_parser_splitsReasoningFromContentExactly) {
  // The prompt opened <think>; the ids before </think> are reasoning, the
  // rest content, both streamed as exact deltas; a repeated <think> in the
  // reasoning is dropped; </think> yields the structural event once.
  const Run run = drive(ids_of("<think>Let me think.</think>Hello, world."));
  require(run.reasoning == "Let me think.", "reasoning: " + run.reasoning);
  require(run.content == "Hello, world.", "content: " + run.content);
  require(run.reasoning_closed == 1, "one </think> event");
  require(run.calls.empty(), "no calls");
  // Deltas arrive per id, in order: every reasoning delta before the close,
  // every content delta after.
  bool closed = false;
  for (const Kind k : run.order) {
    if (k == Kind::kReasoningClosed) closed = true;
    require(k != Kind::kReasoning || !closed, "reasoning before the close");
    require(k != Kind::kContent || closed, "content after the close");
  }
  // Without the opening think (a prompt that does not end in <think>),
  // everything is content and a stray </think> is literal text.
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const Run flat = drive(ids_of("A</think>B"), plain);
  require(flat.reasoning.empty() && flat.content == "A</think>B",
          "no split without the opening think: " + flat.content);
}

DGPP_TEST(tool_parser_oneCallTypedByTheSchema) {
  // Schema-typed values: city (string) stays text even though "123" parses,
  // days (integer) parses, code (string) keeps JSON-looking text verbatim,
  // an unknown key infers JSON when it parses and text otherwise.
  const Run run = drive(ids_of(
      "</think><tool_call>get_weather"
      "<arg_key>city</arg_key><arg_value>123</arg_value>"
      "<arg_key>days</arg_key><arg_value>3</arg_value>"
      "<arg_key>code</arg_key><arg_value>{\"a\": 1}</arg_value>"
      "<arg_key>ratio</arg_key><arg_value>0.5</arg_value>"
      "<arg_key>note</arg_key><arg_value>asked twice</arg_value>"
      "<arg_key>flag</arg_key><arg_value>true</arg_value>"
      "</tool_call>"));
  require(run.calls.size() == 1, "one call");
  require(run.calls[0].name == "get_weather", "name: " + run.calls[0].name);
  require(run.calls[0].arguments ==
              "{\"city\": \"123\", \"days\": 3, \"code\": \"{\\\"a\\\": 1}\", "
              "\"ratio\": 0.5, \"note\": \"asked twice\", \"flag\": true}",
          "arguments: " + run.calls[0].arguments);
  require(run.content.empty() && run.reasoning.empty(), "nothing else");
}

DGPP_TEST(tool_parser_nestedJsonAndNoArgsAndUnicode) {
  const Run run = drive(ids_of(
      "</think>Calling.<tool_call>get_weather"
      "<arg_key>opts</arg_key><arg_value>{\"a\": [1, 2, {\"b\": null}], \"c\": \"x\"}</arg_value>"
      "</tool_call><tool_call>flat_tool</tool_call>"
      "<tool_call>other<arg_key>k</arg_key><arg_value>[1, \"two\"]</arg_value>"
      "<arg_key>city</arg_key><arg_value>Zürich</arg_value></tool_call>Done."));
  require(run.calls.size() == 3, "three calls, got " +
                                     std::to_string(run.calls.size()));
  require(run.calls[0].arguments ==
              "{\"opts\": {\"a\": [1, 2, {\"b\": null}], \"c\": \"x\"}}",
          "nested JSON re-serialized in json.dumps form: " +
              run.calls[0].arguments);
  require(run.calls[1].name == "flat_tool" && run.calls[1].arguments == "{}",
          "a call without arguments is an empty object");
  // A function outside the schema: JSON when it parses, else text (the
  // UTF-8 passes through unescaped, ensure_ascii=False).
  require(run.calls[2].arguments ==
              "{\"k\": [1, \"two\"], \"city\": \"Zürich\"}",
          "unknown function: " + run.calls[2].arguments);
  require(run.content == "Calling.Done.", "content around the calls: " +
                                              run.content);
  // Order: content, call, call, call, content.
  require(run.order.size() >= 5 && run.order.back() == Kind::kContent,
          "trailing content after the calls");
}

DGPP_TEST(tool_parser_malformedBlocksFallBackToLiteralContent) {
  // A value without a key.
  {
    const Run run = drive(ids_of(
        "</think><tool_call>f<arg_value>x</arg_value></tool_call>tail"));
    require(run.calls.empty(), "no call from a value without a key");
    require(run.content == "<tool_call>f<arg_value>x</arg_value></tool_call>tail",
            "the literal block is content: " + run.content);
  }
  // A key without a value, then the close.
  {
    const Run run = drive(ids_of("</think><tool_call>f<arg_key>k</arg_key></tool_call>"));
    require(run.calls.empty() &&
                run.content == "<tool_call>f<arg_key>k</arg_key></tool_call>",
            "key without value: " + run.content);
  }
  // Text between </arg_key> and <arg_value>.
  {
    const Run run = drive(ids_of(
        "</think><tool_call>f<arg_key>k</arg_key> <arg_value>v</arg_value></tool_call>"));
    require(run.calls.empty(), "stray text inside the block");
    require(run.content ==
                "<tool_call>f<arg_key>k</arg_key> <arg_value>v</arg_value></tool_call>",
            "the block aborts at the stray id and everything is literal "
            "content: " + run.content);
  }
  // Unterminated at the end of the generation (the steps cap).
  {
    const Run run = drive(ids_of(
        "</think>Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par"));
    require(run.calls.empty(), "no call from an unterminated block");
    require(run.content ==
                "Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par",
            "unterminated: " + run.content);
  }
  // A nested <tool_call> aborts the open block and starts a fresh one.
  {
    const Run run = drive(ids_of(
        "</think><tool_call>f<arg_key>k</arg_key><tool_call>g<arg_key>x</arg_key>"
        "<arg_value>1</arg_value></tool_call>"));
    require(run.calls.size() == 1 && run.calls[0].name == "g" &&
                run.calls[0].arguments == "{\"x\": 1}",
            "the second block parses");
    require(run.content == "<tool_call>f<arg_key>k</arg_key>",
            "the first block is literal content: " + run.content);
  }
  // Markers in the wrong state outside a block are literal content.
  {
    const Run run = drive(ids_of("</think>a</arg_key>b<arg_value>c"));
    require(run.calls.empty() && run.content == "a</arg_key>b<arg_value>c",
            "stray markers outside a block: " + run.content);
  }
  // An EOS id inside a block decodes to nothing and leaves it open.
  {
    std::vector<int64_t> ids = ids_of("</think><tool_call>f");
    ids.push_back(kEos);
    const Run run = drive(ids);
    require(run.calls.empty() && run.content == "<tool_call>f",
            "EOS inside a block: " + run.content);
  }
}

DGPP_TEST(tool_parser_forcedPrefixSeedsTheBlock) {
  // tool_choice required: the prompt ends in "</think><tool_call>", so the
  // first ids are the name.
  ToolCallParser::Options forced;
  forced.start_in_reasoning = false;
  forced.start_in_tool_call = true;
  forced.forced_prefix_text = "<tool_call>";
  {
    const Run run = drive(ids_of("get_weather<arg_key>city</arg_key>"
                                 "<arg_value>Paris</arg_value></tool_call>"),
                          forced);
    require(run.calls.size() == 1 && run.calls[0].name == "get_weather" &&
                run.calls[0].arguments == "{\"city\": \"Paris\"}",
            "required: the block parses from the seeded state");
  }
  // tool_choice named: the name is in the prompt; the model may extend it
  // and then argues.
  ToolCallParser::Options named = forced;
  named.seeded_name = "get_";
  named.forced_prefix_text = "<tool_call>get_";
  {
    const Run run = drive(ids_of("weather<arg_key>days</arg_key>"
                                 "<arg_value>2</arg_value></tool_call>"),
                          named);
    require(run.calls.size() == 1 && run.calls[0].name == "get_weather" &&
                run.calls[0].arguments == "{\"days\": 2}",
            "named: seeded name + generated tail: " + run.calls[0].name);
  }
  // A forced block that never closes flushes with the prefix text, so the
  // client sees what the model effectively produced.
  {
    const Run run = drive(ids_of("weather<arg_key>days"), named);
    require(run.calls.empty() &&
                run.content == "<tool_call>get_weather<arg_key>days",
            "forced + unterminated: " + run.content);
  }
}

DGPP_TEST(tool_parser_schemasReadBothToolForms) {
  const ToolSchemas s = weather_schemas();
  require(s.has("get_weather") && s.has("flat_tool") && !s.has("none"),
          "names from the wrapped and the flat form");
  require(s.type_of("get_weather", "city") == ToolSchemas::Type::kString,
          "string");
  require(s.type_of("get_weather", "days") == ToolSchemas::Type::kJson,
          "integer is JSON-typed");
  require(s.type_of("get_weather", "nope") == ToolSchemas::Type::kUnknown,
          "unknown key");
  require(s.type_of("flat_tool", "x") == ToolSchemas::Type::kJson, "number");
  require(s.type_of("missing", "x") == ToolSchemas::Type::kUnknown,
          "unknown function");
  // Markers missing from a tokenizer disable the features, never guess.
  ChatMarkers none;
  require(!none.reasoning_available() && !none.tool_calls_available(),
          "no markers, no features");
  ToolCallParser::Options opts;
  ToolCallParser parser(none, fake_decode, ToolSchemas(), opts);
  std::vector<Event> events;
  for (const int64_t id : ids_of("</think><tool_call>f</tool_call>x"))
    parser.feed(id, &events);
  std::string content;
  for (const Event& ev : events) {
    require(ev.kind == Kind::kContent, "everything is content");
    content += ev.text;
  }
  require(content == "</think><tool_call>f</tool_call>x",
          "literal without markers: " + content);
}

}  // namespace
