#pragma once
// The GLM tool-call and reasoning parser (M6 6f, DESIGN §11): a state
// machine over TOKEN IDS that splits one assistant turn's generated ids
// into reasoning text, content text and structured tool calls — the
// inverse of what the chat template renders for an assistant message.
//
// THE FORMAT IS THE TEMPLATE'S. The template renders an assistant turn as
//   <think>{reasoning}</think>{content}<tool_call>{name}<arg_key>{k}
//   </arg_key><arg_value>{v}</arg_value>...</tool_call>...
// and the generation prompt ends in "<think>", so the model's own output
// is the tail of that shape. The markers are ADDED TOKENS of the
// tokenizer (<think> 154841, </think> 154842, <tool_call> 154843,
// </tool_call> 154844, <arg_key>/</arg_key> 154847/8,
// <arg_value>/</arg_value> 154849/50 for the pinned revision): one id
// each, never produced by BPE for ordinary text, so keying on the ids is
// exact where keying on decoded text would be ambiguous. They are NOT
// "special" in the tokenizer's sense — decode() prints them literally —
// which is why a malformed block falls back to its literal text below.
//
// VALUES. The template writes a string argument raw and every other
// value through tojson; the inverse is lossy on its own ("123" and 123
// render alike), so the parser types a value from the request's tool
// schema when it names the key (parameters.properties[key].type: a
// string type keeps the text; a JSON type parses it, the text standing
// when the parse fails) and falls back to "JSON if it parses, else a
// string" for keys the schema does not know — vLLM's GLM parser makes the
// same call. Arguments are re-serialized in Python json.dumps form
// (`{"city": "Paris", "days": 3}`), the template's own tojson dialect.
//
// MALFORMED BLOCKS. A block is a call only when </tool_call> closes a
// consistent name / (key, value)* sequence. Anything else — a value
// without a key, a stray marker, a second <tool_call> before the first
// closed, the stream ending inside a block — flushes the block's literal
// text (the decode of its ids, markers included) as CONTENT, so nothing
// the model produced is silently lost and no half-parsed call reaches a
// client. A <tool_call> nested in an open block starts a fresh block.
//
// STREAMING. Reasoning and content ids stream as suffix-diff deltas of
// their run's full decode (the service's text discipline); tool-call ids
// buffer until the block completes or aborts. Events come out in arrival
// order. The parser is pure host code with no tokenizer dependency beyond
// the Decode callback, so the service gate drives it with a fake and the
// template gate with the real tokenizer.
//
// Threading: one parser per request, driven from one thread at a time.
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp {
class GlmTokenizer;
}

namespace dgpp::glm {

// One marker: its id (-1 = the tokenizer has no such added token) and its
// literal text (what the template writes and decode() prints).
struct ChatMarker {
  int64_t id = -1;
  std::string text;
  bool available() const { return id >= 0; }
};

struct ChatMarkers {
  ChatMarker think_open;        // "<think>"
  ChatMarker think_close;       // "</think>"
  ChatMarker tool_call_open;    // "<tool_call>"
  ChatMarker tool_call_close;   // "</tool_call>"
  ChatMarker arg_key_open;      // "<arg_key>"
  ChatMarker arg_key_close;     // "</arg_key>"
  ChatMarker arg_value_open;    // "<arg_value>"
  ChatMarker arg_value_close;   // "</arg_value>"

  // The reasoning split needs </think>; tool calls need all six markers.
  bool reasoning_available() const { return think_close.available(); }
  bool tool_calls_available() const {
    return tool_call_open.available() && tool_call_close.available() &&
           arg_key_open.available() && arg_key_close.available() &&
           arg_value_open.available() && arg_value_close.available();
  }

  // Looks every marker up by its text among the tokenizer's added tokens
  // (a revision without one leaves it unavailable — never guessed).
  static ChatMarkers from_tokenizer(const GlmTokenizer& tok);
};

// The request's tool schemas, reduced to what value typing needs: for
// every function name, each declared parameter's JSON-schema type.
class ToolSchemas {
 public:
  enum class Type { kUnknown, kString, kJson };

  ToolSchemas() = default;
  // `tools` is the OpenAI array: entries either {type: "function",
  // function: {name, parameters}} or the flat {name, parameters} form the
  // template also accepts. Entries without a usable name are skipped.
  explicit ToolSchemas(const minijson::Value& tools);

  bool has(const std::string& function) const {
    return types_.count(function) != 0;
  }
  Type type_of(const std::string& function, const std::string& key) const;

 private:
  std::map<std::string, std::map<std::string, Type>> types_;
};

class ToolCallParser {
 public:
  // ids -> text, special tokens skipped (the service's decode_ids).
  using Decode = std::function<std::string(const std::vector<int64_t>&)>;

  struct Call {
    std::string name;
    std::string arguments;  // a JSON object, json.dumps form
  };

  struct Event {
    // kReasoningClosed marks the </think> id itself (no text): the fold
    // knob needs the position to reproduce the model's own transcript.
    enum class Kind { kReasoning, kReasoningClosed, kContent, kToolCall };
    Kind kind = Kind::kContent;
    std::string text;  // kReasoning / kContent: the delta
    Call call;         // kToolCall
  };

  struct Options {
    // The prompt ended in <think>: ids route to reasoning until </think>.
    bool start_in_reasoning = true;
    // The prompt itself ends inside a "<tool_call>" block (a prompt-side
    // forced block): the first ids are a function name. The service no
    // longer forces blocks — tool_choice is the grammar of constrained
    // decoding (M6 6g) — but the parser keeps the seam.
    bool start_in_tool_call = false;
    // The name text already in the prompt (tool_choice named a function);
    // the model may extend it.
    std::string seeded_name;
    // The forced prefix's literal text, so an aborted forced block
    // flushes as the text the model effectively produced.
    std::string forced_prefix_text;
  };

  ToolCallParser(ChatMarkers markers, Decode decode, ToolSchemas schemas,
                 Options options);

  // One generated id; appends zero or more events in order.
  void feed(int64_t id, std::vector<Event>* out);
  // End of the generation: flushes an open block as content.
  void finish(std::vector<Event>* out);

  int calls() const { return calls_; }
  bool in_tool_call() const { return state_ == State::kToolCall; }

 private:
  enum class State { kReasoning, kContent, kToolCall };
  enum class Sub { kName, kKey, kAfterKey, kValue, kAfterValue };

  // A streamed text run: its ids and the decode so far.
  struct Run {
    std::vector<int64_t> ids;
    std::string text;
  };

  bool is_marker(int64_t id, const ChatMarker& m) const {
    return m.available() && id == m.id;
  }
  void run_append(Run* run, int64_t id, Event::Kind kind,
                  std::vector<Event>* out);
  void enter_tool_call(int64_t opening_id);
  void abort_block(std::vector<Event>* out);
  void complete_block(std::vector<Event>* out);
  std::string typed_value(const std::string& function, const std::string& key,
                          const std::string& text) const;

  ChatMarkers markers_;
  Decode decode_;
  ToolSchemas schemas_;
  Options options_;

  State state_ = State::kContent;
  Run run_;  // the current reasoning/content run

  // The open block.
  Sub sub_ = Sub::kName;
  std::vector<int64_t> raw_;   // every id since (and including) <tool_call>
  bool raw_has_prefix_ = false;  // raw_ lacks the forced prefix's ids
  std::string seeded_name_;
  std::vector<int64_t> name_ids_, key_ids_, value_ids_;
  std::string name_, key_;
  std::vector<std::pair<std::string, std::string>> args_;  // key, text

  int calls_ = 0;
};

}  // namespace dgpp::glm
