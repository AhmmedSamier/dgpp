// The slow-file frontends: checkpoints that ship vocab.json + merges.txt
// + tokenizer_config.json (or a chat_template.json envelope) instead of
// tokenizer.json (or chat_template.jinja) upstream must tokenize and
// render exactly like the reference-built files. The fixtures pair each
// slow spelling with the reference tool's own output (tokenizer.ref.json
// was built by the reference builder on the qwen3-asr-audio-support
// branch, tools/qwen_asr_tokenizer.py), so this pins the loader to
// the reference implementation, not to itself.
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/chat_template.hpp"
#include "text/tokenizer.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string read_all(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  require(in.good(), "fixture readable: " + path);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

}  // namespace

DGPP_TEST(tokenizer_slow_files_match_reference_json) {
  const std::string dir =
      std::string(DGPP_SOURCE_DIR) + "/tests/data/slow_tokenizer";
  const dgpp::text::Tokenizer from_json =
      dgpp::text::Tokenizer::load(dir + "/tokenizer.ref.json");
  // tokenizer.json is absent here: this is the slow path.
  const dgpp::text::Tokenizer from_slow =
      dgpp::text::Tokenizer::load(dir + "/tokenizer.json");
  for (const char* text :
       {"ab", "a b", "aabbaa", "a b ab ba", "<|pad|>", "<|pad|><|pad|>",
        "ab <|pad|> ba", "  ", "a  b"}) {
    const std::vector<int64_t> want = from_json.encode(text);
    const std::vector<int64_t> got = from_slow.encode(text);
    require(want == got, std::string("slow encode differs for '") + text +
                             "': " + std::to_string(want.size()) + " vs " +
                             std::to_string(got.size()) + " ids");
    require(from_json.decode(want, /*skip_special_tokens=*/false) ==
                from_slow.decode(got, /*skip_special_tokens=*/false),
            std::string("slow decode differs for '") + text + "'");
  }
  require(from_json.added_tokens().size() ==
              from_slow.added_tokens().size(),
          "slow added-token count differs");
  for (size_t i = 0; i < from_json.added_tokens().size(); ++i) {
    const auto& want = from_json.added_tokens()[i];
    const auto& got = from_slow.added_tokens()[i];
    require(want.id == got.id && want.content == got.content &&
                want.special == got.special,
            "slow added token differs at index " + std::to_string(i));
  }
  require(from_json.max_id() == from_slow.max_id(), "slow max_id differs");
  require(from_json.token_by_id_size() == from_slow.token_by_id_size(),
          "slow reverse-map size differs");
  // The fixture's config says special=False (as 14 of the ASR snapshot's
  // 62 do) while the reference builder forces True: decode-with-skip must
  // agree anyway (control markers stay dropped on both spellings).
  const std::vector<int64_t> with_marker = {0, 1, 257, 2};
  require(from_json.decode(with_marker) == from_slow.decode(with_marker),
          "slow decode-with-skip differs on a control marker");
  require(from_json.decode(with_marker).find("<|pad|>") ==
              std::string::npos,
          "the control marker must decode skipped");
}

DGPP_TEST(tokenizer_missing_everything_names_tokenizer_json) {
  try {
    (void)dgpp::text::Tokenizer::load("/nonexistent/tokenizer.json");
  } catch (const std::runtime_error& e) {
    require(std::string(e.what()).find("tokenizer.json") !=
                std::string::npos,
            "the error names the missing file");
    return;
  }
  require(false, "a wholly missing checkpoint must throw");
}

DGPP_TEST(chat_template_json_envelope_matches_jinja) {
  const std::string dir =
      std::string(DGPP_SOURCE_DIR) + "/tests/data/slow_template";
  const dgpp::text::ChatTemplate direct =
      dgpp::text::ChatTemplate::load(dir + "/chat_template.ref.jinja");
  // chat_template.jinja is absent here: this is the envelope path.
  const dgpp::text::ChatTemplate via_envelope =
      dgpp::text::ChatTemplate::load(dir + "/chat_template.jinja");
  require(direct.source_hash() == via_envelope.source_hash(),
          "the envelope parses to byte-identical template source");
  const dgpp::minijson::ParseResult globals = dgpp::minijson::parse(
      R"({"messages":[{"role":"user","content":"hi"}]})");
  const std::string want =
      direct.render(dgpp::text::Value::from_minijson(globals.root));
  const std::string got =
      via_envelope.render(dgpp::text::Value::from_minijson(globals.root));
  require(want == got && want == "user: hi\n", "unexpected render: '" + got +
                                                    "'");
}
