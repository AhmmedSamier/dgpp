#pragma once
// Unicode NFC normalization — the Qwen3.8 tokenizer.json's
// normalizer, implemented from the committed tables (unicode_tables.hpp:
// canonical decompositions, combining classes, primary composites, the
// quick-check set) per UAX #15: full canonical decomposition (Hangul
// algorithmic), canonical ordering of non-starter runs, canonical
// composition with the blocking rule. A string whose codepoints are all
// quick-check YES is returned unchanged (the common case, one table probe
// per non-Latin-1 codepoint). Strict UTF-8: invalid input throws.
#include <string>
#include <string_view>

namespace dgpp::text::unicode {

// True when every codepoint is NFC quick-check YES (the string is NFC).
bool nfc_quick_yes(std::string_view utf8);
// The NFC form of `utf8`.
std::string nfc(std::string_view utf8);

}  // namespace dgpp::text::unicode
