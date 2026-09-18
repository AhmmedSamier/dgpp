#pragma once

#include <memory>
#include <string>
#include <string_view>

namespace dgpp::text {

// Immutable, shared compiled expression. Match state belongs to each call,
// so copies and concurrent requests never share mutable PCRE2 state.
class StringConstraint {
 public:
  enum class Syntax { kPattern, kRegex, kLark, kFormat };
  StringConstraint(std::string expression, Syntax syntax);
  bool accepts(std::string_view text, bool complete) const;
 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

// Decode a JSON string prefix (including its opening quote). Incomplete
// escapes/UTF-8 remain pending until their next byte; invalid encodings fail.
bool decode_json_string_prefix(std::string_view json, std::string* decoded,
                               bool* complete);

}  // namespace dgpp::text
