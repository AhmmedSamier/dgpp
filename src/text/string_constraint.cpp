#include "text/string_constraint.hpp"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <arpa/inet.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace dgpp::text {
namespace {

std::string quote_regex(std::string_view s) {
  std::string out;
  for (char c : s) {
    if (std::string_view("\\.^$|()[]{}*+?").find(c) != std::string_view::npos) out += '\\';
    out += c;
  }
  return out;
}

// Lark's tree annotations (inline rules and aliases) do not affect the
// accepted language. Translate its EBNF into PCRE2 named subroutines;
// right recursion is native, direct left recursion is factored below.
class LarkCompiler {
 public:
  std::string compile(const std::string& text) {
    std::string current;
    for (size_t at = 0; at < text.size();) {
      size_t end = text.find('\n', at);
      if (end == std::string::npos) end = text.size();
      std::string line = text.substr(at, end - at);
      at = end + 1;
      const size_t first = line.find_first_not_of(" \t\r");
      if (first == std::string::npos || line[first] == '#') continue;
      line.erase(0, first);
      if (line.rfind("%import common.", 0) == 0) {
        std::string name = line.substr(15);
        const auto arrow = name.find("->");
        std::string alias;
        if (arrow != std::string::npos) { alias = trim(name.substr(arrow + 2)); name.resize(arrow); }
        name = trim(name);
        auto it = common().find(name);
        if (it == common().end()) fail("unsupported common terminal " + name);
        definitions_[alias.empty() ? name : alias] = it->second;
      } else if (line.rfind("%ignore ", 0) == 0) {
        ignores_.push_back(trim(line.substr(8)));
      } else if (line[0] == '%') {
        fail("unsupported directive " + line);
      } else if (line[0] == '|') {
        if (current.empty()) fail("alternative without a rule");
        definitions_[current] += " " + line;
      } else {
        auto colon = line.find(':');
        if (colon == std::string::npos) fail("expected rule: expression");
        current = trim(line.substr(0, colon));
        if (!current.empty() && (current[0] == '?' || current[0] == '!')) current.erase(0, 1);
        auto priority = current.find('.');
        if (priority != std::string::npos) fail("rule and terminal priorities are not supported");
        if (current.empty() || current.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789") != std::string::npos)
          fail("invalid rule name");
        if (!definitions_.emplace(current, line.substr(colon + 1)).second) fail("duplicate rule " + current);
      }
    }
    if (!definitions_.count("start")) fail("a start rule is required");
    for (const auto& [name, _] : definitions_) names_[name] = "r" + std::to_string(names_.size());
    // Compute nullable/productive rules before emitting subroutine calls.
    // An indirect cycle with no finite derivation otherwise compiles in
    // PCRE2 but produces an empty token mask only after request admission.
    for (size_t pass = 0; pass <= definitions_.size(); ++pass) {
      bool changed = false;
      for (const auto& [name, expr] : definitions_) {
        parse_branches(expr, false);
        auto& known = rules_[name];
        if (known.nullable != analysis_.nullable || known.productive != analysis_.productive) changed = true;
        known = analysis_;
      }
      if (!changed) break;
    }
    for (const auto& [name, _] : definitions_)
      if (!rules_[name].productive) fail("nonproductive recursion in " + name);
    std::string ignored;
    for (const auto& expr : ignores_) {
      if (!ignored.empty()) ignored += '|';
      ignored += parse(expr, false);
    }
    if (!ignored.empty()) skip_ = "(?:(?:" + ignored + "))*";
    std::string result = "\\A" + skip_ + "(?&" + names_.at("start") + ")" + skip_ + "\\z(?(DEFINE)";
    for (const auto& [name, expr] : definitions_) {
      const bool terminal = std::isupper(static_cast<unsigned char>(name[0]));
      auto branches = parse_branches(expr, !terminal);
      const std::string self = "(?&" + names_.at(name) + ")";
      std::vector<std::string> base, recursive;
      for (auto branch : branches) {
        if (branch.rfind(self, 0) == 0) {
          branch.erase(0, self.size());
          if (branch.empty()) fail("nonproductive recursion in " + name);
          recursive.push_back(std::move(branch));
        } else base.push_back(std::move(branch));
      }
      if (base.empty()) fail("nonproductive recursion in " + name);
      for (size_t i = 0; i < branches.size(); ++i) {
        if (branches[i].starts_with(self)) {
          if (rules_[name].nullable) fail("nullable left recursion is not supported in " + name);
          continue;  // the direct left recursion is factored below
        }
        left_[name].insert(branch_analysis_[i].first.begin(), branch_analysis_[i].first.end());
      }
      std::string body = join(base);
      if (!recursive.empty()) body = "(?:" + body + ")(?:(?:" + join(recursive) + "))*";
      result += "(?<" + names_.at(name) + ">" + body + ")";
    }
    std::map<std::string, int> visiting;
    const auto visit = [&](const auto& recurse, const std::string& name) -> void {
      if (visiting[name] == 1) fail("indirect or hidden left recursion is not supported in " + name);
      if (visiting[name] == 2) return;
      visiting[name] = 1;
      for (const auto& next : left_[name]) recurse(recurse, next);
      visiting[name] = 2;
    };
    for (const auto& [name, _] : definitions_) visit(visit, name);
    return result + ')';
  }
 private:
  struct Analysis {
    bool nullable = true, productive = true;
    std::set<std::string> first;
  };
  std::map<std::string, Analysis> rules_;
  std::map<std::string, std::set<std::string>> left_;
  std::map<std::string, bool> regex_nullable_;
  Analysis analysis_;
  std::vector<Analysis> branch_analysis_;
  static std::string trim(std::string s) {
    auto a = s.find_first_not_of(" \t\r");
    if (a == std::string::npos) return {};
    return s.substr(a, s.find_last_not_of(" \t\r") - a + 1);
  }
  [[noreturn]] static void fail(const std::string& why) { throw std::invalid_argument("Lark grammar: " + why); }
  static std::string join(const std::vector<std::string>& parts) {
    std::string out;
    for (const auto& part : parts) { if (!out.empty()) out += '|'; out += "(?:" + part + ")"; }
    return out;
  }
  static const std::map<std::string, std::string>& common() {
    static const std::map<std::string, std::string> c = {
      {"WS", R"(/[ \t\f\r\n]+/)"}, {"WS_INLINE", R"(/[ \t]+/)"},
      {"NEWLINE", R"(/(\r?\n)+/)"}, {"CR", R"(/\r/)"}, {"LF", R"(/\n/)"},
      {"DIGIT", R"(/[0-9]/)"}, {"HEXDIGIT", R"(/[0-9a-fA-F]/)"},
      {"INT", R"(/[0-9]+/)"}, {"SIGNED_INT", R"(/[+-]?[0-9]+/)"},
      {"NUMBER", R"(/(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?/)"},
      {"SIGNED_NUMBER", R"(/[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?/)"},
      {"CNAME", R"(/[a-zA-Z_][a-zA-Z_0-9]*/)"}, {"LETTER", R"(/[a-zA-Z]/)"},
      {"WORD", R"(/[a-zA-Z]+/)"}, {"LCASE_LETTER", R"(/[a-z]/)"}, {"UCASE_LETTER", R"(/[A-Z]/)"},
      {"ESCAPED_STRING", R"(/"(?:[^"\\\n]|\\.)*"/)"}
    };
    return c;
  }
  void space() { while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) ++pos_; }
  std::string atom(bool spacing) {
    space();
    if (pos_ == input_.size()) fail("missing expression");
    char c = input_[pos_++];
    std::string out;
    Analysis info;
    if (c == '(' || c == '[') {
      auto branches = expression(spacing, c == '(' ? ')' : ']');
      out = "(?:" + join(branches) + ")" + (c == '[' ? "?" : "");
      info = analysis_;
      if (c == '[') info.nullable = info.productive = true;
    } else if (c == '"' || c == '\'') {
      std::string literal;
      bool closed = false;
      while (pos_ < input_.size()) {
        char ch = input_[pos_++];
        if (ch == c) { closed = true; break; }
        if (ch == '\\') {
          if (pos_ == input_.size()) fail("unfinished string escape");
          ch = input_[pos_++];
          if (ch == 'u' || ch == 'x' || ch == 'U') {
            const size_t digits = ch == 'x' ? 2 : ch == 'u' ? 4 : 8;
            if (input_.size()-pos_ < digits) fail("unfinished character escape");
            uint32_t codepoint = 0;
            for (size_t i = 0; i < digits; ++i) {
              const char h = input_[pos_++];
              int v = h >= '0' && h <= '9' ? h-'0' : h >= 'a' && h <= 'f' ? h-'a'+10 : h >= 'A' && h <= 'F' ? h-'A'+10 : -1;
              if (v < 0) fail("invalid character escape");
              codepoint = (codepoint << 4) | static_cast<uint32_t>(v);
            }
            if (codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) fail("invalid Unicode code point");
            if (codepoint < 0x80) literal += static_cast<char>(codepoint);
            else if (codepoint < 0x800) { literal += static_cast<char>(0xc0 | (codepoint>>6)); literal += static_cast<char>(0x80 | (codepoint&63)); }
            else if (codepoint < 0x10000) { literal += static_cast<char>(0xe0 | (codepoint>>12)); literal += static_cast<char>(0x80 | ((codepoint>>6)&63)); literal += static_cast<char>(0x80 | (codepoint&63)); }
            else { literal += static_cast<char>(0xf0 | (codepoint>>18)); literal += static_cast<char>(0x80 | ((codepoint>>12)&63)); literal += static_cast<char>(0x80 | ((codepoint>>6)&63)); literal += static_cast<char>(0x80 | (codepoint&63)); }
            continue;
          }
          if (ch == 'n') ch = '\n'; else if (ch == 'r') ch = '\r'; else if (ch == 't') ch = '\t';
          else if (ch == 'b') ch = '\b'; else if (ch == 'f') ch = '\f'; else if (ch == 'v') ch = '\v';
          else if (ch != '\\' && ch != '"' && ch != '\'') fail("unsupported string escape");
        }
        literal += ch;
      }
      if (!closed) fail("unterminated string");
      info.nullable = literal.empty();
      out = quote_regex(literal);
      if (pos_ < input_.size() && input_[pos_] == 'i') { ++pos_; out = "(?i:" + out + ")"; }
      if (input_.substr(pos_, 2) == "..") {
        pos_ += 2; space();
        if (literal.size() != 1 || pos_ >= input_.size() || input_[pos_] != c) fail("invalid character range");
        ++pos_;
        if (pos_ + 1 >= input_.size() || input_[pos_+1] != c) fail("invalid character range");
        out = "[" + quote_regex(literal) + "-" + quote_regex(input_.substr(pos_, 1)) + "]";
        pos_ += 2;
      }
    } else if (c == '/') {
      bool escaped = false, closed = false;
      while (pos_ < input_.size()) {
        char ch = input_[pos_++];
        if (ch == '/' && !escaped) { closed = true; break; }
        out += ch;
        if (ch == '\\') escaped = !escaped; else escaped = false;
      }
      if (!closed) fail("unterminated regular expression");
      std::string flags;
      while (pos_ < input_.size() && std::string_view("imsx").find(input_[pos_]) != std::string_view::npos)
        flags += input_[pos_++];
      out = "(?" + flags + ":" + out + ")";
      auto [it, inserted] = regex_nullable_.emplace(out, false);
      if (inserted) it->second = StringConstraint(out, StringConstraint::Syntax::kRegex).accepts("", true);
      info.nullable = it->second;
    } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t start = pos_ - 1;
      while (pos_ < input_.size() && (std::isalnum(static_cast<unsigned char>(input_[pos_])) || input_[pos_] == '_')) ++pos_;
      auto name = input_.substr(start, pos_ - start);
      if (!names_.count(name)) fail("undefined rule " + name);
      out = "(?&" + names_.at(name) + ")";
      const auto known = rules_.find(name);
      info.nullable = known != rules_.end() && known->second.nullable;
      info.productive = known != rules_.end() && known->second.productive;
      info.first.insert(name);
    } else fail("unexpected character in expression");
    space();
    if (pos_ < input_.size() && std::string_view("*+?").find(input_[pos_]) != std::string_view::npos) {
      const char repeat = input_[pos_++];
      if (repeat != '+') info.nullable = info.productive = true;
      out = "(?:" + out + (spacing ? skip_ : "") + ")" + repeat;
    } else if (pos_ < input_.size() && input_[pos_] == '~') {
      ++pos_; space();
      const size_t start = pos_;
      while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
      auto lo = input_.substr(start, pos_ - start);
      if (lo.empty()) fail("missing repetition bound");
      if (lo.find_first_not_of('0') == std::string::npos) info.nullable = info.productive = true;
      std::string hi;
      if (input_.substr(pos_, 2) == "..") {
        pos_ += 2;
        size_t begin = pos_;
        while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        if (begin == pos_) fail("missing upper repetition bound");
        hi = "," + input_.substr(begin, pos_ - begin);
      }
      out = "(?:" + out + (spacing ? skip_ : "") + "){" + lo + hi + "}";
    }
    analysis_ = std::move(info);
    return out;
  }
  std::vector<std::string> expression(bool spacing, char closer = 0) {
    std::vector<std::string> branches(1);
    std::vector<Analysis> analyses(1);
    while (true) {
      space();
      if (pos_ == input_.size() || input_[pos_] == '#') { if (closer) fail("unclosed group"); break; }
      if (closer && input_[pos_] == closer) { ++pos_; break; }
      if (input_[pos_] == '|') { ++pos_; branches.emplace_back(); analyses.emplace_back(); continue; }
      if (input_.substr(pos_, 2) == "->") {
        pos_ += 2;
        while (pos_ < input_.size() && input_[pos_] != '|' && input_[pos_] != closer) ++pos_;
        continue;
      }
      if (spacing && !branches.back().empty()) branches.back() += skip_;
      branches.back() += atom(spacing);
      auto& sequence = analyses.back();
      if (sequence.nullable) sequence.first.insert(analysis_.first.begin(), analysis_.first.end());
      sequence.nullable = sequence.nullable && analysis_.nullable;
      sequence.productive = sequence.productive && analysis_.productive;
    }
    analysis_ = Analysis{false, false, {}};
    for (const auto& branch : analyses) {
      analysis_.nullable = analysis_.nullable || branch.nullable;
      analysis_.productive = analysis_.productive || branch.productive;
      analysis_.first.insert(branch.first.begin(), branch.first.end());
    }
    branch_analysis_ = std::move(analyses);
    return branches;
  }
  std::vector<std::string> parse_branches(const std::string& s, bool spacing) {
    input_ = s; pos_ = 0; return expression(spacing);
  }
  std::string parse(const std::string& s, bool spacing) { return join(parse_branches(s, spacing)); }
  std::map<std::string, std::string> definitions_, names_;
  std::vector<std::string> ignores_;
  std::string input_, skip_;
  size_t pos_ = 0;
};

std::string format_regex(const std::string& format) {
  const std::string leap = R"((?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:[02468][048]|[13579][26])00))";
  const std::string date = R"((?:[0-9]{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)|02-(?:0[1-9]|1[0-9]|2[0-8]))|)" + leap + "-02-29)";
  const std::string time = R"((?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:[zZ]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9]))";
  if (format == "date") return date;
  if (format == "time") return time;
  if (format == "date-time") return date + "[tT]" + time;
  if (format == "duration") {
    const std::string clock = R"(T(?:[0-9]+H(?:[0-9]+M)?(?:[0-9]+(?:\.[0-9]+)?S)?|[0-9]+M(?:[0-9]+(?:\.[0-9]+)?S)?|[0-9]+(?:\.[0-9]+)?S))";
    return "P(?:[0-9]+W|(?:[0-9]+Y(?:[0-9]+M)?(?:[0-9]+D)?|[0-9]+M(?:[0-9]+D)?|[0-9]+D)(?:" + clock + ")?|" + clock + ")";
  }
  if (format == "email") {
    const std::string atom = R"([a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+)";
    const std::string local = "(?:" + atom + "(?:\\." + atom + ")*|\"(?:[\\x20-\\x21\\x23-\\x5b\\x5d-\\x7e]|\\\\[\\x20-\\x7e])*\")";
    const std::string domain = "(?:" + format_regex("hostname") + "|\\[(?:" + format_regex("ipv4") +
                               "|IPv6:" + format_regex("ipv6") + ")\\])";
    return local + "@" + domain;
  }
  if (format == "hostname") return R"((?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.?)";
  if (format == "ipv4") return R"((?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9]))";
  if (format == "ipv6") {
    const std::string h = "[0-9a-fA-F]{1,4}";
    std::vector<std::string> forms;
    // Eight groups, or a compression replacing at least one group.
    forms.push_back("(?:" + h + ":){7}" + h);
    const auto groups = [&](int count) {
      return count == 0 ? std::string() : h + "(?::" + h + "){" + std::to_string(count-1) + "}";
    };
    for (int left = 0; left < 8; ++left)
      for (int right = 0; right + left < 8; ++right)
        forms.push_back(groups(left) + "::" + groups(right));
    const std::string v4 = format_regex("ipv4");
    forms.push_back("(?:" + h + ":){6}" + v4);
    for (int left = 0; left < 6; ++left)
      for (int right = 0; left + right < 6; ++right)
        forms.push_back(groups(left) + "::" + (right ? groups(right) + ":" : "") + v4);
    std::string out = "(?:";
    for (const auto& form : forms) { if (out != "(?:") out += '|'; out += form; }
    return out + ')';
  }
  if (format == "uuid") return R"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})";
  throw std::invalid_argument("unsupported string format '" + format + "'");
}

bool format_value_ok(const std::string& format, std::string_view s) {
  if (format == "time" || format == "date-time") {
    auto t = format == "time" ? s : s.substr(11);
    if (t.substr(6, 2) == "60") {
      const auto number = [&](size_t at) { return (t[at]-'0')*10 + t[at+1]-'0'; };
      int utc_minutes = number(0)*60 + number(3);
      auto zone = t.find_first_of("+-", 8);
      if (zone != std::string_view::npos)
        utc_minutes += (t[zone] == '+' ? -1 : 1) * (number(zone+1)*60 + number(zone+4));
      if ((utc_minutes%1440 + 1440)%1440 != 1439) return false;
    }
  }
  if (format == "hostname") return s.size() <= 253 + (s.back() == '.');
  if (format == "duration") return s != "P" && s != "PT" && s.back() != 'T';
  if (format == "ipv6") {
    unsigned char buffer[16];
    return inet_pton(AF_INET6, std::string(s).c_str(), buffer) == 1;
  }
  if (format == "date" || format == "date-time") {
    int y = std::stoi(std::string(s.substr(0, 4)));
    int m = std::stoi(std::string(s.substr(5, 2)));
    int d = std::stoi(std::string(s.substr(8, 2)));
    static const int days[] = {31,28,31,30,31,30,31,31,30,31,30,31};
    const bool leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    return d <= days[m-1] + (m == 2 && leap);
  }
  return true;
}
}  // namespace

struct StringConstraint::Impl {
  pcre2_code* code = nullptr;
  std::string format;
  ~Impl() { if (code) pcre2_code_free(code); }
};

StringConstraint::StringConstraint(std::string expression, Syntax syntax) {
  auto impl = std::make_shared<Impl>();
  if (syntax == Syntax::kFormat) { impl->format = expression; expression = format_regex(expression); }
  if (syntax == Syntax::kLark) expression = LarkCompiler().compile(expression);
  else if (syntax == Syntax::kPattern) {
    // Avoid a leading .* before ^: PCRE2's partial matcher cannot know
    // that consuming another byte there makes the later anchor impossible.
    bool top_alternative = false, in_class = false, escaped = false;
    int depth = 0;
    for (char c : expression) {
      if (escaped) { escaped = false; continue; }
      if (c == '\\') { escaped = true; continue; }
      if (c == '[' && !in_class) in_class = true;
      else if (c == ']' && in_class) in_class = false;
      else if (!in_class) {
        if (c == '(') ++depth;
        else if (c == ')') --depth;
        else if (c == '|' && depth == 0) top_alternative = true;
      }
    }
    const bool anchored_start = expression.starts_with("^") && !top_alternative;
    const bool anchored_end = expression.ends_with("$") && (expression.size() < 2 || expression[expression.size()-2] != '\\') && !top_alternative;
    expression = "\\A" + std::string(anchored_start ? "" : "(?s:.*?)") + "(?:" + expression + ")" +
                 (anchored_end ? "" : "(?s:.*)") + "\\z";
  }
  else expression = "\\A(?:" + expression + ")\\z";
  int error = 0;
  PCRE2_SIZE offset = 0;
  impl->code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(expression.data()), expression.size(),
                             PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr);
  if (!impl->code) {
    PCRE2_UCHAR message[256];
    pcre2_get_error_message(error, message, sizeof(message));
    throw std::invalid_argument("invalid string constraint at byte " + std::to_string(offset) + ": " +
                                reinterpret_cast<const char*>(message));
  }
  impl_ = std::move(impl);
}

bool StringConstraint::accepts(std::string_view text, bool complete) const {
  // No byte yet cannot rule out a nonempty match (PCRE2 reports NOMATCH,
  // rather than PARTIAL, for some expressions on an empty subject).
  if (!complete && text.empty()) return true;
  struct Scratch {
    pcre2_match_data* data = pcre2_match_data_create(64, nullptr);
    pcre2_match_context* context = pcre2_match_context_create(nullptr);
    Scratch() { pcre2_set_match_limit(context, 1000000); pcre2_set_depth_limit(context, 1000); }
    ~Scratch() { pcre2_match_context_free(context); pcre2_match_data_free(data); }
  };
  thread_local Scratch scratch;
  const int result = pcre2_match(impl_->code, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(),
                                 0, complete ? 0 : PCRE2_PARTIAL_HARD, scratch.data, scratch.context);
  return result == PCRE2_ERROR_PARTIAL ? !complete :
         result >= 0 && (!complete || format_value_ok(impl_->format, text));
}

bool decode_json_string_prefix(std::string_view json, std::string* out, bool* complete) {
  out->clear(); *complete = false;
  if (json.empty() || json[0] != '"') return false;
  const auto utf8 = [&](unsigned cp) {
    if (cp < 0x80) out->push_back(static_cast<char>(cp));
    else if (cp < 0x800) { out->push_back(0xc0 | (cp >> 6)); out->push_back(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { out->push_back(0xe0 | (cp >> 12)); out->push_back(0x80 | ((cp >> 6) & 63)); out->push_back(0x80 | (cp & 63)); }
    else { out->push_back(0xf0 | (cp >> 18)); out->push_back(0x80 | ((cp >> 12) & 63)); out->push_back(0x80 | ((cp >> 6) & 63)); out->push_back(0x80 | (cp & 63)); }
  };
  const auto hex = [&](size_t pos, unsigned* cp) {
    *cp = 0;
    for (size_t j = pos; j < pos + 4; ++j) {
      char b = json[j];
      int v = b >= '0' && b <= '9' ? b - '0' : b >= 'a' && b <= 'f' ? b - 'a' + 10 : b >= 'A' && b <= 'F' ? b - 'A' + 10 : -1;
      if (v < 0) return false;
      *cp = (*cp << 4) | static_cast<unsigned>(v);
    }
    return true;
  };
  for (size_t i = 1; i < json.size();) {
    unsigned char b = json[i++];
    if (b == '"') { *complete = true; return i == json.size(); }
    if (b == '\\') {
      if (i == json.size()) return true;
      char e = json[i++];
      if (e == 'u') {
        if (json.size() - i < 4) return true;
        unsigned cp;
        if (!hex(i, &cp)) return false;
        i += 4;
        if (cp >= 0xd800 && cp <= 0xdbff) {
          if (json.size() - i < 6) return true;
          unsigned lo;
          if (json.substr(i, 2) != "\\u" || !hex(i+2, &lo) || lo < 0xdc00 || lo > 0xdfff) return false;
          i += 6; cp = 0x10000 + ((cp - 0xd800) << 10) + lo - 0xdc00;
        } else if (cp >= 0xdc00 && cp <= 0xdfff) return false;
        utf8(cp);
      } else {
        const std::string_view escapes = "\"\\/bfnrt";
        auto index = escapes.find(e);
        if (index == std::string_view::npos) return false;
        out->push_back(std::string_view("\"\\/\b\f\n\r\t")[index]);
      }
    } else if (b < 0x20) return false;
    else if (b < 0x80) out->push_back(static_cast<char>(b));
    else {
      int count = b >= 0xc2 && b <= 0xdf ? 1 : b >= 0xe0 && b <= 0xef ? 2 : b >= 0xf0 && b <= 0xf4 ? 3 : -1;
      if (count < 0) return false;
      if (json.size() - i < static_cast<size_t>(count)) return true;
      unsigned cp = b & ((1u << (6-count)) - 1);
      for (int k = 0; k < count; ++k) {
        unsigned char next = json[i++];
        if ((next & 0xc0) != 0x80) return false;
        cp = (cp << 6) | (next & 63);
      }
      if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff) ||
          cp < (count == 1 ? 0x80u : count == 2 ? 0x800u : 0x10000u)) return false;
      utf8(cp);
    }
  }
  return true;
}
}  // namespace dgpp::text
