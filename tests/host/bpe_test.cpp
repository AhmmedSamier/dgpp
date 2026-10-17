#include "text/bpe.hpp"

#include <map>
#include <random>
#include <stdexcept>
#include <string>

#include "common/test.hpp"

namespace {
using Rules = std::map<std::pair<std::string, std::string>, uint32_t>;
using Words = std::vector<std::string>;
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
Words reference(const std::string& word, const Rules& rules) {
  Words symbols;
  for (char c : word) symbols.emplace_back(1, c);
  for (;;) {
    uint32_t best = UINT32_MAX;
    std::pair<std::string, std::string> pair;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto it = rules.find({symbols[i], symbols[i + 1]});
      if (it != rules.end() && it->second < best) {
        best = it->second;
        pair = it->first;
      }
    }
    if (best == UINT32_MAX) return symbols;
    Words next;
    for (size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == pair.first && symbols[i + 1] == pair.second) {
        next.push_back(pair.first + pair.second);
        i += 2;
      } else
        next.push_back(symbols[i++]);
    }
    symbols = std::move(next);
  }
}
Words merge(const std::string& word, const Rules& rules, bool force_heap,
            dgpp::text::detail::BpeWork* work = nullptr) {
  std::vector<std::string_view> symbols;
  for (size_t i = 0; i < word.size(); ++i) symbols.emplace_back(word.data() + i, 1);
  auto rank = [&](std::string_view a, std::string_view b) {
    auto it = rules.find({std::string(a), std::string(b)});
    return it == rules.end() ? UINT32_MAX : it->second;
  };
  if (force_heap && !symbols.empty())
    dgpp::text::detail::bpe_heap(symbols, rank, work);
  else
    dgpp::text::detail::bpe_merge(symbols, rank, work);
  Words result;
  for (auto s : symbols) result.emplace_back(s);
  return result;
}

DGPP_TEST(bpe_phase_defers_rank_inversions_and_consumed_neighbors) {
  const Rules inverted{{{"ab", "a"}, 0}, {{"a", "b"}, 1}};
  require(merge("abab", inverted, true) == Words{"ab", "ab"},
          "new low rank must wait for phase end");
  const Rules overlap{{{"aa", "a"}, 0}, {{"a", "a"}, 1}, {{"aa", "aa"}, 2}};
  for (size_t n = 0; n < 40; ++n) {
    const std::string word(n, 'a');
    require(merge(word, overlap, true) == reference(word, overlap),
            "overlap or stale pair changed result");
  }
}

DGPP_TEST(bpe_generated_rules_match_original_passes) {
  std::mt19937 rng(121);
  Words tokens{"a", "b", "c"};
  for (size_t i = 0; i < 3; ++i)
    for (char c : std::string("abc")) tokens.push_back(tokens[i] + c);
  std::vector<std::pair<std::string, std::string>> pairs;
  for (const auto& a : tokens)
    for (const auto& b : tokens) pairs.emplace_back(a, b);
  for (size_t trial = 0; trial < 32; ++trial) {
    std::shuffle(pairs.begin(), pairs.end(), rng);
    Rules rules;
    for (size_t i = 0; i < pairs.size(); ++i) rules.emplace(pairs[i], i);
    // Exhaust all binary words through length 9, including overlapping rules.
    for (size_t n = 0; n <= 9; ++n)
      for (size_t bits = 0; bits < (size_t{1} << n); ++bits) {
        std::string word(n, 'a');
        for (size_t i = 0; i < n; ++i)
          if (bits & (size_t{1} << i)) word[i] = 'b';
        const auto want = reference(word, rules);
        require(merge(word, rules, true) == want, "heap differs from independent reference");
        require(merge(word, rules, false) == want, "passes differ from independent reference");
      }
    for (size_t n : {256u, 1024u, 4096u}) {
      std::string word(n, 'a');
      for (auto& c : word) c += rng() % 3;
      dgpp::text::detail::BpeWork work;
      require(merge(word, rules, false, &work) == reference(word, rules),
              "switch changed merge order");
      require(work.heap_switches == 1, "hard input must reach the heap");
      require(work.pass_pairs <= 4 * n, "pass work must remain bounded");
      require(work.heap_pairs <= 3 * n, "only adjacent heap pairs should be reconsidered");
      require(work.heap_pushes <= 3 * n && work.heap_peak <= 3 * n,
              "heap work/storage must remain bounded");
    }
  }
}

DGPP_TEST(bpe_switch_handles_long_symbols_and_rank_inversions) {
  // Long operands exercise span growth and stale entries after switching.
  Rules rules{
      {{"ab", "a"}, 0}, {{"a", "b"}, 1}, {{"ab", "ab"}, 2}, {{"a", "a"}, 3}, {{"aa", "aa"}, 4}};
  for (size_t n = 4; n <= 1024; n *= 2)
    rules.emplace(std::make_pair(std::string(n, 'a'), std::string(n, 'a')), rules.size());
  std::string word;
  for (size_t i = 0; i < 512; ++i) word += "ababaa";
  word += std::string(4096, 'a');
  require(merge(word, rules, true) == reference(word, rules), "long phase mismatch");
  require(merge(word, rules, false) == reference(word, rules), "long automatic mismatch");
}

DGPP_TEST(bpe_uniform_runs_do_not_need_the_heap) {
  Rules rules;
  for (size_t n = 1; n <= 128; n *= 2)
    rules.emplace(std::make_pair(std::string(n, 'a'), std::string(n, 'a')), rules.size());
  const std::string word(65536, 'a');
  dgpp::text::detail::BpeWork work;
  require(merge(word, rules, false, &work) == reference(word, rules), "uniform run mismatch");
  require(work.heap_switches == 0 && work.heap_pairs == 0,
          "uniform run should keep the pass fast path");
  require(work.pass_pairs < 2 * word.size(), "uniform run should shrink geometrically");
}
}  // namespace

int main() {
  return dgpp::test::run_all();
}
