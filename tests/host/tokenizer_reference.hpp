#pragma once
// The original pass/rebuild algorithm, retained only as an independent oracle.
#include <stdexcept>

#include "text/bpe.hpp"
#include "text/tokenizer.hpp"
namespace dgpp::text {
struct TokenizerTestAccess {
  static std::vector<int64_t> reference(const Tokenizer& tok, std::string_view mapped) {
    std::vector<int64_t> result;
    auto* out = &result;
    // ignore_merges: a whole-word vocab hit skips the merge loop entirely
    // (GLM); with it off the merges always run (Qwen).
    if (tok.ignore_merges_) {
      if (const auto it = tok.vocab_.find(mapped); it != tok.vocab_.end()) {
        out->push_back(it->second);
        return result;
      }
    }
    // Initial symbols: one per mapped codepoint (1-3 UTF-8 bytes each).
    // `owned` holds every merged string for the WHOLE call — reserved once
    // so push_back can never reallocate (a realloc would dangle the
    // string_views in `symbols`; total merges <= initial count - 1).
    std::vector<std::string_view> symbols;
    std::vector<std::string> owned;
    symbols.reserve(mapped.size());
    for (size_t i = 0; i < mapped.size();) {
      const unsigned char b = static_cast<unsigned char>(mapped[i]);
      const size_t len = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : 3;
      symbols.push_back(mapped.substr(i, len));
      i += len;
    }
    owned.reserve(symbols.size());

    // The merge loop: lowest rank first, ALL occurrences per pass,
    // left-to-right non-overlapping (HF 0.23's bpe semantics).
    for (;;) {
      uint32_t best_rank = UINT32_MAX;
      size_t best_i = 0;
      bool found = false;
      for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        const auto it = tok.merge_rank_.find(Tokenizer::MergeKey{symbols[i], symbols[i + 1]});
        if (it != tok.merge_rank_.end() && it->second < best_rank) {
          best_rank = it->second;
          best_i = i;
          found = true;
        }
      }
      if (!found) break;
      // Capture the pair BY VALUE before the rebuild (symbols[best_i+1] is
      // about to move).
      const std::string first(symbols[best_i]);
      const std::string second(symbols[best_i + 1]);
      std::vector<std::string_view> next;
      next.reserve(symbols.size());
      size_t i = 0;
      while (i < symbols.size()) {
        if (i + 1 < symbols.size() && symbols[i] == first && symbols[i + 1] == second) {
          owned.push_back(first + second);
          next.push_back(owned.back());
          i += 2;
        } else {
          next.push_back(symbols[i]);
          ++i;
        }
      }
      symbols = std::move(next);
    }
    for (const std::string_view s : symbols) {
      const auto it = tok.vocab_.find(s);
      if (it == tok.vocab_.end())
        throw std::runtime_error("BPE symbol missing from vocab (byte-level coverage broken): " +
                                 std::string(s));
      out->push_back(it->second);
    }
    return result;
  }
  static std::vector<int64_t> actual(const Tokenizer& tok, std::string_view mapped) {
    std::vector<int64_t> out;
    tok.encode_word(mapped, &out);
    return out;
  }
  static std::vector<int64_t> heap(const Tokenizer& tok, std::string_view mapped) {
    if (tok.ignore_merges_) {
      auto it = tok.vocab_.find(mapped);
      if (it != tok.vocab_.end()) return {it->second};
    }
    std::vector<std::string_view> symbols;
    for (size_t i = 0; i < mapped.size();) {
      const auto b = static_cast<unsigned char>(mapped[i]);
      const size_t n = b < 128 ? 1 : 2;
      symbols.push_back(mapped.substr(i, n));
      i += n;
    }
    if (!symbols.empty())
      detail::bpe_heap(
          symbols,
          [&](auto a, auto b) {
            auto it = tok.merge_rank_.find(Tokenizer::MergeKey{a, b});
            return it == tok.merge_rank_.end() ? UINT32_MAX : it->second;
          },
          nullptr);
    std::vector<int64_t> out;
    for (auto s : symbols) out.push_back(tok.vocab_.at(s));
    return out;
  }
  static std::vector<std::string> inversions(const Tokenizer& tok) {
    std::unordered_map<std::string, uint32_t> creation;
    for (const auto& [pair, rank] : tok.merge_rank_) {
      auto [it, inserted] =
          creation.emplace(std::string(pair.first) + std::string(pair.second), rank);
      if (!inserted) it->second = std::min(it->second, rank);
    }
    std::vector<std::string> words;
    for (const auto& [pair, rank] : tok.merge_rank_) {
      auto left = creation.find(std::string(pair.first));
      auto right = creation.find(std::string(pair.second));
      if ((left != creation.end() && left->second > rank) ||
          (right != creation.end() && right->second > rank)) {
        const auto a = std::string(pair.first), b = std::string(pair.second);
        words.push_back(a + b + a + b);
        words.push_back(a + a + b + b);
        if (words.size() == 2000) break;
      }
    }
    return words;
  }
  static std::string map(const Tokenizer& tok, std::string_view bytes) {
    std::string out;
    for (unsigned char b : bytes) {
      const auto cp = tok.byte_to_cp_[b];
      if (cp < 128)
        out.push_back(static_cast<char>(cp));
      else {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 63)));
      }
    }
    return out;
  }
};
}  // namespace dgpp::text
