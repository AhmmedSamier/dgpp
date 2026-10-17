#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <string_view>
#include <vector>

namespace dgpp::text::detail {

// Optional work counters for deterministic regression tests. The tokenizer
// passes no counters; all state, including the heap, belongs to this call.
struct BpeWork {
  size_t pass_pairs = 0;
  size_t heap_pairs = 0;
  size_t heap_pushes = 0;
  size_t heap_peak = 0;
  size_t heap_switches = 0;
};

// Symbols are adjacent, nonempty spans in one mapped pretoken. Rank returns
// UINT32_MAX for an absent pair. Merging extends a span; it never copies text.
template <class Rank>
void bpe_heap(std::vector<std::string_view>& symbols, const Rank& rank, BpeWork* work) {
  const size_t end = symbols.size();
  struct Node {
    std::string_view text;
    size_t prev, next;
  };
  struct Pair {
    uint32_t rank;
    size_t pos, left_size, right_size;
    bool operator<(const Pair& other) const {
      return rank != other.rank ? rank > other.rank : pos > other.pos;
    }
  };
  std::vector<Node> nodes;
  nodes.reserve(end);
  for (size_t i = 0; i < end; ++i) nodes.push_back({symbols[i], i == 0 ? end : i - 1, i + 1});
  std::vector<Pair> initial;
  initial.reserve(end);
  const auto candidate = [&](size_t pos, auto&& put) {
    if (pos == end || nodes[pos].text.empty() || nodes[pos].next == end) return;
    const auto left = nodes[pos].text;
    const auto right = nodes[nodes[pos].next].text;
    if (work) ++work->heap_pairs;
    const uint32_t r = rank(left, right);
    if (r != UINT32_MAX) {
      put(Pair{r, pos, left.size(), right.size()});
      if (work) ++work->heap_pushes;
    }
  };
  for (size_t i = 0; i < end; ++i) candidate(i, [&](Pair pair) { initial.push_back(pair); });
  std::priority_queue<Pair> heap(std::less<Pair>{}, std::move(initial));
  std::vector<size_t> changed;
  while (!heap.empty()) {
    if (work) work->heap_peak = std::max(work->heap_peak, heap.size());
    const uint32_t phase = heap.top().rank;
    changed.clear();
    // All occurrences of this rank existed at the START of this phase.
    // New neighbors wait until it ends, even if their rank is lower.
    do {
      const Pair pair = heap.top();
      heap.pop();
      Node& left = nodes[pair.pos];
      if (left.text.size() != pair.left_size || left.next == end) continue;
      Node& right = nodes[left.next];
      if (right.text.size() != pair.right_size) continue;
      // A live node only grows. Changing its next neighbor consumes that
      // neighbor and grows its span, so the two lengths also reject stale
      // entries after either endpoint changes or is consumed.
      left.text = {left.text.data(), left.text.size() + right.text.size()};
      left.next = right.next;
      right.text = {};
      if (left.next != end) nodes[left.next].prev = pair.pos;
      changed.push_back(left.prev);
      changed.push_back(pair.pos);
    } while (!heap.empty() && heap.top().rank == phase);
    for (size_t pos : changed) candidate(pos, [&](Pair pair) { heap.push(pair); });
  }
  symbols.clear();
  for (size_t i = 0; i != end; i = nodes[i].next) symbols.push_back(nodes[i].text);
}

template <class Rank>
void bpe_merge(std::vector<std::string_view>& symbols, const Rank& rank, BpeWork* work = nullptr) {
  const size_t initial = symbols.size();
  // Short words avoid heap setup. For longer words, at most four initial
  // symbol counts of pair probes precede the heap. Geometrically shrinking
  // runs need fewer than two and stay on the cheap pass path throughout.
  size_t budget = initial;
  size_t rounds = 4;
  for (;;) {
    if (symbols.size() < 2) return;
    const size_t pairs = symbols.size() - 1;
    if (initial >= 256) {
      if (pairs > budget) {
        if (--rounds == 0) {
          if (work) ++work->heap_switches;
          bpe_heap(symbols, rank, work);
          return;
        }
        budget += initial;
      }
      budget -= pairs;
    }
    if (work) work->pass_pairs += pairs;
    uint32_t best = UINT32_MAX;
    size_t pos = 0;
    for (size_t i = 0; i < pairs; ++i) {
      const uint32_t r = rank(symbols[i], symbols[i + 1]);
      if (r < best) {
        best = r;
        pos = i;
      }
    }
    if (best == UINT32_MAX) return;
    const auto first = symbols[pos];
    const auto second = symbols[pos + 1];
    size_t out = 0;
    for (size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == first && symbols[i + 1] == second) {
        symbols[out++] = {symbols[i].data(), symbols[i].size() + symbols[i + 1].size()};
        i += 2;
      } else {
        symbols[out++] = symbols[i++];
      }
    }
    symbols.resize(out);
  }
}

}  // namespace dgpp::text::detail
