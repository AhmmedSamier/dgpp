#include "sched/prefix_cache.hpp"

#include <algorithm>
#include <stdexcept>

namespace dgpp::sched {

PrefixCache::PrefixCache(const Config& cfg) : cfg_(cfg) {
  if (cfg_.slots < 0) throw std::invalid_argument("PrefixCache: negative slots");
  if (cfg_.align < 1) throw std::invalid_argument("PrefixCache: align must be >= 1");
  if (cfg_.chunk_tokens < 1)
    throw std::invalid_argument("PrefixCache: chunk_tokens must be >= 1");
  free_.reserve(static_cast<size_t>(cfg_.slots));
  for (int s = 0; s < cfg_.slots; ++s) free_.push_back(s);
}

int PrefixCache::live_entries() const {
  int n = 0;
  for (const Entry& e : entries_) n += e.live ? 1 : 0;
  return n;
}

std::vector<int64_t> PrefixCache::cuts(
    int64_t n, const std::vector<int64_t>& boundaries) const {
  std::vector<int64_t> out;
  for (int64_t m = cfg_.chunk_tokens; m < n; m += cfg_.chunk_tokens) out.push_back(m);
  for (const int64_t b : boundaries) {
    const int64_t a = (b / cfg_.align) * cfg_.align;
    if (a > 0 && a < n) out.push_back(a);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

uint64_t PrefixCache::extend_hash(uint64_t h, int64_t id) {
  const uint64_t v = static_cast<uint64_t>(id);
  for (int i = 0; i < 8; ++i) {
    h ^= (v >> (8 * i)) & 0xffu;
    h *= 0x100000001b3ull;
  }
  return h;
}

uint64_t PrefixCache::hash_prefix(const int64_t* ids, int64_t n, uint64_t seed) {
  uint64_t h = seed;
  for (int64_t i = 0; i < n; ++i) h = extend_hash(h, ids[i]);
  return h;
}

int PrefixCache::find_exact(const int64_t* ids, int64_t n, uint64_t hash) const {
  const auto range = by_hash_.equal_range(hash);
  int best = -1;
  for (auto it = range.first; it != range.second; ++it) {
    const Entry& e = entries_[static_cast<size_t>(it->second)];
    if (!e.live || e.position != n) continue;
    if (!std::equal(e.ids.begin(), e.ids.end(), ids)) continue;
    // Deterministic among duplicates (there should be none): the oldest.
    if (best < 0 || it->second < best) best = it->second;
  }
  return best;
}

int PrefixCache::lookup(const std::vector<int64_t>& prompt,
                        const std::vector<int64_t>& cuts,
                        const std::vector<uint64_t>& cut_hashes) const {
  if (cuts.size() != cut_hashes.size())
    throw std::invalid_argument("PrefixCache::lookup: cuts and hashes differ");
  for (size_t i = cuts.size(); i-- > 0;) {
    const int64_t c = cuts[i];
    if (c <= 0 || c >= static_cast<int64_t>(prompt.size())) continue;
    const int e = find_exact(prompt.data(), c, cut_hashes[i]);
    if (e >= 0) return e;
  }
  return -1;
}

int PrefixCache::take_free_slot() {
  if (free_.empty()) return -1;
  const int s = free_.front();
  free_.erase(free_.begin());
  return s;
}

void PrefixCache::give_back_slot(int slot) {
  if (slot < 0 || slot >= cfg_.slots)
    throw std::out_of_range("PrefixCache: slot out of range");
  const auto it = std::lower_bound(free_.begin(), free_.end(), slot);
  if (it != free_.end() && *it == slot)
    throw std::logic_error("PrefixCache: slot given back twice");
  free_.insert(it, slot);
}

int PrefixCache::evict_lru() {
  int victim = -1;
  for (size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    if (!e.live || e.attached > 0) continue;
    if (victim < 0 ||
        e.last_use < entries_[static_cast<size_t>(victim)].last_use)
      victim = static_cast<int>(i);
  }
  if (victim < 0) return -1;
  Entry& e = entries_[static_cast<size_t>(victim)];
  const int slot = e.slot;
  e.live = false;
  e.slot = -1;
  e.ids.clear();
  e.ids.shrink_to_fit();
  ++stats_.evictions;
  note(3, static_cast<uint64_t>(e.position), static_cast<uint64_t>(slot));
  return slot;
}

int PrefixCache::acquire_slot() {
  const int s = take_free_slot();
  if (s >= 0) return s;
  return evict_lru();
}

int PrefixCache::insert(const int64_t* ids, int64_t position, int slot,
                        uint64_t now) {
  if (position <= 0) throw std::invalid_argument("PrefixCache: empty entry");
  if (slot < 0 || slot >= cfg_.slots)
    throw std::out_of_range("PrefixCache: slot out of range");
  const uint64_t h = hash_prefix(ids, position);
  if (find_exact(ids, position, h) >= 0) {
    ++stats_.duplicates;
    return -1;
  }
  Entry e;
  e.ids.assign(ids, ids + position);
  e.position = position;
  e.slot = slot;
  e.hash = h;
  e.last_use = now;
  e.live = true;
  // Reuse a dead record's index when one exists (bounded memory), else
  // append; either way the choice is deterministic.
  int index = -1;
  for (size_t i = 0; i < entries_.size(); ++i)
    if (!entries_[i].live) {
      index = static_cast<int>(i);
      break;
    }
  if (index < 0) {
    entries_.push_back(std::move(e));
    index = static_cast<int>(entries_.size()) - 1;
  } else {
    // Drop the dead record's stale hash mapping.
    const uint64_t old = entries_[static_cast<size_t>(index)].hash;
    const auto range = by_hash_.equal_range(old);
    for (auto it = range.first; it != range.second; ++it)
      if (it->second == index) {
        by_hash_.erase(it);
        break;
      }
    entries_[static_cast<size_t>(index)] = std::move(e);
  }
  by_hash_.emplace(h, index);
  note(2, static_cast<uint64_t>(position), static_cast<uint64_t>(slot));
  return index;
}

void PrefixCache::attach(int index, uint64_t now) {
  Entry& e = entries_.at(static_cast<size_t>(index));
  if (!e.live) throw std::logic_error("PrefixCache: attach to a dead entry");
  ++e.attached;
  e.last_use = now;
  ++stats_.hits;
  stats_.tokens_saved += e.position;
  note(1, static_cast<uint64_t>(e.position), static_cast<uint64_t>(e.slot));
}

void PrefixCache::touch(int index, uint64_t now) {
  Entry& e = entries_.at(static_cast<size_t>(index));
  if (!e.live) throw std::logic_error("PrefixCache: touch of a dead entry");
  e.last_use = now;
}

void PrefixCache::detach(int index) {
  Entry& e = entries_.at(static_cast<size_t>(index));
  if (e.attached <= 0) throw std::logic_error("PrefixCache: detach below zero");
  --e.attached;
}

int64_t PrefixCache::blocks_pinned(int64_t block_tokens) const {
  if (block_tokens <= 0) return 0;
  int64_t n = 0;
  for (const Entry& e : entries_)
    if (e.live)
      n += e.position / block_tokens + (e.position % block_tokens != 0 ? 1 : 0);
  return n;
}

void PrefixCache::note(uint64_t a, uint64_t b, uint64_t c) {
  digest_ = extend_hash(digest_, static_cast<int64_t>(a));
  digest_ = extend_hash(digest_, static_cast<int64_t>(b));
  digest_ = extend_hash(digest_, static_cast<int64_t>(c));
}

}  // namespace dgpp::sched
