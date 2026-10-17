#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dgpp::text::detail {

// A per-tokenizer, bounded cache of mapped pretokens. Exact key comparison
// makes collisions harmless. Replacement lets new traffic displace old words
// without an LRU list or unbounded growth. Misses compute outside the locks.
class WordCache {
 public:
  static constexpr size_t kMaxEntries = 10000;
  static constexpr size_t kMaxBytes = 256;

  explicit WordCache(size_t entries) : capacity_(entries) {
    if (entries > kMaxEntries)
      throw std::invalid_argument("tokenizer.word_cache_entries must be an integer in [0, 10000]");
    for (size_t i = 0; i < shards_.size(); ++i)
      shards_[i].entries.resize(entries / shards_.size() + (i < entries % shards_.size()));
  }

  template <class Compute>
  void append(std::string_view key, std::vector<int64_t>* out, Compute&& compute) {
    if (capacity_ == 0 || key.empty() || key.size() > kMaxBytes) {
      compute();
      return;
    }
    const size_t index = std::hash<std::string_view>{}(key) % capacity_;
    Shard& shard = shards_[index % shards_.size()];
    Entry& slot = shard.entries[index / shards_.size()];
    {
      std::unique_lock lock(shard.mutex, std::try_to_lock);
      if (lock && slot.key == key) {
        out->insert(out->end(), slot.ids.begin(), slot.ids.end());
        return;
      }
    }
    const size_t start = out->size();
    compute();
    // One byte-level symbol produces at most one id. Guard the value bound
    // too, so the cache cannot retain an unexpectedly large result.
    if (out->size() - start > kMaxBytes) return;
    // Prepare the owned key and ids outside the critical section. A failed
    // try-lock only loses a cache fill; the caller already has its result.
    Entry fresh{std::string(key), {out->begin() + start, out->end()}};
    {
      std::unique_lock lock(shard.mutex, std::try_to_lock);
      if (lock) {
        slot.key.swap(fresh.key);
        slot.ids.swap(fresh.ids);
      }
    }
  }

 private:
  friend struct WordCacheTestAccess;
  struct Entry {
    std::string key;
    std::vector<int64_t> ids;
  };
  struct Shard {
    std::mutex mutex;
    std::vector<Entry> entries;
  };
  const size_t capacity_;
  std::array<Shard, 8> shards_;
};

}  // namespace dgpp::text::detail
