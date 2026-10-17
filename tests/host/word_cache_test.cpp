#include "text/word_cache.hpp"

#include <atomic>
#include <future>
#include <thread>

#include "common/test.hpp"

namespace dgpp::text::detail {
struct WordCacheTestAccess {
  static std::mutex& mutex(WordCache& cache) { return cache.shards_[0].mutex; }
  static size_t slots(const WordCache& cache) {
    size_t count = 0;
    for (const auto& shard : cache.shards_) count += shard.entries.size();
    return count;
  }
};
}  // namespace dgpp::text::detail

namespace {
using dgpp::text::detail::WordCache;
using dgpp::text::detail::WordCacheTestAccess;
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

DGPP_TEST(word_cache_hits_skip_computation_and_own_only_the_word_result) {
  WordCache cache(10);
  int calls = 0;
  std::string key = "hello";
  std::vector<int64_t> out{99};
  cache.append(key, &out, [&] {
    ++calls;
    out.insert(out.end(), {1, 2});
  });
  key.assign("xxxxx");
  out = {88};
  cache.append("hello", &out, [&] { ++calls; });
  require(calls == 1 && out == std::vector<int64_t>({88, 1, 2}),
          "hit must own and append exactly the cached ids");
}

DGPP_TEST(word_cache_bounds_and_replacement) {
  for (size_t slots : {0u, 1u, 7u, 8u, 9u, 10000u}) {
    WordCache cache(slots);
    require(WordCacheTestAccess::slots(cache) == slots,
            "physical slot count must equal configured capacity");
  }
  bool refused = false;
  try {
    WordCache cache(10001);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "oversized cache must be refused");
  WordCache cache(1);
  int calls = 0;
  for (auto key : {"first", "second", "first", "first"}) {
    std::vector<int64_t> out;
    cache.append(key, &out, [&] {
      ++calls;
      out.push_back(key[0]);
    });
    require(out == std::vector<int64_t>{key[0]}, "collision must compare the complete key");
  }
  require(calls == 3, "replacement must adapt after filling the cache");
  for (size_t n : {256u, 257u}) {
    calls = 0;
    for (int i = 0; i < 2; ++i) {
      std::vector<int64_t> out;
      cache.append(std::string(n, 'x'), &out, [&] {
        ++calls;
        out.assign(n, 1);
      });
    }
    require(calls == (n == 256 ? 1 : 2), "mapped byte limit must apply before lookup and fill");
  }
  for (int i = 0; i < 2; ++i) {
    std::vector<int64_t> out;
    cache.append("oversized-result", &out, [&] { out.assign(257, i); });
    require(out.front() == i, "oversized values must not be retained");
  }
  WordCache off(0);
  calls = 0;
  for (int i = 0; i < 2; ++i) {
    std::vector<int64_t> out;
    off.append("word", &out, [&] {
      ++calls;
      out.push_back(i);
    });
  }
  require(calls == 2, "disabled cache must always compute");
}

DGPP_TEST(word_cache_contention_does_not_wait_or_change_results) {
  WordCache cache(1);
  std::unique_lock held(WordCacheTestAccess::mutex(cache));
  auto future = std::async(std::launch::async, [&] {
    std::vector<int64_t> out;
    cache.append("busy", &out, [&] { out.push_back(42); });
    return out;
  });
  const bool ready = future.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  held.unlock();
  require(future.get() == std::vector<int64_t>{42}, "contention changed result");
  require(ready, "lookup or fill blocked on the cache lock");
}

DGPP_TEST(word_cache_parallel_replacement_is_exact) {
  WordCache cache(31);
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t)
    threads.emplace_back([&, t] {
      for (int i = 0; i < 10000; ++i) {
        const int64_t value = (i * 17 + t) % 97;
        const std::string key = std::to_string(value);
        std::vector<int64_t> out{999};
        cache.append(key, &out, [&] { out.insert(out.end(), {value, -value}); });
        if (out != std::vector<int64_t>({999, value, -value})) failed = true;
      }
    });
  for (auto& thread : threads) thread.join();
  require(!failed, "parallel cache hit returned another word's ids");
  require(WordCacheTestAccess::slots(cache) == 31, "cache grew under contention");
}
}  // namespace

int main() {
  return dgpp::test::run_all();
}
