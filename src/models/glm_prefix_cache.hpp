#pragma once
// The prefix cache's host half (M7 stage B, DESIGN §8): the index of
// snapshot entries the scheduler consults at admission, the slot ledger of
// the engine's snapshot arena, and the decision digest every rank compares.
//
// WHAT AN ENTRY IS: a session's state at a pool-aligned position P of a
// token sequence — the KDA slots, the DSA tail rings, the draft block's
// last hidden row, the DSA blocks below P (the full ones pinned by
// reference, the partial one copied) — held in one arena slot on the
// engine, keyed by the sequence's first P ids. A request whose prompt
// begins with those P ids and whose COLD prefill would cut at P attaches
// to it and prefills only the suffix, bitwise the cold run (stage A's
// guarantee: the hot path replays the cold path's chunk sequence).
//
// WHERE ENTRIES COME FROM: (a) a cold prefill takes one at the deepest cut
// of its prompt (the aligned image of the last structural boundary, in a
// chat the assistant header — exactly where the NEXT turn's cold prefill
// cuts, since the boundaries are positions of role-marker tokens in the
// shared prefix); (b) a live request keeps a ROLLING snapshot at its
// latest aligned committed position (one arena slot per live request,
// overwritten every kpool tokens) which becomes an entry when the request
// retires — at floor((end - 1) / kpool) * kpool, the aligned image of its
// EOS token's position, where the next turn's `<|user|>` marker cuts.
//
// DETERMINISM: every method is a pure function of its arguments and the
// cache's own history — the entries are a vector in insertion order, the
// free slots a sorted list, the LRU a scan by (last_use, index) — so the
// ranks of a fabric, fed the same journaled requests, make the same
// decisions; digest() folds every one of them into a value the journal
// carries for the peers to compare.
//
// LOOKUP is exact: a 64-bit prefix hash narrows the candidates and the ids
// are compared in full (a hash collision can never attach the wrong
// state). The candidates are the prompt's own cut positions, so a match
// is by construction a position the cold prefill would also cut at.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dgpp::glm {

class PrefixCache {
 public:
  struct Config {
    int slots = 0;              // arena slots the engine holds (0: off)
    int64_t align = 1;          // kpool: every entry position is a multiple
    int64_t chunk_tokens = 2048;  // the cold prefill's chunk length
  };
  struct Entry {
    std::vector<int64_t> ids;   // the first `position` ids of the sequence
    int64_t position = 0;
    int slot = -1;              // the arena slot; -1 once evicted
    uint64_t hash = 0;
    uint64_t last_use = 0;      // the tick of the last attach / insert
    int attached = 0;           // live requests attached (never evicted)
    bool live = false;
  };
  struct Stats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t tokens_saved = 0;   // sum of attach positions
    int64_t snapshots = 0;      // entries taken at a prefill cut
    int64_t close_entries = 0;  // entries taken from a rolling snapshot
    int64_t rolling = 0;        // rolling snapshots taken (the hops included)
    int64_t hops = 0;           // of them, taken from a two-row step's first row
    int64_t evictions = 0;
    int64_t duplicates = 0;     // an entry already existed at the position
    int64_t skipped_no_slot = 0;  // a snapshot wanted, no slot free or evictable
    int64_t skipped_no_block = 0;  // a snapshot wanted, no pool block for its partial copy
  };

  PrefixCache() = default;
  explicit PrefixCache(const Config& cfg);

  bool enabled() const { return cfg_.slots > 0; }
  const Config& config() const { return cfg_; }
  int slots() const { return cfg_.slots; }
  int free_slots() const { return static_cast<int>(free_.size()); }
  int live_entries() const;
  const Stats& stats() const { return stats_; }
  Stats& stats() { return stats_; }

  // The cold prefill's cut positions in (0, n), ascending and unique: every
  // multiple of chunk_tokens and the aligned image floor(b / align) * align
  // of every boundary b (structural positions in the prompt: role-marker
  // tokens, or whatever the caller derives).
  std::vector<int64_t> cuts(int64_t n,
                            const std::vector<int64_t>& boundaries) const;

  // The prefix hash of ids[0..n): FNV-1a over the id bytes, extendable
  // (hash(ids, n + 1) continues hash(ids, n) with one more id).
  static uint64_t hash_prefix(const int64_t* ids, int64_t n,
                              uint64_t seed = kSeed);
  static uint64_t extend_hash(uint64_t h, int64_t id);
  static constexpr uint64_t kSeed = 0xcbf29ce484222325ull;

  // The deepest live entry whose position is one of `cuts` (all < the
  // prompt's length) and whose ids equal the prompt's first `position`
  // ids. `cut_hashes[i]` is hash_prefix(prompt, cuts[i]). -1: none.
  int lookup(const std::vector<int64_t>& prompt,
             const std::vector<int64_t>& cuts,
             const std::vector<uint64_t>& cut_hashes) const;
  // Whether a live entry with exactly these ids exists (the dedupe check).
  int find_exact(const int64_t* ids, int64_t n, uint64_t hash) const;

  // ---- the slot ledger ----------------------------------------------------
  // The smallest free slot, or -1.
  int take_free_slot();
  void give_back_slot(int slot);
  // The least recently used live entry with no attached request: its slot
  // is freed (the caller releases it on the engine) and the entry dies.
  // Returns the slot, or -1 when nothing is evictable.
  int evict_lru();
  // A free slot, evicting once when none is free. -1 when neither works.
  int acquire_slot();

  // ---- entries -----------------------------------------------------------
  // Inserts an entry over ids[0..position) in `slot`. Returns its index,
  // or -1 when an identical entry is live (the caller keeps the slot out
  // of the entry and gives it back).
  int insert(const int64_t* ids, int64_t position, int slot, uint64_t now);
  const Entry& entry(int index) const { return entries_.at(static_cast<size_t>(index)); }
  void attach(int index, uint64_t now);
  void detach(int index);
  // Refreshes an entry's LRU stamp without a decision (no stats, no digest).
  void touch(int index, uint64_t now);
  int64_t blocks_pinned(int64_t block_tokens) const;  // every live entry's

  // ---- the decision digest -----------------------------------------------
  // Folds a decision (an attach, a snapshot, an eviction ...) into the
  // running digest; the journal carries it so every rank can compare.
  void note(uint64_t a, uint64_t b, uint64_t c);
  uint64_t digest() const { return digest_; }

 private:
  Config cfg_;
  std::vector<Entry> entries_;
  std::unordered_multimap<uint64_t, int> by_hash_;  // hash -> entry index
  std::vector<int> free_;  // sorted ascending
  Stats stats_;
  uint64_t digest_ = kSeed;
};

}  // namespace dgpp::glm
