#pragma once

// GlmResidentImage: the byte image of one rank's resident layers, cached on
// the node's disk so the next startup streams device bytes instead of
// rebuilding them from the checkpoint.
//
// WHY. A rank's resident layer is the END of a pipeline — mmap the shard,
// slice the rank's rows/columns out of each tensor (a column slice pages
// in the whole matrix for a quarter of its bytes), copy to pinned staging,
// H2D, dequantize the bridge tensors, pack. At GLM that is ~130-180 s per
// rank of fault-driven, single-threaded work over 305 GB of mapped shards.
// The pipeline's OUTPUT is a flat byte range per layer (the layer bump:
// weights, scales, dequantized bridges, packs — no pointers), and the
// layout of the views INTO that range is reproducible without touching a
// source byte (the build's counting mode lays out every grant). So: dump
// each bump once, and next time lay out the views, pread the blob into
// staging, H2D. One sequential read of ~82 GiB per rank, no slicing, no
// dequant — the disk's rate is the load time.
//
// FILE. One file per (checkpoint, config, world, rank, head sharding,
// loader format version) — the key is the caller's; a mismatch replaces
// the file. Header + a fixed table (one entry per layer) + 4 KiB-aligned
// blobs. A layer's entry is published AFTER its blob is fully written and
// synced, so a crash mid-write leaves that layer absent (rebuilt and
// re-appended next time) and never a torn blob with a valid entry. Every
// entry carries a 64-bit word fold of its blob; verification on read is
// opt-in (it costs a pass over 82 GiB).
//
// Bitwise: the restored bytes ARE the built bytes — glm_loader_test pins a
// build-vs-restore round trip byte-for-byte on the fixture.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dgpp {

class GlmResidentImage {
 public:
  static constexpr uint32_t kFormatVersion = 1;

  // Opens `dir/<key hex>.img`, creating the directory and the file when
  // absent; a file whose header disagrees (version, key, layer count) is
  // replaced. Throws std::runtime_error on I/O failure.
  GlmResidentImage(const std::string& dir, uint64_t key, int layers);
  ~GlmResidentImage();
  GlmResidentImage(const GlmResidentImage&) = delete;
  GlmResidentImage& operator=(const GlmResidentImage&) = delete;

  int layers() const { return static_cast<int>(entries_.size()); }
  int present() const;  // layers with a published blob
  bool has_layer(int layer) const;
  size_t layer_bytes(int layer) const;  // 0 when absent
  const std::string& path() const { return path_; }

  // pread the layer's blob into `dst` (`bytes` must equal layer_bytes).
  // `verify` re-folds the bytes against the entry. Throws on mismatch.
  void read_layer(int layer, void* dst, size_t bytes, bool verify) const;

  // Appends the blob at the next 4 KiB boundary, fdatasyncs, then
  // publishes the entry. An already-present layer is rewritten (new blob,
  // old bytes orphaned — rebuilds are rare and the file is a cache).
  void write_layer(int layer, const void* src, size_t bytes);

  // 64-bit word fold (xor-multiply, the bus's shape); a byte tail is folded
  // as a partial word. Streams at memory speed, unlike a byte-serial hash.
  static uint64_t fold(const void* data, size_t bytes);

 private:
  struct Entry {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t fold = 0;
    uint64_t reserved = 0;
  };
  void write_fresh(uint64_t key);
  void write_entry(int layer) const;
  uint64_t table_offset(int layer) const;

  std::string path_;
  int fd_ = -1;
  std::vector<Entry> entries_;
};

}  // namespace dgpp
