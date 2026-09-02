#include "models/glm_resident_image.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "common/log.hpp"

namespace dgpp {

namespace {

constexpr char kMagic[8] = {'D', 'G', 'P', 'P', 'R', 'I', 'M', 'G'};
constexpr size_t kBlobAlign = 4096;

struct Header {
  char magic[8];
  uint32_t version;
  uint32_t layers;
  uint64_t key;
  uint64_t reserved[5];
};
static_assert(sizeof(Header) == 64, "resident image header is 64 bytes");

[[noreturn]] void fail(const std::string& what, const std::string& path) {
  throw std::runtime_error("resident image " + path + ": " + what + " (" +
                           std::strerror(errno) + ")");
}

void pwrite_all(int fd, const void* src, size_t bytes, uint64_t offset,
                const std::string& path) {
  const uint8_t* p = static_cast<const uint8_t*>(src);
  while (bytes > 0) {
    // Linux caps a single pwrite at ~2 GiB; loop in 1 GiB pieces.
    const size_t chunk = bytes < (1u << 30) ? bytes : (1u << 30);
    const ssize_t n = ::pwrite(fd, p, chunk, static_cast<off_t>(offset));
    if (n <= 0) fail("write failed", path);
    p += n;
    bytes -= static_cast<size_t>(n);
    offset += static_cast<uint64_t>(n);
  }
}

void pread_all(int fd, void* dst, size_t bytes, uint64_t offset,
               const std::string& path) {
  uint8_t* p = static_cast<uint8_t*>(dst);
  while (bytes > 0) {
    const size_t chunk = bytes < (1u << 30) ? bytes : (1u << 30);
    const ssize_t n = ::pread(fd, p, chunk, static_cast<off_t>(offset));
    if (n <= 0) fail("read failed or short", path);
    p += n;
    bytes -= static_cast<size_t>(n);
    offset += static_cast<uint64_t>(n);
  }
}

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

}  // namespace

uint64_t GlmResidentImage::fold(const void* data, size_t bytes) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint64_t h = 0x9E3779B97F4A7C15ULL ^ bytes;
  size_t i = 0;
  for (; i + 8 <= bytes; i += 8) {
    uint64_t w;
    std::memcpy(&w, p + i, 8);
    h = (h ^ w) * 0x100000001B3ULL;
  }
  if (i < bytes) {
    uint64_t w = 0;
    std::memcpy(&w, p + i, bytes - i);
    h = (h ^ w) * 0x100000001B3ULL;
  }
  return h;
}

GlmResidentImage::GlmResidentImage(const std::string& dir, uint64_t key,
                                   int layers) {
  if (layers <= 0) throw std::invalid_argument("resident image: no layers");
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  char name[32];
  std::snprintf(name, sizeof(name), "%016llx.img",
                static_cast<unsigned long long>(key));
  path_ = (std::filesystem::path(dir) / name).string();
  fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd_ < 0) fail("open failed", path_);
  entries_.assign(static_cast<size_t>(layers), Entry{});

  struct stat st{};
  if (fstat(fd_, &st) != 0) fail("fstat failed", path_);
  const uint64_t table_bytes = sizeof(Header) + sizeof(Entry) * entries_.size();
  bool fresh = static_cast<uint64_t>(st.st_size) < table_bytes;
  if (!fresh) {
    Header h{};
    pread_all(fd_, &h, sizeof(h), 0, path_);
    fresh = std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0 ||
            h.version != kFormatVersion || h.key != key ||
            h.layers != static_cast<uint32_t>(layers);
    if (fresh)
      DGPP_LOG_WARN("resident image {}: header mismatch (version {} key {:#x} "
                    "layers {}) — replacing",
                    path_, h.version, h.key, h.layers);
  }
  if (fresh) {
    write_fresh(key);
  } else {
    pread_all(fd_, entries_.data(), sizeof(Entry) * entries_.size(),
              sizeof(Header), path_);
    // An entry whose blob would run past the file is a torn write that
    // somehow got published — treat it as absent rather than trust it.
    for (Entry& e : entries_)
      if (e.bytes != 0 &&
          e.offset + e.bytes > static_cast<uint64_t>(st.st_size))
        e = Entry{};
  }
}

GlmResidentImage::~GlmResidentImage() {
  if (fd_ >= 0) ::close(fd_);
}

void GlmResidentImage::write_fresh(uint64_t key) {
  if (ftruncate(fd_, 0) != 0) fail("truncate failed", path_);
  Header h{};
  std::memcpy(h.magic, kMagic, sizeof(kMagic));
  h.version = kFormatVersion;
  h.layers = static_cast<uint32_t>(entries_.size());
  h.key = key;
  pwrite_all(fd_, &h, sizeof(h), 0, path_);
  for (Entry& e : entries_) e = Entry{};
  pwrite_all(fd_, entries_.data(), sizeof(Entry) * entries_.size(),
             sizeof(Header), path_);
  if (fdatasync(fd_) != 0) fail("fdatasync failed", path_);
}

int GlmResidentImage::present() const {
  int n = 0;
  for (const Entry& e : entries_) n += e.bytes != 0;
  return n;
}

bool GlmResidentImage::has_layer(int layer) const {
  return layer >= 0 && layer < layers() &&
         entries_[static_cast<size_t>(layer)].bytes != 0;
}

size_t GlmResidentImage::layer_bytes(int layer) const {
  return has_layer(layer) ? entries_[static_cast<size_t>(layer)].bytes : 0;
}

uint64_t GlmResidentImage::table_offset(int layer) const {
  return sizeof(Header) + sizeof(Entry) * static_cast<uint64_t>(layer);
}

void GlmResidentImage::write_entry(int layer) const {
  pwrite_all(fd_, &entries_[static_cast<size_t>(layer)], sizeof(Entry),
             table_offset(layer), path_);
}

void GlmResidentImage::read_layer(int layer, void* dst, size_t bytes,
                                  bool verify) const {
  if (!has_layer(layer))
    throw std::runtime_error("resident image " + path_ + ": layer " +
                             std::to_string(layer) + " is absent");
  const Entry& e = entries_[static_cast<size_t>(layer)];
  if (e.bytes != bytes)
    throw std::runtime_error(
        "resident image " + path_ + ": layer " + std::to_string(layer) +
        " holds " + std::to_string(e.bytes) + " bytes, the build formula says " +
        std::to_string(bytes) + " (stale image for this loader — delete it)");
  pread_all(fd_, dst, bytes, e.offset, path_);
  if (verify && fold(dst, bytes) != e.fold)
    throw std::runtime_error("resident image " + path_ + ": layer " +
                             std::to_string(layer) +
                             " failed verification (corrupt blob)");
}

void GlmResidentImage::write_layer(int layer, const void* src, size_t bytes) {
  if (layer < 0 || layer >= layers() || bytes == 0)
    throw std::invalid_argument("resident image: bad layer/bytes");
  struct stat st{};
  if (fstat(fd_, &st) != 0) fail("fstat failed", path_);
  const uint64_t offset =
      align_up(static_cast<uint64_t>(st.st_size), kBlobAlign);
  pwrite_all(fd_, src, bytes, offset, path_);
  // The blob must be durable before its entry says it exists.
  if (fdatasync(fd_) != 0) fail("fdatasync failed", path_);
  Entry& e = entries_[static_cast<size_t>(layer)];
  e.offset = offset;
  e.bytes = bytes;
  e.fold = fold(src, bytes);
  write_entry(layer);
}

}  // namespace dgpp
