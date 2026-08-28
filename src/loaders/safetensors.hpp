#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"

namespace dgpp {

struct TensorInfo {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  uint64_t data_begin = 0;  // byte offset into file data region
  uint64_t data_end = 0;
  const void* data = nullptr;  // mapped pointer (file lifetime)

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return data_end - data_begin; }
};

class SafetensorsFile {
 public:
  ~SafetensorsFile() {
    if (map_ != MAP_FAILED && map_) munmap(map_, map_len_);
    if (fd_ >= 0) ::close(fd_);
  }
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;

  static std::unique_ptr<SafetensorsFile> open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
      throw std::runtime_error(std::string("cannot open ") + path + ": " +
                               strerror(errno));
    struct stat st{};
    if (fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("fstat failed for " + path);
    }
    size_t len = static_cast<size_t>(st.st_size);
    void* map = mmap(nullptr, len ? len : 1, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("mmap failed for " + path);
    }
    std::unique_ptr<SafetensorsFile> f(new SafetensorsFile());
    f->path_ = path;
    f->fd_ = fd;
    f->map_ = static_cast<uint8_t*>(map);
    f->map_len_ = len;
    f->parse_header();
    return f;
  }

  const TensorInfo* find(std::string_view name) const {
    auto it = tensors_.find(std::string(name));
    return it == tensors_.end() ? nullptr : it->second.get();
  }

  const TensorInfo& at(std::string_view name) const {
    const TensorInfo* t = find(name);
    if (!t)
      throw std::runtime_error("tensor not found: " + std::string(name) +
                               " in " + path_);
    return *t;
  }

  // All tensors whose names match substring (cold-path convenience).
  std::vector<const TensorInfo*> grep(std::string_view substr) const {
    std::vector<const TensorInfo*> out;
    for (const auto& [n, t] : tensors_)
      if (n.find(substr) != std::string::npos) out.push_back(t.get());
    return out;
  }

  size_t tensor_count() const { return tensors_.size(); }
  const uint8_t* map_base() const { return map_; }
  size_t map_size() const { return map_len_; }
  const std::string& path() const { return path_; }
  const minijson::Value& header_meta() const { return meta_; }

  // Header-truth iteration (bind-check tooling): visits every tensor of this
  // shard. Fn takes const TensorInfo&.
  template <typename Fn>
  void for_each(Fn&& fn) const {
    for (const auto& kv : tensors_) fn(*kv.second);
  }

 private:
  SafetensorsFile() = default;

  void parse_header() {
    if (map_len_ < 8)
      throw std::runtime_error(path_ + ": too small for safetensors");
    uint64_t hlen;
    std::memcpy(&hlen, map_, 8);
    if (8 + hlen > map_len_)
      throw std::runtime_error(path_ + ": header length overruns file");
    std::string_view json(reinterpret_cast<const char*>(map_ + 8), hlen);
    auto parsed = minijson::parse(json);
    const minijson::Value& root = parsed.root;
    if (!root.is_object())
      throw std::runtime_error(path_ + ": safetensors header not an object");
    uint64_t data_start = 8 + hlen;
    for (const auto& m : root.members()) {
      if (m.key == "__metadata__") {
        meta_ = m.value;  // Value is copyable now
        continue;
      }
      const minijson::Value& desc = m.value;
      auto t = std::make_unique<TensorInfo>();
      t->name = m.key;
      auto ds = desc.at("dtype").as_string();
      auto dt = dtype_from_string(ds);
      if (!dt)
        throw std::runtime_error(path_ + ": unknown dtype " + std::string(ds));
      t->dtype = *dt;
      for (const auto& d : desc.at("shape").items())
        t->shape.push_back(d.as_int());
      const minijson::Value& offs = desc.at("data_offsets");
      t->data_begin = offs.items().at(0).as_int() + data_start;
      t->data_end = offs.items().at(1).as_int() + data_start;
      if (t->data_end > map_len_)
        throw std::runtime_error(path_ + ": tensor " + m.key + " overruns file");
      t->data = map_ + t->data_begin;
      tensors_[t->name] = std::move(t);
    }
  }

  std::string path_;
  int fd_ = -1;
  uint8_t* map_ = nullptr;
  size_t map_len_ = 0;
  std::unordered_map<std::string, std::unique_ptr<TensorInfo>> tensors_;
  minijson::Value meta_{};
};

}  // namespace dgpp
