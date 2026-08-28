#include "models/glm_dump.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace dgpp {

size_t GlmDumpFile::TensorView::numel() const {
  size_t n = 1;
  for (int64_t d : shape) n *= static_cast<size_t>(d);
  return n;
}

GlmDumpFile GlmDumpFile::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("glm dump: cannot open " + path);
  const std::streamsize size = f.tellg();
  if (size < 16) throw std::runtime_error("glm dump: truncated " + path);
  f.seekg(0);
  GlmDumpFile d;
  d.bytes_.resize(static_cast<size_t>(size));
  f.read(reinterpret_cast<char*>(d.bytes_.data()), size);
  if (!f) throw std::runtime_error("glm dump: short read " + path);

  const uint8_t* p = d.bytes_.data();
  static constexpr uint8_t kMagic[8] = {'D', 'G', 'P', 'P', 'G', 'L', 'M', 'D'};
  if (std::memcmp(p, kMagic, 8) != 0)
    throw std::runtime_error("glm dump: bad magic");
  uint32_t version = 0, header_len = 0;
  std::memcpy(&version, p + 8, 4);
  std::memcpy(&header_len, p + 12, 4);
  if (version != 1)
    throw std::runtime_error("glm dump: unsupported version " +
                             std::to_string(version));
  if (16 + static_cast<uint64_t>(header_len) > d.bytes_.size())
    throw std::runtime_error("glm dump: header length exceeds file");

  const std::string_view header_json(
      reinterpret_cast<const char*>(p + 16), header_len);
  minijson::ParseResult parsed;
  try {
    parsed = minijson::parse(header_json);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("glm dump: header JSON: ") + e.what());
  }
  if (!parsed.root.is_object())
    throw std::runtime_error("glm dump: header is not an object");
  const minijson::Value& root = parsed.root;

  d.model_ = std::string(root.at("model").as_string());
  d.revision_ = std::string(root.at("revision").as_string());
  d.backend_ = std::string(root.at("backend").as_string());

  const minijson::Value& cfg = root.at("config");
  d.hidden_ = static_cast<int>(cfg.at("hidden").as_int());
  d.vocab_ = static_cast<int>(cfg.at("vocab").as_int());
  d.num_layers_ = static_cast<int>(cfg.at("num_layers").as_int());
  d.top_k_ = static_cast<int>(cfg.at("top_k").as_int());
  d.token_count_ = cfg.at("tokens").as_int();
  if (d.hidden_ <= 0 || d.vocab_ <= 0 || d.num_layers_ < 0 || d.top_k_ <= 0 ||
      d.token_count_ <= 0)
    throw std::runtime_error("glm dump: bad config block");

  if (root.find("route_layers")) {
    for (const auto& rl : root.at("route_layers").items()) {
      RouteLayer r;
      r.layer_idx = static_cast<int>(rl.at("layer_idx").as_int());
      r.top_k = static_cast<int>(rl.at("top_k").as_int());
      r.tokens = rl.at("tokens").as_int();
      if (r.layer_idx < 0 || r.top_k <= 0 || r.tokens <= 0 ||
          r.tokens != d.token_count_)
        throw std::runtime_error("glm dump: bad route_layers entry");
      d.route_layers_.push_back(r);
    }
  }

  const size_t payload_base = 16 + header_len;
  const size_t payload_size = d.bytes_.size() - payload_base;
  const minijson::Value& tensors = root.at("tensors");
  if (!tensors.is_object())
    throw std::runtime_error("glm dump: missing tensors map");
  for (const auto& m : tensors.members()) {
    TensorView tv;
    const std::string dtype = std::string(m.value.at("dtype").as_string());
    const auto dt = dtype_from_string(dtype);
    if (!dt) throw std::runtime_error("glm dump: unknown dtype " + dtype);
    tv.dtype = *dt;
    for (const auto& dim : m.value.at("shape").items())
      tv.shape.push_back(dim.as_int());
    const uint64_t offset = m.value.at("offset").as_int();
    const uint64_t nbytes = m.value.at("nbytes").as_int();
    if (offset > payload_size || nbytes > payload_size - offset)
      throw std::runtime_error("glm dump: tensor '" + m.key +
                               "' exceeds payload");
    if (dtype_size(tv.dtype) * tv.numel() != nbytes)
      throw std::runtime_error("glm dump: tensor '" + m.key +
                               "' nbytes disagrees with shape");
    tv.data = d.bytes_.data() + payload_base + offset;
    tv.nbytes = static_cast<size_t>(nbytes);
    d.tensors_[m.key] = tv;
  }
  return d;
}

bool GlmDumpFile::has_tensor(const std::string& name) const {
  return tensors_.count(name) != 0;
}

const GlmDumpFile::TensorView& GlmDumpFile::tensor(
    const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error("glm dump: missing tensor '" + name + "'");
  return it->second;
}

namespace {

const GlmDumpFile::TensorView* require_tensor(const GlmDumpFile& d,
                                              const std::string& name,
                                              DType want) {
  const auto& tv = d.tensor(name);
  if (tv.dtype != want)
    throw std::runtime_error("glm dump: tensor '" + name + "' dtype mismatch");
  return &tv;
}

}  // namespace

const int64_t* GlmDumpFile::tokens() const {
  const auto* tv = require_tensor(*this, "tokens", DType::I64);
  if (tv->shape.size() != 1 ||
      tv->shape[0] != static_cast<int64_t>(token_count_))
    throw std::runtime_error("glm dump: tokens shape mismatch");
  return static_cast<const int64_t*>(tv->data);
}

const uint16_t* GlmDumpFile::final_hidden() const {
  const auto* tv = require_tensor(*this, "final_hidden", DType::BF16);
  if (tv->shape.size() != 2 || tv->shape[0] != token_count_ ||
      tv->shape[1] != hidden_)
    throw std::runtime_error("glm dump: final_hidden shape mismatch");
  return static_cast<const uint16_t*>(tv->data);
}

const int32_t* GlmDumpFile::topk_ids() const {
  const auto* tv = require_tensor(*this, "topk_ids", DType::I32);
  if (tv->shape.size() != 2 || tv->shape[0] != token_count_ ||
      tv->shape[1] != top_k_)
    throw std::runtime_error("glm dump: topk_ids shape mismatch");
  return static_cast<const int32_t*>(tv->data);
}

const float* GlmDumpFile::topk_logits() const {
  const auto* tv = require_tensor(*this, "topk_logits", DType::F32);
  if (tv->shape.size() != 2 || tv->shape[0] != token_count_ ||
      tv->shape[1] != top_k_)
    throw std::runtime_error("glm dump: topk_logits shape mismatch");
  return static_cast<const float*>(tv->data);
}

const int32_t* GlmDumpFile::route_ids() const {
  const auto* tv = require_tensor(*this, "route_ids", DType::I32);
  int64_t total = 0;
  for (const auto& r : route_layers_) total += r.tokens * r.top_k;
  if (tv->shape.size() != 1 || tv->shape[0] != total)
    throw std::runtime_error("glm dump: route_ids shape mismatch");
  return static_cast<const int32_t*>(tv->data);
}

const float* GlmDumpFile::route_weights() const {
  const auto* tv = require_tensor(*this, "route_weights", DType::F32);
  int64_t total = 0;
  for (const auto& r : route_layers_) total += r.tokens * r.top_k;
  if (tv->shape.size() != 1 || tv->shape[0] != total)
    throw std::runtime_error("glm dump: route_weights shape mismatch");
  return static_cast<const float*>(tv->data);
}

}  // namespace dgpp
