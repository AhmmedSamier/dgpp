#include "models/dsa_dump.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace dgpp {

size_t DsaDumpFile::TensorView::numel() const {
  size_t n = 1;
  for (int64_t d : shape) n *= static_cast<size_t>(d);
  return n;
}

DsaDumpFile DsaDumpFile::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("dsa dump: cannot open " + path);
  const std::streamsize size = f.tellg();
  if (size < 16) throw std::runtime_error("dsa dump: truncated " + path);
  f.seekg(0);
  DsaDumpFile d;
  d.bytes_.resize(static_cast<size_t>(size));
  f.read(reinterpret_cast<char*>(d.bytes_.data()), size);
  if (!f) throw std::runtime_error("dsa dump: short read " + path);

  const uint8_t* p = d.bytes_.data();
  static constexpr uint8_t kMagic[8] = {'D', 'G', 'P', 'P', 'D', 'S', 'A', 'D'};
  if (std::memcmp(p, kMagic, 8) != 0)
    throw std::runtime_error("dsa dump: bad magic");
  uint32_t version = 0, header_len = 0;
  std::memcpy(&version, p + 8, 4);
  std::memcpy(&header_len, p + 12, 4);
  if (version != 1)
    throw std::runtime_error("dsa dump: unsupported version " +
                             std::to_string(version));
  if (16 + static_cast<uint64_t>(header_len) > d.bytes_.size())
    throw std::runtime_error("dsa dump: header length exceeds file");

  const std::string_view header_json(
      reinterpret_cast<const char*>(p + 16), header_len);
  minijson::ParseResult parsed;
  try {
    parsed = minijson::parse(header_json);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("dsa dump: header JSON: ") + e.what());
  }
  if (!parsed.root.is_object())
    throw std::runtime_error("dsa dump: header is not an object");
  const minijson::Value& root = parsed.root;

  d.model_ = std::string(root.at("model").as_string());
  d.revision_ = std::string(root.at("revision").as_string());
  d.backend_ = std::string(root.at("backend").as_string());
  d.layer_idx_ = static_cast<int>(root.at("layer_idx").as_int());
  d.config_ = root.at("config");

  const size_t payload_base = 16 + header_len;
  const size_t payload_size = d.bytes_.size() - payload_base;
  const minijson::Value& tensors = root.at("tensors");
  if (!tensors.is_object())
    throw std::runtime_error("dsa dump: missing tensors map");
  for (const auto& m : tensors.members()) {
    TensorView tv;
    const std::string dtype = std::string(m.value.at("dtype").as_string());
    auto dt = dtype_from_string(dtype);
    if (!dt) throw std::runtime_error("dsa dump: unknown dtype " + dtype);
    tv.dtype = *dt;
    for (const auto& dim : m.value.at("shape").items())
      tv.shape.push_back(dim.as_int());
    const uint64_t offset = m.value.at("offset").as_int();
    const uint64_t nbytes = m.value.at("nbytes").as_int();
    if (offset > payload_size || nbytes > payload_size - offset)
      throw std::runtime_error("dsa dump: tensor '" + m.key +
                               "' exceeds payload");
    if (dtype_size(tv.dtype) * tv.numel() != nbytes)
      throw std::runtime_error("dsa dump: tensor '" + m.key +
                               "' nbytes disagrees with shape");
    tv.data = d.bytes_.data() + payload_base + offset;
    tv.nbytes = static_cast<size_t>(nbytes);
    d.tensors_[m.key] = tv;
  }
  return d;
}

DsaConfig DsaDumpFile::single_layer_config() const {
  DsaConfig cfg;
  cfg.hidden = static_cast<int>(config_.at("hidden").as_int());
  cfg.num_heads = static_cast<int>(config_.at("heads").as_int());
  cfg.q_lora_rank = static_cast<int>(config_.at("q_lora").as_int());
  cfg.kv_lora_rank = static_cast<int>(config_.at("kv_lora").as_int());
  cfg.qk_nope_head_dim = static_cast<int>(config_.at("nope").as_int());
  cfg.qk_rope_head_dim = 0;  // rope-free path only
  cfg.v_head_dim = static_cast<int>(config_.at("v_dim").as_int());
  cfg.index_n_heads = static_cast<int>(config_.at("idx_heads").as_int());
  cfg.index_head_dim = static_cast<int>(config_.at("idx_dim").as_int());
  cfg.index_topk = static_cast<int>(config_.at("topk").as_int());
  cfg.index_kpool = static_cast<int>(config_.at("kpool").as_int());
  cfg.num_dsa_layers = 1;
  cfg.block_tokens = 128;
  cfg.tp_size = 1;
  cfg.rms_norm_eps = 1e-5f;
  DsaConfig::validate_config(cfg);
  // The kernel constraint set (must match what the layer checks at
  // construction): the select kernels pin index_n_heads to 32.
  DsaGeometry::from_config(cfg);
  if (cfg.index_n_heads != 32)
    throw std::runtime_error(
        "dsa dump: index_n_heads must be 32 (select kernel pin)");
  return cfg;
}

bool DsaDumpFile::has_tensor(const std::string& name) const {
  return tensors_.count(name) != 0;
}

const DsaDumpFile::TensorView& DsaDumpFile::tensor(
    const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error("dsa dump: missing tensor '" + name + "'");
  return it->second;
}

}  // namespace dgpp
