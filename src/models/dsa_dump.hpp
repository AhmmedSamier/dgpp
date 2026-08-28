#pragma once
// Reader for reference dumps produced by tools/dsa_reference_dump.py
// (M3 deliverable). Same container as the KDA dump: 8-byte magic
// "DGPPDSAD", u32 version, u32 header length, JSON header, then a flat
// little-endian payload the header maps tensor names into. Weights, input
// activations, and reference outputs (layer output, index cache, latent
// rows, tail, top-k) for one DSA layer travel together so the C++ parity
// test exercises the full layer against an independently computed result.
//
// File format details (shared contract with the Python tool):
//   header["tensors"][name] = {"dtype","shape","offset","nbytes"}
//   offsets are relative to the start of the payload region.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"
#include "models/dsa_geometry.hpp"

namespace dgpp {

class DsaDumpFile {
 public:
  struct TensorView {
    DType dtype = DType::BF16;
    std::vector<int64_t> shape;
    const void* data = nullptr;
    size_t nbytes = 0;
    size_t numel() const;
  };

  // Loads and validates the whole file into memory.
  static DsaDumpFile load(const std::string& path);

  const std::string& model() const { return model_; }
  const std::string& revision() const { return revision_; }
  const std::string& backend() const { return backend_; }
  int layer_idx() const { return layer_idx_; }
  const minijson::Value& config_json() const { return config_; }

  // Single-layer DsaConfig from the dump's config block (tp_size=1,
  // num_dsa_layers=1: dumps carry exactly one layer).
  DsaConfig single_layer_config() const;

  bool has_tensor(const std::string& name) const;
  const TensorView& tensor(const std::string& name) const;

 private:
  std::vector<uint8_t> bytes_;
  std::string model_, revision_, backend_;
  int layer_idx_ = -1;
  minijson::Value config_;
  std::map<std::string, TensorView> tensors_;
};

}  // namespace dgpp
