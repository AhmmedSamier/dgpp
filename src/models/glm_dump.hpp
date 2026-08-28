#pragma once
// Reader for full-model reference dumps produced by
// tools/glm_reference_dump.py (M4 chunk 6, DESIGN §7.5). Same container as
// the KDA/DSA dumps: 8-byte magic "DGPPGLMD", u32 version, u32 header
// length, JSON header, flat little-endian payload. A dump carries the input
// token ids, the reference final hidden state, per-token top-k logits, and
// the reference per-layer routing decisions — everything the engine-side
// comparison needs, nothing it does not (weights stay in the checkpoint).
//
// Shared contract with the Python tool:
//   header["tensors"][name] = {"dtype","shape","offset","nbytes"}
//   header["config"] = {"hidden","vocab","num_layers","tokens","top_k"}
//   header["route_layers"] = [{"layer_idx","top_k","tokens"}] aligned with
//     the flat route_ids/route_weights tensors.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"

namespace dgpp {

class GlmDumpFile {
 public:
  struct TensorView {
    DType dtype = DType::BF16;
    std::vector<int64_t> shape;
    const void* data = nullptr;
    size_t nbytes = 0;
    size_t numel() const;
  };

  struct RouteLayer {
    int layer_idx = -1;
    int top_k = 0;
    int64_t tokens = 0;
  };

  static GlmDumpFile load(const std::string& path);

  const std::string& model() const { return model_; }
  const std::string& revision() const { return revision_; }
  const std::string& backend() const { return backend_; }
  int hidden() const { return hidden_; }
  int vocab() const { return vocab_; }
  int num_layers() const { return num_layers_; }
  int top_k() const { return top_k_; }
  int64_t token_count() const { return token_count_; }
  const std::vector<RouteLayer>& route_layers() const { return route_layers_; }

  bool has_tensor(const std::string& name) const;
  const TensorView& tensor(const std::string& name) const;

  // Typed views over the fixed tensor set (throw when absent).
  // tokens: I64 [token_count]; final_hidden: BF16 [token_count, hidden];
  // topk_ids: I32 [token_count, top_k]; topk_logits: F32 [token_count,
  // top_k]; route_ids/route_weights: flat over route_layers().
  const int64_t* tokens() const;
  const uint16_t* final_hidden() const;
  const int32_t* topk_ids() const;
  const float* topk_logits() const;
  const int32_t* route_ids() const;
  const float* route_weights() const;

 private:
  std::vector<uint8_t> bytes_;
  std::string model_, revision_, backend_;
  int hidden_ = 0, vocab_ = 0, num_layers_ = 0, top_k_ = 0;
  int64_t token_count_ = 0;
  std::vector<RouteLayer> route_layers_;
  std::map<std::string, TensorView> tensors_;
};

}  // namespace dgpp
