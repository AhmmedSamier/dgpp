#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "models/glm/vision_config.hpp"

namespace dgpp {
// Replicated BF16 vision tower. Only image prefills execute it; decode graphs
// continue to consume ordinary token ids. The returned device rows belong to
// this object and remain valid until its next encode() call.
class GlmVisionEncoder {
 public:
  GlmVisionEncoder(const GlmVisionConfig& config, const std::string& checkpoint,
                   cudaStream_t stream);
  ~GlmVisionEncoder();
  using Trace = std::function<void(const std::string&, const void*, size_t, DType)>;
  const uint16_t* encode(const std::vector<ImageInput>& images, Trace trace = {});
  uint64_t digest() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dgpp
