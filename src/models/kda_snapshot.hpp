#pragma once
// KDA state-snapshot header (M2 deliverable: import/export with model,
// dtype, TP, and revision headers; DESIGN §8 requires complete snapshots
// only — a snapshot without a valid header is not attachable).
//
// A snapshot file/buffer is: header, then per KDA layer in order:
//   [recurrent state FP32 local_heads*V*K][conv state BF16 channels*width]
// with the speculative conv suffix included verbatim (zeroed unless M8 wrote
// draft positions into it). All integers little-endian; the engine only runs
// on little-endian hosts today, so the header is a raw memcpy struct.
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

#include "common/dtypes.hpp"
#include "models/kda_geometry.hpp"

namespace dgpp {

struct KdaSnapshotHeader {
  static constexpr char kMagic[8] = {'D', 'G', 'P', 'P', 'K', 'D', 'A', '1'};
  static constexpr uint32_t kVersion = 1;
  static constexpr size_t kRevisionBytes = 64;
  static constexpr size_t kNumericsBytes = 32;

  char magic[8] = {};
  uint32_t version = 0;
  uint32_t header_bytes = 0;
  char model_revision[kRevisionBytes] = {};  // NUL-padded checkpoint id
  char numerics_mode[kNumericsBytes] = {};   // e.g. "bf16-act/f32-state"
  uint32_t tp_size = 0;
  uint32_t num_kda_layers = 0;
  uint32_t global_heads = 0;
  uint32_t head_dim = 0;
  uint32_t conv_width = 0;
  uint32_t spec_width = 0;
  uint32_t recurrent_dtype = 0;  // DType as int
  uint32_t conv_dtype = 0;       // DType as int
  uint64_t payload_bytes = 0;
  uint64_t reserved = 0;         // zero; future checksum slot

  static KdaSnapshotHeader make(const KdaConfig& cfg,
                                std::string_view revision,
                                std::string_view numerics) {
    KdaConfig::validate_config(cfg);
    if (revision.size() >= kRevisionBytes)
      throw std::invalid_argument("snapshot revision id too long");
    if (numerics.size() >= kNumericsBytes)
      throw std::invalid_argument("snapshot numerics mode too long");
    KdaSnapshotHeader h{};
    std::memcpy(h.magic, kMagic, sizeof(h.magic));
    h.version = kVersion;
    h.header_bytes = sizeof(KdaSnapshotHeader);
    std::memcpy(h.model_revision, revision.data(), revision.size());
    std::memcpy(h.numerics_mode, numerics.data(), numerics.size());
    h.tp_size = static_cast<uint32_t>(cfg.tp_size);
    h.num_kda_layers = static_cast<uint32_t>(cfg.num_kda_layers);
    h.global_heads = static_cast<uint32_t>(cfg.heads);
    h.head_dim = static_cast<uint32_t>(cfg.head_dim);
    h.conv_width = static_cast<uint32_t>(cfg.conv_width);
    h.spec_width = static_cast<uint32_t>(cfg.spec_width);
    h.recurrent_dtype = static_cast<uint32_t>(DType::F32);
    h.conv_dtype = static_cast<uint32_t>(DType::BF16);
    const KdaGeometry g = KdaGeometry::from_config(cfg);
    h.payload_bytes = g.slot_bytes;
    return h;
  }

  // Throws with a precise reason when the buffer cannot be attached to a
  // model built from `cfg`. Anything unvalidated here becomes a silent
  // wrong-state bug at M7 attach time.
  void validate_against(const KdaConfig& cfg) const {
    auto reject = [&](const std::string& why) {
      throw std::runtime_error("kda snapshot rejected: " + why);
    };
    if (std::memcmp(magic, kMagic, sizeof(magic)) != 0)
      reject("bad magic");
    if (version != kVersion) reject("unsupported version " + std::to_string(version));
    if (header_bytes != sizeof(KdaSnapshotHeader))
      reject("header size mismatch");
    if (model_revision[kRevisionBytes - 1] != '\0' ||
        numerics_mode[kNumericsBytes - 1] != '\0')
      reject("unterminated header string");
    if (tp_size != static_cast<uint32_t>(cfg.tp_size))
      reject("tp mismatch");
    if (num_kda_layers != static_cast<uint32_t>(cfg.num_kda_layers))
      reject("layer count mismatch");
    if (global_heads != static_cast<uint32_t>(cfg.heads)) reject("head count mismatch");
    if (head_dim != static_cast<uint32_t>(cfg.head_dim)) reject("head dim mismatch");
    if (conv_width != static_cast<uint32_t>(cfg.conv_width)) reject("conv width mismatch");
    if (spec_width != static_cast<uint32_t>(cfg.spec_width)) reject("spec width mismatch");
    if (recurrent_dtype != static_cast<uint32_t>(DType::F32))
      reject("recurrent dtype mismatch");
    if (conv_dtype != static_cast<uint32_t>(DType::BF16))
      reject("conv dtype mismatch");
    const KdaGeometry g = KdaGeometry::from_config(cfg);
    if (payload_bytes != g.slot_bytes) reject("payload size mismatch");
    if (reserved != 0) reject("reserved field must be zero");
  }
};

static_assert(sizeof(KdaSnapshotHeader) ==
                  8 + 4 + 4 + KdaSnapshotHeader::kRevisionBytes +
                  KdaSnapshotHeader::kNumericsBytes + 4 * 8 + 8 * 2,
              "KdaSnapshotHeader must stay a fixed-layout struct");

}  // namespace dgpp
