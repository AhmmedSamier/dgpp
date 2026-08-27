// Host-only tests for the KDA geometry contract (DESIGN §7.1) and snapshot
// header validation (DESIGN §8). The byte formulas here are the M2 exit
// criterion "geometry and bytes agree exactly with DESIGN.md §7.1" — the
// numbers are transcribed from that section, not derived from each other.
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "models/kda_geometry.hpp"
#include "models/kda_snapshot.hpp"

namespace {

using dgpp::KdaConfig;
using dgpp::KdaGeometry;
using dgpp::KdaSnapshotHeader;

KdaConfig real_tp4() {
  KdaConfig c;
  c.tp_size = 4;
  return c;  // all other fields at GLM-5.3-Flash defaults
}

void expect_throw(std::function<void()> fn, const char* what) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string("expected exception: ") + what);
}

}  // namespace

DGPP_TEST(kda_geometry_tp4_matches_design_7_1_exactly) {
  const KdaGeometry g = KdaGeometry::from_config(real_tp4());

  // Recurrent state: [16, 128, 128] FP32 per rank/layer.
  if (g.local_heads != 16) throw std::runtime_error("local_heads");
  if (g.recurrent_bytes != 16ull * 128 * 128 * 4)
    throw std::runtime_error("recurrent bytes");
  if (g.recurrent_bytes != 1048576) throw std::runtime_error("1 MiB/layer");

  // Merged q|k|v conv state: 6144 channels x committed width 3 (BF16), with
  // the three-token draft reserve widening the slot to 6.
  if (g.conv_channels != 6144) throw std::runtime_error("conv channels");
  if (g.conv_hist != 3) throw std::runtime_error("conv hist");
  if (g.conv_state_width != 6) throw std::runtime_error("conv slot width");
  if (g.conv_committed_bytes != 6144ull * 3 * 2)
    throw std::runtime_error("conv committed bytes");
  if (g.conv_slot_bytes != 6144ull * 6 * 2)
    throw std::runtime_error("conv slot bytes");

  // Fixed mutable slot per rank/request: 34.0 MiB recurrent + 1.20 MiB
  // committed conv + 1.20 MiB draft reserve = 36.39 MiB (DESIGN §7.1/§8).
  if (g.slot_bytes != 34ull * 1048576 + 34ull * 36864 + 34ull * 36864)
    throw std::runtime_error("slot bytes");
  if (g.slot_bytes != 38158336ull)
    throw std::runtime_error("slot bytes literal");
  const KdaGeometry two = KdaGeometry::from_config(real_tp4(), 2);
  if (two.pool_bytes != 2 * g.slot_bytes)
    throw std::runtime_error("pool scales with slots");

  // Fused in-projection width: 3*local_proj + local_heads + 2*head_dim
  // (q|k|v|b head-sharded, f_a/g_a replicated).
  if (g.in_proj_cols != 3 * 2048 + 16 + 2 * 128)
    throw std::runtime_error("in_proj cols");
}

DGPP_TEST(kda_geometry_tp1_and_pool_sizing) {
  const KdaGeometry g = KdaGeometry::from_config(KdaConfig{});
  if (g.local_heads != 64) throw std::runtime_error("tp1 local heads");
  if (g.recurrent_bytes != 64ull * 128 * 128 * 4)
    throw std::runtime_error("tp1 recurrent");
  if (g.conv_channels != 24576) throw std::runtime_error("tp1 conv channels");
  // 42-request pool at TP=1 for scale sanity (single-node M2 testing shape).
  const KdaGeometry pool = KdaGeometry::from_config(KdaConfig{}, 42);
  if (pool.pool_bytes != 42 * g.slot_bytes)
    throw std::runtime_error("pool bytes");
}

DGPP_TEST(kda_geometry_rejects_invalid_configs) {
  auto bad = [](int heads, int tp) {
    KdaConfig c;
    c.heads = heads;
    c.tp_size = tp;
    return c;
  };
  expect_throw([&] { KdaGeometry::from_config(bad(63, 4)); },
               "heads not divisible by tp");
  expect_throw([&] { KdaGeometry::from_config(bad(0, 1)); }, "zero heads");
  KdaConfig cw = KdaConfig{};
  cw.conv_width = 1;
  expect_throw([&] { KdaGeometry::from_config(cw); }, "conv width 1");
  KdaConfig lb = KdaConfig{};
  lb.lower_bound = 0.5f;
  expect_throw([&] { KdaGeometry::from_config(lb); }, "positive lower bound");
  KdaConfig sw = KdaConfig{};
  sw.spec_width = -1;
  expect_throw([&] { KdaGeometry::from_config(sw); }, "negative spec width");
}

DGPP_TEST(kda_snapshot_header_roundtrip_and_validation) {
  const KdaConfig cfg = real_tp4();
  KdaSnapshotHeader h =
      KdaSnapshotHeader::make(cfg, "unsloth/GLM-5.3-Flash-FP8@a160e22",
                              "bf16-act/f32-state");

  // Wire round-trip must be identity: fixed-layout struct, memcpy-stable.
  std::vector<uint8_t> wire(sizeof(KdaSnapshotHeader));
  std::memcpy(wire.data(), &h, wire.size());
  KdaSnapshotHeader back{};
  std::memcpy(&back, wire.data(), wire.size());
  back.validate_against(cfg);
  if (std::memcmp(&back, &h, sizeof(h)) != 0)
    throw std::runtime_error("roundtrip changed header bytes");

  // Every identity field must be enforced on import.
  auto tamper = [&]<typename M>(M KdaSnapshotHeader::*field, M value,
                                 const char* what) {
    KdaSnapshotHeader t = h;
    t.*field = value;
    expect_throw([&] { t.validate_against(cfg); }, what);
  };
  tamper(&KdaSnapshotHeader::tp_size, 2u, "tp mismatch");
  tamper(&KdaSnapshotHeader::num_kda_layers, 33u, "layer count");
  tamper(&KdaSnapshotHeader::global_heads, 32u, "head count");
  tamper(&KdaSnapshotHeader::head_dim, 64u, "head dim");
  tamper(&KdaSnapshotHeader::conv_width, 3u, "conv width");
  tamper(&KdaSnapshotHeader::spec_width, 0u, "spec width");
  tamper(&KdaSnapshotHeader::recurrent_dtype,
         static_cast<uint32_t>(dgpp::DType::BF16), "recurrent dtype");
  tamper(&KdaSnapshotHeader::conv_dtype,
         static_cast<uint32_t>(dgpp::DType::F32), "conv dtype");
  tamper(&KdaSnapshotHeader::payload_bytes, uint64_t{1}, "payload size");
  tamper(&KdaSnapshotHeader::reserved, uint64_t{7}, "reserved nonzero");

  KdaSnapshotHeader badmagic = h;
  badmagic.magic[0] = 'X';
  expect_throw([&] { badmagic.validate_against(cfg); }, "magic");

  KdaSnapshotHeader unterm = h;
  unterm.model_revision[sizeof(unterm.model_revision) - 1] = 'x';
  expect_throw([&] { unterm.validate_against(cfg); }, "unterminated string");

  // A TP=1 model must reject the TP=4 snapshot (and vice versa).
  KdaSnapshotHeader tp1 = KdaSnapshotHeader::make(KdaConfig{}, "r", "m");
  expect_throw([&] { tp1.validate_against(cfg); }, "cross-tp attach");
  expect_throw([&] { h.validate_against(KdaConfig{}); }, "cross-tp attach 2");

  // Oversized identity strings are refused at creation, not silently cut.
  expect_throw(
      [&] {
        KdaSnapshotHeader::make(cfg, std::string(100, 'r'), "m");
      },
      "long revision");
  expect_throw(
      [&] {
        KdaSnapshotHeader::make(cfg, "r", std::string(40, 'm'));
      },
      "long numerics");
}
