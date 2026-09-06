// Host-only tests for the DSA/MLA geometry contract (DESIGN §7.2). The byte
// formulas here are the M3 exit criterion "measured cache bytes are within 2%
// of the formula": the numbers below are transcribed from DESIGN §7.2's
// table, not derived from each other.
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "models/dsa_geometry.hpp"

namespace {

using dgpp::DsaConfig;
using dgpp::DsaGeometry;

DsaConfig real_tp1() { return DsaConfig{}; }

DsaConfig real_tp4() {
  DsaConfig c;
  c.tp_size = 4;
  return c;  // all other fields at GLM-5.3-Flash defaults
}

}  // namespace

DGPP_TEST(dsa_geometry_tp1_matches_design_7_2_exactly) {
  const DsaGeometry g = DsaGeometry::from_config(real_tp1());

  // Indexer: 32 heads x 128 dim, kpool 4, topk 2048 tokens -> 512 pools.
  if (g.select_k != 512) throw std::runtime_error("select_k");
  if (g.max_selected != 2051) throw std::runtime_error("max_selected");

  // MLA latent: 512 dims BF16 = 1024 B/token/layer.
  if (g.latent_bytes_per_token != 1024)
    throw std::runtime_error("latent bytes per token");

  // Pooled index: 128 B FP8 K + 4 B FP32 scale per pool, one pool per 4
  // tokens -> 33 B/token/layer (DESIGN: "128 FP8 + 4 scale bytes / 4").
  if (g.index_k_bytes_per_pool != 128) throw std::runtime_error("k bytes");
  if (g.index_scale_bytes_per_pool != 4) throw std::runtime_error("scale bytes");
  if (g.index_bytes_per_token != 33)
    throw std::runtime_error("index bytes per token");

  // Incomplete tails: [2, 4, 128] BF16 = 2048 B/request/layer; DESIGN's
  // "11 x 4 x (128 K + 128 gate) x 2" = 22.5 KB/request across 11 layers.
  if (g.tail_bytes_per_request != 2048)
    throw std::runtime_error("tail bytes per request");
  if (g.tail_bytes_per_request_all != 11ull * 2048)
    throw std::runtime_error("tail bytes all layers");
  if (g.tail_bytes_per_request_all != 22528)
    throw std::runtime_error("tail total literal");

  // DESIGN §7.2 table at 300,000 cached tokens, per rank:
  //   MLA latent   300k * 11 * 512 * 2  = 3.379 GB
  //   pooled index 300k * 11 * 132 / 4  = 0.109 GB
  const int64_t tokens = 300000;
  const size_t giB = 1000ull * 1000 * 1000;
  if (g.latent_total_bytes(tokens) != 300000ll * 11 * 512 * 2)
    throw std::runtime_error("latent total");
  if (g.latent_total_bytes(tokens) / giB != 3)
    throw std::runtime_error("latent total whole GB");
  if (g.index_total_bytes(tokens) != 300000ll * 11 * 132 / 4)
    throw std::runtime_error("index total");
  if (g.index_total_bytes(tokens) >= 110 * 1000 * 1000)
    throw std::runtime_error("index total must be ~0.109 GB");

  // Causal bounds: query at p sees floor((p+1)/4) pools; tail is L%4.
  if (g.visible_pools(0) != 0) throw std::runtime_error("visible_pools(0)");
  if (g.visible_pools(2) != 0) throw std::runtime_error("visible_pools(2)");
  if (g.visible_pools(3) != 1) throw std::runtime_error("visible_pools(3)");
  if (g.visible_pools(4) != 1) throw std::runtime_error("visible_pools(4)");
  if (g.visible_pools(2047) != 512)
    throw std::runtime_error("visible_pools(2047)");
  if (g.tail_len(10) != 2) throw std::runtime_error("tail_len(10)");
  if (g.tail_len(12) != 0) throw std::runtime_error("tail_len(12)");

  // Shared block table: 128-token blocks = 32 pools.
  if (g.pools_per_block != 32) throw std::runtime_error("pools per block");
  if (g.latent_block_bytes != 128ull * 512 * 2)
    throw std::runtime_error("latent block bytes");
  if (g.index_k_block_bytes != 32ull * 128)
    throw std::runtime_error("index k block bytes");
  if (g.index_scale_block_bytes != 32ull * 4)
    throw std::runtime_error("index scale block bytes");
}

DGPP_TEST(dsa_geometry_tp4_shards_heads_only) {
  const DsaGeometry g = DsaGeometry::from_config(real_tp4());

  // MLA heads shard 64 -> 16; the indexer (32 heads) and the latent cache
  // stay replicated on every rank (DESIGN §3: DSA indexer replicated).
  if (g.local_heads != 16) throw std::runtime_error("local_heads");
  if (g.local_q_rows != 16 * 256) throw std::runtime_error("local_q_rows");
  if (g.local_v_rows != 16 * 256) throw std::runtime_error("local_v_rows");
  if (g.latent_bytes_per_token != 1024)
    throw std::runtime_error("latent is replicated, not sharded");
  if (g.index_bytes_per_token != 33)
    throw std::runtime_error("index cache is replicated, not sharded");
  if (g.tail_bytes_per_request != 2048)
    throw std::runtime_error("tail is replicated, not sharded");
}

DGPP_TEST(dsa_geometry_rejects_invalid_configs) {
  auto bad = [](DsaConfig c, const char* what) {
    try {
      DsaGeometry::from_config(c);
    } catch (const std::invalid_argument&) {
      return;
    }
    throw std::runtime_error(std::string("expected rejection: ") + what);
  };

  DsaConfig c;
  c.index_topk = 2050;  // not a multiple of kpool
  bad(c, "topk not multiple of kpool");

  c = DsaConfig{};
  c.qk_rope_head_dim = 64;  // engine implements the rope-free path only
  bad(c, "nonzero rope dim");

  c = DsaConfig{};
  c.index_head_dim = 96;  // Hadamard-128 is pinned
  bad(c, "index head dim");

  c = DsaConfig{};
  c.index_kpool = 3;  // template dispatch covers 2/4/8
  bad(c, "kpool not power of two");

  c = DsaConfig{};
  c.block_tokens = 126;  // not a multiple of kpool
  bad(c, "block not pool aligned");

  c = DsaConfig{};
  c.num_heads = 65;
  c.tp_size = 4;
  bad(c, "heads not divisible by tp");

  c = DsaConfig{};
  c.q_lora_rank = 1534;  // fused [q_a|kv_a] alignment
  bad(c, "q_lora alignment");
}

DGPP_TEST(dsa_geometry_latent_formats_size_the_cache) {
  // The KV dtype knob (2026-09-06): bf16 is the DESIGN §7.2 table; fp8
  // halves the row and adds a 4-byte row scale; fp4 packs two e2m1 codes
  // per byte plus one e4m3 scale per 16 elements (288 B at 512 wide).
  const auto at = [](dgpp::LatentFormat f) {
    DsaConfig c;
    c.latent_format = f;
    return DsaGeometry::from_config(c);
  };
  const DsaGeometry bf16 = at(dgpp::LatentFormat::kBf16);
  const DsaGeometry fp8 = at(dgpp::LatentFormat::kFp8);
  const DsaGeometry fp4 = at(dgpp::LatentFormat::kFp4);
  if (bf16.latent_bytes_per_token != 1024 || bf16.latent_scale_bytes_per_token != 0)
    throw std::runtime_error("bf16 row");
  if (fp8.latent_bytes_per_token != 512 || fp8.latent_scale_bytes_per_token != 4)
    throw std::runtime_error("fp8 row");
  if (fp4.latent_bytes_per_token != 288 || fp4.latent_scale_bytes_per_token != 4)
    throw std::runtime_error("fp4 row");
  if (bf16.latent_bytes_per_token_all != 11ull * 1024 ||
      fp8.latent_bytes_per_token_all != 11ull * (512 + 4) ||
      fp4.latent_bytes_per_token_all != 11ull * (288 + 4))
    throw std::runtime_error("per-token totals include the scales");
  if (fp8.latent_block_bytes != 128ull * 512 || fp4.latent_block_bytes != 128ull * 288)
    throw std::runtime_error("block bytes follow the row");
  if (bf16.index_bytes_per_token != fp8.index_bytes_per_token ||
      fp8.index_bytes_per_token != fp4.index_bytes_per_token)
    throw std::runtime_error("the index cache is fp8 in every format");
  // At 262,144 tokens per rank across the 11 main layers: 2.75 GiB of
  // latent rows in bf16, 1.38 GiB in fp8, 0.79 GiB in fp4 (plus scales).
  const int64_t tokens = 262144;
  if (bf16.latent_total_bytes(tokens) != 262144ll * 11 * 1024 ||
      fp8.latent_total_bytes(tokens) != 262144ll * 11 * 516 ||
      fp4.latent_total_bytes(tokens) != 262144ll * 11 * 292)
    throw std::runtime_error("262k totals");
  // The quantized rows' alignment pins.
  auto bad = [](DsaConfig c, const char* what) {
    try {
      DsaGeometry::from_config(c);
    } catch (const std::invalid_argument&) {
      return;
    }
    throw std::runtime_error(std::string("expected rejection: ") + what);
  };
  DsaConfig c;
  c.latent_format = dgpp::LatentFormat::kFp4;
  c.kv_lora_rank = 520;  // not a multiple of 16
  bad(c, "fp4 needs kv_lora % 16");
  c = DsaConfig{};
  c.latent_format = dgpp::LatentFormat::kFp8;
  c.kv_lora_rank = 2048;  // beyond the append kernel's row
  bad(c, "fp8 needs kv_lora <= 1024");
  c = DsaConfig{};
  c.latent_format = dgpp::LatentFormat::kFp4;
  c.kv_lora_rank = 32;  // the CI test geometry stays legal
  DsaGeometry::from_config(c);
}

DGPP_TEST(dsa_geometry_config_defaults_are_real_checkpoint) {
  // The defaults must be the GLM-5.3-Flash config.json values; M4's
  // config-driven assembly reads them explicitly, but every other consumer
  // relies on these defaults being the real ones.
  const DsaConfig c{};
  if (c.num_heads != 64 || c.q_lora_rank != 1536 || c.kv_lora_rank != 512)
    throw std::runtime_error("mla dims");
  if (c.qk_nope_head_dim != 256 || c.qk_rope_head_dim != 0 ||
      c.v_head_dim != 256)
    throw std::runtime_error("head dims");
  if (c.index_n_heads != 32 || c.index_head_dim != 128)
    throw std::runtime_error("indexer dims");
  if (c.index_topk != 2048 || c.index_kpool != 4 || !c.always_select_tail)
    throw std::runtime_error("indexer selection params");
  if (c.num_dsa_layers != 11) throw std::runtime_error("layer count");
}
