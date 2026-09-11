// The GLM-4.7 resident loader on the synthetic fixture: every
// class lands byte-exact at worlds 1, 2 and 4 as the slice formulas of
// docs/glm47_plan.md §2 say (attention heads and biases, the kv pairing,
// the NVFP4 row/column slices in the modelopt layout with the reciprocal
// per-tensor scales, the replicated set verbatim, the draft's BF16 experts
// requantized to NVFP4 within the format's error); the byte formulas equal
// actual usage and the source-byte plan equals the bytes read; the globals;
// resident cache hits and the image round trip; a draft head copy that is
// not the embedding is refused.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "glm4_fixture.hpp"
#include "loaders/nvfp4_quant.hpp"
#include "models/glm4/binding.hpp"
#include "models/glm4/config.hpp"
#include "models/glm4/loader.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::Glm4ExpectedTensor;
using dgpp::Glm4LayerStream;
using dgpp::Glm4TextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  Glm4TextConfig cfg;
  std::string dir;
  std::vector<Glm4ExpectedTensor> table;
  const Glm4ExpectedTensor& expected(const std::string& name) const {
    for (const auto& e : table)
      if (e.name == name) return e;
    throw std::runtime_error("fixture: no expected tensor " + name);
  }
  std::vector<uint8_t> bytes(const std::string& name) const {
    return glm4fx::fixture_bytes(cfg, expected(name));
  }
};

Fixture write_fixture() {
  Fixture fx;
  fx.cfg = glm4fx::tiny_config();
  fx.dir = (fs::current_path() / "glm4_loader_fixture").string();
  glm4fx::write_fixture(fx.cfg, fx.dir);
  fx.table = dgpp::glm4_expected_text_tensors(fx.cfg);
  return fx;
}

std::vector<uint8_t> device_bytes(const void* dev, size_t n) {
  std::vector<uint8_t> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), dev, n, cudaMemcpyDeviceToHost));
  return h;
}
std::vector<uint8_t> host_rows(const std::vector<uint8_t>& t, size_t row_bytes, int64_t r0, int64_t rows) {
  return std::vector<uint8_t>(t.begin() + r0 * row_bytes, t.begin() + (r0 + rows) * row_bytes);
}
std::vector<uint8_t> host_cols(const std::vector<uint8_t>& t, int64_t rows, size_t full, size_t c0, size_t w) {
  std::vector<uint8_t> out;
  out.reserve(rows * w);
  for (int64_t r = 0; r < rows; ++r)
    out.insert(out.end(), t.begin() + r * full + c0, t.begin() + r * full + c0 + w);
  return out;
}
void expect_device_equals(const void* dev, const std::vector<uint8_t>& want, const std::string& what) {
  const std::vector<uint8_t> got = device_bytes(dev, want.size());
  require(got == want, what + ": resident bytes differ from the checkpoint slice");
}
float device_f32(const float* dev) {
  float v;
  DGPP_CUDA_OK(cudaMemcpy(&v, dev, 4, cudaMemcpyDeviceToHost));
  return v;
}
float fixture_f32(const Fixture& fx, const std::string& name) {
  const auto b = fx.bytes(name);
  float v;
  std::memcpy(&v, b.data(), 4);
  return v;
}

// An NVFP4 view against the checkpoint's row / column slice of `base`.
void check_fp4(const Fixture& fx, const dgpp::GlmFp4Matrix& m, const std::string& base, bool rows_slice,
               int64_t start, int64_t count, const std::string& tag) {
  const auto& ep = fx.expected(base + ".weight");
  const int64_t N = ep.shape[0], K = ep.shape[1] * 2;
  if (rows_slice) {
    require(m.rows == count && m.cols == K, tag + " geometry");
    expect_device_equals(m.payload, host_rows(fx.bytes(base + ".weight"), K / 2, start, count), tag + " payload rows");
    expect_device_equals(m.scales, host_rows(fx.bytes(base + ".weight_scale"), K / 16, start, count), tag + " scale rows");
  } else {
    require(m.rows == N && m.cols == count, tag + " geometry");
    expect_device_equals(m.payload, host_cols(fx.bytes(base + ".weight"), N, K / 2, start / 2, count / 2), tag + " payload cols");
    expect_device_equals(m.scales, host_cols(fx.bytes(base + ".weight_scale"), N, K / 16, start / 16, count / 16), tag + " scale cols");
  }
  const float ws2 = fixture_f32(fx, base + ".weight_scale_2");
  require(device_f32(m.global_scale) == 1.0f / ws2, tag + " global = 1 / weight_scale_2");
}

// A requantized draft matrix against its BF16 source slice: the decoded
// values within the format's error.
void check_requant(const Fixture& fx, const dgpp::GlmFp4Matrix& m, const std::string& base, bool rows_slice,
                   int64_t start, int64_t count, const std::string& tag) {
  const auto& e = fx.expected(base + ".weight");
  const int64_t N = e.shape[0], K = e.shape[1];
  const std::vector<uint8_t> src = fx.bytes(base + ".weight");
  const int64_t rows = rows_slice ? count : N, cols = rows_slice ? K : count;
  require(m.rows == rows && m.cols == cols, tag + " geometry");
  const std::vector<uint8_t> payload = device_bytes(m.payload, static_cast<size_t>(rows * cols / 2));
  const std::vector<uint8_t> scales = device_bytes(m.scales, static_cast<size_t>(rows * cols / 16));
  const float global = device_f32(m.global_scale);
  require(global > 0 && std::isfinite(global), tag + " finite global");
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t sr = rows_slice ? start + r : r, sc = rows_slice ? c : start + c;
      uint16_t bits;
      std::memcpy(&bits, src.data() + (sr * K + sc) * 2, 2);
      const float want = dgpp::bf16_bits_to_float(bits);
      const float got = dgpp::nvfp4_decode(payload.data(), scales.data(), global, cols, r, c);
      float bmax = 0.0f;
      for (int j = 0; j < 16; ++j) {
        const int64_t cj = (c / 16) * 16 + j;
        std::memcpy(&bits, src.data() + (sr * K + (rows_slice ? cj : start + cj)) * 2, 2);
        bmax = std::max(bmax, std::fabs(dgpp::bf16_bits_to_float(bits)));
      }
      require(std::fabs(got - want) <= 0.2f * bmax + 1e-6f, tag + " requant element off");  // one block scale (docs/glm47_plan.md D1)
    }
}

void check_layer(const Fixture& fx, const Glm4LayerStream& s, const dgpp::Glm4LayerResident& r) {
  const Glm4TextConfig& cfg = fx.cfg;
  const dgpp::Glm4LocalGeometry& g = s.geometry();
  const int world = s.world(), rank = s.rank();
  const std::string p = dgpp::glm4_layer_prefix(cfg, r.layer);
  const size_t H2 = static_cast<size_t>(cfg.hidden_size) * 2;
  const std::string tag = "world " + std::to_string(world) + " rank " + std::to_string(rank) + " layer " +
                          std::to_string(r.layer) + " ";
  const bool draft = r.layer == cfg.mtp_layer();
  expect_device_equals(r.input_norm, fx.bytes(p + "input_layernorm.weight"), tag + "input norm");
  expect_device_equals(r.post_norm, fx.bytes(p + "post_attention_layernorm.weight"), tag + "post norm");
  // Attention: head slices of the projections and biases, packed o columns.
  const std::string ap = p + "self_attn.";
  const int64_t d = cfg.head_dim;
  expect_device_equals(r.attn.q_proj, host_rows(fx.bytes(ap + "q_proj.weight"), H2, g.head_begin * d, g.local_heads * d), tag + "q rows");
  expect_device_equals(r.attn.k_proj, host_rows(fx.bytes(ap + "k_proj.weight"), H2, g.kv_head_begin * d, g.local_kv_heads * d), tag + "k rows");
  expect_device_equals(r.attn.v_proj, host_rows(fx.bytes(ap + "v_proj.weight"), H2, g.kv_head_begin * d, g.local_kv_heads * d), tag + "v rows");
  expect_device_equals(r.attn.q_bias, host_rows(fx.bytes(ap + "q_proj.bias"), 2, g.head_begin * d, g.local_heads * d), tag + "q bias");
  expect_device_equals(r.attn.k_bias, host_rows(fx.bytes(ap + "k_proj.bias"), 2, g.kv_head_begin * d, g.local_kv_heads * d), tag + "k bias");
  expect_device_equals(r.attn.v_bias, host_rows(fx.bytes(ap + "v_proj.bias"), 2, g.kv_head_begin * d, g.local_kv_heads * d), tag + "v bias");
  const int64_t Q = static_cast<int64_t>(cfg.num_attention_heads) * d;
  expect_device_equals(r.attn.o_proj, host_cols(fx.bytes(ap + "o_proj.weight"), cfg.hidden_size, Q * 2, g.head_begin * d * 2, g.local_heads * d * 2), tag + "o cols");
  expect_device_equals(r.attn.q_norm, fx.bytes(ap + "q_norm.weight"), tag + "q norm");
  expect_device_equals(r.attn.k_norm, fx.bytes(ap + "k_norm.weight"), tag + "k norm");
  require(r.attn.local_heads == g.local_heads && r.attn.local_kv_heads == g.local_kv_heads, tag + "head counts");
  if (world > cfg.num_key_value_heads)
    require(r.attn.kv_head_begin == rank / (world / cfg.num_key_value_heads), tag + "kv pairing");
  if (!r.moe) {
    const int64_t I = g.local_dense_inter;
    require(r.dense.local_inter == I, tag + "dense inter");
    check_fp4(fx, r.dense.gate, p + "mlp.gate_proj", true, rank * I, I, tag + "dense gate");
    check_fp4(fx, r.dense.up, p + "mlp.up_proj", true, rank * I, I, tag + "dense up");
    check_fp4(fx, r.dense.down, p + "mlp.down_proj", false, rank * I, I, tag + "dense down");
  } else {
    const std::string mp = p + "mlp.";
    expect_device_equals(r.moe_w.router, fx.bytes(mp + "gate.weight"), tag + "router");
    expect_device_equals(r.moe_w.router_bias, fx.bytes(mp + "gate.e_score_correction_bias"), tag + "router bias");
    const int64_t I = g.local_inter, S = g.local_shared_inter;
    require(r.moe_w.local_inter == I && r.moe_w.local_shared_inter == S, tag + "moe inter");
    require(static_cast<int>(r.moe_w.experts.size()) == cfg.n_routed_experts * 3, tag + "expert count");
    for (int e = 0; e < cfg.n_routed_experts; ++e) {
      const std::string ep = mp + "experts." + std::to_string(e) + ".";
      if (draft) {
        check_requant(fx, r.moe_w.expert(e, 0), ep + "gate_proj", true, rank * I, I, tag + "draft expert gate");
        check_requant(fx, r.moe_w.expert(e, 1), ep + "up_proj", true, rank * I, I, tag + "draft expert up");
        check_requant(fx, r.moe_w.expert(e, 2), ep + "down_proj", false, rank * I, I, tag + "draft expert down");
      } else {
        check_fp4(fx, r.moe_w.expert(e, 0), ep + "gate_proj", true, rank * I, I, tag + "expert gate");
        check_fp4(fx, r.moe_w.expert(e, 1), ep + "up_proj", true, rank * I, I, tag + "expert up");
        check_fp4(fx, r.moe_w.expert(e, 2), ep + "down_proj", false, rank * I, I, tag + "expert down");
      }
    }
    const std::string sp = mp + "shared_experts.";
    if (draft) {
      check_requant(fx, r.moe_w.shared[0], sp + "gate_proj", true, rank * S, S, tag + "draft shared gate");
      check_requant(fx, r.moe_w.shared[2], sp + "down_proj", false, rank * S, S, tag + "draft shared down");
    } else {
      check_fp4(fx, r.moe_w.shared[0], sp + "gate_proj", true, rank * S, S, tag + "shared gate");
      check_fp4(fx, r.moe_w.shared[1], sp + "up_proj", true, rank * S, S, tag + "shared up");
      check_fp4(fx, r.moe_w.shared[2], sp + "down_proj", false, rank * S, S, tag + "shared down");
    }
  }
  if (draft) {
    expect_device_equals(r.enorm, fx.bytes(p + "enorm.weight"), tag + "enorm");
    expect_device_equals(r.hnorm, fx.bytes(p + "hnorm.weight"), tag + "hnorm");
    expect_device_equals(r.eh_proj, fx.bytes(p + "eh_proj.weight"), tag + "eh_proj");
    expect_device_equals(r.shared_head_norm, fx.bytes(p + "shared_head.norm.weight"), tag + "shared head norm");
  } else {
    require(r.enorm == nullptr && r.eh_proj == nullptr, tag + "no draft head on a main layer");
  }
  require(r.bytes == Glm4LayerStream::layer_bytes(cfg, r.layer, rank, world), tag + "layer bytes formula");
}

}  // namespace

DGPP_TEST(glm4_loader_slices_every_class_byte_exact_at_worlds_1_2_4) {
  const Fixture fx = write_fixture();
  const int layers = fx.cfg.num_hidden_layers + 1;
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      Glm4LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Glm4Residency::Streaming,
                        dgpp::Glm4HeadSharding::Full, /*resident_mtp=*/true);
      for (int l = 0; l < layers; ++l) {
        const uint64_t before = s.source_bytes_read();
        const auto& r = s.load_layer(l);
        require(r.layer == l, "layer index");
        check_layer(fx, s, r);
        require(s.source_bytes_read() - before == Glm4LayerStream::planned_layer_source_bytes(fx.cfg, l, rank, world),
                "the source-byte plan equals the bytes read");
        s.release_layer();
      }
    }
  }
}

DGPP_TEST(glm4_loader_globals_slices) {
  const Fixture fx = write_fixture();
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::Glm4HeadSharding::VocabSharded : dgpp::Glm4HeadSharding::Full;
      Glm4LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Glm4Residency::Streaming, head);
      const auto& g = s.load_globals();
      const size_t H2 = static_cast<size_t>(fx.cfg.hidden_size) * 2;
      expect_device_equals(g.embed, fx.bytes("model.embed_tokens.weight"), "embed");
      expect_device_equals(g.final_norm, fx.bytes("model.norm.weight"), "final norm");
      expect_device_equals(g.lm_head, host_rows(fx.bytes("lm_head.weight"), H2, g.lm_vocab_begin, g.lm_vocab_count), "lm head slice");
      require(g.lm_vocab_count == Glm4LayerStream::lm_vocab_count(fx.cfg, rank, world, head), "lm head count");
      require(g.bytes == Glm4LayerStream::globals_bytes(fx.cfg, rank, world, head), "globals formula");
    }
  }
}

DGPP_TEST(glm4_loader_resident_mode_and_image_round_trip) {
  const Fixture fx = write_fixture();
  const fs::path cache = fs::current_path() / "glm4_loader_image_cache";
  fs::remove_all(cache);
  const std::string saved = Glm4LayerStream::resident_image_dir();
  Glm4LayerStream::set_resident_image_dir(cache.string());
  const int layers = fx.cfg.num_hidden_layers + 1;
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(layers));
  {
    Glm4LayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::Glm4Residency::Resident, dgpp::Glm4HeadSharding::VocabSharded,
                      /*resident_mtp=*/true);
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      check_layer(fx, s, r);
      const auto span = s.resident_layer_span(l);
      require(span.first != nullptr && span.second == r.bytes, "resident span");
      built[static_cast<size_t>(l)] = device_bytes(span.first, span.second);
    }
    (void)s.load_globals();
    require(s.image_layers_captured() == layers, "every layer captured");
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) (void)s.load_layer(l);
    require(s.source_bytes_read() == before, "cache hits read no storage");
    require(Glm4LayerStream::resident_bytes(fx.cfg, 1, 2, dgpp::Glm4HeadSharding::VocabSharded, true) ==
                [&] { size_t t = s.load_globals().bytes; for (int l = 0; l < layers; ++l) t += s.load_layer(l).bytes; return t; }(),
            "resident bytes formula");
    s.release_sources();
    require(s.sources_released(), "sources released");
  }
  {
    Glm4LayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::Glm4Residency::Resident, dgpp::Glm4HeadSharding::VocabSharded,
                      /*resident_mtp=*/true);
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      const auto span = s.resident_layer_span(l);
      require(device_bytes(span.first, span.second) == built[static_cast<size_t>(l)], "restored layer bitwise the built one");
      check_layer(fx, s, r);
    }
    require(s.image_layers_restored() == layers && s.source_bytes_read() == before,
            "every layer restored from the image, no source reads");
    const dgpp::Glm4ReplicatedDigest d = s.hash_replicated();
    require(d.layer.size() == static_cast<size_t>(layers) && d.tensors > 0, "digest shape");
  }
  Glm4LayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
}

DGPP_TEST(glm4_loader_refuses_a_draft_head_that_is_not_the_embedding) {
  const Fixture fx = write_fixture();
  // Flip one byte inside the draft's embedding copy.
  const fs::path shard = fs::path(fx.dir) / "model.safetensors";
  const std::string name = dgpp::glm4_layer_prefix(fx.cfg, fx.cfg.mtp_layer()) + "embed_tokens.weight";
  std::FILE* f = std::fopen(shard.c_str(), "r+b");
  require(f != nullptr, "open shard");
  uint64_t hlen = 0;
  require(std::fread(&hlen, 8, 1, f) == 1, "header length");
  std::string header(hlen, '\0');
  require(std::fread(header.data(), 1, hlen, f) == hlen, "header");
  const size_t at = header.find("\"" + name + "\"");
  require(at != std::string::npos, "tensor in header");
  const size_t off_at = header.find("\"data_offsets\":[", at);
  const uint64_t begin = std::stoull(header.substr(off_at + 16));
  std::fseek(f, static_cast<long>(8 + hlen + begin + 100), SEEK_SET);
  uint8_t byte = 0;
  require(std::fread(&byte, 1, 1, f) == 1, "read byte");
  byte ^= 1;
  std::fseek(f, static_cast<long>(8 + hlen + begin + 100), SEEK_SET);
  require(std::fwrite(&byte, 1, 1, f) == 1, "write byte");
  std::fclose(f);
  bool refused = false;
  try {
    Glm4LayerStream s(fx.cfg, fx.dir, 0, 1);
  } catch (const std::runtime_error& e) {
    refused = std::string(e.what()).find("not a copy") != std::string::npos;
  }
  require(refused, "a distinct draft embedding is refused by name");
}

int main() { return dgpp::test::run_all(); }
