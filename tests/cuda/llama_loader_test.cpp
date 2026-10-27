// The Llama resident loader on the synthetic fixture: every class lands
// byte-exact at worlds 1, 2 and 4 as the slice formulas of
// docs/zeta_plan.md §2 say (attention head slices with the kv pairing,
// MLP row/column slices, the replicated set verbatim, the vocab-sharded
// head); the byte formulas equal actual usage and the source-byte plan
// equals the bytes read; the globals; resident cache hits and the image
// round trip.
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
#include "llama_fixture.hpp"
#include "loaders/fp8_quant.hpp"
#include "models/llama/binding.hpp"
#include "models/llama/config.hpp"
#include "models/llama/loader.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::LlamaExpectedTensor;
using dgpp::LlamaLayerStream;
using dgpp::LlamaTextConfig;
namespace fp8_quant = dgpp::fp8_quant;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  LlamaTextConfig cfg;
  std::string dir;
  std::vector<LlamaExpectedTensor> table;
  const LlamaExpectedTensor& expected(const std::string& name) const {
    for (const auto& e : table)
      if (e.name == name) return e;
    throw std::runtime_error("fixture: no expected tensor " + name);
  }
  std::vector<uint8_t> bytes(const std::string& name) const {
    return llamafx::fixture_bytes(cfg, expected(name));
  }
};

Fixture write_fixture() {
  Fixture fx;
  fx.cfg = llamafx::tiny_config();
  fx.dir = (fs::current_path() / "llama_loader_fixture").string();
  llamafx::write_fixture(fx.cfg, fx.dir);
  fx.table = dgpp::llama_expected_text_tensors(fx.cfg);
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

void check_layer(const Fixture& fx, const LlamaLayerStream& s, const dgpp::LlamaLayerResident& r) {
  const LlamaTextConfig& cfg = fx.cfg;
  const dgpp::LlamaLocalGeometry& g = s.geometry();
  const int world = s.world(), rank = s.rank();
  const std::string p = "model.layers." + std::to_string(r.layer) + ".";
  const size_t H2 = static_cast<size_t>(cfg.hidden_size) * 2;
  const std::string tag = "world " + std::to_string(world) + " rank " + std::to_string(rank) + " layer " +
                          std::to_string(r.layer) + " ";
  expect_device_equals(r.input_norm, fx.bytes(p + "input_layernorm.weight"), tag + "input norm");
  expect_device_equals(r.post_norm, fx.bytes(p + "post_attention_layernorm.weight"), tag + "post norm");
  const std::string ap = p + "self_attn.";
  const int64_t d = cfg.head_dim;
  expect_device_equals(r.attn.q_proj, host_rows(fx.bytes(ap + "q_proj.weight"), H2, g.head_begin * d, g.local_heads * d), tag + "q rows");
  expect_device_equals(r.attn.k_proj, host_rows(fx.bytes(ap + "k_proj.weight"), H2, g.kv_head_begin * d, g.local_kv_heads * d), tag + "k rows");
  expect_device_equals(r.attn.v_proj, host_rows(fx.bytes(ap + "v_proj.weight"), H2, g.kv_head_begin * d, g.local_kv_heads * d), tag + "v rows");
  const int64_t Q = static_cast<int64_t>(cfg.num_attention_heads) * d;
  expect_device_equals(r.attn.o_proj, host_cols(fx.bytes(ap + "o_proj.weight"), cfg.hidden_size, Q * 2, g.head_begin * d * 2, g.local_heads * d * 2), tag + "o cols");
  require(r.attn.local_heads == g.local_heads && r.attn.local_kv_heads == g.local_kv_heads, tag + "head counts");
  if (world > cfg.num_key_value_heads)
    require(r.attn.kv_head_begin == rank / (world / cfg.num_key_value_heads), tag + "kv pairing");
  const std::string mp = p + "mlp.";
  const int64_t I = g.local_inter;
  require(r.mlp.local_inter == I, tag + "mlp inter");
  expect_device_equals(r.mlp.gate, host_rows(fx.bytes(mp + "gate_proj.weight"), H2, rank * I, I), tag + "gate rows");
  expect_device_equals(r.mlp.up, host_rows(fx.bytes(mp + "up_proj.weight"), H2, rank * I, I), tag + "up rows");
  expect_device_equals(r.mlp.down, host_cols(fx.bytes(mp + "down_proj.weight"), cfg.hidden_size, static_cast<size_t>(cfg.intermediate_size) * 2, rank * I * 2, I * 2), tag + "down cols");
  require(r.bytes == LlamaLayerStream::layer_bytes(cfg, r.layer, rank, world), tag + "layer bytes formula");
}

}  // namespace

// The fp8 check: no bf16 projection bytes resident, the fp8 payload and
// scale grid byte-identical to the public host encoder on the same slice
// (the kernels' determinism makes the launcher's own encoding the
// reference), and the formulas auto-deriving through the flag.
void check_fp8_layer(const Fixture& fx, const LlamaLayerStream& s, const dgpp::LlamaLayerResident& r) {
  const LlamaTextConfig& cfg = fx.cfg;
  const dgpp::LlamaLocalGeometry& g = s.geometry();
  const int world = s.world(), rank = s.rank();
  const std::string p = "model.layers." + std::to_string(r.layer) + ".";
  const std::string tag = "fp8 world " + std::to_string(world) + " rank " + std::to_string(rank) + " layer " +
                          std::to_string(r.layer) + " ";
  const int64_t d = cfg.head_dim, H = cfg.hidden_size;
  const size_t H2 = static_cast<size_t>(H) * 2;
  require(!r.attn.q_proj && !r.attn.k_proj && !r.attn.v_proj && !r.attn.o_proj, tag + "bf16 attn pointers null");
  require(!r.mlp.gate && !r.mlp.up && !r.mlp.down, tag + "bf16 mlp pointers null");
  require(r.attn.q_proj_fp8.payload && r.attn.q_proj_fp8.scales, tag + "q fp8 bound");
  require(r.mlp.gate_fp8.payload && r.mlp.down_fp8.payload, tag + "mlp fp8 bound");
  auto encode = [&](const std::vector<uint8_t>& slice, int64_t rows, int64_t cols, const void* payload,
                    const void* scales, const std::string& what) {
    std::vector<uint8_t> want_payload(static_cast<size_t>(rows) * cols);
    std::vector<float> want_scales(static_cast<size_t>(fp8_quant::scale_rows(rows)) * fp8_quant::scale_cols(cols));
    fp8_quant::encode_block128(reinterpret_cast<const uint16_t*>(slice.data()), static_cast<size_t>(cols), rows, cols,
                               want_payload.data(), want_scales.data());
    expect_device_equals(payload, want_payload, what + " codes");
    expect_device_equals(scales,
                         std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(want_scales.data()),
                                              reinterpret_cast<const uint8_t*>(want_scales.data()) +
                                                  want_scales.size() * 4),
                         what + " scales");
  };
  const std::string ap = p + "self_attn.";
  const int64_t q0 = static_cast<int64_t>(g.head_begin) * d, qn = static_cast<int64_t>(g.local_heads) * d;
  const int64_t k0 = static_cast<int64_t>(g.kv_head_begin) * d, kn = static_cast<int64_t>(g.local_kv_heads) * d;
  encode(host_rows(fx.bytes(ap + "q_proj.weight"), H2, q0, qn), qn, H, r.attn.q_proj_fp8.payload,
         r.attn.q_proj_fp8.scales, tag + "q");
  encode(host_rows(fx.bytes(ap + "k_proj.weight"), H2, k0, kn), kn, H, r.attn.k_proj_fp8.payload,
         r.attn.k_proj_fp8.scales, tag + "k");
  encode(host_rows(fx.bytes(ap + "v_proj.weight"), H2, k0, kn), kn, H, r.attn.v_proj_fp8.payload,
         r.attn.v_proj_fp8.scales, tag + "v");
  const int64_t Q = static_cast<int64_t>(cfg.num_attention_heads) * d;
  encode(host_cols(fx.bytes(ap + "o_proj.weight"), H, Q * 2, q0 * 2, qn * 2), H, qn, r.attn.o_proj_fp8.payload,
         r.attn.o_proj_fp8.scales, tag + "o");
  expect_device_equals(r.input_norm, fx.bytes(p + "input_layernorm.weight"), tag + "input norm");
  expect_device_equals(r.post_norm, fx.bytes(p + "post_attention_layernorm.weight"), tag + "post norm");
  const std::string mp = p + "mlp.";
  const int64_t I = g.local_inter;
  require(r.mlp.local_inter == I, tag + "mlp inter");
  encode(host_rows(fx.bytes(mp + "gate_proj.weight"), H2, rank * I, I), I, H, r.mlp.gate_fp8.payload,
         r.mlp.gate_fp8.scales, tag + "gate");
  encode(host_rows(fx.bytes(mp + "up_proj.weight"), H2, rank * I, I), I, H, r.mlp.up_fp8.payload,
         r.mlp.up_fp8.scales, tag + "up");
  encode(host_cols(fx.bytes(mp + "down_proj.weight"), H, static_cast<size_t>(cfg.intermediate_size) * 2, rank * I * 2,
                  I * 2),
         H, I, r.mlp.down_fp8.payload, r.mlp.down_fp8.scales, tag + "down");
}

DGPP_TEST(llama_loader_dense_weights_fp8) {
  const Fixture fx = write_fixture();
  const int layers = fx.cfg.num_hidden_layers;
  {
    struct Fp8On {
      const bool saved;
      Fp8On() : saved(LlamaLayerStream::dense_weights_fp8()) { LlamaLayerStream::set_dense_weights_fp8(true); }
      ~Fp8On() { LlamaLayerStream::set_dense_weights_fp8(saved); }
    } on;
    // Streaming at worlds 1 and 2 (the vocab-sharded head at 2).
    for (const int world : {1, 2}) {
      const int rank = 0;
      const auto head = world > 1 ? dgpp::LlamaHeadSharding::VocabSharded : dgpp::LlamaHeadSharding::Full;
      LlamaLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::LlamaResidency::Streaming, head);
      for (int l = 0; l < layers; ++l) {
        const uint64_t before = s.source_bytes_read();
        const auto& r = s.load_layer(l);
        require(r.layer == l, "fp8 layer index");
        check_fp8_layer(fx, s, r);
        require(r.bytes == LlamaLayerStream::layer_bytes(fx.cfg, l, rank, world), "fp8 layer bytes formula");
        require(s.source_bytes_read() - before == LlamaLayerStream::planned_layer_source_bytes(fx.cfg, l, rank, world),
                "fp8 source-byte plan");
        s.release_layer();
      }
      const auto& g = s.load_globals();
      require(!g.lm_head && g.lm_head_fp8.payload && g.lm_head_fp8.scales, "fp8 head bound, bf16 head null");
      const size_t H2 = static_cast<size_t>(fx.cfg.hidden_size) * 2;
      std::vector<uint8_t> head_payload(static_cast<size_t>(g.lm_vocab_count) * fx.cfg.hidden_size);
      std::vector<float> head_scales(static_cast<size_t>(fp8_quant::scale_rows(g.lm_vocab_count)) *
                                     fp8_quant::scale_cols(fx.cfg.hidden_size));
      fp8_quant::encode_block128(reinterpret_cast<const uint16_t*>(host_rows(fx.bytes("lm_head.weight"), H2,
                                                                                   g.lm_vocab_begin, g.lm_vocab_count)
                                                                            .data()),
                                 static_cast<size_t>(fx.cfg.hidden_size), g.lm_vocab_count, fx.cfg.hidden_size,
                                 head_payload.data(), head_scales.data());
      expect_device_equals(g.lm_head_fp8.payload, head_payload, "fp8 head codes");
      expect_device_equals(g.lm_head_fp8.scales,
                           std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(head_scales.data()),
                                                reinterpret_cast<const uint8_t*>(head_scales.data()) +
                                                    head_scales.size() * 4),
                           "fp8 head scales");
      require(g.lm_vocab_count == LlamaLayerStream::lm_vocab_count(fx.cfg, rank, world, head), "fp8 head count");
      require(g.bytes == LlamaLayerStream::globals_bytes(fx.cfg, rank, world, head), "fp8 globals formula");
    }
  }
  // The flag is process state: the RAII reset hands the next build bf16.
  LlamaLayerStream plain(fx.cfg, fx.dir, 0, 1, dgpp::LlamaResidency::Streaming, dgpp::LlamaHeadSharding::Full);
  const auto& r = plain.load_layer(0);
  require(r.attn.q_proj && r.mlp.gate && !r.attn.q_proj_fp8.payload, "the fp8 flag's scope ends with the test");
}

DGPP_TEST(llama_loader_slices_every_class_byte_exact_at_worlds_1_2_4) {
  const Fixture fx = write_fixture();
  const int layers = fx.cfg.num_hidden_layers;
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      LlamaLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::LlamaResidency::Streaming,
                         dgpp::LlamaHeadSharding::Full);
      for (int l = 0; l < layers; ++l) {
        const uint64_t before = s.source_bytes_read();
        const auto& r = s.load_layer(l);
        require(r.layer == l, "layer index");
        check_layer(fx, s, r);
        require(s.source_bytes_read() - before == LlamaLayerStream::planned_layer_source_bytes(fx.cfg, l, rank, world),
                "the source-byte plan equals the bytes read");
        s.release_layer();
      }
    }
  }
}

DGPP_TEST(llama_loader_globals_slice) {
  const Fixture fx = write_fixture();
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::LlamaHeadSharding::VocabSharded : dgpp::LlamaHeadSharding::Full;
      LlamaLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::LlamaResidency::Streaming, head);
      const auto& g = s.load_globals();
      const size_t H2 = static_cast<size_t>(fx.cfg.hidden_size) * 2;
      expect_device_equals(g.embed, fx.bytes("model.embed_tokens.weight"), "embed");
      expect_device_equals(g.final_norm, fx.bytes("model.norm.weight"), "final norm");
      expect_device_equals(g.lm_head, host_rows(fx.bytes("lm_head.weight"), H2, g.lm_vocab_begin, g.lm_vocab_count), "lm head slice");
      require(g.lm_vocab_count == LlamaLayerStream::lm_vocab_count(fx.cfg, rank, world, head), "lm head count");
      require(g.bytes == LlamaLayerStream::globals_bytes(fx.cfg, rank, world, head), "globals formula");
    }
  }
}

DGPP_TEST(llama_loader_resident_mode_and_image_round_trip) {
  const Fixture fx = write_fixture();
  const fs::path cache = fs::current_path() / "llama_loader_image_cache";
  fs::remove_all(cache);
  const std::string saved = LlamaLayerStream::resident_image_dir();
  LlamaLayerStream::set_resident_image_dir(cache.string());
  const int layers = fx.cfg.num_hidden_layers;
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(layers));
  {
    LlamaLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::LlamaResidency::Resident, dgpp::LlamaHeadSharding::VocabSharded);
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
    require(LlamaLayerStream::resident_bytes(fx.cfg, 1, 2, dgpp::LlamaHeadSharding::VocabSharded, false) ==
                [&] { size_t t = s.load_globals().bytes; for (int l = 0; l < layers; ++l) t += s.load_layer(l).bytes; return t; }(),
            "resident bytes formula");
    s.release_sources();
    require(s.sources_released(), "sources released");
    require(s.staging_released(), "the pinned staging mirror goes with the sources");
  }
  {
    LlamaLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::LlamaResidency::Resident, dgpp::LlamaHeadSharding::VocabSharded);
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      const auto span = s.resident_layer_span(l);
      require(device_bytes(span.first, span.second) == built[static_cast<size_t>(l)], "restored layer bitwise the built one");
      check_layer(fx, s, r);
    }
    require(s.image_layers_restored() == layers && s.source_bytes_read() == before,
            "every layer restored from the image, no source reads");
    const dgpp::LlamaReplicatedDigest d = s.hash_replicated();
    require(d.layer.size() == static_cast<size_t>(layers) && d.tensors > 0, "digest shape");
  }
  LlamaLayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
}

int main() { return dgpp::test::run_all(); }
