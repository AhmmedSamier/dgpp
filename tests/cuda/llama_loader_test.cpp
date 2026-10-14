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
#include "models/llama/binding.hpp"
#include "models/llama/config.hpp"
#include "models/llama/loader.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::LlamaExpectedTensor;
using dgpp::LlamaLayerStream;
using dgpp::LlamaTextConfig;

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
