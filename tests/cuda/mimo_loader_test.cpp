// The MiMo-V2.6-Flash resident loader on the synthetic fixture: every
// class lands byte-exact at worlds 1, 2 and 4 as the slice formulas of
// docs/mimo_v26_flash_plan.md §2 say (the fused projection's pre-sharded
// chunks stacked at the padded stride with zero padding rows and their
// per-chunk scale rows, o_proj's packed columns, the sink's heads widened
// to fp32, the fp8 dense MLP's row / column slices with their re-anchored
// scale grids, the MXFP4 experts' row / column slices, the replicated set
// verbatim); the byte formulas equal actual usage and the source-byte plan
// equals the bytes read; the globals; the checkpoint's unserved tensors
// ignored by the validator; resident cache hits and the image round trip.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "loaders/safetensors.hpp"
#include "mimo_fixture.hpp"
#include "models/mimo/binding.hpp"
#include "models/mimo/config.hpp"
#include "models/mimo/loader.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::MimoExpectedTensor;
using dgpp::MimoLayerStream;
using dgpp::MimoTextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  MimoTextConfig cfg;
  std::string dir;
  std::vector<MimoExpectedTensor> table;
  const MimoExpectedTensor& expected(const std::string& name) const {
    for (const auto& e : table)
      if (e.name == name) return e;
    throw std::runtime_error("fixture: no expected tensor " + name);
  }
  std::vector<uint8_t> bytes(const std::string& name) const {
    return mimofx::tensor_bytes(cfg, expected(name));
  }
};

Fixture write_fixture() {
  Fixture fx;
  fx.cfg = mimofx::tiny_config();
  fx.dir = (fs::current_path() / "mimo_loader_fixture").string();
  mimofx::write_fixture(fx.cfg, fx.dir);
  fx.table = dgpp::mimo_expected_text_tensors(fx.cfg);
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

// An fp8 block-128 view against the checkpoint's row / column slice of
// `name` (the payload) and its re-anchored scale grid.
void check_fp8(const Fixture& fx, const dgpp::GlmQuantMatrix& m, const std::string& name, bool rows_slice,
               int64_t start, int64_t count, const std::string& tag) {
  const auto& ep = fx.expected(name);
  const int64_t N = ep.shape[0], K = ep.shape[1];
  const int64_t sb_full = (K + 127) / 128;
  require(m.scale_block_rows == 128 && m.scale_block_cols == 128, tag + " 128 x 128 grid (the slices are 128-aligned)");
  const std::vector<uint8_t> scales = fx.bytes(name + "_scale_inv");
  if (rows_slice) {
    require(m.rows == count && m.cols == K, tag + " geometry");
    expect_device_equals(m.payload, host_rows(fx.bytes(name), K, start, count), tag + " payload rows");
    expect_device_equals(m.scales, host_rows(scales, sb_full * 4, start / 128, (count + 127) / 128), tag + " scale rows");
  } else {
    require(m.rows == N && m.cols == count, tag + " geometry");
    expect_device_equals(m.payload, host_cols(fx.bytes(name), N, K, start, count), tag + " payload cols");
    expect_device_equals(m.scales, host_cols(scales, (N + 127) / 128, sb_full * 4, (start / 128) * 4, ((count + 127) / 128) * 4),
                         tag + " scale cols");
  }
}

// An MXFP4 view against the checkpoint's row / column slice of `base`.
void check_mxfp4(const Fixture& fx, const dgpp::GlmFp4Matrix& m, const std::string& base, bool rows_slice,
                 int64_t start, int64_t count, const std::string& tag) {
  const auto& ep = fx.expected(base + ".weight");
  const int64_t N = ep.shape[0], K = ep.shape[1] * 2;
  require(m.mxfp4() && m.global_scale == nullptr, tag + " MXFP4 form");
  if (rows_slice) {
    require(m.rows == count && m.cols == K, tag + " geometry");
    expect_device_equals(m.payload, host_rows(fx.bytes(base + ".weight"), K / 2, start, count), tag + " payload rows");
    expect_device_equals(m.scales, host_rows(fx.bytes(base + ".weight_scale"), K / 32, start, count), tag + " scale rows");
  } else {
    require(m.rows == N && m.cols == count, tag + " geometry");
    expect_device_equals(m.payload, host_cols(fx.bytes(base + ".weight"), N, K / 2, start / 2, count / 2), tag + " payload cols");
    expect_device_equals(m.scales, host_cols(fx.bytes(base + ".weight_scale"), N, K / 32, start / 32, count / 32), tag + " scale cols");
  }
}

// The fused projection: the rank's chunks stacked at the padded stride.
void check_qkv(const Fixture& fx, const dgpp::MimoAttnResident& a, const std::string& base, int layer,
               const dgpp::MimoLocalGeometry& g, const std::string& tag) {
  const MimoTextConfig& cfg = fx.cfg;
  const int64_t H = cfg.hidden_size;
  const int64_t chunk_rows = cfg.qkv_chunk_rows(layer);
  const int64_t sr = (chunk_rows + 127) / 128, sb = H / 128, stride = sr * 128;
  require(a.chunks == g.chunks && a.chunk_rows == chunk_rows && a.chunk_stride == stride, tag + " chunk geometry");
  require(a.q_per_chunk == cfg.num_attention_heads / cfg.qkv_chunks() && a.kv_per_chunk == cfg.kv_heads_of(layer) / cfg.qkv_chunks(),
          tag + " heads per chunk");
  const int64_t rows = static_cast<int64_t>(a.chunks - 1) * stride + chunk_rows;
  require(a.qkv.rows == rows && a.qkv.cols == H, tag + " qkv geometry");
  const std::vector<uint8_t> payload = fx.bytes(base + ".weight"), scales = fx.bytes(base + ".weight_scale_inv");
  const std::vector<uint8_t> got_p = device_bytes(a.qkv.payload, static_cast<size_t>(rows) * H);
  const std::vector<uint8_t> got_s = device_bytes(a.qkv.scales, static_cast<size_t>(a.chunks) * sr * sb * 4);
  for (int i = 0; i < a.chunks; ++i) {
    const int64_t c = g.chunk_begin + i;
    const std::vector<uint8_t> want = host_rows(payload, H, c * chunk_rows, chunk_rows);
    require(std::equal(want.begin(), want.end(), got_p.begin() + static_cast<size_t>(i) * stride * H),
            tag + " chunk " + std::to_string(i) + " payload rows");
    if (i + 1 < a.chunks)
      for (int64_t x = chunk_rows * H; x < stride * H; ++x)
        require(got_p[static_cast<size_t>(i) * stride * H + x] == 0, tag + " padding rows are zero");
    const std::vector<uint8_t> ws = host_rows(scales, sb * 4, c * sr, sr);
    require(std::equal(ws.begin(), ws.end(), got_s.begin() + static_cast<size_t>(i) * sr * sb * 4),
            tag + " chunk " + std::to_string(i) + " scale rows");
  }
}

void check_layer(const Fixture& fx, const MimoLayerStream& s, const dgpp::MimoLayerResident& r) {
  const MimoTextConfig& cfg = fx.cfg;
  const dgpp::MimoLocalGeometry& g = s.geometry();
  const int world = s.world(), rank = s.rank();
  const std::string p = dgpp::mimo_layer_prefix(cfg, r.layer);
  const std::string tag = "world " + std::to_string(world) + " rank " + std::to_string(rank) + " layer " +
                          std::to_string(r.layer) + " ";
  const bool draft = r.layer == cfg.mtp_layer();
  require(r.moe == cfg.is_moe_layer(r.layer) && r.swa == cfg.is_swa_layer(r.layer), tag + "layer kind");
  expect_device_equals(r.input_norm, fx.bytes(p + "input_layernorm.weight"), tag + "input norm");
  expect_device_equals(r.post_norm, fx.bytes(p + (draft ? "pre_mlp_layernorm.weight" : "post_attention_layernorm.weight")),
                       tag + "post norm");
  // Attention: the fused projection's chunks, the packed o columns, the sink.
  const std::string ap = p + "self_attn.";
  require(r.attn.local_heads == g.local_heads && r.attn.head_begin == g.head_begin, tag + "head counts");
  require(r.attn.local_kv_heads == g.kv_heads_of(cfg, r.layer) &&
              r.attn.kv_head_begin == (r.swa ? g.swa_kv_head_begin : g.kv_head_begin) && r.attn.swa == r.swa,
          tag + "kv head counts");
  check_qkv(fx, r.attn, ap + "qkv_proj", r.layer, g, tag + "qkv");
  const int64_t O = cfg.o_proj_cols(), dv = cfg.v_head_dim;
  expect_device_equals(r.attn.o_proj, host_cols(fx.bytes(ap + "o_proj.weight"), cfg.hidden_size, O * 2, g.head_begin * dv * 2, g.local_heads * dv * 2),
                       tag + "o cols");
  if (cfg.sink_of(r.layer)) {
    require(r.attn.sink != nullptr, tag + "sink present");
    const std::vector<uint8_t> src = fx.bytes(ap + "attention_sink_bias");
    std::vector<float> want(static_cast<size_t>(g.local_heads));
    for (int h = 0; h < g.local_heads; ++h) {
      uint16_t bits;
      std::memcpy(&bits, src.data() + static_cast<size_t>(g.head_begin + h) * 2, 2);
      want[static_cast<size_t>(h)] = dgpp::bf16_bits_to_float(bits);
    }
    std::vector<uint8_t> wb(want.size() * 4);
    std::memcpy(wb.data(), want.data(), wb.size());
    expect_device_equals(r.attn.sink, wb, tag + "sink (fp32, the rank's heads)");
  } else {
    require(r.attn.sink == nullptr, tag + "no sink on a global layer");
  }
  if (!r.moe) {
    const int64_t I = g.local_dense_inter;
    require(r.dense.local_inter == I, tag + "dense inter");
    check_fp8(fx, r.dense.gate, p + "mlp.gate_proj.weight", true, rank * I, I, tag + "dense gate");
    check_fp8(fx, r.dense.up, p + "mlp.up_proj.weight", true, rank * I, I, tag + "dense up");
    check_fp8(fx, r.dense.down, p + "mlp.down_proj.weight", false, rank * I, I, tag + "dense down");
  } else {
    const std::string mp = p + "mlp.";
    expect_device_equals(r.moe_w.router, fx.bytes(mp + "gate.weight"), tag + "router");
    expect_device_equals(r.moe_w.router_bias, fx.bytes(mp + "gate.e_score_correction_bias"), tag + "router bias");
    const int64_t I = g.local_inter;
    require(r.moe_w.local_inter == I, tag + "moe inter");
    require(static_cast<int>(r.moe_w.experts.size()) == cfg.n_routed_experts * 3, tag + "expert count");
    for (int e = 0; e < cfg.n_routed_experts; ++e) {
      const std::string ep = mp + "experts." + std::to_string(e) + ".";
      check_mxfp4(fx, r.moe_w.expert(e, 0), ep + "gate_proj", true, rank * I, I, tag + "expert gate");
      check_mxfp4(fx, r.moe_w.expert(e, 1), ep + "up_proj", true, rank * I, I, tag + "expert up");
      check_mxfp4(fx, r.moe_w.expert(e, 2), ep + "down_proj", false, rank * I, I, tag + "expert down");
    }
  }
  if (draft) {
    expect_device_equals(r.enorm, fx.bytes(p + "enorm.weight"), tag + "enorm");
    expect_device_equals(r.hnorm, fx.bytes(p + "hnorm.weight"), tag + "hnorm");
    expect_device_equals(r.eh_proj, fx.bytes(p + "eh_proj.weight"), tag + "eh_proj");
    expect_device_equals(r.final_norm, fx.bytes(p + "final_layernorm.weight"), tag + "final_layernorm");
  } else {
    require(r.enorm == nullptr && r.eh_proj == nullptr && r.final_norm == nullptr, tag + "no draft head on a main layer");
  }
  require(r.bytes == MimoLayerStream::layer_bytes(cfg, r.layer, rank, world), tag + "layer bytes formula");
}

}  // namespace

DGPP_TEST(mimo_loader_geometry_at_worlds_1_2_4) {
  const MimoTextConfig cfg = mimofx::tiny_config();
  for (const int world : {1, 2, 4})
    for (int rank = 0; rank < world; ++rank) {
      const dgpp::MimoLocalGeometry g =
          dgpp::MimoLocalGeometry::from_config(cfg, rank, world, dgpp::MimoHeadSharding::VocabSharded);
      require(g.chunks == 4 / world && g.chunk_begin == rank * (4 / world), "chunks");
      require(g.local_heads == 8 / world && g.head_begin == rank * (8 / world), "query heads");
      require(g.local_kv_heads == 4 / world && g.kv_head_begin == rank * (4 / world), "global kv heads");
      require(g.local_swa_kv_heads == 8 / world && g.swa_kv_head_begin == rank * (8 / world), "swa kv heads");
      require(g.local_inter == 128 / world && g.local_dense_inter == 512 / world, "inter slices");
      require(g.lm_vocab_count == 512 / world && g.lm_vocab_begin == rank * (512 / world), "vocab slice");
      require(g.kv_heads_of(cfg, 0) == 4 / world && g.kv_heads_of(cfg, 1) == 8 / world && g.kv_heads_of(cfg, 4) == 8 / world,
              "per-layer kv heads (the draft is SWA)");
    }
  bool refused = false;
  try {
    (void)dgpp::MimoLocalGeometry::from_config(cfg, 0, 8, dgpp::MimoHeadSharding::Full);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "world 8 is refused (four chunks)");
}

DGPP_TEST(mimo_loader_validator_ignores_the_fixtures_unserved_tensors) {
  const Fixture fx = write_fixture();
  std::unordered_map<std::string, dgpp::MimoTensorDesc> present;
  auto f = dgpp::SafetensorsFile::open((fs::path(fx.dir) / "model.safetensors").string());
  f->for_each([&](const dgpp::TensorInfo& t) { present.emplace(t.name, dgpp::MimoTensorDesc{t.dtype, t.shape}); });
  const dgpp::MimoBindReport rep = dgpp::mimo_validate_text_binding(fx.cfg, present);
  require(rep.ok() && rep.matched == fx.table.size(), "the fixture binds");
  require(rep.ignored == 4, "the four unserved tensors are ignored, got " + std::to_string(rep.ignored));
  require(rep.fp8_matrices == 4 + 3 + 4 && rep.fp4_matrices == 3 * 8 * 3, "matrix counts");
}

DGPP_TEST(mimo_loader_slices_every_class_byte_exact_at_worlds_1_2_4) {
  const Fixture fx = write_fixture();
  const int layers = fx.cfg.num_hidden_layers + 1;
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      MimoLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::MimoResidency::Streaming,
                        dgpp::MimoHeadSharding::Full, /*resident_mtp=*/true);
      for (int l = 0; l < layers; ++l) {
        const uint64_t before = s.source_bytes_read();
        const auto& r = s.load_layer(l);
        require(r.layer == l, "layer index");
        check_layer(fx, s, r);
        require(s.source_bytes_read() - before == MimoLayerStream::planned_layer_source_bytes(fx.cfg, l, rank, world),
                "the source-byte plan equals the bytes read");
        s.release_layer();
      }
    }
  }
}

DGPP_TEST(mimo_loader_globals_slices) {
  const Fixture fx = write_fixture();
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::MimoHeadSharding::VocabSharded : dgpp::MimoHeadSharding::Full;
      MimoLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::MimoResidency::Streaming, head);
      const auto& g = s.load_globals();
      const size_t H2 = static_cast<size_t>(fx.cfg.hidden_size) * 2;
      expect_device_equals(g.embed, fx.bytes("model.embed_tokens.weight"), "embed");
      expect_device_equals(g.final_norm, fx.bytes("model.norm.weight"), "final norm");
      expect_device_equals(g.lm_head, host_rows(fx.bytes("lm_head.weight"), H2, g.lm_vocab_begin, g.lm_vocab_count), "lm head slice");
      require(g.lm_vocab_count == MimoLayerStream::lm_vocab_count(fx.cfg, rank, world, head), "lm head count");
      require(g.bytes == MimoLayerStream::globals_bytes(fx.cfg, rank, world, head), "globals formula");
    }
  }
}

DGPP_TEST(mimo_loader_resident_mode_and_image_round_trip) {
  const Fixture fx = write_fixture();
  const fs::path cache = fs::current_path() / "mimo_loader_image_cache";
  fs::remove_all(cache);
  const std::string saved = MimoLayerStream::resident_image_dir();
  MimoLayerStream::set_resident_image_dir(cache.string());
  const int layers = fx.cfg.num_hidden_layers + 1;
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(layers));
  {
    MimoLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::MimoResidency::Resident, dgpp::MimoHeadSharding::VocabSharded,
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
    require(MimoLayerStream::resident_bytes(fx.cfg, 1, 2, dgpp::MimoHeadSharding::VocabSharded, true) ==
                [&] { size_t t = s.load_globals().bytes; for (int l = 0; l < layers; ++l) t += s.load_layer(l).bytes; return t; }(),
            "resident bytes formula");
    s.release_sources();
    require(s.sources_released(), "sources released");
    require(s.staging_released(), "the pinned staging mirror goes with the sources");
  }
  {
    MimoLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::MimoResidency::Resident, dgpp::MimoHeadSharding::VocabSharded,
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
    const dgpp::MimoReplicatedDigest d = s.hash_replicated();
    require(d.layer.size() == static_cast<size_t>(layers) && d.tensors > 0, "digest shape");
  }
  MimoLayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
}

int main() { return dgpp::test::run_all(); }
