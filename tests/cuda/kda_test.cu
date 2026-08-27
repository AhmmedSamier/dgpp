// M2 KDA operator tests. These pin the milestone exit criteria:
//   * kernel parity against host fp32/fp64 references (declared tolerances);
//   * chunked vs unchunked prefill equivalence (bitwise on the GEMM-free
//     path, tolerance on the full layer);
//   * decode continuation, graph-capture determinism, snapshot round-trip;
//   * head-slice (TP-readiness) core-output agreement.
// Reference-dump parity lives in kda_dump_parity.cpp (host-only TU; its
// reader needs minijson, which nvcc refuses) and runs via --dump-file.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "core/graph.hpp"
#include "kernels/kda.hpp"
#include "models/kda_snapshot.hpp"
#include "tests/cuda/kda_test_helpers.hpp"

// Defined in tests/cuda/kda_dump_parity.cpp.
int run_kda_dump_parity(const std::string& path);

namespace {

using namespace dgpp::kda_test;
using dgpp::kda_causal_conv_silu_bf16;
using dgpp::kda_recurrent_fwd;

ptrdiff_t byte_diff(const void* a, const void* b) {
  return reinterpret_cast<const uint8_t*>(a) - reinterpret_cast<const uint8_t*>(b);
}

}  // namespace

// ---------------------------------------------------------------------------
// State pool: geometry wiring, slot isolation, snapshot round-trip
// ---------------------------------------------------------------------------

DGPP_TEST(kda_state_pool_slots_isolation_and_snapshot_roundtrip) {
  KdaConfig cfg;
  cfg.hidden = 64;
  cfg.heads = 4;
  cfg.head_dim = 32;
  cfg.num_kda_layers = 2;
  cfg.spec_width = 3;  // full slot width 6; committed width 3
  const KdaGeometry g = KdaGeometry::from_config(cfg);

  dgpp::Arena arena;
  dgpp::Arena::Config ac;
  ac.persistent_hot = 3 * g.pool_bytes;  // 3 slots
  arena.init(ac);

  KdaStatePool pool;
  pool.init(arena, cfg, 3);
  cudaStream_t s = test_stream();

  // GIVEN distinct garbage in every slot:
  pool.zero_all(s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  for (int slot = 0; slot < 3; ++slot)
    for (int layer = 0; layer < cfg.num_kda_layers; ++layer) {
      DGPP_CUDA_OK(cudaMemsetAsync(pool.recurrent(slot, layer), 0x5A + slot,
                                   g.recurrent_bytes, s));
      DGPP_CUDA_OK(cudaMemsetAsync(pool.conv(slot, layer), 0xA5 + slot,
                                   g.conv_slot_bytes, s));
    }
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  // WHEN one slot is zeroed:
  pool.zero_slot(1, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  // THEN the other slots keep their content and the geometry offsets are
  // exactly the DESIGN layout (recurrent block, then conv block, per layer).
  std::vector<uint8_t> probe(g.slot_bytes);
  pool.export_snapshot(0, probe.data(), probe.size(), s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  for (uint8_t byte : probe)
    if (byte == 0) throw std::runtime_error("zero_slot touched a foreign slot");
  if (byte_diff(pool.conv(0, 0), pool.recurrent(0, 0)) !=
      static_cast<ptrdiff_t>(g.recurrent_bytes))
    throw std::runtime_error("conv block offset");
  if (byte_diff(pool.recurrent(0, 1), pool.recurrent(0, 0)) !=
      static_cast<ptrdiff_t>(g.recurrent_bytes + g.conv_slot_bytes))
    throw std::runtime_error("layer stride");
  if (byte_diff(pool.recurrent(1, 0), pool.recurrent(0, 0)) !=
      static_cast<ptrdiff_t>(g.slot_bytes))
    throw std::runtime_error("slot stride");

  // GIVEN two layers of forward progress on slot 0:
  const int tokens = 5;
  TestWeights tw0 = TestWeights::random(cfg, 99);
  TestWeights tw1 = TestWeights::random(cfg, 100);
  DeviceWeights dw0(tw0), dw1(tw1);
  LayerEnv env(cfg, tokens, s, 2);  // two layers share the scratch arena
  KdaLayer layer0(env.arena, env.gemm, dw0.views(), cfg, tokens, env.ws.p,
                  env.ws.bytes);
  KdaLayer layer1(env.arena, env.gemm, dw1.views(), cfg, tokens, env.ws.p,
                  env.ws.bytes);
  if (!layer0.prepare(tokens) || !layer1.prepare(tokens))
    throw std::runtime_error("gemm plans unavailable");

  std::vector<uint16_t> hidden_in =
      random_bf16_bits(7, int64_t(tokens) * cfg.hidden, -2, 0);
  std::vector<uint16_t> hidden_next =
      random_bf16_bits(8, int64_t(tokens) * cfg.hidden, -2, 0);
  DevBuf in1(hidden_in.size() * 2), in2(hidden_next.size() * 2);
  in1.upload(hidden_in.data(), hidden_in.size() * 2);
  in2.upload(hidden_next.data(), hidden_next.size() * 2);
  DevBuf out1(int64_t(tokens) * cfg.hidden * 2), out2(int64_t(tokens) * cfg.hidden * 2);
  DevBuf out1b(out1.bytes), out2b(out2.bytes);

  pool.zero_slot(0, s);
  layer0.enqueue(in1.p, pool.recurrent(0, 0), pool.conv(0, 0),
                 g.conv_state_width, out1.p, tokens, s);
  layer1.enqueue(out1.p, pool.recurrent(0, 1), pool.conv(0, 1),
                 g.conv_state_width, out2.p, tokens, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  // WHEN the state snapshot is exported and imported into another slot:
  dgpp::KdaSnapshotHeader header =
      pool.snapshot_header("test-rev", "bf16-act/f32-state");
  std::vector<uint8_t> snap(sizeof(header) + pool.snapshot_bytes());
  std::memcpy(snap.data(), &header, sizeof(header));
  pool.export_snapshot(0, snap.data() + sizeof(header), pool.snapshot_bytes(), s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  pool.zero_slot(2, s);
  pool.import_snapshot(2, header, snap.data() + sizeof(header),
                       pool.snapshot_bytes(), s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  // THEN continuing both slots through both layers reproduces bitwise output:
  layer0.enqueue(in2.p, pool.recurrent(0, 0), pool.conv(0, 0),
                 g.conv_state_width, out1.p, tokens, s);
  layer1.enqueue(out1.p, pool.recurrent(0, 1), pool.conv(0, 1),
                 g.conv_state_width, out2.p, tokens, s);
  layer0.enqueue(in2.p, pool.recurrent(2, 0), pool.conv(2, 0),
                 g.conv_state_width, out1b.p, tokens, s);
  layer1.enqueue(out1b.p, pool.recurrent(2, 1), pool.conv(2, 1),
                 g.conv_state_width, out2b.p, tokens, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  std::vector<uint16_t> host_a(out2.bytes / 2), host_b(out2b.bytes / 2);
  out2.download(host_a.data(), out2.bytes);
  out2b.download(host_b.data(), out2b.bytes);
  require_bitwise("snapshot continuation diverged", host_a.data(), host_b.data(),
                  host_a.size() * 2);

  // AND a tampered header is rejected before any bytes land:
  dgpp::KdaSnapshotHeader bad = header;
  bad.tp_size = 2;
  try {
    pool.import_snapshot(2, bad, snap.data() + sizeof(header),
                         pool.snapshot_bytes(), s);
    throw std::runtime_error("tampered snapshot accepted");
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find("tampered") != std::string::npos) throw;
  }
}

// ---------------------------------------------------------------------------
// Conv kernel parity
// ---------------------------------------------------------------------------

DGPP_TEST(kda_conv_kernel_matches_host_reference) {
  cudaStream_t s = test_stream();
  for (int tokens : {1, 2, 3, 64, 2048}) {
    const int channels = 96, conv_width = 4, state_width = 6;
    std::vector<uint16_t> src =
        random_bf16_normal(21, int64_t(tokens) * channels, 1.0f);
    std::vector<uint16_t> weight =
        random_bf16_uniform(22, int64_t(channels) * conv_width, 0.2f);
    std::vector<uint16_t> state_init =
        random_bf16_normal(23, int64_t(channels) * state_width, 0.5f);

    DevBuf dsrc(int64_t(tokens) * channels * 2), dw(weight.size() * 2),
        dstate(state_init.size() * 2), ddst(int64_t(tokens) * channels * 2);
    dsrc.upload(src.data(), src.size() * 2);
    dw.upload(weight.data(), weight.size() * 2);
    dstate.upload(state_init.data(), state_init.size() * 2);

    kda_causal_conv_silu_bf16(dsrc.p, channels, dw.p, dstate.p, state_width,
                              ddst.p, tokens, channels, conv_width, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));

    std::vector<uint16_t> got(src.size()), got_state(state_init.size());
    ddst.download(got.data(), got.size() * 2);
    dstate.download(got_state.data(), got_state.size() * 2);

    // Host reference starts from the same bytes.
    std::vector<uint16_t> want(src.size()), want_state = state_init;
    dgpp::kda_ref::conv_silu<float>(src.data(), channels, weight.data(),
                                    want_state.data(), state_width,
                                    want.data(), tokens, channels, conv_width);

    // Outputs: silu expf differs between device and host by ~1 ulp.
    const Stats st = compare_bf16(got, want, 1);
    require_bf16("conv output T=" + std::to_string(tokens), st, 2e-2, 0.01);
    // Rolled history is a byte-exact copy of the trailing inputs.
    for (int c = 0; c < channels; ++c)
      for (int j = 0; j < conv_width - 1; ++j)
        if (got_state[static_cast<size_t>(c) * state_width + j] !=
            want_state[static_cast<size_t>(c) * state_width + j])
          throw std::runtime_error("conv state mismatch (committed columns)");
    // The speculative reserve columns must be untouched.
    for (int c = 0; c < channels; ++c)
      for (int j = conv_width - 1; j < state_width; ++j)
        if (got_state[static_cast<size_t>(c) * state_width + j] !=
            state_init[static_cast<size_t>(c) * state_width + j])
          throw std::runtime_error("conv kernel wrote reserve columns");
  }

  // A tokens < history-1 chunk must blend old state with new inputs.
  {
    const int channels = 8, conv_width = 4, state_width = 3, tokens = 2;
    std::vector<uint16_t> src = random_bf16_normal(31, tokens * channels, 1.0f);
    std::vector<uint16_t> weight =
        random_bf16_uniform(32, channels * conv_width, 0.2f);
    std::vector<uint16_t> state_init =
        random_bf16_normal(33, channels * state_width, 0.5f);
    DevBuf dsrc(tokens * channels * 2), dw(weight.size() * 2),
        dstate(state_init.size() * 2), ddst(tokens * channels * 2);
    dsrc.upload(src.data(), src.size() * 2);
    dw.upload(weight.data(), weight.size() * 2);
    dstate.upload(state_init.data(), state_init.size() * 2);
    kda_causal_conv_silu_bf16(dsrc.p, channels, dw.p, dstate.p, state_width,
                              ddst.p, tokens, channels, conv_width, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    std::vector<uint16_t> got_state(state_init.size());
    dstate.download(got_state.data(), got_state.size() * 2);
    for (int c = 0; c < channels; ++c) {
      // Expected: drop the oldest `tokens` entries, append the new inputs.
      const uint16_t expect[3] = {state_init[static_cast<size_t>(c) * 3 + 2],
                                  src[static_cast<size_t>(c)],
                                  src[channels + c]};
      for (int j = 0; j < 3; ++j)
        if (got_state[static_cast<size_t>(c) * 3 + j] != expect[j])
          throw std::runtime_error("short-chunk state blend mismatch");
    }
  }
}

// ---------------------------------------------------------------------------
// Recurrent kernel parity (fp32 mirror + fp64 oracle)
// ---------------------------------------------------------------------------

namespace {

struct RecurrentCase {
  int heads, k_dim, v_dim, tokens;
  uint64_t seed;
  bool zero_state;
};

void run_recurrent_case(const RecurrentCase& rc) {
  cudaStream_t s = test_stream();
  const int64_t qkv_stride = 2 * rc.heads * rc.k_dim + rc.heads * rc.v_dim;
  const int64_t n_qkv = int64_t(rc.tokens) * qkv_stride;
  const int64_t n_state = int64_t(rc.heads) * rc.v_dim * rc.k_dim;

  std::vector<uint16_t> qkv = random_bf16_normal(rc.seed, n_qkv, 1.0f);
  // Exercise the l2norm eps: one all-zero token row must not produce NaNs.
  if (rc.tokens >= 2) {
    for (int64_t i = 0; i < qkv_stride; ++i) qkv[qkv_stride + i] = 0;
  }
  std::vector<uint16_t> g1 =
      random_bf16_normal(rc.seed ^ 0x11, int64_t(rc.tokens) * rc.heads * rc.k_dim, 1.0f);
  std::vector<uint16_t> beta =
      random_bf16_normal(rc.seed ^ 0x22, int64_t(rc.tokens) * rc.heads, 1.0f);
  std::vector<float> a_log = random_f32_uniform(rc.seed ^ 0x33, rc.heads, 0.5f);
  std::vector<float> dt_bias =
      random_f32_uniform(rc.seed ^ 0x44, int64_t(rc.heads) * rc.k_dim, 0.5f);
  std::vector<float> state_init =
      rc.zero_state ? std::vector<float>(n_state)
                    : random_f32_uniform(rc.seed ^ 0x55, n_state, 0.05f);

  const float lower_bound = -5.0f;
  const float scale = ref_scale(rc.k_dim);

  DevBuf dqkv(n_qkv * 2), dg1(g1.size() * 2), dbeta(beta.size() * 2),
      dalog(a_log.size() * 4), dtb(dt_bias.size() * 4), dstate(n_state * 4),
      dout(int64_t(rc.tokens) * rc.heads * rc.v_dim * 2);
  dqkv.upload(qkv.data(), qkv.size() * 2);
  dg1.upload(g1.data(), g1.size() * 2);
  dbeta.upload(beta.data(), beta.size() * 2);
  dalog.upload(a_log.data(), a_log.size() * 4);
  dtb.upload(dt_bias.data(), dt_bias.size() * 4);
  dstate.upload(state_init.data(), state_init.size() * 4);

  kda_recurrent_fwd(dqkv.p, dg1.p, dbeta.p, rc.heads, dalog.as<float>(),
                    dtb.as<float>(), dstate.as<float>(), dout.p, rc.tokens,
                    rc.heads, rc.k_dim, rc.v_dim, lower_bound, scale, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  std::vector<uint16_t> got_out(int64_t(rc.tokens) * rc.heads * rc.v_dim);
  std::vector<float> got_state(n_state);
  dout.download(got_out.data(), got_out.size() * 2);
  dstate.download(got_state.data(), got_state.size() * 4);

  // fp32 mirror reference: same precision class as the device.
  {
    std::vector<float> ref_state = state_init;
    std::vector<uint16_t> ref_out(got_out.size());
    dgpp::kda_ref::recurrent<float>(
        qkv.data(), g1.data(), beta.data(), rc.heads, a_log.data(),
        dt_bias.data(), ref_state.data(), ref_out.data(), rc.tokens, rc.heads,
        rc.k_dim, rc.v_dim, lower_bound, scale);
    const std::string tag = "recurrent f32 T=" + std::to_string(rc.tokens) +
                            " H=" + std::to_string(rc.heads);
    const Stats ss = compare_abs_rel(got_state.data(), ref_state.data(),
                                     n_state, 1e-4, 1e-5);
    require_rel(tag + " state", ss, 1e-4, 0.0);
    const Stats so = compare_bf16(got_out, ref_out, 1);
    require_bf16(tag + " out", so, 2e-2, 0.02);
  }
  // fp64 oracle: measures pure fp32 accumulation drift.
  {
    std::vector<double> ref_state(state_init.begin(), state_init.end());
    std::vector<uint16_t> ref_out(got_out.size());
    dgpp::kda_ref::recurrent<double>(
        qkv.data(), g1.data(), beta.data(), rc.heads, a_log.data(),
        dt_bias.data(), ref_state.data(), ref_out.data(), rc.tokens, rc.heads,
        rc.k_dim, rc.v_dim, lower_bound, scale);
    const std::string tag = "recurrent f64 T=" + std::to_string(rc.tokens) +
                            " H=" + std::to_string(rc.heads);
    const Stats ss = compare_abs_rel(got_state.data(), ref_state.data(),
                                     n_state, 1e-4, 1e-5);
    require_rel(tag + " state", ss, 1e-4, 0.0);
    const Stats so = compare_bf16(got_out, ref_out, 2);
    require_bf16(tag + " out", so, 2e-2, 0.02);
  }
}

}  // namespace

DGPP_TEST(kda_recurrent_kernel_matches_host_reference) {
  for (bool zero : {true, false}) {
    const uint64_t z = zero ? 1u : 0u;
    run_recurrent_case({4, 32, 32, 1, 40 + z, zero});
    run_recurrent_case({4, 32, 32, 7, 42 + z, zero});
    run_recurrent_case({2, 64, 64, 257, 44 + z, zero});
  }
  // Real per-rank geometry (TP=4 head count, full head_dim).
  run_recurrent_case({16, 128, 128, 65, 60, true});
  run_recurrent_case({16, 128, 128, 65, 61, false});
}

// ---------------------------------------------------------------------------
// Chunked vs unchunked recurrence: bitwise when no GEMM is involved
// ---------------------------------------------------------------------------

DGPP_TEST(kda_chunked_vs_unchunked_recurrence_is_bitwise) {
  cudaStream_t s = test_stream();
  const int heads = 8, dim = 64, tokens = 512;
  const int64_t qkv_stride = 3 * heads * dim;
  const int64_t n_state = int64_t(heads) * dim * dim;
  const int channels = 3 * heads * dim, conv_width = 4, state_width = 3;

  std::vector<uint16_t> qkv_pre =
      random_bf16_normal(70, int64_t(tokens) * qkv_stride, 1.0f);
  std::vector<uint16_t> g1 =
      random_bf16_normal(71, int64_t(tokens) * heads * dim, 1.0f);
  std::vector<uint16_t> beta =
      random_bf16_normal(72, int64_t(tokens) * heads, 1.0f);
  std::vector<float> a_log = random_f32_uniform(73, heads, 0.5f);
  std::vector<float> dt_bias = random_f32_uniform(74, heads * dim, 0.5f);
  std::vector<uint16_t> conv_w =
      random_bf16_uniform(75, int64_t(channels) * conv_width, 0.2f);
  const float lower_bound = -5.0f, scale = ref_scale(dim);

  DevBuf dpre(qkv_pre.size() * 2), dg1(g1.size() * 2), dbeta(beta.size() * 2),
      dalog(a_log.size() * 4), dtb(dt_bias.size() * 4), dcw(conv_w.size() * 2);
  dpre.upload(qkv_pre.data(), qkv_pre.size() * 2);
  dg1.upload(g1.data(), g1.size() * 2);
  dbeta.upload(beta.data(), beta.size() * 2);
  dalog.upload(a_log.data(), a_log.size() * 4);
  dtb.upload(dt_bias.data(), dt_bias.size() * 4);
  dcw.upload(conv_w.data(), conv_w.size() * 2);

  // Two independent device runs: A unchunked, B in mixed chunks including
  // decode-style single tokens. Both start from zeroed state.
  DevBuf state_a(n_state * 4), state_b(n_state * 4);
  DevBuf conv_a(int64_t(channels) * state_width * 2),
      conv_b(int64_t(channels) * state_width * 2);
  DevBuf qkv_a(int64_t(tokens) * qkv_stride * 2), qkv_b(qkv_a.bytes);
  DevBuf out_a(int64_t(tokens) * heads * dim * 2), out_b(out_a.bytes);
  DGPP_CUDA_OK(cudaMemsetAsync(state_a.p, 0, state_a.bytes, s));
  DGPP_CUDA_OK(cudaMemsetAsync(state_b.p, 0, state_b.bytes, s));
  DGPP_CUDA_OK(cudaMemsetAsync(conv_a.p, 0, conv_a.bytes, s));
  DGPP_CUDA_OK(cudaMemsetAsync(conv_b.p, 0, conv_b.bytes, s));

  // Run A: one call.
  kda_causal_conv_silu_bf16(dpre.p, qkv_stride, dcw.p, conv_a.p, state_width,
                            qkv_a.p, tokens, channels, conv_width, s);
  kda_recurrent_fwd(qkv_a.p, dg1.p, dbeta.p, heads, dalog.as<float>(),
                    dtb.as<float>(), state_a.as<float>(), out_a.p, tokens,
                    heads, dim, dim, lower_bound, scale, s);

  // Run B: 256 + 128 + 127 + 1 (decode tail).
  const int chunks[] = {256, 128, 127, 1};
  int offset = 0;
  for (int len : chunks) {
    const uint16_t* pre = dpre.as<uint16_t>() + int64_t(offset) * qkv_stride;
    const uint16_t* g = dg1.as<uint16_t>() + int64_t(offset) * heads * dim;
    const uint16_t* b = dbeta.as<uint16_t>() + int64_t(offset) * heads;
    uint16_t* qkv_out = qkv_b.as<uint16_t>() + int64_t(offset) * qkv_stride;
    uint16_t* o = out_b.as<uint16_t>() + int64_t(offset) * heads * dim;
    kda_causal_conv_silu_bf16(pre, qkv_stride, dcw.p, conv_b.p, state_width,
                              qkv_out, len, channels, conv_width, s);
    kda_recurrent_fwd(qkv_out, g, b, heads, dalog.as<float>(),
                      dtb.as<float>(), state_b.as<float>(), o, len, heads, dim,
                      dim, lower_bound, scale, s);
    offset += len;
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  if (offset != tokens) throw std::runtime_error("chunk cover");

  std::vector<uint16_t> host_a(out_a.bytes / 2), host_b(out_b.bytes / 2);
  std::vector<float> st_a(n_state), st_b(n_state);
  std::vector<uint16_t> cv_a(conv_a.bytes / 2), cv_b(conv_b.bytes / 2);
  out_a.download(host_a.data(), out_a.bytes);
  out_b.download(host_b.data(), out_b.bytes);
  state_a.download(st_a.data(), state_a.bytes);
  state_b.download(st_b.data(), state_b.bytes);
  conv_a.download(cv_a.data(), conv_a.bytes);
  conv_b.download(cv_b.data(), conv_b.bytes);

  // Chunk boundaries round-trip state through memory exactly (fp32 state,
  // bf16 conv inputs), so equality here is bitwise, not tolerance.
  require_bitwise("chunked outputs", host_a.data(), host_b.data(), out_a.bytes);
  require_bitwise("chunked recurrent state", st_a.data(), st_b.data(),
                  state_a.bytes);
  require_bitwise("chunked conv state", cv_a.data(), cv_b.data(), conv_a.bytes);
}

// ---------------------------------------------------------------------------
// Full layer parity vs host reference
// ---------------------------------------------------------------------------

namespace {

void run_layer_vs_host_reference(const KdaConfig& cfg, int tokens, uint64_t seed,
                                 double out_rel_tol, double state_rel_tol,
                                 const std::string& tag) {
  cudaStream_t s = test_stream();
  const KdaGeometry g = KdaGeometry::from_config(cfg);
  TestWeights tw = TestWeights::random(cfg, seed);
  DeviceWeights dw(tw);
  LayerEnv env(cfg, tokens, s);
  KdaLayer layer(env.arena, env.gemm, dw.views(), cfg, tokens, env.ws.p,
                 env.ws.bytes);
  if (!layer.prepare(tokens)) throw std::runtime_error("gemm plans unavailable");

  std::vector<uint16_t> hidden_in =
      random_bf16_bits(seed ^ 0x900, int64_t(tokens) * cfg.hidden, -2, 0);
  DevBuf din(hidden_in.size() * 2), dout(int64_t(tokens) * cfg.hidden * 2);
  din.upload(hidden_in.data(), hidden_in.size() * 2);

  DevBuf dstate(g.recurrent_elems * 4);
  DevBuf dconv(int64_t(g.conv_channels) * g.conv_state_width * 2);
  DGPP_CUDA_OK(cudaMemsetAsync(dstate.p, 0, dstate.bytes, s));
  DGPP_CUDA_OK(cudaMemsetAsync(dconv.p, 0, dconv.bytes, s));

  layer.enqueue(din.p, dstate.as<float>(), dconv.as<uint16_t>(),
                g.conv_state_width, dout.p, tokens, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  const int64_t n_out = int64_t(tokens) * cfg.hidden;
  const int64_t n_core = int64_t(tokens) * g.local_proj;
  std::vector<uint16_t> got_out(n_out), got_core(n_core);
  std::vector<float> got_state(g.recurrent_elems);
  dout.download(got_out.data(), got_out.size() * 2);
  DGPP_CUDA_OK(cudaMemcpy(got_core.data(), layer.debug_core(), n_core * 2,
                          cudaMemcpyDeviceToHost));
  dstate.download(got_state.data(), dstate.bytes);

  // Host reference, both precision classes. Each pass starts from the same
  // zero conv/recurrent state the device run started from — layer_forward
  // rolls the conv state in place, so it must be fresh per pass (a rolled
  // state poisons the first conv_width-1 tokens' outputs even though its
  // effect on the final recurrent state decays away).
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1 && cfg.hidden > 512) continue;  // fp64 oracle: small only
    std::vector<uint16_t> zero_conv(
        int64_t(g.conv_channels) * g.conv_state_width, 0);
    std::vector<uint16_t> ref_out(n_out), ref_core(n_core);
    // State criterion: the device (fp32, butterfly reductions, device
    // libm) and the host reference (fp32/fp64, sequential reductions,
    // glibc) share bit-identical bf16 inputs, so their difference is pure
    // fp32 arithmetic noise — but cancellation-dominated state elements
    // (small sums of O(1) rank-1 terms) legitimately show amplified
    // RELATIVE error. Judge by drift relative to the state's own scale and
    // the L2 norm; structural bugs produce O(scale) errors everywhere.
    auto require_state = [&](const std::string& what, const auto& ref_state) {
      const Stats ss = compare_abs_rel(got_state.data(), ref_state.data(),
                                       g.recurrent_elems, state_rel_tol, 1e-4);
      const double scale = std::abs(*std::max_element(
          got_state.begin(), got_state.end(),
          [](float x, float y) { return std::abs(x) < std::abs(y); }));
      const double drift = ss.max_abs / std::max(scale, 1e-30);
      if (drift > 5e-3 || ss.l2_rel > state_rel_tol)
        fail(what + " (drift-to-scale=" + std::to_string(drift) + ")", ss);
    };
    if (pass == 0) {
      std::vector<float> ref_state(g.recurrent_elems);
      dgpp::kda_ref::layer_forward<float>(
          host_views(tw), cfg, hidden_in.data(), ref_state.data(),
          zero_conv.data(), g.conv_state_width, ref_out.data(), ref_core.data(),
          tokens);
      require_state(tag + " [f32] state", ref_state);
    } else {
      std::vector<double> ref_state(g.recurrent_elems);
      dgpp::kda_ref::layer_forward<double>(
          host_views(tw), cfg, hidden_in.data(), ref_state.data(),
          zero_conv.data(), g.conv_state_width, ref_out.data(), ref_core.data(),
          tokens);
      require_state(tag + " [f64] state", ref_state);
    }
    const Stats so = compare_bf16(got_out, ref_out, 2);
    require_bf16(tag + " layer_out", so, out_rel_tol, 0.02);
    const Stats sc = compare_bf16(got_core, ref_core, 2);
    require_bf16(tag + " core_out", sc, out_rel_tol, 0.02);
  }
}

}  // namespace

DGPP_TEST(kda_full_layer_matches_host_reference) {
  // Small geometry: exhaustive comparison including the fp64 oracle. The
  // declared tolerance covers bf16 GEMM reduction-order differences plus
  // fp32 recurrence drift; anything structural (layout, offsets, missing
  // silu, wrong gate) blows far past it.
  KdaConfig small;
  small.hidden = 64;
  small.heads = 2;
  small.head_dim = 32;
  small.num_kda_layers = 1;
  small.spec_width = 0;
  run_layer_vs_host_reference(small, 8, 5150, 5e-3, 1e-3, "layer small");

  // Real geometry: production shapes at a token count the naive host
  // reference can still afford (~0.5 GFLOP).
  run_layer_vs_host_reference(KdaConfig{}, 4, 5151, 5e-3, 1e-3, "layer real");
}

// ---------------------------------------------------------------------------
// Full layer: chunked prefill vs unchunked (device vs device)
// ---------------------------------------------------------------------------

DGPP_TEST(kda_full_layer_chunked_prefill_matches_unchunked) {
  cudaStream_t s = test_stream();
  KdaConfig cfg;  // real geometry, TP=1
  const KdaGeometry g = KdaGeometry::from_config(cfg);
  const int tokens = 2048;  // the DESIGN prefill chunk size

  TestWeights tw = TestWeights::random(cfg, 6000);
  DeviceWeights dw(tw);
  LayerEnv env(cfg, tokens, s);
  KdaLayer layer(env.arena, env.gemm, dw.views(), cfg, tokens, env.ws.p,
                 env.ws.bytes);
  if (!layer.prepare(tokens) || !layer.prepare(1024) || !layer.prepare(2047) ||
      !layer.prepare(1))
    throw std::runtime_error("gemm plans unavailable");

  std::vector<uint16_t> hidden_in =
      random_bf16_bits(6001, int64_t(tokens) * cfg.hidden, -2, 0);
  DevBuf din(hidden_in.size() * 2), dout(int64_t(tokens) * cfg.hidden * 2),
      dout2(dout.bytes), dout3(dout.bytes);
  din.upload(hidden_in.data(), hidden_in.size() * 2);

  dgpp::Arena arena;
  dgpp::Arena::Config ac;
  ac.persistent_hot = 3 * g.pool_bytes;  // three slots
  arena.init(ac);
  KdaStatePool pool;
  pool.init(arena, cfg, 3);
  pool.zero_all(s);

  // Unchunked.
  layer.enqueue(din.p, pool.recurrent(0, 0), pool.conv(0, 0),
                g.conv_state_width, dout.p, tokens, s);
  // Chunked 1024 + 1024.
  layer.enqueue(din.p, pool.recurrent(1, 0), pool.conv(1, 0),
                g.conv_state_width, dout2.p, 1024, s);
  layer.enqueue(din.as<uint16_t>() + int64_t(1024) * cfg.hidden,
                pool.recurrent(1, 0), pool.conv(1, 0), g.conv_state_width,
                dout2.as<uint16_t>() + int64_t(1024) * cfg.hidden, 1024, s);
  // Chunked 2047 + 1 (decode-style tail).
  layer.enqueue(din.p, pool.recurrent(2, 0), pool.conv(2, 0),
                g.conv_state_width, dout3.p, 2047, s);
  layer.enqueue(din.as<uint16_t>() + int64_t(2047) * cfg.hidden,
                pool.recurrent(2, 0), pool.conv(2, 0), g.conv_state_width,
                dout3.as<uint16_t>() + int64_t(2047) * cfg.hidden, 1, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  const int64_t n_out = int64_t(tokens) * cfg.hidden;
  std::vector<uint16_t> out_a(n_out), out_b(n_out), out_c(n_out);
  dout.download(out_a.data(), dout.bytes);
  dout2.download(out_b.data(), dout2.bytes);
  dout3.download(out_c.data(), dout3.bytes);
  // Layer-0 recurrent states (the whole-slot export API is exercised by the
  // state-pool test; here we only need this layer's state).
  std::vector<float> st_a(g.recurrent_elems), st_b(g.recurrent_elems),
      st_c(g.recurrent_elems);
  DGPP_CUDA_OK(cudaMemcpy(st_a.data(), pool.recurrent(0, 0),
                          g.recurrent_bytes, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(st_b.data(), pool.recurrent(1, 0),
                          g.recurrent_bytes, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(st_c.data(), pool.recurrent(2, 0),
                          g.recurrent_bytes, cudaMemcpyDeviceToHost));

  // GEMM accumulation order differs across chunk sizes (different cuBLASLt
  // algorithms), so the bf16 inputs to the recurrence differ by ~1 ulp and
  // cancellation-heavy state elements can move much more in relative terms.
  // The meaningful bounds: error relative to the state's own scale, and the
  // L2 drift. The recurrence itself round-trips fp32 state exactly (see the
  // bitwise test).
  const double scale =
      *std::max_element(st_a.begin(), st_a.end(),
                        [](float x, float y) { return std::abs(x) < std::abs(y); });
  const Stats so2 = compare_bf16(out_a, out_b, 2);
  require_bf16("layer chunked 1024+1024 out", so2, 1e-2, 0.02);
  const Stats so3 = compare_bf16(out_a, out_c, 2);
  require_bf16("layer chunked 2047+1 out", so3, 1e-2, 0.02);
  const Stats ss2 = compare_abs_rel(st_a.data(), st_b.data(),
                                    g.recurrent_elems, 1e-3, 1e-4);
  const Stats ss3 = compare_abs_rel(st_a.data(), st_c.data(),
                                    g.recurrent_elems, 1e-3, 1e-4);
  const double drift2 = ss2.max_abs / std::max(scale, 1e-30);
  const double drift3 = ss3.max_abs / std::max(scale, 1e-30);
  if (drift2 > 5e-3 || ss2.l2_rel > 1e-3 || drift3 > 5e-3 || ss3.l2_rel > 1e-3) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "chunked state drift: 1024+1024 abs=%.3g (rel-to-scale %.3g) "
                  "l2=%.3g; 2047+1 abs=%.3g (%.3g) l2=%.3g",
                  ss2.max_abs, drift2, ss2.l2_rel, ss3.max_abs, drift3,
                  ss3.l2_rel);
    throw std::runtime_error(buf);
  }
  std::printf(
      "[INFO] chunked state drift (scale %.3f): 1024+1024 abs=%.3g l2=%.3g; "
      "2047+1 abs=%.3g l2=%.3g\n",
      scale, ss2.max_abs, ss2.l2_rel, ss3.max_abs, ss3.l2_rel);
}

// ---------------------------------------------------------------------------
// Decode step: graph capture/replay vs eager (allocation-free hot path)
// ---------------------------------------------------------------------------

DGPP_TEST(kda_layer_decode_graph_replay_matches_eager) {
  cudaStream_t s = test_stream();
  KdaConfig cfg;
  cfg.hidden = 256;
  cfg.heads = 4;
  cfg.head_dim = 32;
  cfg.num_kda_layers = 1;
  cfg.spec_width = 0;
  const KdaGeometry g = KdaGeometry::from_config(cfg);

  TestWeights tw = TestWeights::random(cfg, 7000);
  DeviceWeights dw(tw);
  LayerEnv env(cfg, 1, s);
  KdaLayer layer(env.arena, env.gemm, dw.views(), cfg, 1, env.ws.p,
                 env.ws.bytes);
  if (!layer.prepare(1)) throw std::runtime_error("gemm plans unavailable");

  dgpp::Arena arena;
  dgpp::Arena::Config ac;
  ac.persistent_hot = 2 * g.pool_bytes;
  arena.init(ac);
  KdaStatePool pool;
  pool.init(arena, cfg, 2);

  dgpp::GraphCache graphs;
  DevBuf din(cfg.hidden * 2), dout(cfg.hidden * 2), dout_ref(cfg.hidden * 2);
  std::vector<uint16_t> in_a = random_bf16_bits(7100, cfg.hidden, -2, 0);
  std::vector<uint16_t> in_b = random_bf16_bits(7101, cfg.hidden, -2, 0);

  for (int variant = 0; variant < 2; ++variant) {
    const std::vector<uint16_t>& in = variant == 0 ? in_a : in_b;
    din.upload(in.data(), in.size() * 2);

    // Eager run from a clean slot.
    pool.zero_slot(0, s);
    layer.enqueue(din.p, pool.recurrent(0, 0), pool.conv(0, 0),
                  g.conv_state_width, dout_ref.p, 1, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));

    // Graph run from a clean slot: capture once, replay for both variants.
    pool.zero_slot(1, s);
    graphs.replay_or_capture(
        1, "kda-decode", s, [&](cudaStream_t cap) {
          layer.enqueue(din.p, pool.recurrent(1, 0), pool.conv(1, 0),
                        g.conv_state_width, dout.p, 1, cap);
        });
    DGPP_CUDA_OK(cudaStreamSynchronize(s));

    std::vector<uint16_t> eager(cfg.hidden), graphed(cfg.hidden);
    dout_ref.download(eager.data(), eager.size() * 2);
    dout.download(graphed.data(), graphed.size() * 2);
    require_bitwise("graph replay output", eager.data(), graphed.data(),
                    eager.size() * 2);
    std::vector<float> st_e(g.recurrent_elems), st_g(g.recurrent_elems);
    DGPP_CUDA_OK(cudaMemcpy(st_e.data(), pool.recurrent(0, 0),
                            g.recurrent_bytes, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(st_g.data(), pool.recurrent(1, 0),
                            g.recurrent_bytes, cudaMemcpyDeviceToHost));
    require_bitwise("graph replay state", st_e.data(), st_g.data(),
                    st_e.size() * 4);
  }
  if (graphs.hits() < 1) throw std::runtime_error("graph was never replayed");
}

// ---------------------------------------------------------------------------
// Head-slice (TP-readiness): a rank-sized layer must reproduce the full
// layer's core outputs for its heads
// ---------------------------------------------------------------------------

DGPP_TEST(kda_head_slice_matches_full_run_core_outputs) {
  cudaStream_t s = test_stream();
  KdaConfig full;  // real geometry, 64 heads
  const KdaGeometry gf = KdaGeometry::from_config(full);
  const int tokens = 4, slice_heads = 16;

  TestWeights tw = TestWeights::random(full, 8000);
  DeviceWeights dw(tw);
  LayerEnv env(full, tokens, s);
  KdaLayer layer_full(env.arena, env.gemm, dw.views(), full, tokens, env.ws.p,
                      env.ws.bytes);
  if (!layer_full.prepare(tokens)) throw std::runtime_error("plans");

  std::vector<uint16_t> hidden_in =
      random_bf16_bits(8001, int64_t(tokens) * full.hidden, -2, 0);
  DevBuf din(hidden_in.size() * 2), dout(int64_t(tokens) * full.hidden * 2);
  din.upload(hidden_in.data(), hidden_in.size() * 2);

  DevBuf dstate(gf.recurrent_elems * 4),
      dconv(int64_t(gf.conv_channels) * gf.conv_state_width * 2);
  DGPP_CUDA_OK(cudaMemsetAsync(dstate.p, 0, dstate.bytes, s));
  DGPP_CUDA_OK(cudaMemsetAsync(dconv.p, 0, dconv.bytes, s));
  layer_full.enqueue(din.p, dstate.as<float>(), dconv.as<uint16_t>(),
                     gf.conv_state_width, dout.p, tokens, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  std::vector<uint16_t> core_full(int64_t(tokens) * gf.local_proj);
  DGPP_CUDA_OK(cudaMemcpy(core_full.data(), layer_full.debug_core(),
                          core_full.size() * 2, cudaMemcpyDeviceToHost));

  // Build the rank-slice weights on the host: heads [0, 16) of q/k/v/b,
  // f_a/g_a replicated in full, sliced f_b/g_b/conv/a_log/dt_bias, sliced
  // o_proj columns, full o_norm — exactly the TP=4 shard rules.
  KdaConfig sliced = full;
  sliced.heads = slice_heads;  // heads [0, 16) of the full head set
  sliced.tp_size = 1;
  const KdaGeometry gs = KdaGeometry::from_config(sliced);
  const int lp_f = gf.local_proj, lp_s = gs.local_proj;

  TestWeights sw;
  {
    // Fused layout is [f_a | g_a | q | k | v | b] (see kda_layer.hpp).
    const int off_q = 2 * full.head_dim;
    const int off_b = off_q + 3 * lp_f;
    auto rows = [&](const std::vector<uint16_t>& src, int64_t row_off,
                    int64_t count, int64_t width) {
      std::vector<uint16_t> out(static_cast<size_t>(count * width));
      for (int64_t r = 0; r < count; ++r)
        std::copy_n(src.begin() + (row_off + r) * width, width,
                    out.begin() + r * width);
      return out;
    };
    auto append = [&](std::vector<uint16_t>& dst,
                      const std::vector<uint16_t>& part) {
      dst.insert(dst.end(), part.begin(), part.end());
    };
    // f_a/g_a are replicated: full rows on every rank.
    sw.in_proj = rows(tw.in_proj, 0, full.head_dim, full.hidden);
    append(sw.in_proj, rows(tw.in_proj, full.head_dim, full.head_dim, full.hidden));
    append(sw.in_proj, rows(tw.in_proj, off_q, lp_s, full.hidden));              // q
    append(sw.in_proj, rows(tw.in_proj, off_q + lp_f, lp_s, full.hidden));       // k
    append(sw.in_proj, rows(tw.in_proj, off_q + 2 * lp_f, lp_s, full.hidden));   // v
    append(sw.in_proj, rows(tw.in_proj, off_b, slice_heads, full.hidden));
    if (int64_t(sw.in_proj.size()) != int64_t(gs.in_proj_cols) * full.hidden)
      throw std::runtime_error("slice in_proj size");

    sw.f_b = rows(tw.f_b, 0, lp_s, full.head_dim);
    sw.g_b = rows(tw.g_b, 0, lp_s, full.head_dim);
    // The merged conv weight is [q(all heads) | k | v]: a rank's slice is
    // per-section, NOT a contiguous prefix. Slicing conv[0:6144] would feed
    // heads 16-47's q-weights as this rank's k/v — exactly the class of bug
    // this test exists to catch for the M5 shard loader.
    sw.conv = rows(tw.conv, 0, lp_s, full.conv_width);
    append(sw.conv, rows(tw.conv, lp_f, lp_s, full.conv_width));
    append(sw.conv, rows(tw.conv, 2 * lp_f, lp_s, full.conv_width));
    sw.o_norm = tw.o_norm;
    // o_proj [hidden, lp_f]: keep columns [0, lp_s).
    sw.o_proj.resize(static_cast<size_t>(full.hidden) * lp_s);
    for (int r = 0; r < full.hidden; ++r)
      std::copy_n(tw.o_proj.begin() + int64_t(r) * lp_f, lp_s,
                  sw.o_proj.begin() + int64_t(r) * lp_s);
    sw.a_log.assign(tw.a_log.begin(), tw.a_log.begin() + slice_heads);
    sw.dt_bias.assign(tw.dt_bias.begin(), tw.dt_bias.begin() + lp_s);
  }

  DeviceWeights dsw(sw);
  LayerEnv env_s(sliced, tokens, s);
  KdaLayer layer_slice(env_s.arena, env_s.gemm, dsw.views(), sliced, tokens,
                       env_s.ws.p, env_s.ws.bytes);
  if (!layer_slice.prepare(tokens)) throw std::runtime_error("slice plans");

  DevBuf dstate_s(gs.recurrent_elems * 4),
      dconv_s(int64_t(gs.conv_channels) * gs.conv_state_width * 2),
      dout_s(int64_t(tokens) * sliced.hidden * 2);
  DGPP_CUDA_OK(cudaMemsetAsync(dstate_s.p, 0, dstate_s.bytes, s));
  DGPP_CUDA_OK(cudaMemsetAsync(dconv_s.p, 0, dconv_s.bytes, s));
  layer_slice.enqueue(din.p, dstate_s.as<float>(), dconv_s.as<uint16_t>(),
                      gs.conv_state_width, dout_s.p, tokens, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  std::vector<uint16_t> core_slice(int64_t(tokens) * lp_s);
  DGPP_CUDA_OK(cudaMemcpy(core_slice.data(), layer_slice.debug_core(),
                          core_slice.size() * 2, cudaMemcpyDeviceToHost));

  // Compare the first slice_heads heads of every token. GEMM algorithm
  // changes with N can shift a few bf16 outputs by an ulp; the tolerance is
  // the same class as the chunked test.
  const int64_t rows = tokens * slice_heads;
  std::vector<uint16_t> got(rows * full.head_dim), want(rows * full.head_dim);
  for (int t = 0; t < tokens; ++t)
    for (int h = 0; h < slice_heads; ++h) {
      std::copy_n(
          core_full.begin() + (int64_t(t) * full.heads + h) * full.head_dim,
          full.head_dim,
          want.begin() + (int64_t(t) * slice_heads + h) * full.head_dim);
      std::copy_n(
          core_slice.begin() + (int64_t(t) * slice_heads + h) * full.head_dim,
          full.head_dim,
          got.begin() + (int64_t(t) * slice_heads + h) * full.head_dim);
    }
  const Stats st = compare_bf16(got, want, 2);
  require_bf16("head slice core outputs", st, 1e-2, 0.02);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  const int rc = dgpp::test::run_all();
  if (rc != 0) return rc;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dump-file") == 0 && i + 1 < argc) {
      const int drc = run_kda_dump_parity(argv[i + 1]);
      if (drc != 0) return drc;
    }
  }
  return 0;
}
