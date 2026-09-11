// The Gated DeltaNet recurrence: the KDA kernel's
// scalar-gate mode against the host oracle (gdn_ref::recurrent), the
// chunk invariance (one call vs many, bitwise: the state round-trips fp32),
// the request-batched form against per-request plain runs (bitwise), and
// the speculative snapshots (the state after row t equals the state of a
// t+1-token run, bitwise).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/kda.hpp"
#include "models/qwen/gdn_reference.hpp"

using namespace dgpp::kda_test;

namespace {
using dgpp::bf16_bits_to_float;

struct Problem {
  int heads = 12, kv_ratio = 3, k = 128, v = 128, tokens = 5;
  int heads_k() const { return heads / kv_ratio; }
  int64_t qkv_row() const { return static_cast<int64_t>(2) * heads_k() * k + static_cast<int64_t>(heads) * v; }
  int64_t state_elems() const { return static_cast<int64_t>(heads) * v * k; }
  std::vector<uint16_t> qkv, a_raw, beta_raw;
  std::vector<float> a_log, dt_bias;
  float scale = 0.f;
  static Problem make(int tokens, uint64_t seed, int heads = 12, int kv_ratio = 3) {
    Problem p;
    p.tokens = tokens;
    p.heads = heads;
    p.kv_ratio = kv_ratio;
    p.qkv = random_bf16_normal(seed + 1, p.qkv_row() * tokens, 0.5f);
    p.a_raw = random_bf16_uniform(seed + 2, static_cast<int64_t>(tokens) * heads, 2.0f);
    p.beta_raw = random_bf16_uniform(seed + 3, static_cast<int64_t>(tokens) * heads, 2.0f);
    p.a_log = random_f32_uniform(seed + 4, heads, 1.0f);
    p.dt_bias = random_f32_uniform(seed + 5, heads, 1.0f);
    p.scale = ref_scale(p.k);
    return p;
  }
};

struct DeviceRun {
  std::vector<uint16_t> out;
  std::vector<float> state;
};

// Runs the device recurrence over `p` starting from `state0`, in `chunks`
// equal pieces; returns the outputs and the final state.
DeviceRun run_device(const Problem& p, const std::vector<float>& state0, int chunks,
                     std::vector<float>* snapshots = nullptr) {
  cudaStream_t s = test_stream();
  DevBuf qkv(p.qkv.size() * 2), a(p.a_raw.size() * 2), b(p.beta_raw.size() * 2);
  DevBuf alog(p.a_log.size() * 4), dtb(p.dt_bias.size() * 4);
  DevBuf state(state0.size() * 4), out(static_cast<size_t>(p.tokens) * p.heads * p.v * 2);
  qkv.upload(p.qkv.data(), p.qkv.size() * 2);
  a.upload(p.a_raw.data(), p.a_raw.size() * 2);
  b.upload(p.beta_raw.data(), p.beta_raw.size() * 2);
  alog.upload(p.a_log.data(), p.a_log.size() * 4);
  dtb.upload(p.dt_bias.data(), p.dt_bias.size() * 4);
  state.upload(state0.data(), state0.size() * 4);
  DevBuf snap(snapshots ? static_cast<size_t>(p.tokens) * p.state_elems() * 4 : 16);
  const int per = p.tokens / chunks;
  for (int c = 0; c < chunks; ++c) {
    const int t0 = c * per;
    const int n = c + 1 == chunks ? p.tokens - t0 : per;
    dgpp::KdaStateSnapshots ks;
    if (snapshots && chunks == 1) {
      ks.states = snap.as<float>();
      ks.stride_elems = p.state_elems();
    }
    dgpp::gdn_recurrent_fwd(qkv.as<uint16_t>() + t0 * p.qkv_row(), a.as<uint16_t>() + t0 * p.heads,
                            p.heads, b.as<uint16_t>() + t0 * p.heads, p.heads, alog.as<float>(),
                            dtb.as<float>(), state.as<float>(),
                            out.as<uint16_t>() + static_cast<int64_t>(t0) * p.heads * p.v, n,
                            p.heads, p.kv_ratio, p.k, p.v, p.scale, s, ks);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  DeviceRun r;
  r.out.resize(static_cast<size_t>(p.tokens) * p.heads * p.v);
  r.state.resize(state0.size());
  out.download(r.out.data(), r.out.size() * 2);
  state.download(r.state.data(), r.state.size() * 4);
  if (snapshots) {
    snapshots->resize(static_cast<size_t>(p.tokens) * p.state_elems());
    snap.download(snapshots->data(), snapshots->size() * 4);
  }
  return r;
}

template <typename Acc>
DeviceRun run_host(const Problem& p, const std::vector<float>& state0) {
  std::vector<Acc> st(state0.begin(), state0.end());
  DeviceRun r;
  r.out.resize(static_cast<size_t>(p.tokens) * p.heads * p.v);
  dgpp::gdn_ref::recurrent<Acc>(p.qkv.data(), p.a_raw.data(), p.heads, p.beta_raw.data(), p.heads,
                                p.a_log.data(), p.dt_bias.data(), st.data(), r.out.data(), p.tokens,
                                p.heads, p.kv_ratio, p.k, p.v, p.scale);
  r.state.assign(st.begin(), st.end());
  return r;
}

std::vector<float> random_state(const Problem& p, uint64_t seed) {
  return random_f32_uniform(seed, p.state_elems(), 0.3f);
}

}  // namespace

DGPP_TEST(gdn_recurrent_matches_the_host_oracle) {
  for (const int tokens : {1, 5, 64}) {
    const Problem p = Problem::make(tokens, 100 + tokens);
    const std::vector<float> s0 = random_state(p, 7);
    const DeviceRun dev = run_device(p, s0, 1);
    const DeviceRun ref = run_host<float>(p, s0);
    const DeviceRun ref64 = run_host<double>(p, s0);
    // Outputs: bf16 rows, the 16-lane split reorders the fp32 reductions.
    require_bf16("gdn out vs fp32 oracle (T=" + std::to_string(tokens) + ")",
                 compare_bf16(dev.out, ref.out, 2), 2e-3, 0.02);
    require_bf16("gdn out vs fp64 oracle (T=" + std::to_string(tokens) + ")",
                 compare_bf16(dev.out, ref64.out, 2), 2e-3, 0.02);
    const Stats ss = compare_abs_rel(dev.state.data(), ref.state.data(),
                                     static_cast<long>(dev.state.size()), 1e-4, 1e-6);
    require_rel("gdn state vs fp32 oracle (T=" + std::to_string(tokens) + ")", ss, 1e-3, 1e-3);
  }
}

DGPP_TEST(gdn_recurrent_is_chunk_invariant_bitwise) {
  const Problem p = Problem::make(64, 200);
  const std::vector<float> s0 = random_state(p, 8);
  const DeviceRun one = run_device(p, s0, 1);
  const DeviceRun eight = run_device(p, s0, 8);
  const DeviceRun sixtyfour = run_device(p, s0, 64);
  require_bitwise("chunked outputs (8)", one.out.data(), eight.out.data(), one.out.size() * 2);
  require_bitwise("chunked state (8)", one.state.data(), eight.state.data(), one.state.size() * 4);
  require_bitwise("chunked outputs (64)", one.out.data(), sixtyfour.out.data(), one.out.size() * 2);
  require_bitwise("chunked state (64)", one.state.data(), sixtyfour.state.data(), one.state.size() * 4);
}

DGPP_TEST(gdn_recurrent_snapshots_are_the_prefix_states_bitwise) {
  const Problem p = Problem::make(4, 300);
  const std::vector<float> s0 = random_state(p, 9);
  std::vector<float> snapshots;
  const DeviceRun full = run_device(p, s0, 1, &snapshots);
  for (int t = 0; t + 1 < p.tokens; ++t) {
    Problem prefix = p;
    prefix.tokens = t + 1;
    prefix.qkv.resize(static_cast<size_t>(prefix.tokens) * p.qkv_row());
    prefix.a_raw.resize(static_cast<size_t>(prefix.tokens) * p.heads);
    prefix.beta_raw.resize(static_cast<size_t>(prefix.tokens) * p.heads);
    const DeviceRun pre = run_device(prefix, s0, 1);
    require_bitwise("snapshot after row " + std::to_string(t),
                    snapshots.data() + static_cast<size_t>(t) * p.state_elems(), pre.state.data(),
                    static_cast<size_t>(p.state_elems()) * 4);
  }
  (void)full;
}

DGPP_TEST(gdn_recurrent_batched_equals_per_request_runs_bitwise) {
  // Two requests in one launch: slot 2 gets rows [0, 3), a padding row, slot
  // 0 gets rows [4, 6); each must equal its own plain run.
  const Problem a = Problem::make(3, 400);
  Problem b = Problem::make(2, 500);
  b.a_log = a.a_log;  // the layer's parameters are shared by every request
  b.dt_bias = a.dt_bias;
  const std::vector<float> sa = random_state(a, 10), sb = random_state(b, 11);
  const DeviceRun ra = run_device(a, sa, 1), rb = run_device(b, sb, 1);
  const int rows = 6;
  Problem batch = Problem::make(rows, 600);
  // rows: a0 a1 a2 pad b0 b1
  auto put = [&](const Problem& src, int src_t, int dst_t) {
    std::memcpy(batch.qkv.data() + dst_t * batch.qkv_row(), src.qkv.data() + src_t * src.qkv_row(), batch.qkv_row() * 2);
    std::memcpy(batch.a_raw.data() + dst_t * batch.heads, src.a_raw.data() + src_t * src.heads, batch.heads * 2);
    std::memcpy(batch.beta_raw.data() + dst_t * batch.heads, src.beta_raw.data() + src_t * src.heads, batch.heads * 2);
  };
  for (int t = 0; t < 3; ++t) put(a, t, t);
  for (int t = 0; t < 2; ++t) put(b, t, 4 + t);
  batch.a_log = a.a_log;
  batch.dt_bias = a.dt_bias;
  // The slots: 3 request slots of state; b in slot 0, a in slot 2.
  const int64_t se = batch.state_elems();
  std::vector<float> slots(static_cast<size_t>(3 * se), 0.f);
  std::copy(sb.begin(), sb.end(), slots.begin());
  std::copy(sa.begin(), sa.end(), slots.begin() + 2 * se);
  std::vector<int32_t> ids = {2, 2, 2, -1, 0, 0};
  std::vector<int64_t> pos = {0, 1, 2, -1, 0, 1};
  std::vector<int32_t> spans = {0, 3, 3, 1, 4, 2};  // (start, len) x 3 spans, the middle all padding
  cudaStream_t s = test_stream();
  DevBuf qkv(batch.qkv.size() * 2), ar(batch.a_raw.size() * 2), br(batch.beta_raw.size() * 2);
  DevBuf alog(batch.a_log.size() * 4), dtb(batch.dt_bias.size() * 4), st(slots.size() * 4);
  DevBuf out(static_cast<size_t>(rows) * batch.heads * batch.v * 2);
  DevBuf dids(ids.size() * 4), dpos(pos.size() * 8), dspans(spans.size() * 4);
  qkv.upload(batch.qkv.data(), batch.qkv.size() * 2);
  ar.upload(batch.a_raw.data(), batch.a_raw.size() * 2);
  br.upload(batch.beta_raw.data(), batch.beta_raw.size() * 2);
  alog.upload(batch.a_log.data(), batch.a_log.size() * 4);
  dtb.upload(batch.dt_bias.data(), batch.dt_bias.size() * 4);
  st.upload(slots.data(), slots.size() * 4);
  dids.upload(ids.data(), ids.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  dspans.upload(spans.data(), spans.size() * 4);
  dgpp::KdaRequestRows req;
  req.request_ids = dids.as<int32_t>();
  req.positions = dpos.as<int64_t>();
  req.spans = dspans.as<int32_t>();
  req.num_requests = 3;
  dgpp::gdn_recurrent_fwd_batched(qkv.as<uint16_t>(), ar.as<uint16_t>(), batch.heads, br.as<uint16_t>(),
                                  batch.heads, alog.as<float>(), dtb.as<float>(), st.as<float>(), se,
                                  out.as<uint16_t>(), rows, batch.heads, batch.kv_ratio, batch.k, batch.v,
                                  batch.scale, req, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> got(static_cast<size_t>(rows) * batch.heads * batch.v);
  std::vector<float> got_slots(slots.size());
  out.download(got.data(), got.size() * 2);
  st.download(got_slots.data(), got_slots.size() * 4);
  const size_t row_elems = static_cast<size_t>(batch.heads) * batch.v;
  require_bitwise("batched a outputs", got.data(), ra.out.data(), 3 * row_elems * 2);
  require_bitwise("batched b outputs", got.data() + 4 * row_elems, rb.out.data(), 2 * row_elems * 2);
  for (size_t i = 0; i < row_elems; ++i)
    if (got[3 * row_elems + i] != 0) throw std::runtime_error("padding row not zero");
  require_bitwise("batched a state (slot 2)", got_slots.data() + 2 * se, ra.state.data(), se * 4);
  require_bitwise("batched b state (slot 0)", got_slots.data(), rb.state.data(), se * 4);
  for (int64_t i = 0; i < se; ++i)
    if (got_slots[static_cast<size_t>(se + i)] != 0.f) throw std::runtime_error("untouched slot changed");
}

int main() { return dgpp::test::run_all(); }
