// Mamba-2 device kernels (src/kernels/mamba2.cu) against the host fp32
// oracle (src/models/nemotron_transcribe/reference.cpp) — milestone 3 of
// docs/nemotron_transcribe_plan.md. Both sides consume the SAME bf16 bits
// (random_bf16_bits), the oracle in fp32/double and the kernels in fp32
// with -ffp-contract=off, so the only gaps are the kernel's fp32 drift
// (bf16 ulp budgets on outputs, rel budgets on the F32 state) and nothing
// else. The stateful composition — the conv step chain and the scan
// resume/step path — must reproduce the fused prefill BITWISE: same bits
// in, same fixed-order arithmetic, no drift budget (a rollback that
// disagrees with the fused walk is a serving bug, not a rounding story).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/mamba2.hpp"
#include "models/nemotron_transcribe/reference.hpp"
#include "models/nemotron_transcribe/state.hpp"

using dgpp::kda_test::compare_abs_rel;
using dgpp::kda_test::compare_bf16;
using dgpp::kda_test::DevBuf;
using dgpp::kda_test::random_bf16_bits;
using dgpp::kda_test::require_bf16;
using dgpp::kda_test::require_bitwise;
using dgpp::kda_test::require_rel;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

template <typename T>
DevBuf upload(const std::vector<T>& v) {
  DevBuf b(v.size() * sizeof(T));
  b.upload(v.data(), v.size() * sizeof(T));
  return b;
}
template <typename T>
std::vector<T> download(const DevBuf& b, size_t n) {
  std::vector<T> v(n);
  b.download(v.data(), n * sizeof(T));
  return v;
}
template <typename T>
std::vector<T> download_ptr(const void* p, size_t n) {
  std::vector<T> v(n);
  DGPP_CUDA_OK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
void dev_sync() { DGPP_CUDA_OK(cudaDeviceSynchronize()); }

float fbf(uint16_t u) { return dgpp::bf16_bits_to_float(u); }
std::vector<uint16_t> to_bf16(const std::vector<float>& v) {
  std::vector<uint16_t> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = dgpp::float_to_bf16_bits(v[i]);
  return out;
}
// The oracle consumes dequantized activations: identical inputs both sides.
std::vector<float> dequant(const std::vector<uint16_t>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = fbf(v[i]);
  return out;
}
// Deterministic F32 scan parameters (A_log / D / dt_bias are F32 in the
// checkpoint; the scan reads them as floats, so no bf16 pin is needed).
std::vector<float> rand_f32(uint64_t seed, size_t n, float lo, float hi) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    const uint64_t x = seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull;
    const uint64_t h = (x ^ (x >> 30)) * 0x94D049BB133111EBull;
    const uint32_t u = static_cast<uint32_t>((h ^ (h >> 32)) >> 16);
    v[i] = lo + (hi - lo) * (static_cast<double>(u) / 65535.0);
  }
  return v;
}

struct ScanCase {
  int heads, hd, groups, ss, seq;
};

// Run the oracle walk on dequantized inputs; returns (out fp32, state fp32).
void oracle_scan(const ScanCase& c, const std::vector<uint16_t>& xb, const std::vector<uint16_t>& Bb,
                 const std::vector<uint16_t>& Cb, const std::vector<uint16_t>& dtb,
                 const std::vector<float>& a_log, const std::vector<float>& D,
                 const std::vector<float>& dt_bias, float dt_min, std::vector<float>& out,
                 std::vector<float>& state) {
  dgpp::nemotron_ref::MambaGeom g;
  g.heads = c.heads;
  g.head_dim = c.hd;
  g.groups = c.groups;
  g.state = c.ss;
  g.dt_min = dt_min;
  out.assign(c.seq * c.heads * c.hd, 0.f);
  state.assign(static_cast<size_t>(c.heads) * c.hd * c.ss, 0.f);
  auto x = dequant(xb), B = dequant(Bb), C = dequant(Cb), dt = dequant(dtb);
  dgpp::nemotron_ref::mamba2_walk(g, x.data(), B.data(), C.data(), dt.data(), a_log.data(),
                                  D.data(), dt_bias.data(), c.seq, out.data(), state.data());
}

}  // namespace

DGPP_TEST(mamba2_conv_prefill_matches_the_oracle) {
  struct C { int seq, ch, kern; uint64_t seed; };
  for (C c : {C{7, 24, 4, 0x2A1}, C{200, 128, 4, 0x2A2}, C{33, 6144, 4, 0x2A3}}) {
    const auto u = random_bf16_bits(c.seed, static_cast<int64_t>(c.seq) * c.ch, -4, 2);
    const auto w = random_bf16_bits(c.seed ^ 0x55, static_cast<int64_t>(c.ch) * c.kern, -3, 1);
    const auto b = random_bf16_bits(c.seed ^ 0x99, c.ch, -2, 1);
    dgpp::nemotron_ref::Tensor wt, bt;
    wt = dequant(w);
    bt = dequant(b);
    std::vector<float> ref(static_cast<size_t>(c.seq) * c.ch);
    dgpp::nemotron_ref::causal_conv_silu(dequant(u).data(), c.seq, c.ch, wt, &bt, c.kern,
                                         ref.data());
    const auto du = upload(u), dw = upload(w), db = upload(b);
    DevBuf dout(static_cast<size_t>(c.seq) * c.ch * 2);
    dgpp::mamba2_conv_silu(du.p, c.ch, dw.p, db.p, dout.p, c.ch, c.seq, c.ch, c.kern, 0);
    dev_sync();
    const auto got = download<uint16_t>(dout, static_cast<size_t>(c.seq) * c.ch);
    require_bf16("conv seq=" + std::to_string(c.seq) + " ch=" + std::to_string(c.ch),
                 compare_bf16(got, to_bf16(ref), 2), 1e-3, 0.0);
  }
}

DGPP_TEST(mamba2_conv_step_chain_is_the_prefill_bitwise) {
  const int seq = 37, ch = 96, kern = 4, hist = kern - 1;
  const uint64_t seed = 0xC07;
  const auto u = random_bf16_bits(seed, static_cast<int64_t>(seq) * ch, -4, 2);
  const auto w = random_bf16_bits(seed ^ 0x55, static_cast<int64_t>(ch) * kern, -3, 1);
  const auto b = random_bf16_bits(seed ^ 0x99, ch, -2, 1);
  // Fused prefill from zero history.
  DevBuf du(upload(u)), dw(upload(w)), db(upload(b));
  DevBuf dpref(static_cast<size_t>(seq) * ch * 2);
  dgpp::mamba2_conv_silu(du.p, ch, dw.p, db.p, dpref.p, ch, seq, ch, kern, 0);
  dev_sync();
  const auto prefill_bits = download<uint16_t>(dpref, static_cast<size_t>(seq) * ch);
  // The decode path: a zeroed state advanced one row at a time.
  DevBuf state(static_cast<size_t>(ch) * hist * 2);
  DGPP_CUDA_OK(cudaMemset(state.p, 0, state.bytes));
  DevBuf drow(ch * 2), dout(ch * 2);
  std::vector<uint16_t> got(static_cast<size_t>(seq) * ch);
  for (int t = 0; t < seq; ++t) {
    drow.upload(&u[static_cast<size_t>(t) * ch], ch * 2);
    dgpp::mamba2_conv_silu_step(drow.p, state.p, dw.p, db.p, dout.p, ch, kern, 0);
    dev_sync();
    auto row = download<uint16_t>(dout, ch);
    std::memcpy(&got[static_cast<size_t>(t) * ch], row.data(), ch * 2);
  }
  require_bitwise("conv step chain == prefill", got.data(), prefill_bits.data(),
                  got.size() * 2);
  // The state holds the last K-1 committed rows, oldest first.
  auto st = download<uint16_t>(state, static_cast<size_t>(ch) * hist);
  for (int j = 0; j < hist; ++j) {
    const int t = seq - hist + j;
    for (int c = 0; c < ch; ++c) require(st[c * hist + j] == u[t * ch + c], "conv state row");
  }
}

DGPP_TEST(mamba2_scan_matches_the_oracle) {
  const float dt_min = 0.003f;
  for (ScanCase c : {ScanCase{4, 8, 2, 32, 33}, ScanCase{8, 64, 8, 128, 96},
                     ScanCase{2, 16, 1, 128, 5}, ScanCase{64, 64, 8, 128, 17}}) {
    const uint64_t seed = 0x5CA7 ^ c.heads;
    const int d_inner = c.heads * c.hd;
    const auto x = random_bf16_bits(seed, static_cast<int64_t>(c.seq) * d_inner, -3, 2);
    const auto B = random_bf16_bits(seed ^ 1, static_cast<int64_t>(c.seq) * c.groups * c.ss, -3, 2);
    const auto C = random_bf16_bits(seed ^ 2, static_cast<int64_t>(c.seq) * c.groups * c.ss, -3, 2);
    // dt spans below the clamp (raw + bias -> softplus < dt_min) and above.
    const auto dt = random_bf16_bits(seed ^ 3, static_cast<int64_t>(c.seq) * c.heads, -6, 2);
    const auto a_log = rand_f32(seed ^ 4, c.heads, -2.f, 2.f);
    const auto D = rand_f32(seed ^ 5, c.heads, -1.f, 1.f);
    const auto dtb = rand_f32(seed ^ 6, c.heads, -2.f, 2.f);
    std::vector<float> ref_out, ref_state;
    oracle_scan(c, x, B, C, dt, a_log, D, dtb, dt_min, ref_out, ref_state);

    const auto dx = upload(x), dB = upload(B), dC = upload(C), ddt = upload(dt);
    const auto da = upload(a_log), dD = upload(D), ddb = upload(dtb);
    DevBuf dstate(static_cast<size_t>(c.heads) * c.hd * c.ss * 4);
    DGPP_CUDA_OK(cudaMemset(dstate.p, 0, dstate.bytes));
    DevBuf dout(static_cast<size_t>(c.seq) * d_inner * 2);
    dgpp::mamba2_scan(dx.p, d_inner, dB.p, c.groups * c.ss, dC.p, c.groups * c.ss, ddt.p, c.heads,
                      da.as<float>(), dD.as<float>(), ddb.as<float>(), dstate.as<float>(), dout.p,
                      d_inner, c.seq, c.heads, c.hd, c.groups, c.ss, dt_min, 0);
    dev_sync();
    const std::string tag = "scan h=" + std::to_string(c.heads) + " hd=" + std::to_string(c.hd) +
                            " ss=" + std::to_string(c.ss) + " seq=" + std::to_string(c.seq);
    require_bf16(tag + " out", compare_bf16(download<uint16_t>(dout, ref_out.size()),
                                             to_bf16(ref_out), 4), 2e-2, 1e-3);
    // The F32 state: the kernel's fp32 chain vs the oracle's double chain.
    const auto got_state = download<float>(dstate, ref_state.size());
    require_rel(tag + " state", compare_abs_rel(got_state.data(), ref_state.data(),
                                                static_cast<long>(ref_state.size()), 1e-3, 1e-6),
                1e-3, 0.0);
  }
}

DGPP_TEST(mamba2_scan_resume_and_step_are_the_prefill_bitwise) {
  const ScanCase c{4, 64, 2, 128, 41};
  const int split = 17;
  const float dt_min = 0.001f;
  const uint64_t seed = 0x9E51;
  const int d_inner = c.heads * c.hd;
  const auto x = random_bf16_bits(seed, static_cast<int64_t>(c.seq) * d_inner, -3, 2);
  const auto B = random_bf16_bits(seed ^ 1, static_cast<int64_t>(c.seq) * c.groups * c.ss, -3, 2);
  const auto C = random_bf16_bits(seed ^ 2, static_cast<int64_t>(c.seq) * c.groups * c.ss, -3, 2);
  const auto dt = random_bf16_bits(seed ^ 3, static_cast<int64_t>(c.seq) * c.heads, -6, 2);
  const auto a_log = rand_f32(seed ^ 4, c.heads, -2.f, 2.f);
  const auto D = rand_f32(seed ^ 5, c.heads, -1.f, 1.f);
  const auto dtb = rand_f32(seed ^ 6, c.heads, -2.f, 2.f);

  const auto dx = upload(x), dB = upload(B), dC = upload(C), ddt = upload(dt);
  const auto da = upload(a_log), dD = upload(D), ddb = upload(dtb);
  DevBuf dref(static_cast<size_t>(c.seq) * d_inner * 2);
  DevBuf dst0(static_cast<size_t>(c.heads) * c.hd * c.ss * 4);
  DevBuf d1(static_cast<size_t>(c.seq) * d_inner * 2);
  DevBuf dst1(static_cast<size_t>(c.heads) * c.hd * c.ss * 4);
  DGPP_CUDA_OK(cudaMemset(dst0.p, 0, dst0.bytes));
  DGPP_CUDA_OK(cudaMemset(dst1.p, 0, dst1.bytes));
  dgpp::mamba2_scan(dx.p, d_inner, dB.p, c.groups * c.ss, dC.p, c.groups * c.ss, ddt.p, c.heads,
                    da.as<float>(), dD.as<float>(), ddb.as<float>(), dst0.as<float>(), dref.p,
                    d_inner, c.seq, c.heads, c.hd, c.groups, c.ss, dt_min, 0);
  // Segment 0..split, then split..end continuing on the carried state.
  dgpp::mamba2_scan(dx.p, d_inner, dB.p, c.groups * c.ss, dC.p, c.groups * c.ss, ddt.p, c.heads,
                    da.as<float>(), dD.as<float>(), ddb.as<float>(), dst1.as<float>(), d1.p,
                    d_inner, split, c.heads, c.hd, c.groups, c.ss, dt_min, 0);
  const size_t row = static_cast<size_t>(split) * d_inner * 2;
  dgpp::mamba2_scan(static_cast<const uint16_t*>(dx.p) + split * d_inner, d_inner,
                    static_cast<const uint16_t*>(dB.p) + split * c.groups * c.ss, c.groups * c.ss,
                    static_cast<const uint16_t*>(dC.p) + split * c.groups * c.ss, c.groups * c.ss,
                    static_cast<const uint16_t*>(ddt.p) + split * c.heads, c.heads, da.as<float>(),
                    dD.as<float>(), ddb.as<float>(), dst1.as<float>(),
                    static_cast<uint16_t*>(d1.p) + split * d_inner, d_inner, c.seq - split,
                    c.heads, c.hd, c.groups, c.ss, dt_min, 0);
  // And the row-by-row decode path from the split's state.
  DevBuf d2(static_cast<size_t>(c.seq) * d_inner * 2);
  DevBuf dst2(static_cast<size_t>(c.heads) * c.hd * c.ss * 4);
  DGPP_CUDA_OK(cudaMemset(dst2.p, 0, dst2.bytes));
  dgpp::mamba2_scan(dx.p, d_inner, dB.p, c.groups * c.ss, dC.p, c.groups * c.ss, ddt.p, c.heads,
                    da.as<float>(), dD.as<float>(), ddb.as<float>(), dst2.as<float>(), d2.p,
                    d_inner, split, c.heads, c.hd, c.groups, c.ss, dt_min, 0);
  for (int t = split; t < c.seq; ++t) {
    dgpp::mamba2_scan_step(static_cast<const uint16_t*>(dx.p) + t * d_inner,
                           static_cast<const uint16_t*>(dB.p) + t * c.groups * c.ss,
                           static_cast<const uint16_t*>(dC.p) + t * c.groups * c.ss,
                           static_cast<const uint16_t*>(ddt.p) + t * c.heads, da.as<float>(),
                           dD.as<float>(), ddb.as<float>(), dst2.as<float>(),
                           static_cast<uint16_t*>(d2.p) + t * d_inner, c.heads, c.hd, c.groups,
                           c.ss, dt_min, 0);
  }
  dev_sync();
  // require_bitwise is a host memcmp: download every device buffer first.
  const auto ref_bits = download<uint16_t>(dref, static_cast<size_t>(c.seq) * d_inner);
  const auto r1 = download<uint16_t>(d1, static_cast<size_t>(c.seq) * d_inner);
  const auto r2 = download<uint16_t>(d2, static_cast<size_t>(c.seq) * d_inner);
  const auto s0 = download<float>(dst0, static_cast<size_t>(c.heads) * c.hd * c.ss);
  const auto s1 = download<float>(dst1, static_cast<size_t>(c.heads) * c.hd * c.ss);
  const auto s2 = download<float>(dst2, static_cast<size_t>(c.heads) * c.hd * c.ss);
  require_bitwise("scan resume == prefill", r1.data(), ref_bits.data(),
                  static_cast<size_t>(c.seq) * d_inner * 2);
  require_bitwise("scan step == prefill", r2.data(), ref_bits.data(),
                  static_cast<size_t>(c.seq) * d_inner * 2);
  require_bitwise("scan resume state", s1.data(), s0.data(), s1.size() * 4);
  require_bitwise("scan step state", s2.data(), s0.data(), s2.size() * 4);
}

DGPP_TEST(mamba2_gated_norm_matches_the_oracle) {
  struct G { int seq, d_inner, group; uint64_t seed; };
  const float eps = 1e-5f;
  for (G g : {G{3, 32, 32, 0x6E1}, G{17, 512, 512, 0x6E2}, G{5, 4096, 512, 0x6E3}}) {
    const auto y = random_bf16_bits(g.seed, static_cast<int64_t>(g.seq) * g.d_inner, -3, 2);
    const auto gate = random_bf16_bits(g.seed ^ 1, static_cast<int64_t>(g.seq) * g.d_inner, -3, 2);
    const auto w = random_bf16_bits(g.seed ^ 2, g.d_inner, -1, 1);
    dgpp::nemotron_ref::Tensor wt;
    wt = dequant(w);
    auto yf = dequant(y), gf = dequant(gate);
    std::vector<float> ref(static_cast<size_t>(g.seq) * g.d_inner, 0.f);
    for (int t = 0; t < g.seq; ++t)
      dgpp::nemotron_ref::rmsnorm_gated(yf.data() + static_cast<size_t>(t) * g.d_inner,
                                        gf.data() + static_cast<size_t>(t) * g.d_inner, wt,
                                        g.d_inner, g.group, eps,
                                        ref.data() + static_cast<size_t>(t) * g.d_inner);
    const auto dy = upload(y), dg(upload(gate)), dw(upload(w));
    DevBuf dout(static_cast<size_t>(g.seq) * g.d_inner * 2);
    dgpp::mamba2_gated_norm(dy.p, g.d_inner, dg.p, g.d_inner, dw.p, dout.p, g.d_inner, g.seq,
                            g.d_inner, g.group, eps, 0);
    dev_sync();
    require_bf16("gated norm d=" + std::to_string(g.d_inner) + " grp=" + std::to_string(g.group),
                 compare_bf16(download<uint16_t>(dout, ref.size()), to_bf16(ref), 2), 1e-3, 0.0);
  }
}

DGPP_TEST(mamba2_state_pool_snapshot_and_zero) {
  dgpp::NemotronMambaGeometry g;
  g.num_mamba_slots = 3;
  g.heads = 4;
  g.head_dim = 8;
  g.groups = 2;
  g.state_size = 128;
  g.conv_dim = 48;
  g.conv_kernel = 4;
  dgpp::Arena arena;
  arena.init({static_cast<size_t>(g.request_bytes()) * 4, 0, 0, 0, 0});
  dgpp::NemotronMambaStatePool pool;
  pool.init(arena, g, 4);
  require(pool.num_requests() == 4, "requests");
  // Scribble request 0's slots, snapshot into request 1, zero request 0.
  const size_t rec_n = static_cast<size_t>(g.heads) * g.head_dim * g.state_size;
  std::vector<float> pat(rec_n);
  for (size_t i = 0; i < pat.size(); ++i) pat[i] = 0.5f + static_cast<float>(i);
  DGPP_CUDA_OK(cudaMemcpy(pool.recurrent(0, 2), pat.data(), pat.size() * 4, cudaMemcpyHostToDevice));
  std::vector<uint16_t> cpat(g.conv_dim * (g.conv_kernel - 1), 0x3F80);
  DGPP_CUDA_OK(cudaMemcpy(pool.conv(0, 2), cpat.data(), cpat.size() * 2, cudaMemcpyHostToDevice));
  pool.copy_request(1, 0, 0);
  pool.zero_request(0, 0);
  dev_sync();
  auto got = download_ptr<float>(pool.recurrent(1, 2), rec_n);
  require(got == pat, "snapshot copied the recurrent state");
  auto gotc = download_ptr<uint16_t>(pool.conv(1, 2), cpat.size());
  require(gotc == cpat, "snapshot copied the conv state");
  auto zeroed = download_ptr<float>(pool.recurrent(0, 2), rec_n);
  for (float f : zeroed) require(f == 0.f, "zero_request cleared the slot");
  bool any = false;
  for (auto u : download_ptr<uint16_t>(pool.conv(0, 2), cpat.size())) any = any || u != 0;
  require(!any, "zero_request cleared the conv history");
  // Out-of-range is a loud failure, not a silent overrun.
  bool threw = false;
  try {
    (void)pool.recurrent(4, 0);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  require(threw, "request bounds");
  threw = false;
  try {
    (void)pool.conv(0, 3);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  require(threw, "slot bounds");
}

int main() { return dgpp::test::run_all(); }
