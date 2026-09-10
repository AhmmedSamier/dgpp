// The gated residual (Q3, 2026-09-09): every device stage against its
// exact host math on the device's own inputs (the elementwise kernels
// bitwise; the GEMV within an ulp of the sequential chain), and the whole
// site — mix then combine — within two bf16 ulps of the reference.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/qwen_gr.hpp"
#include "kernels/qwen_norm.hpp"
#include "models/qwen/gr_reference.hpp"
#include "models/qwen/norm_reference.hpp"

using namespace dgpp::kda_test;

namespace {

struct Site {
  int rows = 3, hc = 4, hidden = 256, rank = 32;
  int width() const { return hc * hidden; }
  std::vector<uint16_t> r, w_norm, w_down, w_up, w_inj, y;
  static Site make(uint64_t seed) {
    Site s;
    s.r = random_bf16_normal(seed + 1, static_cast<int64_t>(s.rows) * s.width(), 1.0f);
    s.w_norm = random_bf16_uniform(seed + 2, s.width(), 0.3f);
    s.w_down = random_bf16_normal(seed + 3, static_cast<int64_t>(s.rank) * s.width(), 0.05f);
    s.w_up = random_bf16_normal(seed + 4, static_cast<int64_t>(s.width()) * s.rank, 0.2f);
    s.w_inj = random_bf16_normal(seed + 5, static_cast<int64_t>(s.hc) * s.width(), 0.05f);
    s.y = random_bf16_normal(seed + 6, static_cast<int64_t>(s.rows) * s.hidden, 0.5f);
    return s;
  }
};

std::vector<uint16_t> download(const DevBuf& b, size_t n) {
  std::vector<uint16_t> v(n);
  b.download(v.data(), n * 2);
  return v;
}

}  // namespace

DGPP_TEST(qwen_gr_stages_match_the_reference_on_the_device_inputs) {
  const Site s = Site::make(10);
  cudaStream_t st = test_stream();
  const int W = s.width();
  DevBuf dr(s.r.size() * 2), dnorm(s.w_norm.size() * 2), ddown(s.w_down.size() * 2),
      dup(s.w_up.size() * 2), dinj(s.w_inj.size() * 2), dy(s.y.size() * 2);
  DevBuf drn(s.r.size() * 2), dt(static_cast<size_t>(s.rows) * s.rank * 2),
      dlog(s.r.size() * 2), dx(static_cast<size_t>(s.rows) * s.hidden * 2);
  dr.upload(s.r.data(), s.r.size() * 2);
  dnorm.upload(s.w_norm.data(), s.w_norm.size() * 2);
  ddown.upload(s.w_down.data(), s.w_down.size() * 2);
  dup.upload(s.w_up.data(), s.w_up.size() * 2);
  dinj.upload(s.w_inj.data(), s.w_inj.size() * 2);
  dy.upload(s.y.data(), s.y.size() * 2);
  // 1. the group norm
  dgpp::qwen_group_rmsnorm_bf16(dr.p, dnorm.p, drn.p, s.rows, s.hc, s.hidden, 1e-6f, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> rn = download(drn, s.r.size());
  {
    std::vector<uint16_t> want(s.r.size());
    dgpp::qwen_ref::group_rmsnorm(s.r.data(), s.w_norm.data(), want.data(), s.rows, s.hc, s.hidden, 1e-6f);
    require_bf16("group norm", compare_bf16(rn, want, 1), 1e-3, 0.01);
  }
  // 2. the down GEMV on the device's Rn
  if (!dgpp::bf16_gemv_accepts(ddown.p, s.rows, W)) throw std::runtime_error("gemv shape refused");
  dgpp::launch_bf16_gemv(drn.as<uint16_t>(), W, ddown.as<uint16_t>(), dt.p, false, s.rows, s.rank, W, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> t_dev = download(dt, static_cast<size_t>(s.rows) * s.rank);
  {
    std::vector<uint16_t> want(t_dev.size());
    dgpp::qwen_ref::gemv_bf16(rn.data(), s.w_down.data(), want.data(), s.rows, s.rank, W);
    require_bf16("down gemv", compare_bf16(t_dev, want, 1), 1e-3, 0.02);
  }
  // 3. the activation (bitwise on the device's t)
  std::vector<uint16_t> t_ref = t_dev;
  dgpp::qwen_ref::gr_gate_act(t_ref.data(), s.rows, s.rank, s.hc);
  dgpp::qwen_gr_gate_act_bf16(dt.p, s.rows, s.rank, s.hc, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> t_act = download(dt, t_dev.size());
  require_bitwise("gate act", t_act.data(), t_ref.data(), t_act.size() * 2);
  // 4. the up GEMV
  dgpp::launch_bf16_gemv(dt.as<uint16_t>(), s.rank, dup.as<uint16_t>(), dlog.p, false, s.rows, W, s.rank, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> logits = download(dlog, s.r.size());
  {
    std::vector<uint16_t> want(logits.size());
    dgpp::qwen_ref::gemv_bf16(t_act.data(), s.w_up.data(), want.data(), s.rows, W, s.rank);
    require_bf16("up gemv", compare_bf16(logits, want, 1), 1e-3, 0.02);
  }
  // 4b. the fused decode GEMVs (2026-09-09): the norm staged into the down
  // GEMV and the activation into the up GEMV — bitwise the chain above.
  {
    DevBuf drn2(s.r.size() * 2), dt2(static_cast<size_t>(s.rows) * s.rank * 2), dlog2(s.r.size() * 2);
    // The inject dots folded into the same launch (2026-09-10) against the
    // standalone dots kernel on the same Rn: bitwise, or the gate that
    // decides every combine has moved.
    DevBuf dg_fold(static_cast<size_t>(s.rows) * s.hc * 4), dg_ref(static_cast<size_t>(s.rows) * s.hc * 4);
    if (!dgpp::qwen_gr_fused_mix_accepts(s.hc, s.hidden, s.rank)) throw std::runtime_error("fused mix refuses the fixture");
    dgpp::qwen_gr_norm_down_bf16(dr.p, static_cast<size_t>(W), dnorm.p, s.hc, s.hidden, 1e-6f, drn2.p,
                                 ddown.p, dt2.p, s.rank, s.rows, st, dinj.p,
                                 static_cast<float*>(dg_fold.p));
    dgpp::qwen_gr_combine_dots_bf16(drn.p, dinj.p, static_cast<float*>(dg_ref.p), s.rows, s.hc,
                                    s.hidden, st);
    dgpp::qwen_gr_act_up_bf16(dt2.p, s.rank, s.hc, dup.p, dlog2.p, s.hidden, s.rows, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> rn2 = download(drn2, s.r.size());
    const std::vector<uint16_t> t2 = download(dt2, t_dev.size());
    const std::vector<uint16_t> logits2 = download(dlog2, s.r.size());
    require_bitwise("fused norm_down: Rn", rn2.data(), rn.data(), rn2.size() * 2);
    require_bitwise("fused norm_down: t", t2.data(), t_dev.data(), t2.size() * 2);
    require_bitwise("fused act_up: logits", logits2.data(), logits.data(), logits2.size() * 2);
    std::vector<float> g_fold(static_cast<size_t>(s.rows) * s.hc), g_ref(g_fold.size());
    DGPP_CUDA_OK(cudaMemcpy(g_fold.data(), dg_fold.p, g_fold.size() * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(g_ref.data(), dg_ref.p, g_ref.size() * 4, cudaMemcpyDeviceToHost));
    require_bitwise("fused norm_down: inject gates", g_fold.data(), g_ref.data(), g_fold.size() * 4);
    std::printf("[ OK ] the fused decode GEMVs are bitwise the norm/GEMV/act/GEMV chain at %d rows, "
                "and the folded inject gates the dots kernel's\n", s.rows);
  }
  // 5. the finish (bitwise on the device's logits and Rn)
  dgpp::qwen_gr_mix_finish_bf16(dlog.p, drn.p, dx.p, s.rows, s.hc, s.hidden, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> x = download(dx, static_cast<size_t>(s.rows) * s.hidden);
  {
    std::vector<uint16_t> want(x.size());
    dgpp::qwen_ref::gr_mix_finish(logits.data(), rn.data(), want.data(), s.rows, s.hc, s.hidden);
    require_bitwise("mix finish", x.data(), want.data(), x.size() * 2);
  }
  // 6. the combine: the inject dots are the kernel's own (a warp chain), so
  // s may differ from the host's sequential dot by an ulp — hold R' to one.
  DevBuf dgates(static_cast<size_t>(s.rows) * s.hc * 4);
  dgpp::qwen_gr_combine_bf16(dr.p, drn.p, dinj.p, dy.p, static_cast<float*>(dgates.p), s.rows,
                             s.hc, s.hidden, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> r2 = download(dr, s.r.size());
  {
    std::vector<uint16_t> want = s.r;
    dgpp::qwen_ref::gr_combine(want.data(), rn.data(), s.w_inj.data(), s.y.data(), s.rows, s.hc, s.hidden);
    require_bf16("combine", compare_bf16(r2, want, 1), 1e-3, 0.01);
  }
  // 6b. the batched rows' down GEMV with the inject rows appended
  // (2026-09-10): t bitwise the chain's GEMV, the gates bitwise the dots
  // kernel's, at the fixture's rows.
  {
    DevBuf dt3(static_cast<size_t>(s.rows) * s.rank * 2), dg3(static_cast<size_t>(s.rows) * s.hc * 4);
    dgpp::qwen_gr_down_inject_bf16(drn.p, ddown.p, dt3.p, s.rank, dinj.p,
                                   static_cast<float*>(dg3.p), s.hc, s.hidden, s.rows, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> t3 = download(dt3, t_dev.size());
    require_bitwise("down_inject: t", t3.data(), t_dev.data(), t3.size() * 2);
    std::vector<float> g3(static_cast<size_t>(s.rows) * s.hc), g_ref(g3.size());
    DGPP_CUDA_OK(cudaMemcpy(g3.data(), dg3.p, g3.size() * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(g_ref.data(), dgates.p, g_ref.size() * 4, cudaMemcpyDeviceToHost));
    require_bitwise("down_inject: gates", g3.data(), g_ref.data(), g3.size() * 4);
    std::printf("[ OK ] the batched down GEMV with the inject rows is bitwise the chain at %d rows\n", s.rows);
  }
  // 7. end to end: the device's x against the host's whole read from R.
  {
    std::vector<uint16_t> rn_ref(s.r.size()), x_ref(x.size());
    dgpp::qwen_ref::gr_mix(s.r.data(), s.w_norm.data(), s.w_down.data(), s.w_up.data(), rn_ref.data(),
                           x_ref.data(), s.rows, s.hc, s.hidden, s.rank, 1e-6f);
    require_bf16("mix end to end", compare_bf16(x, x_ref, 2), 2e-3, 0.02);
  }
}

int main() { return dgpp::test::run_all(); }
