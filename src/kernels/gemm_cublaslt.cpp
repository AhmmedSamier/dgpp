#include "kernels/gemm.hpp"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <format>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

#include "common/cuda_check.hpp"
#include "kernels/bf16_gemv.hpp"

namespace dgpp {

namespace {
constexpr size_t kRecommendedWorkspace = 64ull << 20;

// Full plan identity: shapes, dtypes, and activation leading dimension. The
// old packed uint64 key had no room for the row stride, and a hash-fold risks
// silently aliasing two plans — determinism beats cleverness here.
struct PlanKey {
  int m, n, k;
  DType io;
  GemmOut od;
  size_t act_row_stride;

  bool operator<(const PlanKey& o) const {
    return std::tie(m, n, k, io, od, act_row_stride) <
           std::tie(o.m, o.n, o.k, o.io, o.od, o.act_row_stride);
  }
};
}  // namespace

struct CublasLtGemm::Impl {
  cublasLtHandle_t lt{};
  float* dev_unit_scale{};  // fp8 tensor-wise scale == 1.0f

  struct Plan {
    cublasLtMatmulDesc_t desc{};
    cublasLtMatrixLayout_t la{}, lb{}, ld{};
    cublasLtMatmulAlgo_t algo{};
    size_t ws_bytes = 0;
  };
  std::map<PlanKey, Plan> plans;

  Impl() {
    DGPP_CUBLAS_OK(cublasLtCreate(&lt), "create");
    DGPP_CUDA_OK(cudaMalloc(&dev_unit_scale, sizeof(float)));
    const float one = 1.0f;
    DGPP_CUDA_OK(cudaMemcpy(dev_unit_scale, &one, sizeof(float),
                            cudaMemcpyHostToDevice));
  }

  ~Impl() {
    for (auto& [key, p] : plans) {
      if (p.desc) cublasLtMatmulDescDestroy(p.desc);
      if (p.la) cublasLtMatrixLayoutDestroy(p.la);
      if (p.lb) cublasLtMatrixLayoutDestroy(p.lb);
      if (p.ld) cublasLtMatrixLayoutDestroy(p.ld);
    }
    if (dev_unit_scale) cudaFree(dev_unit_scale);
    if (lt) cublasLtDestroy(lt);
  }

  Plan& get_plan(int m, int n, int k, DType io, GemmOut od,
                 size_t act_row_stride, void* /*ws*/, size_t ws_bytes) {
    PlanKey key{m, n, k, io, od, act_row_stride};
    auto it = plans.find(key);
    if (it != plans.end()) return it->second;

    bool fp8 = io == DType::F8_E4M3;
    bool out_f32 = od == GemmOut::F32;
    cublasComputeType_t comp = CUBLAS_COMPUTE_32F;
    cudaDataType_t ab_type = fp8 ? CUDA_R_8F_E4M3 : CUDA_R_16BF;

    // Convention (see benchmarks/micro/gemm_peak.cu): out row-major [M,N] is
    // issued as column-major D(N,M) = op_T(W cm(K,N)) x Act cm(K,M). Weight
    // arrives row-major [N,K] == col-major (K,N) ld=K. Activation rows of
    // stride S == col-major (K,M) ld=S; S==K reads a contiguous [M,K] tensor.
    Plan p{};
    DGPP_CUBLAS_OK(
        cublasLtMatmulDescCreate(&p.desc, comp, CUDA_R_32F), "desc create");
    cublasOperation_t ta = CUBLAS_OP_T;
    cublasOperation_t tb = CUBLAS_OP_N;
    DGPP_CUBLAS_OK(cublasLtMatmulDescSetAttribute(
                       p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)),
                   "set transa");
    DGPP_CUBLAS_OK(cublasLtMatmulDescSetAttribute(
                       p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)),
                   "set transb");
    if (fp8) {
      DGPP_CUBLAS_OK(
          cublasLtMatmulDescSetAttribute(p.desc,
                                         CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                         &dev_unit_scale, sizeof(void*)),
          "set a scale");
      DGPP_CUBLAS_OK(
          cublasLtMatmulDescSetAttribute(p.desc,
                                         CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                         &dev_unit_scale, sizeof(void*)),
          "set b scale");
    }

    DGPP_CUBLAS_OK(
        cublasLtMatrixLayoutCreate(&p.la, ab_type, k, n, k), "layout a");
    DGPP_CUBLAS_OK(cublasLtMatrixLayoutCreate(&p.lb, ab_type, k, m,
                                              act_row_stride),
                   "layout b");
    DGPP_CUBLAS_OK(cublasLtMatrixLayoutCreate(&p.ld, out_f32 ? CUDA_R_32F
                                                             : CUDA_R_16BF,
                                              n, m, n),
                   "layout d");

    cublasLtMatmulPreference_t pref{};
    DGPP_CUBLAS_OK(cublasLtMatmulPreferenceCreate(&pref), "pref create");
    DGPP_CUBLAS_OK(
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
            sizeof(ws_bytes)),
        "pref ws");

    cublasLtMatmulHeuristicResult_t heur{};
    int nres = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, p.desc, p.la, p.lb,
                                                       p.ld, p.ld, pref, 1,
                                                       &heur, &nres);
    cublasLtMatmulPreferenceDestroy(pref);
    if (st != CUBLAS_STATUS_SUCCESS || nres == 0)
      throw std::runtime_error(std::format(
          "cublasLt no heuristic for m={} n={} k={} dtype={}", m, n, k,
          dtype_name(io)));
    p.algo = heur.algo;
    p.ws_bytes = heur.workspaceSize;

    auto [ins, ok] = plans.emplace(key, p);
    return ins->second;
  }
};

CublasLtGemm::CublasLtGemm() : impl_(new Impl()) {}
CublasLtGemm::~CublasLtGemm() { delete impl_; }

void CublasLtGemm::matmul(const void* act, const void* weight, void* out,
                          int m, int n, int k, DType io_dtype, GemmOut out_dtype,
                          size_t act_row_stride, void* workspace,
                          size_t ws_bytes, cudaStream_t stream) {
  // Decode-shaped bf16 calls take the bandwidth GEMV (bf16_gemv.hpp):
  // cuBLASLt's m=1 kernel sits at ~128 GB/s on this part. Same
  // determinism contract (fixed order, no heuristic), different bits than
  // the Lt path — the layer oracles gate it.
  if (io_dtype == DType::BF16 && bf16_gemv_accepts(weight, m, k)) {
    launch_bf16_gemv(static_cast<const uint16_t*>(act), act_row_stride,
                     static_cast<const uint16_t*>(weight), out,
                     out_dtype == GemmOut::F32, m, n, k, stream);
    return;
  }
  Impl::Plan& p = impl_->get_plan(m, n, k, io_dtype, out_dtype,
                                  act_row_stride, workspace, ws_bytes);
  float alpha = 1.f, beta = 0.f;
  // Heuristic-selected algo + fixed layouts keep replays bitwise-stable in
  // process (graph-capture determinism requirement, DESIGN §11).
  DGPP_CUBLAS_OK(
      cublasLtMatmul(impl_->lt, p.desc, &alpha, weight, p.la, act, p.lb, &beta,
                     out, p.ld, out, p.ld, &p.algo, workspace, ws_bytes,
                     stream),
      std::format("matmul m={} n={} k={} lda={} dtype={}", m, n, k,
                  act_row_stride, dtype_name(io_dtype)));
}

size_t CublasLtGemm::query_workspace_bytes(int, int, int, DType) {
  return kRecommendedWorkspace;
}

bool CublasLtGemm::ensure_plan(int m, int n, int k, DType io_dtype,
                               GemmOut out_dtype, size_t act_row_stride) {
  // Callers must hand a workspace sized by query_workspace_bytes(); pass our
  // recommended cap via a scratchless probe — heuristic query alone does not
  // touch the workspace pointer.
  try {
    impl_->get_plan(m, n, k, io_dtype, out_dtype, act_row_stride, nullptr,
                    kRecommendedWorkspace);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace dgpp
