#include "kernels/gemm.hpp"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <map>
#include <mutex>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "common/cuda_check.hpp"
#include "kernels/bf12_gemv.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/mma_gemv.hpp"

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
  int batch = 1;
  int64_t act_batch_stride = 0, weight_batch_stride = 0, out_batch_stride = 0;
  bool bias = false;
  bool weight_kn = false;
  bool fp32_reductions = false;

  bool operator<(const PlanKey& o) const {
    return std::tie(m, n, k, io, od, act_row_stride, batch, act_batch_stride, weight_batch_stride,
                    out_batch_stride, bias, weight_kn, fp32_reductions) <
           std::tie(o.m, o.n, o.k, o.io, o.od, o.act_row_stride, o.batch, o.act_batch_stride,
                    o.weight_batch_stride, o.out_batch_stride, o.bias, o.weight_kn,
                    o.fp32_reductions);
  }
};
}  // namespace

struct CublasLtGemm::Impl {
  int decode_rows = kGemmDecodeRowsDefault;  // the decode lowering bound (set_decode_rows)
  bool decode_mma = false;                   // the lowering's tensor-core form (set_decode_mma)
  int decode_mma_max_rows = 0;               // its bound (0: every row count)
  int plan_rows = 0;                         // the Lt algorithm's row count (set_plan_rows)
  bool bf12_wide = false;                    // companions take 5..8-row calls too (set_bf12_wide)
  cublasLtHandle_t lt{};
  float* dev_unit_scale{};  // fp8 tensor-wise scale == 1.0f
  // bf16 weight -> its packed companion (register_bf12).
  std::unordered_map<const void*, Bf12Matrix> bf12;

  // The companion of an [n, k] weight, or null.
  const Bf12Matrix* bf12_for(const void* weight, int n, int k) const {
    if (bf12.empty()) return nullptr;
    const auto it = bf12.find(weight);
    if (it == bf12.end() || it->second.n != n || it->second.k != k ||
        !bf12_gemv_accepts(it->second, 1))
      return nullptr;
    return &it->second;
  }
  // The widest bf16 call the GEMV lowering takes: the decode rows — and,
  // for a companion while the caller says its rows are a decode batch
  // (set_bf12_wide), at least one packed launch's rows: bf12_gemv.hpp's
  // wide form reads 0.75 of the bytes once where an Lt algorithm reads them
  // all. A short PREFILL chunk of five to eight rows keeps its Lt algorithm
  // (and the transcripts that went through it).
  int gemv_rows(const Bf12Matrix* packed) const {
    return packed != nullptr && bf12_wide ? std::max(decode_rows, kBf12MaxRows) : decode_rows;
  }

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

  Plan& get_plan(int m, int n, int k, DType io, GemmOut od, size_t act_row_stride, void* /*ws*/,
                 size_t ws_bytes, int batch = 1, int64_t act_batch_stride = 0,
                 int64_t weight_batch_stride = 0, int64_t out_batch_stride = 0,
                 const uint16_t* bias = nullptr, bool weight_kn = false,
                 bool fp32_reductions = false) {
    PlanKey key{m,
                n,
                k,
                io,
                od,
                act_row_stride,
                batch,
                act_batch_stride,
                weight_batch_stride,
                out_batch_stride,
                bias != nullptr,
                weight_kn,
                fp32_reductions};
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
    cublasOperation_t ta = weight_kn ? CUBLAS_OP_N : CUBLAS_OP_T;
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
    if (bias) {
      const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
      DGPP_CUBLAS_OK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &epilogue, sizeof(epilogue)),
                     "bias epilogue");
      DGPP_CUBLAS_OK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &bias, sizeof(bias)),
                     "bias pointer");
    }

    DGPP_CUBLAS_OK(cublasLtMatrixLayoutCreate(&p.la, ab_type, weight_kn ? n : k, weight_kn ? k : n,
                                              weight_kn ? n : k),
                   "layout a");
    DGPP_CUBLAS_OK(cublasLtMatrixLayoutCreate(&p.lb, ab_type, k, m,
                                              act_row_stride),
                   "layout b");
    DGPP_CUBLAS_OK(cublasLtMatrixLayoutCreate(&p.ld, out_f32 ? CUDA_R_32F
                                                             : CUDA_R_16BF,
                                              n, m, n),
                   "layout d");

    if (batch > 1) {
      for (auto layout : {p.la, p.lb, p.ld})
        DGPP_CUBLAS_OK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                        &batch, sizeof(batch)),
                       "layout batch count");
      const auto stride = [](cublasLtMatrixLayout_t layout, int64_t value) {
        DGPP_CUBLAS_OK(
            cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &value, sizeof(value)),
            "layout batch stride");
      };
      stride(p.la, weight_batch_stride);
      stride(p.lb, act_batch_stride);
      stride(p.ld, out_batch_stride);
    }

    cublasLtMatmulPreference_t pref{};
    DGPP_CUBLAS_OK(cublasLtMatmulPreferenceCreate(&pref), "pref create");
    DGPP_CUBLAS_OK(cublasLtMatmulPreferenceSetAttribute(
                       pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)),
                   "pref ws");
    if (fp32_reductions) {
      const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
      DGPP_CUBLAS_OK(
          cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                               &reduction, sizeof(reduction)),
          "FP32 reductions");
    }

    cublasLtMatmulHeuristicResult_t heur{};
    int nres = 0;
    cublasStatus_t st =
        cublasLtMatmulAlgoGetHeuristic(lt, p.desc, p.la, p.lb, p.ld, p.ld, pref, 1, &heur, &nres);
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

int dense_gemv_rows() {
  static const int rows = [] {
    const char* v = std::getenv("DGPP_DENSE_GEMV_ROWS");
    if (!v || !*v) return 4;
    char* end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || x < 1 || x > 4096)
      throw std::invalid_argument("DGPP_DENSE_GEMV_ROWS: an integer in [1, 4096]");
    return static_cast<int>(x);
  }();
  return rows;
}

CublasLtGemm::CublasLtGemm() : impl_(new Impl()) {}
CublasLtGemm::~CublasLtGemm() { delete impl_; }

void CublasLtGemm::set_decode_rows(int rows) {
  if (rows < 1 || rows > kGemmDecodeLoweringRows)
    throw std::invalid_argument("CublasLtGemm::set_decode_rows: rows outside [1, " +
                                std::to_string(kGemmDecodeLoweringRows) + "]");
  impl_->decode_rows = rows;
}

int CublasLtGemm::decode_rows() const { return impl_->decode_rows; }
void CublasLtGemm::set_decode_mma(bool on, int max_rows) {
  if (max_rows < 0) throw std::invalid_argument("CublasLtGemm::set_decode_mma: negative bound");
  impl_->decode_mma = on;
  impl_->decode_mma_max_rows = max_rows;
}
bool CublasLtGemm::decode_mma() const { return impl_->decode_mma; }
int CublasLtGemm::decode_mma_max_rows() const { return impl_->decode_mma_max_rows; }

namespace {
// The rows the next GEMV chunk of a `left`-row call takes (at most four,
// fewer when k's staged rows would pass the smem bound).
int gemv_chunk_rows(const void* weight, int left, int k) {
  int rows = std::min(4, left);
  while (!bf16_gemv_accepts(weight, rows, k)) --rows;
  return rows;
}
}  // namespace

void CublasLtGemm::set_plan_rows(int rows) {
  if (rows < 0) throw std::invalid_argument("CublasLtGemm::set_plan_rows: negative rows");
  impl_->plan_rows = rows;
}

void CublasLtGemm::set_bf12_wide(bool on) { impl_->bf12_wide = on; }

void CublasLtGemm::register_bf12(const void* weight, const Bf12Matrix& packed) {
  if (weight == nullptr || !bf12_gemv_accepts(packed, 1))
    throw std::invalid_argument("CublasLtGemm::register_bf12: not a packed decode matrix");
  impl_->bf12[weight] = packed;
}
size_t CublasLtGemm::bf12_registered() const { return impl_->bf12.size(); }

void CublasLtGemm::resident_view(const void* weight, size_t bytes, int m, const void** ptr,
                                 size_t* view_bytes) const {
  *ptr = weight;
  *view_bytes = bytes;
  const auto it = impl_->bf12.find(weight);
  if (it == impl_->bf12.end()) return;
  const Bf12Matrix& p = it->second;
  // The dispatch rule of matmul: the GEMV lowering's rows.
  if (m < 1 || m > impl_->gemv_rows(&p) || impl_->decode_mma ||
      !bf16_gemv_accepts(weight, 1, p.k) || !bf12_gemv_accepts(p, 1))
    return;
  *ptr = p.packed;
  *view_bytes = bf12_packed_bytes(p.n, p.k);
}

void CublasLtGemm::matmul(const void* act, const void* weight, void* out,
                          int m, int n, int k, DType io_dtype, GemmOut out_dtype,
                          size_t act_row_stride, void* workspace,
                          size_t ws_bytes, cudaStream_t stream) {
  // Decode-shaped bf16 calls take the bandwidth GEMV (bf16_gemv.hpp):
  // cuBLASLt's m=1 kernel sits at ~128 GB/s on this part. Each GEMV row has
  // the scalar reduction order regardless of the rows sharing its launch.
  // The serving ceiling is the model's decode rows (set_decode_rows: its
  // fixed batch, 8 unless it says otherwise), while one launch is capped by
  // both register pressure and the 48-KiB default dynamic-smem limit. Split a wider decode into the
  // largest legal chunks instead of falling through to an Lt algorithm
  // with shape-dependent reduction order. This is the numerical interface that
  // lets a live request move between scalar and batched graph variants
  // without changing its transcript (each chunk past the first re-reads
  // the weights: the batch's byte cost).
  // The tensor-core form takes EVERY row count of an opted-in instance
  // (128-row groups above one launch): a row's chain then never depends on
  // the rows sharing its launch, prefill included — the group prefill's
  // spans are bitwise their prefills alone (an Lt algorithm's split
  // changes with m). Long prefills pay a little on the small bf16 sites.
  if (io_dtype == DType::BF16 && m >= 1 && impl_->decode_mma &&
      (impl_->decode_mma_max_rows == 0 || m <= impl_->decode_mma_max_rows) &&
      mma_gemv_shape_ok(static_cast<const uint16_t*>(weight), static_cast<const uint16_t*>(act),
                        act_row_stride, m, k)) {
    const auto* x = static_cast<const uint16_t*>(act);
    const auto* w = static_cast<const uint16_t*>(weight);
    auto* y = static_cast<uint8_t*>(out);
    if (out_dtype == GemmOut::F32)
      launch_mma_gemv_bf16_f32(x, act_row_stride, w, reinterpret_cast<float*>(y), m, n, k,
                               static_cast<size_t>(n), stream);
    else
      launch_mma_gemv_bf16_bf16(x, act_row_stride, w, reinterpret_cast<uint16_t*>(y), m, n, k,
                                static_cast<size_t>(n), stream);
    return;
  }
  // A registered companion (bf12_gemv.hpp) takes the lowering's launches —
  // bitwise the bf16 GEMV's rows, 0.75 of the bytes — eight rows a launch,
  // so a five-to-eight-row batch reads the packed bytes once instead of
  // falling to an Lt algorithm over the bf16 ones (its rows then carry the
  // scalar chain too).
  const Bf12Matrix* packed =
      io_dtype == DType::BF16 && m >= 1 ? impl_->bf12_for(weight, n, k) : nullptr;
  if (io_dtype == DType::BF16 && m >= 1 && m <= impl_->gemv_rows(packed) &&
      bf16_gemv_accepts(weight, /*m=*/1, k)) {
    const auto* x = static_cast<const uint16_t*>(act);
    const auto* w = static_cast<const uint16_t*>(weight);
    const size_t out_elem = out_dtype == GemmOut::F32 ? sizeof(float)
                                                       : sizeof(uint16_t);
    auto* y = static_cast<uint8_t*>(out);
    for (int row0 = 0; row0 < m;) {
      const int rows = packed != nullptr ? std::min(kBf12MaxRows, m - row0)
                                         : gemv_chunk_rows(weight, m - row0, k);
      const uint16_t* xr = x + static_cast<size_t>(row0) * act_row_stride;
      uint8_t* yr = y + static_cast<size_t>(row0) * n * out_elem;
      if (packed != nullptr)
        launch_bf12_gemv(xr, act_row_stride, *packed, yr, out_dtype == GemmOut::F32, rows,
                         stream);
      else
        launch_bf16_gemv(xr, act_row_stride, w, yr, out_dtype == GemmOut::F32, rows, n, k,
                         stream);
      row0 += rows;
    }
    return;
  }
  Impl::Plan& p = impl_->get_plan(m, n, k, io_dtype, out_dtype,
                                  act_row_stride, workspace, ws_bytes);
  // A row block of a wider chunk takes the chunk's algorithm (set_plan_rows).
  // bf16 calls only: the fp8 dot GEMMs already run in context-sized tiles
  // and keep their own algorithm either way.
  const Impl::Plan& chosen =
      impl_->plan_rows > m && io_dtype == DType::BF16
          ? impl_->get_plan(impl_->plan_rows, n, k, io_dtype, out_dtype, act_row_stride,
                            workspace, ws_bytes)
          : p;
  float alpha = 1.f, beta = 0.f;
  // Heuristic-selected algo + fixed layouts keep replays bitwise-stable in
  // process (graph-capture determinism requirement, DESIGN §11).
  DGPP_CUBLAS_OK(
      cublasLtMatmul(impl_->lt, p.desc, &alpha, weight, p.la, act, p.lb, &beta,
                     out, p.ld, out, p.ld, &chosen.algo, workspace, ws_bytes,
                     stream),
      std::format("matmul m={} n={} k={} lda={} dtype={}", m, n, k,
                  act_row_stride, dtype_name(io_dtype)));
}

size_t CublasLtGemm::query_workspace_bytes(int, int, int, DType) {
  return kRecommendedWorkspace;
}

void CublasLtGemm::matmul_batched_bf16(const uint16_t* act, const uint16_t* weight, float* out,
                                       int m, int n, int k, int batch, int64_t act_stride,
                                       int64_t weight_stride, int64_t out_stride, void* workspace,
                                       size_t ws_bytes, cudaStream_t stream, int plan_rows,
                                       bool weight_kn) {
  if (m <= 0 || n <= 0 || k <= 0 || batch <= 0 || (plan_rows != 0 && plan_rows < m) ||
      act_stride < static_cast<int64_t>(m) * k || weight_stride < static_cast<int64_t>(n) * k ||
      out_stride < static_cast<int64_t>(m) * n)
    throw std::invalid_argument("batched BF16 matmul: invalid shape or overlapping batch stride");
  auto& p = impl_->get_plan(m, n, k, DType::BF16, GemmOut::F32, k, workspace, ws_bytes, batch,
                            act_stride, weight_stride, out_stride, nullptr, weight_kn);
  const auto& chosen =
      plan_rows > m
          ? impl_->get_plan(plan_rows, n, k, DType::BF16, GemmOut::F32, k, workspace, ws_bytes,
                            batch, static_cast<int64_t>(plan_rows) * k, weight_stride,
                            static_cast<int64_t>(plan_rows) * n, nullptr, weight_kn)
          : p;
  const float alpha = 1.f, beta = 0.f;
  DGPP_CUBLAS_OK(cublasLtMatmul(impl_->lt, p.desc, &alpha, weight, p.la, act, p.lb, &beta, out,
                                p.ld, out, p.ld, &chosen.algo, workspace, ws_bytes, stream),
                 "batched BF16 matmul");
}

void CublasLtGemm::matmul_linear_bf16(const uint16_t* act, const uint16_t* weight,
                                      const uint16_t* bias, uint16_t* out, int m, int n, int k,
                                      void* workspace, size_t ws_bytes, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) throw std::invalid_argument("BF16 linear: invalid shape");
  auto& p = impl_->get_plan(m, n, k, DType::BF16, GemmOut::BF16, k, workspace, ws_bytes, 1, 0, 0, 0,
                            bias, false, true);
  if (bias)
    DGPP_CUBLAS_OK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                                  sizeof(bias)),
                   "bias pointer");
  const float alpha = 1.f, beta = 0.f;
  DGPP_CUBLAS_OK(cublasLtMatmul(impl_->lt, p.desc, &alpha, weight, p.la, act, p.lb, &beta, out,
                                p.ld, out, p.ld, &p.algo, workspace, ws_bytes, stream),
                 "BF16 linear");
}

bool CublasLtGemm::ensure_plan(int m, int n, int k, DType io_dtype, GemmOut out_dtype,
                               size_t act_row_stride) {
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
