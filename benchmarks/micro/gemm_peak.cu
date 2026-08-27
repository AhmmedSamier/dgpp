// micro_gemm_peak: cuBLASLt FP8(E4M3)/BF16 GEMM throughput over GLM-5.3-Flash
// representative shapes. Output lines are machine-parseable:
//   RES tag dtype m n k us tflops weight_GBps heuristic valid/returned
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/log.hpp"

namespace {

struct Shape {
  int m, n, k;
};

struct Case {
  const char* tag;
  Shape s;
};

void check_cuda(cudaError_t err, const char* what);

__global__ void clock_warmup_kernel(uint32_t* sink, uint64_t cycles) {
  const uint64_t start = clock64();
  uint32_t value = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  while (clock64() - start < cycles)
    value = value * 1664525u + 1013904223u;
  if (threadIdx.x == 0) atomicAdd(sink, value);
}

void warm_device(void* workspace, cudaStream_t stream) {
  int clock_khz = 0;
  int multiprocessors = 0;
  check_cuda(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0),
             "query clock rate");
  check_cuda(cudaDeviceGetAttribute(&multiprocessors,
                                    cudaDevAttrMultiProcessorCount, 0),
             "query multiprocessor count");
  check_cuda(cudaMemsetAsync(workspace, 0, sizeof(uint32_t), stream),
             "warmup sink clear");
  const uint64_t cycles = static_cast<uint64_t>(clock_khz) * 1000ull;
  clock_warmup_kernel<<<multiprocessors, 256, 0, stream>>>(
      static_cast<uint32_t*>(workspace), cycles);
  check_cuda(cudaGetLastError(), "clock warmup launch");
  check_cuda(cudaStreamSynchronize(stream), "clock warmup sync");
}

// Representative GLM-5.3-Flash projection group-gemm inner kernels.
const std::vector<Case>& cases() {
  static const std::vector<Case> kCases{
      {"attn_o_proj", {1, 4096, 16384}},
      {"qkv_or_qb", {1, 16384, 1536}},
      {"moe_w13", {1, 4096, 4096}},
      {"moe_w2", {1, 4096, 2048}},
      {"dense_gateup", {1, 24576, 4096}},
      {"lm_head", {1, 154880, 4096}},
      {"attn_o_proj_pf", {2048, 4096, 16384}},
      {"dense_gateup_pf", {2048, 24576, 4096}},
  };
  return kCases;
}

void check_cublas(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    DGPP_LOG_ERROR("cublasLt {} failed status={}", what, static_cast<int>(st));
    std::exit(1);
  }
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    DGPP_LOG_ERROR("{} failed: {}", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

struct Buffer {
  void *a{}, *b{}, *d{};
  size_t a_bytes{}, b_bytes{}, d_bytes{};
};

Buffer alloc_case(const Case& c, bool fp8) {
  Buffer buf;
  buf.a_bytes = size_t(c.s.m) * c.s.k * (fp8 ? 1 : 2);
  buf.b_bytes = size_t(c.s.k) * c.s.n * (fp8 ? 1 : 2);
  buf.d_bytes = size_t(c.s.m) * c.s.n * (fp8 ? 2 : 2);  // always BF16 out
  check_cuda(cudaMalloc(&buf.a, buf.a_bytes), "malloc a");
  check_cuda(cudaMalloc(&buf.b, buf.b_bytes), "malloc b");
  check_cuda(cudaMalloc(&buf.d, buf.d_bytes), "malloc d");
  check_cuda(
      cudaMemset(buf.a, 0x3C, buf.a_bytes),  // ~0.01 fp8 / benign bf16 bits
      "memset a");
  check_cuda(cudaMemset(buf.b, 0x3C, buf.b_bytes), "memset b");
  return buf;
}

double bench(bool fp8, const Case& c, cublasLtHandle_t lt, void* ws,
             size_t ws_bytes, cudaStream_t stream) {
  Buffer buf = alloc_case(c, fp8);
  // Convention: C_rm[m,n] = act_rm[m,k] x W_rm[n_out=k?]... concrete:
  // torch weight buffers are [out,in] -> here buf.b holds W[k*n? No:
  // alloc_case sizes b = k*n interpreted as W stored [k rows, n cols].
  // Column-major math: D_cm(n,m) = op_T(W_cm(k,n)) * act_cm(k,m).
  cublasOperation_t ta = CUBLAS_OP_T;  // weights: cm view (k,n) -> (n,k)
  cublasOperation_t tb = CUBLAS_OP_N;  // activations: cm view (k,m)

  cublasComputeType_t comp = CUBLAS_COMPUTE_32F;
  cudaDataType_t ab_type = fp8 ? CUDA_R_8F_E4M3 : CUDA_R_16BF;

  cublasLtMatmulDesc_t desc{};
  check_cublas(cublasLtMatmulDescCreate(&desc, comp, CUDA_R_32F), "desc create");
  check_cublas(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                              &ta, sizeof(ta)),
               "set transa");
  check_cublas(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                              &tb, sizeof(tb)),
               "set transb");

  cublasLtMatrixLayout_t la{}, lb{}, ld{};
  check_cublas(cublasLtMatrixLayoutCreate(&la, ab_type, c.s.k, c.s.n, c.s.k),
               "layout a");  // W cm: rows=k cols=n ld=k
  check_cublas(cublasLtMatrixLayoutCreate(&lb, ab_type, c.s.k, c.s.m, c.s.k),
               "layout b");  // act cm: rows=k cols=m ld=k
  check_cublas(
      cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, c.s.n, c.s.m, c.s.n),
      "layout d");

  float alpha = 1.f, beta = 0.f;
  // FP8 requires alpha/beta scaling pointers in host memory (bf16 scale ptrs
  // supported for tensor-wise scaling; plain fp32 scalars OK).
  cublasLtMatmulPreference_t pref{};
  check_cublas(cublasLtMatmulPreferenceCreate(&pref), "pref create");
  check_cublas(
      cublasLtMatmulPreferenceSetAttribute(pref,
                                           CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &ws_bytes, sizeof(ws_bytes)),
      "pref ws");

  constexpr int kMaxHeuristics = 16;
  cublasLtMatmulHeuristicResult_t heur[kMaxHeuristics]{};
  int nres = 0;
  check_cublas(
      cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, ld, ld, pref,
                                     kMaxHeuristics, heur, &nres),
      "heuristic");
  if (nres == 0) {
    DGPP_LOG_WARN("no heuristic for {} fp8={} skipping", c.tag, int(fp8));
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(desc);
    cudaFree(buf.a);
    cudaFree(buf.b);
    cudaFree(buf.d);
    return -1.0;
  }

  const int iters = c.s.m <= 8 ? 30 : 5;
  cudaEvent_t beg{}, end{};
  check_cuda(cudaEventCreate(&beg), "event create begin");
  check_cuda(cudaEventCreate(&end), "event create end");
  double best_ms = DBL_MAX;
  int best_heuristic = -1;
  int valid_heuristics = 0;
  for (int candidate = 0; candidate < nres; ++candidate) {
    bool valid = true;
    for (int warmup = 0; warmup < 2; ++warmup) {
      const cublasStatus_t status = cublasLtMatmul(
          lt, desc, &alpha, buf.b, la, buf.a, lb, &beta, buf.d, ld, buf.d, ld,
          &heur[candidate].algo, ws, ws_bytes, stream);
      if (status != CUBLAS_STATUS_SUCCESS) {
        valid = false;
        break;
      }
    }
    if (!valid || cudaStreamSynchronize(stream) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }

    constexpr int kSamples = 3;
    std::array<double, kSamples> samples{};
    for (int sample = 0; sample < kSamples && valid; ++sample) {
      check_cuda(cudaEventRecord(beg, stream), "event record begin");
      for (int iteration = 0; iteration < iters; ++iteration) {
        const cublasStatus_t status = cublasLtMatmul(
            lt, desc, &alpha, buf.b, la, buf.a, lb, &beta, buf.d, ld, buf.d,
            ld, &heur[candidate].algo, ws, ws_bytes, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
          valid = false;
          break;
        }
      }
      check_cuda(cudaEventRecord(end, stream), "event record end");
      if (!valid || cudaEventSynchronize(end) != cudaSuccess) {
        cudaGetLastError();
        valid = false;
        break;
      }
      float total_ms = 0.0f;
      check_cuda(cudaEventElapsedTime(&total_ms, beg, end), "event elapsed");
      samples[sample] = static_cast<double>(total_ms) / iters;
    }
    if (!valid) continue;
    std::sort(samples.begin(), samples.end());
    const double candidate_ms = samples[kSamples / 2];
    ++valid_heuristics;
    if (candidate_ms < best_ms) {
      best_ms = candidate_ms;
      best_heuristic = candidate;
    }
  }
  if (best_heuristic < 0) {
    DGPP_LOG_WARN("all heuristics failed for {} fp8={}", c.tag, int(fp8));
    best_ms = -1.0;
  }

  double us = best_ms > 0.0 ? best_ms * 1000.0 : 0.0;
  double flops = 2.0 * c.s.m * c.s.n * c.s.k;
  double tflops = us > 0.0 ? flops / (us * 1e-6) / 1e12 : 0.0;
  double w_gb = double(c.s.k) * c.s.n * (fp8 ? 1 : 2) / 1e9;
  double eff_bw = us > 0.0 ? w_gb / (us * 1e-6) : 0.0;
  std::printf(
      "RES %s %s m=%d n=%d k=%d %.1f us %.1f TFLOPS %.0f GB/s "
      "heuristic=%d valid=%d/%d\n",
      c.tag, fp8 ? "fp8" : "bf16", c.s.m, c.s.n, c.s.k, us, tflops,
      c.s.m <= 8 ? eff_bw : 0.0, best_heuristic, valid_heuristics, nres);

  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(ld);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatmulDescDestroy(desc);
  check_cuda(cudaEventDestroy(beg), "event destroy begin");
  check_cuda(cudaEventDestroy(end), "event destroy end");
  cudaFree(buf.a);
  cudaFree(buf.b);
  cudaFree(buf.d);
  return best_heuristic >= 0 ? tflops : -1.0;
}

}  // namespace

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  cublasLtHandle_t lt{};
  check_cublas(cublasLtCreate(&lt), "lt create");

  size_t ws_bytes = 64ull << 20;
  void* ws = nullptr;
  check_cuda(cudaMalloc(&ws, ws_bytes), "ws malloc");
  cudaStream_t stream{};
  check_cuda(cudaStreamCreate(&stream), "stream create");
  warm_device(ws, stream);

  int failed_cases = 0;
  for (bool fp8 : {true, false}) {
    for (const auto& c : cases()) {
      if (bench(fp8, c, lt, ws, ws_bytes, stream) < 0.0) ++failed_cases;
    }
  }

  check_cuda(cudaStreamDestroy(stream), "stream destroy");
  check_cuda(cudaFree(ws), "workspace free");
  check_cublas(cublasLtDestroy(lt), "lt destroy");
  DGPP_LOG_INFO("done");
  return failed_cases == 0 ? 0 : 1;
}
