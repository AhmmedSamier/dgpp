#pragma once
#include <cstdio>
#include <format>
#include <source_location>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace dgpp {

inline std::runtime_error cuda_error(std::string_view what, cudaError_t err,
                                     std::source_location loc) {
  return std::runtime_error(std::format("cuda failure {}: {} | {} at {}:{}",
                                        static_cast<int>(err),
                                        cudaGetErrorString(err), what,
                                        loc.file_name(), loc.line()));
}

}  // namespace dgpp

#define DGPP_CUDA_OK(expr)                                                    \
  do {                                                                        \
    cudaError_t dgpp_err_ = (expr);                                           \
    if (dgpp_err_ != cudaSuccess) {                                           \
      /* Drain the sticky per-thread error so a failure here cannot */       \
      /* surface later as a stale cudaGetLastError in unrelated code. */     \
      cudaGetLastError();                                                     \
      throw ::dgpp::cuda_error(#expr, dgpp_err_,                              \
                               std::source_location::current());              \
    }                                                                         \
  } while (0)

#define DGPP_CUBLAS_OK(expr, ctx)                                            \
  do {                                                                       \
    cublasStatus_t dgpp_st_ = (expr);                                       \
    if (dgpp_st_ != CUBLAS_STATUS_SUCCESS)                                  \
      throw std::runtime_error(std::format("cublas failure {} ({}) at {}:{}",\
          static_cast<int>(dgpp_st_), ctx,                                   \
          std::source_location::current().file_name(),                       \
          std::source_location::current().line()));                          \
  } while (0)
