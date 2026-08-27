cmake_minimum_required(VERSION 3.24)
project(dgpp VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 121)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DEFINED ENV{DGPP_SANITIZE} AND NOT "$ENV{DGPP_SANITIZE}" STREQUAL "")
  add_compile_options(-fsanitize=$ENV{DGPP_SANITIZE} -fno-omit-frame-pointer)
  add_link_options(-fsanitize=$ENV{DGPP_SANITIZE})
endif()

option(DGPP_ENABLE_IBV "Build ibverbs-based microbenches" ON)
option(DGPP_ENABLE_MLX5 "Use mlx5dv fast paths in microbenches" OFF)

add_library(dgpp_common INTERFACE)
target_include_directories(dgpp_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublasLt)
find_package(Threads REQUIRED)

if(DGPP_ENABLE_IBV)
  find_path(IBVERBS_INCLUDE_DIR NAMES infiniband/verbs.h)
  find_library(IBVERBS_LIBRARY NAMES ibverbs)
  if(NOT IBVERBS_INCLUDE_DIR OR NOT IBVERBS_LIBRARY)
    message(WARNING "libibverbs dev missing; disabling IBV targets")
    set(DGPP_ENABLE_IBV OFF)
  endif()
endif()

if(DGPP_ENABLE_IBV)
  target_compile_definitions(dgpp_common INTERFACE DGPP_HAVE_IBV=1)
endif()

function(dgpp_set_common t)
  target_link_libraries(${t} PRIVATE dgpp_common)
  target_compile_options(${t} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>)
endfunction()

# ---- libraries ------------------------------------------------------------

add_library(dgpp_core STATIC
  src/common/log.cpp
  src/core/trace.cpp)
dgpp_set_common(dgpp_core)

add_library(dgpp_loaders STATIC
  src/loaders/minijson.cpp
  src/loaders/shardspec.cpp)
target_link_libraries(dgpp_loaders PUBLIC dgpp_core)
dgpp_set_common(dgpp_loaders)

add_library(dgpp_kernels STATIC
  src/kernels/gemm_cublaslt.cpp
  src/kernels/kernels.cu
  src/kernels/doll_attention.cu)
target_link_libraries(dgpp_kernels PUBLIC CUDA::cudart CUDA::cublasLt dgpp_core)
dgpp_set_common(dgpp_kernels)

add_library(dgpp_models_doll STATIC
  src/models/gpt_doll.cu)
target_link_libraries(dgpp_models_doll PUBLIC dgpp_kernels dgpp_core)
dgpp_set_common(dgpp_models_doll)

add_executable(dgppctl apps/dgppctl.cpp src/common/log.cpp)
target_link_libraries(dgppctl PRIVATE CUDA::cudart)
dgpp_set_common(dgppctl)

enable_testing()
add_executable(unit_tests
  tests/unit/log_test.cpp
  tests/unit/minijson_test.cpp
  tests/unit/arena_test.cpp
  tests/unit/safetensors_test.cpp
  tests/unit/fp8_test.cpp
  tests/unit/shardspec_test.cpp
  src/core/trace.cpp)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(unit_tests PRIVATE CUDA::cudart dgpp_loaders)
dgpp_set_common(unit_tests)
add_executable(flag_protocol_test tests/cuda/flag_protocol_test.cu)
target_link_libraries(flag_protocol_test PRIVATE CUDA::cudart)
dgpp_set_common(flag_protocol_test)

add_test(NAME unit COMMAND unit_tests)

add_executable(gpt_doll apps/gpt_doll.cpp)
target_link_libraries(gpt_doll PRIVATE dgpp_models_doll dgpp_loaders)
dgpp_set_common(gpt_doll)
# Fast GPU selftest; SKIP_RETURN_CODE 2 = no visible device, ctest skips.
add_test(NAME doll_selftest COMMAND gpt_doll --selftest --layers 2
  --hidden 1024 --vocab 8192 --inter 2048 --max-seq 512)
set_tests_properties(doll_selftest PROPERTIES SKIP_RETURN_CODE 2)

# Flag-protocol regression test (CUDA); skip without a device as above.
add_test(NAME flag_protocol_test COMMAND flag_protocol_test)
set_tests_properties(flag_protocol_test PROPERTIES SKIP_RETURN_CODE 2)

add_executable(micro_mem_bw benchmarks/micro/mem_bw.cu src/common/log.cpp)
target_link_libraries(micro_mem_bw PRIVATE CUDA::cudart Threads::Threads)
dgpp_set_common(micro_mem_bw)

add_executable(micro_zerocopy benchmarks/micro/zerocopy.cu src/common/log.cpp)
target_link_libraries(micro_zerocopy PRIVATE CUDA::cudart Threads::Threads)
dgpp_set_common(micro_zerocopy)

add_executable(micro_gemm_peak benchmarks/micro/gemm_peak.cu src/common/log.cpp)
target_link_libraries(micro_gemm_peak PRIVATE CUDA::cudart CUDA::cublasLt)
dgpp_set_common(micro_gemm_peak)

add_executable(micro_gdr_probe benchmarks/micro/gdr_probe.cpp src/common/log.cpp)
target_link_libraries(micro_gdr_probe PRIVATE CUDA::cudart)
dgpp_set_common(micro_gdr_probe)
if(DGPP_ENABLE_IBV)
  target_link_libraries(micro_gdr_probe PRIVATE ${IBVERBS_LIBRARY})
  target_include_directories(micro_gdr_probe PRIVATE ${IBVERBS_INCLUDE_DIR})
endif()

if(DGPP_ENABLE_IBV)
  add_executable(micro_ibv_smoke benchmarks/micro/ibv_smoke.cpp src/common/log.cpp)
  target_link_libraries(micro_ibv_smoke PRIVATE ${IBVERBS_LIBRARY})
  target_include_directories(micro_ibv_smoke PRIVATE ${IBVERBS_INCLUDE_DIR})
  dgpp_set_common(micro_ibv_smoke)
endif()
