cmake_minimum_required(VERSION 3.24)
project(dgpp VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 121)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DGPP_WERROR "Treat compiler warnings as errors" OFF)
set(DGPP_SANITIZE "" CACHE STRING "Host sanitizer (address or undefined)")
set_property(CACHE DGPP_SANITIZE PROPERTY STRINGS "" address undefined)

if(DGPP_SANITIZE)
  if(NOT DGPP_SANITIZE MATCHES "^(address|undefined)$")
    message(FATAL_ERROR "DGPP_SANITIZE must be address, undefined, or empty")
  endif()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${DGPP_SANITIZE}>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=${DGPP_SANITIZE}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-omit-frame-pointer>)
  # CMake may select CUDA as a target's link language while still invoking the
  # host C++ driver for the final link.  Linker options therefore need the host
  # spelling for both C++-only and mixed CUDA/C++ targets.
  add_link_options(-fsanitize=${DGPP_SANITIZE})
endif()

option(DGPP_ENABLE_IBV "Build ibverbs-based microbenches" ON)

add_library(dgpp_common INTERFACE)
target_include_directories(dgpp_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublasLt)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

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
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;-Xcompiler=-ffp-contract=off>)
  if(DGPP_WERROR)
    target_compile_options(${t} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
  endif()
endfunction()

find_program(DGPP_CLANG_FORMAT
  NAMES clang-format clang-format-19 clang-format-18 clang-format-17)
if(DGPP_CLANG_FORMAT)
  file(GLOB_RECURSE DGPP_FORMAT_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu)
  add_custom_target(format
    COMMAND ${DGPP_CLANG_FORMAT} -i ${DGPP_FORMAT_SOURCES}
    COMMENT "Formatting C++ and CUDA sources")
  add_custom_target(format-check
    COMMAND ${DGPP_CLANG_FORMAT} --dry-run --Werror ${DGPP_FORMAT_SOURCES}
    COMMENT "Checking C++ and CUDA formatting")
else()
  message(STATUS "clang-format not found; format targets are unavailable")
endif()

# ---- libraries ------------------------------------------------------------

add_library(dgpp_core STATIC
  src/common/log.cpp
  src/core/trace.cpp)
dgpp_set_common(dgpp_core)

# M5 control plane: TCP roster/startup/health (host-only; no CUDA).
add_library(dgpp_net STATIC
  src/net/tcp.cpp
  src/net/roster.cpp)
target_link_libraries(dgpp_net PUBLIC dgpp_core Threads::Threads)
dgpp_set_common(dgpp_net)

add_library(dgpp_loaders STATIC
  src/loaders/minijson.cpp
  src/loaders/shardspec.cpp)
target_link_libraries(dgpp_loaders PUBLIC dgpp_core)
dgpp_set_common(dgpp_loaders)

add_library(dgpp_kernels STATIC
  src/kernels/gemm_cublaslt.cpp
  src/kernels/kernels.cu
  src/kernels/doll_attention.cu
  src/kernels/kda.cu
  src/kernels/dsa.cu
  src/kernels/fp8_dequant.cu
  src/kernels/scale_gemm.cu
  src/kernels/glm_mhc.cu
  src/kernels/glm_moe.cu
  src/kernels/glm_norm.cu)
target_link_libraries(dgpp_kernels PUBLIC CUDA::cudart CUDA::cublasLt dgpp_core)
dgpp_set_common(dgpp_kernels)

add_library(dgpp_models_doll STATIC
  src/models/gpt_doll.cu)
target_link_libraries(dgpp_models_doll PUBLIC dgpp_kernels dgpp_core)
dgpp_set_common(dgpp_models_doll)

add_library(dgpp_models_kda STATIC
  src/models/kda_state.cpp
  src/models/kda_layer.cu
  src/models/kda_reference.cpp
  src/models/kda_dump.cpp)
target_link_libraries(dgpp_models_kda PUBLIC dgpp_kernels dgpp_loaders CUDA::cudart)
dgpp_set_common(dgpp_models_kda)

add_library(dgpp_models_dsa STATIC
  src/models/dsa_reference.cpp
  src/models/dsa_dump.cpp
  src/models/dsa_state.cpp
  src/models/dsa_layer.cu)
target_link_libraries(dgpp_models_dsa PUBLIC dgpp_kernels CUDA::cudart)
dgpp_set_common(dgpp_models_dsa)

# Host-only GLM model assembly (config parsing, expected-tensor binding); no
# CUDA dependency so the loader/binding layer builds and tests anywhere.
add_library(dgpp_models_glm STATIC
  src/models/glm_config.cpp
  src/models/glm_binding.cpp
  src/models/glm_mhc_reference.cpp
  src/models/glm_moe_reference.cpp
  src/models/glm_trace.cpp)
target_link_libraries(dgpp_models_glm PUBLIC dgpp_loaders)
dgpp_set_common(dgpp_models_glm)

# Streaming resident loader (M4): managed-memory layer residency over the
# binding table, with the block-scale dequant bridge for the M3 seams.
# Chunk 6 adds the full diagnostic forward (glm_forward) and the reference
# dump reader (glm_dump) on top; they drive the parity-tested M2/M3 layer
# modules, hence the kda/dsa library links.
add_library(dgpp_models_glm_loader STATIC
  src/models/glm_loader.cpp
  src/models/glm_moe_layer.cpp
  src/models/glm_dump.cpp
  src/models/glm_forward.cpp
  src/models/glm_tp.cpp)
target_link_libraries(dgpp_models_glm_loader
  PUBLIC dgpp_models_glm dgpp_kernels dgpp_models_kda dgpp_models_dsa)
dgpp_set_common(dgpp_models_glm_loader)

add_executable(glm_bind_check apps/glm_bind_check.cpp)
target_link_libraries(glm_bind_check PRIVATE dgpp_models_glm dgpp_loaders)
dgpp_set_common(glm_bind_check)

# Deployment evidence: stream real layers, reconcile bytes vs formula.
add_executable(glm_stream_check apps/glm_stream_check.cpp)
target_link_libraries(glm_stream_check PRIVATE dgpp_models_glm_loader)
dgpp_set_common(glm_stream_check)

# M4 deliverables 1+5 deployment run: the curated reference suite over the
# real checkpoint (dumps from tools/glm_reference_dump.py gen-torch) and
# DGPPTC1 route-trace capture for the traffic model.
add_executable(glm_forward_check apps/glm_forward_check.cpp)
target_link_libraries(glm_forward_check PRIVATE dgpp_models_glm_loader CUDA::cudart)
dgpp_set_common(glm_forward_check)

add_executable(dgppctl apps/dgppctl.cpp src/common/log.cpp)
target_link_libraries(dgppctl PRIVATE CUDA::cudart)
dgpp_set_common(dgppctl)

# M5 deliverable 1 deployment driver: roster/startup/health over TCP.
add_executable(roster_check apps/roster_check.cpp)
target_link_libraries(roster_check PRIVATE dgpp_net)
dgpp_set_common(roster_check)

# M5 deliverable 2: the CollectiveBus data plane (verbs + CUDA). Gated on
# ibverbs availability; the control plane (dgpp_net) never requires it.
if(DGPP_ENABLE_IBV)
  add_library(dgpp_bus STATIC
    src/net/bus_types.cpp
    src/net/verbs.cpp
    src/net/collective_bus.cpp
    src/net/bus_kernel.cu
    src/net/nic_visibility.cu)
  target_link_libraries(dgpp_bus
    PUBLIC dgpp_net CUDA::cudart ${IBVERBS_LIBRARY})
  target_include_directories(dgpp_bus PUBLIC ${IBVERBS_INCLUDE_DIR})
  dgpp_set_common(dgpp_bus)

  add_executable(bus_check apps/bus_check.cpp)
  target_link_libraries(bus_check PRIVATE dgpp_bus)
  dgpp_set_common(bus_check)

  # M5 deliverable 5: the transport regression command (NIC→GPU visibility
  # on every node pair after driver/firmware changes).
  add_executable(nic_regress apps/nic_regress.cpp)
  target_link_libraries(nic_regress PRIVATE dgpp_bus CUDA::cudart Threads::Threads)
  dgpp_set_common(nic_regress)
  add_test(NAME nic_regress_selftest COMMAND nic_regress selftest)
  set_tests_properties(nic_regress_selftest PROPERTIES SKIP_RETURN_CODE 2)

  # M5 deliverable 3: TP block-boundary parity over loopback buses
  # (real verbs QPs, one process). Needs the model stack AND the bus.
  # The fixture side is g++-only (nvcc cannot compile minijson).
  add_executable(glm_tp_test tests/cuda/glm_tp_test.cpp
                        tests/cuda/glm_tp_fixture.cpp)
  target_include_directories(glm_tp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(glm_tp_test PRIVATE dgpp_models_glm_loader dgpp_bus
                                         CUDA::cudart Threads::Threads)
  dgpp_set_common(glm_tp_test)
  add_test(NAME glm_tp_test COMMAND glm_tp_test)
  set_tests_properties(glm_tp_test PROPERTIES SKIP_RETURN_CODE 2)

  # M5 deliverable 3 fabric runner: TP forward on real nodes.
  add_executable(glm_tp_check apps/glm_tp_check.cpp)
  target_link_libraries(glm_tp_check PRIVATE dgpp_models_glm_loader dgpp_bus
                                          CUDA::cudart Threads::Threads)
  dgpp_set_common(glm_tp_check)
endif()

enable_testing()
add_executable(unit_tests
  tests/unit/log_test.cpp
  tests/unit/minijson_test.cpp
  tests/unit/arena_test.cpp
  tests/unit/safetensors_test.cpp
  tests/unit/fp8_test.cpp
  tests/unit/shardspec_test.cpp
  tests/unit/kda_geometry_test.cpp
  tests/unit/dsa_geometry_test.cpp
  tests/unit/glm_config_test.cpp
  tests/unit/glm_binding_test.cpp
  tests/unit/glm_trace_test.cpp
  tests/unit/glm_route_audit_test.cpp
  tests/unit/net_test.cpp
  src/core/trace.cpp)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(unit_tests PRIVATE CUDA::cudart dgpp_loaders
                                        dgpp_models_glm dgpp_net Threads::Threads)
dgpp_set_common(unit_tests)
add_executable(flag_protocol_test tests/cuda/flag_protocol_test.cu)
target_link_libraries(flag_protocol_test PRIVATE CUDA::cudart)
dgpp_set_common(flag_protocol_test)

add_test(NAME unit COMMAND unit_tests)
add_test(NAME checkpoint_audit_test
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/checkpoint_audit_test.py)
add_test(NAME route_trace_test
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/route_trace_test.py)

# KDA operator tests (CUDA); skip without a device as above. The dump-parity
# runner is a host-only TU (minijson + nvcc don't mix).
add_executable(kda_test
  tests/cuda/kda_test.cu
  tests/cuda/kda_dump_parity.cpp)
target_include_directories(kda_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kda_test PRIVATE dgpp_models_kda CUDA::cudart Threads::Threads)
dgpp_set_common(kda_test)
add_test(NAME kda_test COMMAND kda_test)
set_tests_properties(kda_test PROPERTIES SKIP_RETURN_CODE 2)

add_executable(dsa_test
  tests/cuda/dsa_test.cu
  tests/cuda/dsa_dump_parity.cpp)
target_include_directories(dsa_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dsa_test PRIVATE dgpp_models_dsa dgpp_models_kda CUDA::cudart Threads::Threads)
dgpp_set_common(dsa_test)
add_test(NAME dsa_test COMMAND dsa_test)
set_tests_properties(dsa_test PROPERTIES SKIP_RETURN_CODE 2)

# M4 streaming loader tests (CUDA): synthetic mini-checkpoint on disk,
# byte-exact layout merges, dequant bridge vs host oracle, formula parity.
add_executable(glm_loader_test tests/cuda/glm_loader_test.cpp)
target_include_directories(glm_loader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(glm_loader_test PRIVATE dgpp_models_glm_loader CUDA::cudart)
dgpp_set_common(glm_loader_test)
add_test(NAME glm_loader_test COMMAND glm_loader_test)
set_tests_properties(glm_loader_test PROPERTIES SKIP_RETURN_CODE 2)

# Scale-aware GEMM parity (M4): block edges, saturation, NaN policy;
# --checkpoint-dir adds real-checkpoint slices (manual deployment test —
# the checkpoint-runner TU is host-only, minijson does not mix with nvcc).
add_executable(scale_gemm_test
  tests/cuda/scale_gemm_test.cu
  tests/cuda/scale_gemm_checkpoint.cpp)
target_include_directories(scale_gemm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(scale_gemm_test PRIVATE dgpp_kernels dgpp_loaders CUDA::cudart)
dgpp_set_common(scale_gemm_test)
add_test(NAME scale_gemm_test COMMAND scale_gemm_test)
set_tests_properties(scale_gemm_test PROPERTIES SKIP_RETURN_CODE 2)

# mHC module parity (M4): compute/update/final-mean kernels vs the double
# oracle; real geometry, edge cases, determinism.
add_executable(glm_mhc_test tests/cuda/glm_mhc_test.cu)
target_include_directories(glm_mhc_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(glm_mhc_test PRIVATE dgpp_kernels dgpp_models_glm CUDA::cudart)
dgpp_set_common(glm_mhc_test)
add_test(NAME glm_mhc_test COMMAND glm_mhc_test)
set_tests_properties(glm_mhc_test PROPERTIES SKIP_RETURN_CODE 2)

# MoE module parity (M4): router vs oracle at real geometry, expert path on
# small geometry, swiglu clamps, accumulation order, determinism.
add_executable(glm_moe_test tests/cuda/glm_moe_test.cu)
target_include_directories(glm_moe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(glm_moe_test PRIVATE dgpp_models_glm_loader dgpp_models_glm CUDA::cudart)
dgpp_set_common(glm_moe_test)
add_test(NAME glm_moe_test COMMAND glm_moe_test)
set_tests_properties(glm_moe_test PROPERTIES SKIP_RETURN_CODE 2)

# Full-model forward parity (M4 chunk 6): this test writes a synthetic
# mini-checkpoint, python computes the full-stack pure reference from the
# SAME checkpoint (one tensor-table source: the C++ binding), and the test
# runs the diagnostic forward against the dump. The torch backend on the
# real checkpoint is the manual deployment run (glm_forward_check).
add_executable(glm_forward_test tests/cuda/glm_forward_test.cpp)
target_link_libraries(glm_forward_test
  PRIVATE dgpp_models_glm_loader CUDA::cudart)
dgpp_set_common(glm_forward_test)
set(GLM_FORWARD_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/glm_forward_fixture)
add_test(NAME glm_forward_fixture
  COMMAND glm_forward_test --write-fixture ${GLM_FORWARD_FIXTURE})
set_tests_properties(glm_forward_fixture PROPERTIES SKIP_RETURN_CODE 2)
add_test(NAME glm_forward_generate
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_reference_dump.py
          gen-pure --checkpoint-dir ${GLM_FORWARD_FIXTURE}
          --out ${CMAKE_CURRENT_BINARY_DIR}/glm_pure_ref.glmdump)
set_tests_properties(glm_forward_generate PROPERTIES
  DEPENDS glm_forward_fixture)
add_test(NAME glm_forward_test
  COMMAND glm_forward_test --checkpoint-dir ${GLM_FORWARD_FIXTURE}
          --dump-file ${CMAKE_CURRENT_BINARY_DIR}/glm_pure_ref.glmdump)
set_tests_properties(glm_forward_test PROPERTIES
  DEPENDS glm_forward_generate SKIP_RETURN_CODE 2)

# DSA reference-dump parity: generate a pure-python oracle dump, then run
# the layer against it (gen-torch is the manual deployment test).
add_test(NAME dsa_dump_generate
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/dsa_reference_dump.py
          gen-pure --out ${CMAKE_CURRENT_BINARY_DIR}/dsa_pure_ref.dsadump)
add_test(NAME dsa_dump_test
  COMMAND dsa_test --dump-file ${CMAKE_CURRENT_BINARY_DIR}/dsa_pure_ref.dsadump)
set_tests_properties(dsa_dump_test PROPERTIES
  DEPENDS dsa_dump_generate SKIP_RETURN_CODE 2)

# Reference-dump parity: generate a pure-python oracle dump, then run the
# layer against it. The torch backend (real checkpoint slices) is a manual
# deployment test documented in benchmarks/README.md.
add_test(NAME kda_dump_generate
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/kda_reference_dump.py
          gen-pure --out ${CMAKE_CURRENT_BINARY_DIR}/kda_pure_ref.kdadump
          --heads 2 --head-dim 32 --hidden 64 --tokens 8)
add_test(NAME kda_dump_test
  COMMAND kda_test --dump-file ${CMAKE_CURRENT_BINARY_DIR}/kda_pure_ref.kdadump)
set_tests_properties(kda_dump_test PROPERTIES
  DEPENDS kda_dump_generate SKIP_RETURN_CODE 2)

add_executable(gpt_doll apps/gpt_doll.cpp)
target_link_libraries(gpt_doll PRIVATE dgpp_models_doll dgpp_loaders)
dgpp_set_common(gpt_doll)
# Fast GPU selftest; SKIP_RETURN_CODE 2 = no visible device, ctest skips.
add_test(NAME doll_selftest COMMAND gpt_doll --selftest --layers 2
  --hidden 1024 --vocab 8192 --inter 2048 --max-seq 512)
set_tests_properties(doll_selftest PROPERTIES SKIP_RETURN_CODE 2)

# Flag-protocol regression test (CUDA); skip without a device as above.
add_test(NAME flag_protocol_test COMMAND flag_protocol_test)

# M5 control-plane app smoke (loopback, in-process).
add_test(NAME roster_selftest COMMAND roster_check selftest)
set_tests_properties(flag_protocol_test PROPERTIES SKIP_RETURN_CODE 2)

# M5 data plane: deep loopback scenarios (verbs + consumers + contention)
# and the app-level smoke. Both skip without a visible device.
if(DGPP_ENABLE_IBV)
  add_executable(bus_test tests/cuda/bus_test.cu)
  target_link_libraries(bus_test PRIVATE dgpp_bus)
  dgpp_set_common(bus_test)
  add_test(NAME bus_test COMMAND bus_test)
  set_tests_properties(bus_test PROPERTIES SKIP_RETURN_CODE 2)
  add_test(NAME bus_selftest COMMAND bus_check selftest)
  set_tests_properties(bus_selftest PROPERTIES SKIP_RETURN_CODE 2)
endif()

add_executable(micro_mem_bw benchmarks/micro/mem_bw.cu src/common/log.cpp)
target_link_libraries(micro_mem_bw PRIVATE CUDA::cudart Threads::Threads)
dgpp_set_common(micro_mem_bw)

add_executable(micro_zerocopy benchmarks/micro/zerocopy.cu src/common/log.cpp)
target_link_libraries(micro_zerocopy PRIVATE CUDA::cudart Threads::Threads)
dgpp_set_common(micro_zerocopy)

add_executable(micro_gemm_peak benchmarks/micro/gemm_peak.cu src/common/log.cpp)
target_link_libraries(micro_gemm_peak PRIVATE CUDA::cudart CUDA::cublasLt)
dgpp_set_common(micro_gemm_peak)

add_executable(kda_bench benchmarks/micro/kda_bench.cu src/common/log.cpp)
target_link_libraries(kda_bench PRIVATE dgpp_models_kda CUDA::cudart CUDA::cublasLt)
dgpp_set_common(kda_bench)

add_executable(dsa_bench benchmarks/micro/dsa_bench.cu src/common/log.cpp)
target_link_libraries(dsa_bench PRIVATE dgpp_models_dsa CUDA::cudart CUDA::cublasLt)
dgpp_set_common(dsa_bench)

if(DGPP_ENABLE_IBV)
  add_executable(micro_gdr_probe benchmarks/micro/gdr_probe.cpp src/common/log.cpp)
  target_link_libraries(micro_gdr_probe PRIVATE CUDA::cudart)
  dgpp_set_common(micro_gdr_probe)
  target_link_libraries(micro_gdr_probe PRIVATE ${IBVERBS_LIBRARY})
  target_include_directories(micro_gdr_probe PRIVATE ${IBVERBS_INCLUDE_DIR})

  add_executable(micro_ibv_smoke
    benchmarks/micro/ibv_smoke.cpp
    src/net/nic_visibility.cu
    src/common/log.cpp)
  target_link_libraries(micro_ibv_smoke PRIVATE ${IBVERBS_LIBRARY} CUDA::cudart)
  target_include_directories(micro_ibv_smoke PRIVATE ${IBVERBS_INCLUDE_DIR})
  dgpp_set_common(micro_ibv_smoke)
endif()
