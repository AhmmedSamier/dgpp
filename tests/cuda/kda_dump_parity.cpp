// Reference-dump parity runner (M2 deliverable 5). Host-only TU: the dump
// reader needs minijson, which nvcc's frontend refuses, so this file is
// compiled by the host compiler and linked into the kda_test binary.
// Called from kda_test.cu's main() when --dump-file is given.
//
// Declared tolerances (the "declared FP32-accumulation tolerance" of the M2
// exit criteria): the pure backend computes in IEEE double with bf16
// boundary rounding, the torch backend matches the reference layer; our
// path is fp32 accumulation with cuBLASLt reduction orders. Outputs agree
// within a couple of bf16 ulps; the FP32 recurrent state within 1e-3
// relative.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/kda_dump.hpp"
#include "tests/cuda/kda_test_helpers.hpp"

namespace {

using namespace dgpp::kda_test;

int check_tensor(const std::string& what, const std::vector<uint16_t>& got,
                 const dgpp::KdaDumpFile::TensorView& tv, int ulps,
                 double rel_tol, double mismatch_frac) {
  std::vector<uint16_t> want(tv.numel());
  std::memcpy(want.data(), tv.data, tv.nbytes);
  const Stats st = compare_bf16(got, want, ulps);
  require_bf16(what, st, rel_tol, mismatch_frac);
  std::printf("[ OK ] %-26s max_rel=%.3g l2_rel=%.3g mismatches=%ld/%ld\n",
              what.c_str(), st.max_rel, st.l2_rel, st.mismatches, st.n);
  return 0;
}

}  // namespace

int run_kda_dump_parity(const std::string& path) {
  try {
    dgpp::KdaDumpFile dump = dgpp::KdaDumpFile::load(path);
    const KdaConfig cfg = dump.single_layer_config();
    const KdaGeometry g = KdaGeometry::from_config(cfg);
    const int tokens = static_cast<int>(dump.config_json().at("tokens").as_int());

    TestWeights tw;  // device uploads use the dump's own bytes
    {
      auto copy16 = [&](const char* name, std::vector<uint16_t>& dst) {
        const auto& tv = dump.tensor(name);
        const auto* p = static_cast<const uint16_t*>(tv.data);
        dst.assign(p, p + tv.numel());
      };
      auto copyf32 = [&](const char* name, std::vector<float>& dst) {
        const auto& tv = dump.tensor(name);
        const auto* p = static_cast<const float*>(tv.data);
        dst.assign(p, p + tv.numel());
      };
      copy16("in_proj", tw.in_proj);
      copy16("f_b", tw.f_b);
      copy16("g_b", tw.g_b);
      copy16("conv", tw.conv);
      copy16("o_norm", tw.o_norm);
      copy16("o_proj", tw.o_proj);
      copyf32("a_log", tw.a_log);
      copyf32("dt_bias", tw.dt_bias);
    }

    cudaStream_t s = test_stream();
    DeviceWeights dw(tw);
    LayerEnv env(cfg, tokens, s);
    KdaLayer layer(env.arena, env.gemm, dw.views(), cfg, tokens, env.ws.p,
                   env.ws.bytes);
    if (!layer.prepare(tokens)) throw std::runtime_error("gemm plans unavailable");

    const auto& in_tv = dump.tensor("hidden_in");
    std::vector<uint16_t> hidden_in(static_cast<const uint16_t*>(in_tv.data),
                                    static_cast<const uint16_t*>(in_tv.data) +
                                        in_tv.numel());
    DevBuf din(hidden_in.size() * 2), dout(int64_t(tokens) * cfg.hidden * 2);
    din.upload(hidden_in.data(), hidden_in.size() * 2);
    DevBuf dstate(g.recurrent_elems * 4);
    DevBuf dconv(int64_t(g.conv_channels) * g.conv_state_width * 2);
    DGPP_CUDA_OK(cudaMemsetAsync(dstate.p, 0, dstate.bytes, s));
    DGPP_CUDA_OK(cudaMemsetAsync(dconv.p, 0, dconv.bytes, s));

    layer.enqueue(din.p, dstate.as<float>(), dconv.as<uint16_t>(),
                  g.conv_state_width, dout.p, tokens, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));

    std::vector<uint16_t> got_out(int64_t(tokens) * cfg.hidden);
    std::vector<uint16_t> got_core(int64_t(tokens) * g.local_proj);
    std::vector<float> got_state(g.recurrent_elems);
    std::vector<uint16_t> got_conv(int64_t(g.conv_channels) * g.conv_state_width);
    dout.download(got_out.data(), got_out.size() * 2);
    DGPP_CUDA_OK(cudaMemcpy(got_core.data(), layer.debug_core(),
                            int64_t(tokens) * g.local_proj * 2,
                            cudaMemcpyDeviceToHost));
    dstate.download(got_state.data(), dstate.bytes);
    dconv.download(got_conv.data(), dconv.bytes);

    const std::string tag = " [" + dump.backend() + "]";
    // The pure backend is a double-precision oracle; the torch backend
    // runs the same pinned equations in fp32 with bf16 boundaries — the
    // same cross-implementation drift class the DSA dump runner budgets
    // for (8 ulps / 5% mismatch). Pure keeps its tighter budget.
    const bool torch_backend = dump.backend() == "torch";
    const int out_ulps = torch_backend ? 8 : 2;
    const double out_mismatch = torch_backend ? 0.05 : 0.02;
    check_tensor("dump layer_out" + tag, got_out, dump.tensor("layer_out"),
                 out_ulps, 2e-2, out_mismatch);
    check_tensor("dump core_out" + tag, got_core, dump.tensor("core_out"),
                 out_ulps, 2e-2, out_mismatch);
    {
      const auto& tv = dump.tensor("recurrent_state_out");
      std::vector<float> want(tv.numel());
      std::memcpy(want.data(), tv.data, tv.nbytes);
      const Stats st = compare_abs_rel(got_state.data(), want.data(),
                                       tv.numel(), 1e-3, 1e-4);
      require_rel("dump recurrent state" + tag, st, 1e-3, 0.0);
      std::printf("[ OK ] %-26s max_rel=%.3g l2_rel=%.3g\n",
                  ("dump recurrent state" + tag).c_str(), st.max_rel, st.l2_rel);
    }
    if (dump.has_tensor("conv_state_out"))
      check_tensor("dump conv state" + tag, got_conv,
                   dump.tensor("conv_state_out"), 2, 5e-2, 0.05);
    return 0;
  } catch (const std::exception& e) {
    std::printf("[FAIL] dump parity: %s\n", e.what());
    return 1;
  }
}
