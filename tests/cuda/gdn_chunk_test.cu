// Chunked GDN parity, workspace isolation/lifetime, and kernel timing.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
#include <thread>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/gdn_chunk.hpp"
#include "kernels/kda.hpp"

using namespace dgpp;
using dgpp::kda_test::DevBuf;

namespace {

uint16_t bf16(float f) {
  const __nv_bfloat16 b = __float2bfloat16_rn(f);
  uint16_t u;
  std::memcpy(&u, &b, 2);
  return u;
}
float fbf(uint16_t u) {
  uint32_t x = static_cast<uint32_t>(u) << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

struct Stream {
  cudaStream_t s = nullptr;
  Stream() { DGPP_CUDA_OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(s); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};

struct Result {
  std::vector<uint16_t> output;
  std::vector<float> state;
};

struct Problem {
  static constexpr int K = 128, V = 128;
  int tokens, heads, kv_ratio;
  float scale = 1.f / std::sqrt(static_cast<float>(K));
  DevBuf qkv, a, b, alog, dtb, initial, state, output, workspace;

  Problem(int t, int h, int ratio, int seed)
      : tokens(t),
        heads(h),
        kv_ratio(ratio),
        qkv(static_cast<size_t>(t) * (2 * h / ratio * K + h * V) * 2),
        a(static_cast<size_t>(t) * h * 2),
        b(a.bytes),
        alog(h * 4),
        dtb(h * 4),
        initial(static_cast<size_t>(h) * V * K * 4),
        state(initial.bytes),
        output(static_cast<size_t>(t) * h * V * 2),
        workspace(gdn_chunked_workspace_bytes(t, h)) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (DevBuf* buffer : {&qkv, &a, &b}) {
      std::vector<uint16_t> values(buffer->bytes / 2);
      for (auto& x : values) x = bf16(nd(rng));
      buffer->upload(values.data(), buffer->bytes);
    }
    std::vector<float> ha(h), hd(h), hs(initial.bytes / 4);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    for (int i = 0; i < h; ++i) {
      ha[i] = std::log(1.f + 15.f * u(rng));
      hd[i] = nd(rng) * 0.5f;
    }
    for (auto& x : hs) x = nd(rng) * 0.1f;
    alog.upload(ha.data(), alog.bytes);
    dtb.upload(hd.data(), dtb.bytes);
    initial.upload(hs.data(), initial.bytes);
  }

  void reset(cudaStream_t stream) {
    DGPP_CUDA_OK(
        cudaMemcpyAsync(state.p, initial.p, state.bytes, cudaMemcpyDeviceToDevice, stream));
  }
  void chunked(cudaStream_t stream) {
    gdn_chunked_fwd(qkv.p, a.p, heads, b.p, heads, alog.as<float>(), dtb.as<float>(),
                    state.as<float>(), output.p, tokens, heads, kv_ratio, K, V, scale, workspace.p,
                    workspace.bytes, stream);
  }
  void recurrent(cudaStream_t stream) {
    gdn_recurrent_fwd(qkv.p, a.p, heads, b.p, heads, alog.as<float>(), dtb.as<float>(),
                      state.as<float>(), output.p, tokens, heads, kv_ratio, K, V, scale, stream);
  }
  Result result(cudaStream_t stream) {
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    Result r{std::vector<uint16_t>(output.bytes / 2), std::vector<float>(state.bytes / 4)};
    output.download(r.output.data(), output.bytes);
    state.download(r.state.data(), state.bytes);
    return r;
  }
};

void run(int tokens, int heads, int kv_ratio, bool time_it) {
  Problem p(tokens, heads, kv_ratio, 7 + tokens);
  Stream stream;
  p.reset(stream.s);
  p.recurrent(stream.s);
  const Result ref = p.result(stream.s);
  p.reset(stream.s);
  p.chunked(stream.s);
  const Result got = p.result(stream.s);
  double on2 = 0, od2 = 0, sn2 = 0, sd2 = 0;
  for (size_t i = 0; i < ref.output.size(); ++i) {
    const double r = fbf(ref.output[i]), c = fbf(got.output[i]);
    on2 += r * r;
    od2 += (r - c) * (r - c);
  }
  for (size_t i = 0; i < ref.state.size(); ++i) {
    sn2 += double(ref.state[i]) * ref.state[i];
    sd2 += (double(ref.state[i]) - got.state[i]) * (double(ref.state[i]) - got.state[i]);
  }
  const double eo = std::sqrt(od2 / on2), es = std::sqrt(sd2 / sn2);
  std::printf("[ .. ] tokens %d heads %d kv_ratio %d: out rel l2 %.4g, state rel l2 %.4g\n", tokens,
              heads, kv_ratio, eo, es);
  require(eo < 2e-2 && es < 2e-2, "chunked GDN exceeds bf16 tolerance");
  if (time_it) {
    cudaEvent_t e0, e1;
    DGPP_CUDA_OK(cudaEventCreate(&e0));
    DGPP_CUDA_OK(cudaEventCreate(&e1));
    auto time = [&](auto f) {
      f();
      DGPP_CUDA_OK(cudaEventRecord(e0, stream.s));
      for (int i = 0; i < 10; ++i) f();
      DGPP_CUDA_OK(cudaEventRecord(e1, stream.s));
      DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
      return ms / 10;
    };
    const float tr = time([&] { p.recurrent(stream.s); });
    const float tc = time([&] { p.chunked(stream.s); });
    DGPP_CUDA_OK(cudaEventDestroy(e0));
    DGPP_CUDA_OK(cudaEventDestroy(e1));
    std::printf("[ .. ] %d tokens x %d heads: recurrent %.3f ms, chunked %.3f ms (%.2fx)\n", tokens,
                heads, tr, tc, tr / tc);
  }
}

}  // namespace

DGPP_TEST(gdn_chunk_recurrence_parity) {
  run(1, 3, 3, false);
  run(63, 3, 3, false);
  run(64, 3, 3, false);
  run(65, 3, 3, false);
  run(300, 6, 3, false);
  run(1000, 6, 3, false);
  run(8192, 24, 3, true);
}

DGPP_TEST(gdn_chunk_concurrent_streams) {
  Problem a(8192, 3, 3, 123), b(8192, 3, 3, 456);
  Stream sa, sb;
  a.reset(sa.s);
  a.chunked(sa.s);
  const Result ref_a = a.result(sa.s);
  b.reset(sb.s);
  b.chunked(sb.s);
  const Result ref_b = b.result(sb.s);
  // All allocations precede the overlap: cudaMalloc/cudaFree must not
  // accidentally serialize the calls and hide workspace aliasing.
  for (int trial = 0; trial < 3; ++trial) {
    a.reset(sa.s);
    a.chunked(sa.s);
    b.reset(sb.s);
    b.chunked(sb.s);
    const Result got_a = a.result(sa.s), got_b = b.result(sb.s);
    require(got_a.output == ref_a.output && got_a.state == ref_a.state,
            "stream A differs from its serialized output/state");
    require(got_b.output == ref_b.output && got_b.state == ref_b.state,
            "stream B differs from its serialized output/state");
  }
}

DGPP_TEST(gdn_chunk_workspace_lifetime) {
  // Run under compute-sanitizer --leak-check full. Each worker releases its
  // own buffers before exit; a hidden TLS allocation would leak per worker.
  for (int trial = 0; trial < 3; ++trial) {
    std::exception_ptr error;
    std::thread worker([&] {
      try {
        run(8192, 24, 3, false);
      } catch (...) {
        error = std::current_exception();
      }
    });
    worker.join();
    if (error) std::rethrow_exception(error);
  }
}

DGPP_TEST(gdn_chunk_workspace_bounds) {
  Problem p(65, 3, 3, 123);
  auto rejected = [&](void* workspace, size_t bytes) {
    bool threw = false;
    try {
      gdn_chunked_fwd(p.qkv.p, p.a.p, p.heads, p.b.p, p.heads, p.alog.as<float>(),
                      p.dtb.as<float>(), p.state.as<float>(), p.output.p, p.tokens, p.heads,
                      p.kv_ratio, p.K, p.V, p.scale, workspace, bytes, nullptr);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "invalid workspace was accepted");
  };
  rejected(nullptr, p.workspace.bytes);
  rejected(p.workspace.p, p.workspace.bytes - 1);
  rejected(p.workspace.as<uint8_t>() + 1, p.workspace.bytes);
}

int main() {
  return dgpp::test::run_all();
}
