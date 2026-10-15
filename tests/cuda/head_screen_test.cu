// The draft head's screen (kernels/head_screen.hpp): the build's codes
// and the prep's int8 rows are the host rotation's bit for bit; the
// screen's block maxima and the select's candidate lists are the host's
// bit for bit from the device's codes and rows; the rescore writes the
// production row chain's logits on the candidate blocks (fp64 tolerance;
// the bf12 form bitwise the bf16 one) and -inf elsewhere; a planted peak
// is always found; rows chunk across launches (17 rows); a partial last
// block masks its missing columns.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf12_gemv.hpp"
#include "kernels/head_screen.hpp"
#include "kernels/packq_head.hpp"
#include "models/quant_matrix.hpp"

namespace {

void require(bool c, const std::string& what) {
  if (!c) throw std::runtime_error(what);
}

template <typename T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&d, v.size() * sizeof(T) + 256));
  DGPP_CUDA_OK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <typename T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  DGPP_CUDA_OK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}

float e4m3_to_float(uint8_t c) {
  const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
  float v;
  if (e == 0) v = std::ldexp(static_cast<float>(m), -9);  // subnormal: m / 8 * 2^-6
  else v = std::ldexp(1.f + static_cast<float>(m) / 8.f, e - 7);
  return s ? -v : v;
}

struct Head {
  int n = 0, k = 0;
  std::vector<uint16_t> bf16;            // the bf16 head [n][k]
  std::vector<uint8_t> fp8;              // e4m3 payload [n][k]
  std::vector<float> scales;             // [n/128 ceil][k/128]
  std::vector<float> served_fp8;         // bf16(fp8 * scale) as fp32, the dequant bridge's value
  std::vector<uint8_t> int8_planes;      // the int8 g128 head in the plane layout (packq_head.hpp)
  std::vector<uint16_t> int8_scales;     // f16 [n][k/128]
  std::vector<float> served_int8;        // (code - 128) * scale as fp32
};

Head make_head(uint64_t seed, int n, int k) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.f, 0.02f);
  std::uniform_real_distribution<float> sd(0.004f, 0.012f);
  Head h;
  h.n = n;
  h.k = k;
  h.bf16.resize(static_cast<size_t>(n) * k);
  for (auto& w : h.bf16) w = dgpp::float_to_bf16_bits(nd(rng));
  h.fp8.resize(static_cast<size_t>(n) * k);
  const int sr = (n + 127) / 128, sc = k / 128;
  h.scales.resize(static_cast<size_t>(sr) * sc);
  for (auto& s : h.scales) s = sd(rng);
  for (auto& c : h.fp8) {
    uint8_t v = static_cast<uint8_t>(rng() & 0xFF);
    if ((v & 0x7F) == 0x7F) v &= 0x7E;  // no NaN codes
    c = v;
  }
  h.served_fp8.resize(h.fp8.size());
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < k; ++c) {
      const float s = h.scales[static_cast<size_t>(r / 128) * sc + c / 128];
      h.served_fp8[static_cast<size_t>(r) * k + c] =
          dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(e4m3_to_float(h.fp8[static_cast<size_t>(r) * k + c]) * s));
    }
  // The int8 g128 head: codes around 128 with signed f16 group scales (the
  // AutoRound head's form), permuted into the plane layout as the loader does.
  std::normal_distribution<float> cd(0.f, 40.f);
  std::uniform_real_distribution<float> isd(2e-4f, 6e-4f);
  h.int8_planes.resize(static_cast<size_t>(n) * k);
  h.int8_scales.resize(static_cast<size_t>(n) * (k / 128));
  h.served_int8.resize(static_cast<size_t>(n) * k);
  for (auto& c : h.int8_planes) c = static_cast<uint8_t>(std::min(255.f, std::max(0.f, std::round(cd(rng)) + 128.f)));
  for (auto& s : h.int8_scales) s = dgpp::float_to_fp16_bits(isd(rng) * ((rng() & 1u) ? 1.f : -1.f));
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < k; ++c) {
      const float s = dgpp::fp16_bits_to_float(h.int8_scales[static_cast<size_t>(r) * (k / 128) + c / 128]);
      h.served_int8[static_cast<size_t>(r) * k + c] = (static_cast<float>(h.int8_planes[static_cast<size_t>(r) * k + c]) - 128.f) * s;
    }
  dgpp::packq_planes_permute_int8(h.int8_planes.data(), n, k);
  return h;
}

// The device's rotation on one fp32 row, in place: the same stage order and
// (e & s) rule, plain adds, then the scale.
void rotate_host(std::vector<float>& x, const std::vector<int8_t>& sign) {
  const int k = static_cast<int>(x.size());
  for (int e = 0; e < k; ++e) x[e] *= static_cast<float>(sign[e]);
  std::vector<float> p(x.size());
  for (int s = 1; s < 128; s <<= 1) {
    for (int e = 0; e < k; ++e) p[e] = x[e ^ s];
    for (int e = 0; e < k; ++e) {
      const float v = x[e];
      x[e] = (e & s) ? p[e] - v : v + p[e];
    }
  }
  for (int e = 0; e < k; ++e) x[e] *= 0.088388347648318f;
}

struct Case {
  int n, k, rows, cap;
  bool peak;
};

void run_case(const Case& cs, uint64_t seed) {
  const int n = cs.n, k = cs.k, rows = cs.rows, cap = cs.cap;
  const int nb = (n + 127) / 128;
  std::printf("[ .. ] n %d k %d rows %d cap %d %s\n", n, k, rows, cap, cs.peak ? "planted peaks" : "random rows");
  Head h = make_head(seed, n, k);
  std::mt19937_64 rng(seed ^ 0x9E37u);
  std::normal_distribution<float> nd(0.f, 0.5f);
  std::vector<float> noise(static_cast<size_t>(rows) * k);
  for (auto& v : noise) v = nd(rng);
  std::vector<int> planted(rows, -1);
  if (cs.peak)
    for (int r = 0; r < rows; ++r) planted[r] = static_cast<int>(rng() % static_cast<uint64_t>(n));
  uint16_t* d_bf16 = upload(h.bf16);
  uint8_t* d_fp8 = upload(h.fp8);
  float* d_scales = upload(h.scales);
  dgpp::GlmQuantMatrix q;
  q.payload = d_fp8;
  q.scales = d_scales;
  q.rows = n;
  q.cols = k;
  uint8_t* d_planes = upload(h.int8_planes);
  uint16_t* d_iscales = upload(h.int8_scales);
  dgpp::GlmPackedMatrix pq;
  pq.packed = reinterpret_cast<const uint32_t*>(d_planes);
  pq.scales = d_iscales;
  pq.rows = n;
  pq.cols = k;
  pq.bits = 8;
  pq.scale_fmt = dgpp::kPackedScaleF16G128;
  pq.layout = dgpp::kPackedLayoutPlanes8;
  const bool packq_ok = dgpp::head_screen_packq_supported(pq);

  const std::vector<float> served_bf16 = [&] {
    std::vector<float> v(h.bf16.size());
    for (size_t i = 0; i < v.size(); ++i) v[i] = dgpp::bf16_bits_to_float(h.bf16[i]);
    return v;
  }();
  for (int form = 0; form < 3; ++form) {  // 0: the fp8 head, 1: the bf16 head (and its bf12 twin), 2: the int8 planes
    if (form == 2 && !packq_ok) {
      std::printf("[ .. ] int8 plane form: K %d outside the packed core's compiled set, skipped\n", k);
      continue;
    }
    const std::vector<float>& served = form == 0 ? h.served_fp8 : form == 1 ? served_bf16 : h.served_int8;
    // The activation rows for this form: noise, plus 300 x the planted
    // column's served row (its logit then leads the rest by many sigmas).
    std::vector<uint16_t> act(static_cast<size_t>(rows) * k);
    std::vector<float> actf(act.size());
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < k; ++c) {
        float v = noise[static_cast<size_t>(r) * k + c];
        if (cs.peak) v += 300.f * served[static_cast<size_t>(planted[r]) * k + c];
        act[static_cast<size_t>(r) * k + c] = dgpp::float_to_bf16_bits(v);
        actf[static_cast<size_t>(r) * k + c] = dgpp::bf16_bits_to_float(act[static_cast<size_t>(r) * k + c]);
      }
    uint16_t* d_act = upload(act);
    void* storage = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&storage, dgpp::head_screen_bytes(n, k)));
    dgpp::HeadScreen s = dgpp::head_screen_carve(storage, n, k);
    const uint64_t rot_seed = 17;
    if (form == 0) dgpp::head_screen_build_fp8(s, q, rot_seed, nullptr);
    else if (form == 1) dgpp::head_screen_build_bf16(s, d_bf16, rot_seed, nullptr);
    else dgpp::head_screen_build_packq(s, pq, rot_seed, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    // ---- the build against the host rotation --------------------------------
    std::vector<int8_t> sign(static_cast<size_t>(k));
    dgpp::head_screen_signs(rot_seed, k, sign.data());
    require(download(s.sign, static_cast<size_t>(k)) == sign, "the device holds the host's signs");
    const std::vector<float> step = download(s.step, static_cast<size_t>(n));
    const std::vector<float> rms_blk = download(s.rms_blk, static_cast<size_t>(nb));
    const std::vector<uint32_t> codes = download(s.codes, static_cast<size_t>(nb) * (k / 16) * 128);
    std::vector<uint8_t> host_codes(static_cast<size_t>(n) * k);
    std::vector<float> host_rms(static_cast<size_t>(n));
    size_t code_mismatch = 0;
    double step_err = 0;
    for (int r = 0; r < n; ++r) {
      std::vector<float> x(served.begin() + static_cast<size_t>(r) * k, served.begin() + static_cast<size_t>(r + 1) * k);
      rotate_host(x, sign);
      double ss = 0, sm = 0;
      for (float v : x) { ss += static_cast<double>(v) * v; sm += v; }
      const double mean = sm / k, var = std::max(ss / k - mean * mean, 0.0);
      float st = 0.9957f * static_cast<float>(std::sqrt(var));
      if (!(st > 0.f)) st = 1.f;
      step_err = std::max(step_err, std::fabs(static_cast<double>(st) - step[r]) / std::max(1e-12, static_cast<double>(st)));
      const float dst = step[r], inv = 1.f / dst;  // the device's step: the codes must then be bit for bit
      double rr = 0;
      for (int c = 0; c < k; ++c) {
        const float v = x[c];
        const float code = std::min(std::max(std::floor(std::fma(v, inv, 2.f)), 0.f), 3.f);
        host_codes[static_cast<size_t>(r) * k + c] = static_cast<uint8_t>(code);
        const float qv = (code - 1.5f) * dst;
        rr += static_cast<double>(v - qv) * (v - qv);
      }
      host_rms[r] = static_cast<float>(std::sqrt(rr / k));
      const int b = r / 128, cc = r % 128, kc = k / 16;
      for (int g = 0; g < kc; ++g) {
        uint32_t w = 0;
        for (int j = 0; j < 4; ++j)
          for (int L = 0; L < 4; ++L) w |= static_cast<uint32_t>(host_codes[static_cast<size_t>(r) * k + 16 * g + 4 * j + L]) << (8 * L + 2 * j);
        if (codes[(static_cast<size_t>(b) * kc + g) * 128 + cc] != w) ++code_mismatch;
      }
    }
    require(step_err < 2e-5, "the column steps match the host's within fp32 reduction error (" + std::to_string(step_err) + ")");
    require(code_mismatch == 0, "the codes are the host's bit for bit (" + std::to_string(code_mismatch) + " words differ)");
    for (int b = 0; b < nb; ++b) {
      float m = 0.f;
      for (int r = b * 128; r < std::min(n, b * 128 + 128); ++r) m = std::max(m, host_rms[r]);
      require(std::fabs(m - rms_blk[b]) <= 1e-4f * m + 1e-7f, "the block rounding rms matches the host's");
    }
    // ---- prep + screen + select ---------------------------------------------
    void* scratch = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&scratch, dgpp::head_screen_scratch_bytes(rows, k, nb, cap)));
    dgpp::HeadScreenScratch sc = dgpp::head_screen_scratch_carve(scratch, rows, k, nb, cap);
    dgpp::launch_head_screen(s, sc, d_act, static_cast<size_t>(k), rows, dgpp::kHeadScreenDelta, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<int8_t> xq = download(sc.xq, static_cast<size_t>(rows) * k);
    const std::vector<float> xmeta = download(sc.xmeta, static_cast<size_t>(rows) * 4);
    const std::vector<float> bmax = download(sc.bmax, static_cast<size_t>(rows) * nb);
    const std::vector<int32_t> cand = download(sc.cand, static_cast<size_t>(rows) * cap);
    std::vector<std::vector<int>> host_cand(rows);
    for (int r = 0; r < rows; ++r) {
      std::vector<float> x(actf.begin() + static_cast<size_t>(r) * k, actf.begin() + static_cast<size_t>(r + 1) * k);
      rotate_host(x, sign);
      float amax = 0.f;
      double ss = 0;
      for (float v : x) { amax = std::max(amax, std::fabs(v)); ss += static_cast<double>(v) * v; }
      float scale = amax / 127.f;
      if (!(scale > 0.f)) scale = 1.f;
      require(std::fabs(scale - xmeta[r * 4 + 0]) <= 1e-6f * scale, "the row scale matches");
      const float dscale = xmeta[r * 4 + 0], inv = 1.f / dscale;
      int sum = 0;
      for (int c = 0; c < k; ++c) {
        const int qv = std::max(-127, std::min(127, static_cast<int>(std::nearbyint(x[c] * inv))));
        require(qv == xq[static_cast<size_t>(r) * k + c], "the int8 row is the host's bit for bit");
        sum += qv;
      }
      require(static_cast<int>(xmeta[r * 4 + 1]) == sum, "the row's code sum matches");
      require(std::fabs(xmeta[r * 4 + 2] - static_cast<float>(std::sqrt(ss))) <= 1e-4f * static_cast<float>(std::sqrt(ss)), "|x|_2 matches");
      // block maxima from the DEVICE's codes and row: integer dots, then the kernel's float formula
      std::vector<float> hb(nb, -INFINITY);
      for (int col = 0; col < n; ++col) {
        const int b = col / 128, cc = col % 128, kc = k / 16;
        long long acc = 0;
        for (int g = 0; g < kc; ++g) {
          const uint32_t w = codes[(static_cast<size_t>(b) * kc + g) * 128 + cc];
          for (int j = 0; j < 4; ++j)
            for (int L = 0; L < 4; ++L) {
              const int code = static_cast<int>((w >> (8 * L + 2 * j)) & 3u);
              acc += static_cast<long long>(xq[static_cast<size_t>(r) * k + 16 * g + 4 * j + L]) * (2 * code);
            }
        }
        const float v = ((dscale * step[col]) * 0.5f) * static_cast<float>(acc - 3LL * sum);
        hb[b] = std::max(hb[b], v);
      }
      for (int b = 0; b < nb; ++b)
        require(hb[b] == bmax[static_cast<size_t>(r) * nb + b], "block maxima are the host's bit for bit (row " + std::to_string(r) + " block " + std::to_string(b) + ")");
      // the select
      const float xnorm = xmeta[r * 4 + 2];
      int best = 0;
      for (int b = 1; b < nb; ++b) if (hb[b] > hb[best]) best = b;
      const float base = hb[best] - dgpp::kHeadScreenDelta * (xnorm * rms_blk[best]);
      std::vector<int> taken(nb, 0);
      for (int c = 0; c < cap; ++c) {
        int pick = -1;
        for (int b = 0; b < nb; ++b)
          if (!taken[b] && hb[b] >= base - dgpp::kHeadScreenDelta * (xnorm * rms_blk[b]) && (pick < 0 || hb[b] > hb[pick])) pick = b;
        if (pick < 0) break;
        taken[pick] = 1;
        host_cand[r].push_back(pick);
      }
      for (int c = 0; c < cap; ++c) {
        const int want = c < static_cast<int>(host_cand[r].size()) ? host_cand[r][c] : -1;
        require(cand[static_cast<size_t>(r) * cap + c] == want, "candidate lists are the host's (row " + std::to_string(r) + ")");
      }
    }
    // ---- the rescore ---------------------------------------------------------
    float* d_out = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&d_out, static_cast<size_t>(rows) * n * 4));
    DGPP_CUDA_OK(cudaMemset(d_out, 0x7F, static_cast<size_t>(rows) * n * 4));  // NaN fill: every column must be written
    if (form == 0) dgpp::launch_head_rescore_fp8(s, sc, q, d_act, static_cast<size_t>(k), rows, d_out, 0, nullptr);
    else if (form == 1) dgpp::launch_head_rescore_bf16(s, sc, d_bf16, d_act, static_cast<size_t>(k), rows, d_out, 0, nullptr);
    else dgpp::launch_head_rescore_packq(s, sc, pq, d_act, static_cast<size_t>(k), rows, d_out, 0, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> out = download(d_out, static_cast<size_t>(rows) * n);
    int found = 0, agree = 0;
    for (int r = 0; r < rows; ++r) {
      std::vector<char> in_cand(nb, 0);
      for (int b : host_cand[r]) in_cand[b] = 1;
      double exact_best = -1e300;
      int exact_arg = -1;
      double masked_best = -1e300;
      int masked_arg = -1;
      double maxabs = 0;
      for (int col = 0; col < n; ++col) {
        double d = 0;
        for (int c = 0; c < k; ++c) d += static_cast<double>(served[static_cast<size_t>(col) * k + c]) * actf[static_cast<size_t>(r) * k + c];
        maxabs = std::max(maxabs, std::fabs(d));
        if (d > exact_best) { exact_best = d; exact_arg = col; }
        const float got = out[static_cast<size_t>(r) * n + col];
        if (in_cand[col / 128]) {
          require(std::isfinite(got), "a candidate column's logit is finite");
          require(std::fabs(got - d) <= 2e-4 * std::max(1.0, maxabs) + 1e-5, "a candidate logit matches fp64 (row " + std::to_string(r) + " col " + std::to_string(col) + ": " + std::to_string(got) + " vs " + std::to_string(d) + ")");
          if (got > masked_best) { masked_best = got; masked_arg = col; }
        } else {
          require(got == -INFINITY, "a non-candidate column is -inf (row " + std::to_string(r) + " col " + std::to_string(col) + ")");
        }
      }
      if (cs.peak) require(exact_arg == planted[r], "the planted peak is the exact argmax");
      if (masked_arg == exact_arg) ++agree;
      if (cs.peak && masked_arg == planted[r]) ++found;
    }
    if (cs.peak) require(found == rows, "every planted peak is found through the screen");
    std::printf("[ OK ] form %s: build, prep, screen, select bitwise the host; rescore within fp64 tolerance; argmax agreement %d/%d\n",
                form == 0 ? "fp8" : form == 1 ? "bf16" : "int8 planes", agree, rows);
    if (form == 1) {
      // The bf12 twin: lossless, so the rescore is the bf16 one bit for bit.
      dgpp::Bf12Host packed = dgpp::bf12_encode(h.bf16.data(), n, k);
      require(packed.ok, "the bf16 head packs to bf12");
      dgpp::Bf12Matrix m;
      m.packed = upload(packed.packed);
      m.rows = upload(packed.rows);
      m.esc = upload(packed.esc);
      m.raw = upload(packed.raw);
      m.n = n;
      m.k = k;
      float* d_out2 = nullptr;
      DGPP_CUDA_OK(cudaMalloc(&d_out2, static_cast<size_t>(rows) * n * 4));
      dgpp::launch_head_rescore_bf12(s, sc, m, d_act, static_cast<size_t>(k), rows, d_out2, 0, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> out2 = download(d_out2, static_cast<size_t>(rows) * n);
      require(std::memcmp(out.data(), out2.data(), out.size() * 4) == 0, "the bf12 rescore is the bf16 one bit for bit");
      std::printf("[ OK ] form bf12: the rescore is the bf16 chain bit for bit (%zu escapes, %d raw rows)\n", packed.escapes, packed.raw_rows);
      if (dgpp::head_screen_bf12_supported(m)) {
        // The build from the companion: the same values, so the same codes, steps and block rms.
        void* storage2 = nullptr;
        DGPP_CUDA_OK(cudaMalloc(&storage2, dgpp::head_screen_bytes(n, k)));
        dgpp::HeadScreen s2 = dgpp::head_screen_carve(storage2, n, k);
        dgpp::head_screen_build_bf12(s2, m, rot_seed, nullptr);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        require(download(s2.codes, codes.size()) == codes, "the bf12 build's codes are the bf16 build's bit for bit");
        require(download(s2.step, static_cast<size_t>(n)) == step, "the bf12 build's steps are the bf16 build's");
        require(download(s2.rms_blk, static_cast<size_t>(nb)) == rms_blk, "the bf12 build's block rms are the bf16 build's");
        std::printf("[ OK ] form bf12: the screen built from the companion is the bf16 build bit for bit\n");
        DGPP_CUDA_OK(cudaFree(storage2));
      } else {
        std::printf("[ .. ] form bf12: k %d carries a tail, the companion build is not offered there\n", k);
      }
      DGPP_CUDA_OK(cudaFree(const_cast<uint8_t*>(m.packed)));
      DGPP_CUDA_OK(cudaFree(const_cast<uint32_t*>(m.rows)));
      DGPP_CUDA_OK(cudaFree(const_cast<uint32_t*>(m.esc)));
      DGPP_CUDA_OK(cudaFree(const_cast<uint16_t*>(m.raw)));
      DGPP_CUDA_OK(cudaFree(d_out2));
    }
    DGPP_CUDA_OK(cudaFree(d_out));
    DGPP_CUDA_OK(cudaFree(scratch));
    DGPP_CUDA_OK(cudaFree(storage));
    DGPP_CUDA_OK(cudaFree(d_act));
  }
  DGPP_CUDA_OK(cudaFree(d_bf16));
  DGPP_CUDA_OK(cudaFree(d_fp8));
  DGPP_CUDA_OK(cudaFree(d_scales));
  DGPP_CUDA_OK(cudaFree(d_planes));
  DGPP_CUDA_OK(cudaFree(d_iscales));
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("SKIP: no CUDA device\n");
    return 2;
  }
  try {
    run_case({2048, 2560, 1, 4, true}, 1);
    run_case({2048, 2560, 8, 4, true}, 2);
    run_case({2048, 2560, 17, 4, true}, 3);   // two screen chunks (16 + 1)
    run_case({2048, 2560, 3, 32, false}, 4);  // random rows: every block a candidate at this cap
    run_case({1000, 256, 5, 3, true}, 5);     // a partial last block, the smallest hidden
    run_case({1024, 5120, 4, 8, true}, 6);    // the Qwen3.8-27B width
    run_case({512, 6144, 9, 4, true}, 7);     // the full GLM-5.3 width: 8 rows would stage exactly 48 KB (4-row chunks)
    run_case({512, 7168, 9, 4, true}, 8);     // a wider head (4-row screen chunks)
    std::printf("[ OK ] head_screen_test\n");
    return 0;
  } catch (const std::exception& e) {
    std::printf("[FAIL] %s\n", e.what());
    return 1;
  }
}
