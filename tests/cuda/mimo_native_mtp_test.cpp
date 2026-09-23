// Native head state survives chunk boundaries, history wrap, slot reuse and
// prefix attachment. Distinct head weights must produce distinct proposals.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "common/bf16_residency.hpp"
#include "common/cuda_check.hpp"
#include "kernels/mimo_mtp_history.hpp"
#include "mimo_fixture.hpp"
#include "models/mimo/forward.hpp"
using namespace dgpp;
static void require(bool v, const char* why) {
  if (!v) throw std::runtime_error(why);
}
static std::vector<std::vector<float>> drafts(MimoModel& m, int req) {
  std::vector<std::vector<float>> out;
  out.push_back(m.session_draft(req, {7}).logits);
  out.push_back(m.session_draft_chain(req, 8, 0, true, false).logits);
  out.push_back(m.session_draft_chain(req, 9, 1, false, true).logits);
  require(out[0] != out[1] && out[1] != out[2], "distinct native heads collapsed");
  return out;
}
// CPU oracle for p-d translation, distinct slots, wrap and invalid padding.
static void history_oracle() {
  constexpr int C = 11, H = 7, T = 6;
  std::vector<uint16_t> history(2 * C * H);
  for (size_t i = 0; i < history.size(); ++i) history[i] = uint16_t(i + 1);
  int64_t positions[T] = {0, 1, 12, 21, -1, 23};
  int32_t ids[T] = {0, 1, 0, 1, 0, -1};
  uint16_t *dh = nullptr, *out = nullptr;
  int64_t *dp = nullptr, *shift = nullptr;
  int32_t* di = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&dh, history.size() * 2));
  DGPP_CUDA_OK(cudaMalloc(&out, T * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&dp, T * 8));
  DGPP_CUDA_OK(cudaMalloc(&shift, T * 8));
  DGPP_CUDA_OK(cudaMalloc(&di, T * 4));
  DGPP_CUDA_OK(cudaMemcpy(dh, history.data(), history.size() * 2, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(dp, positions, T * 8, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(di, ids, T * 4, cudaMemcpyHostToDevice));
  for (int d = 0; d < 3; ++d) {
    mimo_mtp_history_gather(dh, out, dp, shift, di, T, H, C, d, nullptr);
    std::vector<uint16_t> got(T * H);
    int64_t pos[T];
    DGPP_CUDA_OK(cudaMemcpy(got.data(), out, T * H * 2, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(pos, shift, T * 8, cudaMemcpyDeviceToHost));
    for (int r = 0; r < T; ++r) {
      const bool valid = ids[r] >= 0 && positions[r] >= d;
      require(pos[r] == (valid ? positions[r] - d : -1), "native head position oracle");
      for (int j = 0; j < H; ++j)
        require(
            got[r * H + j] == (valid ? history[(ids[r] * C + (positions[r] - d) % C) * H + j] : 0),
            "native head backbone input oracle");
    }
  }
  std::vector<uint16_t> values(T * H);
  for (size_t i = 0; i < values.size(); ++i) values[i] = uint16_t(1000 + i);
  DGPP_CUDA_OK(cudaMemcpy(out, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
  mimo_mtp_history_store(dh, out, dp, di, T, H, C, 0, nullptr);
  auto want = history;
  for (int r = 0; r < T; ++r)
    if (ids[r] >= 0 && positions[r] >= 0)
      for (int j = 0; j < H; ++j) want[(ids[r] * C + positions[r] % C) * H + j] = values[r * H + j];
  DGPP_CUDA_OK(cudaMemcpy(history.data(), dh, history.size() * 2, cudaMemcpyDeviceToHost));
  require(history == want, "native history store CPU oracle");
  cudaFree(dh);
  cudaFree(out);
  cudaFree(dp);
  cudaFree(shift);
  cudaFree(di);
  std::puts("PASS native head CPU offset/slot/wrap/padding oracle");
}
int main() {
  try {
    history_oracle();
    auto cfg = mimofx::tiny_config();
    cfg.mtp_layers_loaded = 3;
    const auto dir = (std::filesystem::current_path() / "mimo_native_fixture").string();
    mimofx::write_fixture(cfg, dir);
    for (auto residency : {Bf16Residency::Checkpoint, Bf16Residency::Bf12}) {
      set_bf16_residency(residency);
      for (auto format : {LatentFormat::kBf16, LatentFormat::kFp8}) {
        MimoModel m(cfg, dir, 64, 8192, MimoResidency::Resident, nullptr, 0, 1, 2, true, 8, format);
        m.session_graph_prepare();
        require(m.session_snapshot_bytes() == MimoModel::session_snapshot_bytes(cfg, 1, true),
                "snapshot plan mismatch");
        void* snap = nullptr;
        DGPP_CUDA_OK(cudaMalloc(&snap, m.session_snapshot_bytes()));
        for (int n : {2, 23, 97, 2111}) {
          std::vector<int64_t> prompt(n);
          for (int i = 0; i < n; ++i) prompt[i] = (i * 13 + 7) % cfg.vocab_size;
          m.session_prefill(0, prompt);
          auto meta = m.session_snapshot(0, snap);
          auto want = drafts(m, 0);
          m.session_draft_rollback(0, 1);
          require(drafts(m, 0) == want, "native eager draft rollback changed proposals");
          m.session_close(0);
          // Different prompt overwrites ring storage before attachment.
          m.session_prefill(1, {41, 42, 43, 44});
          m.session_close(1);
          m.session_attach(1, snap, meta);
          auto got = drafts(m, 1);
          require(got == want, "native draft logits changed after cross-slot prefix restore");
          m.session_close(1);
          m.session_release_snapshot(meta);
          m.session_prefill(0, prompt);
          require(drafts(m, 0) == want, "native draft logits changed after reopen");
          m.session_close(0);
          std::printf("PASS native n=%d residency=%d format=%d three-head restore/reopen\n", n,
                      int(residency), int(format));
        }
        cudaFree(snap);
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
