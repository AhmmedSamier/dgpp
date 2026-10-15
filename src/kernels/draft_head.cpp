#include "kernels/draft_head.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace {

DraftHeadSettings g_settings;
constexpr uint64_t kRotationSeed = 0x6472616674736372ull;  // the rotation's signs: fixed, every rank alike

}  // namespace

void draft_head_configure(const DraftHeadSettings& s) {
  if (s.blocks < 0 || s.blocks > kHeadScreenMaxBlocks)
    throw std::invalid_argument("draft_head_configure: blocks must be in [0, " + std::to_string(kHeadScreenMaxBlocks) + "]");
  g_settings = s;
}
const DraftHeadSettings& draft_head_settings() { return g_settings; }

const char* HeadMatrix::form() const {
  if (fp8 != nullptr && fp8->payload != nullptr) return "fp8";
  if (packq != nullptr && packq->packed != nullptr) return "int8 planes";
  if (bf12 != nullptr && bf12->packed != nullptr) return "bf12";
  if (bf16 != nullptr) return "bf16";
  return "none";
}

DraftHead::~DraftHead() {
  cudaFree(storage_);
  cudaFree(scratch_storage_);
}

size_t DraftHead::plan_bytes(int n, int k, int max_rows) {
  const int blocks = g_settings.blocks;
  if (blocks <= 0 || n <= 0 || k <= 0 || k % kHeadScreenBlock != 0 || k > kHeadScreenMaxK) return 0;
  const int nb = (n + kHeadScreenBlock - 1) / kHeadScreenBlock;
  return head_screen_bytes(n, k) + head_screen_scratch_bytes(std::max(1, max_rows), k, nb, blocks);
}

bool DraftHead::init(const HeadMatrix& head, int max_rows, int rank, cudaStream_t stream, bool quiet) {
  const int blocks = g_settings.blocks;
  if (blocks <= 0) return false;
  if (blocks_ > 0) return true;  // built already
  const int n = head.n, k = head.k;
  const bool fp8 = head.fp8 != nullptr && head.fp8->payload != nullptr;
  const bool packq = head.packq != nullptr && head.packq->packed != nullptr;
  const bool bf12 = head.bf12 != nullptr && head.bf12->packed != nullptr;
  // The bf16 build reads the rows on the device: a residency that keeps only
  // the 12-bit companion resident leaves the bf16 pointer off the device (or
  // released), and the build then comes from the companion.
  bool bf16_dev = false;
  if (!fp8 && !packq && head.bf16 != nullptr) {
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, head.bf16) == cudaSuccess)
      bf16_dev = attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
    else
      (void)cudaGetLastError();
  }
  std::string why;
  if (n <= 0 || k <= 0) why = "no head bound";
  else if (k % kHeadScreenBlock != 0 || k > kHeadScreenMaxK)
    why = "the hidden width is not a multiple of 128 up to " + std::to_string(kHeadScreenMaxK);
  else if (packq && !head_screen_packq_supported(*head.packq))
    why = "the packed head is not the int8 g128 plane form";
  else if (!fp8 && !packq && !bf16_dev && !bf12) {
    if (quiet) return false;  // the companion is not packed yet; the caller asks again after packing
    why = head.bf16 == nullptr ? "the head's form has no screen build"
                               : "the bf16 head is not device-resident and no 12-bit companion is packed";
  } else if (!fp8 && !packq && !bf16_dev && bf12 && !head_screen_bf12_supported(*head.bf12))
    why = "the 12-bit companion carries a tail (k is not a whole number of 1024-column super-blocks)";
  if (!why.empty()) {
    DGPP_LOG_WARN("rank {} draft head screen: off for this model ({}); the drafts read the exact head", rank, why);
    return false;
  }
  if (max_rows < 1) max_rows = 1;
  const auto t0 = std::chrono::steady_clock::now();
  const size_t screen_bytes = head_screen_bytes(n, k);
  DGPP_CUDA_OK(cudaMalloc(&storage_, screen_bytes));
  screen_ = head_screen_carve(storage_, n, k);
  const size_t scratch_bytes = head_screen_scratch_bytes(max_rows, k, screen_.nb, blocks);
  DGPP_CUDA_OK(cudaMalloc(&scratch_storage_, scratch_bytes));
  scratch_ = head_screen_scratch_carve(scratch_storage_, max_rows, k, screen_.nb, blocks);
  const char* built_from = head.form();
  if (fp8)
    head_screen_build_fp8(screen_, *head.fp8, kRotationSeed, stream);
  else if (packq)
    head_screen_build_packq(screen_, *head.packq, kRotationSeed, stream);
  else if (bf16_dev) {
    head_screen_build_bf16(screen_, head.bf16, kRotationSeed, stream);
    built_from = "bf16";
  } else {
    head_screen_build_bf12(screen_, *head.bf12, kRotationSeed, stream);
    built_from = "bf12";
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  blocks_ = blocks;
  max_rows_ = max_rows;
  bytes_ = screen_bytes + scratch_bytes;
  DGPP_LOG_INFO("rank {} draft head screen: {} x {} in 2 bits ({:.1f} MB, {} blocks; cap {} blocks per draft, up to {} "
                "rows; {} form) built in {:.0f} ms",
                rank, n, k, static_cast<double>(screen_.nb) * (k / 16) * 128 * 4 / 1e6, screen_.nb, blocks, max_rows,
                built_from, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  return true;
}

bool DraftHead::logits(const HeadMatrix& head, const uint16_t* hidden, size_t stride, int rows, float* out,
                       size_t out_stride, cudaStream_t stream) {
  if (blocks_ <= 0 || rows <= 0 || rows > max_rows_) return false;
  if (head.n != screen_.n || head.k != screen_.k)
    throw std::invalid_argument("DraftHead::logits: the head's shape is not the screen's");
  if (out_stride == 0) out_stride = static_cast<size_t>(screen_.n);
  launch_head_screen(screen_, scratch_, hidden, stride, rows, kHeadScreenDelta, stream);
  if (head.fp8 != nullptr && head.fp8->payload != nullptr)
    launch_head_rescore_fp8(screen_, scratch_, *head.fp8, hidden, stride, rows, out, out_stride, stream);
  else if (head.packq != nullptr && head.packq->packed != nullptr)
    launch_head_rescore_packq(screen_, scratch_, *head.packq, hidden, stride, rows, out, out_stride, stream);
  else if (head.bf12 != nullptr && head.bf12->packed != nullptr)
    launch_head_rescore_bf12(screen_, scratch_, *head.bf12, hidden, stride, rows, out, out_stride, stream);
  else if (head.bf16 != nullptr)
    launch_head_rescore_bf16(screen_, scratch_, head.bf16, hidden, stride, rows, out, out_stride, stream);
  else
    throw std::invalid_argument("DraftHead::logits: the head's form has no rescore");
  return true;
}

}  // namespace dgpp
