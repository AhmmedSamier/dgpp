#pragma once
// The draft head (2026-10-13): every family's MTP, DFlash2 or DSpark draft
// ends in the vocabulary head, and that read is the same whichever model
// made the hidden row. This object owns the head screen
// (kernels/head_screen.hpp) for one rank's vocab slice in whichever
// encoding the family serves — bf16, its 12-bit companion, fp8 with block
// or channel scales, the int8 plane form — built once from the served
// head, and answers a draft's head call with the masked row (the candidate
// blocks' logits exact, -inf elsewhere) or declines so the caller runs its
// exact head. A family integrates it in three lines: the memory plan item,
// init() after its head is bound, and `if (draft && draft_head_.logits(...))`
// at its draft head site. The verify and prefill heads never see it.
//
// The setting is process-wide (engine.draft_head / engine.draft_head_blocks
// through draft_head_configure before any model is built), like the L2
// prefetcher's, so every family and every rank reads the same cap.
#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "kernels/bf12_gemv.hpp"
#include "kernels/head_screen.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

struct DraftHeadSettings {
  int blocks = 0;  // the candidate cap in 128-row blocks; 0 = every draft reads the exact head
};
void draft_head_configure(const DraftHeadSettings& s);
const DraftHeadSettings& draft_head_settings();

// A family's vocabulary head as served: one form set. bf12 is looked up at
// use (the companion is packed after construction, and the bf16 bytes may
// be released by then), so a caller passes the current pointer every call.
struct HeadMatrix {
  const uint16_t* bf16 = nullptr;        // [n][k]
  const Bf12Matrix* bf12 = nullptr;      // the 12-bit companion of `bf16` (nullable)
  const GlmQuantMatrix* fp8 = nullptr;   // block or channel scales
  const GlmPackedMatrix* packq = nullptr;  // int8 g128 planes
  int n = 0, k = 0;
  const char* form() const;
};

class DraftHead {
 public:
  DraftHead() = default;
  ~DraftHead();
  DraftHead(const DraftHead&) = delete;
  DraftHead& operator=(const DraftHead&) = delete;

  // The screen's device bytes for a head slice and a draft row ceiling
  // under the current setting (0 when the setting is off).
  static size_t plan_bytes(int n, int k, int max_rows);
  // Builds the screen from the served head when the setting is on and the
  // form is screenable (fp8, bf16, the int8 plane form); logs the reason
  // and returns false otherwise, leaving the head exact. `max_rows`: the
  // widest draft head call (the decode row ceiling).
  // Idempotent: a second call after the screen is built returns true at
  // once. A family calls it at construction and again after packing its
  // 12-bit companions: a head whose bf16 bytes are not device-resident
  // builds from the companion, so the first call (companion not packed yet)
  // declines quietly when `quiet` and the second one builds.
  bool init(const HeadMatrix& head, int max_rows, int rank, cudaStream_t stream, bool quiet = false);
  bool enabled() const { return blocks_ > 0; }
  int blocks() const { return blocks_; }
  size_t bytes() const { return bytes_; }
  // The masked draft logits of `rows` bf16 hidden rows (stride `stride`
  // elements) into out [rows][out_stride] (f32; out_stride 0 = n): true
  // when the screen took the call, false (nothing written) when it is off
  // or rows exceed the ceiling — the caller then runs its exact head. The
  // head's current pointers (the bf12 companion in particular) come with
  // each call. Kernel nodes only: capture-safe.
  bool logits(const HeadMatrix& head, const uint16_t* hidden, size_t stride, int rows, float* out,
              size_t out_stride, cudaStream_t stream);

 private:
  int blocks_ = 0;
  int max_rows_ = 0;
  size_t bytes_ = 0;
  uint8_t* storage_ = nullptr;
  uint8_t* scratch_storage_ = nullptr;
  HeadScreen screen_;
  HeadScreenScratch scratch_;
};

}  // namespace dgpp
