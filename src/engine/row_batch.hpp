#pragma once
#include <algorithm>
#include <cstdint>
#include <numeric>

namespace dgpp {
// One walk over request-owned spans.
struct RowRun {
  int req = 0;                   // the request (scalar); 0 for the fixed batch
  const int64_t* ids = nullptr;  // host [T] (prefill; eager decode validation)
  int T = 0;
  int64_t pos0 = 0;              // prefill: the chunk's first position
  bool decode = false;           // decode rows: metadata staged by decode_host_prep
  bool all_rows = false;         // every row's logits (else the last row; decode: all rows)
  bool capture_layers = false;
  bool capture = false;          // graph capture: no syncs, no copies, nothing executes
  int batch_requests = 0;        // > 0: the fixed slot-major batch (decode)
  bool snapshots = false;        // decode: leave every row's post-state in the spec rows
  // Prefill: the call's first and last chunk (a family with a replay
  // segment — DeepSeek-V4.1's bounded decoder — carries a tail between
  // the chunks of one call and runs its segment on the last).
  bool first_chunk = true;
  bool last_chunk = true;
  // A row batch: several requests' continuations as the spans of
  // one walk (host arrays of num_spans; 0: the scalar form above). T is
  // the spans' total and `ids` their prompts concatenated.
  const int32_t* span_reqs = nullptr;
  const int64_t* span_pos0 = nullptr;
  const int32_t* span_lens = nullptr;
  int num_spans = 0;
  // Optional host mask: one last-row output per prefill span (or one for
  // a scalar prefill). Serving cursors only need it at prompt completion.
  const uint8_t* output_spans = nullptr;
  // Mixed walks put speculative decode spans first, followed by prefill
  // spans. Decode spans retain every row's logits and rollback state;
  // prefill spans retain their last row. Only families implementing this
  // contract may set this field.
  const uint8_t* span_first_chunk = nullptr;
  const uint8_t* span_last_chunk = nullptr;
  int decode_spans = 0;
  int decode_rows() const {
    return decode_spans ? std::accumulate(span_lens, span_lens + decode_spans, 0) : 0;
  }

  // Preserve the decoder prefix and request-owned short-tail recipes.
  // A layer may combine adjacent readers only when its arithmetic is
  // invariant under that grouping. The visitor receives the source offset,
  // row count, decode flag and first source span.
  template <class Compatible, class Visit>
  void for_each_kernel_group(Compatible&& compatible, Visit&& visit) const {
    if (!num_spans) { visit(0, T, decode, 0); return; }
    int row0 = 0;
    const int prefix = decode_rows();
    if (prefix) {
      visit(0, prefix, true, 0);
      row0 = prefix;
    }
    for (int span = decode_spans; span < num_spans;) {
      const int first = span;
      int count = span_lens[span++];
      while (span < num_spans && compatible(count, span_lens[span]))
        count += span_lens[span++];
      visit(row0, count, false, first);
      row0 += count;
    }
  }

  // Keep selected reader spans separate while retaining the existing wide
  // batch around them. The executor must support its prefill arithmetic on
  // any decoder rows included in a combined group. Request state still
  // follows the original spans; these groups select projection shapes only.
  template <class Isolate, class Visit>
  void for_each_isolated_reader_group(Isolate&& isolate, Visit&& visit) const {
    if (!num_spans) { visit(0, T, decode, 0); return; }
    const int prefix = decode_rows();
    int row0 = 0, count = prefix, first = 0, at = prefix;
    const auto flush = [&] {
      if (count) visit(row0, count, prefix > 0 && first == 0 && count == prefix, first);
      count = 0;
    };
    for (int span = decode_spans; span < num_spans; ++span) {
      const int rows = span_lens[span];
      if (isolate(span)) {
        flush();
        visit(at, rows, false, span);
      } else {
        if (!count) { row0 = at; first = span; }
        count += rows;
      }
      at += rows;
    }
    flush();
  }

  int output_rows() const {
    return decode || all_rows ? T : decode_rows() + std::max(1, num_spans) - decode_spans;
  }
  // Selected source rows and their stable, packed output positions. Holes
  // correspond to unfinished readers whose logits are not needed yet.
  template <class Visit>
  void for_each_output(Visit&& visit) const {
    if (decode || all_rows) { visit(0, 0, T); return; }
    const int prefix = decode_rows();
    if (prefix) visit(0, 0, prefix);
    int end = prefix, output = prefix;
    for (int span = decode_spans; span < std::max(1, num_spans); ++span, ++output) {
      end += num_spans ? span_lens[span] : T;
      if (!output_spans || output_spans[span]) visit(end - 1, output, 1);
    }
  }

  // After gathering, adjacent selected outputs can share a projection or
  // transfer even when their source rows belonged to different readers.
  template <class Visit>
  void for_each_output_range(Visit&& visit) const {
    int first = 0, count = 0;
    for_each_output([&](int, int output, int rows) {
      if (count && output != first + count) { visit(first, count); count = 0; }
      if (!count) first = output;
      count += rows;
    });
    if (count) visit(first, count);
  }

  bool device_decode_outputs = false;  // mixed verify is consumed by a device pick
  // The scalar prefill's following chunk (host ids, rows, first
  // position; next_T 0 when this is the last), for a family that can
  // stage the next chunk's inputs while this one runs.
  const int64_t* next_ids = nullptr;
  int next_T = 0;
  int64_t next_pos0 = 0;
};

}  // namespace dgpp
