#pragma once

#include <cstdint>
#include <stdexcept>

#include "kernels/glm_spec.hpp"

namespace dgpp {

// The graph's row index and a request's persistent slot are independent.
// Both session facades use the same feed gather, row map and rollback commit.
template <class Core>
struct GraphBatchMapping {
  static void inputs(Core& core, int requests, int rows, int feed_rows) {
    auto* feeds = core.d_tokens_ + core.max_decode_rows();
    core.step_tokens_ = feeds;
    if (core.graph_batch_map_source_)
      glm_upload_i32(core.graph_batch_map_source_, core.d_batch_map_, requests, core.stream_);
    if (feed_rows > rows || core.graph_batch_map_source_) {
      glm_spec_gather_feed(feeds, requests, feed_rows, rows, core.d_tokens_, core.stream_,
                          core.graph_batch_map_source_ ? core.d_batch_map_ : nullptr);
      core.step_tokens_ = core.d_tokens_;
    }
    for (int q = 0; q < requests; ++q) {
      core.h_req_spans_[2 * q] = q * rows;
      core.h_req_spans_[2 * q + 1] = rows;
      for (int r = 0; r < rows; ++r) core.h_req_ids_[q * rows + r] = q;
    }
    if (core.graph_batch_map_source_)
      glm_batch_rows(core.d_batch_map_, requests, rows, core.d_req_ids_, core.d_req_spans_, core.stream_);
    else {
      glm_upload_i32(core.h_req_ids_, core.d_req_ids_, requests * rows, core.stream_);
      glm_upload_i32(core.h_req_spans_, core.d_req_spans_, 2 * requests, core.stream_);
    }
    glm_spec_positions_batched(core.d_session_pos_, core.d_req_ids_, requests * rows, rows,
                               core.max_context_, core.d_step_pos_, core.stream_);
  }

  static void commit(Core& core, const PickVerdict* verdicts, int requests, int rows) {
    for (int q = 0; q < requests; ++q) {
      const int row0 = q * rows;
      if (!core.graph_batch_map_source_) {
        glm_spec_commit(verdicts + q, rows, core.derived().spec_segments(q, row0),
                        core.d_session_pos_ + q, core.stream_);
        continue;
      }
      if (core.max_requests_ < 2)
        throw std::logic_error("compact commit: stride requires at least two request slots");
      auto segments = core.derived().spec_segments(0, row0);
      const auto next = core.derived().spec_segments(1, row0);
      for (int i = 0; i < segments.count; ++i)
        segments.seg[i].request_stride_bytes = reinterpret_cast<uintptr_t>(next.seg[i].dst) -
                                               reinterpret_cast<uintptr_t>(segments.seg[i].dst);
      if (segments.replay_dst != nullptr)
        segments.replay_request_stride_bytes = reinterpret_cast<uintptr_t>(next.replay_dst) -
                                               reinterpret_cast<uintptr_t>(segments.replay_dst);
#ifndef NDEBUG
      for (int req = 0; req < core.max_requests_; ++req) {
        const auto actual = core.derived().spec_segments(req, row0);
        if (actual.count != segments.count)
          throw std::logic_error("compact commit: segment count differs by request");
        for (int i = 0; i < segments.count; ++i) {
          const auto expected = reinterpret_cast<uintptr_t>(segments.seg[i].dst) +
                                req * segments.seg[i].request_stride_bytes;
          if (reinterpret_cast<uintptr_t>(actual.seg[i].dst) != expected)
            throw std::logic_error("compact commit: nonuniform request destination stride");
        }
      }
#endif
      glm_spec_commit(verdicts + q, rows, segments, core.d_session_pos_, core.stream_, core.d_batch_map_, q);
    }
  }
};

}  // namespace dgpp
