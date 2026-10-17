#pragma once

#include <concepts>
#include <cstdint>
#include <vector>

namespace dgpp {

// A layer executor's mixed-row contract, independent of model identity and
// speculative decoding algorithm. Decode spans precede prefill spans in one
// walk. The executor must preserve each decoder's rollback rows, each reader's
// state and snapshots, and any features needed by its configured drafter.
// Device decode outputs must remain available to the graph's pick/commit/draft
// tail while prefill completion outputs are returned to the session layer.
// Merely having grouped prefill or a graph-preparation hook is insufficient.
template <class Model>
concept MixedPrefillModel = requires(
    Model& model, typename Model::PrefillCursor& cursor,
    const std::vector<typename Model::PrefillCursor*>& cursors,
    const std::vector<int64_t>& budgets, const std::vector<int>& decoders,
    const std::vector<std::vector<int64_t>>& feeds) {
  requires Model::kMixedPrefillVerify;
  { model.prepare_mixed_verify(1) } -> std::same_as<void>;
  { model.finish_mixed_prefill(cursor) } -> std::same_as<void>;
  { model.session_prefill_verify_group(cursors, budgets, decoders, feeds, 1) }
      -> std::same_as<std::vector<typename Model::Outputs>>;
};

}  // namespace dgpp
