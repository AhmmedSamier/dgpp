// Nemotron-3.5-Transcribe weight loader (milestone 4). Builds the
// per-slot and global weight views from the checkpoint's safetensors,
// following the ResidentLayerStream lifecycle (loaders/resident_stream.hpp).
// Pure BF16 weights plus the Mamba F32 scalars; no packing, no TP yet
// (world must be 1 — the whole vocabulary and every expert live on rank 0).
// MTP draft block weights are loaded eagerly from llm.mtp.layers.*.

#include "models/nemotron_transcribe/loader.hpp"

#include <algorithm>
#include <cstdio>
#include <format>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/kernels.hpp"

namespace dgpp {
namespace {

std::string mixer_prefix(int slot) {
  return std::format("llm.model.layers.{}.mixer.", slot);
}

// Expected tensors for one LM slot (the slot's actual index — add_slots()
// uses the loop counter as the slot number, so it cannot be reused for a
// single arbitrary slot).
std::vector<NemotronExpectedTensor> slot_expected_tensors(
    const NemotronTranscribeLmConfig& lm, int slot) {
  const std::string p = mixer_prefix(slot);
  const int64_t H = lm.hidden_size;
  const auto mg = nemotron_mamba_geometry(lm);
  const int64_t q_rows = static_cast<int64_t>(lm.num_attention_heads) * lm.head_dim;
  const int64_t kv_rows = static_cast<int64_t>(lm.num_key_value_heads) * lm.head_dim;

  std::vector<NemotronExpectedTensor> out;
  out.push_back({std::format("llm.model.layers.{}.norm.weight", slot), DType::BF16, {H},
                 NemotronWeightClass::SlotNorm, slot});
  const NemotronSlot kind = lm.slots[slot];
  switch (kind) {
    case NemotronSlot::Mamba: {
      const std::string m = p;
      out.push_back({m + "in_proj.weight", DType::BF16, {mg.in_proj_rows, H},
                     NemotronWeightClass::MambaInProj, slot});
      out.push_back({m + "conv1d.weight", DType::BF16, {mg.conv_channels, 1, lm.conv_kernel},
                     NemotronWeightClass::MambaConvWeight, slot});
      if (lm.use_conv_bias)
        out.push_back({m + "conv1d.bias", DType::BF16, {mg.conv_channels},
                       NemotronWeightClass::MambaConvBias, slot});
      out.push_back({m + "_fp32_params.A_log", DType::F32, {lm.mamba_num_heads},
                     NemotronWeightClass::MambaAlog, slot});
      out.push_back({m + "_fp32_params.D", DType::F32, {lm.mamba_num_heads},
                     NemotronWeightClass::MambaD, slot});
      out.push_back({m + "_fp32_params.dt_bias", DType::F32, {lm.mamba_num_heads},
                     NemotronWeightClass::MambaDtBias, slot});
      out.push_back({m + "norm.weight", DType::BF16, {mg.d_inner},
                     NemotronWeightClass::MambaGatedNorm, slot});
      out.push_back({m + "out_proj.weight", DType::BF16, {H, mg.d_inner},
                     NemotronWeightClass::MambaOutProj, slot});
      break;
    }
    case NemotronSlot::Attention: {
      const std::string m = p;
      out.push_back({m + "q_proj.weight", DType::BF16, {q_rows, H},
                     NemotronWeightClass::AttnQ, slot});
      out.push_back({m + "k_proj.weight", DType::BF16, {kv_rows, H},
                     NemotronWeightClass::AttnK, slot});
      out.push_back({m + "v_proj.weight", DType::BF16, {kv_rows, H},
                     NemotronWeightClass::AttnV, slot});
      out.push_back({m + "o_proj.weight", DType::BF16, {H, q_rows},
                     NemotronWeightClass::AttnO, slot});
      break;
    }
    case NemotronSlot::Moe: {
      const std::string m = p;
      const int64_t E = lm.num_routed_experts, I = lm.moe_intermediate_size;
      const int64_t SI = lm.moe_shared_expert_intermediate_size;
      out.push_back({m + "gate.weight", DType::BF16, {E, H},
                     NemotronWeightClass::RouterGate, slot});
      out.push_back({m + "gate.e_score_correction_bias", DType::F32, {E},
                     NemotronWeightClass::RouterBias, slot});
      out.push_back({m + "experts.gate_and_up_projs", DType::BF16, {E, H, I},
                     NemotronWeightClass::ExpertsGateUp, slot});
      out.push_back({m + "experts.down_projs", DType::BF16, {E, I, H},
                     NemotronWeightClass::ExpertsDown, slot});
      out.push_back({m + "shared_experts.up_proj.weight", DType::BF16, {SI, H},
                     NemotronWeightClass::SharedUp, slot});
      out.push_back({m + "shared_experts.down_proj.weight", DType::BF16, {H, SI},
                     NemotronWeightClass::SharedDown, slot});
      break;
    }
  }
  return out;
}

}  // namespace

NemotronLocalGeometry NemotronLocalGeometry::from_config(
    const NemotronTranscribeConfig& cfg, int rank, int world,
    LoaderHeadSharding /*head*/) {
  return from_config(cfg.lm, rank, world, /*head=*/LoaderHeadSharding::Full);
}

NemotronLocalGeometry NemotronLocalGeometry::from_config(
    const NemotronTranscribeLmConfig& lm, int rank, int world,
    LoaderHeadSharding /*head*/) {
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("nemotron_transcribe loader: rank/world out of range");
  if (world > 1)
    throw std::runtime_error("nemotron_transcribe loader: TP (world > 1) is not yet implemented");
  NemotronLocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_inter = lm.moe_intermediate_size;
  g.local_shared_inter = lm.moe_shared_expert_intermediate_size;
  g.lm_vocab_begin = 0;
  g.lm_vocab_count = lm.vocab_size;
  g.embed_vocab_begin = 0;
  g.embed_vocab_count = lm.vocab_size;
  return g;
}

void NemotronLoaderFamily::validate_binding(const Config& c, const PresentMap& present) {
  const NemotronBindReport rep = nemotron_validate_binding(c, present);
  if (rep.ok) return;
  std::string msg = "nemotron_transcribe loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

std::vector<NemotronExpectedTensor> NemotronLoaderFamily::layer_table(
    const Config& c, int layer) {
  if (layer < 0 || layer >= c.lm.num_layers) return {};
  return slot_expected_tensors(c.lm, layer);
}

std::vector<NemotronExpectedTensor> NemotronLoaderFamily::global_table(
    const Config& c) {
  const auto& lm = c.lm;
  const int64_t H = lm.hidden_size, V = lm.vocab_size;
  std::vector<NemotronExpectedTensor> out;
  out.push_back({"llm.model.embed_tokens.weight", DType::BF16, {V, H},
                 NemotronWeightClass::Embed, -1});
  out.push_back({"llm.model.norm.weight", DType::BF16, {H},
                 NemotronWeightClass::FinalNorm, -1});
  out.push_back({"llm.lm_head.weight", DType::BF16, {V, H},
                 NemotronWeightClass::LmHead, -1});
  return out;
}

size_t NemotronLoaderFamily::globals_bytes(const Config& c, int rank, int world,
                                           LoaderHeadSharding /*head*/) {
  const NemotronLocalGeometry geo = NemotronLocalGeometry::from_config(c, rank, world,
                                                                        LoaderHeadSharding::Full);
  const size_t H = static_cast<size_t>(c.lm.hidden_size);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(geo.embed_vocab_count) * H * 2);  // embed
  b += align_up_256(H * 2);                                               // final norm
  b += align_up_256(static_cast<size_t>(geo.lm_vocab_count) * H * 2);     // lm_head
  return b;
}

void NemotronLoaderFamily::build_globals(const Config& c, const Geometry& geo,
                                         const LoaderTensorMap& tensors, LayerBump& bump,
                                         GlobalsResident& out, uint64_t& source_bytes,
                                         uint64_t& verbatim_bytes,
                                         LoaderHeadSharding /*head*/) {
  const size_t H = static_cast<size_t>(c.lm.hidden_size);

  const auto globals = global_table(c);
  std::unordered_map<std::string, const Expected*> by_name;
  for (const auto& e : globals) by_name.emplace(e.name, &e);

  std::vector<DequantJob> jobs;
  std::vector<PackJob> packs;
  // A concrete builder for the globals pass: every global is replicated.
  struct GlobalsBuilder : WeightBuilder<Expected> {
    bool replicated(const Expected&) const override { return true; }
    using WeightBuilder::WeightBuilder;
  } b(globals, by_name, bump, tensors, jobs, packs, true, geo.rank, geo.world,
      "nemotron_transcribe loader");

  out.embed = b.load_bf16_rows("llm.model.embed_tokens.weight", 0,
                               geo.embed_vocab_count);
  out.embed_vocab_begin = geo.embed_vocab_begin;
  out.embed_vocab_count = geo.embed_vocab_count;
  out.final_norm = b.load_bf16("llm.model.norm.weight");
  out.lm_head = b.load_bf16_rows("llm.lm_head.weight", 0, geo.lm_vocab_count);
  out.lm_vocab_begin = geo.lm_vocab_begin;
  out.lm_vocab_count = geo.lm_vocab_count;

  source_bytes = b.source_bytes;
  verbatim_bytes = b.verbatim_bytes;
  out.bytes = align_up_256(static_cast<size_t>(geo.embed_vocab_count) * H * 2) +
              align_up_256(H * 2) +
              align_up_256(static_cast<size_t>(geo.lm_vocab_count) * H * 2);

  DGPP_CUDA_OK(cudaStreamSynchronize(0));
  DGPP_LOG_INFO("nemotron_transcribe: globals resident, {} bytes", out.bytes);
}

// ---- the per-slot builder --------------------------------------------------

struct NemotronLoaderFamily::Builder : WeightBuilder<Expected> {
  const NemotronTranscribeConfig& cfg;
  const NemotronLocalGeometry& geo;
  NemotronLayerResident& out;

  Builder(const NemotronTranscribeConfig& cfg_, const NemotronLocalGeometry& geo_,
          const std::vector<Expected>& table_,
          const std::unordered_map<std::string, const Expected*>& by_name_,
          LayerBump& bump_, NemotronLayerResident& out_,
          const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<Expected>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_,
                                geo_.rank, geo_.world, "nemotron_transcribe loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const Expected&) const override { return true; }
  void build_layer(int layer);
};

void NemotronLoaderFamily::Builder::build_layer(int layer) {
  if (layer < 0 || layer >= main_layers(cfg))
    fail("layer index out of range: " + std::to_string(layer));
  const NemotronSlot kind = cfg.lm.slots[layer];
  const std::string p = mixer_prefix(layer);

  out.layer = layer;
  out.moe = (kind == NemotronSlot::Moe);
  out.input_norm = load_bf16(std::format("llm.model.layers.{}.norm.weight", layer));

  switch (kind) {
    case NemotronSlot::Mamba: {
      out.mamba_in_proj = load_bf16(p + "in_proj.weight");
      out.mamba_conv_weight = load_bf16(p + "conv1d.weight");
      if (cfg.lm.use_conv_bias)
        out.mamba_conv_bias = load_bf16(p + "conv1d.bias");
      out.mamba_A_log = load_f32(p + "_fp32_params.A_log");
      out.mamba_D = load_f32(p + "_fp32_params.D");
      out.mamba_dt_bias = load_f32(p + "_fp32_params.dt_bias");
      out.mamba_norm = load_bf16(p + "norm.weight");
      out.mamba_out_proj = load_bf16(p + "out_proj.weight");
      break;
    }
    case NemotronSlot::Attention: {
      out.attn_q = load_bf16(p + "q_proj.weight");
      out.attn_k = load_bf16(p + "k_proj.weight");
      out.attn_v = load_bf16(p + "v_proj.weight");
      out.attn_o = load_bf16(p + "o_proj.weight");
      break;
    }
    case NemotronSlot::Moe: {
      out.moe_gate = load_bf16(p + "gate.weight");
      out.moe_gate_bias = load_f32(p + "gate.e_score_correction_bias");
      out.moe_experts_gate_and_up = load_bf16(p + "experts.gate_and_up_projs");
      out.moe_experts_down = load_bf16(p + "experts.down_projs");
      out.moe_shared_up = load_bf16(p + "shared_experts.up_proj.weight");
      out.moe_shared_down = load_bf16(p + "shared_experts.down_proj.weight");
      break;
    }
  }
  out.bytes = bump.cursor;
}

template class ResidentLayerStream<NemotronLoaderFamily>;

// NemotronLayerStream implementation.
NemotronLayerStream::NemotronLayerStream(const NemotronTranscribeConfig& cfg,
                                         const std::string& checkpoint_dir, int rank, int world,
                                         NemotronResidency residency, NemotronHeadSharding head,
                                         bool resident_mtp)
    : ResidentLayerStream(cfg, checkpoint_dir, rank, world, residency, head, resident_mtp),
      cfg_(cfg) {}

void NemotronLayerStream::set_resident_image_dir(const std::string& /*dir*/) {}

const std::string& NemotronLayerStream::resident_image_dir() {
  static std::string dir;
  return dir;
}

const std::string& NemotronLayerStream::image_dir() const {
  static std::string dir;
  return dir;
}

}  // namespace dgpp
