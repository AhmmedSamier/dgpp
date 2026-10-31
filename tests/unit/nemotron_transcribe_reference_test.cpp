// The Nemotron-Transcribe host fp32 oracle (docs/nemotron_transcribe_plan.md
// milestone 2) against the HF `nemotron_h` reference. The fixture —
// tests/python/nemotron_transcribe_ref_oracle.py --emit — is a tiny
// nemotron_h (every mixer kind twice, the MoE group policy live at 2/1,
// the dt clamp at 0.003) filled from a fixed seed in fp32, walked on CPU
// through the pure-torch fallback: the whole-model chain (residual in,
// slot-norm out, mixer out, logits) plus the primitives the walk hides
// (causal depthwise conv + silu, the multi-chunk selective scan, the gated
// norm, the bias-corrected grouped router). The oracle replays every one of
// those from the checkpoint-named weights and the numbers must agree to
// fp32 rounding — nothing fused, no kernel library, no GPU.
//
// The hand-computed cases at the bottom pin the corners the random fixture
// cannot: the dt clamp biting, the D skip on the pre-dt x, the gate before
// the group variance, the router's unbiased weights from a biased choice,
// and the bf16 dequant path the real checkpoint arrives through.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "models/nemotron_transcribe/config.hpp"
#include "models/nemotron_transcribe/reference.hpp"

namespace {

using dgpp::minijson::Value;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string fixture_path() {
#ifdef DGPP_SOURCE_DIR
  return std::string(DGPP_SOURCE_DIR) + "/tests/data/nemotron_transcribe_reference_oracle.json";
#else
  return {};
#endif
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::vector<float> vec(const Value& a) {
  std::vector<float> out;
  out.reserve(a.items().size());
  for (const auto& v : a.items()) out.push_back(static_cast<float>(v.as_double()));
  return out;
}

std::vector<std::vector<int64_t>> index_grid(const Value& a) {
  std::vector<std::vector<int64_t>> out;
  for (const auto& row : a.items()) {
    std::vector<int64_t> r;
    for (const auto& v : row.items()) r.push_back(v.as_int());
    out.push_back(std::move(r));
  }
  return out;
}

std::vector<std::vector<float>> float_grid(const Value& a) {
  std::vector<std::vector<float>> out;
  for (const auto& row : a.items()) out.push_back(vec(row));
  return out;
}

void close(const std::vector<float>& got, const std::vector<float>& want, double rel,
           double abs_tol, const std::string& what) {
  require(got.size() == want.size(), what + ": size " + std::to_string(got.size()) + " vs " +
                                          std::to_string(want.size()));
  double worst = 0.0;
  size_t at = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    const double lim = abs_tol + rel * std::abs(static_cast<double>(want[i]));
    if (d / lim > worst) { worst = d / lim; at = i; }
  }
  if (worst > 1.0) {
    throw std::runtime_error(what + ": worst rel/abs at element " + std::to_string(at) + " got " +
                             std::to_string(got[at]) + " want " + std::to_string(want[at]) +
                             " (" + std::to_string(worst) + "x budget)");
  }
}

void near(double got, double want, double tol, const std::string& what) {
  if (!(std::abs(got - want) <= tol))
    throw std::runtime_error(what + ": got " + std::to_string(got) + ", want " +
                             std::to_string(want));
}

dgpp::nemotron_ref::Tensor f32(const std::vector<float>& v) {
  dgpp::nemotron_ref::Tensor T;
  T = v;
  return T;
}

// The oracle's fp32-vs-fp32 budget: both sides sum fp32, different orders,
// ~1e-6 relative per contraction; the walk chains six mixers and the head.
constexpr double kRel = 5e-4, kAbs = 1e-5;

}  // namespace

DGPP_TEST(nemotron_oracle_walk_matches_the_hf_reference) {
  const std::string path = fixture_path();
  if (path.empty() || !std::filesystem::exists(path)) {
    dgpp::test::skip("nemotron reference fixture absent");
    return;
  }
  const std::string text = read_file(path);
  const auto doc = dgpp::minijson::parse_document(text);
  const Value& root = doc.root;
  const Value& cfgj = root.at("config");

  dgpp::NemotronTranscribeLmConfig lm;
  lm.hidden_size = static_cast<int>(cfgj.at("hidden_size").as_int());
  lm.vocab_size = static_cast<int>(cfgj.at("vocab_size").as_int());
  lm.num_attention_heads = static_cast<int>(cfgj.at("num_attention_heads").as_int());
  lm.num_key_value_heads = static_cast<int>(cfgj.at("num_key_value_heads").as_int());
  lm.head_dim = static_cast<int>(cfgj.at("head_dim").as_int());
  lm.rope_theta = cfgj.at("rope_theta").as_double();
  lm.partial_rotary_factor = 1.0;
  lm.mamba_num_heads = static_cast<int>(cfgj.at("mamba_num_heads").as_int());
  lm.mamba_head_dim = static_cast<int>(cfgj.at("mamba_head_dim").as_int());
  lm.mamba_num_groups = static_cast<int>(cfgj.at("n_groups").as_int());
  lm.mamba_state_size = static_cast<int>(cfgj.at("ssm_state_size").as_int());
  lm.conv_kernel = static_cast<int>(cfgj.at("conv_kernel").as_int());
  lm.use_conv_bias = true;
  lm.mamba_hidden_act = "silu";
  lm.time_step_min = cfgj.at("time_step_min").as_double();
  lm.num_routed_experts = static_cast<int>(cfgj.at("n_routed_experts").as_int());
  lm.num_experts_per_tok = static_cast<int>(cfgj.at("num_experts_per_tok").as_int());
  lm.moe_intermediate_size = static_cast<int>(cfgj.at("moe_intermediate_size").as_int());
  lm.moe_shared_expert_intermediate_size =
      static_cast<int>(cfgj.at("moe_shared_expert_intermediate_size").as_int());
  lm.routed_scaling_factor = cfgj.at("routed_scaling_factor").as_double();
  lm.moe_n_groups = static_cast<int>(cfgj.at("n_group").as_int());
  lm.moe_topk_group = static_cast<int>(cfgj.at("topk_group").as_int());
  lm.norm_topk_prob = true;
  lm.mlp_hidden_act = "relu2";
  lm.norm_eps = cfgj.at("norm_eps").as_double();
  lm.num_layers = static_cast<int>(cfgj.at("layers_block_type").items().size());
  for (const auto& s : cfgj.at("layers_block_type").items()) {
    const std::string_view name = s.as_string();
    lm.slots.push_back(name == "mamba"          ? dgpp::NemotronSlot::Mamba
                      : name == "attention"     ? dgpp::NemotronSlot::Attention
                      : name == "linear_attention" ? dgpp::NemotronSlot::Mamba
                      : name == "full_attention"   ? dgpp::NemotronSlot::Attention
                      : dgpp::NemotronSlot::Moe);
  }

  dgpp::nemotron_ref::Weights w;
  const Value& wj = root.at("weights");
  require(wj.is_object(), "weights block");
  for (const auto& m : wj.members()) {
    dgpp::nemotron_ref::Tensor T;
    T = vec(m.value.at("data"));
    w.emplace(m.key, std::move(T));
  }

  std::vector<uint32_t> ids;
  for (const auto& v : root.at("ids").items()) ids.push_back(static_cast<uint32_t>(v.as_int()));
  const auto res = dgpp::nemotron_ref::walk(lm, w, ids.data(), ids.size(),
                                            /*record=*/true, /*last_logits_only=*/false);

  const Value& walk = root.at("walk");
  const Value& rec = walk.at("rec");
  const size_t H = lm.hidden_size;
  for (size_t i = 0; i < lm.slots.size(); ++i) {
    const std::string p = "llm.model.layers." + std::to_string(i) + ".";
    close(res.normed[i], vec(rec.at(p + "norm_out").at("data")), kRel, kAbs, p + "norm_out");
    close(res.mixer_out[i], vec(rec.at(p + "mixer_out").at("data")), kRel, kAbs, p + "mixer_out");
    require(res.mixer_out[i].size() == ids.size() * H, "mixer row count");
    if (lm.slots[i] == dgpp::NemotronSlot::Moe) {
      const Value& rj = rec.at(p + "mixer.router");
      const auto want_idx = index_grid(rj.at("index"));
      const auto want_w = float_grid(rj.at("weight"));
      for (size_t t = 0; t < ids.size(); ++t) {
        auto sel = dgpp::nemotron_ref::router_select(
            res.normed[i].data() + t * H, w.at(p + "mixer.gate.weight"),
            w.at(p + "mixer.gate.e_score_correction_bias").f32.data(), lm.num_routed_experts, H,
            lm.num_experts_per_tok, lm.moe_n_groups, lm.moe_topk_group, lm.norm_topk_prob,
            static_cast<float>(lm.routed_scaling_factor));
        for (size_t k = 0; k < want_idx[t].size(); ++k)
          require(sel.index[k] == static_cast<size_t>(want_idx[t][k]),
                  p + "router index row " + std::to_string(t) + " pos " + std::to_string(k));
        std::vector<float> got_w(sel.weight);
        close(got_w, want_w[t], kRel, kAbs, p + "router weight row " + std::to_string(t));
      }
    }
  }
  close(res.logits, vec(walk.at("logits").at("data")), kRel, kAbs, "logits");
}

DGPP_TEST(nemotron_oracle_primitives_match_the_hf_reference) {
  const std::string path = fixture_path();
  if (path.empty() || !std::filesystem::exists(path)) {
    dgpp::test::skip("nemotron reference fixture absent");
    return;
  }
  const std::string text = read_file(path);
  const auto doc = dgpp::minijson::parse_document(text);
  const Value& cases = doc.root.at("cases");

  {  // causal depthwise conv + silu
    const Value& c = cases.at("conv");
    const size_t seq = static_cast<size_t>(c.at("seq").as_int());
    const size_t ch = static_cast<size_t>(c.at("channels").as_int());
    const size_t kern = static_cast<size_t>(c.at("kernel").as_int());
    std::vector<float> u = vec(c.at("u"));
    std::vector<float> out(seq * ch, 0.f);
    const auto bias = vec(c.at("bias"));
    dgpp::nemotron_ref::Tensor wt, bt;
    wt = vec(c.at("weight"));
    bt = bias;
    dgpp::nemotron_ref::causal_conv_silu(u.data(), seq, ch, wt, &bt, kern, out.data());
    close(out, vec(c.at("out")), kRel, kAbs, "conv");
  }
  {  // the selective scan, multi-chunk
    const Value& c = cases.at("scan");
    dgpp::nemotron_ref::MambaGeom g;
    g.heads = static_cast<size_t>(c.at("heads").as_int());
    g.head_dim = static_cast<size_t>(c.at("head_dim").as_int());
    g.groups = static_cast<size_t>(c.at("groups").as_int());
    g.state = static_cast<size_t>(c.at("state").as_int());
    g.dt_min = static_cast<float>(c.at("dt_min").as_double());
    const size_t seq = static_cast<size_t>(c.at("seq").as_int());
    std::vector<float> x = vec(c.at("x")), dt = vec(c.at("dt")), B = vec(c.at("B")),
                         C = vec(c.at("C")), A_log = vec(c.at("A_log")), D = vec(c.at("D")),
                         dt_bias = vec(c.at("dt_bias"));
    std::vector<float> out(seq * g.heads * g.head_dim, 0.f);
    dgpp::nemotron_ref::mamba2_walk(g, x.data(), B.data(), C.data(), dt.data(), A_log.data(),
                                    D.data(), dt_bias.data(), seq, out.data());
    close(out, vec(c.at("out")), kRel, kAbs, "scan");
  }
  {  // the gated norm, row by row
    const Value& c = cases.at("gated_norm");
    const size_t rows = static_cast<size_t>(c.at("rows").as_int());
    const size_t n = static_cast<size_t>(c.at("n").as_int());
    const size_t group = static_cast<size_t>(c.at("group").as_int());
    const float eps = static_cast<float>(c.at("eps").as_double());
    auto y = vec(c.at("y")), gate = vec(c.at("gate"));
    dgpp::nemotron_ref::Tensor wgt = f32(vec(c.at("weight")));
    std::vector<float> out(rows * n, 0.f);
    for (size_t r = 0; r < rows; ++r)
      dgpp::nemotron_ref::rmsnorm_gated(y.data() + r * n, gate.data() + r * n, wgt, n, group, eps,
                                        out.data() + r * n);
    close(out, vec(c.at("out")), kRel, kAbs, "gated_norm");
  }
  {  // the router: group policy live, weights from the unbiased sigmoid
    const Value& c = cases.at("router");
    const size_t E = static_cast<size_t>(c.at("E").as_int());
    const size_t H = static_cast<size_t>(c.at("H").as_int());
    const size_t topk = static_cast<size_t>(c.at("topk").as_int());
    auto x = vec(c.at("x")), bias = vec(c.at("bias"));
    dgpp::nemotron_ref::Tensor gate = f32(vec(c.at("gate")));
    const auto want_idx = index_grid(c.at("index"));
    const auto want_w = float_grid(c.at("weight"));
    const auto want_logits = vec(c.at("logits"));
    for (size_t r = 0; r < want_idx.size(); ++r) {
      auto sel = dgpp::nemotron_ref::router_select(x.data() + r * H, gate, bias.data(), E, H, topk,
                                                   2, 1, true, 2.5f);
      std::vector<float> logits(E);
      dgpp::nemotron_ref::gemv(logits.data(), gate, x.data() + r * H, E, H);
      close(logits, {want_logits.begin() + r * E, want_logits.begin() + (r + 1) * E}, kRel, kAbs,
            "router logits row " + std::to_string(r));
      for (size_t k = 0; k < topk; ++k)
        require(sel.index[k] == static_cast<size_t>(want_idx[r][k]),
                "router index row " + std::to_string(r) + " pos " + std::to_string(k));
      close(sel.weight, want_w[r], kRel, kAbs, "router weight row " + std::to_string(r));
    }
  }
}

DGPP_TEST(nemotron_oracle_hand_computed_corners) {
  using namespace dgpp::nemotron_ref;

  // dt clamp + the D skip on the pre-dt x, one head, one lane, one state.
  // A_log = 0 -> A = -1; dt raw -50 -> softplus ~0 -> clamped to dt_min.
  {
    MambaGeom g;
    g.heads = 1; g.head_dim = 1; g.groups = 1; g.state = 1; g.dt_min = 0.001f;
    const float x[2] = {2.f, 0.f}, B[2] = {3.f, 0.f}, C[2] = {1.f, 1.f};
    const float dt[2] = {-50.f, -50.f}, A_log[1] = {0.f}, D[1] = {0.5f}, dt_bias[1] = {0.f};
    float out[2] = {};
    mamba2_walk(g, x, B, C, dt, A_log, D, dt_bias, 2, out);
    const float d = 0.001f, dA = std::exp(-d);
    // t0: state = d*B*x = 0.006; out = state*C + D*x (the new state emits).
    near(out[0], 0.006f + 1.0f, 1e-6, "clamped dt step");
    // t1: x = 0: the state only decays, but it still emits through C.
    near(out[1], 0.006f * dA, 1e-6, "decayed state still emits, D skip on the pre-dt x");
  }

  // Gated norm: the gate acts BEFORE the variance (a gate of 2 on one lane
  // changes that group's rms — norm_before_gate would not).
  {
    const float y[2] = {1.f, 1.f}, gate[2] = {0.f, 3.f};  // silu(0)=0, silu(3)=2.858
    Tensor w;
    w = std::vector<float>{1.f, 1.f};
    float out[2];
    rmsnorm_gated(y, gate, w, 2, 2, 1e-5f, out);
    near(out[0], 0.0, 1e-9, "silu(0) gates the lane out");
    const double t1 = 1.0 * (3.0 / (1.0 + std::exp(-3.0)));
    const double rms = std::sqrt((t1 * t1) / 2.0 + 1e-5);
    near(out[1], static_cast<float>(t1 / rms), 1e-5, "gate before the variance");
  }

  // The router: the biased choice takes the biased expert, the weight stays
  // the unbiased sigmoid, renormalized and scaled.
  {
    Tensor gate;
    std::vector<float> g(2 * 1, 0.f);  // E = 2, H = 1
    g[0] = 4.f; g[1] = -4.f;           // logits from x = 1
    gate = g;
    const float bias[2] = {0.f, 3.f};  // lifts expert 1 past expert 0
    const float x[1] = {1.f};
    auto sel = router_select(x, gate, bias, 2, 1, 2, 1, 1, true, 2.5f);
    require(sel.index[0] == 0 && sel.index[1] == 1, "both chosen, canonical order");
    const double s0 = 1.0 / (1.0 + std::exp(-4.0)), s1 = 1.0 / (1.0 + std::exp(4.0));
    near(sel.weight[0], static_cast<float>(s0 / (s0 + s1) * 2.5), 1e-5, "unbiased weight 0");
    near(sel.weight[1], static_cast<float>(s1 / (s0 + s1) * 2.5), 1e-5, "biased pick, unbiased weight 1");
  }

  // The bf16 dequant path (the checkpoint's dtype) and a hand relu2 MLP.
  {
    Tensor W;
    W = std::vector<uint16_t>{dgpp::float_to_bf16_bits(1.f), dgpp::float_to_bf16_bits(2.f),
                              dgpp::float_to_bf16_bits(3.f), dgpp::float_to_bf16_bits(4.f)};
    const float x[2] = {1.f, 1.f};
    float y[2];
    gemv(y, W, x, 2, 2);
    near(y[0], 3.0, 1e-6, "bf16 gemv row 0");
    near(y[1], 7.0, 1e-6, "bf16 gemv row 1");
    Tensor up, down;
    up = std::vector<float>{-1.f, 0.f, 2.f, 0.f};   // [I=2, H=2]
    down = std::vector<float>{1.f, 1.f, 1.f, 1.f};  // [H=2, I=2]
    const float xin[2] = {1.f, 1.f};
    float o[2];
    relu2_mlp(xin, up, down, 2, 2, o);
    // relu2([-1, 2]) = [0, 4]; down sums -> 4 each.
    near(o[0], 4.0, 1e-6, "relu2 mlp lane 0");
    near(o[1], 4.0, 1e-6, "relu2 mlp lane 1");
  }
}
