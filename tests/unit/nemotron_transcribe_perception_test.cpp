// The Nemotron-Transcribe perception oracle (docs/nemotron_transcribe_plan.md
// milestone 5) against a CPU-torch mirror of the NeMo chain. The fixture —
// tests/python/nemotron_transcribe_perception_oracle.py --emit — runs a tiny
// geometry with every structural choice of the release live: the eval-gated
// dither and 0.97 preemphasis, the center-padded 512-framing on the shipped
// (random, here; real, at the checkpoint) window and filterbank tensors,
// the ×8 frame-major zero-tailed stacking into the bias-free pre-encode,
// the pre-norm blocks with the per-head q/k LayerNORM BEFORE the half RoPE
// (pairs within the rotated 64, the tail pass-through), bidirectional
// attention at 1/sqrt(hd) into the BIASED out_proj, the erf-GELU FFN, and
// the fusion: thresholded activity (strict >), LayerNorm(n_spk), the
// built-in sinusoidal kernel (sin on the EVEN columns), the scale on the
// infusion only (0.7, so mis-scaling cannot hide), LayerNorm(d_model) and
// the connector.
//
// Sortformer's own branch (its 380 tensors) is 5b — the activity is this
// oracle's input, exactly as ParallelExpertEncoder._fuse_diar_and_asr
// accepts external speaker targets.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "models/nemotron_transcribe/perception.hpp"

namespace {

using dgpp::minijson::Value;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string fixture_path() {
#ifdef DGPP_SOURCE_DIR
  return std::string(DGPP_SOURCE_DIR) + "/tests/data/nemotron_transcribe_perception_oracle.json";
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

void close(const std::vector<float>& got, const std::vector<float>& want, double rel,
           double abs_tol, const std::string& what) {
  require(got.size() == want.size(),
          what + ": size " + std::to_string(got.size()) + " vs " + std::to_string(want.size()));
  double worst = 0.0;
  size_t at = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    const double lim = abs_tol + rel * std::abs(static_cast<double>(want[i]));
    if (d / lim > worst) { worst = d / lim; at = i; }
  }
  if (worst > 1.0)
    throw std::runtime_error(what + ": worst at " + std::to_string(at) + " got " +
                             std::to_string(got[at]) + " want " + std::to_string(want[at]) +
                             " (" + std::to_string(worst) + "x budget)");
}

dgpp::nemotron_ref::Tensor f32(const std::vector<float>& v) {
  dgpp::nemotron_ref::Tensor T;
  T = v;
  return T;
}

constexpr double kRel = 5e-4, kAbs = 1e-5;

}  // namespace

DGPP_TEST(nemotron_perc_oracle_matches_the_nemo_mirror) {
  const std::string path = fixture_path();
  if (path.empty() || !std::filesystem::exists(path)) {
    dgpp::test::skip("nemotron perception fixture absent");
    return;
  }
  const std::string text = read_file(path);
  const auto doc = dgpp::minijson::parse_document(text);
  const Value& root = doc.root;
  const Value& cfg = root.at("config");

  dgpp::NemotronTranscribePerceptionConfig pc;
  pc.mel_features = static_cast<int>(cfg.at("mel_features").as_int());
  pc.fft_size = static_cast<int>(cfg.at("n_fft").as_int());
  pc.window_size = static_cast<int>(cfg.at("win").as_int());
  pc.window_stride = static_cast<int>(cfg.at("hop").as_int());
  pc.subsampling_factor = static_cast<int>(cfg.at("subsampling").as_int());
  pc.enc_d_model = static_cast<int>(cfg.at("d_model").as_int());
  pc.enc_heads = static_cast<int>(cfg.at("heads").as_int());
  pc.enc_layers = static_cast<int>(cfg.at("layers").as_int());
  pc.enc_ff_expansion = static_cast<int>(cfg.at("ff_expansion").as_int());
  pc.enc_qk_norm = true;
  pc.enc_qkv_bias = false;
  pc.enc_pre_block_norm = true;
  pc.enc_rope_base = cfg.at("rope_base").as_double();
  pc.enc_rotary_fraction = cfg.at("rotary_fraction").as_double();
  pc.n_spk = static_cast<int>(cfg.at("n_spk").as_int());
  pc.speaker_feature_thresholded = true;
  pc.speaker_activity_threshold = cfg.at("threshold").as_double();
  pc.spk_kernel_scale = cfg.at("spk_kernel_scale").as_double();
  pc.output_dim = static_cast<int>(cfg.at("output_dim").as_int());
  pc.connector_d_model = pc.enc_d_model;

  dgpp::nemotron_ref::Weights w;
  const Value& wj = root.at("weights");
  for (const auto& m : wj.members()) w.emplace(m.key, f32(vec(m.value.at("data"))));

  for (const auto& c : root.at("cases").items()) {
    const auto audio = vec(c.at("audio"));
    const auto activity = vec(c.at("activity"));
    const auto out = dgpp::nemotron_perc::perceive(pc, w, audio.data(), audio.size(),
                                                   activity.data());
    const std::string tag = "n=" + std::to_string(audio.size());
    require(out.frames == static_cast<size_t>(c.at("mel_frames").items()[1].as_int()),
            tag + ": frame count");
    require(out.tokens * static_cast<size_t>(pc.enc_d_model) == out.states.size(),
            tag + ": state rows");
    close(out.mel, vec(c.at("mel")), kRel, kAbs, tag + " mel");
    close(out.states, vec(c.at("states")), kRel, kAbs, tag + " states");
    close(out.rows, vec(c.at("rows")), kRel, kAbs, tag + " rows");
  }
}

DGPP_TEST(nemotron_perc_oracle_hand_computed_corners) {
  using namespace dgpp::nemotron_perc;
  using dgpp::nemotron_ref::Tensor;
  using dgpp::nemotron_ref::Weights;

  // LayerNorm over 4 lanes, (w, b) identity.
  {
    const float x[4] = {1.f, 2.f, 3.f, 4.f};
    Tensor w = Tensor(); w = std::vector<float>{1, 1, 1, 1};
    Tensor b = Tensor(); b = std::vector<float>{0, 0, 0, 0};
    float y[4];
    layernorm_row(x, w, b, 4, 1e-5f, y);
    const double inv = 1.0 / std::sqrt(1.25 + 1e-5);
    for (int i = 0; i < 4; ++i)
      require(std::abs(y[i] - (i + 1 - 2.5f) * inv) < 1e-6f, "layernorm lane");
  }

  // rfft512 of a delta at position 3: X[k] = e^{-2 pi i 3 k / 512}.
  {
    std::vector<float> frame(512, 0.f);
    frame[3] = 1.f;
    std::vector<float> re(257), im(257);
    rfft512(frame.data(), re.data(), im.data());
    for (size_t k = 0; k < 257; ++k) {
      const double ang = -2.0 * M_PI * 3.0 * static_cast<double>(k) / 512.0;
      require(std::abs(re[k] - std::cos(ang)) < 2e-5, "delta rfft re bin " + std::to_string(k));
      require(std::abs(im[k] - std::sin(ang)) < 2e-5, "delta rfft im bin " + std::to_string(k));
    }
  }

  // The speaker kernel: sin on EVEN columns, cos on odd; s=0 pins
  // (0, 1) pairs; s=2 column 0 is sin(2 * 10000^0) = sin(2).
  {
    dgpp::NemotronTranscribePerceptionConfig pc;
    pc.n_spk = 4;
    pc.enc_d_model = 64;
    std::vector<float> k;
    build_spk_kernel(pc, k);
    require(k[0] == 0.f && k[1] == 1.f, "speaker row 0 is (sin0, cos0)");
    require(std::abs(k[2 * 64 + 0] - std::sin(2.0)) < 1e-6, "speaker sin on the even column");
    require(std::abs(k[2 * 64 + 1] - std::cos(2.0)) < 1e-6, "speaker cos on the odd column");
    const double d = 10000.0, col = 3 * 2;  // column 6 -> div 10000^(-6/64)
    const double div = std::pow(d, -col / 64.0);
    require(std::abs(k[1 * 64 + 6] - std::sin(1.0 * div)) < 1e-6, "speaker frequency ladder");
  }

  // Threshold semantics: STRICT greater-than (a == threshold is silence).
  {
    dgpp::NemotronTranscribePerceptionConfig pc;
    pc.mel_features = 1;  // unused here
    pc.enc_d_model = 2;
    pc.enc_heads = 1;
    pc.enc_ff_expansion = 1;
    pc.subsampling_factor = 1;
    pc.n_spk = 2;
    pc.speaker_feature_thresholded = true;
    pc.speaker_activity_threshold = 0.5;
    pc.spk_kernel_scale = 1.0;
    pc.output_dim = 2;
    pc.connector_d_model = 2;
    Weights w;
    Tensor t;
    t = std::vector<float>{1, 1};
    w["perception.encoder.diar_norm.weight"] = t;
    t = std::vector<float>{0, 0};
    w["perception.encoder.diar_norm.bias"] = t;
    t = std::vector<float>{1, 1};
    w["perception.encoder.asr_norm.weight"] = t;
    t = std::vector<float>{0, 0};
    w["perception.encoder.asr_norm.bias"] = t;
    t = std::vector<float>{1, 0, 0, 1};
    w["perception.proj.weight"] = t;
    t = std::vector<float>{0, 0};
    w["perception.proj.bias"] = t;
    const float states[2] = {3.f, -3.f};
    const float activity[2] = {0.5f, 0.5001f};  // lane 0 on the dot: silent (STRICT >)
    std::vector<float> rows;
    fuse_and_connect(pc, w, states, activity, 1, rows);
    // thresholds: (0, 1); LN(8-wide=2): (-1, +1); kernel rows: s=0 (0, 1),
    // s=1 (sin 1, cos 1) (div_term[0] = 1 for d_model 2 -> exponent 0).
    // asr LN of (3, -3): (1, -1). rows = asr_ln + 1.0 * (dn @ kernel).
    const double s1 = std::sin(1.0), c1 = std::cos(1.0);
    require(rows.size() == 2, "fusion row width");
    require(std::abs(rows[0] - (1.0 + (-0.0 + s1))) < 1e-4, "strict > on lane 0 (silence)");
    require(std::abs(rows[1] - (-1.0 + (-1.0 + c1))) < 1e-4, "strict > on lane 1 (voice)");
  }
}
