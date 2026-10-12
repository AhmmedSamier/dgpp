#include "serve/audio_inputs.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio/mel_frontend.hpp"
#include "audio/wav_decode.hpp"
#include "common/audio_input.hpp"
#include "common/base64.hpp"
#include "common/test.hpp"
#include "loaders/minijson.hpp"

namespace minijson = dgpp::minijson;

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <class F>
void rejects(F f) {
  bool threw = false;
  try {
    f();
  } catch (const std::exception&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("expected a rejection");
}

std::string le32(uint32_t v) {
  char b[4] = {char(v & 0xff), char((v >> 8) & 0xff), char((v >> 16) & 0xff), char(v >> 24)};
  return {b, 4};
}
std::string le16(uint16_t v) {
  char b[2] = {char(v & 0xff), char((v >> 8) & 0xff)};
  return {b, 2};
}

// RIFF/WAVE carrying an arbitrary block of already-encoded sample bytes.
std::string wav(std::string body, uint16_t format, uint16_t channels, uint32_t rate,
                uint16_t bits) {
  return "RIFF" + le32(4 + 24 + static_cast<uint32_t>(body.size())) + "WAVE" + "fmt " + le32(16) +
         le16(format) + le16(channels) + le32(rate) + le32(rate * channels * bits / 8) +
         le16(channels * bits / 8) + le16(bits) + "data" + le32(static_cast<uint32_t>(body.size())) +
         body;
}
std::string pcm16(const std::vector<int16_t>& p) {
  return {reinterpret_cast<const char*>(p.data()), p.size() * 2};
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

const minijson::Value& goldens() {
  static const std::string text = read_file(std::string(DGPP_SOURCE_DIR) +
                                            "/tests/data/qwen3_asr_feature_goldens.json");
  static const minijson::Value root = [] {
    if (text.empty()) throw std::runtime_error("missing frontend goldens json");
    return minijson::parse(text).root;
  }();
  return root;
}

// int16 samples -> an input_audio content part (base64 WAV), the client shape.
std::string b64_wav_part(std::string_view samples_b64) {
  return R"json({"type":"input_audio","input_audio":{"data":")json" + std::string(samples_b64) +
         R"json(","format":"wav"}})json";
}

std::vector<int16_t> gold_samples(const minijson::Value& case_root) {
  const std::string raw = dgpp::decode_base64(std::string(case_root.at("samples_b64").as_string()),
                                              1 << 22);
  std::vector<int16_t> out(raw.size() / 2);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}
}  // namespace

DGPP_TEST(audio_token_count) {
  using dgpp::audio_token_count;
  // Frames -> tokens, mirroring _get_feat_extract_output_lengths: 13 per
  // full 100-frame chunk, then three (x - 1) / 2 + 1 halvings of the tail.
  require(audio_token_count(50) == 7, "min-length tail");
  require(audio_token_count(100) == 13, "full chunk");
  require(audio_token_count(101) == 14, "one past a chunk");
  require(audio_token_count(220) == 29, "two chunks plus a 20-frame tail");
  require(audio_token_count(300) == 39, "three chunks");
  require(audio_token_count(1020) == 133, "ten chunks plus a 20-frame tail");
  require(audio_token_count(3000) == 390, "thirty seconds");
}

DGPP_TEST(audio_input_validation) {
  using dgpp::AudioInput;
  using dgpp::validate_audio_inputs;
  auto valid = [](int64_t offset, int64_t frames) {
    AudioInput a;
    a.offset = offset;
    a.frames = frames;
    a.tokens = dgpp::audio_token_count(frames);
    a.pcm.assign(static_cast<size_t>(frames * dgpp::kAudioHop), 0.1f);
    return a;
  };
  validate_audio_inputs({valid(0, 100)}, 117);
  validate_audio_inputs({valid(3, 100), valid(116, 50)}, 136);
  rejects([&] {
    auto a = valid(0, 100);
    a.tokens = 14;  // matches no legal frame count
    validate_audio_inputs({a}, 200);
  });
  rejects([&] {
    auto a = valid(0, 220);
    a.frames = 300;  // a consistent count, wrong for this payload
    validate_audio_inputs({a}, 3000);
  });
  rejects([&] {
    auto a = valid(0, 100);
    a.pcm.resize(a.pcm.size() - 1);  // frames and bytes disagree
    validate_audio_inputs({a}, 200);
  });
  rejects([&] { validate_audio_inputs({valid(100, 50), valid(0, 100)}, 200); });  // unordered
  rejects([&] { validate_audio_inputs({valid(0, 100), valid(12, 50)}, 200); });   // overlap
  rejects([&] { validate_audio_inputs({valid(100, 100)}, 110); });                // past the prompt
}

DGPP_TEST(wav_decode) {
  using dgpp::audio::decode_wav;
  std::vector<int16_t> pcm(1600);
  for (size_t i = 0; i < pcm.size(); ++i)
    pcm[i] = static_cast<int16_t>(std::sin(0.01 * static_cast<double>(i)) * 12000.0);
  const auto file = wav(pcm16(pcm), 1, 1, 16000, 16);
  auto w = decode_wav(reinterpret_cast<const uint8_t*>(file.data()), file.size());
  require(w.sample_rate == 16000 && w.channels == 1 && w.samples.size() == pcm.size(), "geometry");
  require(std::abs(w.samples[7] - pcm[7] / 32768.0f) < 1e-7f, "scale");

  // Stereo: L=+x, R=-x averages to silence, and the interleaving order is right.
  std::string inter;
  for (auto v : pcm) inter += pcm16({v, static_cast<int16_t>(-v)});
  auto sw = decode_wav(reinterpret_cast<const uint8_t*>((wav(inter, 1, 2, 22050, 16)).data()),
                       wav(inter, 1, 2, 22050, 16).size());
  require(sw.channels == 2 && sw.samples.size() == 2 * pcm.size(), "stereo decode");
  const auto mono = dgpp::audio::to_mono_16k(sw);
  require(mono.size() == static_cast<size_t>(static_cast<double>(pcm.size() - 1) * 16000 / 22050) +
                                1,
          "downmix and resample length");
  for (size_t i = 0; i < mono.size(); ++i)
    require(std::abs(mono[i]) < 1e-6f, "L/R cancellation");

  // 8-bit unsigned and IEEE float32 payloads.
  std::string u8(pcm.size(), '\0');
  for (size_t i = 0; i < pcm.size(); ++i)
    u8[i] = static_cast<char>(128 + (pcm[i] >> 8));
  auto bw = decode_wav(reinterpret_cast<const uint8_t*>(wav(u8, 1, 1, 16000, 8).data()),
                       wav(u8, 1, 1, 16000, 8).size());
  require(std::abs(bw.samples[64] - pcm[64] / 32768.0f) < 1.0f / 128.0f, "8-bit scale");
  std::string f32;
  for (auto v : pcm) {
    const float x = v / 32768.0f;
    f32 += {reinterpret_cast<const char*>(&x)[0], reinterpret_cast<const char*>(&x)[1],
            reinterpret_cast<const char*>(&x)[2], reinterpret_cast<const char*>(&x)[3]};
  }
  auto fw = decode_wav(reinterpret_cast<const uint8_t*>(wav(f32, 3, 1, 16000, 32).data()),
                       wav(f32, 3, 1, 16000, 32).size());
  require(std::abs(fw.samples[7] - pcm[7] / 32768.0f) < 1e-9f, "float payload");

  // Refusals: not RIFF, packed-integer (A-law), a data chunk past the file.
  const char* ogg = "OggS\0\0\0\0\0\0";
  rejects([&] { decode_wav(reinterpret_cast<const uint8_t*>(ogg), 10); });
  rejects([&] {
    const auto bad = wav(pcm16(pcm), 6, 1, 16000, 8);
    decode_wav(reinterpret_cast<const uint8_t*>(bad.data()), bad.size());
  });
  rejects([&] {
    auto bad = wav(pcm16(pcm), 1, 1, 16000, 16);
    bad.replace(4, 4, le32(0xfffffff0));  // RIFF length far past EOF
    bad.resize(64);
    decode_wav(reinterpret_cast<const uint8_t*>(bad.data()), bad.size());
  });
}

DGPP_TEST(mel_frontend_goldens) {
  const auto& cases = goldens().at("cases");
  double worst = 0.0;
  for (const auto& entry : cases.members()) {
    const auto& c = entry.value;
    auto samples = gold_samples(c);
    std::vector<float> f_samples(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) f_samples[i] = samples[i] / 32768.0f;
    if (f_samples.size() < static_cast<size_t>(dgpp::kAudioMinSamples))
      f_samples.resize(static_cast<size_t>(dgpp::kAudioMinSamples), 0.f);
    f_samples.resize(f_samples.size() - f_samples.size() % dgpp::kAudioHop);
    const auto f = dgpp::audio::compute_fbank(f_samples.data(), f_samples.size());
    require(f.frames == c.at("valid_frames").as_int(), "frame count matches the oracle");

    const std::string want =
        dgpp::decode_base64(std::string(c.at("features_b64").as_string()), 1 << 24);
    require(want.size() == static_cast<size_t>(f.frames) * dgpp::kAudioMelBins * 4,
            "golden feature size");
    for (size_t i = 0; i < want.size() / 4; ++i) {
      float w;
      std::memcpy(&w, want.data() + 4 * i, 4);
      const double d = std::abs(w - f.features[i]);
      if (d > worst) worst = d;
      if (d > 2e-4) throw std::runtime_error("log-mel feature drifted from the oracle");
    }
  }
  std::printf("  worst |delta| = %.3e\n", worst);
}

DGPP_TEST(audio_prepare_end_to_end) {
  const auto& cases = goldens().at("cases");
  for (const char* name : {"chirp_1s", "voice_2s2", "tone_0s2", "chirp_10s2"}) {
    const auto* c = cases.find(name);
    require(c != nullptr, "goldens case");
    const auto w = wav(pcm16(gold_samples(*c)), 1, 1, 16000, 16);
    auto au = dgpp::serve::prepare_audio(
        minijson::parse(b64_wav_part(dgpp::encode_base64(w))).root, "x");
    require(au.tokens == static_cast<int>(c->at("tokens").as_int()), "token count");
    require(au.frames == static_cast<int64_t>(au.pcm.size()) / dgpp::kAudioHop, "hop align");
    require(au.frames == c->at("valid_frames").as_int(), "valid frames");
  }
  // Refusals: remote URLs, non-WAV payloads, malformed base64, missing data.
  for (const std::string& bad :
       {std::string(R"({"type":"audio_url","audio_url":{"url":"https://x/a.wav"}})"),
        std::string(R"({"type":"audio_url","audio_url":{"url":"data:audio/mpeg;base64,AAAA"}})"),
        std::string(R"({"type":"input_audio","input_audio":{"data":"no$e64"}})"),
        std::string(R"({"type":"input_audio","input_audio":{}})"),
        std::string(R"json({"type":"input_audio","input_audio":{"data":")json") +
            dgpp::encode_base64("not a wav at all, not even close") +
            R"json(","format":"wav"}})json"}) {
    rejects([&] {
      const auto doc = minijson::parse(bad);
      dgpp::serve::prepare_audio(doc.root, "x");
    });
  }
}

DGPP_TEST(mel_frontend_invariants) {
  const auto& filt = dgpp::audio::mel_filter_bank();
  require(filt.size() == 201 * 128, "bank shape");
  int active = 0;
  for (int m = 0; m < 128; ++m) {
    bool any = false;
    for (int k = 0; k < 201; ++k)
      if (filt[static_cast<size_t>(k) * 128 + m] > 0) any = true;
    active += any;
  }
  require(active == 128, "every mel filter has support");
  const auto& win = dgpp::audio::hann_window_400();
  // Periodic: the endpoints are 0 and (near) 0 but never both exact; the
  // symmetry w[n] == w[N - n] and the peak at N/2 are the real contract.
  require(win.front() == 0.f && std::abs(win[200] - 1.f) < 1e-6f &&
              std::abs(win.back() - win[1]) < 1e-6f && win[399] > 0.f && win[399] < 3e-4f,
          "periodic hann");
  // All-silence hits the clamp: (log10(1e-10) + 4) / 4 == -1.5 exactly.
  std::vector<float> zeros(16000 * 2, 0.f);
  const auto f = dgpp::audio::compute_fbank(zeros.data(), zeros.size());
  require(f.frames == 200, "silence frames");
  for (size_t i = 0; i < f.features.size(); i += 97)
    require(std::abs(f.features[i] + 1.5f) < 1e-5f, "silence sits on the clamp");
  rejects([&] { dgpp::audio::compute_fbank(zeros.data(), zeros.size() - 1); });  // partial hop
}
