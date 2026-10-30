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
#include "audio/compressed_decode.hpp"
#include "audio/fft400.hpp"
#include "audio/mp3_decode.hpp"
#include "audio/wav_decode.hpp"
#include "serve/audio_transcriptions.hpp"
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

DGPP_TEST(mp3_decode) {
  using dgpp::audio::decode_mp3;
  const std::string mono = read_file(std::string(DGPP_SOURCE_DIR) +
                                     "/tests/data/mp3_tone_1s_16k_mono.mp3");
  require(!mono.empty(), "mp3 mono fixture present");
  auto m = decode_mp3(reinterpret_cast<const uint8_t*>(mono.data()), mono.size());
  require(m.sample_rate == 16000 && m.channels == 1, "mp3 geometry");
  // One second of audio plus encoder delay/padding: near 16000, never exact.
  require(m.samples.size() > 15000 && m.samples.size() < 19000,
          "mp3 frame count near one second");
  double sumsq = 0;
  for (float v : m.samples) sumsq += static_cast<double>(v) * v;
  const double rms = std::sqrt(sumsq / m.samples.size());
  require(rms > 0.18 && rms < 0.30, "mp3 tone energy near the source sine");
  // Content check against the known source (12000/32768 * sin(0.02 i)):
  // best normalized correlation over the encoder-delay lag range.
  double e1 = 0;
  for (size_t i = 0; i < 16000; ++i) {
    const double src = 12000.0 / 32768.0 * std::sin(0.02 * i);
    e1 += src * src;
  }
  double best = 0;
  for (size_t lag = 0; lag + 16000 <= m.samples.size() && lag < 3000; lag += 16) {
    double dot = 0, e2 = 0;
    for (size_t i = 0; i < 16000; ++i) {
      const double src = 12000.0 / 32768.0 * std::sin(0.02 * i);
      const double got = m.samples[lag + i];
      dot += src * got;
      e2 += got * got;
    }
    if (e2 > 0) best = std::max(best, dot / std::sqrt(e1 * e2));
  }
  require(best > 0.9, "mp3 tone matches the source sine");

  // Stereo at a native rate: geometry holds and the shared mono/16k path
  // still lands near one second.
  const std::string stereo = read_file(std::string(DGPP_SOURCE_DIR) +
                                       "/tests/data/mp3_tone_1s_44100_stereo.mp3");
  require(!stereo.empty(), "mp3 stereo fixture present");
  auto st = decode_mp3(reinterpret_cast<const uint8_t*>(stereo.data()), stereo.size());
  require(st.sample_rate == 44100 && st.channels == 2, "stereo mp3 geometry");
  require(st.samples.size() / 2 > 43000 && st.samples.size() / 2 < 50000,
          "stereo mp3 frame count near one second");
  const auto dm = dgpp::audio::to_mono_16k(st);
  require(dm.size() > 16000 && dm.size() < 18000, "stereo mp3 downmixes near one second");

  // Refusals: empty, non-audio bytes, a truncated head, an ID3 tag alone.
  rejects([&] { decode_mp3(nullptr, 0); });
  const char* text = "not audio at all, not even close to an mp3 sync";
  rejects([&] { decode_mp3(reinterpret_cast<const uint8_t*>(text), std::strlen(text)); });
  rejects([&] {
    decode_mp3(reinterpret_cast<const uint8_t*>(mono.data()), 64);
  });
  const std::string tag_only("ID3\x04\x00\x00\x00\x00\x00\x64", 10);
  rejects([&] {
    decode_mp3(reinterpret_cast<const uint8_t*>(tag_only.data()), tag_only.size());
  });
}

DGPP_TEST(mp3_gapless_trim) {
  using dgpp::audio::gapless_trim;
  // A synthetic MPEG-1 Layer-III frame (128 kbps, 44.1 kHz, stereo):
  // 4-byte header, 32-byte side info, Xing tag, LAME tag.
  auto frame = [] {
    std::string f(417, '\0');
    f[0] = '\xff';
    f[1] = '\xfb';  // MPEG-1, layer III, no CRC
    f[2] = '\x90';  // 128 kbps, 44.1 kHz, no padding
    f[3] = '\x00';  // stereo
    std::memcpy(&f[36], "Xing", 4);
    f[43] = 0x03;  // flags are big-endian: frames + bytes present
    std::memcpy(&f[52], "LAME3.99r", 9);  // incl. the 'r': version is 5 chars
    f[52 + 21] = '\x84';  // delay 2112 = 0x840
    f[52 + 22] = '\x07';  // pad 1900 = 0x76c (high nibble 0, low 7)
    f[52 + 23] = '\x6c';
    return f;
  };
  const std::string good = frame();
  const auto trim = gapless_trim(reinterpret_cast<const uint8_t*>(good.data()), good.size());
  require(trim.has_value() && trim->first == 2112 && trim->second == 1900, "lame tag parses");

  // An ID3v2 tag ahead of the frame is skipped, not mistaken for audio.
  const std::string id3("ID3\x04\x00\x00\x00\x00\x00\x00", 10);
  const std::string tagged = id3 + good;
  const auto trim2 =
      gapless_trim(reinterpret_cast<const uint8_t*>(tagged.data()), tagged.size());
  require(trim2.has_value() && trim2->first == 2112, "tag found past id3v2");

  // Structural misses: no Xing where the header says it is (a blind
  // "LAME" later in the frame must NOT match), insane values,
  // unprintable versions, truncation.
  std::string noxing = good;
  std::memset(&noxing[36], 0, 4);
  std::memcpy(&noxing[200], "LAME3.99r", 8);
  rejects([&] {
    (void)gapless_trim(reinterpret_cast<const uint8_t*>(noxing.data()), noxing.size())
        .value();
  });
  std::string full_range = good;
  full_range[52 + 21] = '\xff';  // delay 0xfff = 4095: legal field width,
  full_range[52 + 22] = '\xff';  // kept here, bounded at apply time
  const auto trim3 = gapless_trim(reinterpret_cast<const uint8_t*>(full_range.data()),
                                  full_range.size());
  require(trim3.has_value() && trim3->first == 4095, "full-range delay passes through");
  std::string badver = good;
  badver[52 + 8] = '\x01';
  rejects([&] {
    (void)gapless_trim(reinterpret_cast<const uint8_t*>(badver.data()), badver.size())
        .value();
  });
  rejects([&] {
    (void)gapless_trim(reinterpret_cast<const uint8_t*>(good.data()), 60).value();
  });
  const char* garbage = "not an mp3 frame at all, not even close";
  rejects([&] {
    (void)gapless_trim(reinterpret_cast<const uint8_t*>(garbage), std::strlen(garbage))
        .value();
  });
}

DGPP_TEST(flac_vorbis_decode) {
  // FLAC is lossless: geometry exact, samples bit-near the source sine.
  const std::string flac = read_file(std::string(DGPP_SOURCE_DIR) +
                                     "/tests/data/flac_tone_1s_16k_mono.flac");
  require(!flac.empty(), "flac fixture present");
  auto fl = dgpp::audio::decode_flac(reinterpret_cast<const uint8_t*>(flac.data()),
                                     flac.size());
  require(fl.sample_rate == 16000 && fl.channels == 1, "flac geometry");
  require(fl.samples.size() == 16000, "flac frame count exact");
  double worst = 0;
  for (size_t i = 0; i < fl.samples.size(); ++i) {
    const double want = 12000.0 / 32768.0 * std::sin(0.02 * i);
    worst = std::max(worst, std::abs(fl.samples[i] - want));
  }
  require(worst < 1e-4, "flac samples match the source sine");

  // Vorbis is lossy: geometry exact, tone correlation high.
  const std::string ogg = read_file(std::string(DGPP_SOURCE_DIR) +
                                    "/tests/data/vorbis_tone_1s_44100_stereo.ogg");
  require(!ogg.empty(), "vorbis fixture present");
  auto vo = dgpp::audio::decode_vorbis(reinterpret_cast<const uint8_t*>(ogg.data()),
                                       ogg.size());
  require(vo.sample_rate == 44100 && vo.channels == 2, "vorbis geometry");
  require(vo.samples.size() / 2 > 43000 && vo.samples.size() / 2 < 45000,
          "vorbis frame count near one second");
  const auto dm = dgpp::audio::to_mono_16k(vo);
  // Reference is the stereo mix the downmixer averages, resampled to the
  // same grid (linear resample has no delay; skip the codec lead-in).
  double e1 = 0, dot = 0, e2 = 0;
  for (size_t i = 2000; i + 1000 < dm.size() && i < 16000; ++i) {
    const double x = static_cast<double>(i) * 44100.0 / 16000.0;
    const size_t j = static_cast<size_t>(x);
    const double f = x - j;
    const auto at = [&](size_t k, double hz) {
      return std::sin(2 * 3.14159265358979 * hz * k / 44100);
    };
    const double l = at(j, 440) * (1 - f) + at(j + 1, 440) * f;
    const double r = at(j, 660) * (1 - f) + at(j + 1, 660) * f;
    const double src = 10000.0 / 32768.0 * 0.5 * (l + r);
    dot += src * dm[i];
    e1 += src * src;
    e2 += dm[i] * dm[i];
  }
  require(e1 > 0 && e2 > 0 && dot / std::sqrt(e1 * e2) > 0.9,
          "vorbis tone matches the source sine");

  // Refusals: cross-codec bytes, garbage, truncation, and Ogg Opus,
  // which is refused by name.
  rejects([&] {
    dgpp::audio::decode_flac(reinterpret_cast<const uint8_t*>(ogg.data()), ogg.size());
  });
  rejects([&] {
    dgpp::audio::decode_vorbis(reinterpret_cast<const uint8_t*>(flac.data()), flac.size());
  });
  const char* text = "not audio at all";
  rejects([&] {
    dgpp::audio::decode_flac(reinterpret_cast<const uint8_t*>(text), std::strlen(text));
  });
  rejects([&] {
    dgpp::audio::decode_flac(reinterpret_cast<const uint8_t*>(flac.data()), 64);
  });
  const std::string opus = read_file(std::string(DGPP_SOURCE_DIR) +
                                     "/tests/data/opus_tone_1s.opus");
  require(!opus.empty(), "opus fixture present");
  bool named_opus = false;
  try {
    dgpp::audio::decode_vorbis(reinterpret_cast<const uint8_t*>(opus.data()), opus.size());
  } catch (const std::exception& e) {
    named_opus = std::string(e.what()).find("Opus") != std::string::npos;
  }
  require(named_opus, "opus refusal names opus");

  // Admission: flac/ogg bytes under their format labels decode to the
  // same hop-aligned spans as wav/mp3.
  const std::string flac_part = R"json({"type":"input_audio","input_audio":{"data":")json" +
                                dgpp::encode_base64(flac) +
                                R"json(","format":"flac"}})json";
  auto au_flac = dgpp::serve::prepare_audio(minijson::parse(flac_part).root, "x");
  require(au_flac.frames == 100, "flac admission spans one second");
  const std::string ogg_part = R"json({"type":"input_audio","input_audio":{"data":")json" +
                               dgpp::encode_base64(ogg) +
                               R"json(","format":"ogg"}})json";
  auto au_ogg = dgpp::serve::prepare_audio(minijson::parse(ogg_part).root, "x");
  require(au_ogg.frames > 90 && au_ogg.frames < 110, "ogg admission near one second");
}

DGPP_TEST(mp3_prepare_end_to_end) {  const std::string mono = read_file(std::string(DGPP_SOURCE_DIR) +
                                     "/tests/data/mp3_tone_1s_16k_mono.mp3");
  require(!mono.empty(), "mp3 mono fixture present");
  const std::string part = R"json({"type":"input_audio","input_audio":{"data":")json" +
                           dgpp::encode_base64(mono) + R"json(","format":"mp3"}})json";
  auto au = dgpp::serve::prepare_audio(minijson::parse(part).root, "x");
  require(au.frames > 90 && au.frames < 120, "mp3 admission near one second");
  require(au.tokens == dgpp::audio_token_count(au.frames), "mp3 token count");
  require(au.frames == static_cast<int64_t>(au.pcm.size()) / dgpp::kAudioHop, "mp3 hop align");
  // The container is sniffed from the bytes: a WAV labeled mp3 still
  // decodes, while an unknown format label is refused up front.
  std::vector<int16_t> pcm(1600, 1000);
  const std::string wav_bytes = wav(pcm16(pcm), 1, 1, 16000, 16);
  const std::string wav_as_mp3 = R"json({"type":"input_audio","input_audio":{"data":")json" +
                                 dgpp::encode_base64(wav_bytes) +
                                 R"json(","format":"mp3"}})json";
  auto au2 = dgpp::serve::prepare_audio(minijson::parse(wav_as_mp3).root, "x");
  require(au2.frames == 50, "wav bytes decode under an mp3 label");
  const std::string opus_label = R"json({"type":"input_audio","input_audio":{"data":")json" +
                                 dgpp::encode_base64(wav_bytes) +
                                 R"json(","format":"opus"}})json";
  rejects([&] {
    const auto doc = minijson::parse(opus_label);
    dgpp::serve::prepare_audio(doc.root, "x");
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

DGPP_TEST(fft400_matches_direct_dft) {
  // Random complex input: the mixed-radix FFT must match the naive
  // matrix product to float-rounding distance.
  double in_re[400], in_im[400], out_re[400], out_im[400];
  uint64_t state = 0x12345678u;
  auto rand = [&] {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<double>((state >> 33) & 0xffffff) / 0xffffff - 0.5;
  };
  for (int n = 0; n < 400; ++n) {
    in_re[n] = rand();
    in_im[n] = rand();
  }
  dgpp::audio::fft400_forward(in_re, in_im, out_re, out_im);
  double worst = 0;
  for (int k = 0; k < 400; ++k) {
    double re = 0, im = 0;
    for (int n = 0; n < 400; ++n) {
      const double a = 2 * 3.14159265358979 * k * n / 400;
      re += in_re[n] * std::cos(a) + in_im[n] * std::sin(a);
      im += in_im[n] * std::cos(a) - in_re[n] * std::sin(a);
    }
    worst = std::max({worst, std::abs(out_re[k] - re), std::abs(out_im[k] - im)});
  }
  require(worst < 1e-9, "fft400 matches the direct dft");
}

DGPP_TEST(audio_piece_takes) {
  using dgpp::audio_piece_takes;
  using dgpp::kAudioMinSamples;
  // The 30.04 s killer: 480640 samples stage as 464000 + 16640, never a
  // 640-sample tail below the frontend floor nor a piece past the tower's
  // 3000-frame window.
  require(audio_piece_takes(16000) == std::vector<int64_t>{16000}, "short clip");
  require(audio_piece_takes(480640) == (std::vector<int64_t>{464000, 16640}),
          "window lends the tail a chunk");
  require(audio_piece_takes(488000) == (std::vector<int64_t>{480000, 8000}),
          "exact-minimum tail stands alone");
  require(audio_piece_takes(960000) == (std::vector<int64_t>{480000, 480000}),
          "two full windows");
  require(audio_piece_takes(968640) == (std::vector<int64_t>{480000, 480000, 8640}),
          "tall tail stands alone");
  require(audio_piece_takes(7999).empty(), "below-minimum total");
  require(audio_piece_takes(0).empty(), "empty total");
  for (int64_t total : {8000ll, 16000ll, 479840ll, 480000ll, 480160ll, 487840ll,
                        488000ll, 488160ll, 960640ll, 1440000ll, 2000000ll}) {
    const auto takes = audio_piece_takes(total);
    int64_t sum = 0;
    for (auto t : takes) {
      require(t >= kAudioMinSamples, "every take clears the floor");
      require(t <= 480000, "no take exceeds the tower window");
      require(t % dgpp::kAudioHop == 0, "takes stay hop-aligned");
      sum += t;
    }
    require(sum == total, "takes partition the clip");
  }
}

DGPP_TEST(transcription_segment_bodies) {
  using dgpp::serve::MultipartPart;
  // A 60 s synthetic WAV part (6000 frames, 780 audio tokens).
  std::string pcm;
  for (int i = 0; i < 16000 * 60; ++i) {
    const auto v = static_cast<int16_t>(12000.0 * std::sin(0.02 * i));
    pcm += static_cast<char>(v & 0xff);
    pcm += static_cast<char>(static_cast<unsigned>(v) >> 8);
  }
  const std::string wav_bytes = wav(pcm, 1, 1, 16000, 16);
  const std::vector<MultipartPart> parts = {
      {"file", "long.wav", "audio/wav", std::string_view(wav_bytes)},
      {"model", "", "", "test-model"},
  };
  // Unknown windows and fitting clips ride the single-body path.
  require(dgpp::serve::transcription_segment_bodies(parts, "test-model", 0).size() == 1,
          "unknown window singles");
  require(dgpp::serve::transcription_segment_bodies(parts, "test-model", 2048).size() == 1,
          "fitting clip singles");
  // A 300-row window segments into whole-chunk bodies summing exactly.
  auto segs = dgpp::serve::transcription_segment_bodies(parts, "test-model", 300);
  require(segs.size() == 5, "five segments");
  int64_t frames_total = 0;
  for (const auto& seg : segs) {
    const auto* messages = seg.body.find("messages");
    require(messages != nullptr && messages->is_array(), "segment messages");
    const auto& content = messages->items().back().at("content").items();
    const auto& ia = content.back().at("input_audio");
    require(ia.at("format").as_string() == "wav", "segments rewrap as wav");
    const std::string raw =
        dgpp::decode_base64(std::string(ia.at("data").as_string()), 1 << 24);
    auto w = dgpp::audio::decode_wav(reinterpret_cast<const uint8_t*>(raw.data()),
                                     raw.size());
    const int64_t frames = static_cast<int64_t>(w.samples.size()) / dgpp::kAudioHop;
    require(frames % dgpp::kAudioChunkFrames == 0 || &seg == &segs.back(),
            "whole-chunk splits");
    frames_total += frames;
    const auto* steps = seg.body.find("max_tokens");
    require(steps != nullptr && steps->as_int() >= 256, "per-segment cap");
  }
  require(frames_total == 6000, "segments partition the clip");
  // A window that cannot hold one chunk plus the reserve refuses loudly.
  bool threw = false;
  try {
    dgpp::serve::transcription_segment_bodies(parts, "test-model", 100);
  } catch (const dgpp::serve::MultipartError& e) {
    threw = std::string(e.what()).find("segmented") != std::string::npos;
  }
  require(threw, "unsegmentable window refuses");
}

DGPP_TEST(transcription_language_tag_format) {
  using dgpp::serve::format_transcription_text;
  using dgpp::serve::split_language_tag;
  // The user's verbatim case: glued tag splits onto its own line.
  auto hello = split_language_tag(
      "language EnglishHmm. Oh yeah, yeah. He wasn't even that big");
  require(hello.first == "language English", "tag splits");
  require(hello.second == "Hmm. Oh yeah, yeah. He wasn't even that big", "body splits");
  require(format_transcription_text(
              "language EnglishHmm. Oh yeah, yeah. He wasn't even that big") ==
              "language English\nHmm. Oh yeah, yeah. He wasn't even that big",
          "tag renders above the body");
  // Corpus shapes (LibriSpeech gate): lone tags stay whole, lowercase
  // glue passes through rather than eating speech, and anything else is
  // verbatim.
  require(split_language_tag("language None").second.empty(), "lone tag unsplit");
  require(format_transcription_text("language None") == "language None", "lone tag verbatim");
  require(format_transcription_text("language Englishsaid more") == "language Englishsaid more",
          "lowercase glue verbatim");
  require(format_transcription_text("Hello world") == "Hello world", "no tag verbatim");
  require(format_transcription_text("language english lower") == "language english lower",
          "lowercase tag verbatim");
  require(format_transcription_text("") == "", "empty verbatim");
  require(format_transcription_text("language English Hello") == "language English Hello",
          "spaced tag verbatim");
}

DGPP_TEST(audio_prepare_end_to_end) {  const auto& cases = goldens().at("cases");
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
  // Refusals: remote URLs, non-audio payloads, malformed base64, missing data.
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
