#include "serve/audio_transcriptions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/audio_input.hpp"
#include "common/base64.hpp"
#include "audio/wav_decode.hpp"
#include "serve/audio_inputs.hpp"

namespace dgpp::serve {
namespace {
// Multipart framing bounds: curl and the SDKs emit a handful of small
// fields; anything past these is a malformed or hostile body.
constexpr size_t kMaxParts = 32;
constexpr size_t kMaxFieldName = 256;
constexpr size_t kMaxTextField = 8 * 1024;

// A line within [pos, end): LF-terminated (an optional CR is stripped).
// Returns npos past a final line without a terminator.
size_t line_end(std::string_view body, size_t pos) {
  const size_t nl = body.find('\n', pos);
  return nl == std::string_view::npos ? body.size() : nl;
}

std::string_view strip_cr(std::string_view line) {
  if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
  return line;
}

[[noreturn]] void fail(const std::string& message, const std::string& param) {
  throw MultipartError(message, param);
}

// boundary= value from a multipart content-type: token or quoted-string.
std::string parse_boundary(std::string_view content_type) {
  // The header name is matched case-insensitively by the caller contract
  // (HttpRequest lowercases keys; values keep their case).
  std::string lower(content_type);
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  const size_t semi = lower.find(';');
  if (semi == std::string::npos) fail("multipart body needs a boundary parameter", "content-type");
  size_t pos = lower.find("boundary=", semi);
  if (pos == std::string_view::npos) fail("multipart body needs a boundary parameter", "content-type");
  pos += 9;
  while (pos < lower.size() && lower[pos] == ' ') ++pos;
  // Slice the ORIGINAL value (case matters only for matching the body).
  std::string_view raw = content_type.substr(pos);
  if (!raw.empty() && raw.front() == '"') {
    const size_t end = raw.find('"', 1);
    if (end == std::string_view::npos || end > 78)
      fail("multipart boundary is malformed", "content-type");
    return std::string(raw.substr(1, end - 1));
  }
  const size_t end = raw.find_first_of(" ;\t\r\n");
  std::string boundary(raw.substr(0, end));
  if (boundary.empty() || boundary.size() > 70)
    fail("multipart boundary is malformed", "content-type");
  return boundary;
}

struct RawPart {
  std::string headers;
  std::string_view data;
};

// Split the body on exact delimiter lines (--boundary / --boundary--).
// The payload is never substring-searched: a WAV may contain the boundary
// text, so part data runs line-by-line to the next delimiter line.
std::vector<RawPart> split_parts(std::string_view body, const std::string& boundary) {
  const std::string dash = "--" + boundary;
  const auto advance = [&](size_t eol) {
    return eol < body.size() ? eol + 1 : body.size() + 1;
  };
  size_t pos = 0;
  // The preamble (bytes before the first delimiter) is skipped.
  for (;;) {
    if (pos > body.size()) fail("multipart body is truncated", "file");
    const size_t eol = line_end(body, pos);
    const std::string_view line = strip_cr(body.substr(pos, eol - pos));
    pos = advance(eol);
    if (line == dash) break;
    if (line == dash + "--") return {};
  }
  std::vector<RawPart> out;
  bool closed = false;
  while (!closed) {
    // Part headers run to the empty line.
    std::string headers;
    for (;;) {
      if (pos > body.size()) fail("multipart part is truncated", "file");
      const size_t heol = line_end(body, pos);
      const std::string_view hline = strip_cr(body.substr(pos, heol - pos));
      pos = advance(heol);
      if (hline.empty()) break;
      headers += std::string(hline);
      headers += "\n";
      if (headers.size() > 8 * 1024) fail("multipart part headers are too large", "file");
    }
    const size_t part_start = pos;
    size_t scan = pos;
    size_t data_end = pos;
    for (;;) {
      if (scan > body.size()) fail("multipart part is truncated", "file");
      const size_t seol = line_end(body, scan);
      const std::string_view sline = strip_cr(body.substr(scan, seol - scan));
      if (sline == dash || sline == dash + "--") {
        // Back over the line break that ended the data.
        data_end = scan >= 2 && body[scan - 2] == '\r' && body[scan - 1] == '\n' ? scan - 2
                   : scan >= 1 && body[scan - 1] == '\n'                         ? scan - 1
                                                                                : scan;
        closed = (sline == dash + "--");
        pos = advance(seol);
        break;
      }
      scan = advance(seol);
    }
    out.push_back({std::move(headers), body.substr(part_start, data_end - part_start)});
    if (out.size() > kMaxParts) fail("multipart body has too many parts", "file");
  }
  if (!closed) fail("multipart body is missing its closing delimiter", "file");
  return out;
}
}  // namespace

std::vector<MultipartPart> parse_multipart(std::string_view content_type, std::string_view body) {
  if (content_type.size() < 19) fail("transcriptions need multipart/form-data", "content-type");
  std::string lower(content_type.substr(0, 19));
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower != "multipart/form-data")
    fail("transcriptions need multipart/form-data", "content-type");
  const std::string boundary = parse_boundary(content_type);
  const auto raw = split_parts(body, boundary);
  std::vector<MultipartPart> parts;
  for (const auto& rp : raw) {
    MultipartPart part;
    // Minimal header parse: Content-Disposition (name, filename) and
    // Content-Type. Continuations are not supported and unnecessary here.
    size_t hpos = 0;
    bool disposition = false;
    while (hpos < rp.headers.size()) {
      const size_t heol = rp.headers.find('\n', hpos);
      const std::string_view line(rp.headers.data() + hpos,
                                  (heol == std::string::npos ? rp.headers.size() : heol) - hpos);
      hpos = heol == std::string::npos ? rp.headers.size() : heol + 1;
      const size_t colon = line.find(':');
      if (colon == std::string_view::npos) continue;
      std::string name(line.substr(0, colon));
      std::transform(name.begin(), name.end(), name.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      std::string_view value = line.substr(colon + 1);
      while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
        value.remove_prefix(1);
      if (name == "content-disposition") {
        disposition = true;
        // form-data; name="file"; filename="audio.wav"
        size_t vpos = 0;
        while (vpos < value.size()) {
          const size_t vend = value.find(';', vpos);
          std::string_view item = value.substr(vpos, vend == std::string_view::npos
                                                          ? std::string_view::npos
                                                          : vend - vpos);
          vpos = vend == std::string_view::npos ? value.size() : vend + 1;
          while (!item.empty() && item.front() == ' ') item.remove_prefix(1);
          if (item.starts_with("name=")) {
            std::string_view v = item.substr(5);
            if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
            part.name = std::string(v);
          } else if (item.starts_with("filename=")) {
            std::string_view v = item.substr(9);
            if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
            part.filename = std::string(v);
          }
        }
      } else if (name == "content-type") {
        part.content_type = std::string(value);
      }
    }
    if (!disposition) fail("multipart part is missing its disposition", "file");
    if (part.name.empty() || part.name.size() > kMaxFieldName)
      fail("multipart part has a bad field name", "file");
    part.data = rp.data;
    parts.push_back(std::move(part));
  }
  return parts;
}

namespace {
// Canonical 44-byte WAV header sniff for the max_tokens duration rule:
// rate@24, channels@22, bits@34, "data"@36 with size@40. Anything else
// (extensible fmt, extra chunks, MP3) falls back to the decoded probe
// duration at the call site.
double wav_seconds(std::string_view bytes) {
  if (bytes.size() < 44) return -1.0;
  const auto u32 = [&](size_t at) {
    return static_cast<uint32_t>(static_cast<unsigned char>(bytes[at])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[at + 1])) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[at + 2])) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(bytes[at + 3])) << 24);
  };
  const auto u16 = [&](size_t at) {
    return static_cast<uint16_t>(static_cast<unsigned char>(bytes[at])) |
           (static_cast<uint16_t>(static_cast<unsigned char>(bytes[at + 1])) << 8);
  };
  if (std::string_view(bytes.data(), 4) != "RIFF" ||
      std::string_view(bytes.data() + 8, 4) != "WAVE" ||
      std::string_view(bytes.data() + 12, 4) != "fmt " || u32(16) != 16 ||
      std::string_view(bytes.data() + 36, 4) != "data")
    return -1.0;
  const uint32_t rate = u32(24);
  const uint32_t align = static_cast<uint32_t>(u16(22)) * u16(34) / 8;
  if (rate == 0 || align == 0) return -1.0;
  return static_cast<double>(u32(40)) / (static_cast<double>(rate) * align);
}
}  // namespace

TranscriptionFields parse_transcription_fields(const std::vector<MultipartPart>& parts,
                                                 const std::string& model_id) {
  const MultipartPart* file = nullptr;
  const MultipartPart* model = nullptr;
  const MultipartPart* language = nullptr;
  const MultipartPart* prompt = nullptr;
  const MultipartPart* format = nullptr;
  const MultipartPart* temperature = nullptr;
  for (const auto& part : parts) {
    auto first = [&](const MultipartPart*& slot) {
      if (slot == nullptr) slot = &part;
    };
    if (part.name == "file") first(file);
    else if (part.name == "model") first(model);
    else if (part.name == "language") first(language);
    else if (part.name == "prompt") first(prompt);
    else if (part.name == "response_format") first(format);
    else if (part.name == "temperature") first(temperature);
    // Unknown fields are ignored for forward compatibility.
  }
  if (file == nullptr || file->data.empty())
    fail("the audio file is required", "file");
  if (model == nullptr || model->data.empty()) fail("model is required", "model");
  if (model->data.size() > 256) fail("model name is too long", "model");
  TranscriptionFields f;
  f.file_bytes = file->data;
  f.filename = file->filename;
  f.model = std::string(model->data);
  if (f.model != model_id)
    throw MultipartError("the model '" + f.model + "' does not exist on this server",
                         "model", 404);
  if (format != nullptr) {
    if (format->data.size() > kMaxTextField) fail("response_format is too long", "response_format");
    f.response_format = std::string(format->data);
    if (f.response_format != "json" && f.response_format != "text")
      fail("response_format '" + f.response_format +
               "' is not supported (json and text are)",
           "response_format");
  }
  if (temperature != nullptr) {
    if (temperature->data.size() > 32) fail("temperature is not a number", "temperature");
    const std::string t(temperature->data);
    char* end = nullptr;
    f.temperature = std::strtod(t.c_str(), &end);
    if (end == t.c_str() || *end != '\0' || !std::isfinite(f.temperature) ||
        f.temperature < 0.0 || f.temperature > 2.0)
      fail("temperature must be a number in [0, 2]", "temperature");
    f.has_temperature = true;
  }
  if (language != nullptr) {
    if (language->data.empty() || language->data.size() > 64)
      fail("language must be a non-empty string", "language");
    f.language = std::string(language->data);
  }
  if (prompt != nullptr) {
    if (prompt->data.size() > 4096) fail("prompt is too long", "prompt");
    f.prompt = std::string(prompt->data);
  }
  return f;
}

namespace {
// Completion cap from the clip length (~4 text tokens per audio second
// covers dense English with margin), floored so short clips still close
// cleanly and capped so prompt + cap fits any admittable pool. The
// engine's reserve arithmetic still refuses the truly oversized.
int transcription_steps(double seconds) {
  int steps = 2048;
  if (seconds >= 0.0)
    steps = std::max(256, std::min(6000, static_cast<int>(seconds * 4.0) + 128));
  return steps;
}
}  // namespace

TranscriptionRequest transcription_body_for_wav(const TranscriptionFields& f,
                                                 std::string_view wav_bytes) {
  using V = minijson::Value;
  // The audio must decode before anything is admitted: undecodable bytes
  // are a 400 naming the file, never scheduler work. The decoded span is
  // discarded here (admission re-decodes from the same bytes); this pass
  // only proves the clip is servable and measures it.
  const AudioInput probe = prepare_audio_bytes(
      reinterpret_cast<const uint8_t*>(wav_bytes.data()), wav_bytes.size(), "file");
  double seconds = wav_seconds(wav_bytes);
  if (seconds < 0.0 && probe.frames > 0)
    seconds = static_cast<double>(probe.frames) * kAudioHop / kAudioSampleRate;
  const int steps = transcription_steps(seconds);
  std::vector<minijson::Member> members;
  members.push_back({"model", V::make_owned_string(f.model)});
  std::vector<V> messages;
  if (!f.prompt.empty())
    messages.push_back(V::make_object({{"role", V::make_string("system")},
                                       {"content", V::make_owned_string(f.prompt)}}));
  std::vector<minijson::Member> ia_members;
  ia_members.push_back({"data", V::make_owned_string(encode_base64(wav_bytes))});
  ia_members.push_back({"format", V::make_string("wav")});
  std::vector<V> content_parts;
  content_parts.push_back(V::make_object({{"type", V::make_string("input_audio")},
                                          {"input_audio",
                                           V::make_object(std::move(ia_members))}}));
  messages.push_back(V::make_object({{"role", V::make_string("user")},
                                     {"content", V::make_array(std::move(content_parts))}}));
  members.push_back({"messages", V::make_array(std::move(messages))});
  members.push_back({"max_tokens", V::make_int(steps)});
  if (f.has_temperature) members.push_back({"temperature", V::make_double(f.temperature)});
  members.push_back({"stream", V::make_bool(false)});
  V body = V::make_object(std::move(members));
  return TranscriptionRequest{std::move(body), f.response_format};
}

std::pair<std::string, std::string> split_language_tag(std::string_view text) {
  // "language " + CapitalizedWord, the word ending exactly where an
  // uppercase letter begins the transcript (or at end of input).
  constexpr std::string_view kPrefix = "language ";
  if (text.size() <= kPrefix.size() || text.substr(0, kPrefix.size()) != kPrefix)
    return {"", std::string(text)};
  size_t i = kPrefix.size();
  if (i >= text.size() || text[i] < 'A' || text[i] > 'Z') return {"", std::string(text)};
  ++i;
  while (i < text.size() && text[i] >= 'a' && text[i] <= 'z') ++i;
  if (i < text.size() && (text[i] < 'A' || text[i] > 'Z')) return {"", std::string(text)};
  std::string tag(text.substr(0, i));
  std::string body = i < text.size() ? std::string(text.substr(i)) : "";
  while (!body.empty() && (body.front() == ' ' || body.front() == '\t'))
    body.erase(body.begin());
  return {std::move(tag), std::move(body)};
}

std::string format_transcription_text(std::string_view text) {
  auto [tag, body] = split_language_tag(text);
  if (tag.empty() || body.empty()) return std::string(text);
  return tag + "\n" + body;
}

TranscriptionRequest transcription_chat_body(const std::vector<MultipartPart>& parts,
                                             const std::string& model_id) {
  const TranscriptionFields f = parse_transcription_fields(parts, model_id);
  return transcription_body_for_wav(f, f.file_bytes);
}
std::vector<TranscriptionRequest> transcription_segment_bodies(
    const std::vector<MultipartPart>& parts, const std::string& model_id,
    int64_t token_budget) {
  const TranscriptionFields f = parse_transcription_fields(parts, model_id);
  // Single-body fast path: unknown windows, and clips that fit, ride
  // today's path byte-identically (same probe, same seconds, same body).
  const AudioInput probe = prepare_audio_bytes(
      reinterpret_cast<const uint8_t*>(f.file_bytes.data()), f.file_bytes.size(), "file");
  // Prompt overhead plus slack against the engine window: template, the
  // system prompt and the span markers. The per-segment admission check
  // re-verifies with the rendered prompt as backstop.
  constexpr int64_t kSegmentReserve = 128;
  if (token_budget <= 0 || probe.tokens + kSegmentReserve <= token_budget)
    return {transcription_body_for_wav(f, f.file_bytes)};
  // Whole-chunk segments of C chunks (13C audio tokens), the last taking
  // the tail: chunk-aligned splits keep token counts exact, so the parts
  // sum to the admission total.
  const int64_t chunks =
      (token_budget - kSegmentReserve - kAudioChunkTokens) / kAudioChunkTokens;
  if (chunks < 1)
    throw MultipartError(
        "the audio cannot fit one engine forward even segmented; submit a shorter clip",
        "file", 400, "context_length_exceeded");
  const int64_t seg_frames = chunks * kAudioChunkFrames;
  std::vector<TranscriptionRequest> out;
  int64_t first = 0;
  const int64_t total = probe.frames;
  while (total - first > seg_frames) {
    const std::string wav = audio::encode_wav_16mono_16k(
        probe.pcm.data() + static_cast<size_t>(first) * kAudioHop,
        static_cast<size_t>(seg_frames) * kAudioHop);
    out.push_back(transcription_body_for_wav(f, wav));
    first += seg_frames;
  }
  const int64_t tail_frames = total - first;
  const std::string wav = audio::encode_wav_16mono_16k(
      probe.pcm.data() + static_cast<size_t>(first) * kAudioHop,
      static_cast<size_t>(tail_frames) * kAudioHop);
  out.push_back(transcription_body_for_wav(f, wav));
  return out;
}
}  // namespace dgpp::serve
