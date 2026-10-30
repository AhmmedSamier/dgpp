# Audio inputs

Audio reaches this server through `POST /v1/chat/completions` content parts
and `POST /v1/audio/transcriptions`, on the same admission spine as
[images](vision.md): base64 audio (WAV, MP3, FLAC or Ogg Vorbis) is decoded
to 16 kHz hop-aligned mono at admission, its encoder token count is written
into the prompt as a delimiter span, and the serving model re-featurizes the
PCM identically on every rank. No URL fetching and no Python runtime are
involved.

The transport, admission and frontend stack below is complete; **no served
family has an audio encoder yet** — the first one (NVIDIA Nemotron-3.5
Transcribe) is in progress, and every frontend refuses `input_audio`/
`audio_url` parts until a family declares `supports_audios()`. The
delimiter-span geometry and the host log-mel frontend are pinned to the
reference processor of the Qwen3-ASR family
(`tools/qwen3_asr_feature_goldens.py`, held within 1e-4): the first served
family re-pins both to its own preprocessor before enabling audio.

## Request format

Either OpenAI-style `input_audio` (raw base64) or an `audio_url` data URI, in
a user message alongside any text:

```python
{"role": "user", "content": [
    {"type": "text", "text": "Transcribe."},
    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
]}
```

MP3 submits the same shape with `"format": "mp3"`:

```python
{"type": "input_audio", "input_audio": {"data": b64_mp3, "format": "mp3"}}
```

FLAC and Ogg Vorbis submit under `"format": "flac"` and `"format": "ogg"`.
```python
{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + b64}}
```

```python
{"type": "audio_url", "audio_url": {"url": "data:audio/mpeg;base64," + b64_mp3}}
```

Accepted payloads are PCM (8/16/24/32-bit) or IEEE-float WAV, MP3
(MPEG-1/2/2.5, layers 1-3, 8-48 kHz mono/stereo, decoded by the vendored
minimp3 - `src/audio/mp3_decode.{hpp,cpp}`), FLAC (any rate the admission
bounds accept, vendored dr_flac - `src/audio/compressed_decode.{hpp,cpp}`),
and Ogg Vorbis (same module, vendored stb_vorbis); anything else - other
containers, opus - is refused by naming the format, never guessed at.
The container is sniffed from the bytes, not the filename or MIME: a WAV
labeled mp3 still decodes, and undecodable bytes are a 400 naming the
field. MP3 clips with a LAME/Xing gapless tag drop their encoder
delay/padding (`gapless_trim`); untagged MP3 keeps ~2k samples of lead as
audio rather than trimmed, so such clips report slightly longer than
their WAV masters. Decoded clips may be up to half an hour (20 MiB base64 per part,
256 MiB per request), and stereo or off-rate audio is downmixed and resampled
(linear) to 16 kHz mono.

The prompt contract is the checkpoint's own: `...` (151669) +
`...` (151676) * N + `...` (151670), with N set by the
encoder's chunking - 13 tokens per full 10-second chunk,
`audio_token_count` in `common/audio_input.hpp`. Markers typed directly into
message text are rejected as conflicting with the decoded audio. These are
the reference processor's constants; a new family supplies its own delimiter
ids (from its config, through `audio_token_ids()`) and its own chunk
geometry.

## Raw snapshots

Checkpoints that ship no `tokenizer.json` or `chat_template.jinja` serve as
downloaded — no preparation step: `Tokenizer::load` falls back to
`vocab.json` + `merges.txt` + `tokenizer_config.json` (gated on the
`Qwen2Tokenizer` class, added tokens registered special exactly as the
reference `add_special_tokens` builder does), and `ChatTemplate::load` falls
back to the `chat_template.json` envelope. Both fallbacks are pinned
byte-identical to the reference-built files
(`tests/unit/tokenizer_load_test.cpp`, `tests/data/slow_tokenizer/`,
`tests/data/slow_template/`).

## Chunked uploads

Streaming uploaders (OpenWebUI's STT path, the OpenAI SDKs) frame multipart
bodies with `Transfer-Encoding: chunked` and no `Content-Length`; the server
de-chunks them incrementally (`HttpServer::dechunk_step`) under the same
body cap. Unknown codings stay a 501, and Content-Length beside
Transfer-Encoding (the smuggling shape) is a 400.

## Transcriptions

`POST /v1/audio/transcriptions` speaks the OpenAI transcription API over
the same engine: multipart/form-data carrying the audio `file` plus
`model`, `language`, `prompt`, `response_format` (`json` or `text`) and
`temperature`. The `file` may be WAV, MP3, FLAC or Ogg (OpenWebUI and the
OpenAI SDKs submit mp3 chunks). The route validates the parts, folds them
into the chat-completions body the engine already serves (the file as one
`input_audio` part, the prompt as the system message, a
duration-proportional `max_tokens` - from the container header or the
decoded duration), rides the normal preparation pool and admission path,
and answers `{"text": ...}` or the raw transcript with no usage block.
The reference checkpoint prefixes every transcript with its language tag
glued onto the text (`language EnglishHmm. ...`); the server splits the tag
onto its own line above the body (`split_language_tag`, camelCase boundary —
2619/2620 test-clean outputs on the reference; anything else passes through
verbatim rather than risk eating speech).

Long-form clips (past one engine forward, ~2.5 minutes at the default
window) transcribe in whole-chunk segments: one scheduler request per
segment, strictly in order through a transcription chain, each with its
own duration-proportional `max_tokens`, answered once with the
space-joined transcripts. Each segment formats its own tag above its
body, so long transcripts repeat the tag per segment. Chunk-aligned splits keep token counts exact;
a trailing partial chunk rides the last segment. A window too small for one chunk plus
the prompt reserve refuses with a 400 naming the file.

The `language` field is validated as a string and documented, not
rendered: the reference checkpoint auto-detects the language and prefixes
its own output, and a live probe showed system text does not condition it.
Only `json` and `text` are accepted as `response_format`; anything else is
a 400 naming the field, as is an undecodable file (naming `file`) or a
wrong model (404). Errors reuse the chat contract otherwise.

## Scheduling and admission

Audio requests run monolithic (no chunked prefill, no prefix-cache attach -
neither snapshots across an audio span yet), and one request may carry many
clips.

Admission accepts up to 16384 audio tokens per request, but the engine
stage holds only its prefill window (the family's `prefill_audio_token_limit`):
longer clips are refused at admission with a 400 naming the audio
(`context_length_exceeded`), before any engine work. An engine
op that throws fails the whole service by design (DESIGN §9 item 6), so
run `dgpp-serve` under a supervisor — `deploy/dgpp-serve.example.service`
restarts it on failure.

## Where the work lives

| Piece                                              | File |
| -------------------------------------------------- | ---- |
| Span record, geometry, validators | `src/common/audio_input.hpp` |
| WAV container decode, mono 16 kHz | `src/audio/wav_decode.{hpp,cpp}` |
| MP3 stream decode (vendored minimp3) | `src/audio/mp3_decode.{hpp,cpp}` |
| FLAC + Ogg Vorbis decode (vendored dr_flac, stb_vorbis) | `src/audio/compressed_decode.{hpp,cpp}` |
| 400-point mixed-radix FFT for the frontend | `src/audio/fft400.{hpp,cpp}` |
| Log-mel frontend (reference-pinned oracle) | `src/audio/mel_frontend.{hpp,cpp}` |
| Content-part admission | `src/serve/audio_inputs.{hpp,cpp}` |
| Prompt rewrite (delimiter span) | `src/serve/audio_frontend.{hpp,cpp}` |
| Transcription route, multipart parser, chains | `src/serve/audio_transcriptions.{hpp,cpp}`, `src/serve/generation_service.*` |
| Chunked request bodies | `src/serve/http_server.{hpp,cpp}` |
| Request/queue/engine seams | `src/sched/scheduler.hpp`, `src/engine/*_engine.hpp` |

The mel frontend mirrors `Qwen3ASRFeatureExtractor` exactly (center-padded
400/160 STFT with a periodic Hann, torch's reflect edge; the 201x128 Slaney
bank; log10 with the per-clip 8-decay floor; `(x + 4) / 4`), and
`tests/unit/audio_frontend_test.cpp` pins it to tensors dumped by
`tools/qwen3_asr_feature_goldens.py`. Clips under 0.5 s are zero-extended to
8000 samples, matching the reference processor. A new family forks this
module against its own preprocessor's goldens, the same oracle-first ladder
the vision path uses.
