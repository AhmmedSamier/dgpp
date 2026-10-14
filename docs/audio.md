# Audio inputs

The ASR family (Qwen3-ASR) reaches this server through
`POST /v1/chat/completions` content parts, on the same admission spine as
[images](vision.md): base64 audio (WAV, MP3, FLAC or Ogg Vorbis) is decoded
to 16 kHz hop-aligned mono at admission, its encoder token count is trained into the prompt as a delimiter
span, and the tower re-featurizes the PCM identically on every rank. No URL
fetching and no Python runtime are involved.

The tower's fp32 CPU oracle is in (`models/qwen_asr/audio_tower_reference`,
pinned to a torch replay of the checkpoint's own encoder within 2.4e-05),
and the language half is served: `QwenAsrModel` (`models/qwen_asr/`) is the
checkpoint's plain Qwen3 decoder (28x2048, GQA 16/8, q/k-norm, full 128-wide
rotary, bf16 SwiGLU, the tied head aliasing the embedding) on the shared
GLM-4.7 attention/pool machinery, with the tower's projected rows spliced
into the prefill embeddings at each audio span. `dgpp-serve` detects the
`Qwen3ASR*` architecture (`qwen3_asr` family) and renders audio parts through
`QwenAsrAudioFrontend`, so `GET /v1/models` lists `"audio"` for it. The
remaining frontend milestone is an FFT frontend (the host DFT is exact
and quadratic); the device tower milestone is served (BF16, golden-pinned
— see below).

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
message text are rejected as conflicting with the decoded audio.

## Serving it

The snapshot serves as downloaded — no preparation step. Upstream ships
no `tokenizer.json` or `chat_template.jinja`, so the server reads the
slow spelling directly: `Tokenizer::load` falls back to `vocab.json` +
`merges.txt` + `tokenizer_config.json` (gated on the `Qwen2Tokenizer`
class, added tokens registered special exactly as the reference
`add_special_tokens` builder does), and `ChatTemplate::load` falls back
to the `chat_template.json` envelope. Both fallbacks are pinned
byte-identical to the reference-built files
(`tests/unit/tokenizer_load_test.cpp`, `tests/data/slow_tokenizer/`,
`tests/data/slow_template/`).

```bash
./build/dgpp-serve --model Qwen/Qwen3-ASR-1.7B --port 18081
```

`--checkpoint-dir` at a raw snapshot works the same way.

### Cluster deployment

`deploy/cluster_qwen3-asr-1.7b_bf16_w1.example.json` is the ready-to-run
single-node shape (eight slots, the 8192-token pool, port 8000, served
id `Qwen3-ASR-1.7B` — OpenWebUI's STT model setting must name the
alias, not the repo):

```bash
python3 scripts/download_model.py --config deploy/cluster_qwen3-asr-1.7b_bf16_w1.example.json
scripts/dgpp-cluster up --config deploy/cluster_qwen3-asr-1.7b_bf16_w1.example.json
```

The snapshot needs no preparation: the server reads the upstream files
directly (above), so peers sync the raw snapshot and boot straight
into it.

`GET /v1/models` then lists `"audio"` in `input_modalities`, and a user
message with an `input_audio` (base64 WAV) or `audio_url` (data URI) part
transcribes: admission decodes the WAV, the frontend lets the template emit
its single `<|audio_pad|>` per part and expands it to the clip's encoder
row count post-encode, the tower splices its rows into the prefill, and the
decoder generates greedily. Verified end to end against the upstream demo
clip (15 s English speech → 211 prompt tokens → 49 fluent tokens), and
gated on LibriSpeech test-clean at **6.7% WER** (2620/2620 utterances,
`benchmarks/results/2026-10-14-asr-wer/`, deletion-heavy short-function-word
profile — content words land).
The device tower (`models/qwen_asr/tower.{hpp,cpp}`, kernels in
`kernels/qwen_asr_tower.{hpp,cu}`) runs the encoder in BF16: direct-fused
stride-2 convs, `matmul_linear_bf16` linears, vision-kernel LayerNorm /
exact-erf GELU / residual / softmax, and 104-row-windowed bidirectional
attention through the same batched-GEMM sequence as the vision tower.
`tests/cuda/qwen_asr_tower_cuda_test` pins it to the torch oracle within
5e-2 absolute (measured worst 3.0e-02, mean 2.7e-04 on the 37 s
window-crossing clip); the demo transcription is byte-identical between
the two paths. The host log-mel frontend is a mixed-radix 16x25 FFT
(`src/audio/fft400.{hpp,cpp}`) in double precision - bit-identical
features to the quadratic DFT it replaced (golden worst |delta|
unchanged at 9.6e-05) at ~2.3x the speed (5.6 → 2.4 ms per audio second);
the 201x128 mel projection is now the frontend's dominant cost.

## Performance

The demo clip's prefill fell 25.5 s → 2.8 s → 149 ms across the two
performance changes below (same 49-token transcription throughout). The
remaining 149 ms is roughly host log-mel (86 ms), device tower (~30 ms)
and LM walk (~35 ms); the mixed-radix FFT frontend (§Serving it) takes
the log-mel share to ~36 ms, leaving the mel projection itself as the
frontend's cost.

## Transcriptions

`POST /v1/audio/transcriptions` speaks the OpenAI transcription API over
the same engine: multipart/form-data carrying the audio `file` plus
`model`, `language`, `prompt`, `response_format` (`json` or `text`) and
`temperature`. The `file` may be WAV or MP3 (OpenWebUI and the OpenAI SDKs
submit mp3 chunks). The route validates the parts, folds them into the
chat-completions body the engine already serves (the file as one
`input_audio` part, the prompt as the system message, a
duration-proportional `max_tokens` - WAV from a header sniff, MP3 from the
decoded duration), and answers `{"text": ...}` or the
raw transcript with no usage block. The checkpoint prefixes every
transcript with its language tag glued onto the text (`language
EnglishHmm. ...`); the server splits the tag onto its own line above the
body (`split_language_tag`, camelCase boundary — 2619/2620 test-clean
outputs; anything else passes through verbatim rather than risk eating
speech).

Long-form clips (past one engine forward, ~2.5 minutes at the default
window) transcribe in whole-chunk segments: one scheduler request per
segment, strictly in order through a transcription chain, each with its
own duration-proportional `max_tokens`, answered once with the
space-joined transcripts. Each segment formats its own tag above its
body, so long transcripts repeat the tag per segment. Chunk-aligned splits keep token counts exact;
a trailing partial chunk rides the last segment. A window too small for one chunk plus
the prompt reserve refuses with a 400 naming the file.

Two checkpoint facts shape the semantics. `language` is validated as a
string and documented, not rendered: the checkpoint auto-detects and
prefixes its own output with `language <NAME>`, and a live probe showed
system text does not condition this model. And the raw transcript is
returned verbatim for both formats: the model glues the language prefix
straight onto the text with no `<asr_text>` marker to split on. Only
`json` and `text` are accepted as `response_format`; anything else is a
400 naming the field, as is an undecodable file (naming `file`) or a
wrong model (404). Errors reuse the chat contract otherwise.

The first-request tax deserves its own note: the first cublasLt use in a
process pays lazy library loads and kernel JIT (profiled at 2.6 s on a
first 96-row forward vs 34 ms warm). The model constructor warms one dummy
tower piece and one 8-token forward, moving ~2.5 s from the first request
to boot (construction 1.6 s → 4.2 s). Per-shape plan queries after that
are microseconds.

## Where the work lives

| Piece                                              | File |
| -------------------------------------------------- | ---- |
| Span record, geometry, validators | `src/common/audio_input.hpp` |
| WAV container decode, mono 16 kHz | `src/audio/wav_decode.{hpp,cpp}` |
| MP3 stream decode (vendored minimp3) | `src/audio/mp3_decode.{hpp,cpp}` |
| FLAC + Ogg Vorbis decode (vendored dr_flac, stb_vorbis) | `src/audio/compressed_decode.{hpp,cpp}` |
| 400-point mixed-radix FFT for the frontend | `src/audio/fft400.{hpp,cpp}` |
| FLAC + Ogg Vorbis decode (vendored dr_flac, stb_vorbis) | `src/audio/compressed_decode.{hpp,cpp}` |
| Log-mel frontend (oracle path) | `src/audio/mel_frontend.{hpp,cpp}` |
| Content-part admission | `src/serve/audio_inputs.{hpp,cpp}` |
| Prompt rewrite (delimiter span) | `src/serve/audio_frontend.{hpp,cpp}` |
| Request/queue/engine seams | `src/sched/scheduler.hpp`, `src/engine/*_engine.hpp` |
| Tower config + fp32 oracle (CUDA gate) | `src/models/qwen_asr/` |
| Language half (served decoder + audio splice) | `src/models/qwen_asr/model.{hpp,cpp}` |
| Serve family + audio frontend | `apps/dgpp_serve.cpp`, `src/serve/qwen_asr_audio_frontend.*` |

The frontend mirrors `Qwen3ASRFeatureExtractor` exactly (center-padded
400/160 STFT with a periodic Hann, torch's reflect edge; the 201x128 Slaney
bank; log10 with the per-clip 8-decay floor; `(x + 4) / 4`), and
`tests/unit/audio_frontend_test.cpp` pins it to tensors dumped by
`tools/qwen3_asr_feature_goldens.py`. Clips under 0.5 s are zero-extended to
8000 samples, matching the reference processor. The host DFT is exact and
quadratic; the tower milestone replaces it with an FFT kernel held to the
same goldens.

Scheduling for now: audio requests run monolithic (no chunked prefill, no
prefix-cache attach - neither snapshots across an audio span yet), and one
request may carry many clips.

Admission accepts up to 16384 audio tokens per request, but the engine
stage holds only its prefill window (2048 rows ≈ 2.5 minutes at the
default window): longer clips are refused at admission with a 400 naming
the audio (`context_length_exceeded`), before any engine work. An engine
op that throws fails the whole service by design (DESIGN §9 item 6), so
run `dgpp-serve` under a supervisor — `deploy/dgpp-serve.example.service`
restarts it on failure.
