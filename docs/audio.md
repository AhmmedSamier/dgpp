# Audio inputs

The ASR family (Qwen3-ASR) reaches this server through
`POST /v1/chat/completions` content parts, on the same admission spine as
[images](vision.md): a base64 WAV is decoded to 16 kHz hop-aligned mono at
admission, its encoder token count is trained into the prompt as a delimiter
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
`QwenAsrAudioFrontend`, so `GET /v1/models` lists `"audio"` for it. The open
milestones are the CUDA tower (the oracle path runs fp32 on the host today)
and an FFT frontend (the host DFT is exact and quadratic).

## Request format

Either OpenAI-style `input_audio` (raw base64) or an `audio_url` data URI, in
a user message alongside any text:

```python
{"role": "user", "content": [
    {"type": "text", "text": "Transcribe."},
    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
]}
```

```python
{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + b64}}
```

Accepted payloads are PCM (8/16/24/32-bit) or IEEE-float WAV; anything else -
other containers, mp3, opus - is refused by naming the format, never
guessed at. Decoded clips may be up to half an hour (20 MiB base64 per part,
256 MiB per request), and stereo or off-rate audio is downmixed and resampled
(linear) to 16 kHz mono.

The prompt contract is the checkpoint's own: `...` (151669) +
`...` (151676) * N + `...` (151670), with N set by the
encoder's chunking - 13 tokens per full 10-second chunk,
`audio_token_count` in `common/audio_input.hpp`. Markers typed directly into
message text are rejected as conflicting with the decoded audio.

## Serving it

The snapshot is serve-ready with two generated files beside it (neither
ships upstream): the `tokenizer.json` dgpp's Tokenizer loads, and the
`chat_template.jinja` the frontend renders.

```bash
SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/<rev>
Q3=~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<rev>
python3 tools/qwen_asr_tokenizer.py --snapshot "$SNAP" \
    --qwen3-tokenizer "$Q3/tokenizer.json" --out "$SNAP/tokenizer.json"
python3 -c "import json
print(json.load(open('$SNAP/chat_template.json'))['chat_template'], end='')" \
    > "$SNAP/chat_template.jinja"
./build/dgpp-serve --checkpoint-dir "$SNAP" --port 18081
```

`GET /v1/models` then lists `"audio"` in `input_modalities`, and a user
message with an `input_audio` (base64 WAV) or `audio_url` (data URI) part
transcribes: admission decodes the WAV, the frontend lets the template emit
its single `<|audio_pad|>` per part and expands it to the clip's encoder
row count post-encode, the tower splices its rows into the prefill, and the
decoder generates greedily. Verified end to end against the upstream demo
clip (15 s English speech → 211 prompt tokens → 49 fluent tokens).
The device tower (`models/qwen_asr/tower.{hpp,cpp}`, kernels in
`kernels/qwen_asr_tower.{hpp,cu}`) runs the encoder in BF16: direct-fused
stride-2 convs, `matmul_linear_bf16` linears, vision-kernel LayerNorm /
exact-erf GELU / residual / softmax, and 104-row-windowed bidirectional
attention through the same batched-GEMM sequence as the vision tower.
`tests/cuda/qwen_asr_tower_cuda_test` pins it to the torch oracle within
5e-2 absolute (measured worst 3.0e-02, mean 2.7e-04 on the 37 s
window-crossing clip); the demo transcription is byte-identical between
the two paths. The remaining prefill cost is the host log-mel frontend
(exact, quadratic) and the LM walk itself — the FFT frontend is the open
compute milestone.

## Performance

The demo clip's prefill fell 25.5 s → 2.8 s → 149 ms across the two
performance changes below (same 49-token transcription throughout). The
remaining 149 ms is roughly host log-mel (86 ms), device tower (~30 ms)
and LM walk (~35 ms); the FFT frontend would take the first of those.

## Transcriptions

`POST /v1/audio/transcriptions` speaks the OpenAI transcription API over
the same engine: multipart/form-data carrying the audio `file` plus
`model`, `language`, `prompt`, `response_format` (`json` or `text`) and
`temperature`. The route validates the parts, folds them into the
chat-completions body the engine already serves (the file as one
`input_audio` part, the prompt as the system message, a
duration-proportional `max_tokens`), and answers `{"text": ...}` or the
raw transcript with no usage block.

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
