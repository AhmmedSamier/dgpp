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
