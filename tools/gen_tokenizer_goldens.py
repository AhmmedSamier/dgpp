#!/usr/bin/env python3
"""Generates tests/data/glm_tokenizer_goldens.jsonl — the differential
goldens for the GLM tokenizer (M6 Stage 3).

The GOLDEN SOURCE is HF tokenizers 0.23.1 (the pinned reference — the
venv at /tmp/opencode/tokref) applied to this script's case list. The
C++ gate (tests/host/glm_tokenizer_test.cpp) loads the SAME corpus and
asserts byte-exact encode/decode parity, refusing to run against a
tokenizer.json whose FNV-1a-64 revision hash differs from the corpus
header (the revision key; the C++ and this script share the hash
definition).

Case selection: the corpus covers the pretokenizer's decision surface —
contractions in both cases, the whitespace alternatives (space-prefix,
trailing-space, newline runs, tabs before letters, nbsp), digit
clumping, Unicode letter/number classes, multi-script text, emoji
(ZWJ sequences, skin tones), code fences, and the added tokens
standalone and inline. Deliberately avoids codepoints added to Unicode
after 13.0 so table-version skew between this generator (Python 3.12 /
Unicode 15.0) and the C++ tables cannot flake the gate (see
tools/gen_unicode_tables.py).

Regenerating: /tmp/opencode/tokref/bin/python tools/gen_tokenizer_goldens.py
"""
import glob
import json
import sys

MODEL = "unsloth/GLM-5.3-Flash-FP8"


def fnv1a64(data: bytes) -> int:
    # Matches the C++ (glm_tokenizer.cpp + glm_loader.cpp's house hash):
    # the project's established basis, not the FNV standard basis.
    h = 1469598103934665603
    for b in data:
        h = (h ^ b) * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return h


THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"
USER = "<|" + "user|>"
ASSIST = "<|" + "assistant|>"
EOS = "<|" + "end" + "of" + "text|>"

CASES = [
    # The fabric-run anchors (the M6 Stage 1d/2 prompts and generations).
    "The capital of France is",
    " Paris",
    ".",
    " In",
    " French,",
    " is",
    " spelled",
    # Contractions: every form, both cases, embedded.
    "I don't think it's CAN'T 'LL 'Re we're I'm 'll 'Ve 'd 'm 'T 'S",
    "can't won't shouldn't they're we've y'all",
    # Whitespace alternatives.
    "  5", " a", "   abc", "\tabc", "\n\n  x", "!!!\n", "a \n b",
    "abc   ", " \n", "\r\n\r\n", "hello\n\nworld\n", "x\u00a0y",
    # Digit clumping and numeric forms.
    "1234567", "0000", "0.5", "1,000,000.50", "123456789012345",
    # Unicode letter/number classes (Nl, No included — not just ASCII
    # digits).
    "½ Ⅻ ² ⅓",
    # Multi-script text.
    "你好，世界",
    "Привет мир",
    "Γεια σου",
    "สวัสดีครับ",
    "שלום עולם",
    "مرحبا بالعالم",
    "naïve café Straße",
    # Emoji (pictographic, skin tone, ZWJ sequence).
    "emoji 👍🏽 test",
    "family 👨‍👩‍👧‍👦 walk",
    # Code and punctuation runs.
    "code ```python\nprint(1)\n``` end",
    "!!!???***", "a-b_c+d=e",
    # Added tokens: standalone, inline, and a template-shaped run.
    THINK_OPEN,
    THINK_CLOSE,
    USER,
    ASSIST,
    EOS,
    "a" + THINK_OPEN + "b",
    "x" + THINK_OPEN + " r" + THINK_CLOSE + "y" + USER + "hi" + ASSIST,
    EOS + THINK_OPEN + " e" + THINK_CLOSE + USER + "u" + ASSIST,
    "before " + USER + " after",
    # Degenerate inputs.
    "", " ", "  ", "\n", "x",
    # Realistic mixed sample.
    "def add(a, b):\n    return a + b  # inline comment\n\nprint(add(2, 3))\n",
    "The quick brown fox jumps over the lazy dog. 0123456789 !@#$%^&*()",
]


def main():
    out_path = (
        sys.argv[1] if len(sys.argv) > 1 else
        "tests/data/glm_tokenizer_goldens.jsonl"
    )
    snap = glob.glob(f"/home/user/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/*/")
    if not snap:
        sys.exit(f"model snapshot for {MODEL} not found in the HF cache")
    tok_path = snap[0] + "tokenizer.json"
    with open(tok_path, "rb") as f:
        raw = f.read()
    import tokenizers
    tok = tokenizers.Tokenizer.from_file(tok_path)

    with open(out_path, "w", encoding="utf-8") as f:
        header = {
            "model": MODEL,
            "tokenizers": tokenizers.__version__,
            "revision_hash": f"{fnv1a64(raw):016x}",
            "cases": len(CASES),
        }
        f.write(json.dumps(header) + "\n")
        for text in CASES:
            ids = tok.encode(text).ids
            # HF decode() defaults to skip_special_tokens=True — special
            # added tokens (the EOS here) decode to nothing. The gate's
            # round trip pins the VERBATIM semantics explicitly.
            decoded = tok.decode(ids, skip_special_tokens=False)
            if decoded != text:
                sys.exit(f"HF verbatim round-trip failed for {text!r} -> {decoded!r}")
            f.write(json.dumps({"text": text, "ids": ids}) + "\n")
    print(f"wrote {out_path}: {len(CASES)} cases, revision "
          f"{header['revision_hash']} (tokenizers {tokenizers.__version__})")


if __name__ == "__main__":
    main()
