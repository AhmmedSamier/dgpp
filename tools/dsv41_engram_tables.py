#!/usr/bin/env python3
"""Derive DeepSeek-V4.1-Flash's Engram hash constants into a sidecar the loader reads.

The Engram module (docs/deepseek_v41_flash_plan.md §1.6) hashes n-grams of
COMPRESSED token ids: every token's decoded text is normalized (NFKC, NFD,
strip accents, lowercase, collapse whitespace, strip) and tokens that
normalize alike share one class; the class count must equal the config's
`engram_compressed_vocab_size` (99,092) because every hash multiplier is
derived from it. The per-(layer, n-gram, head) bucket sizes are consecutive
primes above `engram_vocab_size - 1`, handed out in order and never reused;
the multipliers come from numpy's PCG64 seeded per layer. All of it is the
reference implementation's (`inference/engram.py`), reproduced here with
the same libraries so the values are the training run's, and written once
as JSON beside the checkpoint:

  {
    "format": "dgpp-dsv41-engram-tables-1",
    "tokenizer_sha256": ..., "vocab_size": 129280,
    "compressed_vocab_size": 99092, "pad_id": 2, "pad_class": ...,
    "layer_ids": [1, 14], "max_ngram_size": 4, "n_heads": 8,
    "num_embeddings": [384006168, 384016682],
    "primes": [[[p ...] x n_heads] x (max_ngram_size - 1)] x layers,
    "offsets": the same shape (row offset of each bucket range),
    "multipliers": [[m0, m1, m2, m3]] x layers,
    "token_map": [class of token 0, class of token 1, ...]
  }

Usage:
  python3 tools/dsv41_engram_tables.py --model-dir <snapshot or repo dir> [--out FILE] [--check]

The venv needs tokenizers and numpy (requirements-tools.txt); sympy is used
for the primality check when present, else a deterministic Miller-Rabin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SIDECAR_NAME = "dgpp_engram_tables.json"
FORMAT = "dgpp-dsv41-engram-tables-1"


def is_prime(n: int) -> bool:
    try:
        from sympy import isprime  # type: ignore

        return bool(isprime(n))
    except ImportError:
        pass
    # Deterministic Miller-Rabin for n < 3.3e24 (the first 12 primes as bases).
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer_path: Path) -> tuple[list[int], int]:
    """The reference's build_compressed_token_map over the raw tokenizers backend."""
    from tokenizers import Regex, Tokenizer, normalizers

    backend = Tokenizer.from_file(str(tokenizer_path))
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    vocab_size = backend.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = [0] * vocab_size
    for token_id in range(vocab_size):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
            if key is None:
                raise ValueError(f"token id {token_id} has no string form")
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids: list[int], max_ngram_size: int, compressed_vocab: int) -> list[list[int]]:
    max_long = np.iinfo(np.int64).max
    bound = max(1, (int(max_long) // compressed_vocab) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append([int(v) * 2 + 1 for v in values])
    return rows


def derive(model_dir: Path) -> dict:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config") or config
    layer_ids = [int(x) for x in text["engram_layer_ids"]]
    max_ngram = int(text["engram_max_ngram_size"])
    n_heads = int(text["engram_n_heads"])
    vocab_base = int(text["engram_vocab_size"])
    num_embeddings = [int(x) for x in text["engram_num_embeddings"]]
    pad_id = int(text["engram_pad_token_id"])
    expected_classes = int(text["engram_compressed_vocab_size"])
    vocab_size = int(text["vocab_size"])
    if len(num_embeddings) != len(layer_ids):
        raise ValueError("engram_num_embeddings and engram_layer_ids disagree in length")

    tokenizer_path = model_dir / "tokenizer.json"
    token_map, classes = build_compressed_token_map(tokenizer_path)
    if len(token_map) != vocab_size:
        raise ValueError(f"tokenizer has {len(token_map)} ids, config vocab_size {vocab_size}")
    if classes != expected_classes:
        raise ValueError(
            f"compressed vocab has {classes} classes, config engram_compressed_vocab_size {expected_classes}: "
            "the hash multipliers would differ from training"
        )

    primes: list[list[list[int]]] = []
    offsets: list[list[list[int]]] = []
    seen: set[int] = set()
    for layer_index, rows in enumerate(num_embeddings):
        per_ngram_p, per_ngram_o = [], []
        running = 0
        for _n in range(max_ngram - 1):
            sizes, offs = [], []
            current = vocab_base - 1
            for _h in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
                offs.append(running)
                running += current
            per_ngram_p.append(sizes)
            per_ngram_o.append(offs)
        if running != rows:
            raise ValueError(
                f"layer {layer_ids[layer_index]}: the bucket ranges total {running} rows, "
                f"engram_num_embeddings says {rows}"
            )
        primes.append(per_ngram_p)
        offsets.append(per_ngram_o)

    multipliers = compute_hash_multipliers(layer_ids, max_ngram, classes)
    return {
        "format": FORMAT,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "vocab_size": vocab_size,
        "compressed_vocab_size": classes,
        "pad_id": pad_id,
        "pad_class": token_map[pad_id],
        "layer_ids": layer_ids,
        "max_ngram_size": max_ngram,
        "n_heads": n_heads,
        "num_embeddings": num_embeddings,
        "primes": primes,
        "offsets": offsets,
        "multipliers": multipliers,
        "token_map": token_map,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True, help="directory holding config.json and tokenizer.json")
    parser.add_argument("--out", type=Path, help=f"output path (default: <model-dir>/{SIDECAR_NAME})")
    parser.add_argument("--check", action="store_true", help="compare against an existing sidecar instead of writing")
    args = parser.parse_args(argv)
    model_dir = args.model_dir.expanduser()
    out = (args.out or model_dir / SIDECAR_NAME).expanduser()
    tables = derive(model_dir)
    summary = (
        f"classes {tables['compressed_vocab_size']}, pad class {tables['pad_class']}, "
        f"{sum(len(h) for n in tables['primes'] for h in n)} primes, layers {tables['layer_ids']}, "
        f"multipliers {tables['multipliers']}"
    )
    if args.check:
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing != tables:
            print(f"sidecar {out} differs from the derivation ({summary})", file=sys.stderr)
            return 1
        print(f"sidecar {out} matches ({summary})")
        return 0
    out.write_text(json.dumps(tables, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {out}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
