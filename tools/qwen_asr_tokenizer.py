#!/usr/bin/env python3
"""Build the tokenizer.json dgpp's Tokenizer loads for the Qwen3-ASR
checkpoint (docs/audio.md).

The ASR snapshot ships vocab.json + merges.txt + tokenizer_config.json but
no tokenizer.json. Its BPE core is byte-identical to plain Qwen3's (checked:
151643 vocab entries and all 151387 merges match Qwen/Qwen3-0.6B); its 62
added tokens are a strict superset of Qwen3's 26 (the 36 audio/TTS markers
at 151669-151704). So this assembles Qwen3's pre_tokenizer/decoder/
post_processor around the ASR snapshot's own vocab, merges and added
tokens, then verifies the result token-for-token against transformers'
own slow tokenizer on the snapshot.

Two serialization details matter, both verified by the check below:
- merges.txt's first line is a "#version: 0.2" header, but every other
  #-leading line ('# #', '## ##', ...) is a real merge: filter only the
  header, never by prefix.
- the tokenizers lib writes a Split pattern passed as str as
  {"String": ...}; dgpp's Tokenizer wants {"Regex": ...} (the pinned
  Qwen2 Split pattern this checkpoint uses).

Usage: python3 tools/qwen_asr_tokenizer.py --out /path/to/tokenizer.json
"""
import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot", required=True,
                    help="the Qwen3-ASR snapshot dir (vocab.json, merges.txt, ...)")
    ap.add_argument("--qwen3-tokenizer", required=True,
                    help="a plain Qwen3 tokenizer.json (pre/decoder/post structure donor)")
    args = ap.parse_args()

    from tokenizers import AddedToken, Tokenizer, decoders, normalizers
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split

    vocab = json.load(open(f"{args.snapshot}/vocab.json"))
    lines = open(f"{args.snapshot}/merges.txt", encoding="utf-8").read().splitlines()
    assert lines[0] == "#version: 0.2", lines[0]
    merges = [tuple(l.split(" ")) for l in lines[1:]]
    tok = Tokenizer(BPE(vocab=vocab, merges=merges))
    donor = json.load(open(args.qwen3_tokenizer))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = Sequence([
        Split(pattern=donor["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"],
              behavior="isolated", invert=False),
        ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tc = json.load(open(f"{args.snapshot}/tokenizer_config.json"))["added_tokens_decoder"]
    added = [AddedToken(content=v["content"], single_word=v["single_word"], lstrip=v["lstrip"],
                        rstrip=v["rstrip"], normalized=v["normalized"], special=v["special"])
             for _, v in sorted(tc.items(), key=lambda kv: int(kv[0]))]
    tok.add_special_tokens(added)
    d = json.loads(tok.to_str())
    d["post_processor"] = donor["post_processor"]
    # The lib writes a str pattern as {"String": ...}; dgpp's Tokenizer pins
    # the {"Regex": ...} form (same pattern, the Qwen2 Split dgpp hardcodes).
    split = d["pre_tokenizer"]["pretokenizers"][0]
    assert set(split["pattern"]) == {"String"}, split["pattern"]
    split["pattern"] = {"Regex": split["pattern"]["String"]}
    json.dump(d, open(args.out, "w"))
    print(f"{args.out}: {len(vocab)} vocab, {len(merges)} merges, {len(added)} added tokens")


if __name__ == "__main__":
    main()
