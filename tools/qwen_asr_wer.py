#!/usr/bin/env python3
"""LibriSpeech WER gate for the served Qwen3-ASR endpoint (docs/audio.md).

Streams every utterance of a LibriSpeech split (default test-clean) through
POST /v1/audio/transcriptions as multipart file uploads — the same shape
OpenWebUI submits — and scores word error rate against the reference
transcripts with the standard normalization (uppercase, punctuation
stripped, whitespace collapsed).

The checkpoint prefixes its transcript with `language <NAME>` (no marker
to split on, so the harness strips one leading `language <word>`); the
raw hypothesis is kept in the per-utterance record for audit.

Usage:
  python3 tools/qwen_asr_wer.py --split test-clean --url http://127.0.0.1:8000 \
      --model asr_ckpt --out benchmarks/results/2026-10-14-asr-wer/ \
      [--limit N] [--resume]

Writes <out>/wer.json (aggregate) and <out>/utterances.jsonl (per
utterance). --resume skips utterance ids already present in
utterances.jsonl. Only the standard library is used.
"""
import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DGPP_LIBRISPEECH_DIR", "/tmp/opencode/LibriSpeech")


def norm(text):
    text = unicodedata.normalize("NFKC", text).upper()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2014", " ").replace("\u2013", " ")
    words = re.sub(r"[^A-Z' ]", " ", text).split()
    out = []
    for w in words:
        # The standard LibriSpeech expansions; surrounding quotes are
        # stripping artifacts of possessives, not words.
        if w == "MR":
            out.append("MISTER")
        elif w == "MRS":
            out.append("MISSUS")
        else:
            out.append(w.strip("'"))
    return " ".join(x for x in out if x)


def strip_language_prefix(text):
    return re.sub(r"^LANGUAGE\s+[A-Z]+\s*", "", norm(text), count=1)


def edit_distance(ref, hyp):
    # Levenshtein over word lists; O(min) memory over the shorter side.
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def transcribe(url, model, path):
    boundary = "------------------------dgpp-wer-1"
    with open(path, "rb") as f:
        audio = f.read()
    body = []
    body.append(("--" + boundary + "\r\n").encode())
    body.append(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    body.append(model.encode() + b"\r\n")
    body.append(("--" + boundary + "\r\n").encode())
    body.append(
        ('Content-Disposition: form-data; name="file"; filename="%s"\r\n'
         % os.path.basename(path)).encode())
    body.append(b"Content-Type: audio/flac\r\n\r\n")
    body.append(audio + b"\r\n")
    body.append(("--" + boundary + "--\r\n").encode())
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/audio/transcriptions",
        data=b"".join(body),
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read().decode())
    except Exception as e:  # HTTPError carries the status; connection errors do not
        status = getattr(e, "code", 0)
        detail = ""
        try:
            detail = e.read().decode()[:300]
        except Exception:
            pass
        return status, {"error": "%s %s" % (e, detail)}


def collect(split_dir):
    refs = {}
    paths = {}
    for root, _, files in os.walk(split_dir):
        for name in files:
            if name.endswith(".trans.txt"):
                with open(os.path.join(root, name)) as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        uid, text = line.split(" ", 1)
                        refs[uid] = norm(text)
            elif name.endswith(".flac"):
                paths[name.replace(".flac", "")] = os.path.join(root, name)
    return refs, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="asr_ckpt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute wer.json from stored utterances.jsonl "
                         "(no server traffic)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(ROOT, "benchmarks", "results", "asr-wer")
    if args.rescore:
        os.makedirs(out_dir, exist_ok=True)
        split_dir = os.path.join(DATA, args.split)
        refs, _ = collect(split_dir)
        latest = {}
        with open(os.path.join(out_dir, "utterances.jsonl")) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    latest[r["id"]] = r
                except Exception:
                    pass
        ref_words = hyp_words = errs = failed = 0
        for uid, r in latest.items():
            if r.get("status") == 200 and uid in refs:
                hyp = strip_language_prefix(r.get("hyp_raw", ""))
                a, b = refs[uid].split(), hyp.split()
                ref_words += len(a)
                hyp_words += len(b)
                errs += edit_distance(a, b)
            else:
                failed += 1
        wer = errs / ref_words if ref_words else 0
        summary = {"split": args.split, "utterances": len(latest),
                   "scored": len(latest) - failed, "failed": failed,
                   "ref_words": ref_words, "hyp_words": hyp_words,
                   "errors": errs, "wer": wer}
        with open(os.path.join(out_dir, "wer.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return

    split_dir = os.path.join(DATA, args.split)
    if not os.path.isdir(split_dir):
        sys.exit("missing split dir: %s (set DGPP_LIBRISPEECH_DIR)" % split_dir)
    os.makedirs(out_dir, exist_ok=True)

    refs, paths = collect(split_dir)
    uids = sorted(set(refs) & set(paths))
    if args.limit:
        uids = uids[:args.limit]
    done = set()
    if args.resume:
        jl = os.path.join(out_dir, "utterances.jsonl")
        if os.path.exists(jl):
            with open(jl) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        # Only successes are final: failures (refusals,
                        # restarts, disconnects) are retried.
                        if r.get("status") == 200:
                            done.add(r["id"])
                    except Exception:
                        pass
    uids = [u for u in uids if u not in done]
    print("scoring %d utterances (%d already done)" % (len(uids), len(done)), flush=True)

    jl = open(os.path.join(out_dir, "utterances.jsonl"), "a")
    ref_words = hyp_words = errs = 0
    failed = 0
    for i, uid in enumerate(uids):
        status, res = transcribe(args.url, args.model, paths[uid])
        hyp_raw = res.get("text", "") if isinstance(res, dict) else ""
        if status != 200 or not hyp_raw:
            failed += 1
            jl.write(json.dumps({"id": uid, "status": status, "error": res}) + "\n")
        else:
            hyp = strip_language_prefix(hyp_raw)
            ref = refs[uid].split()
            hyp_w = hyp.split()
            e = edit_distance(ref, hyp_w)
            ref_words += len(ref)
            hyp_words += len(hyp_w)
            errs += e
            jl.write(json.dumps({"id": uid, "status": 200, "ref": refs[uid],
                                 "hyp_raw": hyp_raw, "hyp": hyp,
                                 "errors": e, "ref_words": len(ref)}) + "\n")
        jl.flush()
        if (i + 1) % 50 == 0:
            wer = errs / ref_words if ref_words else 0
            print("  %d/%d wer=%.4f failed=%d" % (i + 1, len(uids), wer, failed), flush=True)
    jl.close()
    # Aggregate over the whole record (latest entry per id wins), so
    # resumed runs report the full split, not just their own delta.
    latest = {}
    with open(os.path.join(out_dir, "utterances.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
                latest[r["id"]] = r
            except Exception:
                pass
    ref_words = hyp_words = errs = failed = 0
    for r in latest.values():
        if r.get("status") == 200 and "errors" in r:
            ref_words += r.get("ref_words", 0)
            hyp_words += len(r.get("hyp", "").split())
            errs += r["errors"]
        else:
            failed += 1
    wer = errs / ref_words if ref_words else 0
    summary = {"split": args.split, "utterances": len(latest),
               "scored": len(latest) - failed, "failed": failed,
               "ref_words": ref_words, "hyp_words": hyp_words,
               "errors": errs, "wer": wer}
    with open(os.path.join(out_dir, "wer.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
