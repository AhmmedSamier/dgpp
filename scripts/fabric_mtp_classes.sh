#!/bin/bash
# Kept in scripts/ since 2026-09-06 (the closure pass ran it from a scratch directory).
# MTP acceptance per prompt class (M9's sign-off item): glm_gen_check
# --decode-graph --mtp on the four nodes, greedy, 300 steps per class.
set -u
ROOT=/home/user/workspace/dgpp
OUT=${1:?OUT_DIR}; shift
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"; cd "$ROOT" || exit 1
declare -A P
P[chat]="Explain, in a few paragraphs, why a CUDA graph replay can be faster than launching the same kernels eagerly, and what it costs."
P[code]="Write a Python module that parses an OpenAI-style SSE stream of chat completion chunks into a single message object, with type hints, docstrings, and a small set of unit tests using pytest."
P[prose]="Write a long, detailed history of the Roman Republic from its founding to the rise of Augustus, one era per paragraph."
P[json]="Return a JSON array of 25 objects, each with the fields country, capital, population_millions and currency, for 25 different countries. Output only the JSON."
P[math]="A train leaves city A at 60 km/h and another leaves city B, 450 km away, at 90 km/h toward it 30 minutes later. Work out step by step when and where they meet, then generalize the formula and check it with two other examples."
for cls in "$@"; do
  "$ROOT/scripts/fabric_run.sh" --force --log-dir "$OUT/$cls" -- \
    --model unsloth/GLM-5.3-Flash-FP8 --chat "${P[$cls]}" --steps 300 --decode-graph --mtp \
    > "$OUT/$cls.out" 2>&1
  echo "== $cls"
  grep -h "speculative summary" "$OUT/$cls/r0.log" | tail -1 | sed 's/^.*INFO  //' | cut -c1-260
  grep -h "rank consistency" "$OUT/$cls.out" | tail -1
done
