#!/bin/bash
# Kept in scripts/ since 2026-09-06 (the closure pass ran it from a scratch directory).
# MTP acceptance per prompt class (M9's sign-off item): glm_gen_check
# --decode-graph --mtp on the four nodes, greedy, 300 steps per class.
#   fabric_mtp_classes.sh [--config CLUSTER.json] OUT_DIR CLASS...
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The checkpoint and the fabric come from a cluster config (dgpp-cluster's
# file: deploy/cluster.json by default, or --config F as the first argument)
# — its `model`, `nodes` (node 0 is the head, where this runs) and `ssh_user`
# — never from the environment or a hardcoded id.
CONFIG="$ROOT/deploy/cluster.json"
if [[ "${1:-}" == "--config" ]]; then CONFIG="$2"; shift 2; fi
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
[[ -f "$CONFIG" ]] || { echo "no cluster config at $CONFIG" >&2; exit 1; }
MODEL=$(jq -r '.model' "$CONFIG")
NODES=($(jq -r '.nodes[]' "$CONFIG"))
export DGPP_FABRIC_HEAD="${NODES[0]}"
export DGPP_FABRIC_PEERS="${NODES[*]:1}"
export DGPP_FABRIC_USER="$(jq -r '.ssh_user // env.USER' "$CONFIG")"
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
    --model "$MODEL" --chat "${P[$cls]}" --steps 300 --decode-graph --mtp \
    > "$OUT/$cls.out" 2>&1
  echo "== $cls"
  grep -h "speculative summary" "$OUT/$cls/r0.log" | tail -1 | sed 's/^.*INFO  //' | cut -c1-260
  grep -h "rank consistency" "$OUT/$cls.out" | tail -1
done
