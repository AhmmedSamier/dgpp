#!/bin/bash
# Grow-on-demand admission on the fabric (M6 6d): the same four short-answer
# requests with a large max_tokens against a tight KV pool, once under
# full-reserve and once under grow. Full-reserve pins prompt + max_tokens
# per request, so the pool admits them one at a time; grow reserves prompt
# + window and lets all four run at once, growing only if an answer runs
# long. Prints, per policy, each request's finish and token count, the wall
# time of the batch, the metrics' growth and shed counters, and the
# four-way op-stream md5 (the growth decisions must be rank-identical).
#
# Usage: scripts/serve_admission_check.sh [OUT_DIR]
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/scripts/cluster_env.sh" || exit 1
OUT=${1:-$ROOT/build-ci/fabric-runs/admission_gate_$(date +%Y-%m-%d_%H%M%S)}
HOST=$(dgpp_client_host) || exit 1
URL=http://$HOST:$(dgpp_http_port)/v1/chat/completions
BASE="--max-concurrency 4 --kv-capacity 1536 --default-max-tokens 1024 --queue-limit 8 --decode-graph --seed 20260904"
mkdir -p "$OUT"
cd "$ROOT" || exit 1

declare -a PROMPTS=(
  "In one sentence: what is the capital of Portugal?"
  "In one sentence: what does a lighthouse do?"
  "In one sentence: why is the sky blue?"
  "In one sentence: who wrote Hamlet?"
)

run_policy() {  # run_policy NAME KNOBS
  local name=$1 knobs=$2
  local dir=$OUT/$name
  mkdir -p "$dir"
  export DGPP_SERVE_LOG=$dir/serve
  export DGPP_SERVE_KNOBS="$BASE $knobs"
  echo "=== policy $name: $DGPP_SERVE_KNOBS"
  scripts/serve_run.sh up || return 1
  local M
  M=$(dgpp_served_model) || return 1
  sleep 2
  local t0 t1 i=0
  t0=$(date +%s.%N)
  for p in "${PROMPTS[@]}"; do
    i=$((i + 1))
    curl -s --max-time 600 "$URL" -H 'Content-Type: application/json' \
      -d '{"model":"'$M'","messages":[{"role":"user","content":"'"$p"'"}],"max_tokens":1024}' \
      > "$dir/req_$i.json" &
  done
  wait
  t1=$(date +%s.%N)
  curl -s "http://$HOST:$(dgpp_http_port)/v1/metrics" > "$dir/metrics.json"
  scripts/serve_run.sh down
  python3 "$ROOT/scripts/admission_report.py" "$dir" "$t0" "$t1" "$name"
  grep -h "admitted to slot\|deferred\|reservation grown\|pool exhausted" "$dir/serve/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* INFO  //; s/^[0-9-]* [0-9:.]* WARN  //' | cut -c1-150 | head -16
  md5sum "$dir"/serve/serve_rank*.ops | awk '{print $1}' | sort | uniq -c
}

run_policy full "--admission full"
run_policy grow "--admission grow --admission-window 256"
echo "=== admission check done: $OUT"
