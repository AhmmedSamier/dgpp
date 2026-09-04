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
OUT=${1:-$ROOT/build-ci/fabric-runs/admission_gate_$(date +%Y-%m-%d_%H%M%S)}
HOST=${DGPP_SERVE_HOST:-192.0.2.11}
URL=http://$HOST:18080/v1/chat/completions
M=unsloth/GLM-5.3-Flash-FP8
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
  curl -s "http://$HOST:18080/v1/metrics" > "$dir/metrics.json"
  scripts/serve_run.sh down
  python3 - "$dir" "$t0" "$t1" "$name" <<'PY'
import json, sys, glob, os
d, t0, t1, name = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
for f in sorted(glob.glob(os.path.join(d, "req_*.json"))):
    r = json.load(open(f))
    if "error" in r:
        print(f"  {os.path.basename(f)}: ERROR {r['error']}")
        continue
    ch = r["choices"][0]
    print(f"  {os.path.basename(f)}: finish {ch['finish_reason']}, completion_tokens {r['usage']['completion_tokens']}: {ch['message']['content'][:80]!r}")
m = json.load(open(os.path.join(d, "metrics.json")))
sv, sc = m.get("service", {}), m.get("scheduler", {})
print(f"  metrics: admission {sv.get('admission')}, reservations_grown {sv.get('reservations_grown')}, requests_shed_pool {sv.get('requests_shed_pool')}, pool {sc.get('pool_blocks_in_use')}/{sc.get('pool_blocks_total')} blocks, tokens {sc.get('tokens_generated')}")
print(f"  batch wall time under {name}: {t1 - t0:.1f} s")
PY
  grep -h "admitted to slot\|deferred\|reservation grown\|pool exhausted" "$dir/serve/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* INFO  //; s/^[0-9-]* [0-9:.]* WARN  //' | cut -c1-150 | head -16
  md5sum "$dir"/serve/serve_rank*.ops | awk '{print $1}' | sort | uniq -c
}

run_policy full "--admission full"
run_policy grow "--admission grow --admission-window 256"
echo "=== admission check done: $OUT"
