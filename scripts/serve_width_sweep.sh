#!/bin/bash
# The sampling-width sweep (M6 6b's open sizing decision): boots the
# four-node service once per (candidate width, graph mode), sends the same
# request set every time — half at the card's defaults (temperature 1.0 /
# top_p 0.95), half at a hot setting (temperature 1.2 / top_p 1.0) that
# flattens the distribution — with fixed seeds, so the same tokens come
# back at every width (the pick is width-independent by construction) and
# only the step time and the fallback count change. Prints, per boot, the
# per-request fallback counts and pace, and appends one TSV row per request
# to $OUT/sweep.tsv for the analysis.
#
# Usage: scripts/serve_width_sweep.sh [OUT_DIR] [WIDTHS] [MODES]
#   WIDTHS defaults to "128 64 32 16 4", MODES to "plain mtp".
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/scripts/cluster_env.sh" || exit 1
OUT=${1:-$ROOT/build-ci/fabric-runs/width_sweep_$(date +%Y-%m-%d_%H%M%S)}
WIDTHS=${2:-"128 64 32 16 4"}
MODES=${3:-"plain mtp"}
HOST=$(dgpp_client_host) || exit 1
URL=http://$HOST:$(dgpp_http_port)/v1/chat/completions
mkdir -p "$OUT"
cd "$ROOT" || exit 1
TSV=$OUT/sweep.tsv
echo -e "width\tmode\trequest\tsetting\tcompletion_tokens\tfinish\tsampled\tfallbacks\treplays\ttok_per_replay\tms_per_replay\tms_per_token\tsha" > "$TSV"

# The request set: three prompts, each at the defaults and at the hot
# setting, 384 tokens, one live at a time (the scalar pace).
declare -a PROMPTS=(
  "Write a short story about a lighthouse keeper who finds a message in a bottle."
  "Explain how a transformer language model generates text, for a curious teenager."
  "List and describe five unusual sports played around the world."
)

run_requests() {  # run_requests WIDTH MODE RUNDIR
  local width=$1 mode=$2 dir=$3 i=0
  local M
  M=$(dgpp_served_model) || return 1
  for setting in default hot; do
    for p in "${PROMPTS[@]}"; do
      i=$((i + 1))
      local extra=""
      if [ "$setting" = hot ]; then extra=',"temperature":1.2,"top_p":1.0'; fi
      local body='{"model":"'$M'","messages":[{"role":"user","content":"'"$p"'"}],"max_tokens":384,"seed":'$((20260904 + i))"$extra"'}'
      curl -s --max-time 600 "$URL" -H 'Content-Type: application/json' -d "$body" > "$dir/req_${setting}_$i.json"
    done
  done
}

for mode in $MODES; do
  for width in $WIDTHS; do
    RUN=$OUT/w${width}_$mode
    mkdir -p "$RUN"
    export DGPP_SERVE_LOG=$RUN/serve
    GRAPH="--decode-graph"
    if [ "$mode" = mtp ]; then GRAPH="--decode-graph --mtp"; fi
    export DGPP_SERVE_KNOBS="--max-concurrency 2 --kv-capacity 4096 --default-max-tokens 512 --queue-limit 8 $GRAPH --sampling-candidates $width"
    echo "=== width $width mode $mode"
    if ! scripts/serve_run.sh up; then
      echo "boot failed for width $width mode $mode"
      continue
    fi
    sleep 2
    run_requests "$width" "$mode" "$RUN"
    scripts/serve_run.sh down
    python3 "$ROOT/scripts/width_sweep_collect.py" "$RUN" "$width" "$mode" "$TSV"
  done
done
echo "=== sweep done: $TSV"
