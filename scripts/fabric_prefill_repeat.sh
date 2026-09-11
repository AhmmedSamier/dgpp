#!/bin/bash
# The steady-state prefill procedure:
#   fabric_prefill_repeat.sh [--config CLUSTER.json] [--ids-file FILE] OUT_DIR LEN...
# For each LEN: the first LEN of the hard-prose ids (tiled x4 above 2100),
# glm_gen_check on the four nodes with --prefill-repeat 3 (the timed prefill
# is the third), two decode steps; prints the prefill line and the 4-way
# generated-ids verdict.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
# The JSON selects the checkpoint and world size; .env supplies the site.
CONFIG=$(dgpp_config) || exit 1
if [[ "${1:-}" == "--config" ]]; then CONFIG="$2"; shift 2; fi
IDS=""
if [[ "${1:-}" == "--ids-file" ]]; then IDS="$2"; shift 2; fi
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
export DGPP_CLUSTER_CONFIG="$CONFIG"
[[ -f "$CONFIG" ]] || { echo "no cluster config at $CONFIG" >&2; exit 1; }
MODEL=$(jq -r '.model' "$CONFIG")
NODE_LIST=$(dgpp_nodes --config "$CONFIG") || exit 1
read -r -a NODES <<< "$NODE_LIST"

IDS=${IDS:-${DGPP_DATA_DIR:-$ROOT/data}/hard_ids.csv}
[[ -f "$IDS" ]] || { echo "missing $IDS; run python3 scripts/prepare_data.py tokens --text /path/to/long-prompt.txt, or pass --ids-file" >&2; exit 2; }
OUT=${1:?OUT_DIR}; shift
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"
cd "$ROOT" || exit 1
for n in "$@"; do
  prompt=$(python3 "$ROOT/scripts/prefill_report.py" repeat-prompt "$IDS" "$n")
  "$ROOT/scripts/fabric_run.sh" --force --log-dir "$OUT/n$n" -- \
    --model "$MODEL" --prompt "$prompt" --steps 2 \
    --prefill-repeat 3 > "$OUT/n$n.out" 2>&1
  echo "== $n tokens"
  grep -h "rank 0 prefill:" "$OUT/n$n/r0.log" | tail -1
  grep -h "rank consistency\|rc=" "$OUT/n$n.out" | tail -2
  [ -f "$OUT/n$n/r0.log" ] || tail -5 "$OUT/n$n.out"
done
