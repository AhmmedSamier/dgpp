#!/bin/bash
# The decode-optimization A/B on the fabric (docs/mimo_v26_flash_plan.md §7.1):
# one world per leg (the MiMo w4 template unless AB_CONFIG names another) from
# a given dgpp-serve binary, C1/C2/C4 greedy
# timed_load with MTP depth 1 (the template), then C1 plain (--no-mtp);
# legs in the order given (base first and last: the closing baseline).
#
#   run_ab.sh <out-dir> <label>=<binary>[=<ENV=VAL;ENV=VAL>] [...]
#
# The optional third field sets DGPP_* environment knobs for the leg (the
# head's DGPP_* environment reaches every rank through dgpp-cluster).
#
# Each leg writes <out-dir>/<label>/{up.txt,mtp.json,mtp.log,plain.json,plain.log,down.txt}.
# summarize_ab.py renders the table (ms/step per class, mean over classes).
set -uo pipefail
ROOT=/home/stephen/workspace/dgpp
OUT=$1; shift
CONFIG=${AB_CONFIG:-$ROOT/deploy/cluster_mimo-v2.6-flash_mxfp4-fp8_w4.json}   # AB_CONFIG overrides (the GLM regression check)
MODES=${AB_MODES:-mtp plain}                                                      # AB_MODES="mtp" skips the plain leg
mkdir -p "$OUT"
for leg in "$@"; do
  label=${leg%%=*}; rest=${leg#*=}; bin=${rest%%=*}; envs=""; [[ $rest == *=* ]] && envs=${rest#*=}
  d=$OUT/$label; mkdir -p "$d"
  echo "== $(date -u +%H:%M:%S) leg $label ($bin) env [${envs}]"
  sha256sum "$bin" > "$d/binary.sha256"; echo "$envs" > "$d/env.txt"
  for kv in ${envs//;/ }; do export "$kv"; done
  for mode in $MODES; do
    knobs=(); [[ $mode == plain ]] && knobs=(--knobs=--no-mtp)
    conc="1,2,4"; [[ $mode == plain ]] && conc="1"
    [[ -s $d/$mode.json ]] && { echo "   $label/$mode done already"; continue; }
    python3 "$ROOT/scripts/dgpp-cluster" up --config "$CONFIG" --bin "$bin" --log-dir "$d/server-$mode" "${knobs[@]}" > "$d/up-$mode.txt" 2>&1 || { echo "up failed: $d/up-$mode.txt"; exit 1; }
    python3 "$ROOT/scripts/timed_load.py" 127.0.0.1 18080 --concurrency "$conc" --classes all --repeat 3 --max-tokens 256 --json-out "$d/$mode.json" > "$d/$mode.log" 2>&1 || echo "timed_load ($mode) failed: $d/$mode.log"
    python3 "$ROOT/scripts/dgpp-cluster" down --config "$CONFIG" --log-dir "$d/server-$mode" > "$d/down-$mode.txt" 2>&1 || true
  done
  for kv in ${envs//;/ }; do unset "${kv%%=*}"; done
done
echo "== $(date -u +%H:%M:%S) done"
python3 "$OUT/../ab/summarize_ab.py" "$OUT"
