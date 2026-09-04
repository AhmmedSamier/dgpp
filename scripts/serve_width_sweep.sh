#!/bin/bash
# The sampling-width sweep (M6 6b's open sizing decision): boots the
# four-node service once per (candidate width, graph mode), sends the same
# request set every time — half at the card's defaults (temperature 1.0 /
# top_p 0.95), half at a hot setting (temperature 1.2 / top_p 1.0) that
# flattens the distribution — with fixed seeds, so the SAME tokens come
# back at every width (the pick is width-independent by construction) and
# only the step time and the fallback count change. Prints, per boot, the
# per-request fallback counts and pace, and appends one TSV row per request
# to $OUT/sweep.tsv for the analysis.
#
# Usage: scripts/serve_width_sweep.sh [OUT_DIR] [WIDTHS] [MODES]
#   WIDTHS defaults to "128 64 32 16 4", MODES to "plain mtp".
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$ROOT/build-ci/fabric-runs/width_sweep_$(date +%Y-%m-%d_%H%M%S)}
WIDTHS=${2:-"128 64 32 16 4"}
MODES=${3:-"plain mtp"}
HOST=${DGPP_SERVE_HOST:-192.0.2.11}
URL=http://$HOST:18080/v1/chat/completions
M=unsloth/GLM-5.3-Flash-FP8
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
    python3 - "$RUN" "$width" "$mode" "$TSV" <<'PY'
import json, re, sys, os, hashlib, glob
run, width, mode, tsv = sys.argv[1:5]
log = os.path.join(run, "serve", "serve_r0.log")
# Per-request fallback counts: the engine logs "slot S closed: N sampled
# decode steps, F fallbacks" at close; the scheduler logs "request 'id'
# admitted to slot S" at admission. Slots are reused in order, so the k-th
# close on a slot belongs to the k-th admission to it.
admitted = {}   # slot -> [request ids in order]
closed = {}     # request id -> (sampled, fallbacks)
for line in open(log, errors="replace"):
    m = re.search(r"request '([^']+)' admitted to slot (\d+)", line)
    if m:
        admitted.setdefault(int(m.group(2)), []).append(m.group(1))
        continue
    m = re.search(r"slot (\d+) closed: (\d+) sampled decode steps, (\d+) fallbacks", line)
    if m:
        slot = int(m.group(1))
        pending = [r for r in admitted.get(slot, []) if r not in closed]
        if pending:
            closed[pending[0]] = (int(m.group(2)), int(m.group(3)))
# The pace per request from serve_pace.py's table.
import subprocess
table = subprocess.run([sys.executable, "scripts/serve_pace.py", log], capture_output=True, text=True).stdout
pace = {}
for line in table.splitlines():
    parts = line.split()
    if len(parts) >= 9 and parts[0].startswith("chatcmpl-"):
        try:
            pace[parts[0]] = (int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        except ValueError:
            pass
rows = []
for f in sorted(glob.glob(os.path.join(run, "req_*.json"))):
    name = os.path.basename(f)[4:-5]
    setting = name.split("_")[0]
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"  {name}: unreadable ({e})"); continue
    if "error" in d:
        print(f"  {name}: ERROR {d['error']}"); continue
    rid = d["id"]
    ch = d["choices"][0]
    text = ch["message"].get("content") or ""
    sha = hashlib.sha256(((ch["message"].get("reasoning_content") or "") + "\x00" + text).encode()).hexdigest()[:12]
    sampled, fallbacks = closed.get(rid, (-1, -1))
    tok, replays, tpr, mspt, mspr = pace.get(rid, (-1, -1, float("nan"), float("nan"), float("nan")))
    rows.append((width, mode, name, setting, d["usage"]["completion_tokens"], ch["finish_reason"], sampled, fallbacks, replays, tpr, mspr, mspt, sha))
with open(tsv, "a") as out:
    for r in rows:
        out.write("\t".join(str(x) for x in r) + "\n")
        print("  " + "  ".join(str(x) for x in r))
PY
  done
done
echo "=== sweep done: $TSV"
