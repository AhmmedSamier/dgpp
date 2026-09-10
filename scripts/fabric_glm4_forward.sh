#!/bin/bash
# The GLM-4.7 real-checkpoint TP forward gate (docs/glm47_plan.md,
# 2026-09-09): glm4_forward_check on the first WORLD nodes of the cluster
# config over the fabric bus, one rank per node — the same token ids on
# every rank; every rank's per-layer hyper-state digests must agree
# bitwise, and the merged argmax (the highest logit across the ranks'
# vocab slices) must equal the world-1 argmax.
#   fabric_glm4_forward.sh [--config CLUSTER.json] OUT_DIR WORLD IDS [extra args...]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/deploy/cluster.json"
if [[ "${1:-}" == "--config" ]]; then CONFIG="$2"; shift 2; fi
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
NODES=($(jq -r '.nodes[]' "$CONFIG"))
USER_=$(jq -r '.ssh_user // env.USER' "$CONFIG")
STAGE=$(jq -r '.paths.stage_dir // "/tmp/bus4"' "$CONFIG")
OUT=${1:?OUT_DIR}; WORLD=${2:?WORLD}; IDS=${3:?IDS}; shift 3
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"
BIN="$ROOT/build-ci/glm4_forward_check"
MODEL=${DGPP_GLM4_MODEL:-nvidia/GLM-4.7-NVFP4}
PORT=${DGPP_GLM4_PORT:-29950}
HEAD=${NODES[0]}
pids=()
for ((r=0; r<WORLD; r++)); do
  ip=${NODES[$r]}
  if [[ $r -eq 0 ]]; then
    ( "$BIN" --model "$MODEL" --ids "$IDS" --world "$WORLD" --rank 0 --port "$PORT" "$@" > "$OUT/w${WORLD}_r0.log" 2>&1 ) &
    pids+=($!)
  else
    ssh -o BatchMode=yes "$USER_@$ip" "mkdir -p $STAGE && pkill -x glm4_forward_check; true" >/dev/null 2>&1
    scp -q "$BIN" "$USER_@$ip:$STAGE/glm4_forward_check"
    ( ssh -o BatchMode=yes "$USER_@$ip" "$STAGE/glm4_forward_check --model $MODEL --ids $IDS --world $WORLD --rank $r --peer $HEAD --port $PORT $*" > "$OUT/w${WORLD}_r$r.log" 2>&1 ) &
    pids+=($!)
  fi
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "world $WORLD: exit $rc"
for ((r=0; r<WORLD; r++)); do
  echo "== rank $r"
  grep -h 'digest\|argmax\|boot\|ERROR\|glm4_forward_check:' "$OUT/w${WORLD}_r$r.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-200
done
# Cross-rank digests must agree; the merged argmax is the per-position max over the ranks' argmax logits.
python3 - "$OUT" "$WORLD" <<'PY'
import re, sys
out, world = sys.argv[1], int(sys.argv[2])
digests, ids, logits = [], [], []
for r in range(world):
    txt = open(f"{out}/w{world}_r{r}.log").read()
    digests.append(re.findall(r"digest ([0-9a-f]{16})", txt))
    m_ids = re.search(r"argmax ids: (.*)", txt); m_lg = re.search(r"argmax logits: (.*)", txt)
    if not m_ids or not m_lg:
        print("rank", r, "has no argmax lines"); sys.exit(1)
    ids.append([int(v) for v in m_ids.group(1).split(",")])
    logits.append([float(v) for v in m_lg.group(1).split(",")])
same = all(d == digests[0] for d in digests) and len(digests[0]) > 0
print("cross-rank digests identical:", same, f"({len(digests[0])} digests)")
merged = []
for t in range(len(ids[0])):
    best = max(range(world), key=lambda r: (logits[r][t], -ids[r][t]))
    merged.append(ids[best][t])
print("merged argmax ids:", ",".join(str(v) for v in merged))
sys.exit(0 if same else 1)
PY
