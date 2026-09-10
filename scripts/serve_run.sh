#!/bin/bash
# The fabric launcher is scripts/dgpp-cluster over deploy/cluster.json
# (2026-09-06, the productionizing pass). This shim keeps the evidence
# scripts' interface: DGPP_SERVE_KNOBS becomes --knobs (flags appended to
# every rank, overriding the config), DGPP_SERVE_LOG becomes --log-dir.
#   serve_run.sh up | down | status
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
args=("${1:-}")
[ -n "${DGPP_SERVE_KNOBS:-}" ] && args+=(--knobs "$DGPP_SERVE_KNOBS")
[ -n "${DGPP_SERVE_LOG:-}" ] && args+=(--log-dir "$DGPP_SERVE_LOG")
exec python3 "$ROOT/scripts/dgpp-cluster" "${args[@]}"
