#!/bin/bash
# The fabric launcher combines a deployment JSON with site settings in .env.
# This shim keeps the test scripts' interface:
# DGPP_SERVE_KNOBS becomes --knobs (flags appended to
# every rank, overriding the config), DGPP_SERVE_LOG becomes --log-dir.
#   serve_run.sh up --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1.json
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
args=("$@")
[ -n "${DGPP_SERVE_KNOBS:-}" ] && args+=(--knobs "$DGPP_SERVE_KNOBS")
[ -n "${DGPP_SERVE_LOG:-}" ] && args+=(--log-dir "$DGPP_SERVE_LOG")
exec python3 "$ROOT/scripts/dgpp-cluster" "${args[@]}"
