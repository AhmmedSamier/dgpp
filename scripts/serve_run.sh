#!/bin/bash
# The fabric launcher combines a deployment JSON with site settings in .env.
# This shim keeps the test scripts' interface:
# DGPP_SERVE_KNOBS becomes --knobs (flags appended to
# every rank, overriding the config), DGPP_SERVE_LOG becomes --log-dir.
#   serve_run.sh up --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
args=("$@")
# The selected deployment, explicitly: a world started under --log-dir is
# stopped only with its --config and the same --log-dir (the launcher's
# rule), and `down` without --config scans the default log dir instead —
# the soak ritual's teardown was silently failing that way (2026-09-13).
case "${1:-}" in
  up|down|status)
    if ! printf '%s\n' "$@" | grep -qx -- --config; then
      cfg=$(dgpp_config 2>/dev/null) && [ -n "$cfg" ] && args+=(--config "$cfg")
    fi;;
esac
[ -n "${DGPP_SERVE_KNOBS:-}" ] && args+=(--knobs "$DGPP_SERVE_KNOBS")
[ -n "${DGPP_SERVE_LOG:-}" ] && args+=(--log-dir "$DGPP_SERVE_LOG")
exec python3 "$ROOT/scripts/dgpp-cluster" "${args[@]}"
