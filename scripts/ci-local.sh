#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
PRESET="${DGPP_PRESET:-ci}"
BUILD="${DGPP_BUILD_DIR:-$ROOT/build-$PRESET}"

cmake -S "$ROOT" -B "$BUILD" --preset "$PRESET"
NPROC=$(nproc)
cmake --build "$BUILD" -j"$NPROC"
ctest --test-dir "$BUILD" --output-on-failure
