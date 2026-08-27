#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${DGPP_BUILD_DIR:-$ROOT/build-ci}"
PRESET="${DGPP_PRESET:-ci}"

cmake -S "$ROOT" -B "$BUILD" --preset "$PRESET" || cmake -S "$ROOT" -B "$BUILD"
NPROC=$(nproc)
cmake --build "$BUILD" -j"$NPROC"
ctest --test-dir "$BUILD" --output-on-failure
