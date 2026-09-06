#!/bin/bash
# The release artifact (2026-09-06, the productionizing pass): builds the
# release preset, installs the layout into dist/stage/dgpp-<version>/,
# writes the manifest with per-file checksums, and packs
# dist/dgpp-<version>.tar.zst. The version is the tree's
# (cmake/version.cmake): <base>+g<sha12>, ".dirty" when uncommitted.
#
#   scripts/release.sh            # build, stage, verify, pack; prints the tarball
#   scripts/release.sh --no-build # re-stage and pack the existing release build
#
# Layout inside the tarball (dgpp-<version>/):
#   bin/dgpp-serve                 the server, rpath $ORIGIN/../lib
#   lib/libcudart.so.13, lib/libcublasLt.so.13   the CUDA runtime it was built against
#   scripts/dgpp-cluster, scripts/serve_api_check.py
#   deploy/cluster.example.json    the config template a site copies and edits
#   doc/README.md, doc/operations.md
#   MANIFEST                       version, git sha, CUDA, host, date
#   MANIFEST.sha256                sha256 of every other file (sha256sum -c)
# A node needs only the driver, rdma-core, libnl and libstdc++ beyond this.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=$ROOT/build-release
DIST=$ROOT/dist
if [ "${1:-}" != "--no-build" ]; then
  cmake --preset release -S "$ROOT" >/dev/null
  cmake --build "$BUILD" -j --target dgpp_serve_app
fi
VERSION=$(sed -n 's/^#define DGPP_VERSION "\(.*\)"$/\1/p' "$BUILD/generated/dgpp_version.hpp")
[ -n "$VERSION" ] || { echo "no version stamp in $BUILD/generated/dgpp_version.hpp"; exit 1; }
NAME=dgpp-$VERSION
STAGE=$DIST/stage/$NAME
rm -rf "$STAGE"; mkdir -p "$STAGE"
cmake --install "$BUILD" --prefix "$STAGE" >/dev/null
# The manifest: the identity, then every file's checksum.
{
  echo "name: $NAME"
  echo "version: $VERSION"
  echo "git: $(git -C "$ROOT" rev-parse HEAD)"
  echo "cuda: $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname) ($(uname -m))"
  echo "build_type: Release"
} > "$STAGE/MANIFEST"
(cd "$STAGE" && find . -type f ! -name MANIFEST.sha256 -printf '%P\n' | sort | xargs sha256sum > MANIFEST.sha256)
# Verify before packing: the checksums, the version the binary reports, and
# that the binary resolves the CUDA runtime from its own lib/.
(cd "$STAGE" && sha256sum -c --quiet MANIFEST.sha256)
REPORTED=$("$STAGE/bin/dgpp-serve" --version)
case "$REPORTED" in *"$VERSION"*) ;; *) echo "version mismatch: binary says '$REPORTED', stamp is $VERSION"; exit 1 ;; esac
for lib in libcudart.so libcublasLt.so; do
  RESOLVED=$(ldd "$STAGE/bin/dgpp-serve" | awk -v l="$lib" '$1 ~ l {print $3}' | head -1)
  if [ -z "$RESOLVED" ] || [ "$(readlink -f "$RESOLVED")" != "$(readlink -f "$STAGE"/lib/$lib.*)" ]; then
    echo "the staged binary does not resolve $lib from its lib/ (got '$RESOLVED'):"; ldd "$STAGE/bin/dgpp-serve" | grep -E "cudart|cublasLt"; exit 1
  fi
done
mkdir -p "$DIST"
TARBALL=$DIST/$NAME.tar.zst
rm -f "$TARBALL"
tar -C "$DIST/stage" -I 'zstd -19 -T0' -cf "$TARBALL" "$NAME"
echo "$REPORTED"
echo "$(du -h "$TARBALL" | cut -f1)  $TARBALL"
