# Dependencies

The supported target is DGX Spark / GB10, Linux/aarch64, CUDA SM 12.1.
The development environment used for this change is Ubuntu 24.04.4,
GCC 13.3, CMake 3.28.3 and CUDA toolkit 13.0.88. This is a tested baseline,
not a claim that every compiler or driver combination works.

| Task | Required dependencies |
| --- | --- |
| Build serving binaries | CMake 3.25+, C++20 compiler (GCC 13 tested), CUDA toolkit 13 with cudart/cuBLASLt, libibverbs development headers/library, Python 3.10+, make or Ninja |
| Run a development binary | Compatible NVIDIA driver, CUDA 13 cudart/cuBLASLt shared libraries, libibverbs and its provider libraries, libnl, libstdc++, glibc |
| Run a packaged binary | Same runtime dependencies, except cudart/cuBLASLt are bundled in `lib/` |
| Launch/manage a cluster | Python 3.10+ standard library, Bash, OpenSSH client (`ssh`, `scp`), GNU coreutils (`timeout`), curl, jq; peer Python available as `python3` |
| Download checkpoints | A venv with `requirements-download.txt`; Hugging Face login only when the repository requires it |
| HTTP clients and reports | Python standard library; Bash wrappers also use curl, jq, awk, sed, GNU utilities and util-linux (`flock`) |
| Generate tokenizer/template goldens or token-ID fixtures | `requirements-tools.txt`; local checkpoint metadata |
| Torch reference dumps | Compatible PyTorch installed separately for the intended CPU/CUDA backend; see the individual generator's help |
| Package/install releases | `tar`, `zstd`, `sha256sum`, `ldd`; Git and the CUDA toolkit on the packaging machine |
| Profiling and optional diagnostics | Nsight Systems (`nsys`), `iproute2`, `rdma-core` diagnostic utilities, and clang-format for formatting targets |

On Ubuntu 24.04, the non-CUDA build/operations packages can be installed with:

```bash
sudo apt-get install build-essential cmake ninja-build pkg-config \
  python3 python3-venv libibverbs-dev rdma-core libibverbs1 ibverbs-providers \
  openssh-client curl jq zstd iproute2
```

Use the DGX OS-supported driver/toolkit installation for CUDA; the command
above does not install or upgrade either. Peer runtime compatibility is
checked by `dgpp-cluster doctor`. A staged development binary's exact peer
linkage cannot be checked before staging; that check is reported as a warning.

libibverbs is used by the production multi-node transport, not just debugging.
The current single-node server also links it, although world 1 does not use
the RDMA bus. Keep `DGPP_ENABLE_IBV=ON` for serving and releases.
`-DDGPP_ENABLE_IBV=OFF` is a development-only subset: it omits the server,
release installation and RDMA targets. With the option on, missing headers
or libraries cause a configure error rather than silently omitting serving.

Python packages are optional for the native server. Install tooling into a
venv, not the system interpreter. The requirements files give compatible
version ranges; save `pip freeze` with benchmark results when exact tooling
reproducibility matters. Historical goldens record the versions that generated
them; regenerating with another version is a deliberate test-data change.
