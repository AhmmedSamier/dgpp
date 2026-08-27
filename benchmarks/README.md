# DGPP Benchmarks

These programs are validation probes, not production runtime components. Run
high-bandwidth tests in an agreed maintenance window: they can consume both
RoCE lanes and more than 25 GB/s of the shared memory fabric.

The dated result record for the audit remediation is
`benchmarks/results/2026-08-27-dgx-spark.md`. Curated conclusions also appear
in `docs/measurements.md`.

## Build

```bash
cmake --preset release
cmake --build --preset release -j
```

Examples below use:

```bash
BUILD=build-release
PEER_F0=192.0.2.12
PEER_F1=198.51.100.12
DEV_F0=rocep1s0f0
DEV_F1=roceP2p1s0f0
GID_INDEX=3
```

Use the addresses and GID indices reported by `show_gids` on the target
cluster; do not copy these lab values blindly.

## Memory probes

`micro_mem_bw` reports streaming read, write, and copy rates. Its working-set
argument is MiB:

```bash
"$BUILD/micro_mem_bw" -s 4096
```

`micro_zerocopy` compares GPU access to `cudaMalloc` and `cudaHostAlloc`
memory, CPU-writer contention proxies, copy-engine staging, and the
CPU→GPU→CPU flag round trip:

```bash
"$BUILD/micro_zerocopy"
```

CPU writers touch separate buffers and are only memory-pressure proxies; the
benchmark intentionally avoids racing CPU writes with GPU reads of the same
object. For the production margin, repeat the pinned-read measurement while
real remote `ib_write_bw` streams target the node, as shown in the dated
result record.

## GEMM heuristic sweep

```bash
"$BUILD/micro_gemm_peak"
```

The program first loads all SMs to avoid timing the transition from an idle
clock state. For every model shape and datatype, it then times every algorithm
returned by the current cuBLASLt heuristic query (up to 16), measures three
batches per algorithm, and reports the candidate with the fastest median.
Sample clocks/power externally when recording a result. Small weights may
remain L2-resident across iterations, so their effective GB/s is not a DRAM
result. This is a best-of-heuristics baseline, not proof of silicon peak.

## KDA operator benchmark

```bash
"$BUILD/kda_bench" [--iters N] [--warmup N]
```

Measures the M2 KDA operators at real geometry on an idle node: the
recurrent-kernel-only decode state traffic across all 34 layers (TP=1 and
the TP=4 rank share), and the full layer's decode step and 2048-token
prefill chunk. Reports effective bytes/s against the 230 GB/s planning
floor. The decode recurrence is latency-bound at M2 (few blocks, 34
sequential launches) — see `benchmarks/results/2026-08-27-kda-m2.md` for
the numbers and the documented optimization path. This is profiling
evidence for the M2 exit criterion, not a production throughput claim.

## KDA reference dumps (correctness, not bandwidth)

`tools/kda_reference_dump.py` generates parity inputs for the C++ layer:

```bash
# stdlib-only synthetic oracle (wired into CTest):
python3 tools/kda_reference_dump.py gen-pure --out FILE \
    --heads 2 --head-dim 32 --hidden 64 --tokens 8
"$BUILD/kda_test" --dump-file FILE

# self-contained format/determinism check:
python3 tools/kda_reference_dump.py selftest

# real checkpoint slices (requires torch; run where the checkpoint lives):
python3 tools/kda_reference_dump.py gen-torch --model-dir MODEL_DIR \
    --layer 0 --out FILE --tokens 32
"$BUILD/kda_test" --dump-file FILE
```

`gen-torch` reads safetensors shards with the stdlib (no safetensors
dependency) and runs the reference layer equations in torch; record its
invocation and the parity result when run against the pinned revision.

## Direct-device registration probe

```bash
"$BUILD/micro_gdr_probe"
```

The result distinguishes `SUPPORTED`, `UNSUPPORTED`, and probe errors. On DGX
Spark, `UNSUPPORTED` is expected and returns success because the experiment
completed. Registered `cudaHostAlloc` memory is the zero-copy receive path;
the copy timings are diagnostic comparisons, not a prescribed fallback.

## RC SEND protocol benchmark

Copy the same freshly built binary to the peer. The binary is architecture and
runtime dependent; do not reuse an old copy after source or CUDA changes.

### Latency

Peer:

```bash
"$BUILD/micro_ibv_smoke" serve 4791 --once --dev "$DEV_F0" \
  --gid-index "$GID_INDEX"
```

Initiator:

```bash
"$BUILD/micro_ibv_smoke" ping --peer "$PEER_F0:4791" --iters 1000 \
  --dev "$DEV_F0" --gid-index "$GID_INDEX"
```

Every SEND is signaled and sourced from its registered MR. The reported
one-way value is half the application echo round trip and includes software
overhead; use `ib_send_lat` for the authoritative transport latency.

### Single-lane bandwidth

Peer:

```bash
"$BUILD/micro_ibv_smoke" serve 4792 --once --dev "$DEV_F0" \
  --gid-index "$GID_INDEX"
```

Initiator:

```bash
"$BUILD/micro_ibv_smoke" bw --peer "$PEER_F0:4792" \
  --msg-size 1048576 --window 64 --dev "$DEV_F0" \
  --gid-index "$GID_INDEX"
```

The project tool exercises signaled RC SEND and receive-ring behavior. Use
`ib_write_bw` for the link ceiling.

### Concurrent lanes

Start one peer responder per lane on distinct ports, then run:

```bash
"$BUILD/micro_ibv_smoke" bw \
  --peer "$PEER_F0:4793" --dev "$DEV_F0" \
  --peer "$PEER_F1:4794" --dev "$DEV_F1" \
  --msg-size 1048576 --window 64 --gid-index "$GID_INDEX"
```

Repeated `--peer` and `--dev` arguments map by position. Supplying multiple
peers with only one device intentionally creates multiple QPs on that one
device; it is not lane striping.

## NIC DMA → GPU visibility test

This is the deployment regression for the pinned receive contract. The peer
server allocates an ibverbs-registered `cudaHostAlloc` slab and launches a GPU
consumer. The initiator sends a changing 64-byte payload followed by a
64-byte sequence doorbell as two ordered RC SENDs. The GPU polls the doorbell
with a system-scope acquire, hashes the payload, and publishes an acknowledgement.

Peer:

```bash
"$BUILD/micro_ibv_smoke" serve 4795 --once --dev "$DEV_F0" \
  --gid-index "$GID_INDEX"
```

Initiator:

```bash
"$BUILD/micro_ibv_smoke" verify --peer "$PEER_F0:4795" --iters 10000 \
  --dev "$DEV_F0" --gid-index "$GID_INDEX"
```

Run it independently on every lane and node pair after any CUDA, kernel,
mlx5, NIC firmware, or topology change. A pass is empirical validation for
that exact stack; a NIC is not a C++ atomic participant, so this hardware test
cannot be replaced by the host-only CUDA test.

## Authoritative perftest commands

Per-lane latency:

```bash
ib_send_lat -d "$DEV_F0" -x "$GID_INDEX" -F -n 10000 -s 2 "$PEER_F0"
```

Per-lane bandwidth:

```bash
ib_write_bw -d "$DEV_F0" -x "$GID_INDEX" -F -N -D 5 \
  -s 8388608 -q 4 -t 128 --report_gbits "$PEER_F0"
```

Run a matching server command without the peer argument first. For an
aggregate result, start the `f0` and `P2...f0` server/client pairs concurrently
on distinct TCP ports; adding their separately timed averages is valid only
when their measurement intervals overlap.

## Result-record requirements

Record at least:

- source commit or base commit plus dirty-tree identifier;
- node model, kernel, CUDA, driver, mlx5 firmware, and perftest versions;
- interface, GID, MTU, negotiated speed, and PCIe link state;
- exact command and direction;
- warmups, duration/iterations, all runs or variability—not only the best;
- concurrent workloads and whether a model server was resident;
- when diagnosing a congestion symptom, pause, discard, retry, and
  out-of-buffer counter deltas over the affected interval.
