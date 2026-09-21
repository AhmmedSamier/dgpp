# Compare effective rank settings at startup

Base: `ff8294550d9dae6004ed445982c0d9855f4043cb`.
Issue: [#18](https://github.com/HawkBearPig/dgpp/issues/18).

Rank 0 resolves `graph_batch_min_live=0` to `min(2, max_concurrency)`
before broadcasting the world settings. Peers compared their unresolved
zero with the adopted value and logged a settings-override warning even
when their configurations agreed.

The canonical settings string now renders the effective threshold using
that rank's current concurrency. This suppresses the false warning for
both omitted and explicit zero defaults, including when the head names
the equivalent explicit threshold. Real threshold and concurrency
differences still warn. The change leaves settings serialization,
adoption, validation and engine configuration unchanged.

## Reproduction and regression

`serve_startup_test` runs the actual server in two- and four-process worlds
over loopback. Each process receives `--max-connections 0`, an invalid
local HTTP capacity that is checked after settings adoption and before
checkpoint access, CUDA initialization or RDMA setup. The test hides CUDA
devices, uses an ephemeral journal port and verifies the intended exit
on every rank. Invalid KV capacity cannot serve this purpose because
the journal decoder rejects it before comparing settings.

The four tests cover 12 configurations: automatic defaults at concurrency
1, 2, 4 and 8; omitted, zero and equivalent explicit thresholds on three
peers; differing thresholds in both directions; and different concurrency
with automatic thresholds. Every peer's adopted canonical settings must
match the head's entire settings string.

The new tests failed against the base binary: all eight matching-settings
configurations produced false warnings. Two additional assertions exposed
the unresolved zero in diagnostics for genuine mismatches. The fixed
binary passes all four tests and all 12 configurations.

## Four-node startup validation

The production four-node GLM-5.3-Flash service was running. Temporary
copies of the base and fixed `ci` binaries ran on the four physical Spark
nodes, using a separate ephemeral TCP journal port for each run. They
used the same invalid local HTTP capacity as the loopback regression,
with CUDA hidden and a 20-second process timeout. No model or RDMA world
was created, and production was not restarted. Temporary peer binaries
were removed afterward.

The common server arguments were:

```text
--checkpoint-dir /tmp/dgpp-issue18-unused --world 4 --rank <rank>
--journal-port <ephemeral-port> --fabric-port 0 --max-connections 0
--rendezvous-timeout-ms 10000 --max-concurrency <slots>
--graph-batch-min-live <threshold>
```

Peers also received `--peer <rank-0-address>`. Each run checked all four
exit codes, the expected capacity error, warning counts and equality of
all adopted settings with the head. Across base/fixed runs, effective
settings matched exactly after normalizing only the ephemeral port.

| Configuration | Base warnings, ranks 1/2/3 | Fixed warnings, ranks 1/2/3 | Adopted threshold |
| --- | --- | --- | ---: |
| One slot, automatic on every rank | 1 / 1 / 1 | 0 / 0 / 0 | 1 |
| Four slots, automatic on every rank | 1 / 1 / 1 | 0 / 0 / 0 | 2 |
| Four slots, rank 1 explicitly requests 1 | 1 / 1 / 1 | 1 / 0 / 0 | 2 |
| Four slots, head explicitly requests 2, peers automatic | 1 / 1 / 1 | 0 / 0 / 0 | 2 |

## Build and host validation

```bash
cmake --preset ci
cmake --build --preset ci -j 4
ctest --test-dir build-ci -L host --output-on-failure -j 1
CUDA_VISIBLE_DEVICES="" ctest --test-dir build-ci -R '^(unit_tests|unit)$' -V -j 1
```

The full native aarch64 build passed with warnings treated as errors.
The first host run took 177.50 seconds: 26 of 28 CTest entries passed,
including the new startup regression, all nine checkpoint-backed gates
and the service tests. The two failing entries both run `unit_tests`:
its three CUDA arena cases could not allocate GPU memory while production
was running. These cases sit inside a binary labeled `host`.

Both unit entries passed when explicitly rerun with CUDA hidden (17.81
seconds). Each ran 221 host cases and skipped the three GPU arena cases;
no checkpoint gate was skipped. All 28 host CTest entries therefore have
passing results, with those GPU cases excluded. An earlier attempt to
use `--rerun-failed` selected stale GPU entries instead: with CUDA hidden,
`glm4_forward_test` failed for no visible device and the DSA fixture
generator passed. Those entries are not counted as GPU validation.

Changed C++ ranges were formatted with clang-format 19.1.7; the formatting
check and `git diff --check` passed. The production rank processes were
still alive with the same PIDs after validation.

## Performance and accuracy scope

The only production code change is the startup string comparison. No
per-token work, kernels, sampling, scheduling or model arithmetic changed.
The head still broadcasts the same resolved threshold, and the four-node
comparison confirms unchanged effective settings on every rank. The
canonical string used by the later configuration digest is unchanged for
valid configurations because defaults are already resolved at that point.

This validates the changed TCP startup diagnostic across real nodes. It
does not measure GPU throughput or model accuracy. No successful GPU/RDMA
execution tests, live model API checks or op-stream digest comparisons
are claimed: these test worlds deliberately stop before an engine or op
stream exists, and the diagnostic change does not alter that execution
path.
