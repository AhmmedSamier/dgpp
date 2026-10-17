# HTTP output delivery

The default `http.stream_flush: "pass"` keeps the 25 ms quiet-loop poll.
`"wake"` signals ready output through an eventfd without moving socket writes
off the HTTP thread. `--stream-flush` overrides the deployment JSON. The
maintenance timeout remains 25 ms in both production modes.
Request-preparation and file completions always notify HTTP; `stream_flush`
controls additional notifications for generated output.

## Host comparison

Build the standalone probe from the checkout being evaluated:

```bash
mkdir -p build-http-probe
g++ -std=c++20 -O2 -pthread -Isrc tools/http_stream_bench.cpp \
  src/serve/http_server.cpp src/common/log.cpp -o build-http-probe/http_stream_bench
for mode in pass25 pass10 wake; do
  build-http-probe/http_stream_bench "$mode" 125 1 500
  build-http-probe/http_stream_bench "$mode" 125 4 500
  build-http-probe/http_stream_bench "$mode" 125 16 500
  build-http-probe/http_stream_bench "$mode" 0 64 1
done
```

Repeat the panel in alternating mode order on an idle host. The probe uses the
real HTTP server, a synthetic producer, staggered streams and approximately
300-byte event payloads. It measures publication-to-receive delay, event
batching, HTTP-thread CPU time and loop passes. `pass10` is a probe-only
comparison; it does not add a production configuration setting. CPU percentage
uses one CPU core as 100%. Idle tests include connection setup CPU time.

These measurements isolate HTTP delivery. They do not include tokenization,
model execution, the generation-service record pump, proxies or network hops.
SSE events and socket reads are not model token counts, and one event per read
is an observation rather than a protocol guarantee.

## Real-model validation

Use the same release binary and deployment for `pass` and `wake`, changing
only `http.stream_flush`. Follow the hardware, warmup and repetition rules in
[benchmarks](benchmarks.md). Measure client-side first visible output and
update-gap distributions, server-reported generation throughput, CPU time and
the final usage/output. Include concurrent requests, logprobs and a slow reader.
Internal TTFT and inter-token metrics measure engine events and exclude HTTP
delivery, so they cannot establish this latency improvement by themselves.

Keep raw results in ignored storage and summarize the configuration, source,
repetitions, before/after measurements and limitations in the PR. A lower
global polling timeout changes idle CPU work for every deployment; retain
the existing default unless broader load evidence justifies that change.
