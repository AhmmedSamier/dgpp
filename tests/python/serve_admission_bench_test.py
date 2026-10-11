#!/usr/bin/env python3
"""Exercise benchmark evidence checks over real HTTP/SSE transport."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "serve_admission_bench.py"
spec = importlib.util.spec_from_file_location("admission_bench", SCRIPT)
bench_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench_module)


@contextmanager
def server(fault=None):
    state = {"requests": 0, "active": 0, "tokens": 0, "prompt": 0, "prefills": {}}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_json({"data": [{"id": "fixture"}]})
                return
            with lock:
                readers = [{"id": "wrong" if fault == "identity" else rid, "processed_tokens": 4}
                           for rid in state["prefills"]]
                self.send_json({"service": {"engine_failed": False, "requests_total": state["requests"]},
                                "scheduler": {"active": state["active"], "queued": 0,
                                              "tokens_generated": state["tokens"],
                                              "prompt_tokens_computed": state["prompt"]},
                                "prefill": {"requests": readers}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            long = body["prompt"].startswith("Document:")
            with lock:
                state["requests"] += 1
                rid = "cmpl-" + str(state["requests"])
                state["active"] += 1
                if long:
                    state["prefills"][rid] = True
            try:
                if long:
                    time.sleep(.06)
                with lock:
                    state["prefills"].pop(rid, None)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i in range(body["max_tokens"]):
                    self.wfile.write(("data: " + json.dumps({"id": rid, "choices": [{"text": str(i)}]})
                                      + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(.002)
                if fault != "usage":
                    usage = {"prompt_tokens": 16, "completion_tokens": body["max_tokens"] + (fault == "length"),
                             "prompt_tokens_details": {"cached_tokens": int(fault == "cache")}}
                    self.wfile.write(("data: " + json.dumps({"id": rid, "choices": [], "usage": usage})
                                      + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    state["active"] -= 1
                    state["prompt"] += 16
                    state["tokens"] += body["max_tokens"]
                    if fault == "traffic":
                        state["requests"] += 1

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever)
    thread.start()
    args = SimpleNamespace(url=f"http://127.0.0.1:{http.server_port}", timeout=2,
                           seed=104, prompt_words=128, arrivals=2, interval=.001,
                           decode_tokens=128, arrival_tokens=8)
    try:
        yield bench_module.Bench(args)
    finally:
        http.shutdown()
        thread.join()
        http.server_close()


class AdmissionBenchmarkTest(unittest.TestCase):
    def test_native_overlap_and_fixed_work_are_recorded(self):
        with server() as bench:
            result = bench.run("mixed_short", 0)
        self.assertEqual(len(result["requests"]), 3)
        long = next(r for r in result["requests"] if r["name"] == "long")
        self.assertEqual(result["native_prefill_observation"][0]["id"], long["id"])
        self.assertEqual(result["generated_tokens"], 137)
        self.assertEqual(result["computed_prompt_tokens"], 48)

    def test_overlap_must_match_the_long_response(self):
        with server("identity") as bench:
            with self.assertRaisesRegex(RuntimeError, "did not match"):
                bench.run("sustained_idle", 0)

    def test_unrelated_traffic_invalidates_the_run(self):
        with server("traffic") as bench:
            with self.assertRaisesRegex(RuntimeError, "unrelated traffic"):
                bench.run("solo_decode", 0)

    def test_usage_output_length_and_cache_evidence_are_required(self):
        for fault, message in (("usage", "incomplete"), ("length", "output length"), ("cache", "cache hit")):
            with self.subTest(fault=fault), server(fault) as bench:
                with self.assertRaisesRegex(RuntimeError, message):
                    bench.stream("probe", "hello", 4)

    def test_cancellation_drains_before_the_next_run(self):
        with server() as bench:
            record = bench.stream("cancel", "hello", 16, cancel=True)
            self.assertTrue(record["cancelled"])
            self.assertEqual(bench.idle()["scheduler"]["active"], 0)
            self.assertEqual(bench.stream("recovery", "hello", 4)["usage"]["completion_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
