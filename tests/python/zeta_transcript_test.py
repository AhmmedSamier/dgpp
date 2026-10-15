"""Zeta-2.1 end-to-end transcript gate (docs/zeta_plan.md slice 5).

Boots the real server on the landed zeta-2.1 checkpoint (world 1),
POSTs the checkpoint's own sample.prompt to /v1/completions, and asserts
the greedy completion is byte-exact against the checkpoint's
sample.output — the only committed gate that runs the real 32-layer
weights end to end (the fixture tests use a synthetic 2-layer model;
the loader tests pin byte placement but never run the math; the
tokenizer gates never touch the weights).

Skips unless DGPP_SERVE_TEST_BINARY points at a built server AND the
checkpoint sits in the HF cache. Needs a GPU and ~20 GiB of resident
cache for the first boot (later boots reuse the resident image).
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import cluster_doctor

MODEL = "zed-industries/zeta-2.1"


def free_port():
    with closing(socket.socket()) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class ZetaTranscriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = os.environ.get("DGPP_SERVE_TEST_BINARY")
        if not binary:
            raise unittest.SkipTest("set DGPP_SERVE_TEST_BINARY to the freshly built server")
        cls.binary = str(Path(binary).resolve(strict=True))
        hub = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache/huggingface/hub")
        try:
            snapshot = cluster_doctor.cached_snapshot(MODEL, hub)
            prompt = snapshot / "sample.prompt"
            output = snapshot / "sample.output"
            if not prompt.is_file() or not output.is_file():
                raise ValueError("sample.prompt/sample.output missing")
            size = cluster_doctor.checkpoint_size(snapshot)
        except ValueError as error:
            raise unittest.SkipTest(f"zeta-2.1 checkpoint unavailable: {error}")
        cls.snapshot = snapshot
        cls.prompt_text = prompt.read_text()
        cls.expected = output.read_text()
        cls.checkpoint_bytes = size

    def boot_server(self, directory):
        port = free_port()
        resolved = {
            "model": MODEL,
            "engine": {
                "max_concurrency": 1,
                "kv_capacity": 4096,
                "default_max_tokens": 512,
                "queue_limit": 2,
                "max_connections": 8,
                "decode_graph": True,
                "mtp": False,
                "prefix_cache_gib": 0.25,
                "stats_interval_s": 60,
            },
            "paths": {
                "resident_cache": "",
                "log_dir": str(Path(directory) / "log"),
                "stage_dir": str(Path(directory) / "stage"),
                "release_dir": str(Path(directory) / "releases"),
            },
            "http": {"bind_host": "127.0.0.1", "port": port},
            "nodes": ["127.0.0.1"],
            "ports": {"http": port, "fabric": free_port(), "journal": free_port()},
            "node_env": [{"HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"),
                          "DGPP_RESIDENT_CACHE_DIR": os.environ.get("DGPP_RESIDENT_CACHE_DIR",
                                                                     "~/.cache/dgpp/resident")}],
        }
        config = Path(directory) / "resolved.json"
        config.write_text(json.dumps(resolved))
        log = open(Path(directory) / "serve.log", "w+")
        process = subprocess.Popen([self.binary, "--config", str(config), "--rank", "0"],
                                   stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    self.fail(f"server exited rc={process.returncode}:\n{log.read()[-4000:]}")
                try:
                    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=5)) as conn:
                        conn.request("GET", "/v1/models")
                        if conn.getresponse().status == 200:
                            return process, log, port
                except OSError:
                    pass
                time.sleep(2)
            self.fail("server did not serve within 600 s")
        except Exception:
            if process.poll() is None:
                process.kill()
            raise
        return process, log, port  # unreachable; keeps the shape obvious

    def complete(self, port, max_tokens):
        body = json.dumps({"model": MODEL, "prompt": self.prompt_text,
                           "max_tokens": max_tokens, "temperature": 0})
        with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=300)) as conn:
            conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
            response = conn.getresponse()
            payload = response.read()
            self.assertEqual(response.status, 200, payload[:500].decode("utf8", "replace"))
            return json.loads(payload)

    def test_greedy_completion_matches_sample_output(self):
        with tempfile.TemporaryDirectory(prefix="dgpp-zeta-transcript-") as directory:
            process, log, port = self.boot_server(directory)
            try:
                first = self.complete(port, 512)
                self.assertEqual(first["choices"][0]["finish_reason"], "stop")
                self.assertEqual(first["choices"][0]["text"], self.expected,
                                 "greedy transcript drifted from the checkpoint's sample.output")
                second = self.complete(port, 512)
                self.assertEqual(second["choices"][0]["text"], self.expected,
                                 "repeat completion is not deterministic")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                log.close()


if __name__ == "__main__":
    unittest.main()
