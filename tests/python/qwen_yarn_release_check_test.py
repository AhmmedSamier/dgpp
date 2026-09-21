#!/usr/bin/env python3
"""Offline release-check regressions; no checkpoint, GPU or sockets required."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_yarn_release_check as check


class FakeLane:
    """Only the most recently submitted prompt fits in this prefix cache."""

    def __init__(self):
        self.name, self.url, self.model = "dgpp", "test:1234", "reasoning-checkpoint"
        self.surface = {"context": {"request_limit_tokens": 524288},
                        "rope_scaling": {"factor": 2}}
        self.calls = []
        self.last_prompt = None
        self.fail_at = None
        self.fail_concurrent = False
        self.computed = 0
        self.lock = threading.Lock()

    def reported_limit(self):
        return self.surface["context"]["request_limit_tokens"]

    def metrics(self):
        return {"active": 0, "queued": 0, "prompt_tokens_computed": self.computed,
                "prefill_ms": float(self.computed)}

    def ask(self, prompt, max_tokens, timeout=None):
        with self.lock:
            self.calls.append((prompt, max_tokens, timeout))
            if len(self.calls) == self.fail_at or (timeout is not None and self.fail_concurrent):
                raise TimeoutError("injected interruption")
            cached = 5000 if prompt == self.last_prompt else 0
            self.last_prompt = prompt
            self.computed += 5000 - cached
        question = re.search(r"What is the vault code for shipment (VL-\d+)", prompt)
        answer = "still reasoning"
        if question and max_tokens >= 512:
            answer = re.search(r"vault code for shipment " + question[1] + r" is (\d+)", prompt)[1]
        return {"usage": {"prompt_tokens": 5000, "prompt_tokens_details": {"cached_tokens": cached}},
                "ttft_ms": 100., "last_ms": 264., "tokens": 11, "wall_ms": 270.,
                "finish": "stop", "text": answer,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


class ReleaseCheckTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "record.json"
        self.lane = FakeLane()
        self.argv = ["check", "--run", "--host", "test", "--port", "1234",
                     "--lengths", "2048", "--json-out", str(self.path)]

    def run_check(self, *extra):
        sampler = MagicMock()
        sampler.report.return_value = {"peak_mib": None, "note": "offline fixture"}
        with patch.object(sys, "argv", self.argv + list(extra)), \
                patch.object(check, "Lane", return_value=self.lane), \
                patch.object(check, "MemorySampler", return_value=sampler), \
                redirect_stdout(io.StringIO()):
            return check.main()

    def record(self):
        return json.loads(self.path.read_text())

    def slot(self):
        return self.record()["results"]["dgpp"]["lengths"]["2048"]

    def test_defaults_allow_reasoning_and_match_two_slot_template(self):
        self.assertEqual(self.run_check(), 0)
        record = self.record()
        self.assertEqual(record["args"]["probe_tokens"], 768)
        self.assertEqual(record["args"]["concurrency"], 2)
        self.assertEqual(self.slot()["retrieval"]["hits"], 5)
        self.assertEqual(record["status"], "completed")

    def test_decode_and_concurrent_pace_use_milliseconds_once(self):
        self.run_check()
        self.assertEqual(self.slot()["decode"]["ms_per_token"], 16.4)
        self.assertEqual(self.slot()["concurrent"]["per_token_ms"], [16.4, 16.4])

    def test_missing_or_single_token_stamps_have_no_decode_rate(self):
        for first, last, tokens in ((None, None, 0), (100, None, 2), (100, 100, 1)):
            self.assertIsNone(check.decode_ms_per_token(
                {"ttft_ms": first, "last_ms": last, "tokens": tokens}))

    def test_cache_reuse_repeats_last_probe_before_eviction(self):
        self.run_check()
        slot = self.slot()
        self.assertEqual(slot["cache_reuse"]["prompt_sha256"],
                         slot["retrieval"]["probes"][-1]["prompt_sha256"])
        self.assertEqual(slot["cache_reuse"]["cached_ratio"], 1.)
        self.assertNotIn("cache_prime", slot)

    def test_timeouts_cover_serial_execution_of_all_streams(self):
        prompts = ["short", "x" * 4000, "last"]
        check.concurrent_phase(self.lane, prompts, 768)
        self.assertEqual([call[2] for call in self.lane.calls], [2706] * 3)

    def test_requested_concurrency_can_exceed_needle_count(self):
        self.run_check("--depths", "0.5", "--concurrency", "3")
        self.assertEqual(self.slot()["concurrent"]["completed"], 3)

    def test_interruption_preserves_probes_and_resume_skips_completed_work(self):
        self.lane.fail_at = 4  # Calibration and two probes completed.
        with self.assertRaises(TimeoutError):
            self.run_check()
        partial = self.record()
        probes = partial["results"]["dgpp"]["lengths"]["2048"]["retrieval"]["probes"]
        self.assertEqual(len(probes), 2)
        self.assertEqual(partial["status"], "error")
        self.assertIn("TimeoutError", partial["error"])
        self.lane.calls.clear()
        self.lane.fail_at = None
        self.assertEqual(self.run_check("--resume"), 0)
        # Three remaining probes, reuse, decode, two streams, five control probes.
        self.assertEqual(len(self.lane.calls), 12)
        self.assertEqual(self.slot()["retrieval"]["probes"][:2], probes)
        self.assertNotIn("error", self.record())
        self.assertEqual(len(self.record()["attempts"]), 2)

    def test_resume_primes_cache_when_retrieval_was_already_complete(self):
        original = check.save_record
        def interrupt_after_retrieval(path, record):
            original(path, record)
            slot = record["results"].get("dgpp", {}).get("lengths", {}).get("2048", {})
            if slot.get("retrieval", {}).get("requests") == 5 and record["status"] == "running":
                raise KeyboardInterrupt()
        with patch.object(check, "save_record", side_effect=interrupt_after_retrieval):
            with self.assertRaises(KeyboardInterrupt):
                self.run_check()
        self.assertEqual(self.record()["status"], "interrupted")
        self.lane.last_prompt = None  # The server was restarted between invocations.
        self.run_check("--resume")
        self.assertEqual(self.slot()["cache_prime"]["cached_tokens"], 0)
        self.assertEqual(self.slot()["cache_reuse"]["cached_ratio"], 1.)

    def test_completed_resume_sends_no_generation_requests(self):
        self.run_check()
        self.lane.calls.clear()
        self.assertEqual(self.run_check("--resume"), 0)
        self.assertEqual(self.lane.calls, [])

    def test_resume_rejects_changed_recipe_without_overwriting_record(self):
        self.run_check()
        before = self.path.read_bytes()
        self.lane.calls.clear()
        with self.assertRaisesRegex(ValueError, "same check arguments"):
            self.run_check("--resume", "--seed", "123")
        self.lane.surface["rope_scaling"]["factor"] = 3
        with self.assertRaisesRegex(ValueError, "context settings"):
            self.run_check("--resume")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.lane.calls, [])

    def test_failed_streams_fail_verdict_and_are_retried_on_resume(self):
        self.lane.fail_concurrent = True
        self.assertEqual(self.run_check(), 1)
        self.assertEqual(self.slot()["concurrent"]["completed"], 0)
        self.lane.fail_concurrent = False
        self.lane.calls.clear()
        self.assertEqual(self.run_check("--resume"), 0)
        self.assertEqual(len(self.lane.calls), 2)

    def test_atomic_write_preserves_previous_record_on_serialization_failure(self):
        check.save_record(self.path, {"valid": 1})
        with self.assertRaises(ValueError):
            check.save_record(self.path, {"invalid": float("nan")})
        self.assertEqual(self.record(), {"valid": 1})
        self.assertEqual(list(Path(self.directory.name).iterdir()), [self.path])

    def test_opt_in_and_resume_path_checks_happen_before_endpoint_access(self):
        for argv in (["check"], ["check", "--run", "--resume"]):
            with patch.object(sys, "argv", argv), patch.dict("os.environ", {}, clear=True), \
                    patch.object(check, "Lane") as lane, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    check.main()
                self.assertEqual(error.exception.code, 2)
                lane.assert_not_called()


if __name__ == "__main__":
    unittest.main()
