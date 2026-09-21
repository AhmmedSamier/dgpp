"""Exercise the server's settings handshake before model loading or GPU use."""
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest


class ServeStartupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = os.environ.get("DGPP_SERVE_TEST_BINARY")
        if not binary:
            raise unittest.SkipTest("set DGPP_SERVE_TEST_BINARY to the freshly built server")
        cls.binary = str(Path(binary).resolve(strict=True))

    def run_world(self, settings):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix="dgpp-startup-") as directory:
            # The invalid local HTTP capacity ends every rank after the handshake,
            # before checkpoint access, CUDA initialization or RDMA setup.
            common = [self.binary, "--checkpoint-dir", str(Path(directory) / "unused"),
                      "--world", str(len(settings)), "--journal-port", str(port),
                      "--fabric-port", "0", "--max-connections", "0",
                      "--rendezvous-timeout-ms", "5000"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "DGPP_LOG_LEVEL": "info"}
            processes, files = [], []
            try:
                for rank, (slots, threshold) in enumerate(settings):
                    args = common + ["--rank", str(rank), "--max-concurrency", str(slots)]
                    if rank:
                        args += ["--peer", "127.0.0.1"]
                    if threshold is not None:
                        args += ["--graph-batch-min-live", str(threshold)]
                    output = open(Path(directory) / f"rank{rank}.log", "w+")
                    files.append(output)
                    processes.append(subprocess.Popen(args, env=env, stdout=output,
                                                      stderr=subprocess.STDOUT))
                logs = []
                for process, output in zip(processes, files):
                    process.wait(timeout=15)
                    output.seek(0)
                    log = output.read()
                    self.assertEqual(process.returncode, 1, log)
                    self.assertIn("all capacity knobs must be >= 1", log)
                    logs.append(log)
                self.assertIn(f"settings pushed to {len(settings) - 1} peer(s)", logs[0])
                head = logs[0].split("peer(s): ", 1)[1].splitlines()[0]
                for log in logs[1:]:
                    marker = " ; runs: " if "settings override" in log else "settings from rank 0: "
                    self.assertIn(marker, log)
                    self.assertEqual(log.split(marker, 1)[1].splitlines()[0], head)
                return logs
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                for output in files:
                    output.close()

    def test_matching_defaults_are_quiet(self):
        for slots in (1, 2, 4, 8):
            with self.subTest(slots=slots):
                logs = self.run_world([(slots, None), (slots, 0)])
                self.assertNotIn("settings override", logs[1])
                self.assertIn("settings from rank 0:", logs[1])
                self.assertIn(f"batchmin={min(2, slots)} ", logs[1])

    def test_explicit_default_matches_automatic_on_every_peer(self):
        for slots in (1, 4):
            default = min(2, slots)
            for head in (0, default):
                with self.subTest(slots=slots, head=head):
                    logs = self.run_world([(slots, head), (slots, None),
                                           (slots, 0), (slots, default)])
                    for log in logs[1:]:
                        self.assertNotIn("settings override", log)
                        self.assertIn("settings from rank 0:", log)
                        self.assertIn(f"batchmin={default} ", log)

    def test_different_threshold_still_warns_and_adopts_head(self):
        for head, peer, effective in ((0, 1, 2), (3, 0, 3), (3, 4, 3)):
            with self.subTest(head=head, peer=peer):
                log = self.run_world([(4, head), (4, peer)])[1]
                self.assertIn("WARN", log)
                self.assertIn("rank 0's settings override this rank's own", log)
                own, adopted = log.split(" ; runs: ", 1)
                self.assertIn(f"batchmin={peer or 2} ", own)
                self.assertIn(f"batchmin={effective} ", adopted)

    def test_different_concurrency_still_warns(self):
        log = self.run_world([(4, 0), (1, 0)])[1]
        self.assertIn("rank 0's settings override this rank's own", log)
        own, adopted = log.split(" ; runs: ", 1)
        self.assertIn("conc=1 ", own)
        self.assertIn("batchmin=1 ", own)
        self.assertIn("conc=4 ", adopted)
        self.assertIn("batchmin=2 ", adopted)


if __name__ == "__main__":
    unittest.main()
