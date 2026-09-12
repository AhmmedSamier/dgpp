"""The cluster launcher's recorded-deployment scan: `down` and `status`
without --config (or with --all) act on every deployment staged under
DGPP_LOG_DIR/deployments, not on the default deployment file. Single-node
namespaces only (no SSH); the "ranks" are recorded `sleep` processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import cluster_process

LAUNCHER = ROOT / "scripts" / "dgpp-cluster"


class LauncherScanTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.log_root = self.root / "log"
        (self.root / ".env").write_text(
            'DGPP_NODES="127.0.0.1"\n'
            f'DGPP_LOG_DIR="{self.log_root}"\n'
            f'DGPP_STAGE_DIR="{self.root}/stage"\n'
            f'DGPP_RELEASE_DIR="{self.root}/releases"\n'
        )
        self.environ = {
            "DGPP_ENV_FILE": str(self.root / ".env"),
            "DGPP_BUILD_DIR": str(self.root / "no-build"),  # status's --version probe: "?"
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": os.environ.get("HOME", str(self.root)),
        }

    def namespace(self, name, model, record_source=True):
        """A staged single-node deployment, as the launcher leaves it."""
        d = self.log_root / "deployments" / name
        d.mkdir(parents=True)
        (d / "cluster.resolved.json").write_text(json.dumps({
            "model": model, "engine": {}, "nodes": ["127.0.0.1"], "ssh_user": "tester",
            "ports": {"http": 18080, "fabric": 29970, "journal": 29971},
            "http": {"bind_host": "127.0.0.1", "port": 18080},
            "paths": {"log_dir": str(d), "stage_dir": f"{self.root}/stage/deployments/{name}",
                      "release_dir": f"{self.root}/releases"},
            "node_env": [{}],
        }))
        if record_source:
            (d / "deployment.path").write_text(f"{self.root}/{name}.json\n")
        return d

    def launch(self, d):
        """Rank 0 of the namespace: a recorded session-leader sleep."""
        proc = cluster_process.launch(d / "rank0.process.json", ["sleep", "300"], d / "serve_r0.log", d)
        self.addCleanup(self.kill, proc)
        return proc

    @staticmethod
    def kill(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()

    def launcher(self, *argv):
        return subprocess.run([sys.executable, str(LAUNCHER), *argv], env=self.environ,
                              text=True, capture_output=True, timeout=120)

    def test_down_without_config_stops_every_running_deployment(self):
        running = self.namespace("aaaa", "org/running")
        also = self.namespace("bbbb", "org/also-running", record_source=False)
        self.namespace("cccc", "org/idle")
        first, second = self.launch(running), self.launch(also)
        result = self.launcher("down")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = result.stdout
        self.assertIn(f"=== stopping org/running on 1 node(s) [aaaa] ({self.root}/aaaa.json): "
                      f"rank 0 (127.0.0.1) pid {first.pid}", out)
        self.assertIn(f"=== stopping org/also-running on 1 node(s) [bbbb] (deployment file not recorded): "
                      f"rank 0 (127.0.0.1) pid {second.pid}", out)
        self.assertIn("org/idle on 1 node(s) [cccc]", out)
        self.assertIn("): not running", out)
        self.assertIn("2 deployment(s) stopped", out)
        for proc in (first, second):
            proc.wait(timeout=10)
        for name in ("aaaa", "bbbb"):
            self.assertIsNone(cluster_process.running(self.log_root / "deployments" / name / "rank0.process.json"))

    def test_all_flag_is_the_same_scan(self):
        proc = self.launch(self.namespace("dddd", "org/model"))
        result = self.launcher("down", "--all")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"rank 0 (127.0.0.1) pid {proc.pid}", result.stdout)
        self.assertIn("1 deployment(s) stopped", result.stdout)
        proc.wait(timeout=10)

    def test_status_without_config_lists_every_recorded_deployment(self):
        live = self.namespace("eeee", "org/live")
        (live / "serve_r0.log").write_text("2026-09-12 15:31:40.001 INFO  serve: listening on :18080\n")
        proc = self.launch(live)
        stopped = self.namespace("ffff", "org/stopped")
        (stopped / "serve_r0.log").write_text(
            "2026-09-12 05:26:48.583 INFO  rank 0: memory plan total 108.38 GiB\n"
            "2026-09-12 05:26:48.665 ERROR serve: rank 0: the configuration needs 108.38 GiB plus 8.00 GiB headroom\n")
        self.namespace("gggg", "org/never-booted")
        result = self.launcher("status")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = result.stdout
        # The running deployment: its ranks and the end of rank 0's log.
        self.assertIn(f"=== org/live on 1 node(s) [eeee] ({self.root}/eeee.json)", out)
        self.assertIn(f"rank 0 (127.0.0.1): alive ({proc.pid})", out)
        self.assertIn("  2026-09-12 15:31:40.001 INFO  serve: listening on :18080", out)
        # Stopped ones: one line each, saying how they ended; no rank listing.
        self.assertIn(f"org/stopped on 1 node(s) [ffff] ({self.root}/ffff.json): not running "
                      "(rank 0's log ends 2026-09-12 05:26:48: ERROR serve: rank 0: the configuration needs", out)
        self.assertIn("org/never-booted on 1 node(s) [gggg]", out)
        self.assertIn(": not running (no rank 0 log)", out)
        self.assertNotIn("=== org/stopped", out)
        self.assertNotIn("DOWN", out)
        self.assertIsNotNone(cluster_process.running(self.log_root / "deployments/eeee/rank0.process.json"))

    def test_nothing_recorded_is_not_an_error(self):
        for verb in ("down", "status"):
            result = self.launcher(verb)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"no deployment recorded under {self.log_root}/deployments", result.stdout)
        self.namespace("gggg", "org/idle")
        result = self.launcher("down")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no deployment was running; nothing stopped", result.stdout)

    def test_all_rejects_a_config_a_log_dir_and_other_verbs(self):
        config = self.root / "x.json"
        config.write_text(json.dumps({"model": "org/model", "world_size": 1}))
        for argv, message in ((("down", "--all", "--config", str(config)), "--all takes no --config"),
                              (("down", "--log-dir", str(self.root / "elsewhere")), "--all scans"),
                              (("up", "--all", "--config", str(config)), "--all applies to down and status")):
            result = self.launcher(*argv)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn(message, result.stderr)

    def test_down_with_config_names_a_deployment_that_is_not_running(self):
        config = self.root / "never.json"
        config.write_text(json.dumps({"model": "org/never", "world_size": 1}))
        result = self.launcher("down", "--config", str(config))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("=== org/never on 1 node(s) [", result.stdout)
        self.assertIn("no rank recorded as running; nothing to stop", result.stdout)
        self.assertNotIn("op-stream identity", result.stdout)  # never staged: nothing to collect


if __name__ == "__main__":
    unittest.main()
