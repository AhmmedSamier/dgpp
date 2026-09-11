"""Checkpoint distribution and NIC discovery tests; no real models or SSH."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import cache_sync
import cluster_doctor
import discover_roce
import download_model


class SetupToolsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "head/hub"
        self.snapshot = self.cache / "models--org--model/snapshots/revision"
        self.snapshot.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json"):
            (self.snapshot / name).write_text("{}")
        (self.snapshot / "chat_template.jinja").write_text("{{ messages }}")
        blob = self.snapshot.parent.parent / "blobs/weights"
        blob.parent.mkdir()
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        blob.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
        (self.snapshot / "model.safetensors").symlink_to("../../blobs/weights")
        cache_sync.activate(self.snapshot)
        self.config = self.root / "cluster.json"
        self.config.write_text(json.dumps({"model": "org/model", "world_size": 4}))
        self.env_file = self.root / ".env"
        self.env_file.write_text(
            f'DGPP_NODES="127.0.0.1 peer1 peer2 peer3"\nDGPP_SSH_USER="tester"\n'
            f'HF_HUB_CACHE="{self.cache}"\nHF_ACCESS_TOKEN="not-for-peers"\n'
            'DGPP_NODE_OVERRIDES=\'{"peer2":{"HF_HUB_CACHE":"/peer/cache"}}\'\n')
        self.env = {"PATH": os.environ["PATH"], "HOME": str(self.root / "head"),
                    "DGPP_ENV_FILE": str(self.env_file),
                    "DGPP_CLUSTER_CONFIG": str(self.root / "wrong.json")}
        self.output = io.StringIO()
        redirect = contextlib.redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def invoke(self, *args):
        with patch.dict(os.environ, self.env, clear=True), patch.object(download_model.shutil, "which", return_value="available"):
            download_model.main(["--config", str(self.config), *args])

    def test_one_hub_download_then_ordered_peer_sync(self):
        events = []
        def download(*args):
            events.append(("hub", args))
            return self.snapshot
        def sync(snapshot, target, env):
            events.append((target, env))
            self.assertEqual(snapshot, self.snapshot)
            self.assertNotIn("HF_ACCESS_TOKEN", env)
        with patch.object(download_model, "download", side_effect=download) as hub, patch.object(download_model, "sync_snapshot", side_effect=sync):
            self.invoke()
        hub.assert_called_once_with("org/model", "main", self.cache)
        self.assertEqual([event[0] for event in events], ["hub", "tester@peer1", "tester@peer2", "tester@peer3"])
        self.assertEqual(events[2][1]["HF_HUB_CACHE"], "/peer/cache")

    def test_sync_only_never_contacts_hub(self):
        with patch.object(download_model, "download") as hub, patch.object(download_model, "sync_snapshot") as sync:
            self.invoke("--sync-only")
        hub.assert_not_called()
        self.assertEqual(sync.call_count, 3)

    def test_verify_only_does_not_download_sync_or_activate(self):
        with patch.object(download_model, "download") as hub, patch.object(download_model, "sync_snapshot") as sync, patch.object(download_model, "activate") as activate, patch.object(download_model, "peer_cache") as verify:
            self.invoke("--verify-only")
        hub.assert_not_called()
        sync.assert_not_called()
        activate.assert_not_called()
        self.assertEqual(verify.call_count, 3)
        self.assertTrue(all(call.kwargs == {"verify": True, "revision": "revision"} for call in verify.call_args_list))

    def test_local_only_skips_peers(self):
        with patch.object(download_model, "sync_snapshot") as sync, patch.object(download_model, "peer_cache") as verify:
            self.invoke("--verify-only", "--local-only")
        sync.assert_not_called()
        verify.assert_not_called()

    def test_standalone_model_uses_standard_cache(self):
        self.env_file.write_text("")
        # Put the fixture at the normal per-user cache location.
        standard = self.root / "head/.cache/huggingface/hub"
        shutil.copytree(self.cache, standard, symlinks=True)
        snapshot = standard / "models--org--model/snapshots/revision"
        with patch.dict(os.environ, self.env, clear=True), patch.object(download_model, "download", return_value=snapshot) as hub, patch.object(download_model, "sync_snapshot") as sync:
            download_model.main(["--model", "org/model"])
        hub.assert_called_once_with("org/model", "main", standard)
        sync.assert_not_called()

    def test_cache_precedence(self):
        self.assertEqual(cluster_doctor.cache_root({"HF_HOME": "/hf"}), Path("/hf/hub"))
        self.assertEqual(cluster_doctor.cache_root({"HF_HOME": "/hf", "HF_HUB_CACHE": "/hub"}), Path("/hub"))

    def test_hub_api_downloads_full_snapshot_into_selected_cache(self):
        snapshot_download = Mock(return_value=str(self.snapshot))
        with patch.dict(sys.modules, {"huggingface_hub": Mock(snapshot_download=snapshot_download)}):
            self.assertEqual(download_model.download("org/model", "commit", self.cache), self.snapshot)
        snapshot_download.assert_called_once_with(repo_id="org/model", revision="commit", cache_dir=str(self.cache), max_workers=4)

    def test_pinned_revision_requires_activation_before_sync(self):
        pinned = self.snapshot.parent / "pinned"
        shutil.copytree(self.snapshot, pinned, symlinks=True)
        with patch.object(download_model, "download", return_value=pinned), patch.object(download_model, "sync_snapshot") as sync:
            with self.assertRaisesRegex(ValueError, "another revision is active"):
                self.invoke("--revision", "pinned")
            sync.assert_not_called()
            self.invoke("--revision", "pinned", "--activate")
        self.assertEqual((pinned.parent.parent / "refs/main").read_text().strip(), "pinned")
        self.assertEqual(sync.call_count, 3)

    def test_failed_peer_stops_sync(self):
        with patch.object(download_model, "sync_snapshot", side_effect=RuntimeError("transfer failed")) as sync:
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                self.invoke("--sync-only")
        sync.assert_called_once()

    def test_wrong_coordinator_rejected_before_download(self):
        with patch.object(cluster_doctor.socket, "gethostbyname", side_effect=OSError("not local")), patch.object(download_model, "download") as hub:
            with self.assertRaisesRegex(ValueError, "rank 0"):
                self.invoke()
        hub.assert_not_called()

    def test_manifest_contains_only_selected_snapshot_and_blobs(self):
        repository = self.snapshot.parent.parent
        (repository / "snapshots/other").mkdir()
        (repository / "snapshots/other/private").write_text("unrelated")
        (self.cache / "token").write_text("not-for-peers")
        self.assertEqual(set(cache_sync.snapshot_files(self.snapshot).splitlines()), {
            "blobs/weights", "snapshots/revision/config.json", "snapshots/revision/tokenizer.json",
            "snapshots/revision/chat_template.jinja", "snapshots/revision/model.safetensors"})

    def test_absolute_and_escaping_symlinks_rejected(self):
        link = self.snapshot / "bad-link"
        for target in (str(self.snapshot / "config.json"), "../../refs/main", "../../blobs/missing"):
            with self.subTest(target=target):
                link.symlink_to(target)
                with self.assertRaises((ValueError, OSError)):
                    cache_sync.snapshot_files(self.snapshot)
                link.unlink()

    def test_activation_keeps_old_ref_if_checkpoint_invalid(self):
        ref = self.snapshot.parent.parent / "refs/main"
        ref.write_text("old-revision")
        (self.snapshot / "model.safetensors").unlink()
        with self.assertRaises(OSError):
            cache_sync.activate(self.snapshot)
        self.assertEqual(ref.read_text(), "old-revision")

    def test_failed_transfer_never_activates_peer(self):
        with patch.object(cache_sync, "peer_cache", return_value={"repository": "/cache/models--org--model"}), patch.object(cache_sync, "ssh", return_value="/tmp/dgpp-cache-sync.12345678") as ssh, patch.object(cache_sync.subprocess, "run", side_effect=[Mock(), subprocess.CalledProcessError(23, "rsync")]):
            with self.assertRaises(subprocess.CalledProcessError):
                cache_sync.sync_snapshot(self.snapshot, "tester@fake-peer", {})
        self.assertFalse(any("--activate" in call.args[1] for call in ssh.call_args_list))
        self.assertIn("rmdir", ssh.call_args_list[-1].args[1])

    @unittest.skipUnless(shutil.which("rsync"), "rsync integration requires rsync")
    def test_real_rsync_over_local_test_transport(self):
        peer = self.root / "peer"
        peer.mkdir()
        binaries = self.root / "bin"
        binaries.mkdir()
        fake_ssh = binaries / "ssh"
        shutil.copyfile(ROOT / "tests/fixtures/local_ssh.py", fake_ssh)
        fake_ssh.chmod(0o755)
        env = {**self.env, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
               "DGPP_TEST_PEER_HOME": str(peer)}
        destination = peer / ".cache/huggingface/hub/models--org--model"
        (destination / "refs").mkdir(parents=True)
        (destination / "refs/main").write_text("old-revision")
        (destination / "snapshots/old-revision").mkdir(parents=True)
        unrelated = destination / "snapshots/old-revision/keep"
        unrelated.write_text("keep this")
        # Existing same-size corrupt content must be repaired, not size-skipped.
        blob = destination / "blobs/weights"
        blob.parent.mkdir()
        original = (self.snapshot / "model.safetensors").read_bytes()
        blob.write_bytes(b"x" * len(original))
        with patch.dict(os.environ, env, clear=True):
            cache_sync.sync_snapshot(self.snapshot, "tester@fake-peer", {})
            verified = cache_sync.peer_cache("tester@fake-peer", "org/model", {}, verify=True, revision="revision")
            with self.assertRaisesRegex(RuntimeError, "revision"):
                cache_sync.peer_cache("tester@fake-peer", "org/model", {}, verify=True, revision="wrong")
        self.assertEqual((destination / "refs/main").read_text().strip(), "revision")
        copied = destination / "snapshots/revision/model.safetensors"
        self.assertTrue(copied.is_symlink())
        self.assertEqual(copied.read_bytes(), original)
        self.assertEqual(unrelated.read_text(), "keep this")
        self.assertEqual(verified["repository"], str(destination))
        self.assertFalse((peer / ".cache/huggingface/token").exists())

    def test_explicit_config_forwarded_by_bash_wrapper(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/serve_run.sh"), "resolve", "--config", str(self.config)], env=self.env, text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["model"], "org/model")

    def test_discovery_maps_spark_device_to_interface(self):
        port = self.root / "infiniband/rocep1s0f0/ports/1"
        for directory in ("gid_attrs/types", "gid_attrs/ndevs", "gids"):
            (port / directory).mkdir(parents=True)
        (port / "state").write_text("4: ACTIVE")
        (port / "link_layer").write_text("Ethernet")
        for index, gid in (("0", "fe80::1"), ("3", "::ffff:192.0.2.11")):
            (port / "gid_attrs/types" / index).write_text("RoCE v2")
            (port / "gid_attrs/ndevs" / index).write_text("enp1s0f0np0")
            (port / "gids" / index).write_text(gid)
        addresses = [{"ifname": "enp1s0f0np0", "mtu": 9000,
                      "addr_info": [{"local": "192.0.2.11", "prefixlen": 24}]}]
        with patch.object(cluster_doctor, "command", return_value=Mock(returncode=0, stdout=json.dumps(addresses))):
            rows = cluster_doctor.roce_inventory(self.root / "infiniband")
        self.assertTrue(rows[0]["usable"])
        self.assertEqual(rows[0]["gids"], [{"index": 3, "address": "192.0.2.11", "interface": "enp1s0f0np0", "mtu": 9000, "ips": ["192.0.2.11/24"]}])
        self.assertEqual(discover_roce.suggestions(rows), {"DGPP_ROCE_DEVICES": "rocep1s0f0", "DGPP_ROCE_GID_INDICES": "3"})
        rows[0]["gids"].append({"index": 4})
        self.assertNotIn("DGPP_ROCE_GID_INDICES", discover_roce.suggestions(rows))
        rows[0]["usable"] = False
        self.assertIsNone(discover_roce.suggestions(rows))

    def test_discovery_does_not_guess_more_than_two_lanes(self):
        self.assertIsNone(discover_roce.suggestions([]))
        self.assertIsNone(discover_roce.suggestions([{"usable": True}] * 3))


if __name__ == "__main__":
    unittest.main()
