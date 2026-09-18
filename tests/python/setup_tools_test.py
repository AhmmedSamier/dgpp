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
import site_env


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

    def test_empty_explicit_config_is_rejected_by_every_entry_point(self):
        commands = [("dgpp-cluster", "resolve"), ("site_env.py", "resolve"),
                    ("download_model.py",), ("discover_roce.py",),
                    ("prepare_data.py", "tokens")]
        for command in commands:
            for value in ("", "   "):
                with self.subTest(command=command, value=value):
                    result = subprocess.run([sys.executable, str(ROOT / "scripts" / command[0]),
                                             *command[1:], "--config", value], env=self.env,
                                            capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("--config must not be empty", result.stderr)
                    self.assertEqual(result.stdout, "")

    @unittest.skipUnless(shutil.which("cmake") and shutil.which("c++"), "requires CMake and a host compiler")
    def test_missing_cuda_compiler_reports_setup_action(self):
        result = subprocess.run(["cmake", "-S", str(ROOT), "-B", str(self.root / "cmake-missing"),
                                 "-DCMAKE_CUDA_COMPILER=NOTFOUND"], env=self.env,
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA compiler not found", result.stderr)
        self.assertIn("CUDACXX=", result.stderr)
        self.assertNotIn("Failed to detect a default CUDA architecture", result.stderr)

    @unittest.skipUnless(shutil.which("cmake") and shutil.which("c++"), "requires CMake and a host compiler")
    def test_invalid_explicit_cuda_root_does_not_fall_back(self):
        result = subprocess.run(["cmake", "-S", str(ROOT), "-B", str(self.root / "cmake-root"),
                                 f"-DCUDAToolkit_ROOT={self.root / 'absent'}"], env=self.env,
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No nvcc under CUDAToolkit_ROOT=", result.stderr)

    def test_release_packaging_preserves_testing_build_directory(self):
        build = self.root / "build-ci"
        build.mkdir()
        cache = build / "CMakeCache.txt"
        for build_type in ("Debug", "RelWithDebInfo", ""):
            original = f"CMAKE_BUILD_TYPE:STRING={build_type}\n"
            cache.write_text(original)
            for arguments in ([], ["--no-build"]):
                with self.subTest(build_type=build_type, arguments=arguments):
                    result = subprocess.run(["bash", str(ROOT / "scripts/release.sh"), *arguments],
                                            env={**self.env, "DGPP_BUILD_DIR": str(build)},
                                            capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("release requires a Release build", result.stderr)
                    self.assertEqual(cache.read_text(), original)

    @unittest.skipUnless(shutil.which("c++") and shutil.which("readelf"), "requires a host compiler and readelf")
    def test_release_packaging_rejects_debug_artifact(self):
        build = self.root / "build-release"
        build.mkdir()
        (build / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\nDGPP_SANITIZE:STRING=\n")
        subprocess.run(["c++", "-g", "-x", "c++", "-", "-o", str(build / "dgpp-serve")],
                       input="int main() { return 0; }\n", text=True, capture_output=True, check=True, timeout=30)
        result = subprocess.run(["bash", str(ROOT / "scripts/release.sh"), "--no-build"],
                                env={**self.env, "DGPP_BUILD_DIR": str(build)},
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("release binary contains debug information", result.stderr)

    def test_release_packaging_rejects_sanitizer_build(self):
        build = self.root / "build-release"
        build.mkdir()
        (build / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\nDGPP_SANITIZE:STRING=address\n")
        result = subprocess.run(["bash", str(ROOT / "scripts/release.sh"), "--no-build"],
                                env={**self.env, "DGPP_BUILD_DIR": str(build)},
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("release cannot contain sanitizer instrumentation", result.stderr)

    @unittest.skipUnless(shutil.which("cmake") and shutil.which("c++"), "requires CMake and a host compiler")
    def test_bundled_pcre2_install_requires_only_linked_library(self):
        # Exercise the actual dependency setup without CUDA or downloads.
        # Like PCRE2, this fixture installs both its core and POSIX libraries,
        # but the consumer links/builds only the core library.
        source = self.root / "release-fixture"
        dependency = source / "pcre2"
        dependency.mkdir(parents=True)
        (dependency / "core.cpp").write_text("int pcre_fixture() { return 0; }\n")
        (dependency / "posix.cpp").write_text("int unused_posix_fixture() { return 0; }\n")
        (dependency / "pcre2.h").write_text("int pcre_fixture();\n")
        (dependency / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.25)\nproject(pcre2 LANGUAGES CXX)\n"
            "add_library(pcre2-8-static STATIC core.cpp)\n"
            "add_library(pcre2-posix-static STATIC posix.cpp)\n"
            "install(TARGETS pcre2-8-static pcre2-posix-static ARCHIVE DESTINATION lib)\n"
            "install(FILES pcre2.h DESTINATION include)\n")
        (source / "main.cpp").write_text("int pcre_fixture(); int main() { return pcre_fixture(); }\n")
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.25)\nproject(release_fixture LANGUAGES CXX)\n"
            # Empty defined results force the bundled branch even if the host
            # has development headers; FetchContent uses our local fixture.
            'set(PCRE2_INCLUDE_DIR "")\nset(PCRE2_LIBRARY "")\n'
            f'set(FETCHCONTENT_SOURCE_DIR_PCRE2 "{dependency.as_posix()}")\n'
            f'include("{(ROOT / "cmake/pcre2.cmake").as_posix()}")\n'
            "add_executable(consumer main.cpp)\n"
            "target_link_libraries(consumer PRIVATE dgpp_pcre2)\n"
            "install(TARGETS consumer RUNTIME DESTINATION bin)\n")
        build, stage = self.root / "release-build", self.root / "release-stage"
        commands = (["cmake", "-S", str(source), "-B", str(build)],
                    ["cmake", "--build", str(build), "--target", "consumer"],
                    ["cmake", "--install", str(build), "--prefix", str(stage)])
        for command in commands:
            result = subprocess.run(command, env=self.env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(list(build.rglob("*posix*.a")), "unused POSIX library must not be required")
        self.assertEqual({p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()},
                         {"bin/consumer"}, "dependency development files must not enter the release")
        subprocess.run([str(stage / "bin/consumer")], check=True, timeout=10)

    def test_discovery_selection_matches_device_order_and_overrides(self):
        rows = [{"device": device, "port": port, "active": active}
                for device, port, active in (("z", "1", True), ("a", "1", True),
                                            ("b", "2", True), ("c", "1", False))]
        self.assertEqual(discover_roce.selection(rows, {}),
                         {"mode": "automatic", "devices": ["a", "z"], "gid_indices": None})
        self.assertEqual(discover_roce.selection(rows, {"DGPP_ROCE_DEVICES": "z a", "DGPP_ROCE_GID_INDICES": "3 4"}),
                         {"mode": "configured", "devices": ["z", "a"], "gid_indices": ["3", "4"]})

    def test_discovery_preserves_partial_results_and_checks_later_peers(self):
        rows = [{"device": "nic0", "port": "1", "active": True, "usable": True,
                 "gids": [{"interface": "eth0", "ips": ["192.0.2.1/24"], "address": "192.0.2.1", "mtu": 9000, "index": 3}]}]
        for json_mode in (False, True):
            for error in (Mock(returncode=255, stderr="Host key verification failed"),
                          subprocess.TimeoutExpired("ssh", 30)):
                with self.subTest(json_mode=json_mode, error=error):
                    output = io.StringIO()
                    with patch.dict(os.environ, self.env, clear=True), contextlib.redirect_stdout(output), \
                            patch.object(cluster_doctor, "roce_inventory", return_value=rows), \
                            patch.object(discover_roce.subprocess, "run", side_effect=[error,
                                         Mock(returncode=0, stdout=json.dumps(rows)),
                                         Mock(returncode=0, stdout="[]")]) as remote:
                        result = discover_roce.main(["--config", str(self.config), *(["--json"] if json_mode else [])])
                    self.assertEqual(result, 1)
                    self.assertEqual(remote.call_count, 3)
                    if json_mode:
                        data = json.loads(output.getvalue())
                        self.assertEqual(set(data["inventories"]), {"127.0.0.1", "peer2", "peer3"})
                        self.assertEqual(set(data["errors"]), {"peer1"})
                        self.assertNotIn("peer1", data["selections"])
                    else:
                        self.assertIn("locally eligible", output.getvalue())
                        self.assertIn("Deployment lane order (automatic): nic0", output.getvalue())
                        self.assertIn("FAIL peer1: inventory unknown", output.getvalue())
                        self.assertIn("peer2: verbs device", output.getvalue())

    def test_local_discovery_success_and_single_node_selection(self):
        with patch.object(cluster_doctor, "roce_inventory", return_value=[]), \
                patch.object(discover_roce.subprocess, "run") as remote:
            self.assertEqual(discover_roce.main([]), 0)
        remote.assert_not_called()
        self.config.write_text(json.dumps({"model": "org/model", "world_size": 1}))
        with patch.dict(os.environ, self.env, clear=True), patch.object(cluster_doctor, "roce_inventory", return_value=[]):
            self.assertEqual(discover_roce.main(["--config", str(self.config)]), 0)
        self.assertIn("not needed (single node)", self.output.getvalue())

    def test_doctor_does_not_invent_lane_counts_for_failed_probes(self):
        with patch.dict(os.environ, self.env, clear=True):
            cfg = site_env.resolve_config(self.config)
        report = {"rank": 0, "devices": ["nic0", "nic1"], "checks": []}
        with patch.object(cluster_doctor, "probe", return_value=report), \
                patch.object(cluster_doctor.subprocess, "run", return_value=Mock(returncode=255, stderr="Host key verification failed")):
            self.assertEqual(cluster_doctor.check_cluster(cfg, None, "/log", "/stage", "tester"), 1)
        output = self.output.getvalue()
        self.assertNotIn("lane counts differ", output)
        self.assertIn("lane counts are unknown", output)
        self.assertIn("Preflight: 3 failed check(s)", output)

    def test_real_lane_mismatch_still_prints_summary(self):
        with patch.dict(os.environ, self.env, clear=True):
            cfg = site_env.resolve_config(self.config)
        report = {"rank": 0, "devices": ["nic0", "nic1"], "checks": []}
        peer = {"rank": 1, "devices": ["nic0"], "checks": []}
        with patch.object(cluster_doctor, "probe", return_value=report), \
                patch.object(cluster_doctor.subprocess, "run", return_value=Mock(returncode=0, stdout=json.dumps(peer))):
            self.assertEqual(cluster_doctor.check_cluster(cfg, None, "/log", "/stage", "tester"), 1)
        output = self.output.getvalue()
        self.assertIn("lane counts differ between successfully inspected nodes", output)
        self.assertIn("Preflight: 1 failed check(s)", output)

    def test_missing_checkpoint_recovery_uses_head_download_and_sync(self):
        spec = {"rank": 0, "nodes": ["127.0.0.1"], "model": "org/missing", "env": {"HF_HUB_CACHE": str(self.cache)},
                "paths": {}, "http_bind": "127.0.0.1", "ports": {"http": 0}, "binary": None}
        with patch.object(cluster_doctor, "command", return_value=Mock(returncode=0, stdout="")):
            report = cluster_doctor.probe(spec)
        check = next(check for check in report["checks"] if check["check"] == "checkpoint")
        self.assertEqual(check["status"], "fail")
        self.assertIn("on rank 0 run python3 scripts/download_model.py --config FILE", check["detail"])
        self.assertIn("--sync-only", check["detail"])
        self.assertNotIn("download the complete checkpoint on this node", check["detail"])


if __name__ == "__main__":
    unittest.main()
