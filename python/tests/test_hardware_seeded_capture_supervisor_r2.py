# SPDX-License-Identifier: AGPL-3.0-only
"""Failure-path tests for the offline smoke supervisor; no host policy writes."""

from argparse import Namespace
import base64
import hashlib
from pathlib import Path
import json
import os
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

from cpu2tensor.examples import hardware_seeded_capture_supervisor_r2 as guard


class HardwareSeededCaptureSupervisorR2Tests(unittest.TestCase):
    def test_unit_journald_does_not_populate_required_empty_spool(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        unit = (repo / "ops/systemd/cpu2tensor-seeded-smoke-r2.service.in").read_text()
        self.assertIn("StandardOutput=journal\n", unit)
        self.assertIn("StandardError=journal\n", unit)
        self.assertNotIn("StandardOutput=append:", unit)
        self.assertNotIn("/supervisor.log", unit)
        self.assertIn("LogRateLimitBurst=100\n", unit)

    def test_manual_invocation_and_missing_restore_hook_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cgroup = Path(directory) / "cgroup"
            cgroup.write_text("0::/system.slice/cpu2tensor-seeded-smoke-r2.service\n")
            with mock.patch.object(guard, "CGROUP", cgroup), \
                    mock.patch.dict(guard.os.environ, {"INVOCATION_ID": ""}):
                with self.assertRaisesRegex(ValueError, "reviewed systemd unit"):
                    guard._require_service_context()
            active = mock.Mock(returncode=0)
            wrong = mock.Mock(stdout="ExecStopPost=\nRuntimeMaxUSec=20min\nKillMode=control-group\n")
            with mock.patch.object(guard, "CGROUP", cgroup), \
                    mock.patch.dict(guard.os.environ, {"INVOCATION_ID": "unit-id"}), \
                    mock.patch.object(guard.subprocess, "run", side_effect=[active, wrong]):
                with self.assertRaisesRegex(ValueError, "restore hook"):
                    guard._require_service_context()

    def test_memory_headroom_is_bounded_by_system_and_cgroup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meminfo = root / "meminfo"
            cgroup = root / "cgroup"
            accounting = root / "system.slice/unit"
            accounting.mkdir(parents=True)
            meminfo.write_text("MemAvailable: 5242880 kB\n")
            cgroup.write_text("0::/system.slice/unit\n")
            (accounting / "memory.max").write_text(str(4 * guard.GIB))
            (accounting / "memory.current").write_text(str(guard.GIB))
            with mock.patch.object(guard, "MEMINFO", meminfo), \
                    mock.patch.object(guard, "CGROUP", cgroup), \
                    mock.patch.object(guard, "CGROUP_ROOT", root):
                guard._check_memory(start=True)
                (accounting / "memory.current").write_text(str(3 * guard.GIB))
                with self.assertRaisesRegex(RuntimeError, "cgroup memory"):
                    guard._check_memory(start=True)
                (accounting / "memory.current").write_text(str(guard.GIB))
                meminfo.write_text("MemAvailable: 1048576 kB\n")
                with self.assertRaisesRegex(RuntimeError, "MemAvailable"):
                    guard._check_memory(start=False)
                meminfo.write_text("MemAvailable: 5242880 kB\n")
                legacy = root / "memory/system.slice/unit"
                legacy.mkdir(parents=True)
                (legacy / "memory.limit_in_bytes").write_text(str(4 * guard.GIB))
                (legacy / "memory.usage_in_bytes").write_text(str(guard.GIB))
                cgroup.write_text("10:memory:/system.slice/unit\n")
                guard._check_memory(start=True)

    def test_package_sensor_must_match_cpu2_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "name").write_text("coretemp\n")
            (root / "temp1_label").write_text("Package id 0\n")
            sensor = root / "temp1_input"
            sensor.write_text("65000\n")
            package = root / "physical_package_id"
            package.write_text("0\n")
            with mock.patch.object(guard, "CPU2_PACKAGE", package):
                self.assertEqual(guard.package_temperature_millic(sensor), 65_000)
                package.write_text("1\n")
                with self.assertRaisesRegex(ValueError, "package sensor"):
                    guard.package_temperature_millic(sensor)

    def test_sealed_files_require_group_read_without_group_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spool = Path(directory)
            shard_dir = spool / "raw"
            shard_dir.mkdir(mode=0o750)
            shard = shard_dir / "shard.pt"
            shard.write_bytes(b"sealed")
            shard.chmod(0o640)
            guard._verify_mac_readability(spool, owner_uid=os.getuid(), user_gid=os.getgid())
            shard.chmod(0o600)
            with self.assertRaisesRegex(RuntimeError, "read-only transferable"):
                guard._verify_mac_readability(spool, owner_uid=os.getuid(), user_gid=os.getgid())

    def test_portable_source_bundle_detects_change_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            package = repo / "python/cpu2tensor"
            native = repo / "native"
            package.mkdir(parents=True)
            native.mkdir()
            (repo / "pyproject.toml").write_text("[build-system]\n")
            (repo / "AGENTS.md").write_text("instructions\n")
            module = package / "module.py"
            module.write_text("value = 1\n")
            (native / "target.c").write_text("int value = 1;\n")
            first = guard.source_bundle_manifest(repo)
            self.assertEqual(first, guard.source_bundle_manifest(repo))
            module.write_text("value = 2\n")
            self.assertNotEqual(first["digest_sha256"],
                                guard.source_bundle_manifest(repo)["digest_sha256"])
            (native / "link.c").symlink_to(native / "target.c")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                guard.source_bundle_manifest(repo)

    def test_post_capture_binary_rehash_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "binary"
            binary.write_bytes(b"reviewed")
            repo = Path(guard.__file__).resolve().parents[3]
            args = Namespace(
                binary=binary, binary_sha256=guard.sha256(binary),
                runner_sha256=guard.sha256(Path(guard.collector.__file__)),
                planner_sha256=guard.sha256(Path(guard.plans.__file__)),
                supervisor_sha256=guard.sha256(Path(guard.__file__)),
                source_bundle_sha256=guard.source_bundle_manifest(repo)["digest_sha256"],
            )
            guard._verify_frozen_sources(args, repo)
            binary.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "binary or source changed"):
                guard._verify_frozen_sources(args, repo)

    def test_policy_restore_is_idempotent_and_preserves_bad_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            no_turbo = root / "no_turbo"
            maximum = root / "max_freq"
            state = root / "policy.json"
            no_turbo.write_text("1\n")
            maximum.write_text("1800000\n")
            state.write_text(json.dumps({"no_turbo": "0", "maximum_khz": "4900000"}))
            with mock.patch.object(guard, "NO_TURBO", no_turbo), \
                    mock.patch.object(guard, "MAX_FREQ", maximum):
                guard.restore_policy(state)
                self.assertEqual(no_turbo.read_text().strip(), "0")
                self.assertEqual(maximum.read_text().strip(), "4900000")
                self.assertFalse(state.exists())
                guard.restore_policy(state)
                state.write_text('{"no_turbo":"bad","maximum_khz":"4900000"}')
                with self.assertRaisesRegex(ValueError, "invalid saved policy"):
                    guard.restore_policy(state)
                self.assertTrue(state.exists())

    def test_timer_restore_skips_active_and_deactivating_service(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "policy.json"
            state.write_text('{"no_turbo":"0","maximum_khz":"4900000"}')
            with mock.patch.object(guard, "POLICY_STATE", state), \
                    mock.patch.object(guard, "restore_policy") as restore, \
                    mock.patch.object(guard.subprocess, "run") as run:
                for service_state in ("active", "activating", "deactivating"):
                    run.return_value = mock.Mock(stdout=service_state + "\n")
                    guard.restore_if_inactive()
                restore.assert_not_called()
                run.return_value = mock.Mock(stdout="failed\n")
                guard.restore_if_inactive()
                restore.assert_called_once_with(state)

    def test_process_group_receives_term_kill_and_is_reaped(self) -> None:
        process = mock.Mock(pid=1234)
        process.wait.side_effect = [subprocess.TimeoutExpired("collector", 5), 0]
        with mock.patch.object(guard.os, "killpg") as killpg:
            guard.terminate_group(process)
        self.assertEqual(killpg.call_args_list, [
            mock.call(1234, signal.SIGTERM), mock.call(1234, signal.SIGKILL),
        ])
        self.assertEqual(process.wait.call_count, 2)

    def test_unit_rehearsal_changes_and_restores_policy_without_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(spool=root)
            process = mock.Mock(pid=1234)
            values = {guard.NO_TURBO: "0", guard.MAX_FREQ: "4900000"}
            with mock.patch.dict(guard.os.environ, {"C2T_SEEDED_REHEARSAL": "1"}), \
                    mock.patch.object(guard, "preflight", return_value=("0", "4900000")), \
                    mock.patch.object(guard, "_seal_policy_state"), \
                    mock.patch.object(guard, "_write", side_effect=lambda path, value: values.__setitem__(path, value)), \
                    mock.patch.object(guard, "_read", side_effect=lambda path: values[path]), \
                    mock.patch.object(guard.subprocess, "Popen", return_value=process) as popen, \
                    mock.patch.object(guard, "source_bundle_manifest") as bundle, \
                    mock.patch.object(guard.collector, "subject_manifest") as subject, \
                    mock.patch.object(guard, "terminate_group") as terminate, \
                    mock.patch.object(guard, "restore_policy") as restore:
                guard.run_smoke(args)
            popen.assert_called_once_with(("/bin/sleep", "30"), start_new_session=True)
            process.wait.assert_called_once_with(timeout=35)
            bundle.assert_not_called()
            subject.assert_not_called()
            terminate.assert_called_once_with(process)
            restore.assert_called_once_with(guard.POLICY_STATE)

    def test_root_child_python_must_resolve_frozen_checkout(self) -> None:
        repo = Path(guard.__file__).resolve().parents[3]
        paths = {
            "package": "/another-checkout/python/cpu2tensor/__init__.py",
            "torch": str(Path(guard.__file__)),
            "weights_only": True,
            "runner": str(repo / "python/cpu2tensor/examples/hardware_multimodal_experiment.py"),
            "planner": str(repo / "python/cpu2tensor/examples/hardware_seeded_capture_plan_r2.py"),
            "guard": str(Path(guard.__file__)),
        }
        with mock.patch.object(guard.subprocess, "run", return_value=mock.Mock(stdout=json.dumps(paths))):
            with self.assertRaisesRegex(ValueError, "another cpu2tensor checkout"):
                guard._check_child_resolution(repo)
        paths["package"] = str(repo / "python/cpu2tensor/__init__.py")
        with mock.patch.object(guard.subprocess, "run", return_value=mock.Mock(stdout=json.dumps(paths))):
            guard._check_child_resolution(repo)

    def test_local_smoke_hash_and_completeness_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            plan = guard.plans.make_plan("smoke", "a" * 64, 71)
            sealed_plan = artifact / "execution-plan.json"
            sealed_plan.write_text(json.dumps(plan))
            (artifact / "raw").mkdir()
            (artifact / "derived").mkdir()
            entries = []
            binary = "/approved/hardware_kernel_workload"
            for row in plan["rows"]:
                execution_id = row["execution_id"]
                execution = {key: row[key] for key in (
                    "execution_id", "family", "repetition", "partition",
                )}
                frame = guard.collector.input_frame(row["input_seed"])
                frame_hash = hashlib.sha256(frame).hexdigest()
                argv = [binary, row["family"], str(row["loops"])]
                stdout = b"42\n"
                raw = {
                    "schema": guard.collector.RAW_SCHEMA,
                    "execution": execution, "loops": row["loops"],
                    "invocation": {
                        "argv": argv, "input_seed": row["input_seed"],
                        "stdin_sha256": frame_hash,
                        "stdin": guard.torch.tensor(list(frame), dtype=guard.torch.uint8),
                    },
                    "stdout": guard.torch.tensor(list(stdout), dtype=guard.torch.uint8),
                    "batches": [{
                        "source": 123, "tid": 123, "cpu": 2,
                        "status": [{"signal": signal, "requested": True,
                                    "available": True, "lost": False,
                                    "time_enabled_ns": 100 if signal != "intel_pt" else None,
                                    "time_running_ns": 100 if signal != "intel_pt" else None}
                                   for signal in ("intel_pt", "memory_loads", "counters")],
                        "pt": {"signal": "intel_pt", "trace_bytes": guard.torch.tensor([1], dtype=guard.torch.uint8)},
                        "pebs": {"signal": "memory_loads", "ip": guard.torch.tensor([1]),
                                 "address": guard.torch.tensor([1]), "exact_ip": guard.torch.tensor([True]),
                                 "cpu": guard.torch.tensor([2])},
                        "counters": {"available": True, "lost": False,
                                     "time_enabled_ns": 100, "time_running_ns": 100},
                    }],
                }
                paths = {}
                raw_relative = Path("raw") / f"{execution_id}.pt"
                guard.torch.save(raw, artifact / raw_relative)
                raw_hash = guard.sha256(artifact / raw_relative)
                derived_relative = Path("derived") / f"{execution_id}.pt"
                guard.torch.save({
                    "schema": guard.collector.DERIVED_SCHEMA,
                    "execution": execution, "raw_sha256": raw_hash,
                    "batch": {f"{name}_available": guard.torch.tensor([True])
                              for name in ("pt", "pebs", "pmu")},
                }, artifact / derived_relative)
                paths.update({
                    "raw_path": str(raw_relative), "raw_sha256": raw_hash,
                    "derived_path": str(derived_relative),
                    "derived_sha256": guard.sha256(artifact / derived_relative),
                })
                entries.append({
                    **execution, "loops": row["loops"], "raw_retained": True,
                    "admission": {"attempt": 1},
                    "invocation": {"argv": argv, "input_seed": row["input_seed"],
                                   "stdin_sha256": frame_hash},
                    "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                    "stdout_base64": base64.b64encode(stdout).decode("ascii"),
                    "capture": {"pt_bytes": 1, "pebs_samples": 1,
                                "pebs_usable_samples": 1, "pebs_censored_samples": 0,
                                "pebs_exact_ip": 1, "pebs_nonzero_address": 1,
                                "lost_sources": 0, "missing_sources": 0,
                                "multiplexed_sources": 0},
                    **paths,
                })
            manifest = {
                "identity_sha256": "a" * 64,
                "build": {"workload_path": binary},
                "split": {"explicit_plan": {
                    "kind": "smoke", "sha256": guard.sha256(sealed_plan),
                }},
                "collection": {
                    "executions": 51, "total_pt_bytes": 51,
                    "total_pebs_samples": 51, "total_pebs_usable_samples": 51,
                    "total_pebs_censored_samples": 0,
                    "loss_count": 0,
                    "admission": {"rejected_attempts": 0, "total_attempts": 51},
                    "missing_modality_executions": {"pt": 0, "pebs": 0, "pmu": 0},
                },
                "entries": entries,
            }
            def seal_manifest() -> None:
                manifest["manifest_content_sha256"] = guard.collector._json_hash({
                    key: value for key, value in manifest.items()
                    if key != "manifest_content_sha256"
                })
                (artifact / "capture-manifest.json").write_text(json.dumps(manifest))

            seal_manifest()
            guard.verify_smoke_artifact(artifact, "a" * 64, plan)
            retained_schema_raw = guard.torch.load(artifact / entries[0]["raw_path"], weights_only=True)
            retained_schema_derived = guard.torch.load(artifact / entries[0]["derived_path"], weights_only=True)
            retained_schema_raw["batches"][0]["pebs"]["signal"] = "memory_stores"
            retained_schema_raw["batches"][0]["status"][1]["signal"] = "memory_stores"
            self.assertEqual(
                guard._verified_sensor_counts(retained_schema_raw, retained_schema_derived,
                                              pebs_signal="memory_stores"), entries[0]["capture"],
            )
            with self.assertRaisesRegex(RuntimeError, "status contract"):
                guard._verified_sensor_counts(retained_schema_raw, retained_schema_derived)
            first = entries[0]
            raw = guard.torch.load(artifact / first["raw_path"], weights_only=True)
            raw["batches"][0]["status"][1]["lost"] = True
            guard.torch.save(raw, artifact / first["raw_path"])
            first["raw_sha256"] = guard.sha256(artifact / first["raw_path"])
            derived = guard.torch.load(artifact / first["derived_path"], weights_only=True)
            derived["raw_sha256"] = first["raw_sha256"]
            guard.torch.save(derived, artifact / first["derived_path"])
            first["derived_sha256"] = guard.sha256(artifact / first["derived_path"])
            seal_manifest()
            with self.assertRaisesRegex(RuntimeError, "raw sensor reports loss"):
                guard.verify_smoke_artifact(artifact, "a" * 64, plan)
            raw["batches"][0]["status"][1]["lost"] = False
            guard.torch.save(raw, artifact / first["raw_path"])
            first["raw_sha256"] = guard.sha256(artifact / first["raw_path"])
            derived["raw_sha256"] = first["raw_sha256"]
            guard.torch.save(derived, artifact / first["derived_path"])
            first["derived_sha256"] = guard.sha256(artifact / first["derived_path"])
            seal_manifest()
            guard.verify_smoke_artifact(artifact, "a" * 64, plan)
            derived["batch"]["pebs_available"] = guard.torch.tensor([False])
            guard.torch.save(derived, artifact / first["derived_path"])
            first["derived_sha256"] = guard.sha256(artifact / first["derived_path"])
            seal_manifest()
            with self.assertRaisesRegex(RuntimeError, "derived sensor modality"):
                guard.verify_smoke_artifact(artifact, "a" * 64, plan)
            derived["batch"]["pebs_available"] = guard.torch.tensor([True])
            guard.torch.save(derived, artifact / first["derived_path"])
            first["derived_sha256"] = guard.sha256(artifact / first["derived_path"])
            manifest["collection"]["total_pt_bytes"] = 50
            seal_manifest()
            with self.assertRaisesRegex(RuntimeError, "aggregate differs"):
                guard.verify_smoke_artifact(artifact, "a" * 64, plan)
            manifest["collection"]["total_pt_bytes"] = 51
            seal_manifest()
            original_raw = (artifact / entries[0]["raw_path"]).read_bytes()
            (artifact / entries[0]["raw_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                guard.verify_smoke_artifact(artifact, "a" * 64, plan)
            (artifact / entries[0]["raw_path"]).write_bytes(original_raw)
            first = entries[0]
            first["loops"] += 1
            seal_manifest()
            with self.assertRaisesRegex(RuntimeError, "exact execution plan"):
                guard.verify_smoke_artifact(artifact, "a" * 64, plan)
            first["loops"] -= 1
            seal_manifest()
            planned = next(row for row in plan["rows"]
                           if row["execution_id"] == first["execution_id"])
            changed_frame = b"s" + (planned["input_seed"] ^ 1).to_bytes(8, "little")
            raw = guard.torch.load(artifact / first["raw_path"], weights_only=True)
            raw["invocation"]["stdin"] = guard.torch.tensor(
                list(changed_frame), dtype=guard.torch.uint8,
            )
            guard.torch.save(raw, artifact / first["raw_path"])
            first["raw_sha256"] = guard.sha256(artifact / first["raw_path"])
            derived = guard.torch.load(artifact / first["derived_path"], weights_only=True)
            derived["raw_sha256"] = first["raw_sha256"]
            guard.torch.save(derived, artifact / first["derived_path"])
            first["derived_sha256"] = guard.sha256(artifact / first["derived_path"])
            seal_manifest()
            with self.assertRaisesRegex(RuntimeError, "raw smoke payload"):
                guard.verify_smoke_artifact(artifact, "a" * 64, plan)
            manifest["manifest_content_sha256"] = "0" * 64
            (artifact / "capture-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "manifest content hash"):
                guard.verify_smoke_artifact(artifact, "a" * 64, plan)

    def test_preflight_rejects_wrong_mount_hot_package_and_changed_boot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool = root / "spool"
            spool.mkdir()
            binary = root / "binary"
            binary.write_bytes(b"target")
            no_turbo = root / "no_turbo"
            no_turbo.write_text("0\n")
            max_freq = root / "max_freq"
            max_freq.write_text("4900000\n")
            boot = root / "boot"
            boot.write_text(guard.EXPECTED_BOOT)
            args = Namespace(
                source_root=Path(guard.__file__).resolve().parents[3],
                binary=binary, binary_sha256=guard.sha256(binary),
                runner_sha256=guard.sha256(Path(guard.collector.__file__)),
                planner_sha256=guard.sha256(Path(guard.plans.__file__)),
                supervisor_sha256=guard.sha256(Path(guard.__file__)),
                source_bundle_sha256=guard.source_bundle_manifest(
                    Path(guard.__file__).resolve().parents[3]
                )["digest_sha256"],
                cohort_seed=71,
                spool=spool, temperature_input=root / "temperature",
                state_path=root / "policy.json",
            )
            stat = mock.Mock(f_blocks=896, f_frsize=guard.MIB)
            with mock.patch.object(guard.os, "geteuid", return_value=0), \
                    mock.patch.object(guard.os.path, "ismount", return_value=True), \
                    mock.patch.object(guard.os, "statvfs", return_value=stat), \
                    mock.patch.object(guard.shutil, "disk_usage", return_value=mock.Mock(free=7 * guard.GIB)), \
                    mock.patch.object(guard, "NO_TURBO", no_turbo), \
                    mock.patch.object(guard, "MAX_FREQ", max_freq), \
                    mock.patch.object(guard, "BOOT_ID", boot), \
                    mock.patch.object(guard.platform, "system", return_value="Linux"), \
                    mock.patch.object(guard.platform, "release", return_value=guard.EXPECTED_KERNEL), \
                    mock.patch.object(guard, "_check_child_resolution"), \
                    mock.patch.object(guard, "_check_spool_access"), \
                    mock.patch.object(guard, "_require_service_context"), \
                    mock.patch.object(guard, "_check_memory"), \
                    mock.patch.object(guard, "POLICY_STATE", root / "policy.json"), \
                    mock.patch.object(guard, "CPU2_PACKAGE", root / "physical_package_id"), \
                    mock.patch.object(guard, "package_temperature_millic", return_value=60_000) as temp, \
                    mock.patch.object(guard, "_mount_fields", return_value=("tmpfs", "tmpfs", str(spool.resolve()))) as mount:
                self.assertEqual(guard.preflight(args), ("0", "4900000"))
                args.runner_sha256 = "b" * 64
                with self.assertRaisesRegex(ValueError, "source changed"):
                    guard.preflight(args)
                args.runner_sha256 = guard.sha256(Path(guard.collector.__file__))
                mount.return_value = ("/dev/sda1", "ext4", str(spool.resolve()))
                with self.assertRaisesRegex(ValueError, "tmpfs"):
                    guard.preflight(args)
                mount.return_value = ("tmpfs", "tmpfs", str(spool.resolve()))
                temp.return_value = 80_000
                with self.assertRaisesRegex(ValueError, "too hot"):
                    guard.preflight(args)
                temp.return_value = 60_000
                boot.write_text("changed")
                with self.assertRaisesRegex(ValueError, "boot or kernel"):
                    guard.preflight(args)

    def test_limit_aborts_still_kill_group_and_restore_policy(self) -> None:
        for reason, message in (
            ("temperature", "80 C"), ("timeout", "20 minutes"),
            ("root", "5 GiB"), ("tmpfs", "128 MiB"),
        ):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                spool = Path(directory)
                args = Namespace(
                    spool=spool, binary=spool / "binary", cohort_seed=71,
                    temperature_input=spool / "temperature",
                    state_path=spool / "policy.json",
                    source_bundle_sha256="a" * 64,
                )
                process = mock.Mock(pid=1234)
                process.poll.return_value = None
                values = {guard.NO_TURBO: "0", guard.MAX_FREQ: "4900000"}

                def write(path: Path, value: str) -> None:
                    values[path] = value

                stat = mock.Mock(f_bavail=1 if reason == "tmpfs" else 100,
                                 f_frsize=guard.MIB)
                free = [mock.Mock(free=6 * guard.GIB),
                        mock.Mock(free=4 * guard.GIB if reason == "root" else 6 * guard.GIB)]
                clock = (0, guard.MAX_SECONDS) if reason == "timeout" else (0, 1)
                temperatures = (60_000, 80_000 if reason == "temperature" else 60_000)
                with mock.patch.object(guard, "preflight", return_value=("0", "4900000")), \
                        mock.patch.object(guard, "_seal_policy_state"), \
                        mock.patch.object(guard, "_write", side_effect=write), \
                        mock.patch.object(guard, "_read", side_effect=lambda path: values[path]), \
                        mock.patch.object(guard, "source_bundle_manifest", return_value={"digest_sha256": "a" * 64}), \
                        mock.patch.object(guard, "_check_memory"), \
                        mock.patch.object(guard.collector, "subject_manifest", return_value={"identity_sha256": "a" * 64}), \
                        mock.patch.object(guard.subprocess, "Popen", return_value=process), \
                        mock.patch.object(guard, "package_temperature_millic", side_effect=temperatures), \
                        mock.patch.object(guard.shutil, "disk_usage", side_effect=free), \
                        mock.patch.object(guard.os, "statvfs", return_value=stat), \
                        mock.patch.object(guard.time, "monotonic", side_effect=clock), \
                        mock.patch.object(guard, "terminate_group") as terminate, \
                        mock.patch.object(guard, "restore_policy") as restore:
                    with self.assertRaisesRegex((RuntimeError, TimeoutError), message):
                        guard.run_smoke(args)
                terminate.assert_called_once_with(process)
                restore.assert_called_once_with(guard.POLICY_STATE)


if __name__ == "__main__":
    unittest.main()
