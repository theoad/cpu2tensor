# SPDX-License-Identifier: AGPL-3.0-only

import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

import torch

from cpu2tensor.examples.hardware_foundation_corpus import (
    PLAN_SCHEMA,
    TarShardWriter,
    WorkloadRow,
    _validate_capture,
    load_plan,
    seal_custody,
)
from cpu2tensor.examples.hardware_foundation_audit import audit
from cpu2tensor.examples.hardware_foundation_plan import _run_twice
from cpu2tensor.hardware import (
    HardwareBatch, HardwareCaptureEnvelope, HardwareCounterBatch,
    HardwareMultimodalBatch, HardwareSourceStatus,
)


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_plan_preflight_never_inherits_interactive_stdin() -> None:
    with tempfile.TemporaryDirectory() as directory:
        stdout, stderr = _run_twice(
            (sys.executable, "-c", "import sys; print(len(sys.stdin.read()))"),
            cwd=Path(directory), environment=dict(os.environ), exit_code=0,
        )
    assert stdout == _hash(b"0\n")
    assert stderr == _hash(b"")


def test_plan_requires_whole_application_holdout_and_exact_inputs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binary = root / "target"
        input_path = root / "input"
        binary.write_bytes(b"binary")
        input_path.write_bytes(b"input")
        row = {
            "execution_id": "gzip-0001",
            "family": "compression",
            "application": "gzip",
            "partition": "training",
            "session": "session-a",
            "matched_input_partition": None,
            "input_seed": 17,
            "argv": [str(binary), "-c", str(input_path)],
            "stdin_base64": base64.b64encode(b"").decode(),
            "cwd": str(root),
            "input_paths": [str(input_path)],
            "input_sha256": [_hash(b"input")],
            "support_paths": [],
            "expected_exit_code": 0,
            "expected_stdout_sha256": _hash(b"output"),
            "expected_stderr_sha256": _hash(b""),
            "timeout_seconds": 2,
        }
        plan = {
            "schema": PLAN_SCHEMA,
            "corpus_id": "pilot",
            "environment": {"LANG": "C"},
            "rows": [row],
        }
        path = root / "plan.json"
        path.write_text(json.dumps(plan))
        _, rows = load_plan(path)
        assert rows[0].application == "gzip"
        plan["rows"].append({
            **row,
            "execution_id": "gzip-0002",
            "partition": "heldout_application",
            "session": "session-b",
            "matched_input_partition": "heldout_application",
        })
        path.write_text(json.dumps(plan))
        try:
            load_plan(path)
        except ValueError as error:
            assert "leak" in str(error)
        else:
            raise AssertionError("application leakage was accepted")


def test_plan_rejects_input_content_leakage_under_different_paths() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binary = root / "target"
        training_input = root / "train-input"
        calibration_input = root / "calibration-input"
        binary.write_bytes(b"binary")
        training_input.write_bytes(b"identical-content")
        calibration_input.write_bytes(b"identical-content")

        def row(execution_id: str, partition: str, path: Path) -> dict[str, object]:
            return {
                "execution_id": execution_id,
                "family": "compression",
                "application": "gzip",
                "partition": partition,
                "session": "session-a",
                "matched_input_partition": None,
                "input_seed": 17,
                "argv": [str(binary), str(path)],
                "stdin_base64": "",
                "cwd": str(root),
                "input_paths": [str(path)],
                "input_sha256": [_hash(b"identical-content")],
                "support_paths": [],
                "expected_exit_code": 0,
                "expected_stdout_sha256": _hash(b""),
                "expected_stderr_sha256": _hash(b""),
                "timeout_seconds": 2,
            }

        plan = {
            "schema": PLAN_SCHEMA,
            "corpus_id": "leak-test",
            "environment": {"LANG": "C"},
            "rows": [
                row("gzip-train", "training", training_input),
                row("gzip-calibration", "calibration", calibration_input),
            ],
        }
        path = root / "plan.json"
        path.write_text(json.dumps(plan))
        try:
            load_plan(path)
        except ValueError as error:
            assert "input-content leakage" in str(error)
        else:
            raise AssertionError("content-identical split inputs were accepted")


def test_release_audit_requires_complete_exact_sessions() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binary = root / "target"
        binary.write_bytes(b"binary")

        def row(execution_id: str, session: str) -> dict[str, object]:
            heldout = session == "session-b"
            return {
                "execution_id": execution_id,
                "family": "primitive",
                "application": "fixture",
                "partition": "heldout_session" if heldout else "training",
                "session": session,
                "matched_input_partition": "familiar_validation" if heldout else None,
                "input_seed": 1,
                "argv": [str(binary)],
                "stdin_base64": "",
                "cwd": str(root),
                "input_paths": [],
                "input_sha256": [],
                "support_paths": [],
                "expected_exit_code": 0,
                "expected_stdout_sha256": _hash(b""),
                "expected_stderr_sha256": _hash(b""),
                "timeout_seconds": 1,
            }

        plan = {
            "schema": PLAN_SCHEMA, "corpus_id": "audit-fixture",
            "environment": {"LANG": "C"},
            "rows": [row("fixture-a", "session-a"), row("fixture-b", "session-b")],
        }
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        plan_digest = _hash(plan_path.read_bytes())
        releases = []
        for session, execution_id in (
            ("session-a", "fixture-a"), ("session-b", "fixture-b"),
        ):
            entry = {
                "execution_id": execution_id, "family": "primitive",
                "application": "fixture",
                "partition": "training" if session == "session-a" else "heldout_session",
                "session": session, "pt_bytes": 1, "pebs_samples": 0,
                "instructions": 1, "cycles": 1, "ref_cycles": 1,
                "pebs_user_samples": 0, "pebs_kernel_samples": 0,
                "pebs_usable_samples": 0, "pebs_inexact_samples": 0,
                "pebs_zero_address_samples": 0,
            }
            manifest = {
                "plan_sha256": plan_digest, "rejections": [],
                "stopped_at_byte_target": False, "executions": 1,
                "planned_executions": 1,
                "subject": {
                    "identity_sha256": "a" * 64,
                    "subject": {
                        "scope": "process_user_kernel",
                        "modalities": ["intel_pt", "memory_loads", "counters"],
                    },
                },
                "entries": [entry], "shards": [],
            }
            release = {
                "schema": "cpu2tensor-hardware-foundation-release-v1",
                "corpus_id": "audit-fixture", "session": session,
                "capture_manifest": manifest, "shards": [],
            }
            path = root / f"{session}.json"
            path.write_text(json.dumps(release))
            releases.append(path)
        binary.unlink()  # The independent auditor runs away from collector paths.
        report = audit(
            plan_path, releases, minimum_bytes=0,
            minimum_application_pt_bytes=0, bucket=None, region="us-east-1",
            enforce_quality=False,
        )
        assert report["accepted"] is True
        assert report["executions"] == 2


def test_tar_shards_are_content_addressed_and_range_indexed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        writer = TarShardWriter(root, target_bytes=1024)
        raw = root / "row.pt"
        torch.save({"value": torch.arange(32)}, raw)
        writer.add(raw, {
            "execution_id": "row-0001",
            "family": "primitive",
            "application": "fixture",
            "partition": "training",
            "session": "session-a",
        })
        writer.close()
        descriptor = writer.published[0]
        tar_path = root / "ready" / descriptor["tar"]
        assert hashlib.sha256(tar_path.read_bytes()).hexdigest() == descriptor["sha256"]
        index_path = root / "ready" / descriptor["index"]
        index = json.loads(index_path.read_text().strip())
        with tarfile.open(tar_path) as archive:
            member = archive.getmember(index["member"])
            assert member.offset_data == index["offset"]
            extracted = archive.extractfile(member)
            assert extracted is not None
            assert hashlib.sha256(extracted.read()).hexdigest() == index["sha256"]


def test_custody_deduplicates_inputs_and_preserves_plan() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root / "artifact"
        artifact.mkdir()
        binary = root / "binary"
        input_path = root / "input"
        plan = root / "plan.json"
        binary.write_bytes(b"binary")
        input_path.write_bytes(b"same")
        plan.write_text("{}\n")
        row = WorkloadRow(
            execution_id="row-1", family="fixture", application="fixture",
            partition="training", session="session-a",
            matched_input_partition=None, input_seed=1,
            argv=(str(binary),), stdin=b"", cwd=str(root),
            input_paths=(str(input_path), str(input_path)),
            input_sha256=(_hash(b"same"), _hash(b"same")), support_paths=(),
            expected_exit_code=0,
            expected_stdout_sha256=_hash(b""), expected_stderr_sha256=_hash(b""),
            timeout_seconds=1,
        )
        manifest = seal_custody(
            artifact, plan, (row,), binary, (os.getuid(), os.getgid())
        )
        digests = {item["sha256"] for item in manifest["unique_objects"]}
        assert {_hash(b"binary"), _hash(b"same")} <= digests
        assert sum(
            item["sha256"] == _hash(b"same")
            for item in manifest["unique_objects"]
        ) == 1
        assert (artifact / "custody/plan.json").read_text() == "{}\n"


def test_inexact_pebs_is_retained_and_counted_without_unsigned_underflow() -> None:
    empty = torch.empty(0, dtype=torch.int64)
    pt = HardwareBatch(
        source=77, signal="intel_pt", ip=empty, pid=empty, tid=empty,
        time=empty, cpu=empty, period=empty, address=empty, weight=empty,
        data_source=empty, exact_ip=torch.empty(0, dtype=torch.bool),
        trace_bytes=torch.tensor([1], dtype=torch.uint8),
        perf_records=torch.empty(0, dtype=torch.uint8),
    )
    pebs = HardwareBatch(
        source=77, signal="memory_loads", ip=torch.tensor([1]),
        pid=torch.tensor([77]), tid=torch.tensor([77]), time=torch.tensor([1]),
        cpu=torch.tensor([2]), period=torch.tensor([10_000]),
        address=torch.tensor([0]), weight=torch.tensor([1]),
        data_source=torch.tensor([1]), exact_ip=torch.tensor([False]),
        trace_bytes=empty, perf_records=None,
    )
    counters = HardwareCounterBatch(
        77, 77, -1, ("instructions", "cycles", "ref_cycles"),
        torch.tensor([1, 2, 3]), 10, 10, True, False,
    )
    batch = HardwareMultimodalBatch(
        77, 77, -1, HardwareCaptureEnvelope("CLOCK_MONOTONIC_RAW", 1, 2, 3, 4),
        (
            HardwareSourceStatus("intel_pt", True, True, False),
            HardwareSourceStatus("memory_loads", True, True, False, 10, 10),
            HardwareSourceStatus("counters", True, True, False, 10, 10),
        ),
        pt, pebs, counters,
    )
    counts = _validate_capture((batch,), 77, 2)
    assert counts["pebs_samples"] == 1
    assert counts["pebs_inexact_samples"] == 1
    assert counts["pebs_zero_address_samples"] == 1
    assert counts["pebs_usable_samples"] == 0
