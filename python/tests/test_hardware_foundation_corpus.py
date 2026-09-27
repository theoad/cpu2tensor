# SPDX-License-Identifier: AGPL-3.0-only

import base64
import hashlib
import json
import os
from pathlib import Path
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
from cpu2tensor.hardware import (
    HardwareBatch, HardwareCaptureEnvelope, HardwareCounterBatch,
    HardwareMultimodalBatch, HardwareSourceStatus,
)


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
