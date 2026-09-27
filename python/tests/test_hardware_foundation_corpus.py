# SPDX-License-Identifier: AGPL-3.0-only

import base64
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

import torch

from cpu2tensor.examples.hardware_foundation_corpus import (
    PLAN_SCHEMA,
    TarShardWriter,
    load_plan,
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
            "input_seed": 17,
            "argv": [str(binary), "-c", str(input_path)],
            "stdin_base64": base64.b64encode(b"").decode(),
            "cwd": str(root),
            "input_paths": [str(input_path)],
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
        })
        path.write_text(json.dumps(plan))
        try:
            load_plan(path)
        except ValueError as error:
            assert "leak" in str(error)
        else:
            raise AssertionError("application leakage was accepted")


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
