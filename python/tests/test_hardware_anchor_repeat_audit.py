# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded exact-plan repeat audit metrics."""
import hashlib
import json

import pytest
import torch

from cpu2tensor.examples.hardware_anchor_repeat_audit import (
    _cosine,
    _load_plan,
    _relative_difference,
    _summary,
    _validate_independent_sessions,
    _validate_plan_rows,
)


def test_cosine_handles_missing_and_identical_vectors():
    zero = torch.zeros(3, dtype=torch.float64)
    one = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert _cosine(zero, zero) == 1.0
    assert _cosine(zero, one) == 0.0
    assert _cosine(one, one) == 1.0


def test_relative_difference_is_symmetric_and_bounded_for_zero():
    assert _relative_difference(0, 0) == 0.0
    assert _relative_difference(10, 20) == _relative_difference(20, 10)
    assert _relative_difference(10, 20) == pytest.approx(2 / 3)


def test_summary_reports_distribution_without_refitting():
    result = _summary([0.0, 0.5, 1.0])
    assert result["rows"] == 3
    assert result["median"] == 0.5
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    with pytest.raises(ValueError, match="empty"):
        _summary([])


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_plan_verifies_declared_file_and_rows(tmp_path):
    plan = {
        "cohort_seed": 7,
        "kind": "smoke",
        "rows": [{"execution_id": "one"}],
    }
    digest = _write(tmp_path / "execution-plan.json", plan)
    manifest = {"split": {"explicit_plan": {
        "cohort_seed": 7, "kind": "smoke", "sha256": digest,
    }}}
    loaded, rows = _load_plan(tmp_path, manifest)
    assert loaded == plan
    assert rows == {"one": {"execution_id": "one"}}
    manifest["split"]["explicit_plan"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="plan hash"):
        _load_plan(tmp_path, manifest)


def test_validate_plan_rows_checks_seed_and_retention():
    plan_rows = {"one": {
        "execution_id": "one", "family": "read", "loops": 2,
        "repetition": 3, "partition": "collection", "input_seed": 4,
        "retain_raw": True,
    }}
    entry = {
        "family": "read", "loops": 2, "repetition": 3,
        "partition": "collection", "invocation": {"input_seed": 4},
        "raw_retained": True,
    }
    _validate_plan_rows({"one": entry}, plan_rows)
    entry["invocation"]["input_seed"] = 5
    with pytest.raises(ValueError, match="input seed"):
        _validate_plan_rows({"one": entry}, plan_rows)


def test_independent_sessions_reject_same_root_and_reused_raw(tmp_path):
    session_a, session_b = tmp_path / "a", tmp_path / "b"
    session_a.mkdir()
    session_b.mkdir()
    (session_a / "capture-manifest.json").write_text("a")
    (session_b / "capture-manifest.json").write_text("b")
    entries_a = {"one": {"raw_sha256": "a"}}
    entries_b = {"one": {"raw_sha256": "b"}}
    _validate_independent_sessions(session_a, session_b, entries_a, entries_b)
    with pytest.raises(ValueError, match="distinct session roots"):
        _validate_independent_sessions(session_a, session_a, entries_a, entries_a)
    entries_b["one"]["raw_sha256"] = "a"
    with pytest.raises(ValueError, match="reuse"):
        _validate_independent_sessions(session_a, session_b, entries_a, entries_b)
