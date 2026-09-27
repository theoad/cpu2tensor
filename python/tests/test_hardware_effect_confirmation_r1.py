# SPDX-License-Identifier: AGPL-3.0-only
import pytest
import torch
from pathlib import Path

from cpu2tensor.examples.hardware_effect_confirmation_r1 import (
    conjunctive_gate,
    raw_hashes,
    validate_confirmation_protocol,
    validate_family_counts,
)


def test_conjunctive_gate_requires_both_scores_above_their_maxima() -> None:
    fused = torch.tensor([0.9, 1.1, 0.9, 1.1])
    pmu = torch.tensor([0.9, 0.9, 1.1, 1.1])
    assert conjunctive_gate(fused, pmu, 1.0, 1.0).tolist() == [
        False, False, False, True,
    ]


def test_conjunctive_gate_rejects_misaligned_scores() -> None:
    with pytest.raises(ValueError, match="aligned one-dimensional"):
        conjunctive_gate(torch.zeros(2), torch.zeros(2, 1), 1.0, 1.0)


def test_raw_hashes_and_family_counts_reject_reuse_or_wrong_cohort() -> None:
    manifest = {"entries": [
        {"family": "a", "raw_sha256": "one"},
        {"family": "b", "raw_sha256": "two"},
    ]}
    assert raw_hashes(manifest) == {"one", "two"}
    validate_family_counts(manifest, {"a": 1, "b": 1})
    with pytest.raises(ValueError, match="family counts"):
        validate_family_counts(manifest, {"a": 2})
    manifest["entries"][1]["raw_sha256"] = "one"
    with pytest.raises(ValueError, match="reuses a raw capture"):
        raw_hashes(manifest)


def test_confirmation_protocol_rejects_missing_frozen_plan(tmp_path: Path) -> None:
    benign = {"split": {}, "entries": []}
    effect = {"split": {}, "entries": []}
    with pytest.raises(ValueError, match="execution plan differs"):
        validate_confirmation_protocol(tmp_path, benign, effect)
