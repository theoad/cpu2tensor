# SPDX-License-Identifier: AGPL-3.0-only
"""Offline deterministic pairing and score-summary tests for proxy R2."""

import json
from pathlib import Path

import pytest
import torch

from cpu2tensor.examples import hardware_crossmodal_proxy_r2 as proxy


def test_distance_uses_only_matching_exposure() -> None:
    values = [[100, 1_000, 100], [105, 1_050, 110], [150, 1_000, 100]]
    assert proxy._distance(values, 0, 1) is not None
    assert proxy._distance(values, 0, 2) is None


def test_pair_list_is_deterministic_disjoint_and_score_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [f"row-{index}" for index in range(6)]
    entries = [{
        "execution_id": execution_id, "partition": "familiar_validation",
        "family": "pipe", "loops": 8, "admission": {"attempt": 1},
        "capture": {"lost_sources": 0, "missing_sources": 0,
                    "multiplexed_sources": 0},
        "lanes": [{"migration_verified": True}],
    } for execution_id in ids]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": entries}))
    monkeypatch.setattr(proxy, "SEALED_MANIFEST_SHA256", proxy.sha256(manifest))
    data = {
        "schema": "cpu2tensor-autoresearch-eval-v2",
        "source_manifest_sha256": proxy.SEALED_MANIFEST_SHA256,
        "execution_ids": ids, "partition": torch.tensor([2] * 6),
        "families": ["pipe"] * 6, "loops": [8] * 6,
        "exposure_raw": torch.tensor([[100 + index, 1_000 + index, 100 + index]
                                      for index in range(6)], dtype=torch.int64),
    }
    monkeypatch.setattr(proxy, "_load_eval", lambda *args: data)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    a = proxy.make_pairs(tmp_path / "unused.pt", manifest, first, expected_manifest_entries=6)
    b = proxy.make_pairs(tmp_path / "unused.pt", manifest, second, expected_manifest_entries=6)
    assert first.read_bytes() == second.read_bytes()
    assert a == b
    assert len(a["pairs"]) == 3
    selected = [pair[key] for pair in a["pairs"] for key in ("anchor", "donor")]
    assert sorted(selected) == sorted(ids)
    assert a["coverage"]["2/pipe/8"]["unpaired_rows"] == 0


def test_summary_separates_hybrid_from_metadata_artifact() -> None:
    scores = {
        "anchor": torch.tensor([0.2, 0.2]),
        "donor": torch.tensor([0.3, 0.3]),
        "hybrid": torch.tensor([0.9, 0.4]),
        "sham": torch.tensor([0.2, 0.2]),
        "metadata_only": torch.tensor([0.2, 0.8]),
    }
    result = proxy._summary(scores, 0.5, torch.tensor([True, True]))
    assert result["hybrid_alerts"] == 1
    assert result["metadata_only_alerts"] == 1
    assert result["only_hybrid_alerts"] == 1
    assert result["hybrid_vs_original_max_win_rate"] == 1.0


def test_full_unmodified_benign_counts_include_unpaired_rows() -> None:
    payload = {
        "partition": torch.tensor([2, 2, 3, 3], dtype=torch.uint8),
        "families": ["pipe", "pipe", "memfd", "memfd"],
        "loops": [8, 16, 8, 16],
    }
    result = proxy._benign_counts(torch.tensor([0.1, 0.9, 0.1, 0.9]),
                                  payload, threshold=0.5)
    assert result["familiar_validation"]["all"]["alerts"] == 1
    assert result["heldout_family"]["by_family"]["memfd"]["all"]["rows"] == 2
    assert 0.5 < result["familiar_validation"]["all"]["one_sided_95pct_upper"] <= 1
