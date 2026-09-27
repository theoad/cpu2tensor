# SPDX-License-Identifier: AGPL-3.0-only
"""Cross-session transfer stability diagnostics are label-honest and deterministic."""
import pytest
import torch

from cpu2tensor.examples.hardware_transfer_stability import auc, diagnose_modality


def batch(rows: torch.Tensor) -> dict[str, torch.Tensor]:
    count = rows.shape[0]
    values = rows[:, None, None, :].to(torch.float32)
    available = torch.ones((count, 1, 1), dtype=torch.bool)
    return {
        "pt": values,
        "pebs": values,
        "pmu": values,
        "pt_available": available,
        "pebs_available": available,
        "pmu_available": available,
        "time_bounds": values,
        "timing_quality": values,
    }


def test_auc_counts_ties_as_half():
    assert auc(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0])) == 0.875


def test_stable_effect_transfers_despite_orthogonal_session_shift():
    benign = batch(torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]))
    train = {
        "neutral": batch(torch.tensor([[0.0, 0.0], [0.1, 0.0]])),
        "effect": batch(torch.tensor([[2.0, 0.0], [2.1, 0.0]])),
    }
    test = {
        "neutral": batch(torch.tensor([[0.0, 5.0], [0.1, 5.0]])),
        "effect": batch(torch.tensor([[2.0, 5.0], [2.1, 5.0]])),
    }
    result = diagnose_modality(benign, train, test, "pmu")
    assert result["test_auc"] == 1.0
    assert result["test_paired_wins"] == 2
    assert result["cross_session_effect_cosine"] > 0.999


def test_reversed_effect_is_exposed_as_nontransfer():
    benign = batch(torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]))
    train = {
        "neutral": batch(torch.tensor([[0.0, 0.0], [0.0, 0.0]])),
        "effect": batch(torch.tensor([[2.0, 0.0], [2.0, 0.0]])),
    }
    test = {
        "neutral": batch(torch.tensor([[2.0, 0.0], [2.0, 0.0]])),
        "effect": batch(torch.tensor([[0.0, 0.0], [0.0, 0.0]])),
    }
    result = diagnose_modality(benign, train, test, "pmu")
    assert result["test_auc"] == 0.0
    assert result["cross_session_effect_cosine"] < -0.999


def test_unknown_timing_values_are_encoded_as_missing_zeroes():
    rows = batch(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    rows["time_bounds"][0, 0, 0, 0] = torch.nan
    rows["timing_quality"][1, 0, 0] = torch.nan
    benign = batch(torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]))
    result = diagnose_modality(
        benign,
        {"neutral": rows, "effect": batch(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))},
        {"neutral": rows, "effect": batch(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))},
        "timing",
    )
    assert result["pairs_per_session"] == 2


def test_diagnosis_rejects_unequal_or_degenerate_arms():
    benign = batch(torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]))
    equal = {
        "neutral": batch(torch.tensor([[0.0, 0.0], [0.0, 0.0]])),
        "effect": batch(torch.tensor([[1.0, 0.0], [1.0, 0.0]])),
    }
    unequal = {**equal, "effect": batch(torch.tensor([[1.0, 0.0]]))}
    with pytest.raises(ValueError, match="nonempty and equal"):
        diagnose_modality(benign, unequal, equal, "pmu")
    degenerate = {arm: batch(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
                  for arm in ("effect", "neutral")}
    with pytest.raises(ValueError, match="zero pmu training effect"):
        diagnose_modality(benign, degenerate, degenerate, "pmu")
