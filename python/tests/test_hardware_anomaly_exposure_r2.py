# SPDX-License-Identifier: AGPL-3.0-only
"""Focused tests for training-only hardware-exposure conditioning."""

import torch

from cpu2tensor.examples.hardware_anomaly_exposure_r2 import (
    condition_features,
    make_design,
    normalize,
    score,
    tail_result,
)


def test_fit_ignores_calibration_and_holdout_targets() -> None:
    partition = torch.tensor([0] * 12 + [1] * 3 + [2] * 2 + [3] * 3)
    exposures = torch.stack((
        torch.linspace(1, 20, 20),
        torch.linspace(2, 40, 20),
        torch.linspace(3, 60, 20),
    ), dim=1)
    pt_design, pebs_design, _ = make_design(exposures, partition, "volume_time")
    features = torch.zeros((20, 789))
    features[:, 0] = (2 * pt_design[:, 0] + pt_design[:, 1]).float()
    features[:, 512] = (3 * pebs_design[:, 0]).float()
    first, pt_beta, pebs_beta = condition_features(
        features, pt_design, pebs_design, partition,
    )
    changed = features.clone()
    changed[partition != 0, 0] += 10_000
    changed[partition != 0, 512] += 10_000
    _, new_pt_beta, new_pebs_beta = condition_features(
        changed, pt_design, pebs_design, partition,
    )
    assert torch.equal(pt_beta, new_pt_beta)
    assert torch.equal(pebs_beta, new_pebs_beta)
    assert float(first[:12, 0].abs().max()) < 0.1
    assert float(first[:12, 512].abs().max()) < 0.1


def test_normalizer_is_training_only_and_tail_is_calibration_only() -> None:
    partition = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 3])
    conditioned = torch.zeros((9, 789), dtype=torch.float64)
    conditioned[:, 0] = torch.arange(9)
    standardized, center, scale = normalize(conditioned, partition)
    assert center[0] == 1
    changed = conditioned.clone()
    changed[partition != 0, 0] += 1000
    _, new_center, new_scale = normalize(changed, partition)
    assert torch.equal(center, new_center)
    assert torch.equal(scale, new_scale)
    scores = score(standardized)
    result = tail_result(scores, partition, 0.25)
    assert result["calibration_tail_points"] == 1
    scores[partition == 3] += 1000
    assert tail_result(scores, partition, 0.25)["threshold"] == result["threshold"]


def test_predeclared_design_widths() -> None:
    exposure = torch.arange(30, dtype=torch.float64).reshape(10, 3)
    partition = torch.tensor([0] * 6 + [1, 1, 2, 3])
    assert make_design(exposure, partition, "volume")[0].shape[1] == 1
    assert make_design(exposure, partition, "volume_time")[0].shape[1] == 2
    assert make_design(exposure, partition, "volume_time_quadratic")[0].shape[1] == 5
    pt, pebs, state = make_design(exposure, partition, "volume_time_quadratic")
    replay_pt, replay_pebs, _ = make_design(
        exposure, torch.full_like(partition, 3), "volume_time_quadratic", state,
    )
    assert torch.equal(pt, replay_pt)
    assert torch.equal(pebs, replay_pebs)
