# SPDX-License-Identifier: AGPL-3.0-only
"""Focused checks for the offline compact-feature experiment."""

import torch

from cpu2tensor.examples.hardware_anomaly_model_r1 import (
    DenoisingAutoencoder,
    compact_features,
    score_autoencoder,
    threshold_and_counts,
)


def test_compact_features_ignore_unavailable_pebs() -> None:
    pebs = torch.zeros((1, 1, 16, 136), dtype=torch.float32)
    pebs[0, 0, 0] = 2
    pebs[0, 0, 1] = float("nan")
    available = torch.zeros((1, 1, 16), dtype=torch.bool)
    available[0, 0, 0] = True
    payload = {"batch": {
        "pt": torch.ones((1, 1, 16, 256), dtype=torch.float32),
        "pebs": pebs,
        "pmu": torch.ones((1, 1, 1, 4), dtype=torch.float32),
        "pebs_available": available,
    }}
    features = compact_features(payload)
    assert features.shape == (789,)
    assert bool(torch.isfinite(features).all())
    assert torch.allclose(features[512:648], torch.full((136,), 2.0))


def test_scores_and_tail_count() -> None:
    torch.manual_seed(1)
    model = DenoisingAutoencoder(6)
    samples = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    assert score_autoencoder(model, samples, 3).shape == (10,)
    calibration = torch.arange(10, dtype=torch.float32)
    result = threshold_and_counts(calibration, torch.tensor([0.0, 9.0]),
                                  torch.tensor([8.0, 9.0]), 0.1)
    assert result["calibration_alerts"] == 1
    assert result["familiar_alerts"] == 1
    assert result["heldout_alerts"] == 1
