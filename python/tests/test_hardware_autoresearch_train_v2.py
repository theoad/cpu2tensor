# SPDX-License-Identifier: AGPL-3.0-only
"""Nontraining checks for the single-file exposure-conditioned candidate."""

from pathlib import Path

import pytest
import torch

from cpu2tensor.examples.hardware_autoresearch_train import (
    CHECKPOINT_SCHEMA,
    DenoisingModel,
    EXPOSURE_FIELDS,
    FrozenScorer,
    condition_features,
    fit_exposure_correction,
    fit_normalizer,
    load_frozen,
    save_frozen_v2,
    score_features,
    validate_train_cache_v2,
)


IDENTITY = {
    "source_manifest_sha256": "a" * 64,
    "source_cache_sha256": "b" * 64,
    "exposure_fields": EXPOSURE_FIELDS,
}


def fixture_data(rows: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn((rows, 789), generator=generator)
    exposure = torch.stack((
        torch.arange(1, rows + 1, dtype=torch.int64) * 100,
        torch.arange(1, rows + 1, dtype=torch.int64) * 1_000_000,
        torch.arange(rows, dtype=torch.int64),
    ), dim=1)
    return features, exposure


def test_v2_train_only_schema_rejects_labels_and_invalid_exposure() -> None:
    features, exposure = fixture_data()
    payload = {
        "schema": "cpu2tensor-autoresearch-train-v2",
        "features": features,
        "exposure_raw": exposure,
        **IDENTITY,
    }
    observed_features, observed_exposure, identity = validate_train_cache_v2(
        payload, expected_rows=16,
    )
    assert observed_features is features
    assert observed_exposure is exposure
    assert identity == IDENTITY
    with pytest.raises(ValueError, match="only agreed train-only"):
        validate_train_cache_v2({**payload, "families": ["pipe"] * 16}, expected_rows=16)
    with pytest.raises(ValueError, match="field order"):
        validate_train_cache_v2({**payload, "exposure_fields": EXPOSURE_FIELDS[::-1]},
                                expected_rows=16)
    bad_exposure = exposure.clone()
    bad_exposure[0, 0] = 0
    with pytest.raises(ValueError, match="valid counts"):
        validate_train_cache_v2({**payload, "exposure_raw": bad_exposure},
                                expected_rows=16)


def test_train_only_correction_is_reused_for_new_rows() -> None:
    features, exposure = fixture_data(32)
    correction = fit_exposure_correction(features, exposure)
    first = condition_features(features[:8], exposure[:8], correction)
    all_rows = condition_features(features, exposure, correction)
    assert torch.equal(first, all_rows[:8])
    altered = features.clone()
    altered[8:] += 1000
    assert torch.equal(
        first,
        condition_features(altered[:8], exposure[:8], correction),
    )
    assert all(tensor.dtype == torch.float64 for tensor in correction.values())


def test_v2_checkpoint_replays_exposure_scores_without_training(tmp_path: Path) -> None:
    features, exposure = fixture_data()
    correction = fit_exposure_correction(features, exposure)
    conditioned = condition_features(features, exposure, correction)
    center, scale = fit_normalizer(conditioned)
    torch.manual_seed(1729)
    model = DenoisingModel().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint = tmp_path / "seed-1729.pt"
    identity = {**IDENTITY, "train_cache_sha256": "c" * 64}
    save_frozen_v2(checkpoint, model, center, scale, correction,
                   seed=1729, steps=0, cache_identity=identity)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["train_cache_identity"] == identity
    original = FrozenScorer(model, center, scale, 1729, correction)
    restored = load_frozen(checkpoint)
    assert torch.equal(
        score_features(original, features, exposure, batch_size=3),
        score_features(restored, features, exposure, batch_size=5),
    )
    with pytest.raises(ValueError, match="valid counts"):
        score_features(restored, features, exposure.to(torch.float32))
    with pytest.raises(ValueError, match="valid counts"):
        score_features(restored, features)
