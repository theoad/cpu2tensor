# SPDX-License-Identifier: AGPL-3.0-only
"""Focused checks for the single-file train-only autoresearch boundary."""

from pathlib import Path
import time

import pytest
import torch

from cpu2tensor.examples.hardware_autoresearch_train import (
    FEATURES,
    FrozenScorer,
    fit_normalizer,
    load_frozen,
    save_frozen,
    score_features,
    train_one,
    validate_train_cache,
)


IDENTITY = {
    "source_manifest_sha256": "a" * 64,
    "source_cache_sha256": "b" * 64,
}


def test_cache_rejects_partitions_and_nonfinite_rows() -> None:
    payload = {
        "schema": "cpu2tensor-autoresearch-train-v1",
        "features": torch.ones((8, FEATURES), dtype=torch.float32),
        **IDENTITY,
    }
    features, identity = validate_train_cache(payload, expected_rows=8)
    assert features.shape == (8, FEATURES)
    assert identity == IDENTITY
    with pytest.raises(ValueError, match="only the agreed"):
        validate_train_cache({**payload, "partition": torch.zeros(8)}, expected_rows=8)
    bad = {**payload, "features": features.clone()}
    bad["features"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_train_cache(bad, expected_rows=8)


def test_checkpoint_replays_training_score(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(9)
    features = torch.randn((64, FEATURES), generator=generator)
    center, scale = fit_normalizer(features)
    model, metrics = train_one(
        features, center, scale, seed=1729, device=torch.device("cpu"),
        deadline=time.perf_counter() + 30, steps=2, batch_size=16,
    )
    assert metrics["steps"] == 2
    checkpoint = tmp_path / "seed-1729.pt"
    save_frozen(checkpoint, model, center, scale, seed=1729,
                steps=2, cache_identity=IDENTITY)
    original = FrozenScorer(model, center, scale, 1729)
    restored = load_frozen(checkpoint)
    expected = score_features(original, features, batch_size=7)
    observed = score_features(restored, features, batch_size=7)
    assert torch.equal(expected, observed)
    assert bool(torch.isfinite(observed).all())
    assert observed.shape == (64,)


def test_training_seed_reproduces_cpu_weights() -> None:
    torch.set_num_threads(2)
    features = torch.arange(32 * FEATURES, dtype=torch.float32).reshape(32, FEATURES) / 1000
    center, scale = fit_normalizer(features)
    first, _ = train_one(features, center, scale, seed=1730,
                         device=torch.device("cpu"), deadline=time.perf_counter() + 30,
                         steps=2, batch_size=8)
    second, _ = train_one(features, center, scale, seed=1730,
                          device=torch.device("cpu"), deadline=time.perf_counter() + 30,
                          steps=2, batch_size=8)
    assert all(torch.equal(first.state_dict()[key], second.state_dict()[key])
               for key in first.state_dict())
