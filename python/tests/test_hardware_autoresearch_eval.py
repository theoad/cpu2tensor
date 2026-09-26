# SPDX-License-Identifier: AGPL-3.0-only
"""Offline fixtures for immutable, proxy-only checkpoint evaluation."""

from pathlib import Path

import pytest
import torch

from cpu2tensor.examples.hardware_autoresearch_eval import (
    _strata,
    calibrate_threshold,
    evaluate,
    sha256,
    validate_eval_cache,
)
from cpu2tensor.examples.hardware_autoresearch_train import (
    DenoisingModel,
    FEATURES,
    save_frozen,
)


COUNTS = {1: 4, 2: 3, 3: 3}
IDENTITY = {
    "source_manifest_sha256": "a" * 64,
    "source_cache_sha256": "b" * 64,
}


def eval_payload() -> dict[str, object]:
    return {
        "schema": "cpu2tensor-autoresearch-eval-v1",
        "features": torch.arange(10 * FEATURES, dtype=torch.float32).reshape(10, FEATURES) / 1000,
        "partition": torch.tensor([1] * 4 + [2] * 3 + [3] * 3, dtype=torch.uint8),
        "execution_ids": [f"row-{index}" for index in range(10)],
        "families": ["pipe"] * 2 + ["mmap"] * 2 + ["pipe", "mmap", "pipe"]
                    + ["heldout"] * 3,
        "loops": [8, 8, 16, 32, 8, 16, 16, 8, 16, 32],
        **IDENTITY,
    }


def checkpoint(path: Path, seed: int, identity: dict[str, str] = IDENTITY) -> None:
    torch.manual_seed(seed)
    model = DenoisingModel()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    save_frozen(path, model, torch.zeros(FEATURES), torch.ones(FEATURES),
                seed=seed, steps=0, cache_identity=identity)


def test_eval_cache_rejects_bad_custody_and_nonfinite_rows() -> None:
    payload = eval_payload()
    assert validate_eval_cache(payload, expected_counts=COUNTS) is payload
    with pytest.raises(ValueError, match="partition counts"):
        validate_eval_cache({**payload, "partition": torch.zeros(10, dtype=torch.uint8)},
                            expected_counts=COUNTS)
    bad = {**payload, "features": payload["features"].clone()}
    bad["features"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_eval_cache(bad, expected_counts=COUNTS)
    with pytest.raises(ValueError, match="identities"):
        validate_eval_cache({**payload, "execution_ids": ["duplicate"] * 10},
                            expected_counts=COUNTS)


def test_calibration_is_strict_and_uses_only_its_scores() -> None:
    scores = torch.tensor([0.9, 0.5, 0.5, 0.1])
    threshold, allowed = calibrate_threshold(scores, target_fpr=0.25)
    assert (threshold, allowed) == (0.5, 1)
    assert int((scores > threshold).sum()) == 1
    threshold, allowed = calibrate_threshold(scores, target_fpr=1e-4)
    assert threshold == pytest.approx(0.9)
    assert allowed == 0
    with pytest.raises(ValueError, match="finite"):
        calibrate_threshold(torch.tensor([float("inf")]), target_fpr=1e-4)


def test_partition_family_and_intensity_alert_metrics() -> None:
    scores = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.7, 0.3])
    partition = torch.tensor([1, 1, 2, 2, 3, 3], dtype=torch.uint8)
    metrics = _strata(scores, partition,
                      ["pipe", "pipe", "pipe", "mmap", "heldout", "heldout"],
                      [8, 16, 8, 16, 8, 16], threshold=0.5)
    assert metrics["calibration"]["alerts"] == 1
    assert metrics["familiar_validation"]["by_family"]["pipe"]["alerts"] == 1
    assert metrics["familiar_validation"]["by_intensity_loops"]["16"]["alerts"] == 0
    assert metrics["heldout_family"]["by_family_and_intensity"]["heldout"]["8"]["alerts"] == 1


def test_evaluator_seals_two_seed_scores_and_never_authorizes_decision(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    eval_path = tmp_path / "development-eval.pt"
    torch.save(eval_payload(), eval_path)
    first = tmp_path / "seed-1729.pt"
    second = tmp_path / "seed-1730.pt"
    checkpoint(first, 1729)
    checkpoint(second, 1730)
    output = tmp_path / "report"
    report = evaluate(eval_path, sha256(eval_path), (first, second), output,
                      target_fpr=0.25, batch_size=3, expected_counts=COUNTS)
    assert report["status"] == "proxy_benign_only_not_real_bug_gate"
    assert report["keep_discard_authorized"] is False
    assert [entry["seed"] for entry in report["seeds"]] == [1729, 1730]
    assert all(entry["scoring_rows_per_second"] > 0 for entry in report["seeds"])
    assert report["seeds"][0]["metrics"]["calibration"]["rows"] == 4
    assert report["seeds"][1]["metrics"]["heldout_family"]["rows"] == 3
    assert sha256(output / "scores.pt") == report["scores_sha256"]
    saved = torch.load(output / "scores.pt", weights_only=True)
    assert set(saved["scores_by_seed"]) == {"1729", "1730"}
    assert all(values.shape == (10,) for values in saved["scores_by_seed"].values())
    assert (output / "evaluation.json").is_file()
    with pytest.raises(FileExistsError):
        evaluate(eval_path, sha256(eval_path), (first,), output,
                 expected_counts=COUNTS)
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate(eval_path, "0" * 64, (first,), tmp_path / "bad-hash",
                 expected_counts=COUNTS)
    wrong = tmp_path / "wrong.pt"
    checkpoint(wrong, 2000, {**IDENTITY, "source_cache_sha256": "c" * 64})
    with pytest.raises(ValueError, match="source custody"):
        evaluate(eval_path, sha256(eval_path), (wrong,), tmp_path / "bad-source",
                 expected_counts=COUNTS)
