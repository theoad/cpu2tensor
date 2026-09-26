# SPDX-License-Identifier: AGPL-3.0-only
"""Offline exposure-cache contract tests; no training or physical capture."""

from pathlib import Path

import pytest
import torch

from cpu2tensor.examples import hardware_autoresearch_eval as evaluator
from cpu2tensor.examples.hardware_autoresearch_train import (
    DenoisingModel,
    save_frozen_v2,
)


COUNTS = {1: 2, 2: 2, 3: 2}
SOURCE_IDENTITY = {
    "source_manifest_sha256": "a" * 64,
    "source_cache_sha256": "b" * 64,
}


def payload() -> dict[str, object]:
    return {
        "schema": evaluator.EVAL_SCHEMA_V2,
        "features": torch.zeros((6, evaluator.FEATURES), dtype=torch.float32),
        "exposure_raw": torch.tensor(
            [[100 + index, 1_000 + index, index] for index in range(6)],
            dtype=torch.int64,
        ),
        "exposure_fields": evaluator.EXPOSURE_FIELDS,
        "partition": torch.tensor([1, 1, 2, 2, 3, 3], dtype=torch.uint8),
        "execution_ids": [f"row-{index}" for index in range(6)],
        "families": ["pipe", "pipe", "pipe", "mmap", "heldout", "heldout"],
        "loops": [8, 16, 8, 16, 8, 16],
        **SOURCE_IDENTITY,
    }


def checkpoint(path: Path, *, schema: str = evaluator.CHECKPOINT_SCHEMA_V2,
               train_hash: str = evaluator.SEALED_V2_TRAIN_SHA256) -> None:
    identity = {
        **SOURCE_IDENTITY,
        "exposure_fields": evaluator.EXPOSURE_FIELDS,
        "train_cache_sha256": train_hash,
    }
    torch.save({"schema": schema, "seed": 1729, "train_cache_identity": identity}, path)


def test_v2_cache_requires_exact_exposure_contract() -> None:
    good = payload()
    assert evaluator.validate_eval_cache(good, expected_counts=COUNTS) is good
    for key, bad_value in (
        ("exposure_fields", ("elapsed_ns", "pt_bytes", "pebs_samples")),
        ("exposure_raw", good["exposure_raw"].float()),
        ("exposure_raw", good["exposure_raw"][:, :2]),
        ("exposure_raw", torch.zeros((6, 3), dtype=torch.int64)),
    ):
        with pytest.raises(ValueError, match="exposure"):
            evaluator.validate_eval_cache({**good, key: bad_value}, expected_counts=COUNTS)
    with pytest.raises(ValueError, match="incomplete"):
        evaluator.validate_eval_cache(
            {key: value for key, value in good.items() if key != "exposure_raw"},
            expected_counts=COUNTS,
        )


def test_v2_evaluator_forwards_raw_exposure_and_never_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "development-eval-v2.pt"
    torch.save(payload(), cache)
    digest = evaluator.sha256(cache)
    monkeypatch.setattr(evaluator, "SEALED_V2_EVAL_SHA256", digest)
    frozen = tmp_path / "seed-1729.pt"
    checkpoint(frozen)
    seen = []

    def score_features(scorer: object, features: torch.Tensor, *,
                       exposure_raw: torch.Tensor, batch_size: int) -> torch.Tensor:
        assert scorer is sentinel
        assert batch_size == 3
        assert torch.equal(exposure_raw, payload()["exposure_raw"])
        seen.append(exposure_raw.clone())
        return exposure_raw[:, 0].float()

    sentinel = object()
    monkeypatch.setattr(evaluator, "load_frozen", lambda path, *, device: sentinel)
    monkeypatch.setattr(evaluator, "score_features", score_features)
    output = tmp_path / "report"
    report = evaluator.evaluate(cache, digest, [frozen], output,
                                expected_counts=COUNTS, target_fpr=0.5, batch_size=3)
    assert len(seen) == 1
    assert report["schema"] == evaluator.REPORT_SCHEMA_V2
    assert report["development_eval_schema"] == evaluator.EVAL_SCHEMA_V2
    assert report["checkpoint_schema"] == evaluator.CHECKPOINT_SCHEMA_V2
    assert report["source_identity"]["train_cache_sha256"] == evaluator.SEALED_V2_TRAIN_SHA256
    assert report["keep_discard_authorized"] is False
    assert report["status"] == "proxy_benign_only_not_real_bug_gate"
    assert report["seeds"][0]["metrics"]["heldout_family"]["rows"] == 2
    saved = torch.load(output / "scores.pt", weights_only=True)
    assert saved["schema"] == evaluator.SCORES_SCHEMA_V2


def test_v2_rejects_wrong_train_hash_or_checkpoint_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "development-eval-v2.pt"
    torch.save(payload(), cache)
    digest = evaluator.sha256(cache)
    monkeypatch.setattr(evaluator, "SEALED_V2_EVAL_SHA256", digest)
    wrong_train = tmp_path / "wrong-train.pt"
    checkpoint(wrong_train, train_hash="c" * 64)
    wrong_version = tmp_path / "wrong-version.pt"
    checkpoint(wrong_version, schema=evaluator.CHECKPOINT_SCHEMA)
    for index, frozen in enumerate((wrong_train, wrong_version)):
        with pytest.raises(ValueError, match="source custody"):
            evaluator.evaluate(cache, digest, [frozen], tmp_path / f"report-{index}",
                               expected_counts=COUNTS)
    monkeypatch.setattr(evaluator, "SEALED_V2_EVAL_SHA256", "d" * 64)
    with pytest.raises(ValueError, match="sealed cache"):
        evaluator.evaluate(cache, digest, [wrong_train], tmp_path / "wrong-seal",
                           expected_counts=COUNTS)


def test_v2_real_frozen_scorer_interface_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "development-eval-v2.pt"
    torch.save(payload(), cache)
    digest = evaluator.sha256(cache)
    monkeypatch.setattr(evaluator, "SEALED_V2_EVAL_SHA256", digest)
    model = DenoisingModel()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    correction = {
        "exposure_center": torch.zeros(3, dtype=torch.float64),
        "exposure_scale": torch.ones(3, dtype=torch.float64),
        "pt_design_center": torch.zeros(2, dtype=torch.float64),
        "pebs_design_center": torch.zeros(2, dtype=torch.float64),
        "pt_coefficients": torch.zeros((2, 512), dtype=torch.float64),
        "pebs_coefficients": torch.zeros((2, 2), dtype=torch.float64),
    }
    frozen = tmp_path / "frozen-v2.pt"
    save_frozen_v2(
        frozen, model, torch.zeros(evaluator.FEATURES),
        torch.ones(evaluator.FEATURES), correction, seed=1729, steps=0,
        cache_identity={
            **SOURCE_IDENTITY,
            "exposure_fields": evaluator.EXPOSURE_FIELDS,
            "train_cache_sha256": evaluator.SEALED_V2_TRAIN_SHA256,
        },
    )
    report = evaluator.evaluate(cache, digest, [frozen], tmp_path / "real-interface",
                                expected_counts=COUNTS, target_fpr=0.5)
    assert report["seeds"][0]["metrics"]["calibration"]["rows"] == 2
    assert report["seeds"][0]["scoring_rows_per_second"] > 0
