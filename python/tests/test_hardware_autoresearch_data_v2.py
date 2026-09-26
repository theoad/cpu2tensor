# SPDX-License-Identifier: AGPL-3.0-only
"""Custody and leakage checks for exposure-aware split preparation."""

import json
from pathlib import Path

import pytest
import torch

from cpu2tensor.examples.hardware_autoresearch_data_v2 import (
    EVAL_SCHEMA,
    EXPOSURE_FIELDS,
    TRAIN_SCHEMA,
    prepare,
    sha256,
)


COUNTS = (2, 1, 1, 1)
PARTITIONS = ("training", "training", "calibration",
              "familiar_validation", "heldout_family")


def fixture_inputs(root: Path, *, mutate=None) -> tuple[Path, Path, str, str]:
    entries = []
    for index, partition in enumerate(PARTITIONS):
        entries.append({
            "execution_id": f"row-{index}",
            "partition": partition,
            "family": "known" if index < 4 else "unseen",
            "loops": index + 1,
            "elapsed_ns": (index + 1) * 1_000_000,
            "capture": {"pt_bytes": (index + 1) * 100,
                        "pebs_samples": index},
        })
    manifest = {"schema": "cpu2tensor-kernel-multimodal-experiment-v1",
                "entries": entries}
    if mutate is not None:
        mutate(manifest)
    manifest_path = root / "capture-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    manifest_hash = sha256(manifest_path)
    source = {
        "schema": "cpu2tensor-compact-anomaly-r1",
        "manifest_sha256": manifest_hash,
        "features": torch.arange(5 * 789, dtype=torch.float32).reshape(5, 789),
        "partition": torch.tensor([0, 0, 1, 2, 3], dtype=torch.uint8),
    }
    source_path = root / "compact-features.pt"
    torch.save(source, source_path)
    return source_path, manifest_path, sha256(source_path), manifest_hash


def prepare_fixture(root: Path, *, mutate=None) -> tuple[dict, Path]:
    source, manifest, source_hash, manifest_hash = fixture_inputs(root, mutate=mutate)
    output = root / "v2"
    report = prepare(source, manifest, output,
                     expected_source_sha256=source_hash,
                     expected_manifest_sha256=manifest_hash,
                     expected_counts=COUNTS)
    return report, output


def test_seals_disjoint_exposure_and_no_train_labels(tmp_path: Path) -> None:
    report, output = prepare_fixture(tmp_path)
    train = torch.load(output / "train-only.pt", weights_only=True)
    dev = torch.load(output / "development-eval.pt", weights_only=True)
    assert set(train) == {"schema", "features", "exposure_raw", "exposure_fields",
                          "source_manifest_sha256", "source_cache_sha256"}
    assert train["schema"] == TRAIN_SCHEMA
    assert dev["schema"] == EVAL_SCHEMA
    assert train["exposure_fields"] == EXPOSURE_FIELDS
    assert dev["exposure_fields"] == EXPOSURE_FIELDS
    assert train["features"].shape == (2, 789)
    assert dev["features"].shape == (3, 789)
    assert train["exposure_raw"].dtype == torch.int64
    assert train["exposure_raw"].tolist() == [
        [100, 1_000_000, 0], [200, 2_000_000, 1],
    ]
    assert dev["exposure_raw"].tolist() == [
        [300, 3_000_000, 2], [400, 4_000_000, 3], [500, 5_000_000, 4],
    ]
    assert dev["partition"].tolist() == [1, 2, 3]
    assert dev["execution_ids"] == ["row-2", "row-3", "row-4"]
    assert dev["families"] == ["known", "known", "unseen"]
    assert report == json.loads((output / "seal.json").read_text())
    assert report["train_cache_sha256"] == sha256(output / "train-only.pt")
    assert report["development_eval_sha256"] == sha256(output / "development-eval.pt")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        prepare_fixture(tmp_path)


def test_rejects_wrong_hash_and_source_manifest_link(tmp_path: Path) -> None:
    source, manifest, source_hash, manifest_hash = fixture_inputs(tmp_path)
    with pytest.raises(ValueError, match="locked input"):
        prepare(source, manifest, tmp_path / "v2", expected_source_sha256="0" * 64,
                expected_manifest_sha256=manifest_hash, expected_counts=COUNTS)
    payload = torch.load(source, weights_only=True)
    payload["manifest_sha256"] = "0" * 64
    torch.save(payload, source)
    with pytest.raises(ValueError, match="manifest identity"):
        prepare(source, manifest, tmp_path / "v2", expected_source_sha256=sha256(source),
                expected_manifest_sha256=manifest_hash, expected_counts=COUNTS)
    assert not (tmp_path / "v2").exists()
    assert source_hash != sha256(source)


@pytest.mark.parametrize("mutate,match", [
    (lambda manifest: manifest["entries"][0].update(
        {"partition": "heldout_family"}), "partition mismatch"),
    (lambda manifest: manifest["entries"][1].update(
        {"execution_id": "row-0"}), "duplicate or invalid execution identity"),
    (lambda manifest: manifest["entries"][0]["capture"].update(
        {"pt_bytes": 0}), "PT bytes"),
    (lambda manifest: manifest["entries"][0]["capture"].update(
        {"pebs_samples": -1}), "PEBS samples"),
    (lambda manifest: manifest["entries"][0].update(
        {"elapsed_ns": 0}), "elapsed ns"),
])
def test_rejects_row_identity_or_invalid_exposure(tmp_path: Path, mutate, match: str) -> None:
    source, manifest, source_hash, manifest_hash = fixture_inputs(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match=match):
        prepare(source, manifest, tmp_path / "v2",
                expected_source_sha256=source_hash,
                expected_manifest_sha256=manifest_hash,
                expected_counts=COUNTS)
    assert not (tmp_path / "v2").exists()
