# SPDX-License-Identifier: AGPL-3.0-only
"""Seal physically separate v2 caches with same-execution exposure observables.

This is immutable data preparation, not a candidate model or an evaluator.
The train-only artifact has no partition, family, loop, or effect labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SOURCE_SCHEMA = "cpu2tensor-compact-anomaly-r1"
MANIFEST_SCHEMA = "cpu2tensor-kernel-multimodal-experiment-v1"
TRAIN_SCHEMA = "cpu2tensor-autoresearch-train-v2"
EVAL_SCHEMA = "cpu2tensor-autoresearch-eval-v2"
SOURCE_SHA256 = "5c689576cfd8d1576aa6f063e598a02cfee21d0fa629141e4c9913a8fda1fcdc"
MANIFEST_SHA256 = "0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d"
PARTITIONS = ("training", "calibration", "familiar_validation", "heldout_family")
EXPECTED_COUNTS = (56_000, 21_000, 7_000, 18_000)
EXPOSURE_FIELDS = ("pt_bytes", "elapsed_ns", "pebs_samples")
MAX_INT64 = (1 << 63) - 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: object, *, zero_allowed: bool, name: str) -> int:
    if (type(value) is not int or value < (0 if zero_allowed else 1)
            or value > MAX_INT64):
        raise ValueError(f"{name} must be an in-range nonnegative integer")
    return value


def prepare(
    source_cache: Path, manifest_path: Path, output: Path, *,
    expected_source_sha256: str = SOURCE_SHA256,
    expected_manifest_sha256: str = MANIFEST_SHA256,
    expected_counts: tuple[int, int, int, int] = EXPECTED_COUNTS,
) -> dict[str, object]:
    """Validate custody and row alignment before writing new sealed files."""
    if output.exists():
        raise FileExistsError(f"refusing to replace existing cache directory: {output}")
    source_hash = sha256(source_cache)
    manifest_hash = sha256(manifest_path)
    if source_hash != expected_source_sha256 or manifest_hash != expected_manifest_sha256:
        raise ValueError("source cache or manifest SHA-256 differs from locked input")

    source = torch.load(source_cache, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text())
    if (not isinstance(source, dict) or set(source) != {
            "schema", "manifest_sha256", "features", "partition"}
            or source["schema"] != SOURCE_SCHEMA
            or source["manifest_sha256"] != manifest_hash):
        raise ValueError("compact source schema or manifest identity mismatch")
    if (not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA
            or not isinstance(manifest.get("entries"), list)):
        raise ValueError("capture manifest schema or entries mismatch")
    features = source["features"]
    partition = source["partition"]
    entries = manifest["entries"]
    total = sum(expected_counts)
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32
            or features.device.type != "cpu" or features.shape != (total, 789)
            or not bool(torch.isfinite(features).all())):
        raise ValueError("compact features must be finite CPU float32 [rows,789]")
    if (not isinstance(partition, torch.Tensor) or partition.dtype != torch.uint8
            or partition.device.type != "cpu" or partition.shape != (total,)
            or tuple(int((partition == index).sum()) for index in range(4)) != expected_counts
            or len(entries) != total):
        raise ValueError("compact partition counts or manifest row count mismatch")

    exposure_rows: list[tuple[int, int, int]] = []
    execution_ids: list[str] = []
    families: list[str] = []
    loops: list[int] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry {index} is not an object")
        execution_id = entry.get("execution_id")
        if (not isinstance(execution_id, str) or not execution_id
                or execution_id in seen_ids):
            raise ValueError(f"duplicate or invalid execution identity at row {index}")
        seen_ids.add(execution_id)
        if entry.get("partition") != PARTITIONS[int(partition[index])]:
            raise ValueError(f"manifest/source partition mismatch at {execution_id}")
        family = entry.get("family")
        if not isinstance(family, str) or not family:
            raise ValueError(f"invalid family at {execution_id}")
        loop_count = _positive_int(entry.get("loops"), zero_allowed=False,
                                   name=f"loops of {execution_id}")
        capture = entry.get("capture")
        if not isinstance(capture, dict):
            raise ValueError(f"missing capture of {execution_id}")
        pt_bytes = _positive_int(capture.get("pt_bytes"), zero_allowed=False,
                                 name=f"PT bytes of {execution_id}")
        elapsed_ns = _positive_int(entry.get("elapsed_ns"), zero_allowed=False,
                                   name=f"elapsed ns of {execution_id}")
        pebs_samples = _positive_int(capture.get("pebs_samples"), zero_allowed=True,
                                     name=f"PEBS samples of {execution_id}")
        exposure_rows.append((pt_bytes, elapsed_ns, pebs_samples))
        execution_ids.append(execution_id)
        families.append(family)
        loops.append(loop_count)

    exposure = torch.tensor(exposure_rows, dtype=torch.int64)
    train_mask = partition == 0
    eval_mask = ~train_mask
    identity = {
        "source_manifest_sha256": manifest_hash,
        "source_cache_sha256": source_hash,
        "exposure_fields": EXPOSURE_FIELDS,
    }
    train = {
        "schema": TRAIN_SCHEMA,
        "features": features[train_mask].contiguous(),
        "exposure_raw": exposure[train_mask].contiguous(),
        **identity,
    }
    development_eval = {
        "schema": EVAL_SCHEMA,
        "features": features[eval_mask].contiguous(),
        "exposure_raw": exposure[eval_mask].contiguous(),
        "partition": partition[eval_mask].contiguous(),
        "execution_ids": [value for value, selected in zip(execution_ids, eval_mask.tolist()) if selected],
        "families": [value for value, selected in zip(families, eval_mask.tolist()) if selected],
        "loops": [value for value, selected in zip(loops, eval_mask.tolist()) if selected],
        **identity,
    }
    if sha256(source_cache) != source_hash or sha256(manifest_path) != manifest_hash:
        raise ValueError("input changed while v2 caches were being prepared")

    output.mkdir(parents=True, exist_ok=False)
    train_path = output / "train-only.pt"
    eval_path = output / "development-eval.pt"
    torch.save(train, train_path)
    torch.save(development_eval, eval_path)
    report: dict[str, object] = {
        "schema": "cpu2tensor-autoresearch-data-seal-v2",
        "source_cache_sha256": source_hash,
        "source_manifest_sha256": manifest_hash,
        "train_cache_sha256": sha256(train_path),
        "development_eval_sha256": sha256(eval_path),
        "training_rows": expected_counts[0],
        "development_rows": sum(expected_counts[1:]),
        "exposure_fields": list(EXPOSURE_FIELDS),
    }
    (output / "seal.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_cache", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_cache, args.manifest, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
