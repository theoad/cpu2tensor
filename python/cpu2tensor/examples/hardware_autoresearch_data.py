# SPDX-License-Identifier: AGPL-3.0-only
"""Seal disjoint training and development-evaluation caches for autoresearch.

The mutable training program is given only the training cache. This is a
research-integrity boundary, not an OS sandbox: the coordinator must still
review candidate code and keep the protected cache path out of its arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SOURCE_SCHEMA = "cpu2tensor-compact-anomaly-r1"
TRAIN_SCHEMA = "cpu2tensor-autoresearch-train-v1"
EVAL_SCHEMA = "cpu2tensor-autoresearch-eval-v1"
EXPECTED_PARTITIONS = (56_000, 21_000, 7_000, 18_000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source_cache: Path, manifest_path: Path, output: Path) -> dict[str, str]:
    """Create new sealed files; refuse to replace existing research evidence."""
    train_path = output / "train-only.pt"
    eval_path = output / "development-eval.pt"
    if train_path.exists() or eval_path.exists():
        raise FileExistsError("autoresearch split files already exist")
    source_hash = sha256(source_cache)
    manifest_hash = sha256(manifest_path)
    source = torch.load(source_cache, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text())
    features = source["features"]
    partition = source["partition"]
    entries = manifest["entries"]
    if source["schema"] != SOURCE_SCHEMA or source["manifest_sha256"] != manifest_hash:
        raise ValueError("source cache does not match the sealed manifest")
    if (features.dtype != torch.float32 or features.shape != (102_000, 789)
            or partition.dtype != torch.uint8 or partition.shape != (102_000,)
            or len(entries) != 102_000 or not bool(torch.isfinite(features).all())):
        raise ValueError("source cache shape, dtype, identity, or finiteness failed")
    counts = tuple(int((partition == value).sum()) for value in range(4))
    if counts != EXPECTED_PARTITIONS:
        raise ValueError(f"unexpected whole-execution split: {counts}")
    names = ("training", "calibration", "familiar_validation", "heldout_family")
    for index, entry in enumerate(entries):
        if entry["partition"] != names[int(partition[index])]:
            raise ValueError(f"split identity mismatch: {entry['execution_id']}")

    train_mask = partition == 0
    eval_mask = ~train_mask
    train = {
        "schema": TRAIN_SCHEMA,
        "features": features[train_mask].contiguous(),
        "source_manifest_sha256": manifest_hash,
        "source_cache_sha256": source_hash,
    }
    eval_entries = [entry for entry, selected in zip(entries, eval_mask.tolist()) if selected]
    development_eval = {
        "schema": EVAL_SCHEMA,
        "features": features[eval_mask].contiguous(),
        "partition": partition[eval_mask].contiguous(),
        "execution_ids": [entry["execution_id"] for entry in eval_entries],
        "families": [entry["family"] for entry in eval_entries],
        "loops": [entry["loops"] for entry in eval_entries],
        "source_manifest_sha256": manifest_hash,
        "source_cache_sha256": source_hash,
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(train, train_path)
    torch.save(development_eval, eval_path)
    return {
        "source_cache_sha256": source_hash,
        "source_manifest_sha256": manifest_hash,
        "train_cache_sha256": sha256(train_path),
        "development_eval_sha256": sha256(eval_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_cache", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_cache, args.manifest, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
