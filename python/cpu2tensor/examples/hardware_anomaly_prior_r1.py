# SPDX-License-Identifier: AGPL-3.0-only
"""Rescore already-frozen hardware models at matched benign tail budgets.

This reads only calibration and benign validation rows of the sealed corpus.
The archived vulnerability canary is deliberately outside this comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import torch

from cpu2tensor.examples.hardware_anomaly_model_r1 import sha256, threshold_and_counts
from cpu2tensor.examples.hardware_multimodal import (
    load_frozen_multimodal_model, multimodal_anomaly_score,
)
from cpu2tensor.examples.hardware_multimodal_experiment import (
    MarginalBaseline, PtPcaBaseline, _concatenate, _model_batch,
)


PARTITIONS = ("calibration", "familiar_validation", "heldout_family")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("d128", type=Path)
    parser.add_argument("d512", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.corpus / "capture-manifest.json").read_text())
    corpus_hash = sha256(args.corpus / "capture-manifest.json")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    names = ("marginal", "pt_pca", "d128_fused", "d128_span",
             "d512_fused", "d512_span")
    scorers = {}
    for key, checkpoint_name, baseline_type in (
        ("marginal", "marginal-baseline.pt", MarginalBaseline),
        ("pt_pca", "pt-only-pca-baseline.pt", PtPcaBaseline),
    ):
        checkpoint = torch.load(args.d128 / checkpoint_name, map_location="cpu", weights_only=True)
        scorers[key] = baseline_type(**checkpoint["state"])
    for prefix, folder in (("d128", args.d128), ("d512", args.d512)):
        for variant, filename in (("fused", "fused.pt"), ("span", "span-only.pt")):
            model, _ = load_frozen_multimodal_model(folder / filename, device=device)
            if model.config.model_dimensions != int(prefix[1:]):
                raise ValueError("checkpoint width mismatch")
            scorers[f"{prefix}_{variant}"] = model
    scores = {name: {partition: [] for partition in PARTITIONS} for name in names}
    timing = {name: 0.0 for name in names}
    counts = {name: 0 for name in PARTITIONS}
    started = time.perf_counter()
    pending = []
    current_partition = None

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        batch_cpu = _concatenate(pending)
        batch_device = batch_cpu.to(device)
        for name in names:
            t0 = time.perf_counter()
            with torch.inference_mode():
                if name in ("marginal", "pt_pca"):
                    values = scorers[name].score(batch_cpu)
                else:
                    values = multimodal_anomaly_score(scorers[name], batch_device)
                    if device.type == "mps":
                        torch.mps.synchronize()
            timing[name] += time.perf_counter() - t0
            scores[name][current_partition].append(values.cpu())
        counts[current_partition] += len(pending)
        pending = []

    for partition in PARTITIONS:
        current_partition = partition
        for entry in manifest["entries"]:
            if entry["partition"] != partition:
                continue
            derived = args.corpus / entry["derived_path"]
            if sha256(derived) != entry["derived_sha256"]:
                raise ValueError(f"derived hash mismatch: {entry['execution_id']}")
            payload = torch.load(derived, map_location="cpu", weights_only=True)
            if payload["execution"]["execution_id"] != entry["execution_id"]:
                raise ValueError("derived execution identity mismatch")
            pending.append(_model_batch(payload["batch"]))
            if len(pending) == 128:
                flush()
        flush()
    if [counts[name] for name in PARTITIONS] != [21_000, 7_000, 18_000]:
        raise ValueError("unexpected split counts")
    result = {}
    for name in names:
        partition_scores = [torch.cat(scores[name][partition]) for partition in PARTITIONS]
        result[name] = {
            "fpr_1e-3": threshold_and_counts(*partition_scores, 1e-3),
            "fpr_1e-4_exploratory": threshold_and_counts(*partition_scores, 1e-4),
            "scoring_seconds": timing[name],
            "scoring_executions_per_second": 46_000 / timing[name],
        }
    report = {
        "schema": "cpu2tensor-hardware-anomaly-prior-rescore-r1",
        "dataset_manifest_sha256": corpus_hash,
        "host": platform.node(), "machine": platform.machine(),
        "device": str(device), "total_wall_seconds": time.perf_counter() - started,
        "scores": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
