# SPDX-License-Identifier: AGPL-3.0-only
"""Audit frozen hardware representations using benign-only geometry metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics

import torch
from torch.nn import functional as F

from cpu2tensor.examples.hardware_fewshot_transfer_r1 import concatenate
from cpu2tensor.examples.hardware_futex_transfer_r1 import pooled, sha256
from cpu2tensor.examples.hardware_multimodal import (
    MaskedHardwareModel,
    MultimodalConfig,
    freeze_multimodal_model,
    make_training_masks,
)
from cpu2tensor.examples.hardware_multimodal_experiment import load_dataset
from cpu2tensor.examples.hardware_objective_ablation_r3 import (
    OBJECTIVES,
    masked_pool,
)


SCHEMA = "cpu2tensor-hardware-latent-geometry-r4"
R3_REPORT_SHA256 = "b4b6517c2948b7c33e34227c5bc9d1df1feb0e60e87f15b38daa2252752969cb"
CHUNK = 64
VIEW_ROWS_PER_STRATUM = 5
VIEW_SEED = 4401


def effective_rank(embeddings: torch.Tensor) -> float:
    centered = embeddings - embeddings.mean(0)
    singular = torch.linalg.svdvals(centered)
    weights = singular.square()
    weights = weights[weights > 0] / weights.sum()
    return float(torch.exp(-(weights * weights.log()).sum()))


def nearest_centroid_accuracy(
    training: torch.Tensor,
    training_labels: list[str],
    evaluation: torch.Tensor,
    evaluation_labels: list[str],
) -> float:
    classes = sorted(set(training_labels))
    if set(evaluation_labels) - set(classes):
        raise ValueError("evaluation contains an unseen class")
    mean = training.mean(0)
    scale = training.std(0, correction=0).clamp_min(1e-4)
    training = (training - mean) / scale
    evaluation = (evaluation - mean) / scale
    centroids = torch.stack([
        training[
            torch.tensor([label == name for label in training_labels])
        ].mean(0)
        for name in classes
    ])
    similarities = F.normalize(evaluation, dim=1) @ F.normalize(centroids, dim=1).T
    predicted = similarities.argmax(1)
    expected = torch.tensor([classes.index(label) for label in evaluation_labels])
    return float((predicted == expected).to(torch.float32).mean())


def embed(model, batch, device: str) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, batch.batch_size, CHUNK):
            indices = torch.arange(start, min(start + CHUNK, batch.batch_size))
            values.append(pooled(model, batch.index_select(indices).to(device)).cpu())
    return torch.cat(values)


def load_cohort(root: Path):
    manifest, rows = load_dataset(root)
    entries = {entry["execution_id"]: entry for entry in manifest["entries"]}
    identifiers = sorted(rows)
    batch = concatenate([rows[identifier] for identifier in identifiers])
    families = [entries[identifier]["family"] for identifier in identifiers]
    intensities = [identifier.split("-i", 1)[1].split("-s", 1)[0] for identifier in identifiers]
    strata = [f"{family}/i{intensity}" for family, intensity in zip(families, intensities)]
    return manifest, identifiers, batch, families, intensities, strata


def view_subset(identifiers, batch, strata):
    selected = []
    for stratum in sorted(set(strata)):
        candidates = [index for index, value in enumerate(strata) if value == stratum]
        candidates.sort(key=lambda index: hashlib.sha256(
            f"latent-geometry-r4\0{VIEW_SEED}\0{identifiers[index]}".encode()
        ).digest())
        selected.extend(candidates[:VIEW_ROWS_PER_STRATUM])
    selected.sort()
    indices = torch.tensor(selected, dtype=torch.long)
    return batch.index_select(indices), [identifiers[index] for index in selected]


def view_metrics(model, batch, device: str) -> dict[str, float]:
    generator = torch.Generator().manual_seed(VIEW_SEED)
    values = [[], []]
    with torch.no_grad():
        for start in range(0, batch.batch_size, CHUNK):
            indices = torch.arange(start, min(start + CHUNK, batch.batch_size))
            part = batch.index_select(indices).to(device)
            left = make_training_masks(part, generator=generator)
            right = make_training_masks(part, generator=generator)
            values[0].append(masked_pool(model, part, left).cpu())
            values[1].append(masked_pool(model, part, right).cpu())
    left, right = map(torch.cat, values)
    left, right = F.normalize(left, dim=1), F.normalize(right, dim=1)
    similarities = left @ right.T
    expected = torch.arange(left.shape[0])
    return {
        "mean_paired_cosine": float((left * right).sum(1).mean()),
        "identity_top1": float((similarities.argmax(1) == expected).float().mean()),
    }


def centroid_agreement(
    left: torch.Tensor,
    left_labels: list[str],
    right: torch.Tensor,
    right_labels: list[str],
) -> float:
    classes = sorted(set(left_labels))
    values = []
    for name in classes:
        left_centroid = left[torch.tensor([x == name for x in left_labels])].mean(0)
        right_centroid = right[torch.tensor([x == name for x in right_labels])].mean(0)
        values.append(float(F.cosine_similarity(left_centroid[None], right_centroid[None])))
    return statistics.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("r3_artifact", type=Path)
    parser.add_argument("pilot_a", type=Path)
    parser.add_argument("pilot_b", type=Path)
    parser.add_argument("pilot_c", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output directory")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    report_path = args.r3_artifact / "report.json"
    if sha256(report_path) != R3_REPORT_SHA256:
        raise ValueError("R3 artifact differs")
    r3 = json.loads(report_path.read_text())
    cohorts = [load_cohort(root) for root in (args.pilot_a, args.pilot_b, args.pilot_c)]
    if [sha256(root / "capture-manifest.json") for root in (
        args.pilot_a, args.pilot_b, args.pilot_c
    )] != [r3["manifests"][name] for name in ("pilot_a", "pilot_b", "pilot_c")]:
        raise ValueError("benign cohort manifests differ")
    view_batch, view_identifiers = view_subset(
        cohorts[2][1], cohorts[2][2], cohorts[2][5]
    )

    args.output.mkdir(parents=True)
    runs = []
    for prior in r3["runs"]:
        checkpoint = args.r3_artifact / prior["checkpoint"]
        if sha256(checkpoint) != prior["checkpoint_sha256"]:
            raise ValueError("R3 checkpoint differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = MaskedHardwareModel(MultimodalConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model = model.to(args.device)
        freeze_multimodal_model(model)
        embeddings = [embed(model, cohort[2], args.device) for cohort in cohorts]
        family_accuracy = {
            name: nearest_centroid_accuracy(
                embeddings[0], cohorts[0][3], embeddings[index], cohorts[index][3]
            )
            for name, index in (("session_b", 1), ("session_c", 2))
        }
        intensity_accuracy = {
            name: nearest_centroid_accuracy(
                embeddings[0], cohorts[0][4], embeddings[index], cohorts[index][4]
            )
            for name, index in (("session_b", 1), ("session_c", 2))
        }
        stratum_accuracy = {
            name: nearest_centroid_accuracy(
                embeddings[0], cohorts[0][5], embeddings[index], cohorts[index][5]
            )
            for name, index in (("session_b", 1), ("session_c", 2))
        }
        runs.append({
            "objective": prior["objective"],
            "pretraining_seed": prior["pretraining_seed"],
            "effective_rank_session_a": effective_rank(embeddings[0]),
            "family_accuracy": family_accuracy,
            "intensity_accuracy": intensity_accuracy,
            "stratum_accuracy": stratum_accuracy,
            "family_centroid_cosine_a_b": centroid_agreement(
                embeddings[0], cohorts[0][3], embeddings[1], cohorts[1][3]
            ),
            "family_centroid_cosine_a_c": centroid_agreement(
                embeddings[0], cohorts[0][3], embeddings[2], cohorts[2][3]
            ),
            "view": view_metrics(model, view_batch, args.device),
            "checkpoint": prior["checkpoint"],
            "checkpoint_sha256": prior["checkpoint_sha256"],
        })

    summary = {}
    for objective in OBJECTIVES:
        group = [run for run in runs if run["objective"] == objective]
        summary[objective] = {
            key: statistics.median(values)
            for key, values in {
                "effective_rank": [run["effective_rank_session_a"] for run in group],
                "family_accuracy_b": [run["family_accuracy"]["session_b"] for run in group],
                "family_accuracy_c": [run["family_accuracy"]["session_c"] for run in group],
                "intensity_accuracy_b": [run["intensity_accuracy"]["session_b"] for run in group],
                "intensity_accuracy_c": [run["intensity_accuracy"]["session_c"] for run in group],
                "stratum_accuracy_b": [run["stratum_accuracy"]["session_b"] for run in group],
                "stratum_accuracy_c": [run["stratum_accuracy"]["session_c"] for run in group],
                "view_identity_top1": [run["view"]["identity_top1"] for run in group],
                "view_paired_cosine": [run["view"]["mean_paired_cosine"] for run in group],
            }.items()
        }
    report = {
        "schema": SCHEMA,
        "scope": "benign-only frozen latent-geometry diagnostic",
        "claim_boundary": "diagnostic only; no effect labels and no objective promotion",
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": args.device,
        "r3_report_sha256": R3_REPORT_SHA256,
        "view_seed": VIEW_SEED,
        "view_rows": view_batch.batch_size,
        "view_identifiers_sha256": hashlib.sha256(
            "\n".join(view_identifiers).encode()
        ).hexdigest(),
        "chance": {"family": 1 / 17, "intensity": 1 / 3, "stratum": 1 / 51,
                   "view_identity_top1": 1 / view_batch.batch_size},
        "source_sha256": sha256(Path(__file__)),
        "summary_medians": summary,
        "runs": runs,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
