# SPDX-License-Identifier: AGPL-3.0-only
"""Run the bounded v4 futex effect/transfer diagnostic.

This is a lawful sensor proxy, not vulnerability validation.  Session A defines
the effect direction and trains every head; session B contributes no fitted
statistics, gradients, or model selection.  The script never captures hardware
data or executes a workload.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch import nn

from cpu2tensor.examples.hardware_multimodal import (
    HardwareMultimodalBatch,
    MaskedHardwareModel,
    MaskedTokens,
    MultimodalConfig,
    freeze_multimodal_model,
    train_masked_model,
)
from cpu2tensor.examples.hardware_multimodal_experiment import load_dataset


SCHEMA = "cpu2tensor-hardware-futex-transfer-r1"
EFFECT_FAMILIES = ("futex_mismatch", "futex_wake")
MODALITIES = ("pt", "pebs", "pmu", "timing", "fused")
SEEDS = (2702, 2703, 2704)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combine(rows: list[HardwareMultimodalBatch]) -> HardwareMultimodalBatch:
    if not rows:
        raise ValueError("cannot combine an empty row set")
    return HardwareMultimodalBatch(*(
        torch.cat([getattr(row, field.name) for row in rows])
        for field in fields(HardwareMultimodalBatch)
    ))


def load_rows(root: Path, families: set[str] | None = None) -> tuple[
    dict[str, object], HardwareMultimodalBatch, torch.Tensor, list[str]
]:
    manifest, rows = load_dataset(root)
    entries = {entry["execution_id"]: entry for entry in manifest["entries"]}
    if len(entries) != len(manifest["entries"]):
        raise ValueError("manifest repeats an execution identity")
    identifiers = sorted(rows)
    if families is not None:
        identifiers = [
            identifier for identifier in identifiers
            if entries[identifier]["family"] in families
        ]
    labels = torch.tensor([
        float(entries[identifier]["family"] == "futex_wake")
        for identifier in identifiers
    ])
    return manifest, combine([rows[identifier] for identifier in identifiers]), labels, identifiers


def validate_capture(manifest: dict[str, object], expected_rows: int) -> None:
    entries = manifest["entries"]
    if len(entries) != expected_rows or manifest["collection"]["executions"] != expected_rows:
        raise ValueError("capture row count differs from the preregistered plan")
    if manifest["collection"]["loss_count"] or manifest["collection"]["admission"]["rejected_attempts"]:
        raise ValueError("capture has loss or rejected attempts")
    for entry in entries:
        capture = entry["capture"]
        if (entry["admission"]["attempt"] != 1 or not entry["raw_retained"] or
                any(capture[name] for name in (
                    "lost_sources", "missing_sources", "multiplexed_sources"
                ))):
            raise ValueError("capture entry is incomplete or retried")


def ensure_independent(manifests: list[dict[str, object]], roots: list[Path]) -> None:
    identity = ("subject_identity_sha256", "event_identity_sha256", "feature_schema")
    if any(manifest[field] != manifests[0][field]
           for manifest in manifests[1:] for field in identity):
        raise ValueError("datasets do not share one subject/event/feature identity")
    manifest_hashes = [sha256(root / "capture-manifest.json") for root in roots]
    if len(set(manifest_hashes)) != len(manifest_hashes):
        raise ValueError("datasets reuse a capture manifest")
    raw_sets = [{entry["raw_sha256"] for entry in manifest["entries"]}
                for manifest in manifests]
    for index, left in enumerate(raw_sets):
        if any(left & right for right in raw_sets[index + 1:]):
            raise ValueError("datasets reuse one or more raw captures")


def vector(batch: HardwareMultimodalBatch, modality: str) -> torch.Tensor:
    if modality == "pt":
        values = (batch.pt, batch.pt_available.to(torch.float32))
    elif modality == "pebs":
        values = (batch.pebs, batch.pebs_available.to(torch.float32))
    elif modality == "pmu":
        values = (batch.pmu, batch.pmu_available.to(torch.float32))
    elif modality == "timing":
        values = (torch.nan_to_num(batch.time_bounds), torch.nan_to_num(batch.timing_quality))
    elif modality == "fused":
        return torch.cat([vector(batch, name) for name in MODALITIES[:-1]], dim=1)
    else:
        raise ValueError(f"unknown modality: {modality}")
    return torch.cat([value.flatten(1) for value in values], dim=1)


def auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    positive, negative = scores[labels == 1], scores[labels == 0]
    comparisons = (positive[:, None] > negative[None, :]).to(torch.float32)
    ties = (positive[:, None] == negative[None, :]).to(torch.float32)
    return float((comparisons + 0.5 * ties).mean())


def direction_probe(
    training: HardwareMultimodalBatch,
    training_labels: torch.Tensor,
    evaluation: HardwareMultimodalBatch,
    evaluation_labels: torch.Tensor,
) -> dict[str, object]:
    report = {}
    for modality in MODALITIES:
        left, right = vector(training, modality), vector(evaluation, modality)
        mean, scale = left.mean(0), left.std(0, correction=0)
        keep = scale > 1e-8
        left = (left[:, keep] - mean[keep]) / scale[keep]
        right = (right[:, keep] - mean[keep]) / scale[keep]
        direction = left[training_labels == 1].mean(0) - left[training_labels == 0].mean(0)
        evaluation_direction = (
            right[evaluation_labels == 1].mean(0) - right[evaluation_labels == 0].mean(0)
        )
        training_scores, evaluation_scores = left @ direction, right @ direction
        cosine = 0.0
        if direction.norm() and evaluation_direction.norm():
            cosine = float(torch.nn.functional.cosine_similarity(
                direction[None], evaluation_direction[None]
            ))
        report[modality] = {
            "coordinates": int(keep.sum()),
            "training_auroc": auc(training_labels, training_scores),
            "evaluation_auroc": auc(evaluation_labels, evaluation_scores),
            "effect_direction_cosine": cosine,
        }
    return report


def empty_masks(batch: HardwareMultimodalBatch) -> MaskedTokens:
    return MaskedTokens(
        torch.zeros_like(batch.pt_available),
        torch.zeros_like(batch.pebs_available),
        torch.zeros_like(batch.pmu_available),
    )


def pooled(model: MaskedHardwareModel, batch: HardwareMultimodalBatch) -> torch.Tensor:
    tokens = model.encode(batch, empty_masks(batch))
    available = batch.token_availability.to(tokens.dtype)[..., None]
    return (tokens * available).sum((1, 2)) / available.sum((1, 2)).clamp_min(1)


def copy_normalization(source: MaskedHardwareModel, destination: MaskedHardwareModel) -> None:
    for name in (
        "pt_mean", "pt_scale", "pebs_mean", "pebs_scale", "pmu_mean", "pmu_scale",
        "normalization_fitted",
    ):
        getattr(destination, name).copy_(getattr(source, name))


def transfer(
    pretrained: MaskedHardwareModel,
    training: HardwareMultimodalBatch,
    training_labels: torch.Tensor,
    evaluation: HardwareMultimodalBatch,
    evaluation_labels: torch.Tensor,
    device: str,
    output: Path,
) -> list[dict[str, object]]:
    config, state = pretrained.config, copy.deepcopy(pretrained.state_dict())
    results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        shared_head = copy.deepcopy(nn.Linear(config.model_dimensions, 1).state_dict())
        for arm in ("scratch", "frozen", "fine_tuned"):
            torch.manual_seed(seed)
            model = MaskedHardwareModel(config).to(device)
            if arm == "scratch":
                copy_normalization(pretrained, model)
            else:
                model.load_state_dict(state)
            head = nn.Linear(config.model_dimensions, 1).to(device)
            head.load_state_dict(shared_head)
            if arm == "frozen":
                freeze_multimodal_model(model)
            else:
                model.train()
                for parameter in model.parameters():
                    parameter.requires_grad_(True)
            groups = [{"params": head.parameters(), "lr": 3e-3}]
            if arm != "frozen":
                groups.append({"params": model.parameters(), "lr": 1e-4})
            optimizer = torch.optim.AdamW(groups, weight_decay=1e-3)
            row_generator = torch.Generator().manual_seed(seed)
            device_training = training.to(device)
            device_labels = training_labels.to(device)
            started = time.monotonic()
            first_loss = final_loss = 0.0
            for step in range(80):
                indices = torch.randperm(
                    training_labels.numel(), generator=row_generator
                )[:24].to(device)
                batch = device_training.index_select(indices)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    head(pooled(model, batch)).squeeze(-1), device_labels[indices]
                )
                loss.backward()
                optimizer.step()
                if step == 0:
                    first_loss = float(loss.detach())
                final_loss = float(loss.detach())
            model.eval()
            head.eval()
            with torch.no_grad():
                training_scores = head(pooled(model, training.to(device))).squeeze(-1).cpu()
                evaluation_scores = head(pooled(model, evaluation.to(device))).squeeze(-1).cpu()
            checkpoint = output / f"transfer-seed{seed}-{arm}.pt"
            torch.save({
                "schema": SCHEMA,
                "arm": arm,
                "seed": seed,
                "config": config.__dict__,
                "model_state_dict": model.state_dict(),
                "head_state_dict": head.state_dict(),
            }, checkpoint)
            results.append({
                "seed": seed,
                "arm": arm,
                "first_loss": first_loss,
                "final_loss": final_loss,
                "seconds": time.monotonic() - started,
                "training_auroc": auc(training_labels, training_scores),
                "evaluation_auroc": auc(evaluation_labels, evaluation_scores),
                "evaluation_cross_class_wins": int((
                    evaluation_scores[evaluation_labels == 1][:, None] >
                    evaluation_scores[evaluation_labels == 0][None, :]
                ).sum()),
                "training_scores": training_scores.tolist(),
                "evaluation_scores": evaluation_scores.tolist(),
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256(checkpoint),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anchor_session", type=Path)
    parser.add_argument("effect_session_a", type=Path)
    parser.add_argument("effect_session_b", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output directory")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)

    anchor_manifest, anchor, _, _ = load_rows(args.anchor_session)
    manifest_a, training, training_labels, _ = load_rows(
        args.effect_session_a, set(EFFECT_FAMILIES)
    )
    manifest_b, evaluation, evaluation_labels, _ = load_rows(
        args.effect_session_b, set(EFFECT_FAMILIES)
    )
    validate_capture(anchor_manifest, 51)
    validate_capture(manifest_a, 48)
    validate_capture(manifest_b, 48)
    ensure_independent(
        [anchor_manifest, manifest_a, manifest_b],
        [args.anchor_session, args.effect_session_a, args.effect_session_b],
    )
    if (training_labels.numel() != 32 or evaluation_labels.numel() != 32 or
            training_labels.sum() != 16 or evaluation_labels.sum() != 16):
        raise ValueError("effect sessions are not balanced 16-by-16")

    config = MultimodalConfig(
        pebs_features=140,
        pmu_features=4,
        model_dimensions=128,
        attention_heads=4,
        feedforward_dimensions=256,
        local_layers=2,
        cross_cpu_layers=2,
    )
    torch.manual_seed(2701)
    model = MaskedHardwareModel(config).to(args.device)
    started = time.monotonic()
    losses = train_masked_model(
        model, anchor.to(args.device), steps=60, batch_size=32, seed=2701
    )
    pretraining_seconds = time.monotonic() - started
    results = transfer(
        model, training, training_labels, evaluation, evaluation_labels,
        args.device, args.output,
    )

    checkpoint = args.output / "pretrained-v4-d128.pt"
    torch.save({"schema": SCHEMA, "config": config.__dict__, "state_dict": model.state_dict()}, checkpoint)
    report = {
        "schema": SCHEMA,
        "scope": "lawful futex kernel-path sensor and transfer proxy",
        "vulnerability_sensitivity": False,
        "operational_go": False,
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": args.device,
        "source_sha256": sha256(Path(__file__)),
        "manifests": {
            name: sha256(path / "capture-manifest.json") for name, path in (
                ("anchor", args.anchor_session),
                ("effect_session_a", args.effect_session_a),
                ("effect_session_b", args.effect_session_b),
            )
        },
        "subject_identity_sha256": manifest_a["subject_identity_sha256"],
        "event_identity_sha256": manifest_a["event_identity_sha256"],
        "feature_schema": manifest_a["feature_schema"],
        "direction_probe": direction_probe(
            training, training_labels, evaluation, evaluation_labels
        ),
        "pretraining": {
            "rows": anchor.batch_size,
            "steps": 60,
            "seed": 2701,
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "seconds": pretraining_seconds,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint),
        },
        "transfer": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
