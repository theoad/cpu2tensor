# SPDX-License-Identifier: AGPL-3.0-only
"""Compare label-free representation objectives on retained hardware tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from cpu2tensor.examples.hardware_effect_anomaly_r1 import PAIRS, family_batch
from cpu2tensor.examples.hardware_fewshot_transfer_r1 import (
    PILOT_REPORT_SHA256,
    concatenate,
)
from cpu2tensor.examples.hardware_futex_transfer_r1 import sha256
from cpu2tensor.examples.hardware_multimodal import (
    HardwareMultimodalBatch,
    MaskedHardwareModel,
    MaskedTokens,
    MultimodalConfig,
    freeze_multimodal_model,
    make_training_masks,
    masked_reconstruction_loss,
)
from cpu2tensor.examples.hardware_multimodal_experiment import load_dataset
from cpu2tensor.examples.hardware_scaling_pilot_r1 import (
    EFFECT_MANIFESTS,
    balanced_training_rows,
    validation_loss,
)
from cpu2tensor.examples.hardware_compute_scaling_r2 import (
    replicated_transfer_probe,
)


SCHEMA = "cpu2tensor-hardware-objective-ablation-r3"
OBJECTIVES = ("reconstruction", "contrastive", "vicreg")
PRETRAINING_SEEDS = (3901, 3902, 3903)
MODEL_WIDTH = 128
TRAINING_ROWS = 2040
OPTIMIZER_STEPS = 320
BATCH_SIZE = 32
CONTRASTIVE_WEIGHT = 0.03
CONTRASTIVE_TEMPERATURE = 0.2
VICREG_WEIGHT = 0.01
VICREG_INVARIANCE = 25.0
VICREG_VARIANCE = 25.0
VICREG_COVARIANCE = 1.0


def masked_pool(
    model: MaskedHardwareModel,
    batch: HardwareMultimodalBatch,
    masked: MaskedTokens,
) -> torch.Tensor:
    tokens = model.encode(batch, masked)
    available = batch.token_availability.to(tokens.dtype)[..., None]
    return (tokens * available).sum((1, 2)) / available.sum((1, 2)).clamp_min(1)


def contrastive_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    temperature: float = CONTRASTIVE_TEMPERATURE,
) -> torch.Tensor:
    """Symmetric in-batch instance discrimination for two masked views."""
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 2:
        raise ValueError("contrastive views must be matching 2D batches")
    left = F.normalize(left, dim=1)
    right = F.normalize(right, dim=1)
    logits = left @ right.T / temperature
    labels = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """VICReg-style invariance, variance, and covariance regularization."""
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 2:
        raise ValueError("VICReg views must be matching 2D batches")
    invariance = F.mse_loss(left, right)
    left_centered = left - left.mean(0)
    right_centered = right - right.mean(0)
    left_std = torch.sqrt(left_centered.var(0, unbiased=False) + 1e-4)
    right_std = torch.sqrt(right_centered.var(0, unbiased=False) + 1e-4)
    variance = 0.5 * (
        F.relu(1.0 - left_std).mean() + F.relu(1.0 - right_std).mean()
    )
    denominator = left.shape[0] - 1
    left_covariance = left_centered.T @ left_centered / denominator
    right_covariance = right_centered.T @ right_centered / denominator
    covariance = (
        _off_diagonal(left_covariance).square().sum()
        + _off_diagonal(right_covariance).square().sum()
    ) / left.shape[1]
    return (
        VICREG_INVARIANCE * invariance
        + VICREG_VARIANCE * variance
        + VICREG_COVARIANCE * covariance
    )


def train_objective(
    model: MaskedHardwareModel,
    training: HardwareMultimodalBatch,
    objective: str,
    seed: int,
) -> tuple[list[float], list[float], float]:
    if objective not in OBJECTIVES:
        raise ValueError("unknown objective")
    if not bool(model.normalization_fitted):
        model.fit_normalization(training)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-2
    )
    generator = torch.Generator().manual_seed(seed)
    losses = []
    auxiliary_losses = []
    if training.pt.device.type == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    for _ in range(OPTIMIZER_STEPS):
        indices = torch.randperm(training.batch_size, generator=generator)[:BATCH_SIZE]
        batch = training.index_select(indices)
        left_masks = make_training_masks(batch, generator=generator)
        right_masks = make_training_masks(batch, generator=generator)
        reconstruction = 0.5 * (
            masked_reconstruction_loss(model, batch, left_masks)
            + masked_reconstruction_loss(model, batch, right_masks)
        )
        left = masked_pool(model, batch, left_masks)
        right = masked_pool(model, batch, right_masks)
        if objective == "reconstruction":
            auxiliary = left.square().mean() * 0.0 + right.square().mean() * 0.0
            loss = reconstruction
        elif objective == "contrastive":
            auxiliary = contrastive_loss(left, right)
            loss = reconstruction + CONTRASTIVE_WEIGHT * auxiliary
        else:
            auxiliary = vicreg_loss(left, right)
            loss = reconstruction + VICREG_WEIGHT * auxiliary
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        auxiliary_losses.append(float(auxiliary.detach()))
    if training.pt.device.type == "mps":
        torch.mps.synchronize()
    return losses, auxiliary_losses, time.perf_counter() - started


def objective_decision(runs: list[dict[str, object]]) -> dict[str, object]:
    """Apply the preregistered paired transfer and grammar-retention gate."""
    by_objective = {
        objective: sorted(
            (run for run in runs if run["objective"] == objective),
            key=lambda run: int(run["pretraining_seed"]),
        )
        for objective in OBJECTIVES
    }
    if any(len(group) != len(PRETRAINING_SEEDS) for group in by_objective.values()):
        raise ValueError("objective matrix is incomplete")
    baseline = by_objective["reconstruction"]
    baseline_auc = statistics.median(
        float(run["transfer"]["median_read_auroc"]) for run in baseline
    )
    baseline_loss = statistics.median(
        float(run["heldout_masked_loss"]) for run in baseline
    )
    candidates = {}
    for objective in OBJECTIVES[1:]:
        group = by_objective[objective]
        median_auc = statistics.median(
            float(run["transfer"]["median_read_auroc"]) for run in group
        )
        median_loss = statistics.median(
            float(run["heldout_masked_loss"]) for run in group
        )
        paired_wins = sum(
            float(candidate["transfer"]["median_read_auroc"])
            > float(control["transfer"]["median_read_auroc"])
            for candidate, control in zip(group, baseline, strict=True)
        )
        candidates[objective] = {
            "median_read_auroc": median_auc,
            "median_heldout_masked_loss": median_loss,
            "paired_transfer_wins": paired_wins,
            "keeps_grammar_loss": median_loss <= 1.10 * baseline_loss,
            "passes": (
                median_auc >= 0.75
                and median_auc >= baseline_auc + 0.10
                and paired_wins >= 2
                and median_loss <= 1.10 * baseline_loss
            ),
        }
    promoted = [name for name, result in candidates.items() if result["passes"]]
    return {
        "baseline_median_read_auroc": baseline_auc,
        "baseline_median_heldout_masked_loss": baseline_loss,
        "candidates": candidates,
        "promoted_objectives": promoted,
        "passes_preregistered_gate": bool(promoted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_artifact", type=Path)
    parser.add_argument("pilot_a", type=Path)
    parser.add_argument("pilot_b", type=Path)
    parser.add_argument("pilot_c", type=Path)
    parser.add_argument("effect_a", type=Path)
    parser.add_argument("effect_b", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output directory")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")

    pilot_path = args.pilot_artifact / "report.json"
    if sha256(pilot_path) != PILOT_REPORT_SHA256:
        raise ValueError("pilot artifact differs")
    pilot = json.loads(pilot_path.read_text())
    roots = (args.pilot_a, args.pilot_b, args.pilot_c)
    if {
        name: sha256(root / "capture-manifest.json")
        for name, root in zip(("a", "b", "c"), roots, strict=True)
    } != pilot["manifests"]:
        raise ValueError("pilot manifests differ")
    if {
        "effect_a": sha256(args.effect_a / "capture-manifest.json"),
        "effect_b": sha256(args.effect_b / "capture-manifest.json"),
    } != EFFECT_MANIFESTS:
        raise ValueError("effect manifests differ")

    training, identifiers = balanced_training_rows(
        (args.pilot_a, args.pilot_b), TRAINING_ROWS
    )
    training = training.to(args.device)
    _, validation_rows = load_dataset(args.pilot_c)
    validation = concatenate([validation_rows[key] for key in sorted(validation_rows)])
    training_pairs = {
        pair: family_batch(args.effect_a, PAIRS[pair])
        for pair in ("fstat", "openat")
    }
    read_evaluation = family_batch(args.effect_b, PAIRS["read"])

    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    runs = []
    for objective in OBJECTIVES:
        for seed in PRETRAINING_SEEDS:
            torch.manual_seed(seed)
            config = MultimodalConfig(
                pebs_features=140,
                pmu_features=4,
                model_dimensions=MODEL_WIDTH,
                attention_heads=4,
                feedforward_dimensions=2 * MODEL_WIDTH,
                local_layers=2,
                cross_cpu_layers=2,
            )
            model = MaskedHardwareModel(config).to(args.device)
            losses, auxiliary_losses, training_seconds = train_objective(
                model, training, objective, seed
            )
            if args.device == "mps":
                torch.mps.synchronize()
            heldout_loss = validation_loss(model, validation, args.device)
            freeze_multimodal_model(model)
            transfer = replicated_transfer_probe(
                model, training_pairs, read_evaluation, args.device
            )
            checkpoint = args.output / f"{objective}-seed{seed}.pt"
            torch.save({
                "schema": SCHEMA,
                "objective": objective,
                "seed": seed,
                "config": config.__dict__,
                "state_dict": model.state_dict(),
            }, checkpoint)
            runs.append({
                "objective": objective,
                "pretraining_seed": seed,
                "training_identifiers_sha256": sha256_identifiers(identifiers),
                "first_training_loss": losses[0],
                "final_training_loss": losses[-1],
                "first_auxiliary_loss": auxiliary_losses[0],
                "final_auxiliary_loss": auxiliary_losses[-1],
                "heldout_masked_loss": heldout_loss,
                "training_seconds": training_seconds,
                "transfer": transfer,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256(checkpoint),
            })

    report = {
        "schema": SCHEMA,
        "scope": "label-free objective ablation on retained hardware tensors",
        "claim_boundary": "development objective selection, not vulnerability detection",
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": args.device,
        "objectives": list(OBJECTIVES),
        "pretraining_seeds": list(PRETRAINING_SEEDS),
        "model_width": MODEL_WIDTH,
        "training_rows": TRAINING_ROWS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "batch_size": BATCH_SIZE,
        "weights": {
            "contrastive": CONTRASTIVE_WEIGHT,
            "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
            "vicreg": VICREG_WEIGHT,
            "vicreg_invariance": VICREG_INVARIANCE,
            "vicreg_variance": VICREG_VARIANCE,
            "vicreg_covariance": VICREG_COVARIANCE,
        },
        "decision": objective_decision(runs),
        "source_sha256": sha256(Path(__file__)),
        "pilot_report_sha256": PILOT_REPORT_SHA256,
        "manifests": {
            name: sha256(root / "capture-manifest.json")
            for name, root in zip(("pilot_a", "pilot_b", "pilot_c"), roots, strict=True)
        } | {
            "effect_a": sha256(args.effect_a / "capture-manifest.json"),
            "effect_b": sha256(args.effect_b / "capture-manifest.json"),
        },
        "runs": runs,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(encoded)
    print(encoded, end="")


def sha256_identifiers(identifiers: list[str]) -> str:
    return hashlib.sha256("\n".join(identifiers).encode()).hexdigest()


if __name__ == "__main__":
    main()
