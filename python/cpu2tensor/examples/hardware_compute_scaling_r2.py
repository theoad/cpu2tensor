# SPDX-License-Identifier: AGPL-3.0-only
"""Cross data volume with optimizer compute for hardware pretraining."""
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

from cpu2tensor.examples.hardware_effect_anomaly_r1 import PAIRS, family_batch
from cpu2tensor.examples.hardware_fewshot_transfer_r1 import (
    PILOT_REPORT_SHA256,
    concatenate,
    deterministic_examples,
    score,
)
from cpu2tensor.examples.hardware_futex_transfer_r1 import auc, pooled, sha256
from cpu2tensor.examples.hardware_multimodal import (
    HardwareMultimodalBatch,
    MaskedHardwareModel,
    MultimodalConfig,
    freeze_multimodal_model,
    train_masked_model,
)
from cpu2tensor.examples.hardware_multimodal_experiment import load_dataset
from cpu2tensor.examples.hardware_scaling_pilot_r1 import (
    EFFECT_MANIFESTS,
    balanced_training_rows,
    encoder_throughput,
    local_slope,
    validation_loss,
)


SCHEMA = "cpu2tensor-hardware-compute-scaling-r2"
MODEL_WIDTHS = (128, 256)
DATA_ROWS = (255, 1020, 2040)
OPTIMIZER_STEPS = (80, 160, 320)
HEAD_SEEDS = (2801, 2802, 2803)
PRETRAINING_SEED = 3901
BATCH_SIZE = 32
TRANSFER_STEPS = 80


def replicated_transfer_probe(
    model: MaskedHardwareModel,
    training_pairs: dict[str, tuple],
    evaluation: tuple,
    device: str,
) -> dict[str, object]:
    """Measure ranking transfer with fixed, independent labeled-row seeds."""
    runs = []
    for seed in HEAD_SEEDS:
        selected_batches = []
        selected_labels = []
        selected_ids = []
        for pair_name in ("fstat", "openat"):
            batch, labels, identifiers = training_pairs[pair_name]
            indices = deterministic_examples(identifiers, labels, 2, seed)
            selected_batches.append(batch.index_select(indices))
            selected_labels.append(labels[indices])
            selected_ids.extend(
                f"{pair_name}/{identifiers[index]}" for index in indices.tolist()
            )
        training = concatenate(selected_batches).to(device)
        labels = torch.cat(selected_labels).to(device)
        torch.manual_seed(seed)
        head = nn.Linear(model.config.model_dimensions, 1).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=3e-3, weight_decay=1e-3
        )
        generator = torch.Generator().manual_seed(seed)
        losses = []
        for _ in range(TRANSFER_STEPS):
            indices = torch.randint(
                0, labels.numel(), (labels.numel(),), generator=generator
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                head(pooled(model, training.index_select(indices))).squeeze(-1),
                labels[indices],
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        batch, evaluation_labels, identifiers = evaluation
        scores = score(model, head, batch, device)
        runs.append({
            "seed": seed,
            "training_identifiers": selected_ids,
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "read_auroc": auc(evaluation_labels, scores),
            "evaluation_identifiers": identifiers,
            "evaluation_labels": evaluation_labels.tolist(),
            "evaluation_scores": scores.tolist(),
        })
    return {
        "shots_per_class_per_training_pair": 2,
        "runs": runs,
        "median_read_auroc": statistics.median(
            run["read_auroc"] for run in runs
        ),
    }


def scaling_decision(runs: list[dict[str, object]]) -> dict[str, object]:
    """Apply the preregistered compute and data scaling gates."""
    trajectories = []
    for width in MODEL_WIDTHS:
        for rows in DATA_ROWS:
            points = sorted(
                (
                    int(run["optimizer_steps"]),
                    float(run["heldout_masked_loss"]),
                )
                for run in runs
                if run["model_dimensions"] == width
                and run["training_rows"] == rows
            )
            if len(points) != len(OPTIMIZER_STEPS):
                raise ValueError("compute trajectory is incomplete")
            trajectories.append({
                "model_dimensions": width,
                "training_rows": rows,
                "loss_80": points[0][1],
                "loss_320": points[-1][1],
                "log_slope": local_slope(points),
                "improved": points[-1][1] < points[0][1],
            })
    compute_improvements = sum(item["improved"] for item in trajectories)

    data_slopes = {}
    relative_improvements = {}
    for width in MODEL_WIDTHS:
        points = sorted(
            (
                int(run["training_rows"]),
                float(run["heldout_masked_loss"]),
            )
            for run in runs
            if run["model_dimensions"] == width
            and run["optimizer_steps"] == max(OPTIMIZER_STEPS)
        )
        if len(points) != len(DATA_ROWS):
            raise ValueError("largest-compute data trajectory is incomplete")
        data_slopes[str(width)] = local_slope(points)
        relative_improvements[str(width)] = (points[0][1] - points[-1][1]) / points[0][1]

    compute_scaling = compute_improvements >= 5
    data_scaling = (
        all(slope < 0 for slope in data_slopes.values())
        and max(relative_improvements.values()) >= 0.02
    )
    return {
        "compute_scaling": compute_scaling,
        "compute_improvements": compute_improvements,
        "compute_trajectories": trajectories,
        "data_scaling_emerges": data_scaling,
        "largest_compute_data_log_slopes": data_slopes,
        "largest_compute_relative_improvements": relative_improvements,
        "passes_preregistered_gate": compute_scaling and data_scaling,
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

    _, validation_rows = load_dataset(args.pilot_c)
    validation = concatenate([validation_rows[key] for key in sorted(validation_rows)])
    training_sets = {
        count: balanced_training_rows((args.pilot_a, args.pilot_b), count)
        for count in DATA_ROWS
    }
    training_pairs = {
        pair: family_batch(args.effect_a, PAIRS[pair])
        for pair in ("fstat", "openat")
    }
    read_evaluation = family_batch(args.effect_b, PAIRS["read"])

    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    runs = []
    for width in MODEL_WIDTHS:
        for row_count in DATA_ROWS:
            for steps in OPTIMIZER_STEPS:
                torch.manual_seed(PRETRAINING_SEED)
                config = MultimodalConfig(
                    pebs_features=140,
                    pmu_features=4,
                    model_dimensions=width,
                    attention_heads=4 if width == 128 else 8,
                    feedforward_dimensions=2 * width,
                    local_layers=2,
                    cross_cpu_layers=2,
                )
                model = MaskedHardwareModel(config).to(args.device)
                training, identifiers = training_sets[row_count]
                started = time.perf_counter()
                losses = train_masked_model(
                    model,
                    training.to(args.device),
                    steps=steps,
                    batch_size=BATCH_SIZE,
                    seed=PRETRAINING_SEED,
                )
                if args.device == "mps":
                    torch.mps.synchronize()
                train_seconds = time.perf_counter() - started
                heldout_loss = validation_loss(model, validation, args.device)
                freeze_multimodal_model(model)
                throughput = encoder_throughput(model, validation, args.device)
                transfer = replicated_transfer_probe(
                    model, training_pairs, read_evaluation, args.device
                )
                checkpoint = args.output / (
                    f"d{width}-n{row_count}-s{steps}-seed{PRETRAINING_SEED}.pt"
                )
                torch.save({
                    "schema": SCHEMA,
                    "seed": PRETRAINING_SEED,
                    "training_rows": row_count,
                    "optimizer_steps": steps,
                    "config": config.__dict__,
                    "state_dict": model.state_dict(),
                }, checkpoint)
                parameters = sum(
                    parameter.numel() for parameter in model.parameters()
                )
                runs.append({
                    "model_dimensions": width,
                    "training_rows": row_count,
                    "optimizer_steps": steps,
                    "optimizer_examples": steps * BATCH_SIZE,
                    "parameters": parameters,
                    "parameter_examples": parameters * steps * BATCH_SIZE,
                    "training_identifiers_sha256": hashlib.sha256(
                        "\n".join(identifiers).encode()
                    ).hexdigest(),
                    "first_training_loss": losses[0],
                    "final_training_loss": losses[-1],
                    "heldout_masked_loss": heldout_loss,
                    "training_seconds": train_seconds,
                    "encoder_rows_per_second": throughput,
                    "transfer": transfer,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": sha256(checkpoint),
                })

    decision = scaling_decision(runs)
    report = {
        "schema": SCHEMA,
        "scope": "local data-by-optimizer-compute scaling pilot",
        "claim_boundary": "development curves, not universal scaling laws",
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": args.device,
        "pretraining_seed": PRETRAINING_SEED,
        "head_seeds": list(HEAD_SEEDS),
        "batch_size": BATCH_SIZE,
        "model_widths": list(MODEL_WIDTHS),
        "data_rows": list(DATA_ROWS),
        "optimizer_steps": list(OPTIMIZER_STEPS),
        "decision": decision,
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


if __name__ == "__main__":
    main()
