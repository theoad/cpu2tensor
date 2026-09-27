# SPDX-License-Identifier: AGPL-3.0-only
"""Run a bounded capacity-by-data hardware-pretraining scaling pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
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
    make_training_masks,
    masked_reconstruction_loss,
    train_masked_model,
)
from cpu2tensor.examples.hardware_multimodal_experiment import load_dataset


SCHEMA = "cpu2tensor-hardware-scaling-pilot-r1"
MODEL_WIDTHS = (64, 128, 256)
DATA_ROWS = (255, 1020, 2040)
SEED = 3901
STEPS = 80
BATCH_SIZE = 32
VALIDATION_CHUNK = 64
TRANSFER_STEPS = 80
EFFECT_MANIFESTS = {
    "effect_a": "a0f65a27d8253fe9fe2d8f6d948f740d2bebde655bd2b66aa39fabaa363376d3",
    "effect_b": "c0e24dfaa84ca500bc538616bc522950823448e8b2323c685d9bb82ff922008e",
}


def balanced_training_rows(
    roots: tuple[Path, Path], count: int
) -> tuple[HardwareMultimodalBatch, list[str]]:
    if count % 51 or count not in DATA_ROWS:
        raise ValueError("training row count must be a preregistered balanced budget")
    candidates: dict[tuple[str, int], list[tuple[str, HardwareMultimodalBatch]]] = {}
    for session, root in zip(("a", "b"), roots, strict=True):
        manifest, rows = load_dataset(root)
        entries = {entry["execution_id"]: entry for entry in manifest["entries"]}
        for identifier, row in rows.items():
            entry = entries[identifier]
            intensity = int(identifier.split("-i", 1)[1].split("-s", 1)[0])
            candidates.setdefault((entry["family"], intensity), []).append(
                (f"{session}/{identifier}", row)
            )
    quota = count // 51
    selected = []
    for key in sorted(candidates):
        rows = candidates[key]
        rows.sort(key=lambda item: hashlib.sha256(
            f"scaling-pilot-r1\0{SEED}\0{item[0]}".encode()
        ).digest())
        if len(rows) < quota:
            raise ValueError("not enough rows in one family/intensity stratum")
        selected.extend(rows[:quota])
    selected.sort(key=lambda item: item[0])
    if len(selected) != count:
        raise ValueError("balanced selection has the wrong row count")
    return concatenate([row for _, row in selected]), [name for name, _ in selected]


def validation_loss(
    model: MaskedHardwareModel,
    validation: HardwareMultimodalBatch,
    device: str,
) -> float:
    generator = torch.Generator().manual_seed(99173)
    total = 0.0
    rows = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, validation.batch_size, VALIDATION_CHUNK):
            indices = torch.arange(
                start, min(start + VALIDATION_CHUNK, validation.batch_size)
            )
            batch = validation.index_select(indices).to(device)
            masks = make_training_masks(batch, generator=generator)
            count = batch.batch_size
            total += float(masked_reconstruction_loss(model, batch, masks)) * count
            rows += count
    return total / rows


def transfer_probe(
    model: MaskedHardwareModel,
    training_pairs: dict[str, tuple],
    evaluation: tuple,
    device: str,
) -> dict[str, object]:
    selected_batches = []
    selected_labels = []
    selected_ids = []
    for pair_name in ("fstat", "openat"):
        batch, labels, identifiers = training_pairs[pair_name]
        indices = deterministic_examples(identifiers, labels, 2, SEED)
        selected_batches.append(batch.index_select(indices))
        selected_labels.append(labels[indices])
        selected_ids.extend(
            f"{pair_name}/{identifiers[index]}" for index in indices.tolist()
        )
    training = concatenate(selected_batches).to(device)
    labels = torch.cat(selected_labels).to(device)
    torch.manual_seed(SEED)
    head = nn.Linear(model.config.model_dimensions, 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(SEED)
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
    return {
        "training_identifiers": selected_ids,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "read_auroc": auc(evaluation_labels, scores),
        "evaluation_identifiers": identifiers,
        "evaluation_labels": evaluation_labels.tolist(),
        "evaluation_scores": scores.tolist(),
    }


def encoder_throughput(
    model: MaskedHardwareModel,
    validation: HardwareMultimodalBatch,
    device: str,
) -> float:
    def synchronize() -> None:
        if device == "mps":
            torch.mps.synchronize()

    indices = torch.arange(min(VALIDATION_CHUNK, validation.batch_size))
    with torch.no_grad():
        pooled(model, validation.index_select(indices).to(device))
        synchronize()
        started = time.perf_counter()
        rows = 0
        for start in range(0, validation.batch_size, VALIDATION_CHUNK):
            indices = torch.arange(
                start, min(start + VALIDATION_CHUNK, validation.batch_size)
            )
            pooled(model, validation.index_select(indices).to(device))
            rows += indices.numel()
        synchronize()
    return rows / (time.perf_counter() - started)


def local_slope(points: list[tuple[int, float]]) -> float:
    x = [math.log(float(value)) for value, _ in points]
    y = [math.log(loss) for _, loss in points]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    return sum(
        (left - x_mean) * (right - y_mean) for left, right in zip(x, y)
    ) / denominator


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
            torch.manual_seed(SEED)
            config = MultimodalConfig(
                pebs_features=140,
                pmu_features=4,
                model_dimensions=width,
                attention_heads=4 if width <= 128 else 8,
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
                steps=STEPS,
                batch_size=BATCH_SIZE,
                seed=SEED,
            )
            if args.device == "mps":
                torch.mps.synchronize()
            train_seconds = time.perf_counter() - started
            heldout_loss = validation_loss(model, validation, args.device)
            freeze_multimodal_model(model)
            throughput = encoder_throughput(model, validation, args.device)
            transfer = transfer_probe(
                model, training_pairs, read_evaluation, args.device
            )
            checkpoint = args.output / f"d{width}-n{row_count}-seed{SEED}.pt"
            torch.save({
                "schema": SCHEMA,
                "seed": SEED,
                "training_rows": row_count,
                "config": config.__dict__,
                "state_dict": model.state_dict(),
            }, checkpoint)
            parameters = sum(parameter.numel() for parameter in model.parameters())
            runs.append({
                "model_dimensions": width,
                "training_rows": row_count,
                "parameters": parameters,
                "parameter_examples": parameters * STEPS * BATCH_SIZE,
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

    data_slopes = {
        str(width): local_slope([
            (run["training_rows"], run["heldout_masked_loss"])
            for run in runs if run["model_dimensions"] == width
        ]) for width in MODEL_WIDTHS
    }
    capacity_slopes = {
        str(rows): local_slope([
            (run["parameters"], run["heldout_masked_loss"])
            for run in runs if run["training_rows"] == rows
        ]) for rows in DATA_ROWS
    }
    coherent_data = sum(slope < 0 for slope in data_slopes.values()) >= 2
    low = next(
        run for run in runs
        if run["model_dimensions"] == 64 and run["training_rows"] == 2040
    )
    high = next(
        run for run in runs
        if run["model_dimensions"] == 256 and run["training_rows"] == 2040
    )
    report = {
        "schema": SCHEMA,
        "scope": "local fixed-compute capacity-by-data scaling pilot",
        "claim_boundary": "development curves, not universal scaling laws",
        "coherent_scaling": (
            coherent_data
            and high["heldout_masked_loss"] < low["heldout_masked_loss"]
        ),
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": args.device,
        "seed": SEED,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "model_widths": list(MODEL_WIDTHS),
        "data_rows": list(DATA_ROWS),
        "data_log_slopes": data_slopes,
        "capacity_log_slopes": capacity_slopes,
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
