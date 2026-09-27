# SPDX-License-Identifier: AGPL-3.0-only
"""Compare scratch and pretrained few-shot transfer on lawful kernel effects."""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from torch import nn

from cpu2tensor.examples.hardware_effect_anomaly_r1 import PAIRS, family_batch
from cpu2tensor.examples.hardware_effect_confirmation_r1 import (
    DEVELOPMENT_REPORT_SHA256,
    EXPECTED_SEEDS,
    selected_batch,
)
from cpu2tensor.examples.hardware_futex_transfer_r1 import (
    auc,
    copy_normalization,
    empty_masks,
    ensure_independent,
    pooled,
    sha256,
)
from cpu2tensor.examples.hardware_multimodal import (
    HardwareMultimodalBatch,
    MaskedHardwareModel,
    MultimodalConfig,
    freeze_multimodal_model,
)
from cpu2tensor.examples.hardware_multimodal_experiment import load_dataset


SCHEMA = "cpu2tensor-hardware-fewshot-transfer-r1"
PILOT_REPORT_SHA256 = "176546fc014182ec1c95cebeb4cbaadb4d24a52070fc628f00ee4c649711bdde"
SHOTS = (1, 2, 4, 8, 16)
ARMS = ("scratch", "frozen", "fine_tuned")
TRAINING_PAIRS = ("fstat", "openat")
EVALUATION_PAIRS = ("fstat", "openat", "read")
STEPS = 80
SCORE_CHUNK = 64


def deterministic_examples(
    identifiers: list[str],
    labels: torch.Tensor,
    shots: int,
    seed: int,
) -> torch.Tensor:
    selected = []
    for label in (0, 1):
        candidates = [
            index for index, value in enumerate(labels.tolist()) if value == label
        ]
        candidates.sort(key=lambda index: hashlib.sha256(
            f"fewshot-r1\0{seed}\0{identifiers[index]}".encode()
        ).digest())
        if len(candidates) < shots:
            raise ValueError("not enough examples for the requested shot count")
        selected.extend(candidates[:shots])
    return torch.tensor(selected, dtype=torch.long)


def concatenate(batches: list[HardwareMultimodalBatch]) -> HardwareMultimodalBatch:
    return HardwareMultimodalBatch(*(
        torch.cat([getattr(batch, field.name) for batch in batches])
        for field in fields(HardwareMultimodalBatch)
    ))


def score(
    model: MaskedHardwareModel,
    head: nn.Linear,
    batch: HardwareMultimodalBatch,
    device: str,
) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, batch.batch_size, SCORE_CHUNK):
            indices = torch.arange(
                start, min(start + SCORE_CHUNK, batch.batch_size)
            )
            part = batch.index_select(indices).to(device)
            values.append(head(pooled(model, part)).squeeze(-1).cpu())
    return torch.cat(values)


def build_arm(
    payload: dict[str, object], arm: str, seed: int
) -> tuple[MaskedHardwareModel, nn.Linear]:
    if arm not in ARMS:
        raise ValueError("unknown transfer arm")
    torch.manual_seed(seed)
    pretrained = MaskedHardwareModel(MultimodalConfig(**payload["config"]))
    shared_head = copy.deepcopy(
        nn.Linear(pretrained.config.model_dimensions, 1).state_dict()
    )
    torch.manual_seed(seed)
    model = MaskedHardwareModel(pretrained.config)
    pretrained.load_state_dict(payload["state_dict"])
    if arm == "scratch":
        copy_normalization(pretrained, model)
    else:
        model.load_state_dict(payload["state_dict"])
    head = nn.Linear(pretrained.config.model_dimensions, 1)
    head.load_state_dict(shared_head)
    return model, head


def train_arm(
    payload: dict[str, object],
    arm: str,
    seed: int,
    training: HardwareMultimodalBatch,
    labels: torch.Tensor,
    device: str,
) -> tuple[MaskedHardwareModel, nn.Linear, list[float], float]:
    model, head = build_arm(payload, arm, seed)
    model = model.to(device)
    head = head.to(device)
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
    device_training = training.to(device)
    device_labels = labels.to(device)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    started = time.monotonic()
    for _ in range(STEPS):
        indices = torch.randint(
            0, labels.numel(), (min(24, labels.numel()),), generator=generator
        ).to(device)
        batch = device_training.index_select(indices)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(
            head(pooled(model, batch)).squeeze(-1), device_labels[indices]
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    head.eval()
    return model, head, losses, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_artifact", type=Path)
    parser.add_argument("pilot_artifact", type=Path)
    parser.add_argument("effect_session_a", type=Path)
    parser.add_argument("effect_session_b", type=Path)
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
    development_path = args.development_artifact / "report.json"
    pilot_path = args.pilot_artifact / "report.json"
    if sha256(development_path) != DEVELOPMENT_REPORT_SHA256:
        raise ValueError("development artifact differs")
    if sha256(pilot_path) != PILOT_REPORT_SHA256:
        raise ValueError("pilot artifact differs")
    development = json.loads(development_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    if tuple(run["seed"] for run in development["runs"]) != EXPECTED_SEEDS:
        raise ValueError("pretrained checkpoint seeds differ")
    source_paths = {
        "hardware.py": Path(__file__).parent.parent / "hardware.py",
        "hardware_effect_anomaly_r1.py": Path(__file__).with_name(
            "hardware_effect_anomaly_r1.py"
        ),
        "hardware_futex_transfer_r1.py": Path(__file__).with_name(
            "hardware_futex_transfer_r1.py"
        ),
        "hardware_multimodal.py": Path(__file__).with_name("hardware_multimodal.py"),
        "hardware_multimodal_experiment.py": Path(__file__).with_name(
            "hardware_multimodal_experiment.py"
        ),
        "hardware_multimodal_features.py": Path(__file__).with_name(
            "hardware_multimodal_features.py"
        ),
    }
    if {name: sha256(path) for name, path in source_paths.items()} != development[
        "source_hashes"
    ]:
        raise ValueError("pretraining model or feature source differs")

    roots = [
        args.effect_session_a, args.effect_session_b,
        args.pilot_a, args.pilot_b, args.pilot_c,
    ]
    manifests = [load_dataset(root)[0] for root in roots]
    ensure_independent(manifests, roots)
    if {
        "effect_a": sha256(args.effect_session_a / "capture-manifest.json"),
        "effect_b": sha256(args.effect_session_b / "capture-manifest.json"),
    } != {
        "effect_a": development["manifests"]["effect_a"],
        "effect_b": development["manifests"]["effect_b"],
    }:
        raise ValueError("effect manifests differ from pretraining evaluation")
    if {
        name: sha256(root / "capture-manifest.json")
        for name, root in zip(("a", "b", "c"), roots[2:], strict=True)
    } != pilot["manifests"]:
        raise ValueError("pilot manifests differ")

    training_pairs = {}
    evaluation_pairs = {}
    for pair_name in EVALUATION_PAIRS:
        training_pairs[pair_name] = family_batch(
            args.effect_session_a, PAIRS[pair_name]
        )
        evaluation_pairs[pair_name] = family_batch(
            args.effect_session_b, PAIRS[pair_name]
        )
    _, pilot_a, _, _ = selected_batch(args.pilot_a)
    _, pilot_b, _, _ = selected_batch(args.pilot_b)
    _, pilot_c, _, _ = selected_batch(args.pilot_c)
    benign_evaluation = concatenate([pilot_b, pilot_c])

    torch.set_num_threads(4)
    runs = []
    args.output.mkdir(parents=True)
    for declared in development["runs"]:
        pretrained_seed = declared["seed"]
        checkpoint = args.development_artifact / declared["checkpoint"]
        if sha256(checkpoint) != declared["checkpoint_sha256"]:
            raise ValueError("pretrained checkpoint differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for shots in SHOTS:
            selected_batches = []
            selected_labels = []
            training_ids = []
            for pair_name in TRAINING_PAIRS:
                batch, labels, identifiers = training_pairs[pair_name]
                indices = deterministic_examples(
                    identifiers, labels, shots, pretrained_seed
                )
                selected_batches.append(batch.index_select(indices))
                selected_labels.append(labels[indices])
                training_ids.extend(
                    f"{pair_name}/{identifiers[index]}" for index in indices.tolist()
                )
            training = concatenate(selected_batches)
            labels = torch.cat(selected_labels)
            for arm in ARMS:
                model, head, losses, seconds = train_arm(
                    payload, arm, pretrained_seed, training, labels, args.device
                )
                calibration_scores = score(model, head, pilot_a, args.device)
                threshold = float(calibration_scores.max())
                benign_scores = score(
                    model, head, benign_evaluation, args.device
                )
                pair_results = {}
                for pair_name in EVALUATION_PAIRS:
                    batch, pair_labels, identifiers = evaluation_pairs[pair_name]
                    scores = score(model, head, batch, args.device)
                    pair_results[pair_name] = {
                        "auroc": auc(pair_labels, scores),
                        "effect_alerts": int(
                            (scores[pair_labels == 1] > threshold).sum()
                        ),
                        "control_alerts": int(
                            (scores[pair_labels == 0] > threshold).sum()
                        ),
                        "identifiers": identifiers,
                        "labels": pair_labels.tolist(),
                        "scores": scores.tolist(),
                    }
                checkpoint_path = args.output / (
                    f"seed{pretrained_seed}-k{shots}-{arm}.pt"
                )
                torch.save({
                    "schema": SCHEMA,
                    "seed": pretrained_seed,
                    "shots": shots,
                    "arm": arm,
                    "config": payload["config"],
                    "model_state_dict": model.state_dict(),
                    "head_state_dict": head.state_dict(),
                }, checkpoint_path)
                runs.append({
                    "seed": pretrained_seed,
                    "shots_per_class_per_pair": shots,
                    "arm": arm,
                    "training_identifiers": training_ids,
                    "first_loss": losses[0],
                    "final_loss": losses[-1],
                    "seconds": seconds,
                    "calibration_threshold": threshold,
                    "benign_rows": benign_scores.numel(),
                    "benign_alerts": int((benign_scores > threshold).sum()),
                    "benign_alert_rate": float(
                        (benign_scores > threshold).float().mean()
                    ),
                    "pairs": pair_results,
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_sha256": sha256(checkpoint_path),
                })

    low_shot = [run for run in runs if run["shots_per_class_per_pair"] <= 4]
    advantages = []
    for shots in (1, 2, 4):
        for seed in EXPECTED_SEEDS:
            scratch = next(
                run for run in low_shot
                if run["shots_per_class_per_pair"] == shots
                and run["seed"] == seed and run["arm"] == "scratch"
            )
            for arm in ("frozen", "fine_tuned"):
                candidate = next(
                    run for run in low_shot
                    if run["shots_per_class_per_pair"] == shots
                    and run["seed"] == seed and run["arm"] == arm
                )
                advantages.append({
                    "shots": shots,
                    "seed": seed,
                    "arm": arm,
                    "read_auroc_advantage": (
                        candidate["pairs"]["read"]["auroc"]
                        - scratch["pairs"]["read"]["auroc"]
                    ),
                    "benign_alert_delta": (
                        candidate["benign_alerts"] - scratch["benign_alerts"]
                    ),
                })
    advantage_summary = []
    for shots in (1, 2, 4):
        for arm in ("frozen", "fine_tuned"):
            rows = [
                item for item in advantages
                if item["shots"] == shots and item["arm"] == arm
            ]
            median_auroc = statistics.median(
                item["read_auroc_advantage"] for item in rows
            )
            wins = sum(item["read_auroc_advantage"] > 0 for item in rows)
            median_alert_delta = statistics.median(
                item["benign_alert_delta"] for item in rows
            )
            advantage_summary.append({
                "shots": shots,
                "arm": arm,
                "median_read_auroc_advantage": median_auroc,
                "paired_seed_wins": wins,
                "median_benign_alert_delta": median_alert_delta,
                "success": (
                    median_auroc >= 0.10 and wins >= 2
                    and median_alert_delta <= 0
                ),
            })
    report = {
        "schema": SCHEMA,
        "scope": "paired few-shot representation-transfer diagnosis",
        "claim_boundary": (
            "lawful effect proxy only; no vulnerability or operational claim"
        ),
        "success": (
            "at k<=4, a pretrained arm has median read AUROC advantage >=0.10, "
            "wins at least 2/3 paired seeds, and does not increase median benign alerts"
        ),
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": args.device,
        "steps": STEPS,
        "shots": list(SHOTS),
        "arms": list(ARMS),
        "source_sha256": sha256(Path(__file__)),
        "development_report_sha256": DEVELOPMENT_REPORT_SHA256,
        "pilot_report_sha256": PILOT_REPORT_SHA256,
        "manifests": {
            name: sha256(root / "capture-manifest.json")
            for name, root in zip(
                ("effect_a", "effect_b", "pilot_a", "pilot_b", "pilot_c"),
                roots,
                strict=True,
            )
        },
        "advantages": advantages,
        "advantage_summary": advantage_summary,
        "pretraining_advantage": any(
            item["success"] for item in advantage_summary
        ),
        "runs": runs,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
