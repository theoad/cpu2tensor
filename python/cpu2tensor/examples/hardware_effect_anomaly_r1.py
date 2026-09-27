# SPDX-License-Identifier: AGPL-3.0-only
"""Score lawful held-out kernel error paths with benign-only v4 pretraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import torch

from cpu2tensor.examples.hardware_futex_transfer_r1 import (
    auc,
    combine,
    ensure_independent,
    sha256,
    validate_capture,
)
from cpu2tensor.examples.hardware_multimodal import (
    MaskedHardwareModel,
    MultimodalConfig,
    freeze_multimodal_model,
    multimodal_anomaly_score,
    train_masked_model,
)
from cpu2tensor.examples.hardware_multimodal_experiment import load_dataset


SCHEMA = "cpu2tensor-hardware-effect-anomaly-r1"
PAIRS = {
    "fstat": ("fstat_ok", "fstat_badfd"),
    "openat": ("openat_ok", "openat_missing"),
    "read": ("read_copy", "read_efault"),
}
SEEDS = (2801, 2802, 2803)


def family_batch(root: Path, pair: tuple[str, str]):
    manifest, rows = load_dataset(root)
    entries = {entry["execution_id"]: entry for entry in manifest["entries"]}
    identifiers = sorted(
        identifier for identifier in rows if entries[identifier]["family"] in pair
    )
    labels = torch.tensor([
        float(entries[identifier]["family"] == pair[1]) for identifier in identifiers
    ])
    if labels.numel() != 32 or labels.sum() != 16:
        raise ValueError("effect pair is not balanced 16-by-16")
    return combine([rows[identifier] for identifier in identifiers]), labels, identifiers


def anchor_batch(root: Path):
    manifest, rows = load_dataset(root)
    identifiers = sorted(rows)
    return manifest, combine([rows[identifier] for identifier in identifiers]), identifiers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anchor_session_a", type=Path)
    parser.add_argument("anchor_session_b", type=Path)
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

    anchor_a_manifest, anchor_a, anchor_a_ids = anchor_batch(args.anchor_session_a)
    anchor_b_manifest, anchor_b, anchor_b_ids = anchor_batch(args.anchor_session_b)
    effect_a_manifest, _ = load_dataset(args.effect_session_a)
    effect_b_manifest, _ = load_dataset(args.effect_session_b)
    validate_capture(anchor_a_manifest, 51)
    validate_capture(anchor_b_manifest, 51)
    validate_capture(effect_a_manifest, 96)
    validate_capture(effect_b_manifest, 96)
    roots = [
        args.anchor_session_a, args.anchor_session_b,
        args.effect_session_a, args.effect_session_b,
    ]
    manifests = [
        anchor_a_manifest, anchor_b_manifest, effect_a_manifest, effect_b_manifest,
    ]
    ensure_independent(manifests, roots)

    effect_batches = {
        (pair_name, session_name): family_batch(root, pair)
        for pair_name, pair in PAIRS.items()
        for session_name, root in (
            ("session_a", args.effect_session_a),
            ("session_b", args.effect_session_b),
        )
    }
    config = MultimodalConfig(
        pebs_features=140,
        pmu_features=4,
        model_dimensions=128,
        attention_heads=4,
        feedforward_dimensions=256,
        local_layers=2,
        cross_cpu_layers=2,
    )
    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = MaskedHardwareModel(config).to(args.device)
        losses = train_masked_model(
            model, anchor_a.to(args.device), steps=60, batch_size=32, seed=seed
        )
        freeze_multimodal_model(model)
        with torch.no_grad():
            calibration_scores = multimodal_anomaly_score(
                model, anchor_a.to(args.device)
            ).cpu()
            benign_scores = multimodal_anomaly_score(
                model, anchor_b.to(args.device)
            ).cpu()
        threshold = float(calibration_scores.max())
        pair_reports = {}
        for pair_name in PAIRS:
            session_reports = {}
            for session_name in ("session_a", "session_b"):
                batch, labels, identifiers = effect_batches[(pair_name, session_name)]
                with torch.no_grad():
                    scores = multimodal_anomaly_score(
                        model, batch.to(args.device)
                    ).cpu()
                session_reports[session_name] = {
                    "auroc": auc(labels, scores),
                    "effect_alerts": int((scores[labels == 1] > threshold).sum()),
                    "control_alerts": int((scores[labels == 0] > threshold).sum()),
                    "identifiers": identifiers,
                    "labels": labels.tolist(),
                    "scores": scores.tolist(),
                }
            pair_reports[pair_name] = session_reports
        checkpoint = args.output / f"pretrained-seed{seed}.pt"
        torch.save({
            "schema": SCHEMA,
            "seed": seed,
            "config": config.__dict__,
            "state_dict": model.state_dict(),
        }, checkpoint)
        runs.append({
            "seed": seed,
            "pretraining_first_loss": losses[0],
            "pretraining_final_loss": losses[-1],
            "calibration_threshold": threshold,
            "calibration_identifiers": anchor_a_ids,
            "calibration_scores": calibration_scores.tolist(),
            "benign_identifiers": anchor_b_ids,
            "benign_scores": benign_scores.tolist(),
            "benign_alerts": int((benign_scores > threshold).sum()),
            "benign_rows": benign_scores.numel(),
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint),
            "pairs": pair_reports,
        })

    report = {
        "schema": SCHEMA,
        "scope": "benign-only pretraining against lawful held-out kernel error paths",
        "vulnerability_sensitivity": False,
        "operational_go": False,
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": args.device,
        "source_hashes": {
            path.name: sha256(path) for path in (
                Path(__file__),
                Path(__file__).with_name("hardware_futex_transfer_r1.py"),
                Path(__file__).with_name("hardware_multimodal.py"),
                Path(__file__).with_name("hardware_multimodal_experiment.py"),
                Path(__file__).with_name("hardware_multimodal_features.py"),
                Path(__file__).parent.parent / "hardware.py",
            )
        },
        "manifests": {
            name: sha256(root / "capture-manifest.json")
            for name, root in zip(
                ("anchor_a", "anchor_b", "effect_a", "effect_b"), roots, strict=True
            )
        },
        "subject_identity_sha256": effect_a_manifest["subject_identity_sha256"],
        "event_identity_sha256": effect_a_manifest["event_identity_sha256"],
        "feature_schema": effect_a_manifest["feature_schema"],
        "threshold": (
            "training/resubstitution maximum anomaly score among the same 51 "
            "anchor-session-A rows used to fit normalization and model weights"
        ),
        "runs": runs,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
