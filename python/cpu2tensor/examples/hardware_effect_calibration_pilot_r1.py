# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate the frozen effect detector on 3,060 fresh benign executions."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import platform

import torch

from cpu2tensor.examples.hardware_effect_confirmation_r1 import (
    DEVELOPMENT_REPORT_SHA256,
    EXPECTED_SEEDS,
    PMU_MODALITY_INDEX,
    conjunctive_gate,
    raw_hashes,
    selected_batch,
    validate_exact_plan_rows,
    validate_family_counts,
)
from cpu2tensor.examples.hardware_futex_transfer_r1 import ensure_independent, sha256
from cpu2tensor.examples.hardware_multimodal import (
    HardwareMultimodalBatch,
    MaskedHardwareModel,
    MultimodalConfig,
    freeze_multimodal_model,
    multimodal_anomaly_evidence,
)
from cpu2tensor.examples.hardware_multimodal_experiment import (
    WORKLOAD_LOOPS,
    load_dataset,
)


SCHEMA = "cpu2tensor-hardware-effect-calibration-pilot-r1"
CONFIRMATION_REPORT_SHA256 = "5142e6ec6c96c4861b16e0845c7d7bd6385cd96e02bbde0ac9eb8e78c26f07a8"
COHORTS = (
    ("a", 2026092710, "03f674f56bb86fa8d6f3e7d92916a3db2a3eb0e25759bd4dc865bdb62e4757fd"),
    ("b", 2026092711, "1016866829c322a349a394c96082d070e6b651e4de62464b314a89214ed28ba5"),
    ("c", 2026092712, "1cec0d7ee65edbd5e80870255a00e2a714b5c7dd22e8115651d8e5c4e3750264"),
)
ROWS_PER_COHORT = 1020
RAW_PER_COHORT = 204
SCORE_CHUNK = 64


def validate_capture_quality(manifest: dict[str, object]) -> None:
    entries = manifest["entries"]
    if (
        len(entries) != ROWS_PER_COHORT
        or manifest["collection"]["executions"] != ROWS_PER_COHORT
        or manifest["collection"]["loss_count"]
        or manifest["collection"]["admission"]["rejected_attempts"]
        or sum(bool(entry["raw_retained"]) for entry in entries) != RAW_PER_COHORT
    ):
        raise ValueError("calibration cohort is incomplete")
    for entry in entries:
        capture = entry["capture"]
        if entry["admission"]["attempt"] != 1 or any(
            capture[name]
            for name in ("lost_sources", "missing_sources", "multiplexed_sources")
        ):
            raise ValueError("calibration cohort has a retry or invalid source")


def validate_cohort_protocol(
    root: Path,
    manifest: dict[str, object],
    cohort_seed: int,
    plan_sha256: str,
) -> None:
    plan_path = root / "execution-plan.json"
    if not plan_path.is_file() or sha256(plan_path) != plan_sha256:
        raise ValueError("calibration execution plan differs from preregistration")
    plan = json.loads(plan_path.read_text())
    split = manifest["split"]
    if split.get("explicit_plan") != {
        "cohort_seed": cohort_seed,
        "kind": "session-a",
        "sha256": plan_sha256,
    } or split.get("loop_divisors") != [1, 2, 5]:
        raise ValueError("calibration manifest does not declare the frozen plan")
    validate_family_counts(manifest, {family: 60 for family in WORKLOAD_LOOPS})
    validate_exact_plan_rows(plan, manifest)


def slice_batch(
    batch: HardwareMultimodalBatch, start: int, end: int
) -> HardwareMultimodalBatch:
    return HardwareMultimodalBatch(*(
        getattr(batch, field.name)[start:end]
        for field in fields(HardwareMultimodalBatch)
    ))


def score_batch(
    model: MaskedHardwareModel,
    batch: HardwareMultimodalBatch,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    fused = []
    pmu = []
    with torch.no_grad():
        for start in range(0, batch.batch_size, SCORE_CHUNK):
            evidence = multimodal_anomaly_evidence(
                model, slice_batch(batch, start, start + SCORE_CHUNK).to(device)
            )
            fused.append(evidence.score.cpu())
            pmu.append(evidence.modality_scores[:, PMU_MODALITY_INDEX].cpu())
    return torch.cat(fused), torch.cat(pmu)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_artifact", type=Path)
    parser.add_argument("confirmation_artifact", type=Path)
    parser.add_argument("cohort_a", type=Path)
    parser.add_argument("cohort_b", type=Path)
    parser.add_argument("cohort_c", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prior-root", action="append", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output directory")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    development_path = args.development_artifact / "report.json"
    confirmation_path = args.confirmation_artifact / "report.json"
    if sha256(development_path) != DEVELOPMENT_REPORT_SHA256:
        raise ValueError("development report differs from the frozen input")
    if sha256(confirmation_path) != CONFIRMATION_REPORT_SHA256:
        raise ValueError("confirmation report differs from the frozen input")
    development = json.loads(development_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    if tuple(run["seed"] for run in development["runs"]) != EXPECTED_SEEDS:
        raise ValueError("development checkpoints differ from the frozen seeds")
    if not confirmation["confirmation_pass"]:
        raise ValueError("frozen confirmation did not pass")

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
        raise ValueError("development scoring or loader source differs")
    confirmation_source = Path(__file__).with_name(
        "hardware_effect_confirmation_r1.py"
    )
    if sha256(confirmation_source) != confirmation["source_sha256"]:
        raise ValueError("confirmed gate implementation differs")

    expected_prior_hashes = set(development["manifests"].values()) | {
        confirmation["manifests"]["fresh_benign"],
        confirmation["manifests"]["fresh_effect"],
    }
    prior_manifests = []
    observed_prior_hashes = set()
    for root in args.prior_root:
        observed_prior_hashes.add(sha256(root / "capture-manifest.json"))
        manifest, _ = load_dataset(root)
        prior_manifests.append(manifest)
    if observed_prior_hashes != expected_prior_hashes:
        raise ValueError("prior capture set differs from preregistration")

    roots = (args.cohort_a, args.cohort_b, args.cohort_c)
    manifests = []
    batches = []
    identifiers = []
    for root, (name, cohort_seed, plan_hash) in zip(roots, COHORTS, strict=True):
        manifest, batch, labels, row_ids = selected_batch(root)
        if labels.sum():
            raise ValueError("calibration cohort contains an effect label")
        validate_capture_quality(manifest)
        validate_cohort_protocol(root, manifest, cohort_seed, plan_hash)
        manifests.append(manifest)
        batches.append(batch)
        identifiers.append([f"{name}/{identifier}" for identifier in row_ids])
    ensure_independent(
        prior_manifests + manifests,
        list(args.prior_root) + list(roots),
    )
    prior_raw = set().union(*(raw_hashes(manifest) for manifest in prior_manifests))
    calibration_raw = set().union(*(raw_hashes(manifest) for manifest in manifests))
    if prior_raw & calibration_raw:
        raise ValueError("calibration reuses a prior raw capture")

    torch.set_num_threads(4)
    runs = []
    for expected, declared, confirmed in zip(
        EXPECTED_SEEDS, development["runs"], confirmation["runs"], strict=True
    ):
        if confirmed["seed"] != expected:
            raise ValueError("confirmation threshold order differs")
        checkpoint = args.development_artifact / declared["checkpoint"]
        if sha256(checkpoint) != declared["checkpoint_sha256"]:
            raise ValueError("development checkpoint hash differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = MaskedHardwareModel(MultimodalConfig(**payload["config"])).to(
            args.device
        )
        model.load_state_dict(payload["state_dict"])
        freeze_multimodal_model(model)
        fused_parts = []
        pmu_parts = []
        flat_ids = []
        for batch, cohort_ids in zip(batches, identifiers, strict=True):
            fused, pmu = score_batch(model, batch, args.device)
            fused_parts.append(fused)
            pmu_parts.append(pmu)
            flat_ids.extend(cohort_ids)
        fused = torch.cat(fused_parts)
        pmu = torch.cat(pmu_parts)
        fused_threshold = float(confirmed["fused_training_maximum"])
        pmu_threshold = float(confirmed["pmu_training_maximum"])
        gate = conjunctive_gate(fused, pmu, fused_threshold, pmu_threshold)
        joint_margin = torch.minimum(
            fused - fused_threshold, pmu - pmu_threshold
        )
        order = joint_margin.argsort(descending=True)[:100].tolist()
        runs.append({
            "seed": expected,
            "fused_training_maximum": fused_threshold,
            "pmu_training_maximum": pmu_threshold,
            "alerts": int(gate.sum()),
            "empirical_alert_rate": float(gate.float().mean()),
            "alert_identifiers": [
                flat_ids[index] for index in gate.nonzero().flatten().tolist()
            ],
            "top_100": [{
                "identifier": flat_ids[index],
                "joint_margin": float(joint_margin[index]),
                "fused": float(fused[index]),
                "pmu": float(pmu[index]),
            } for index in order],
        })
    pilot_pass = all(run["alerts"] == 0 for run in runs)
    zero_alert_upper_95 = 1.0 - 0.05 ** (1.0 / (3 * ROWS_PER_COHORT))
    report = {
        "schema": SCHEMA,
        "scope": "fresh benign-only false-positive calibration pilot",
        "gate": "confirmed fused > training maximum AND PMU > training maximum",
        "acceptance": "all three frozen models produce 0/3060 alerts",
        "pilot_pass": pilot_pass,
        "operational_go": False,
        "vulnerability_sensitivity": False,
        "rows_per_model": 3 * ROWS_PER_COHORT,
        "zero_alert_one_sided_95_upper_bound": zero_alert_upper_95,
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": args.device,
        "development_report_sha256": DEVELOPMENT_REPORT_SHA256,
        "confirmation_report_sha256": CONFIRMATION_REPORT_SHA256,
        "source_sha256": sha256(Path(__file__)),
        "manifests": {
            name: sha256(root / "capture-manifest.json")
            for root, (name, _, _) in zip(roots, COHORTS, strict=True)
        },
        "plans": {name: plan_hash for name, _, plan_hash in COHORTS},
        "runs": runs,
    }
    args.output.mkdir(parents=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
