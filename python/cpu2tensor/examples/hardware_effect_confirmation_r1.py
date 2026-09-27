# SPDX-License-Identifier: AGPL-3.0-only
"""Confirm the preregistered fused-plus-PMU anomaly gate on fresh captures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import torch

from cpu2tensor.examples.hardware_futex_transfer_r1 import (
    combine,
    ensure_independent,
    sha256,
    validate_capture,
)
from cpu2tensor.examples.hardware_multimodal import (
    MaskedHardwareModel,
    MultimodalConfig,
    freeze_multimodal_model,
    multimodal_anomaly_evidence,
)
from cpu2tensor.examples.hardware_multimodal_experiment import load_dataset
from cpu2tensor.examples.hardware_multimodal_experiment import WORKLOAD_LOOPS


SCHEMA = "cpu2tensor-hardware-effect-confirmation-r1"
DEVELOPMENT_REPORT_SHA256 = "2d0efd99db73c03562b4a196c4054c8f03b5edfab473acbd3123a1ac437da04c"
READ_FAMILIES = ("read_copy", "read_efault")
EXPECTED_SEEDS = (2801, 2802, 2803)
PMU_MODALITY_INDEX = 2
BENIGN_PLAN_SHA256 = "d76c443b7221a429c4dd63e41b862c56a554f822e7b392d135bf6827205bd679"
BENIGN_COHORT_SEED = 2026092706
EFFECT_COHORT_SEED = 2026092705


def selected_batch(root: Path, families: set[str] | None = None):
    manifest, rows = load_dataset(root)
    entries = {entry["execution_id"]: entry for entry in manifest["entries"]}
    identifiers = sorted(rows)
    if families is not None:
        identifiers = [
            identifier for identifier in identifiers
            if entries[identifier]["family"] in families
        ]
    labels = torch.tensor([
        float(entries[identifier]["family"] == "read_efault")
        for identifier in identifiers
    ])
    return manifest, combine([rows[identifier] for identifier in identifiers]), labels, identifiers


def raw_hashes(manifest: dict[str, object]) -> set[str]:
    hashes = {entry["raw_sha256"] for entry in manifest["entries"]}
    if len(hashes) != len(manifest["entries"]):
        raise ValueError("manifest reuses a raw capture")
    return hashes


def validate_family_counts(
    manifest: dict[str, object], expected: dict[str, int]
) -> None:
    observed: dict[str, int] = {}
    for entry in manifest["entries"]:
        family = entry["family"]
        observed[family] = observed.get(family, 0) + 1
    if observed != expected:
        raise ValueError("fresh cohort differs from the preregistered family counts")


def validate_exact_plan_rows(
    plan: dict[str, object], manifest: dict[str, object]
) -> None:
    planned = {
        row["execution_id"]: (
            row["family"], row["loops"], row["repetition"], row["partition"],
            row["input_seed"], row["retain_raw"],
        )
        for row in plan["rows"]
    }
    observed = {
        entry["execution_id"]: (
            entry["family"], entry["loops"], entry["repetition"],
            entry["partition"], entry.get("invocation", {}).get("input_seed"),
            entry.get("raw_retained"),
        )
        for entry in manifest["entries"]
    }
    if observed != planned:
        raise ValueError("fresh benign rows differ from the frozen execution plan")


def validate_confirmation_protocol(
    benign_root: Path,
    benign_manifest: dict[str, object],
    effect_manifest: dict[str, object],
) -> None:
    plan_path = benign_root / "execution-plan.json"
    if not plan_path.is_file() or sha256(plan_path) != BENIGN_PLAN_SHA256:
        raise ValueError("fresh benign execution plan differs from preregistration")
    plan = json.loads(plan_path.read_text())
    benign_split = benign_manifest["split"]
    if benign_split.get("explicit_plan") != {
        "cohort_seed": BENIGN_COHORT_SEED,
        "kind": "smoke",
        "sha256": BENIGN_PLAN_SHA256,
    }:
        raise ValueError("fresh benign manifest does not declare the frozen plan")
    if (
        benign_split.get("loop_divisors") != [1, 2, 5]
        or benign_split.get("vary_input_seed") is not True
        or any(
            entry.get("invocation", {}).get("input_seed") is None
            or entry.get("raw_retained") is not True
            for entry in benign_manifest["entries"]
        )
    ):
        raise ValueError("fresh benign acquisition protocol differs")
    benign_loops: dict[str, set[int]] = {}
    for entry in benign_manifest["entries"]:
        benign_loops.setdefault(entry["family"], set()).add(entry["loops"])
    expected_loops = {
        family: {base, base // 2, base // 5}
        for family, base in WORKLOAD_LOOPS.items()
    }
    if benign_loops != expected_loops:
        raise ValueError("fresh benign intensity schedule differs")
    validate_exact_plan_rows(plan, benign_manifest)

    effect_split = effect_manifest["split"]
    if (
        effect_split.get("explicit_plan") is not None
        or effect_split.get("loop_divisors") != [1]
        or effect_split.get("vary_input_seed") is not True
        or effect_split.get("repetitions") != 16
        or effect_split.get("seed") != EFFECT_COHORT_SEED
        or any(
            entry.get("loops") != 5000
            or entry.get("invocation", {}).get("input_seed") is None
            or entry.get("raw_retained") is not True
            for entry in effect_manifest["entries"]
        )
    ):
        raise ValueError("fresh effect acquisition protocol differs")


def conjunctive_gate(
    fused: torch.Tensor,
    pmu: torch.Tensor,
    fused_threshold: float,
    pmu_threshold: float,
) -> torch.Tensor:
    if fused.shape != pmu.shape or fused.ndim != 1:
        raise ValueError("gate inputs must be aligned one-dimensional scores")
    return (fused > fused_threshold) & (pmu > pmu_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_anchor", type=Path)
    parser.add_argument("development_anchor_b", type=Path)
    parser.add_argument("development_effect_a", type=Path)
    parser.add_argument("development_effect_b", type=Path)
    parser.add_argument("development_artifact", type=Path)
    parser.add_argument("fresh_benign", type=Path)
    parser.add_argument("fresh_effect", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output directory")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if sha256(args.development_artifact / "report.json") != DEVELOPMENT_REPORT_SHA256:
        raise ValueError("development report differs from the preregistered input")
    development = json.loads((args.development_artifact / "report.json").read_text())
    if tuple(run["seed"] for run in development["runs"]) != EXPECTED_SEEDS:
        raise ValueError("development checkpoints differ from the preregistered seeds")

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

    development_roots = [
        args.training_anchor, args.development_anchor_b,
        args.development_effect_a, args.development_effect_b,
    ]
    development_names = ("anchor_a", "anchor_b", "effect_a", "effect_b")
    development_manifests = []
    for name, root in zip(development_names, development_roots, strict=True):
        manifest, _ = load_dataset(root)
        if sha256(root / "capture-manifest.json") != development["manifests"][name]:
            raise ValueError(f"development {name} manifest differs")
        development_manifests.append(manifest)

    training_manifest, training, _, training_ids = selected_batch(args.training_anchor)
    benign_manifest, benign, benign_labels, benign_ids = selected_batch(args.fresh_benign)
    effect_manifest, effect, effect_labels, effect_ids = selected_batch(
        args.fresh_effect, set(READ_FAMILIES)
    )
    validate_capture(training_manifest, 51)
    validate_capture(benign_manifest, 51)
    validate_capture(effect_manifest, 32)
    ensure_independent(
        [training_manifest, benign_manifest, effect_manifest],
        [args.training_anchor, args.fresh_benign, args.fresh_effect],
    )
    if benign_labels.sum() or effect_labels.numel() != 32 or effect_labels.sum() != 16:
        raise ValueError("fresh confirmation cohorts differ from the frozen plan")
    validate_family_counts(benign_manifest, {family: 3 for family in WORKLOAD_LOOPS})
    validate_family_counts(effect_manifest, {"read_copy": 16, "read_efault": 16})
    validate_confirmation_protocol(
        args.fresh_benign, benign_manifest, effect_manifest
    )
    prior_raw = set().union(*(raw_hashes(manifest) for manifest in development_manifests))
    if prior_raw & (raw_hashes(benign_manifest) | raw_hashes(effect_manifest)):
        raise ValueError("fresh confirmation reuses a development raw capture")

    torch.set_num_threads(4)
    runs = []
    for expected, declared in zip(EXPECTED_SEEDS, development["runs"], strict=True):
        checkpoint = args.development_artifact / declared["checkpoint"]
        if sha256(checkpoint) != declared["checkpoint_sha256"]:
            raise ValueError("development checkpoint hash differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["seed"] != expected or payload["schema"] != development["schema"]:
            raise ValueError("development checkpoint metadata differs")
        model = MaskedHardwareModel(MultimodalConfig(**payload["config"])).to(args.device)
        model.load_state_dict(payload["state_dict"])
        freeze_multimodal_model(model)
        with torch.no_grad():
            training_evidence = multimodal_anomaly_evidence(model, training.to(args.device))
            benign_evidence = multimodal_anomaly_evidence(model, benign.to(args.device))
            effect_evidence = multimodal_anomaly_evidence(model, effect.to(args.device))
        fused_threshold = float(training_evidence.score.max())
        pmu_threshold = float(training_evidence.modality_scores[:, PMU_MODALITY_INDEX].max())
        benign_gate = conjunctive_gate(
            benign_evidence.score.cpu(),
            benign_evidence.modality_scores[:, PMU_MODALITY_INDEX].cpu(),
            fused_threshold,
            pmu_threshold,
        )
        effect_gate = conjunctive_gate(
            effect_evidence.score.cpu(),
            effect_evidence.modality_scores[:, PMU_MODALITY_INDEX].cpu(),
            fused_threshold,
            pmu_threshold,
        )
        runs.append({
            "seed": expected,
            "fused_training_maximum": fused_threshold,
            "pmu_training_maximum": pmu_threshold,
            "training_identifiers": training_ids,
            "benign_identifiers": benign_ids,
            "benign_fused_scores": benign_evidence.score.cpu().tolist(),
            "benign_pmu_scores": benign_evidence.modality_scores[
                :, PMU_MODALITY_INDEX
            ].cpu().tolist(),
            "benign_alerts": int(benign_gate.sum()),
            "effect_identifiers": effect_ids,
            "effect_labels": effect_labels.tolist(),
            "effect_fused_scores": effect_evidence.score.cpu().tolist(),
            "effect_pmu_scores": effect_evidence.modality_scores[
                :, PMU_MODALITY_INDEX
            ].cpu().tolist(),
            "effect_alerts": int(effect_gate[effect_labels == 1].sum()),
            "control_alerts": int(effect_gate[effect_labels == 0].sum()),
        })
    confirmation_pass = all(
        run["benign_alerts"] == 0 and run["effect_alerts"] == 16 and
        run["control_alerts"] == 0 for run in runs
    )
    report = {
        "schema": SCHEMA,
        "scope": "fresh-session confirmation of a preregistered development gate",
        "gate": "fused > training maximum AND PMU residual > training maximum",
        "acceptance": "all seeds: 0/51 benign, 16/16 EFAULT, 0/16 control",
        "confirmation_pass": confirmation_pass,
        "operational_go": False,
        "vulnerability_sensitivity": False,
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": args.device,
        "development_report_sha256": DEVELOPMENT_REPORT_SHA256,
        "benign_plan_sha256": BENIGN_PLAN_SHA256,
        "source_sha256": sha256(Path(__file__)),
        "manifests": {
            "training_anchor": sha256(args.training_anchor / "capture-manifest.json"),
            "fresh_benign": sha256(args.fresh_benign / "capture-manifest.json"),
            "fresh_effect": sha256(args.fresh_effect / "capture-manifest.json"),
        },
        "runs": runs,
    }
    args.output.mkdir(parents=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
