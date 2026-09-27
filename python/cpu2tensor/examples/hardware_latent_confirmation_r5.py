# SPDX-License-Identifier: AGPL-3.0-only
"""Prospectively confirm frozen benign hardware-representation geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics

import torch

from cpu2tensor.examples.hardware_futex_transfer_r1 import sha256
from cpu2tensor.examples.hardware_latent_geometry_r4 import (
    R3_REPORT_SHA256,
    embed,
    load_cohort,
    nearest_centroid_accuracy,
    view_metrics,
    view_subset,
)
from cpu2tensor.examples.hardware_multimodal import (
    MaskedHardwareModel,
    MultimodalConfig,
    freeze_multimodal_model,
)


SCHEMA = "cpu2tensor-hardware-latent-confirmation-r5"
EXPECTED_PLAN_SHA256 = "0724fa230f8913fc1037be11b2a929e6857f47fba52e6553bad23a5fce556863"
EXPECTED_COHORT_SEED = 2026092740


def confirmation_decision(runs: list[dict[str, object]]) -> dict[str, object]:
    by_objective = {
        name: [run for run in runs if run["objective"] == name]
        for name in ("reconstruction", "vicreg")
    }
    if any(len(group) != 3 for group in by_objective.values()):
        raise ValueError("confirmation matrix is incomplete")
    medians = {}
    for name, group in by_objective.items():
        medians[name] = {
            metric: statistics.median(float(run[metric]) for run in group)
            for metric in ("family_accuracy", "intensity_accuracy", "stratum_accuracy",
                           "view_identity_top1")
        }
    vicreg = by_objective["vicreg"]
    floors = {
        "family_accuracy": min(float(run["family_accuracy"]) for run in vicreg) >= 0.93,
        "intensity_accuracy": min(float(run["intensity_accuracy"]) for run in vicreg) >= 0.55,
        "stratum_accuracy": min(float(run["stratum_accuracy"]) for run in vicreg) >= 0.93,
        "view_identity_top1": min(float(run["view_identity_top1"]) for run in vicreg) >= 0.58,
    }
    median_thresholds = {
        "family_accuracy": medians["vicreg"]["family_accuracy"] >= 0.94,
        "intensity_accuracy": medians["vicreg"]["intensity_accuracy"] >= 0.58,
        "stratum_accuracy": medians["vicreg"]["stratum_accuracy"] >= 0.94,
        "view_identity_top1": medians["vicreg"]["view_identity_top1"] >= 0.60,
    }
    margins = {
        "family_accuracy": (
            medians["vicreg"]["family_accuracy"]
            - medians["reconstruction"]["family_accuracy"] >= 0.03
        ),
        "intensity_accuracy": (
            medians["vicreg"]["intensity_accuracy"]
            - medians["reconstruction"]["intensity_accuracy"] >= 0.10
        ),
        "view_identity_top1": (
            medians["vicreg"]["view_identity_top1"]
            - medians["reconstruction"]["view_identity_top1"] >= 0.05
        ),
    }
    return {
        "medians": medians,
        "all_seed_floors": floors,
        "median_thresholds": median_thresholds,
        "reconstruction_margins": margins,
        "passes_preregistered_gate": all((*floors.values(), *median_thresholds.values(),
                                            *margins.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("r3_artifact", type=Path)
    parser.add_argument("pilot_a", type=Path)
    parser.add_argument("prospective_d", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output directory")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    r3_path = args.r3_artifact / "report.json"
    if sha256(r3_path) != R3_REPORT_SHA256:
        raise ValueError("R3 artifact differs")
    r3 = json.loads(r3_path.read_text())
    cohort_a = load_cohort(args.pilot_a)
    cohort_d = load_cohort(args.prospective_d)
    manifest_d = cohort_d[0]
    plan = args.prospective_d / "execution-plan.json"
    if sha256(plan) != EXPECTED_PLAN_SHA256:
        raise ValueError("prospective plan differs")
    explicit_plan = manifest_d["split"]["explicit_plan"]
    if (manifest_d["collection"]["executions"] != 1020 or
            manifest_d["collection"]["admission"]["rejected_attempts"] != 0 or
            manifest_d["collection"]["loss_count"] != 0 or
            explicit_plan["cohort_seed"] != EXPECTED_COHORT_SEED or
            explicit_plan["sha256"] != EXPECTED_PLAN_SHA256 or
            sum(bool(entry["raw_retained"]) for entry in manifest_d["entries"]) != 204 or
            any(entry["admission"]["attempt"] != 1 or
                any(entry["capture"][name] for name in (
                    "lost_sources", "missing_sources", "multiplexed_sources"
                )) for entry in manifest_d["entries"])):
        raise ValueError("prospective collection is incomplete")
    view_batch, view_identifiers = view_subset(
        cohort_d[1], cohort_d[2], cohort_d[5]
    )

    runs = []
    for prior in r3["runs"]:
        if prior["objective"] not in ("reconstruction", "vicreg"):
            continue
        checkpoint = args.r3_artifact / prior["checkpoint"]
        if sha256(checkpoint) != prior["checkpoint_sha256"]:
            raise ValueError("R3 checkpoint differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = MaskedHardwareModel(MultimodalConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model = model.to(args.device)
        freeze_multimodal_model(model)
        training = embed(model, cohort_a[2], args.device)
        evaluation = embed(model, cohort_d[2], args.device)
        view = view_metrics(model, view_batch, args.device)
        runs.append({
            "objective": prior["objective"],
            "pretraining_seed": prior["pretraining_seed"],
            "family_accuracy": nearest_centroid_accuracy(
                training, cohort_a[3], evaluation, cohort_d[3]
            ),
            "intensity_accuracy": nearest_centroid_accuracy(
                training, cohort_a[4], evaluation, cohort_d[4]
            ),
            "stratum_accuracy": nearest_centroid_accuracy(
                training, cohort_a[5], evaluation, cohort_d[5]
            ),
            "view_identity_top1": view["identity_top1"],
            "view_paired_cosine": view["mean_paired_cosine"],
            "checkpoint": prior["checkpoint"],
            "checkpoint_sha256": prior["checkpoint_sha256"],
        })
    report = {
        "schema": SCHEMA,
        "scope": "prospective benign-only frozen geometry confirmation",
        "claim_boundary": "foundation representation confirmation, not bug detection",
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": args.device,
        "r3_report_sha256": R3_REPORT_SHA256,
        "prospective_manifest_sha256": sha256(args.prospective_d / "capture-manifest.json"),
        "prospective_plan_sha256": EXPECTED_PLAN_SHA256,
        "view_identifiers_sha256": hashlib.sha256(
            "\n".join(view_identifiers).encode()
        ).hexdigest(),
        "decision": confirmation_decision(runs),
        "source_sha256": sha256(Path(__file__)),
        "runs": runs,
    }
    args.output.mkdir(parents=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
