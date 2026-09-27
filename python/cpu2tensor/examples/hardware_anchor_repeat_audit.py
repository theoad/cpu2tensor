# SPDX-License-Identifier: AGPL-3.0-only
"""Audit two exact-plan benign sessions under one anchored feature schema.

The audit loads both sealed datasets through the production custody checks and
compares only rows with identical execution/input identities.  It fits no model,
uses no vulnerability labels, and does not select a representation on session B.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from cpu2tensor.examples.hardware_multimodal import HardwareMultimodalBatch
from cpu2tensor.examples.hardware_multimodal_experiment import load_dataset
from cpu2tensor.examples.hardware_multimodal_features import KernelTextAnchor


SCHEMA = "cpu2tensor-hardware-anchor-repeat-audit-v1"
MODALITIES = ("pt", "pebs", "pmu", "timing")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(batch: HardwareMultimodalBatch, modality: str) -> torch.Tensor:
    if batch.batch_size != 1:
        raise ValueError("repeat audit expects one execution per dataset row")
    if modality == "pt":
        values = (batch.pt, batch.pt_available.to(torch.float32))
    elif modality == "pebs":
        values = (batch.pebs, batch.pebs_available.to(torch.float32))
    elif modality == "pmu":
        values = (batch.pmu, batch.pmu_available.to(torch.float32))
    elif modality == "timing":
        values = (
            torch.nan_to_num(batch.time_bounds),
            torch.nan_to_num(batch.timing_quality),
        )
    else:
        raise ValueError(f"unknown modality: {modality}")
    return torch.cat([value.flatten().to(torch.float64) for value in values])


def _pebs_available(batch: HardwareMultimodalBatch) -> bool:
    return bool(batch.pebs_available.flatten().any().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm, right_norm = left.norm(), right.norm()
    if left_norm == 0 and right_norm == 0:
        return 1.0
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(torch.dot(left, right) / (left_norm * right_norm))


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty repeat metric")
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "rows": len(values),
        "minimum": float(tensor.min()),
        "median": float(tensor.median()),
        "p10": float(torch.quantile(tensor, 0.1)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "maximum": float(tensor.max()),
    }


def _relative_difference(left: int, right: int) -> float:
    return abs(left - right) / max((left + right) / 2.0, 1.0)


def _entry_map(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    entries = manifest["entries"]
    result = {entry["execution_id"]: entry for entry in entries}
    if len(result) != len(entries):
        raise ValueError("dataset repeats an execution identity")
    return result


def _load_plan(
    root: Path, manifest: dict[str, object]
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    declared = manifest["split"]["explicit_plan"]
    if not isinstance(declared, dict):
        raise ValueError("repeat audit requires an explicit execution plan")
    path = root / "execution-plan.json"
    if sha256(path) != declared.get("sha256"):
        raise ValueError("execution plan hash differs from manifest declaration")
    plan = json.loads(path.read_text())
    for field in ("cohort_seed", "kind"):
        if plan.get(field) != declared.get(field):
            raise ValueError(f"execution plan {field} differs from manifest declaration")
    rows = {row["execution_id"]: row for row in plan["rows"]}
    if len(rows) != len(plan["rows"]):
        raise ValueError("execution plan repeats an execution identity")
    return plan, rows


def _validate_plan_rows(
    entries: dict[str, dict[str, object]], plan_rows: dict[str, dict[str, object]]
) -> None:
    if set(entries) != set(plan_rows):
        raise ValueError("manifest execution identities differ from execution plan")
    for execution_id, entry in entries.items():
        row = plan_rows[execution_id]
        comparable = ("family", "loops", "repetition", "partition")
        if any(entry[field] != row[field] for field in comparable):
            raise ValueError(f"manifest row differs from execution plan: {execution_id}")
        if entry["invocation"]["input_seed"] != row["input_seed"]:
            raise ValueError(f"manifest input seed differs from plan: {execution_id}")
        if bool(entry["raw_retained"]) != bool(row["retain_raw"]):
            raise ValueError(f"manifest raw-retention state differs from plan: {execution_id}")


def _validate_independent_sessions(
    session_a: Path,
    session_b: Path,
    entries_a: dict[str, dict[str, object]],
    entries_b: dict[str, dict[str, object]],
) -> None:
    if session_a.resolve() == session_b.resolve():
        raise ValueError("repeat audit requires two distinct session roots")
    if sha256(session_a / "capture-manifest.json") == sha256(
        session_b / "capture-manifest.json"
    ):
        raise ValueError("repeat audit requires distinct capture manifests")
    raw_a = {entry["raw_sha256"] for entry in entries_a.values()}
    raw_b = {entry["raw_sha256"] for entry in entries_b.values()}
    if raw_a & raw_b:
        raise ValueError("sessions reuse one or more raw captures")


def _anchor(root: Path, manifest: dict[str, object]) -> dict[str, object]:
    references = manifest["kernel_decode_states"]
    if len(references) != 1:
        raise ValueError("repeat audit requires one exact kernel state")
    reference = references[0]
    path = root / reference["path"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    anchor = KernelTextAnchor.from_symbols(
        state["kernel_symbols"].numpy().tobytes(), reference["kernel_state_sha256"]
    )
    return {**asdict(anchor), "state_file_sha256": sha256(path)}


def audit(session_a: Path, session_b: Path) -> dict[str, object]:
    manifest_a, rows_a = load_dataset(session_a)
    manifest_b, rows_b = load_dataset(session_b)
    identity_fields = (
        "subject_identity_sha256",
        "event_identity_sha256",
        "feature_schema",
    )
    if any(manifest_a[field] != manifest_b[field] for field in identity_fields):
        raise ValueError("session subject/event/feature identity differs")
    plan_a, plan_rows_a = _load_plan(session_a, manifest_a)
    plan_b, plan_rows_b = _load_plan(session_b, manifest_b)
    if plan_a != plan_b:
        raise ValueError("sessions do not share one explicit execution plan")
    entries_a, entries_b = _entry_map(manifest_a), _entry_map(manifest_b)
    if set(entries_a) != set(entries_b) or set(rows_a) != set(rows_b):
        raise ValueError("session execution identities differ")
    _validate_plan_rows(entries_a, plan_rows_a)
    _validate_plan_rows(entries_b, plan_rows_b)
    _validate_independent_sessions(
        session_a, session_b, entries_a, entries_b
    )
    anchor_a, anchor_b = _anchor(session_a, manifest_a), _anchor(session_b, manifest_b)
    if anchor_a != anchor_b:
        raise ValueError("session exact-boot kernel anchors differ")

    cosine = {name: [] for name in MODALITIES}
    exact = {name: 0 for name in MODALITIES}
    exposure = {name: [] for name in ("pt_bytes", "pebs_samples", "elapsed_ns")}
    pebs_presence = {"both_present": 0, "both_missing": 0, "mismatched": 0}
    for execution_id in sorted(entries_a):
        left, right = entries_a[execution_id], entries_b[execution_id]
        comparable = (
            "family", "loops", "repetition", "partition", "invocation"
        )
        if any(left[field] != right[field] for field in comparable):
            raise ValueError(f"matched input identity differs: {execution_id}")
        left_has_pebs = _pebs_available(rows_a[execution_id])
        right_has_pebs = _pebs_available(rows_b[execution_id])
        for name in MODALITIES:
            left_vector, right_vector = _vector(rows_a[execution_id], name), _vector(
                rows_b[execution_id], name
            )
            if name != "pebs" or (left_has_pebs and right_has_pebs):
                cosine[name].append(_cosine(left_vector, right_vector))
            exact[name] += int(torch.equal(left_vector, right_vector))
        for name in ("pt_bytes", "pebs_samples"):
            exposure[name].append(_relative_difference(
                int(left["capture"][name]), int(right["capture"][name])
            ))
        exposure["elapsed_ns"].append(_relative_difference(
            int(left["elapsed_ns"]), int(right["elapsed_ns"])
        ))
        if left_has_pebs and right_has_pebs:
            pebs_presence["both_present"] += 1
        elif not left_has_pebs and not right_has_pebs:
            pebs_presence["both_missing"] += 1
        else:
            pebs_presence["mismatched"] += 1
    return {
        "schema": SCHEMA,
        "scope": "two-session exact-plan benign representation stability",
        "operational_go": False,
        "session_a_manifest_sha256": sha256(session_a / "capture-manifest.json"),
        "session_b_manifest_sha256": sha256(session_b / "capture-manifest.json"),
        "subject_identity_sha256": manifest_a["subject_identity_sha256"],
        "event_identity_sha256": manifest_a["event_identity_sha256"],
        "feature_schema": manifest_a["feature_schema"],
        "execution_plan": manifest_a["split"]["explicit_plan"],
        "kernel_anchor": anchor_a,
        "rows": len(entries_a),
        "raw_sha256_overlap_rows": 0,
        "cosine": {name: _summary(values) for name, values in cosine.items()},
        "pebs_cosine_scope": "rows sampled in both sessions",
        "exact_rows": exact,
        "relative_difference": {
            name: _summary(values) for name, values in exposure.items()
        },
        "pebs_presence": pebs_presence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_a", type=Path)
    parser.add_argument("session_b", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output path")
    torch.set_num_threads(1)
    result = audit(args.session_a, args.session_b)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
