# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate a frozen kernel model on matched safe Dirty Pipe canary arms."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re

import torch

from cpu2tensor.examples.hardware_multimodal import (
    frozen_multimodal_metadata,
    load_frozen_multimodal_model,
    multimodal_anomaly_evidence,
)
from cpu2tensor.examples.hardware_multimodal_experiment import (
    PlannedExecution,
    _atomic_json,
    _atomic_torch_save,
    _featurize_capture,
    _model_capture,
    _model_payload,
    _sha256,
    capture_execution,
    raw_capture_payload,
    seal_kernel_decode_state,
    subject_manifest,
)
from cpu2tensor.examples.hardware_multimodal_features import HardwareFeatureError
from cpu2tensor.hardware import HardwareCaptureError


SCHEMA = "cpu2tensor-dirty-pipe-hardware-validation-v1"
_OUTPUT = re.compile(r"^mode=(effect|neutral) loops=(\d+) mutations=(\d+)$")


def _auc(effect: list[float], neutral: list[float]) -> float:
    wins = sum(left > right for left in effect for right in neutral)
    ties = sum(left == right for left in effect for right in neutral)
    return (wins + 0.5 * ties) / (len(effect) * len(neutral))


def _parse_output(output: bytes, arm: str, loops: int) -> int:
    match = _OUTPUT.fullmatch(output.decode("ascii").strip())
    if match is None or match.group(1) != arm or int(match.group(2)) != loops:
        raise RuntimeError(f"unexpected canary output: {output!r}")
    mutations = int(match.group(3))
    if arm == "neutral" and mutations != 0:
        raise RuntimeError(
            f"neutral canary produced {mutations} mutations, expected zero"
        )
    if arm == "effect" and mutations not in (0, loops):
        raise RuntimeError(
            f"effect canary produced partial manifestation {mutations}/{loops}"
        )
    return mutations


def run(args: argparse.Namespace) -> dict[str, object]:
    if platform.system() != "Linux":
        raise RuntimeError("Dirty Pipe hardware validation requires Linux")
    if args.runs <= 0 or args.loops <= 0 or args.capture_retries < 0:
        raise ValueError("runs and loops must be positive and retries nonnegative")
    binary = args.binary.resolve()
    checkpoint = args.checkpoint.resolve()
    artifact = args.artifact.resolve()
    if not binary.is_file() or not checkpoint.is_file():
        raise ValueError("binary and checkpoint must be files")
    if (artifact / "report.json").exists():
        raise ValueError("artifact already contains a sealed report")
    allowed = os.sched_getaffinity(0)
    if args.target_cpu == args.controller_cpu:
        raise ValueError("target and controller CPUs must be distinct")
    if args.target_cpu not in allowed or args.controller_cpu not in allowed:
        raise ValueError("target and controller CPUs must be allowed")
    os.sched_setaffinity(0, {args.controller_cpu})
    torch.set_num_threads(1)
    subject = subject_manifest(
        binary, target_cpu=args.target_cpu, controller_cpu=args.controller_cpu,
        data_pages=args.data_pages, aux_pages=args.aux_pages,
    )
    metadata = frozen_multimodal_metadata(checkpoint)
    if (
        metadata.get("schema") != "cpu2tensor-frozen-hardware-subject-v1"
        or metadata.get("subject_identity_sha256")
        != subject["subject_identity_sha256"]
        or metadata.get("event_identity_sha256")
        != subject["event_identity_sha256"]
    ):
        raise ValueError("checkpoint differs from the exact validation subject")
    model, threshold = load_frozen_multimodal_model(checkpoint, device="cpu")

    schedule = [
        (pair, arm)
        for pair in range(args.runs)
        for arm in ("effect", "neutral")
    ]
    random.Random(args.seed).shuffle(schedule)
    rows = []
    kernel_decode_states: dict[str, dict[str, str]] = {}
    try:
        for pair, arm in schedule:
            execution = PlannedExecution(
                execution_id=f"{arm}-{pair:05d}",
                family=arm,
                repetition=pair,
                partition="known_cve_validation",
            )
            for attempt in range(args.capture_retries + 1):
                try:
                    captured = capture_execution(
                        binary, execution, loops=args.loops,
                        target_cpu=args.target_cpu,
                        data_pages=args.data_pages, aux_pages=args.aux_pages,
                        timeout=args.timeout,
                    )
                    batch, lanes = _featurize_capture(
                        _model_capture(captured.batches)
                    )
                    break
                except (HardwareCaptureError, HardwareFeatureError):
                    if attempt == args.capture_retries:
                        raise
            mutations = _parse_output(captured.output, arm, args.loops)
            evidence = multimodal_anomaly_evidence(model, batch)
            state_id = captured.decode_sideband.kernel_state_sha256
            kernel_decode_state = kernel_decode_states.get(state_id)
            if kernel_decode_state is None:
                kernel_decode_state = seal_kernel_decode_state(
                    artifact, captured.decode_sideband
                )
                kernel_decode_states[state_id] = kernel_decode_state
            raw_path = artifact / "raw" / f"{execution.execution_id}.pt"
            raw_hash = _atomic_torch_save(
                raw_path,
                raw_capture_payload(
                    captured.batches, decode_sideband=captured.decode_sideband,
                    kernel_decode_state=kernel_decode_state,
                    execution=execution, loops=args.loops,
                    stdout=captured.output, elapsed_ns=captured.elapsed_ns,
                ),
            )
            evidence_path = artifact / "evidence" / f"{execution.execution_id}.pt"
            evidence_hash = _atomic_torch_save(evidence_path, {
                "execution": asdict(execution),
                "raw_sha256": raw_hash,
                "batch": _model_payload(batch),
                "score": evidence.score,
                "modality_scores": evidence.modality_scores,
                "pt_token_error": evidence.pt_token_error,
                "pebs_token_error": evidence.pebs_token_error,
                "pmu_token_error": evidence.pmu_token_error,
                "timing_token_error": evidence.timing_token_error,
            })
            score = float(evidence.score[0])
            rows.append({
                "anonymous_pair": pair,
                "arm": arm,
                "mutations": mutations,
                "output": captured.output.decode("ascii").strip(),
                "score": score,
                "alert": score > threshold,
                "threshold_margin": score - threshold,
                "modality_scores": {
                    name: float(evidence.modality_scores[0, index])
                    for index, name in enumerate(("pt", "pebs", "pmu"))
                    if bool(evidence.modality_present[0, index])
                },
                "capture": captured.counts,
                "attempt": attempt + 1,
                "raw_path": str(raw_path.relative_to(artifact)),
                "raw_sha256": raw_hash,
                "evidence_path": str(evidence_path.relative_to(artifact)),
                "evidence_sha256": evidence_hash,
                "lanes": [int(getattr(lane, "tid")) for lane in lanes],
            })
    finally:
        os.sched_setaffinity(0, allowed)

    effect = [float(row["score"]) for row in rows if row["arm"] == "effect"]
    neutral = [float(row["score"]) for row in rows if row["arm"] == "neutral"]
    paired = {
        pair: {
            str(row["arm"]): float(row["score"])
            for row in rows if row["anonymous_pair"] == pair
        }
        for pair in range(args.runs)
    }
    effect_mutations = {
        int(row["mutations"]) for row in rows if row["arm"] == "effect"
    }
    if len(effect_mutations) != 1:
        raise RuntimeError("Dirty Pipe manifestation changed across effect repeats")
    manifested = next(iter(effect_mutations)) == args.loops
    if args.expected_manifestation != "either" and manifested != (
        args.expected_manifestation == "present"
    ):
        raise RuntimeError(
            f"Dirty Pipe manifestation was {'present' if manifested else 'absent'}, "
            f"expected {args.expected_manifestation}"
        )
    report = {
        "schema": SCHEMA,
        "subject": subject,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "threshold": threshold,
        },
        "kernel_decode_state": (
            next(iter(kernel_decode_states.values()))
            if len(kernel_decode_states) == 1 else None
        ),
        "kernel_decode_states": list(kernel_decode_states.values()),
        "protocol": {
            "seed": args.seed,
            "runs_per_arm": args.runs,
            "loops": args.loops,
            "schedule": [arm for _, arm in schedule],
            "labels_used_by_model": False,
            "labels_unblinded_after_scoring": True,
            "expected_manifestation": args.expected_manifestation,
            "effect": "splice/write against a caller-owned read-only file",
            "neutral": "matched pipe/file operations without the splice primitive",
        },
        "separation": {
            "effect_mean": sum(effect) / len(effect),
            "neutral_mean": sum(neutral) / len(neutral),
            "effect_minus_neutral_mean": (
                sum(effect) / len(effect) - sum(neutral) / len(neutral)
            ),
            "effect_vs_neutral_auc": _auc(effect, neutral),
            "manifested": manifested,
            "effect_mutations_per_execution": next(iter(effect_mutations)),
            "paired_effect_wins": sum(
                values["effect"] > values["neutral"] for values in paired.values()
            ),
            "effect_alerts": sum(
                row["alert"] for row in rows if row["arm"] == "effect"
            ),
            "neutral_alerts": sum(
                row["alert"] for row in rows if row["arm"] == "neutral"
            ),
        },
        "rows": rows,
    }
    report["content_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(artifact / "report.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("artifact", type=Path)
    result.add_argument("--binary", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--runs", type=int, default=12)
    result.add_argument("--loops", type=int, default=20)
    result.add_argument("--seed", type=int, default=20260925)
    result.add_argument("--target-cpu", type=int, default=2)
    result.add_argument("--controller-cpu", type=int, default=3)
    result.add_argument("--data-pages", type=int, default=1024)
    result.add_argument("--aux-pages", type=int, default=8192)
    result.add_argument("--capture-retries", type=int, default=2)
    result.add_argument(
        "--expected-manifestation", choices=("present", "absent", "either"),
        default="either",
    )
    result.add_argument("--timeout", type=float, default=30.0)
    return result


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
