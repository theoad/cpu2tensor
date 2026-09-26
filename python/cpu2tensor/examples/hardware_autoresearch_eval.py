# SPDX-License-Identifier: AGPL-3.0-only
"""Locked, benign-only development evaluation of autoresearch checkpoints.

This file is outside the mutable training candidate. Its scores are a proxy for
lawful-family transfer, not evidence of sensitivity to a physical kernel bug.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Sequence

import torch

from cpu2tensor.examples.hardware_autoresearch_train import (
    load_frozen,
    score_features,
)


EVAL_SCHEMA = "cpu2tensor-autoresearch-eval-v1"
EVAL_SCHEMA_V2 = "cpu2tensor-autoresearch-eval-v2"
CHECKPOINT_SCHEMA = "cpu2tensor-autoresearch-checkpoint-v1"
CHECKPOINT_SCHEMA_V2 = "cpu2tensor-autoresearch-checkpoint-v2"
REPORT_SCHEMA = "cpu2tensor-autoresearch-development-evaluation-v1"
REPORT_SCHEMA_V2 = "cpu2tensor-autoresearch-development-evaluation-v2"
SCORES_SCHEMA = "cpu2tensor-autoresearch-development-scores-v1"
SCORES_SCHEMA_V2 = "cpu2tensor-autoresearch-development-scores-v2"
EXPOSURE_FIELDS = ("pt_bytes", "elapsed_ns", "pebs_samples")
SEALED_V2_TRAIN_SHA256 = "f949c7dd6df006a5cf84e2da59950e3ddad8fd0a31f9e78573814bd798b04292"
SEALED_V2_EVAL_SHA256 = "9b875bab80db03afcc6a612bba6216c0e42af2fa6c8063cc51ae63ade4b40d26"
FEATURES = 789
PARTITIONS = {1: "calibration", 2: "familiar_validation", 3: "heldout_family"}
EXPECTED_COUNTS = {1: 21_000, 2: 7_000, 3: 18_000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def validate_eval_cache(payload: object, *,
                        expected_counts: dict[int, int] = EXPECTED_COUNTS,
                        ) -> dict[str, object]:
    """Reject incomplete, relabeled, or nonfinite development-eval caches."""
    keys = {
        "schema", "features", "partition", "execution_ids", "families", "loops",
        "source_manifest_sha256", "source_cache_sha256",
    }
    if not isinstance(payload, dict) or payload.get("schema") not in (EVAL_SCHEMA, EVAL_SCHEMA_V2):
        raise ValueError("unsupported or incomplete development-eval cache")
    is_v2 = payload["schema"] == EVAL_SCHEMA_V2
    if set(payload) != (keys | {"exposure_raw", "exposure_fields"} if is_v2 else keys):
        raise ValueError("unsupported or incomplete development-eval cache")
    count = sum(expected_counts.values())
    features = payload["features"]
    partition = payload["partition"]
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32 or
            features.device.type != "cpu" or features.shape != (count, FEATURES) or
            not bool(torch.isfinite(features).all())):
        raise ValueError("development features must be finite CPU float32 [rows,789]")
    if (not isinstance(partition, torch.Tensor) or partition.dtype != torch.uint8 or
            partition.device.type != "cpu" or partition.shape != (count,)):
        raise ValueError("development partitions must be CPU uint8 [rows]")
    if any(int((partition == code).sum()) != expected for code, expected in expected_counts.items()) or \
            any(int((partition == code).sum()) for code in set(range(4)) - set(expected_counts)):
        raise ValueError("development partition counts changed")
    ids, families, loops = (payload[key] for key in ("execution_ids", "families", "loops"))
    if (not all(isinstance(values, list) and len(values) == count
                for values in (ids, families, loops)) or
            any(not isinstance(value, str) or not value for value in ids + families) or
            len(set(ids)) != count or
            any(type(value) is not int or value <= 0 for value in loops)):
        raise ValueError("development row identities or loop intensities are invalid")
    if not all(_digest(payload[key]) for key in
               ("source_manifest_sha256", "source_cache_sha256")):
        raise ValueError("development source identity is invalid")
    if is_v2:
        exposure = payload["exposure_raw"]
        if payload["exposure_fields"] != EXPOSURE_FIELDS:
            raise ValueError("development exposure field order is invalid")
        if (not isinstance(exposure, torch.Tensor) or exposure.dtype != torch.int64 or
                exposure.device.type != "cpu" or exposure.shape != (count, len(EXPOSURE_FIELDS)) or
                not bool((exposure[:, :2] > 0).all()) or
                not bool((exposure[:, 2] >= 0).all())):
            raise ValueError("development exposure_raw must be valid CPU int64 [rows,3]")
    return payload


def calibrate_threshold(scores: torch.Tensor, *, target_fpr: float) -> tuple[float, int]:
    """Use only calibration scores; strict > admits at most floor(alpha*n) rows."""
    if (scores.ndim != 1 or scores.numel() == 0 or
            not bool(torch.isfinite(scores).all()) or
            not math.isfinite(target_fpr) or not 0 < target_fpr < 1):
        raise ValueError("calibration requires finite scores and 0 < target_fpr < 1")
    allowed = math.floor(target_fpr * scores.numel())
    threshold = float(torch.sort(scores, descending=True).values[allowed])
    return threshold, allowed


def _alert_metrics(scores: torch.Tensor, selected: torch.Tensor,
                   threshold: float) -> dict[str, float | int]:
    values = scores[selected]
    count = values.numel()
    alerts = int((values > threshold).sum())
    return {
        "rows": count, "alerts": alerts,
        "alert_rate": alerts / count if count else 0.0,
        "scores_mean": float(values.mean()) if count else 0.0,
        "scores_median": float(values.median()) if count else 0.0,
        "scores_min": float(values.min()) if count else 0.0,
        "scores_max": float(values.max()) if count else 0.0,
    }


def _strata(scores: torch.Tensor, partition: torch.Tensor, families: list[str],
            loops: list[int], threshold: float) -> dict[str, object]:
    family_values = sorted(set(families))
    intensity_values = sorted(set(loops))
    family_mask = {name: torch.tensor([value == name for value in families])
                   for name in family_values}
    intensity_mask = {value: torch.tensor([loop == value for loop in loops])
                      for value in intensity_values}
    result: dict[str, object] = {}
    for code, name in PARTITIONS.items():
        selected = partition == code
        result[name] = {
            **_alert_metrics(scores, selected, threshold),
            "by_family": {
                family: _alert_metrics(scores, selected & family_mask[family], threshold)
                for family in family_values if bool((selected & family_mask[family]).any())
            },
            "by_intensity_loops": {
                str(loops_value): _alert_metrics(
                    scores, selected & intensity_mask[loops_value], threshold,
                ) for loops_value in intensity_values
                if bool((selected & intensity_mask[loops_value]).any())
            },
            "by_family_and_intensity": {
                family: {
                    str(loops_value): _alert_metrics(
                        scores, selected & family_mask[family] & intensity_mask[loops_value],
                        threshold,
                    ) for loops_value in intensity_values
                    if bool((selected & family_mask[family] &
                             intensity_mask[loops_value]).any())
                } for family in family_values if bool((selected & family_mask[family]).any())
            },
        }
    return result


def evaluate(eval_cache: Path, expected_eval_sha256: str, checkpoints: Sequence[Path],
             output: Path, *, device: str = "cpu", target_fpr: float = 1e-4,
             batch_size: int = 1024,
             expected_counts: dict[int, int] = EXPECTED_COUNTS,
             ) -> dict[str, object]:
    """Score sealed benign rows without making a keep/discard decision."""
    if output.exists():
        raise FileExistsError("evaluation output already exists")
    evaluation_started = time.perf_counter()
    if not _digest(expected_eval_sha256) or sha256(eval_cache) != expected_eval_sha256:
        raise ValueError("development-eval SHA-256 differs from the locked digest")
    if not checkpoints or batch_size <= 0:
        raise ValueError("at least one checkpoint and a positive batch size are required")
    payload = validate_eval_cache(
        torch.load(eval_cache, map_location="cpu", weights_only=True),
        expected_counts=expected_counts,
    )
    identity = {key: payload[key] for key in
                ("source_manifest_sha256", "source_cache_sha256")}
    is_v2 = payload["schema"] == EVAL_SCHEMA_V2
    if is_v2:
        if expected_eval_sha256 != SEALED_V2_EVAL_SHA256:
            raise ValueError("v2 development-eval SHA-256 differs from the sealed cache")
        identity["exposure_fields"] = EXPOSURE_FIELDS
        identity["train_cache_sha256"] = SEALED_V2_TRAIN_SHA256
    checkpoint_schema = CHECKPOINT_SCHEMA_V2 if is_v2 else CHECKPOINT_SCHEMA
    partition = payload["partition"]
    scores_by_seed: dict[str, torch.Tensor] = {}
    checkpoint_reports = []
    for checkpoint in checkpoints:
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (not isinstance(checkpoint_payload, dict) or
                checkpoint_payload.get("schema") != checkpoint_schema or
                checkpoint_payload.get("train_cache_identity") != identity):
            raise ValueError("checkpoint and development cache have different source custody")
        seed = checkpoint_payload.get("seed")
        if type(seed) is not int or str(seed) in scores_by_seed:
            raise ValueError("checkpoint seeds must be unique integers")
        scorer = load_frozen(checkpoint, device=device)
        score_started = time.perf_counter()
        if is_v2:
            scores = score_features(
                scorer, payload["features"], exposure_raw=payload["exposure_raw"],
                batch_size=batch_size,
            )
        else:
            scores = score_features(scorer, payload["features"], batch_size=batch_size)
        score_seconds = time.perf_counter() - score_started
        if (scores.dtype != torch.float32 or scores.shape != (len(partition),) or
                not bool(torch.isfinite(scores).all())):
            raise ValueError("checkpoint produced incomplete or nonfinite scores")
        threshold, allowed = calibrate_threshold(
            scores[partition == 1], target_fpr=target_fpr,
        )
        scores_by_seed[str(seed)] = scores
        checkpoint_reports.append({
            "seed": seed, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "threshold": threshold, "target_fpr": target_fpr,
            "calibration_allowed_alerts": allowed,
            "scoring_seconds": score_seconds,
            "scoring_rows_per_second": scores.numel() / score_seconds,
            "metrics": _strata(scores, partition, payload["families"],
                                payload["loops"], threshold),
        })
    output.mkdir(parents=True)
    scores_path = output / "scores.pt"
    torch.save({
        "schema": SCORES_SCHEMA_V2 if is_v2 else SCORES_SCHEMA,
        "development_eval_sha256": expected_eval_sha256,
        "execution_ids": payload["execution_ids"], "scores_by_seed": scores_by_seed,
    }, scores_path)
    scores_hash = sha256(scores_path)
    report = {
        "schema": REPORT_SCHEMA_V2 if is_v2 else REPORT_SCHEMA,
        "status": "proxy_benign_only_not_real_bug_gate",
        "keep_discard_authorized": False,
        "missing_gates": ["separate_session_seed_varied_benign_test",
                          "matched_physical_effect_and_lawful_control"],
        "development_eval_sha256": expected_eval_sha256,
        "development_eval_schema": payload["schema"],
        "checkpoint_schema": checkpoint_schema,
        "source_identity": identity,
        "evaluation_rows": len(partition),
        "intensity_definition": "exact target loop count within each family",
        "host": platform.node(), "machine": platform.machine(),
        "device": device, "torch": torch.__version__,
        "pre_report_wall_seconds": time.perf_counter() - evaluation_started,
        "scores_file": scores_path.name, "scores_sha256": scores_hash,
        "seeds": checkpoint_reports,
    }
    (output / "evaluation.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-cache", required=True, type=Path)
    parser.add_argument("--eval-sha256", required=True)
    parser.add_argument("--checkpoint", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-fpr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    report = evaluate(args.eval_cache, args.eval_sha256, args.checkpoint,
                      args.output, device=args.device, target_fpr=args.target_fpr,
                      batch_size=args.batch_size)
    print(json.dumps({
        "schema": report["schema"], "status": report["status"],
        "keep_discard_authorized": report["keep_discard_authorized"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
