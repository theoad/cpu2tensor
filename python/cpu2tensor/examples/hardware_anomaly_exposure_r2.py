# SPDX-License-Identifier: AGPL-3.0-only
"""Training-only exposure conditioning of the sealed 102k compact corpus.

Three fixed ablations remove predictable PT-byte volume and PEBS sample-count
effects.  No held-out family or vulnerability label participates in fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch


SCHEMA = "cpu2tensor-hardware-anomaly-exposure-r2"
MANIFEST_SHA256 = "0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d"
CACHE_SHA256 = "5c689576cfd8d1576aa6f063e598a02cfee21d0fa629141e4c9913a8fda1fcdc"
PARTITIONS = ("training", "calibration", "familiar_validation", "heldout_family")
VARIANTS = ("volume", "volume_time", "volume_time_quadratic")
RIDGE_PER_ROW = 0.001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exposure_matrix(manifest: dict[str, object]) -> torch.Tensor:
    values = []
    for entry in manifest["entries"]:
        capture = entry["capture"]
        pt_bytes = capture["pt_bytes"]
        pebs_samples = capture["pebs_samples"]
        elapsed_ns = entry["elapsed_ns"]
        if pt_bytes <= 0 or pebs_samples < 0 or elapsed_ns <= 0:
            raise ValueError(f"invalid exposure for {entry['execution_id']}")
        values.append((pt_bytes, elapsed_ns / 1_000_000.0, pebs_samples))
    return torch.log1p(torch.tensor(values, dtype=torch.float64))


def make_design(exposure: torch.Tensor, partition: torch.Tensor,
                variant: str,
                state: dict[str, torch.Tensor] | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if variant not in VARIANTS:
        raise ValueError("unknown exposure ablation")
    exposure = exposure.to(torch.float64)
    training = partition == 0
    if state is None:
        center = exposure[training].mean(dim=0)
        scale = exposure[training].std(dim=0, unbiased=False).clamp(min=0.01)
    else:
        center = state["exposure_center"]
        scale = state["exposure_scale"]
    normalized = (exposure - center) / scale
    pt = normalized[:, [0]] if variant == "volume" else normalized[:, [0, 1]]
    pebs = normalized[:, [2]] if variant == "volume" else normalized[:, [2, 1]]
    if variant == "volume_time_quadratic":
        pt = torch.cat((pt, pt.square(), (pt[:, 0] * pt[:, 1])[:, None]), dim=1)
        pebs = torch.cat((pebs, pebs.square(), (pebs[:, 0] * pebs[:, 1])[:, None]), dim=1)
    # Polynomial terms must also be centered against training only.  The
    # intercept lives in the later training-only feature normalizer.
    if state is None:
        pt_center = pt[training].mean(dim=0)
        pebs_center = pebs[training].mean(dim=0)
    else:
        pt_center = state["pt_design_center"]
        pebs_center = state["pebs_design_center"]
    pt = pt - pt_center
    pebs = pebs - pebs_center
    return pt, pebs, {
        "exposure_center": center, "exposure_scale": scale,
        "pt_design_center": pt_center, "pebs_design_center": pebs_center,
    }


def ridge_coefficients(design: torch.Tensor, targets: torch.Tensor,
                       training: torch.Tensor) -> torch.Tensor:
    x = design[training]
    y = targets[training]
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("nonfinite training exposure or target")
    gram = x.T @ x
    ridge = RIDGE_PER_ROW * x.shape[0] * torch.eye(x.shape[1], dtype=x.dtype)
    return torch.linalg.solve(gram + ridge, x.T @ y)


def condition_features(features: torch.Tensor, pt_design: torch.Tensor,
                       pebs_design: torch.Tensor, partition: torch.Tensor,
                       pt_coefficients: torch.Tensor | None = None,
                       pebs_coefficients: torch.Tensor | None = None,
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if features.shape[1] != 789:
        raise ValueError("expected the R1 789-feature cache")
    training = partition == 0
    source = features.to(torch.float64)
    pebs_indices = torch.tensor((512, 648))
    if pt_coefficients is None:
        pt_coefficients = ridge_coefficients(pt_design, source[:, :512], training)
    if pebs_coefficients is None:
        pebs_coefficients = ridge_coefficients(
            pebs_design, source[:, pebs_indices], training,
        )
    conditioned = source.clone()
    conditioned[:, :512] -= pt_design @ pt_coefficients
    conditioned[:, pebs_indices] -= pebs_design @ pebs_coefficients
    return conditioned, pt_coefficients, pebs_coefficients


def normalize(conditioned: torch.Tensor, partition: torch.Tensor,
              center: torch.Tensor | None = None,
              scale: torch.Tensor | None = None,
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    training = partition == 0
    if center is None:
        center = conditioned[training].mean(dim=0)
    if scale is None:
        scale = conditioned[training].std(dim=0, unbiased=False).clamp(min=0.05)
    standardized = ((conditioned - center) / scale).clamp(-20, 20)
    if not bool(torch.isfinite(standardized).all()):
        raise ValueError("nonfinite normalized feature")
    return standardized, center, scale


def score(standardized: torch.Tensor) -> torch.Tensor:
    return standardized.square().mean(dim=1).to(torch.float32)


def tail_result(scores: torch.Tensor, partition: torch.Tensor,
                false_positive_rate: float) -> dict[str, object]:
    calibration = scores[partition == 1]
    tail_count = max(1, int(calibration.numel() * false_positive_rate))
    threshold = float(torch.topk(calibration, tail_count + 1).values[-1])
    return {
        "threshold": threshold,
        "calibration_tail_points": tail_count,
        "calibration_alerts": int((calibration > threshold).sum()),
        "familiar_alerts": int((scores[partition == 2] > threshold).sum()),
        "familiar_executions": int((partition == 2).sum()),
        "heldout_alerts": int((scores[partition == 3] > threshold).sum()),
        "heldout_executions": int((partition == 3).sum()),
    }


def stratified_alerts(manifest: dict[str, object], scores: torch.Tensor,
                      partition: torch.Tensor, threshold: float) -> dict[str, object]:
    groups = {}
    for index, entry in enumerate(manifest["entries"]):
        if partition[index] < 2:
            continue
        key = f"{entry['partition']}/{entry['family']}/loops={entry['loops']}"
        row = groups.setdefault(key, {"executions": 0, "alerts": 0, "score_sum": 0.0})
        row["executions"] += 1
        row["alerts"] += int(float(scores[index]) > threshold)
        row["score_sum"] += float(scores[index])
    return {key: {"executions": row["executions"], "alerts": row["alerts"],
                  "mean_score": row["score_sum"] / row["executions"]}
            for key, row in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("compact_cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if sha256(args.corpus / "capture-manifest.json") != MANIFEST_SHA256:
        raise ValueError("unexpected immutable corpus manifest")
    if sha256(args.compact_cache) != CACHE_SHA256:
        raise ValueError("unexpected R1 compact feature cache")
    manifest = json.loads((args.corpus / "capture-manifest.json").read_text())
    cache = torch.load(args.compact_cache, map_location="cpu", weights_only=True)
    if cache["manifest_sha256"] != MANIFEST_SHA256 or cache["schema"] != "cpu2tensor-compact-anomaly-r1":
        raise ValueError("compact cache identity mismatch")
    features = cache["features"]
    partition = cache["partition"]
    if features.shape != (102_000, 789) or partition.shape != (102_000,):
        raise ValueError("unexpected immutable split shape")
    if [int((partition == index).sum()) for index in range(4)] != [56_000, 21_000, 7_000, 18_000]:
        raise ValueError("unexpected immutable split counts")
    exposure_start = time.perf_counter()
    exposure = exposure_matrix(manifest)
    exposure_seconds = time.perf_counter() - exposure_start
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    checkpoints = {}
    for variant in VARIANTS:
        fit_start = time.perf_counter()
        pt_design, pebs_design, exposure_state = make_design(exposure, partition, variant)
        conditioned, pt_beta, pebs_beta = condition_features(
            features, pt_design, pebs_design, partition,
        )
        standardized, center, scale = normalize(conditioned, partition)
        fit_seconds = time.perf_counter() - fit_start
        scoring_start = time.perf_counter()
        # Ten in-RAM passes prevent a single tiny launch from becoming a
        # misleading throughput claim.  The scores used below are last-pass.
        for _ in range(10):
            scores = score(standardized)
        score_seconds = time.perf_counter() - scoring_start
        tail_1e3 = tail_result(scores, partition, 1e-3)
        tail_1e4 = tail_result(scores, partition, 1e-4)
        checkpoint = args.output / f"{variant}.pt"
        torch.save({
            "schema": SCHEMA, "variant": variant,
            "manifest_sha256": MANIFEST_SHA256, "compact_cache_sha256": CACHE_SHA256,
            "pt_coefficients": pt_beta, "pebs_coefficients": pebs_beta,
            "feature_center": center, "feature_scale": scale,
            **exposure_state,
            "threshold_1e3": tail_1e3["threshold"],
            "threshold_1e4_exploratory": tail_1e4["threshold"],
        }, checkpoint)
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        replay_pt_design, replay_pebs_design, _ = make_design(
            exposure, partition, variant, restored,
        )
        replay, _, _ = condition_features(
            features, replay_pt_design, replay_pebs_design, partition,
            restored["pt_coefficients"], restored["pebs_coefficients"],
        )
        replay_standardized, _, _ = normalize(
            replay, partition, restored["feature_center"], restored["feature_scale"],
        )
        reload_exact = torch.equal(score(replay_standardized), scores)
        if not reload_exact:
            raise RuntimeError("checkpoint reload changed frozen scores")
        checkpoint_hash = sha256(checkpoint)
        checkpoints[variant] = checkpoint_hash
        results[variant] = {
            "parameters_excluding_shared_feature_normalizer": pt_beta.numel() + pebs_beta.numel(),
            "shared_feature_normalizer_coefficients": center.numel() + scale.numel(),
            "exposure_normalizer_coefficients": sum(
                tensor.numel() for tensor in exposure_state.values()
            ),
            "fit_seconds": fit_seconds,
            "fit_training_rows_per_second_including_full_transform": 56_000 / fit_seconds,
            "scoring_seconds_per_pass": score_seconds / 10,
            "scoring_executions_per_second_in_memory": 102_000 * 10 / score_seconds,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_reload_bit_exact_cpu": reload_exact,
            "fpr_1e-3": tail_1e3,
            "fpr_1e-4_exploratory": tail_1e4,
            "stratified_at_1e-3": stratified_alerts(manifest, scores, partition, tail_1e3["threshold"]),
            "stratified_at_1e-4_exploratory": stratified_alerts(manifest, scores, partition, tail_1e4["threshold"]),
        }
        print(f"{variant}: familiar {tail_1e3['familiar_alerts']}, heldout {tail_1e3['heldout_alerts']}", flush=True)
    report = {
        "schema": SCHEMA, "host": platform.node(), "machine": platform.machine(),
        "device": "cpu", "torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
        "manifest_sha256": MANIFEST_SHA256, "compact_cache_sha256": CACHE_SHA256,
        "partitions": dict(zip(PARTITIONS, (56_000, 21_000, 7_000, 18_000))),
        "exposure_extraction_seconds": exposure_seconds,
        "variants": results,
        "warning": "21k calibration has only about two tail points at 1e-4; this does not certify an operational false-positive rate.",
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"report_sha256={sha256(report_path)}", flush=True)


if __name__ == "__main__":
    main()
