# SPDX-License-Identifier: AGPL-3.0-only
"""Benign-only cross-modal conditional-residual experiment on a sealed cache.

This is an offline research score, not a production scorer or a bug detector.
The source and target are different sensor modalities from the same execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch


SCHEMA = "cpu2tensor-conditional-pebs-r1"
CACHE_SCHEMA = "cpu2tensor-compact-anomaly-r1"
PARTITIONS = ("training", "calibration", "familiar_validation", "heldout_family")
PARTITION_COUNTS = (56_000, 21_000, 7_000, 18_000)
PT_END = 512
PEBS_END = 784
FEATURE_COUNT = 789
RIDGE_FRACTION = 0.1
SCALE_FLOOR = 0.05
CLIP = 8.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache(cache: dict[str, object], manifest: dict[str, object],
                   manifest_hash: str) -> tuple[torch.Tensor, torch.Tensor]:
    if cache.get("schema") != CACHE_SCHEMA or cache.get("manifest_sha256") != manifest_hash:
        raise ValueError("cache schema or sealed manifest identity mismatch")
    features, partition = cache.get("features"), cache.get("partition")
    if not isinstance(features, torch.Tensor) or not isinstance(partition, torch.Tensor):
        raise ValueError("cache tensors missing")
    if (features.shape != (sum(PARTITION_COUNTS), FEATURE_COUNT) or
            features.dtype != torch.float32 or partition.shape != (sum(PARTITION_COUNTS),) or
            partition.dtype != torch.uint8 or not bool(torch.isfinite(features).all())):
        raise ValueError("cache tensor shape, dtype, or finiteness mismatch")
    counts = tuple(int((partition == index).sum()) for index in range(4))
    if counts != PARTITION_COUNTS:
        raise ValueError("cache partition counts mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != features.shape[0]:
        raise ValueError("manifest entries mismatch")
    expected = torch.tensor([PARTITIONS.index(entry["partition"]) for entry in entries],
                            dtype=torch.uint8)
    if not torch.equal(partition, expected):
        raise ValueError("cache partitions do not align with manifest")
    if manifest.get("event", {}).get("scope") != "process_kernel":
        raise ValueError("wrong capture scope")
    return features, partition


def standardize(train: torch.Tensor, other: list[torch.Tensor]) -> tuple[
        torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    center = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp(min=SCALE_FLOOR)
    return center, scale, [((part - center) / scale).clamp(-CLIP, CLIP) for part in other]


def fit_conditional(training: torch.Tensor, nuisance: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit ridge E[PEBS | PT, PMU] only on benign training executions."""
    context = torch.cat((training[:, :PT_END], training[:, PEBS_END:]), dim=1)
    if nuisance is not None:
        if nuisance.shape != (len(training), 1):
            raise ValueError("nuisance must have one value per execution")
        context = torch.cat((context, nuisance), dim=1)
    target = training[:, PT_END:PEBS_END]
    bias = torch.ones((context.shape[0], 1), dtype=context.dtype)
    design = torch.cat((context, bias), dim=1)
    covariance = design.T @ design
    regularizer = torch.eye(covariance.shape[0], dtype=covariance.dtype)
    regularizer[-1, -1] = 0.0  # The conditional mean is not penalized.
    weights = torch.linalg.solve(
        covariance + RIDGE_FRACTION * design.shape[0] * regularizer,
        design.T @ target,
    )
    residual = target - design @ weights
    residual_scale = residual.square().mean(dim=0).sqrt().clamp(min=SCALE_FLOOR)
    return weights, residual_scale, target.square().mean(dim=0).sqrt().clamp(min=SCALE_FLOOR)


def score(values: torch.Tensor, weights: torch.Tensor,
          residual_scale: torch.Tensor, *, nuisance: torch.Tensor | None = None,
          batch_size: int = 2048) -> torch.Tensor:
    if nuisance is not None and nuisance.shape != (len(values), 1):
        raise ValueError("nuisance must have one value per execution")
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            chunk = values[start:start + batch_size]
            context = torch.cat((chunk[:, :PT_END], chunk[:, PEBS_END:]), dim=1)
            if nuisance is not None:
                context = torch.cat((context, nuisance[start:start + batch_size]), dim=1)
            design = torch.cat((context, torch.ones((len(chunk), 1))), dim=1)
            residual = (chunk[:, PT_END:PEBS_END] - design @ weights) / residual_scale
            chunks.append(residual.square().mean(dim=1))
    return torch.cat(chunks)


def marginal_score(values: torch.Tensor, target_scale: torch.Tensor) -> torch.Tensor:
    return ((values[:, PT_END:PEBS_END] / target_scale).square()).mean(dim=1)


def threshold_report(scores: list[torch.Tensor], partition: torch.Tensor,
                     entries: list[dict[str, object]], rate: float) -> dict[str, object]:
    calibration, familiar, heldout = scores
    tail = max(1, int(len(calibration) * rate))
    threshold = float(torch.topk(calibration, tail + 1).values[-1])
    result: dict[str, object] = {
        "threshold": threshold, "calibration_alerts": int((calibration > threshold).sum()),
        "calibration_executions": len(calibration),
    }
    for name, values, code in (("familiar_validation", familiar, 2),
                               ("heldout_family", heldout, 3)):
        mask = values > threshold
        result[name] = {"executions": len(values), "alerts": int(mask.sum()),
                        "alerts_per_million": 1_000_000 * float(mask.float().mean())}
        names = [entry["family"] for entry, part in zip(entries, partition.tolist()) if part == code]
        result[name]["by_family"] = {
            family: {"executions": sum(item == family for item in names),
                     "alerts": sum(bool(hit) for item, hit in zip(names, mask.tolist()) if item == family)}
            for family in sorted(set(names))
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-sha256", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    manifest_path = args.corpus / "capture-manifest.json"
    manifest_hash = sha256(manifest_path)
    cache_hash = sha256(args.cache)
    if cache_hash != args.cache_sha256:
        raise ValueError("compact cache SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    features, partition = validate_cache(cache, manifest, manifest_hash)
    loaded_seconds = time.perf_counter() - started
    parts = [features[partition == index] for index in range(4)]
    center, scale, standardized = standardize(parts[0], parts)
    training = standardized[0]
    # Operator-requested loop count is known before capture and contains no
    # result label.  This diagnostic tests one explicit nuisance covariate.
    loop_values = torch.tensor([float(entry["loops"]) for entry in manifest["entries"]])
    if not bool((loop_values > 0).all()):
        raise ValueError("loop count must be positive")
    log_loops = loop_values.log2()
    train_loops = log_loops[partition == 0]
    loop_center = train_loops.mean()
    loop_scale = train_loops.std(unbiased=False).clamp(min=SCALE_FLOOR)
    nuisance_parts = [((log_loops[partition == index] - loop_center) / loop_scale)
                      .clamp(-CLIP, CLIP).reshape(-1, 1) for index in range(4)]
    fit_started = time.perf_counter()
    weights, residual_scale, target_scale = fit_conditional(training)
    fit_seconds = time.perf_counter() - fit_started
    evaluated = standardized[1:]
    score_started = time.perf_counter()
    conditional = [score(values, weights, residual_scale) for values in evaluated]
    score_seconds = time.perf_counter() - score_started
    ablation_started = time.perf_counter()
    conditioned_weights, conditioned_scale, _ = fit_conditional(training, nuisance_parts[0])
    conditioned = [score(values, conditioned_weights, conditioned_scale,
                         nuisance=nuisance_parts[index])
                   for index, values in enumerate(evaluated, start=1)]
    ablation_seconds = time.perf_counter() - ablation_started
    marginal = [marginal_score(values, target_scale) for values in evaluated]
    if not all(bool(torch.isfinite(row).all()) for row in conditional + conditioned + marginal):
        raise ValueError("nonfinite anomaly score")
    result = {
        "schema": SCHEMA, "host": platform.node(), "machine": platform.machine(),
        "device": "cpu", "torch": torch.__version__,
        "manifest_sha256": manifest_hash, "cache_sha256": cache_hash,
        "subject_identity_sha256": manifest["subject_identity_sha256"],
        "feature_count": FEATURE_COUNT, "context_count": PT_END + FEATURE_COUNT - PEBS_END,
        "target_count": PEBS_END - PT_END,
        "parameter_count": weights.numel() + residual_scale.numel(),
        "load_seconds": loaded_seconds, "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "score_executions_per_second": sum(len(x) for x in evaluated) / score_seconds,
        "loop_ablation_fit_and_score_seconds": ablation_seconds,
        "loop_ablation_parameter_count": conditioned_weights.numel() + conditioned_scale.numel(),
        "ridge_fraction": RIDGE_FRACTION, "scale_floor": SCALE_FLOOR, "clip": CLIP,
        "train_conditional_residual_mse": float((
            (training[:, PT_END:PEBS_END] -
             torch.cat((training[:, :PT_END], training[:, PEBS_END:],
                        torch.ones((len(training), 1))), dim=1) @ weights).square().mean())),
        "conditional": {str(rate): threshold_report(conditional, partition, manifest["entries"], rate)
                        for rate in (1e-3, 1e-4)},
        "conditional_with_log2_loops": {
            str(rate): threshold_report(conditioned, partition, manifest["entries"], rate)
            for rate in (1e-3, 1e-4)
        },
        "marginal_control": {str(rate): threshold_report(marginal, partition, manifest["entries"], rate)
                             for rate in (1e-3, 1e-4)},
        "warning": "Single-session development evidence; 21k calibration has only about two tail observations at 1e-4.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "conditional-pebs-r1.pt"
    torch.save({"schema": SCHEMA, "manifest_sha256": manifest_hash,
                "cache_sha256": cache_hash, "center": center, "scale": scale,
                "weights": weights, "residual_scale": residual_scale}, checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not torch.equal(conditional[0][:128], score(evaluated[0][:128], restored["weights"],
                                                   restored["residual_scale"])):
        raise RuntimeError("checkpoint reload changed scores")
    result["checkpoint_sha256"] = sha256(checkpoint)
    result["checkpoint_reload_bit_exact_cpu"] = True
    (args.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
