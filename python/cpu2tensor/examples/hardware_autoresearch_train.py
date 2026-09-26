# SPDX-License-Identifier: AGPL-3.0-only
"""One-file hardware anomaly training candidate for bounded autoresearch trials.

The CLI accepts only a sealed v2 TRAIN-ONLY compact cache.  It never opens a
calibration, validation, canary, or protected final-test artifact.  A separate
immutable evaluator imports ``load_frozen`` and ``score_features`` and controls
all thresholds and keep/discard decisions.

Architecture, objective, and optimizer choices intentionally live in this one
mutable file.  This candidate first removes train-predictable PT-byte and
PEBS-count exposure, then fits a small denoising autoencoder. Its score is a
mean standardized-feature reconstruction residual. Historical v1 checkpoints
remain loadable for read-only comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


TRAIN_SCHEMA_V1 = "cpu2tensor-autoresearch-train-v1"
TRAIN_SCHEMA = "cpu2tensor-autoresearch-train-v2"
CHECKPOINT_SCHEMA_V1 = "cpu2tensor-autoresearch-checkpoint-v1"
CHECKPOINT_SCHEMA = "cpu2tensor-autoresearch-checkpoint-v2"
TRIAL_SCHEMA = "cpu2tensor-autoresearch-trial-v2"
FEATURES = 789
TRAIN_ROWS = 56_000
EXPOSURE_FIELDS = ("pt_bytes", "elapsed_ns", "pebs_samples")
PEBS_INDICES = (512, 648)
RIDGE_PER_ROW = 0.001
SEEDS = (1729, 1730)
HIDDEN = 256
BOTTLENECK = 64
STEPS_PER_SEED = 12_000
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MASK_PROBABILITY = 0.15
NOISE_STD = 0.05
CPU_THREADS = 4
MAX_TRAIN_SECONDS = 240.0
SCORE_BATCH_SIZE = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DenoisingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(FEATURES, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, BOTTLENECK), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, FEATURES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(features))


class FrozenScorer:
    """Frozen model and training-only feature normalizer for evaluator use."""

    def __init__(self, model: DenoisingModel, center: torch.Tensor,
                 scale: torch.Tensor, seed: int,
                 exposure_correction: dict[str, torch.Tensor] | None = None) -> None:
        self.model = model
        self.center = center
        self.scale = scale
        self.seed = seed
        self.exposure_correction = exposure_correction


def validate_train_cache(payload: object, expected_rows: int = TRAIN_ROWS) -> tuple[torch.Tensor, dict[str, str]]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "features", "source_manifest_sha256", "source_cache_sha256",
    }:
        raise ValueError("training cache must contain only the agreed train-only fields")
    if payload["schema"] != TRAIN_SCHEMA_V1:
        raise ValueError("unsupported train-only cache schema")
    features = payload["features"]
    if (not isinstance(features, torch.Tensor) or
            features.dtype != torch.float32 or
            features.shape != (expected_rows, FEATURES) or
            features.device.type != "cpu" or
            not bool(torch.isfinite(features).all())):
        raise ValueError("train-only features must be finite CPU float32 [rows,789]")
    identity = {}
    for key in ("source_manifest_sha256", "source_cache_sha256"):
        value = payload[key]
        if (not isinstance(value, str) or len(value) != 64 or
                any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"invalid {key}")
        identity[key] = value
    return features, identity


def _valid_digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def validate_train_cache_v2(
    payload: object, expected_rows: int = TRAIN_ROWS,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Reject labels, nonfinite features, and malformed same-run exposures."""
    keys = {
        "schema", "features", "exposure_raw", "exposure_fields",
        "source_manifest_sha256", "source_cache_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != keys or payload["schema"] != TRAIN_SCHEMA:
        raise ValueError("v2 training cache must contain only agreed train-only fields")
    features = payload["features"]
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32 or
            features.device.type != "cpu" or features.shape != (expected_rows, FEATURES) or
            not bool(torch.isfinite(features).all())):
        raise ValueError("v2 features must be finite CPU float32 [rows,789]")
    exposure = payload["exposure_raw"]
    validate_exposure(exposure, expected_rows)
    if payload["exposure_fields"] != EXPOSURE_FIELDS:
        raise ValueError("unexpected v2 exposure field order")
    if not all(_valid_digest(payload[key]) for key in
               ("source_manifest_sha256", "source_cache_sha256")):
        raise ValueError("invalid v2 source identity")
    identity = {
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "source_cache_sha256": payload["source_cache_sha256"],
        "exposure_fields": EXPOSURE_FIELDS,
    }
    return features, exposure, identity


def validate_exposure(exposure: torch.Tensor, rows: int) -> None:
    if (not isinstance(exposure, torch.Tensor) or exposure.dtype != torch.int64 or
            exposure.device.type != "cpu" or exposure.shape != (rows, 3) or
            (rows and (not bool((exposure[:, :2] > 0).all()) or
                       not bool((exposure[:, 2] >= 0).all())))):
        raise ValueError("exposure_raw must be CPU int64 [rows,3] with valid counts")


def _exposure_design(
    exposure_raw: torch.Tensor, correction: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    values = exposure_raw.to(torch.float64)
    values[:, 1] /= 1_000_000.0  # elapsed nanoseconds to milliseconds
    logged = torch.log1p(values)
    normalized = (logged - correction["exposure_center"]) / correction["exposure_scale"]
    pt = normalized[:, (0, 1)] - correction["pt_design_center"]
    pebs = normalized[:, (2, 1)] - correction["pebs_design_center"]
    return pt, pebs


def fit_exposure_correction(features: torch.Tensor,
                            exposure_raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Fit R2 linear volume/time nuisance correction on train-only rows."""
    validate_exposure(exposure_raw, features.shape[0])
    values = exposure_raw.to(torch.float64)
    values[:, 1] /= 1_000_000.0
    logged = torch.log1p(values)
    exposure_center = logged.mean(dim=0)
    exposure_scale = logged.std(dim=0, unbiased=False).clamp(min=0.01)
    normalized = (logged - exposure_center) / exposure_scale
    pt = normalized[:, (0, 1)]
    pebs = normalized[:, (2, 1)]
    pt_design_center = pt.mean(dim=0)
    pebs_design_center = pebs.mean(dim=0)
    pt -= pt_design_center
    pebs -= pebs_design_center
    ridge = RIDGE_PER_ROW * features.shape[0] * torch.eye(2, dtype=torch.float64)
    pt_coefficients = torch.linalg.solve(
        pt.T @ pt + ridge, pt.T @ features[:, :512].to(torch.float64),
    )
    pebs_coefficients = torch.linalg.solve(
        pebs.T @ pebs + ridge,
        pebs.T @ features[:, PEBS_INDICES].to(torch.float64),
    )
    return {
        "exposure_center": exposure_center,
        "exposure_scale": exposure_scale,
        "pt_design_center": pt_design_center,
        "pebs_design_center": pebs_design_center,
        "pt_coefficients": pt_coefficients,
        "pebs_coefficients": pebs_coefficients,
    }


def condition_features(features: torch.Tensor, exposure_raw: torch.Tensor,
                       correction: dict[str, torch.Tensor]) -> torch.Tensor:
    validate_exposure(exposure_raw, features.shape[0])
    if (features.dtype != torch.float32 or features.ndim != 2 or
            features.shape[1] != FEATURES or not bool(torch.isfinite(features).all())):
        raise ValueError("conditioning requires finite float32 [rows,789]")
    pt, pebs = _exposure_design(exposure_raw, correction)
    conditioned = features.clone()
    conditioned[:, :512] -= (pt @ correction["pt_coefficients"]).to(torch.float32)
    conditioned[:, PEBS_INDICES] -= (
        pebs @ correction["pebs_coefficients"]
    ).to(torch.float32)
    if not bool(torch.isfinite(conditioned).all()):
        raise ValueError("exposure correction produced nonfinite features")
    return conditioned


def fit_normalizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp(min=0.05)
    return center, scale


def normalize(features: torch.Tensor, center: torch.Tensor,
              scale: torch.Tensor) -> torch.Tensor:
    return ((features - center) / scale).clamp(-20, 20)


def train_one(features: torch.Tensor, center: torch.Tensor, scale: torch.Tensor,
              *, seed: int, device: torch.device, deadline: float,
              steps: int = STEPS_PER_SEED, batch_size: int = BATCH_SIZE,
              ) -> tuple[DenoisingModel, dict[str, float | int]]:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("training steps and batch size must be positive")
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    model = DenoisingModel().to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    normalized = normalize(features, center, scale).to(device)
    started = time.perf_counter()
    first_loss = None
    last_loss = None
    for step in range(steps):
        if step % 64 == 0 and time.perf_counter() >= deadline:
            raise TimeoutError("training exceeded its per-seed deadline")
        indices = torch.randint(features.shape[0], (batch_size,), device=device)
        clean = normalized.index_select(0, indices)
        corrupt = torch.where(
            torch.rand_like(clean) < MASK_PROBABILITY, 0, clean,
        ) + NOISE_STD * torch.randn_like(clean)
        optimizer.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(model(corrupt), clean)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        if step == steps - 1:
            last_loss = float(loss.detach().cpu())
    if device.type == "mps":
        torch.mps.synchronize()
    training_seconds = time.perf_counter() - started
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert first_loss is not None and last_loss is not None
    return model, {
        "seed": seed, "steps": steps, "batch_size": batch_size,
        "first_loss": first_loss, "last_loss": last_loss,
        "training_seconds": training_seconds,
        "sampled_training_examples_per_second": steps * batch_size / training_seconds,
    }


def save_frozen(path: Path, model: DenoisingModel, center: torch.Tensor,
                scale: torch.Tensor, *, seed: int, steps: int,
                cache_identity: dict[str, str]) -> None:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("checkpoint requires a frozen model")
    torch.save({
        "schema": CHECKPOINT_SCHEMA_V1,
        "architecture": {"features": FEATURES, "hidden": HIDDEN, "bottleneck": BOTTLENECK},
        "state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "center": center.detach().cpu(), "scale": scale.detach().cpu(),
        "seed": seed, "steps": steps, "train_cache_identity": cache_identity,
    }, path)


def save_frozen_v2(path: Path, model: DenoisingModel, center: torch.Tensor,
                   scale: torch.Tensor, exposure_correction: dict[str, torch.Tensor],
                   *, seed: int, steps: int,
                   cache_identity: dict[str, object]) -> None:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("checkpoint requires a frozen model")
    if set(cache_identity) != {
            "source_manifest_sha256", "source_cache_sha256",
            "exposure_fields", "train_cache_sha256"}:
        raise ValueError("v2 checkpoint identity lacks physical training bytes")
    _validate_correction(exposure_correction)
    torch.save({
        "schema": CHECKPOINT_SCHEMA,
        "architecture": {"features": FEATURES, "hidden": HIDDEN, "bottleneck": BOTTLENECK},
        "state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "center": center.detach().cpu(), "scale": scale.detach().cpu(),
        "exposure_correction": {key: value.detach().cpu()
                                for key, value in exposure_correction.items()},
        "seed": seed, "steps": steps, "train_cache_identity": cache_identity,
    }, path)


def _validate_correction(correction: object) -> None:
    shapes = {
        "exposure_center": (3,), "exposure_scale": (3,),
        "pt_design_center": (2,), "pebs_design_center": (2,),
        "pt_coefficients": (2, 512), "pebs_coefficients": (2, 2),
    }
    if (not isinstance(correction, dict) or set(correction) != set(shapes) or
            any(not isinstance(correction[key], torch.Tensor) or
                correction[key].shape != shape or
                correction[key].dtype != torch.float64 or
                not bool(torch.isfinite(correction[key]).all())
                for key, shape in shapes.items()) or
            not bool((correction["exposure_scale"] > 0).all())):
        raise ValueError("invalid frozen exposure correction")


def load_frozen(path: str | Path, *, device: str | torch.device = "cpu") -> FrozenScorer:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") not in (CHECKPOINT_SCHEMA_V1, CHECKPOINT_SCHEMA) or payload.get("architecture") != {
        "features": FEATURES, "hidden": HIDDEN, "bottleneck": BOTTLENECK,
    }:
        raise ValueError("unsupported frozen autoresearch checkpoint")
    center = payload["center"]
    scale = payload["scale"]
    if (not isinstance(center, torch.Tensor) or center.shape != (FEATURES,) or
            not isinstance(scale, torch.Tensor) or scale.shape != (FEATURES,) or
            not bool(torch.isfinite(center).all() and torch.isfinite(scale).all()) or
            not bool((scale > 0).all())):
        raise ValueError("invalid frozen normalizer")
    model = DenoisingModel()
    model.load_state_dict(payload["state"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    correction = None
    if payload["schema"] == CHECKPOINT_SCHEMA:
        correction = payload.get("exposure_correction")
        _validate_correction(correction)
        identity = payload.get("train_cache_identity")
        if (not isinstance(identity, dict) or set(identity) != {
                "source_manifest_sha256", "source_cache_sha256",
                "exposure_fields", "train_cache_sha256"} or
                identity["exposure_fields"] != EXPOSURE_FIELDS or
                not all(_valid_digest(identity[key]) for key in (
                    "source_manifest_sha256", "source_cache_sha256",
                    "train_cache_sha256"))):
            raise ValueError("v2 checkpoint has invalid training-byte provenance")
    return FrozenScorer(model, center.to(device), scale.to(device),
                        int(payload["seed"]), correction)


def score_features(scorer: FrozenScorer, features: torch.Tensor,
                   exposure_raw: torch.Tensor | None = None,
                   *, batch_size: int = SCORE_BATCH_SIZE) -> torch.Tensor:
    """Return CPU float32 scores in caller order; evaluator owns input custody."""
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32 or
            features.ndim != 2 or features.shape[1] != FEATURES or
            not bool(torch.isfinite(features).all()) or batch_size <= 0):
        raise ValueError("scoring requires finite float32 [rows,789] and positive batch size")
    device = next(scorer.model.parameters()).device
    if scorer.exposure_correction is None and exposure_raw is not None:
        raise ValueError("v1 scorer does not accept v2 exposure")
    if scorer.exposure_correction is not None:
        validate_exposure(exposure_raw, features.shape[0])
    chunks = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start:start + batch_size]
            if scorer.exposure_correction is not None:
                batch = condition_features(
                    batch, exposure_raw[start:start + batch_size],
                    scorer.exposure_correction,
                )
            clean = normalize(batch.to(device), scorer.center, scorer.scale)
            residual = scorer.model(clean) - clean
            chunks.append(residual.square().mean(dim=1).to("cpu"))
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)


def run(train_cache: Path, output: Path, train_seconds: float) -> dict[str, Any]:
    if not 0 < train_seconds <= MAX_TRAIN_SECONDS:
        raise ValueError("train-seconds must be in (0, 240]")
    torch.set_num_threads(CPU_THREADS)
    started = time.perf_counter()
    train_cache_hash = sha256(train_cache)
    payload = torch.load(train_cache, map_location="cpu", weights_only=True)
    features, exposure_raw, identity = validate_train_cache_v2(payload)
    identity["train_cache_sha256"] = train_cache_hash
    correction = fit_exposure_correction(features, exposure_raw)
    conditioned = condition_features(features, exposure_raw, correction)
    center, scale = fit_normalizer(conditioned)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    output.mkdir(parents=True, exist_ok=False)
    training_started = time.perf_counter()
    deadline = training_started + train_seconds
    results = []
    for index, seed in enumerate(SEEDS):
        # Equal per-seed reservations ensure the second independent seed is
        # attempted without allowing the first to consume the whole budget.
        seed_deadline = training_started + train_seconds * (index + 1) / len(SEEDS)
        model, metrics = train_one(
            conditioned, center, scale, seed=seed, device=device,
            deadline=min(seed_deadline, deadline),
        )
        checkpoint = output / f"seed-{seed}.pt"
        save_frozen_v2(checkpoint, model, center, scale, correction, seed=seed,
                       steps=STEPS_PER_SEED, cache_identity=identity)
        restored = load_frozen(checkpoint, device=device)
        scorer = FrozenScorer(model, center.to(device), scale.to(device), seed, correction)
        preview = features[:128]
        reload_exact = torch.equal(
            score_features(scorer, preview, exposure_raw[:128]),
            score_features(restored, preview, exposure_raw[:128]),
        )
        if not reload_exact:
            raise RuntimeError("frozen checkpoint reload changed training-row scores")
        score_started = time.perf_counter()
        preview_scores = score_features(restored, features[:8192], exposure_raw[:8192])
        score_seconds = time.perf_counter() - score_started
        results.append({
            **metrics, "checkpoint": checkpoint.name,
            "checkpoint_reload_bit_exact": reload_exact,
            "training_score_preview_mean": float(preview_scores.mean()),
            "training_score_preview_rows": preview_scores.numel(),
            "score_seconds": score_seconds,
            "in_memory_scoring_executions_per_second": preview_scores.numel() / score_seconds,
        })
    if sha256(train_cache) != train_cache_hash:
        raise ValueError("train-only cache changed during v2 training")
    report = {
        "schema": TRIAL_SCHEMA, "host": platform.node(),
        "machine": platform.machine(), "device": str(device),
        "torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
        "train_cache_schema": TRAIN_SCHEMA, "train_cache_identity": identity,
        "train_rows": features.shape[0], "feature_dimension": FEATURES,
        "exposure_fields": EXPOSURE_FIELDS,
        "train_seconds_cap": train_seconds, "training_wall_seconds": time.perf_counter() - training_started,
        "total_wall_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in DenoisingModel().parameters()),
        "config": {
            "seeds": SEEDS, "steps_per_seed": STEPS_PER_SEED, "batch_size": BATCH_SIZE,
            "hidden": HIDDEN, "bottleneck": BOTTLENECK,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "mask_probability": MASK_PROBABILITY, "noise_std": NOISE_STD,
            "exposure_correction": "train_only_linear_pt_and_pebs_volume_time",
            "ridge_per_row": RIDGE_PER_ROW,
        },
        "seeds": results,
    }
    (output / "train.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-seconds", type=float, default=MAX_TRAIN_SECONDS)
    args = parser.parse_args()
    report = run(args.train_cache, args.output, args.train_seconds)
    print(json.dumps({
        "schema": report["schema"], "host": report["host"],
        "device": report["device"], "total_wall_seconds": report["total_wall_seconds"],
        "seeds": [entry["seed"] for entry in report["seeds"]],
        "parameter_count": report["parameter_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
