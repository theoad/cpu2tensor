# SPDX-License-Identifier: AGPL-3.0-only
"""Offline compact-feature anomaly baselines for one sealed hardware corpus.

This experiment never captures a target or reads a vulnerability canary.  Its
three benign scores are frozen before either validation partition is inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F


PARTITIONS = ("training", "calibration", "familiar_validation", "heldout_family")
FEATURE_SCHEMA = "cpu2tensor-compact-anomaly-r1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_features(payload: dict[str, object]) -> torch.Tensor:
    batch = payload["batch"]
    pt = batch["pt"][0, 0]
    pebs = batch["pebs"][0, 0]
    available = batch["pebs_available"][0, 0]
    safe_pebs = torch.where(available[:, None], pebs, 0.0)
    count = available.sum().clamp(min=1)
    pebs_mean = safe_pebs.sum(dim=0) / count
    pebs_std = torch.sqrt((safe_pebs.square().sum(dim=0) / count - pebs_mean.square()).clamp(min=0))
    pmu = batch["pmu"][0, 0, 0]
    # Segment moments preserve the byte vocabulary and PEBS feature channels,
    # while discarding the exact 16-step order.  This is a representation limit.
    features = torch.cat((
        pt.mean(dim=0), pt.std(dim=0, unbiased=False),
        pebs_mean, pebs_std,
        torch.sign(pmu) * torch.log1p(pmu.abs()),
        available.float().mean().reshape(1),
    ))
    if not bool(torch.isfinite(features).all()):
        raise ValueError("nonfinite compact feature")
    return features


def load_compact(corpus: Path, cache: Path) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    manifest_path = corpus / "capture-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = sha256(manifest_path)
    if (manifest.get("event", {}).get("scope") != "process_kernel" or
            len(manifest["entries"]) != 102_000):
        raise ValueError("this experiment expects the sealed 102k kernel-only corpus")
    if cache.exists():
        cached = torch.load(cache, map_location="cpu", weights_only=True)
        if (cached["schema"] != FEATURE_SCHEMA or
                cached["manifest_sha256"] != manifest_hash or
                cached["features"].shape[0] != len(manifest["entries"])):
            raise ValueError("compact cache does not match corpus identity")
        return manifest, cached["features"], cached["partition"]
    vectors = []
    partitions = []
    for index, entry in enumerate(manifest["entries"]):
        path = corpus / entry["derived_path"]
        if sha256(path) != entry["derived_sha256"]:
            raise ValueError(f"derived custody hash mismatch: {entry['execution_id']}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload["execution"]["execution_id"] != entry["execution_id"] or
                payload["raw_sha256"] != entry["raw_sha256"]):
            raise ValueError(f"derived identity mismatch: {entry['execution_id']}")
        vectors.append(compact_features(payload))
        partitions.append(PARTITIONS.index(entry["partition"]))
        if (index + 1) % 10_000 == 0:
            print(f"extracted {index + 1}/{len(manifest['entries'])}", flush=True)
    features = torch.stack(vectors)
    partition = torch.tensor(partitions, dtype=torch.uint8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": FEATURE_SCHEMA, "manifest_sha256": manifest_hash,
                "features": features, "partition": partition}, cache)
    return manifest, features, partition


class DenoisingAutoencoder(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dimension, 256), nn.GELU(),
                                     nn.Linear(256, 64), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(64, 256), nn.GELU(),
                                     nn.Linear(256, dimension))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(features))


def score_autoencoder(model: DenoisingAutoencoder, values: torch.Tensor,
                      batch_size: int) -> torch.Tensor:
    scores = []
    model.eval()
    with torch.inference_mode():
        for chunk in values.split(batch_size):
            error = (model(chunk) - chunk).square()
            # Keep a broad residual; a single unstable feature cannot dominate.
            scores.append(error.mean(dim=1))
    return torch.cat(scores)


def score_density(values: torch.Tensor, center: torch.Tensor,
                  components: torch.Tensor, scales: torch.Tensor,
                  residual_scale: torch.Tensor, batch_size: int) -> torch.Tensor:
    scores = []
    with torch.inference_mode():
        for chunk in values.split(batch_size):
            centered = chunk - center
            projected = centered @ components
            residual = centered.square().sum(dim=1) - projected.square().sum(dim=1)
            scores.append((projected.square() / scales).mean(dim=1) +
                          residual.clamp(min=0) / residual_scale)
    return torch.cat(scores)


def score_prototypes(values: torch.Tensor, prototypes: torch.Tensor,
                     batch_size: int) -> torch.Tensor:
    """Nearest training family/intensity centroid, without test-family labels."""
    scores = []
    with torch.inference_mode():
        prototype_norm = prototypes.square().sum(dim=1)
        for chunk in values.split(batch_size):
            distance = (chunk.square().sum(dim=1, keepdim=True) +
                        prototype_norm[None, :] - 2 * chunk @ prototypes.T)
            scores.append(distance.clamp(min=0).min(dim=1).values / values.shape[1])
    return torch.cat(scores)


def stratified_alerts(manifest: dict[str, object], partitions: torch.Tensor,
                      scores: list[torch.Tensor], threshold: float) -> dict[str, object]:
    rows = {}
    for partition_index, score_values in enumerate(scores, start=1):
        indices = torch.nonzero(partitions == partition_index).flatten().tolist()
        if len(indices) != len(score_values):
            raise ValueError("score/manifest partition order mismatch")
        for score, index in zip(score_values.tolist(), indices):
            entry = manifest["entries"][index]
            key = f"{entry['partition']}/{entry['family']}/loops={entry['loops']}"
            counts = rows.setdefault(key, {"executions": 0, "alerts": 0, "score_sum": 0.0})
            counts["executions"] += 1
            counts["alerts"] += int(score > threshold)
            counts["score_sum"] += score
    return {key: {"executions": value["executions"], "alerts": value["alerts"],
                  "mean_score": value["score_sum"] / value["executions"]}
            for key, value in sorted(rows.items())}


def threshold_and_counts(calibration: torch.Tensor, familiar: torch.Tensor,
                         heldout: torch.Tensor, false_positive_rate: float) -> dict[str, object]:
    # Conservative order statistic: the threshold is a calibration observation,
    # and an alert requires strictly greater score.
    tail_points = max(1, int(calibration.numel() * false_positive_rate))
    threshold = float(torch.topk(calibration, tail_points + 1).values[-1])
    return {
        "threshold": threshold, "calibration_tail_points": tail_points,
        "calibration_alerts": int((calibration > threshold).sum()),
        "familiar_alerts": int((familiar > threshold).sum()),
        "familiar_executions": familiar.numel(),
        "heldout_alerts": int((heldout > threshold).sum()),
        "heldout_executions": heldout.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    torch.manual_seed(240926)
    torch.set_num_threads(4)
    start = time.perf_counter()
    cache = args.output / "compact-features.pt"
    manifest, features, partitions = load_compact(args.corpus, cache)
    extraction_seconds = time.perf_counter() - start
    if features.shape[0] != 102_000 or features.shape[1] != 789:
        raise ValueError(f"unexpected compact feature shape: {tuple(features.shape)}")
    slices = [features[partitions == index] for index in range(4)]
    if [part.shape[0] for part in slices] != [56_000, 21_000, 7_000, 18_000]:
        raise ValueError("unexpected immutable split")
    train, calibration, familiar, heldout = slices
    center = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp(min=0.05)
    standardized = [((part - center) / scale).clamp(-20, 20) for part in slices]
    training, calibrated, familiar_test, heldout_test = standardized
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DenoisingAutoencoder(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    train_device = training.to(device)
    model.train()
    train_start = time.perf_counter()
    last_loss = None
    for step in range(args.steps):
        indices = torch.randint(training.shape[0], (args.batch_size,), device=device)
        clean = train_device.index_select(0, indices)
        corrupt = torch.where(torch.rand_like(clean) < 0.15, 0, clean)
        corrupt = corrupt + 0.05 * torch.randn_like(corrupt)
        optimizer.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(model(corrupt), clean)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    if device.type == "mps":
        torch.mps.synchronize()
    train_seconds = time.perf_counter() - train_start
    model.eval()
    # Training-only covariance fit.  A 64-D low-rank Gaussian supplies a second,
    # non-neural density objective; the diagonal score is its simpler control.
    covariance_start = time.perf_counter()
    covariance = training.T @ training / training.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, -64:].contiguous().to(device)
    scales = eigenvalues[-64:].clamp(min=0.05).to(device)
    residual_scale = (training.square().sum(dim=1).mean() - eigenvalues[-64:].sum()).clamp(min=0.05).to(device)
    covariance_seconds = time.perf_counter() - covariance_start
    train_entries = [entry for entry in manifest["entries"] if entry["partition"] == "training"]
    group_names = sorted({(entry["family"], entry["loops"]) for entry in train_entries})
    prototypes = torch.stack([
        training[torch.tensor([(entry["family"], entry["loops"]) == group
                               for entry in train_entries])].mean(dim=0)
        for group in group_names
    ]).to(device)
    center_zero = torch.zeros(features.shape[1], device=device)
    density_values = [part.to(device) for part in standardized[1:]]
    results = {}
    for name in ("denoising_autoencoder", "low_rank_density", "diagonal_density",
                 "nearest_training_prototype"):
        score_start = time.perf_counter()
        if name == "denoising_autoencoder":
            score = lambda data: score_autoencoder(model, data, args.batch_size)
        elif name == "low_rank_density":
            score = lambda data: score_density(data, center_zero, components, scales,
                                               residual_scale, args.batch_size)
        elif name == "nearest_training_prototype":
            score = lambda data: score_prototypes(data, prototypes, args.batch_size)
        else:
            score = lambda data: data.square().mean(dim=1)
        score_sets = [score(data) for data in density_values]
        if device.type == "mps":
            torch.mps.synchronize()
        score_seconds = time.perf_counter() - score_start
        cpu_scores = [set_.cpu() for set_ in score_sets]
        results[name] = {
            "parameter_count": (sum(p.numel() for p in model.parameters()) if name == "denoising_autoencoder" else
                                features.shape[1] * 64 + 64 if name == "low_rank_density" else
                                prototypes.numel() if name == "nearest_training_prototype" else features.shape[1]),
            "shared_normalizer_coefficients": 2 * features.shape[1],
            "scoring_seconds": score_seconds,
            "scoring_executions_per_second": 46_000 / score_seconds,
            "fpr_1e-3": threshold_and_counts(*cpu_scores, 1e-3),
            "fpr_1e-4_exploratory": threshold_and_counts(*cpu_scores, 1e-4),
            "stratified_at_1e-3": stratified_alerts(
                manifest, partitions, cpu_scores,
                threshold_and_counts(*cpu_scores, 1e-3)["threshold"],
            ),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "offline-model-r1.pt"
    torch.save({"schema": FEATURE_SCHEMA, "dataset_manifest_sha256": sha256(args.corpus / "capture-manifest.json"),
                "center": center, "scale": scale, "model": model.cpu().state_dict(),
                "components": components.cpu(), "density_scales": scales.cpu(),
                "residual_scale": residual_scale.cpu(), "prototypes": prototypes.cpu(),
                "prototype_group_names": group_names, "steps": args.steps}, checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored_model = DenoisingAutoencoder(features.shape[1])
    restored_model.load_state_dict(restored["model"])
    restored_model.eval()
    model.eval()
    with torch.inference_mode():
        reload_bit_exact = torch.equal(
            score_autoencoder(model, calibrated[:128], 64),
            score_autoencoder(restored_model, calibrated[:128], 64),
        )
    if not reload_bit_exact:
        raise RuntimeError("offline checkpoint reload changed scores")
    report = {
        "schema": FEATURE_SCHEMA, "host": platform.node(), "machine": platform.machine(),
        "device": str(device), "torch": torch.__version__,
        "dataset_manifest_sha256": sha256(args.corpus / "capture-manifest.json"),
        "dataset_identity_sha256": manifest["identity_sha256"],
        "partitions": dict(zip(PARTITIONS, [p.shape[0] for p in slices])),
        "feature_dimension": features.shape[1], "feature_extraction_seconds": extraction_seconds,
        "steps": args.steps, "batch_size": args.batch_size,
        "last_training_loss": last_loss, "training_seconds": train_seconds,
        "training_executions_per_second": args.steps * args.batch_size / train_seconds,
        "covariance_fit_seconds": covariance_seconds,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_reload_bit_exact_cpu": reload_bit_exact, "scores": results,
        "warning": "21k calibration provides only about two tail observations at 1e-4; neither threshold is an operational FPR certification.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
