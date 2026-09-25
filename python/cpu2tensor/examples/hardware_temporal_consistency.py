# SPDX-License-Identifier: AGPL-3.0-only
"""Experimental temporal consistency objective for retained hardware captures.

The experiment keeps raw Intel PT undecoded.  Negative examples change only
PEBS timestamps inside each lane's own capture envelope, then pass the complete
capture through the production featurizer again.  No CPU number or cross-lane
event order enters the model.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import time
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from cpu2tensor.examples.hardware_multimodal import (
    HardwareMultimodalBatch as ModelBatch,
    MaskedHardwareModel,
    MaskedTokens,
    MultimodalConfig,
    freeze_multimodal_model,
    make_training_masks,
    masked_reconstruction_loss,
    multimodal_anomaly_score,
)
from cpu2tensor.examples.hardware_multimodal_experiment import (
    RAW_SCHEMA,
    _concatenate,
    _partition,
    load_dataset,
)
from cpu2tensor.examples.hardware_multimodal_features import (
    SEGMENTS,
    featurize_hardware_capture,
)
from cpu2tensor.hardware import (
    HardwareBatch,
    HardwareCaptureEnvelope,
    HardwareCounterBatch,
    HardwareMultimodalBatch,
    HardwareSourceStatus,
)


TEMPORAL_REPORT_SCHEMA = "cpu2tensor-temporal-consistency-experiment-v1"
BLOCK_SWAP = tuple((*range(8, 12), *range(4, 8), *range(0, 4), *range(12, 16)))


def cyclic_permutation(shift_bins: int) -> tuple[int, ...]:
    """Map each source time bin to another bin in the same lane."""
    if shift_bins == 0 or abs(shift_bins) >= SEGMENTS:
        raise ValueError("timestamp shift must be between one and fifteen bins")
    return tuple((index + shift_bins) % SEGMENTS for index in range(SEGMENTS))


def _hardware_batch(payload: dict[str, object] | None) -> HardwareBatch | None:
    if payload is None:
        return None
    names = (
        "ip", "pid", "tid", "time", "cpu", "period", "address", "weight",
        "data_source", "exact_ip", "trace_bytes",
    )
    return HardwareBatch(
        int(payload["source"]), str(payload["signal"]),
        *(payload[name] for name in names),
    )


def _counter_batch(payload: dict[str, object] | None) -> HardwareCounterBatch | None:
    if payload is None:
        return None
    return HardwareCounterBatch(
        int(payload["source"]), int(payload["tid"]), int(payload["cpu"]),
        tuple(payload["names"]), payload["values"],
        int(payload["time_enabled_ns"]), int(payload["time_running_ns"]),
        bool(payload["available"]), bool(payload["lost"]),
    )


def captures_from_raw_payload(payload: dict[str, object]) -> tuple[HardwareMultimodalBatch, ...]:
    """Rebuild immutable capture values without changing the custody file."""
    if payload.get("schema") != RAW_SCHEMA or not isinstance(payload.get("batches"), list):
        raise ValueError("raw capture schema changed")
    captures = []
    for row in payload["batches"]:
        envelope = row["envelope"]
        statuses = tuple(HardwareSourceStatus(**status) for status in row["status"])
        captures.append(HardwareMultimodalBatch(
            int(row["source"]), int(row["tid"]), int(row["cpu"]),
            HardwareCaptureEnvelope(**envelope), statuses,
            _hardware_batch(row["pt"]), _hardware_batch(row["pebs"]),
            _counter_batch(row["counters"]),
        ))
    return tuple(captures)


def _remap_sample_times(
    batch: HardwareBatch,
    envelope: HardwareCaptureEnvelope,
    permutation: Sequence[int],
) -> HardwareBatch:
    if batch.signal != "memory_loads" or len(permutation) != SEGMENTS:
        raise ValueError("PEBS remapping needs one destination for every time bin")
    if sorted(permutation) != list(range(SEGMENTS)):
        raise ValueError("timestamp destinations must be a permutation")
    start = envelope.arm_before_ns
    stop = envelope.stop_after_ns
    duration = stop - start
    offsets = batch.time - start
    source_bins = torch.clamp(offsets * SEGMENTS // duration, max=SEGMENTS - 1)
    mapped = torch.empty_like(batch.time)
    for source, destination in enumerate(permutation):
        selected = source_bins == source
        if not bool(selected.any()):
            continue
        source_start = duration * source // SEGMENTS
        source_stop = duration * (source + 1) // SEGMENTS
        destination_start = duration * destination // SEGMENTS
        destination_stop = duration * (destination + 1) // SEGMENTS
        source_width = max(1, source_stop - source_start)
        destination_width = max(1, destination_stop - destination_start)
        within = offsets[selected] - source_start
        mapped[selected] = (
            start + destination_start + within * destination_width // source_width
        ).clamp(max=stop - 1)
    order = mapped.argsort(stable=True)
    columns = {}
    for name in (
        "ip", "pid", "tid", "cpu", "period", "address", "weight",
        "data_source", "exact_ip",
    ):
        columns[name] = getattr(batch, name).index_select(0, order)
    return HardwareBatch(
        batch.source, batch.signal, columns["ip"], columns["pid"], columns["tid"],
        mapped.index_select(0, order), columns["cpu"], columns["period"],
        columns["address"], columns["weight"], columns["data_source"],
        columns["exact_ip"], batch.trace_bytes,
    )


def refeature_with_pebs_time_permutation(
    captures: Sequence[HardwareMultimodalBatch],
    permutation: Sequence[int],
) -> ModelBatch:
    """Re-featurize a coherent within-lane PEBS timestamp intervention."""
    shifted = []
    for capture in captures:
        if capture.pebs is None:
            raise ValueError("temporal consistency needs PEBS in every lane")
        shifted.append(replace(
            capture,
            pebs=_remap_sample_times(capture.pebs, capture.envelope, permutation),
        ))
    return featurize_hardware_capture(tuple(shifted)).batch


def load_temporal_interventions(
    artifact: Path,
    manifest: dict[str, object],
    permutations: dict[str, Sequence[int]],
) -> dict[str, dict[str, ModelBatch]]:
    """Read each verified raw shard once and derive all requested interventions."""
    result = {name: {} for name in permutations}
    for entry in manifest["entries"]:
        payload = torch.load(
            artifact / entry["raw_path"], map_location="cpu", weights_only=True
        )
        if payload.get("execution") != {
            name: entry[name] for name in (
                "execution_id", "family", "repetition", "partition"
            )
        }:
            raise ValueError(f"raw identity mismatch for {entry['execution_id']}")
        captures = captures_from_raw_payload(payload)
        for name, permutation in permutations.items():
            result[name][entry["execution_id"]] = refeature_with_pebs_time_permutation(
                captures, permutation
            )
    return result


def _empty_masks(batch: ModelBatch) -> MaskedTokens:
    return MaskedTokens(
        torch.zeros_like(batch.pt_available),
        torch.zeros_like(batch.pebs_available),
        torch.zeros_like(batch.pmu_available),
    )


class TemporalConsistencyModel(nn.Module):
    """One shared temporal energy head applied independently to every lane."""

    def __init__(self, config: MultimodalConfig) -> None:
        super().__init__()
        self.base = MaskedHardwareModel(config)
        self.temporal_head = nn.Linear(config.model_dimensions, 1)
        self.register_buffer("energy_mean", torch.tensor(0.0))
        self.register_buffer("energy_scale", torch.tensor(1.0))
        self.register_buffer("energy_calibrated", torch.tensor(False))

    def token_energy(self, batch: ModelBatch) -> torch.Tensor:
        tokens = self.base.encoded_tokens(batch, _empty_masks(batch))
        start = self.base.config.segments
        stop = 2 * start
        return self.temporal_head(tokens[:, :, start:stop]).squeeze(-1)

    def lane_energy(self, batch: ModelBatch) -> torch.Tensor:
        token = self.token_energy(batch)
        available = batch.pebs_available
        if bool((available.sum(2) == 0).any()):
            raise ValueError("temporal consistency needs PEBS in every lane")
        return torch.where(available, token, torch.zeros_like(token)).sum(2) / (
            available.sum(2).clamp_min(1)
        )

    def execution_energy(self, batch: ModelBatch) -> torch.Tensor:
        # All lanes are peers.  Mean aggregation is invariant to their order.
        return self.lane_energy(batch).mean(1)

    @torch.no_grad()
    def fit_energy_calibration(self, clean: ModelBatch) -> None:
        energy = self.execution_energy(clean)
        self.energy_mean.copy_(energy.mean())
        self.energy_scale.copy_(energy.std(correction=0).clamp_min(1e-4))
        self.energy_calibrated.fill_(True)

    def temporal_score(self, batch: ModelBatch) -> torch.Tensor:
        if not bool(self.energy_calibrated):
            raise RuntimeError("fit temporal calibration on clean training data")
        return F.relu((self.execution_energy(batch) - self.energy_mean) / self.energy_scale)


def train_temporal_consistency(
    model: TemporalConsistencyModel,
    clean: ModelBatch,
    shifted: Sequence[ModelBatch],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    rank_weight: float = 1.0,
    margin: float = 1.0,
    seed: int = 0,
) -> tuple[dict[str, float], ...]:
    """Joint masked reconstruction and clean-before-shifted energy ranking."""
    if not shifted or any(row.batch_size != clean.batch_size for row in shifted):
        raise ValueError("shifted training views must align with every clean execution")
    if steps <= 0 or batch_size <= 0 or rank_weight <= 0 or margin <= 0:
        raise ValueError("temporal training arguments must be positive")
    model.base.fit_normalization(clean)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    history = []
    for step in range(steps):
        count = min(batch_size, clean.batch_size)
        indices = torch.randperm(clean.batch_size, generator=generator)[:count]
        positive = clean.index_select(indices)
        negative = shifted[step % len(shifted)].index_select(indices)
        masks = make_training_masks(
            positive, generator=generator, whole_modality_probability=0.3
        )
        reconstruction = masked_reconstruction_loss(model.base, positive, masks)
        clean_energy = model.execution_energy(positive)
        shifted_energy = model.execution_energy(negative)
        ranking = F.softplus(clean_energy - shifted_energy + margin).mean()
        loss = reconstruction + rank_weight * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({
            "loss": float(loss.detach()),
            "reconstruction": float(reconstruction.detach()),
            "ranking": float(ranking.detach()),
        })
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.fit_energy_calibration(clean)
    freeze_multimodal_model(model.base)
    return tuple(history)


@torch.no_grad()
def temporal_consistency_anomaly_score(
    model: TemporalConsistencyModel,
    batch: ModelBatch,
    *,
    temporal_weight: float = 1.0,
) -> torch.Tensor:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("freeze the temporal model before anomaly scoring")
    return multimodal_anomaly_score(model.base, batch) + temporal_weight * model.temporal_score(batch)


def _auc(clean: torch.Tensor, shifted: torch.Tensor) -> float:
    comparison = shifted[:, None] - clean[None, :]
    return float(((comparison > 0).float() + 0.5 * (comparison == 0).float()).mean())


def _partition_from_rows(
    manifest: dict[str, object], rows: dict[str, ModelBatch], partition: str,
) -> ModelBatch:
    return _concatenate([
        rows[entry["execution_id"]]
        for entry in manifest["entries"] if entry["partition"] == partition
    ])


def _metrics(
    clean: torch.Tensor, shifted: torch.Tensor, threshold: float | None = None,
) -> dict[str, float]:
    clean_mean = float(clean.mean())
    shifted_mean = float(shifted.mean())
    result = {
        "clean_mean": clean_mean,
        "shifted_mean": shifted_mean,
        "mean_ratio": shifted_mean / max(clean_mean, 1e-12),
        "mean_delta": shifted_mean - clean_mean,
        "auroc": _auc(clean, shifted),
    }
    if threshold is not None:
        result.update({
            "threshold": threshold,
            "clean_alerts_per_thousand": (
                float((clean > threshold).sum()) * 1_000 / clean.numel()
            ),
            "shifted_alerts_per_thousand": (
                float((shifted > threshold).sum()) * 1_000 / shifted.numel()
            ),
        })
    return result


def _family_metrics(
    manifest: dict[str, object], clean_rows: dict[str, ModelBatch],
    shifted_rows: dict[str, ModelBatch], model: TemporalConsistencyModel,
) -> dict[str, dict[str, float | str]]:
    result = {}
    for family in manifest["split"]["families"]:
        entries = [entry for entry in manifest["entries"] if
                   entry["family"] == family and entry["partition"] in (
                       "familiar_validation", "heldout_family")]
        if not entries:
            continue
        clean = _concatenate([clean_rows[entry["execution_id"]] for entry in entries])
        shifted = _concatenate([shifted_rows[entry["execution_id"]] for entry in entries])
        result[family] = {
            "partition": entries[0]["partition"],
            **_metrics(model.temporal_score(clean), model.temporal_score(shifted)),
        }
    return result


def _localization(
    clean: ModelBatch, shifted: ModelBatch, model: TemporalConsistencyModel,
    affected: set[int],
) -> dict[str, float | int]:
    clean_token = model.token_energy(clean)
    shifted_token = model.token_energy(shifted)
    delta = (shifted_token - clean_token).abs().masked_fill(~shifted.pebs_available, -torch.inf)
    usable = shifted.pebs_available.all(2)
    if not bool(usable.any()):
        return {"executions": 0, "top_k_overlap": 0.0}
    delta = delta[usable]
    take = min(len(affected), SEGMENTS)
    top = delta.topk(take, dim=1).indices
    overlap = torch.zeros(top.shape[0])
    for index in affected:
        overlap += (top == index).any(1)
    return {
        "executions": int(usable.sum()),
        "top_k_overlap": float((overlap / take).mean()),
        "random_overlap": take / SEGMENTS,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    artifact = args.artifact.resolve()
    manifest, clean_rows = load_dataset(artifact)
    permutations = {
        "train_shift_1": cyclic_permutation(1),
        "train_shift_4": cyclic_permutation(4),
        "unseen_shift_2": cyclic_permutation(2),
        "unseen_shift_8": cyclic_permutation(8),
        "unseen_block_swap": BLOCK_SWAP,
    }
    started = time.perf_counter()
    interventions = load_temporal_interventions(artifact, manifest, permutations)
    refeature_seconds = time.perf_counter() - started

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    training = _partition(manifest, clean_rows, "training").to(device)
    shifted_training = [
        _partition_from_rows(manifest, interventions[name], "training").to(device)
        for name in ("train_shift_1", "train_shift_4")
    ]
    torch.manual_seed(args.seed)
    model = TemporalConsistencyModel(MultimodalConfig(
        pebs_features=training.pebs.shape[-1],
        pmu_features=training.pmu.shape[-1],
        model_dimensions=args.model_dimensions,
        attention_heads=args.attention_heads,
        feedforward_dimensions=args.feedforward_dimensions,
        local_layers=args.local_layers,
        cross_cpu_layers=args.cross_cpu_layers,
    )).to(device)
    started = time.perf_counter()
    history = train_temporal_consistency(
        model, training, shifted_training, steps=args.steps,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, rank_weight=args.rank_weight,
        margin=args.margin, seed=args.seed,
    )
    training_seconds = time.perf_counter() - started

    result: dict[str, object] = {
        "schema": TEMPORAL_REPORT_SCHEMA,
        "dataset_identity_sha256": manifest["identity_sha256"],
        "device": str(device),
        "host": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "system": platform.system(),
            "torch_version": torch.__version__,
            "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        },
        "training": {
            "steps": args.steps,
            "seconds": training_seconds,
            "loss_first": history[0],
            "loss_last": history[-1],
        },
        "raw_refeaturization_seconds": refeature_seconds,
        "partitions": {},
    }
    calibration = _partition(manifest, clean_rows, "calibration").to(device)
    calibration_temporal = model.temporal_score(calibration)
    calibration_combined = temporal_consistency_anomaly_score(model, calibration)
    quantile = 1.0 - args.reviews_per_million / 1_000_000.0
    temporal_threshold = float(torch.quantile(
        calibration_temporal, quantile, interpolation="higher"
    ))
    combined_threshold = float(torch.quantile(
        calibration_combined, quantile, interpolation="higher"
    ))
    result["calibration"] = {
        "executions": calibration.batch_size,
        "reviews_per_million": args.reviews_per_million,
        "pilot_only": True,
        "temporal_threshold": temporal_threshold,
        "combined_threshold": combined_threshold,
    }
    for partition in ("familiar_validation", "heldout_family"):
        clean = _partition(manifest, clean_rows, partition).to(device)
        clean_base = multimodal_anomaly_score(model.base, clean)
        clean_temporal = model.temporal_score(clean)
        clean_combined = temporal_consistency_anomaly_score(model, clean)
        views = {}
        for name in ("unseen_shift_2", "unseen_shift_8", "unseen_block_swap"):
            shifted = _partition_from_rows(manifest, interventions[name], partition).to(device)
            shifted_base = multimodal_anomaly_score(model.base, shifted)
            views[name] = {
                "base": _metrics(clean_base, shifted_base),
                "temporal": _metrics(
                    clean_temporal, model.temporal_score(shifted), temporal_threshold
                ),
                "combined": _metrics(
                    clean_combined, temporal_consistency_anomaly_score(model, shifted),
                    combined_threshold,
                ),
            }
            if name == "unseen_block_swap":
                views[name]["localization"] = _localization(
                    clean, shifted, model,
                    {index for index, destination in enumerate(BLOCK_SWAP)
                     if index != destination},
                )
        result["partitions"][partition] = views

    heldout_shift = interventions["unseen_shift_2"]
    result["heldout_family_shift_2"] = _family_metrics(
        manifest, clean_rows, heldout_shift, model
    )
    evaluation = _concatenate((
        _partition(manifest, clean_rows, "familiar_validation"),
        _partition(manifest, clean_rows, "heldout_family"),
    )).to(device)
    for _ in range(3):
        multimodal_anomaly_score(model.base, evaluation)
        temporal_consistency_anomaly_score(model, evaluation)
    repeats = args.throughput_repeats
    started = time.perf_counter()
    for _ in range(repeats):
        multimodal_anomaly_score(model.base, evaluation)
    base_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        temporal_consistency_anomaly_score(model, evaluation)
    temporal_seconds = time.perf_counter() - started
    result["throughput"] = {
        "executions": evaluation.batch_size,
        "repeats": repeats,
        "base_executions_per_second": repeats * evaluation.batch_size / base_seconds,
        "temporal_executions_per_second": repeats * evaluation.batch_size / temporal_seconds,
        "overhead_fraction": temporal_seconds / base_seconds - 1.0,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--model-dimensions", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dimensions", type=int, default=128)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--cross-cpu-layers", type=int, default=2)
    parser.add_argument("--throughput-repeats", type=int, default=10)
    parser.add_argument("--reviews-per-million", type=float, default=10_000.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_experiment(args)
    path = args.artifact / "temporal-consistency-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
