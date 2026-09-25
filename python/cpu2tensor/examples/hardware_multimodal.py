# SPDX-License-Identifier: AGPL-3.0-only
"""Small masked model for aligned PT, PEBS, and PMU fixture tensors.

The model treats CPUs as an unordered set.  A shared local encoder first models
each CPU independently, then a second encoder exchanges one summary per CPU.
Neither encoder receives an absolute CPU number or a cross-CPU event order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MultimodalConfig:
    """Fixed tensor and small-model dimensions."""

    pebs_features: int
    pmu_features: int
    segments: int = 16
    model_dimensions: int = 64
    attention_heads: int = 4
    feedforward_dimensions: int = 128
    local_layers: int = 2
    cross_cpu_layers: int = 2

    def __post_init__(self) -> None:
        dimensions = (
            self.pebs_features,
            self.pmu_features,
            self.segments,
            self.model_dimensions,
            self.attention_heads,
            self.feedforward_dimensions,
            self.local_layers,
            self.cross_cpu_layers,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all multimodal dimensions must be positive")
        if self.segments != 16:
            raise ValueError("the first multimodal contract has exactly 16 segments")
        if self.model_dimensions % self.attention_heads:
            raise ValueError("model dimensions must divide evenly across attention heads")

    @property
    def tokens_per_cpu(self) -> int:
        return 2 * self.segments + 1


@dataclass(frozen=True)
class HardwareMultimodalBatch:
    """Owned fixed tensors for complete executions.

    Timing bounds use one caller-selected clock.  They are intervals rather than
    a total order: simultaneous observations on different CPUs remain peers.
    Timing quality is in ``[0, 1]`` and lets coarse PT timing remain explicit.
    """

    pt: torch.Tensor
    pebs: torch.Tensor
    pmu: torch.Tensor
    pt_available: torch.Tensor
    pebs_available: torch.Tensor
    pmu_available: torch.Tensor
    time_bounds: torch.Tensor
    timing_quality: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.pt,
            self.pebs,
            self.pmu,
            self.pt_available,
            self.pebs_available,
            self.pmu_available,
            self.time_bounds,
            self.timing_quality,
        )
        if any(tensor.device != self.pt.device for tensor in tensors):
            raise ValueError("all multimodal tensors must be on one device")
        if any(tensor.dtype != torch.float32 for tensor in (
            self.pt, self.pebs, self.pmu, self.time_bounds, self.timing_quality
        )):
            raise ValueError("features, times, and timing quality must be float32")
        if any(tensor.dtype != torch.bool for tensor in (
            self.pt_available, self.pebs_available, self.pmu_available
        )):
            raise ValueError("availability tensors must be boolean")
        if self.pt.ndim != 4 or self.pt.shape[-2:] != (16, 256):
            raise ValueError("PT must have shape [batch, cpu, 16, 256]")
        batch, cpus = self.pt.shape[:2]
        if self.pebs.ndim != 4 or self.pebs.shape[:3] != (batch, cpus, 16):
            raise ValueError("PEBS must have shape [batch, cpu, 16, features]")
        if self.pmu.ndim != 4 or self.pmu.shape[:3] != (batch, cpus, 1):
            raise ValueError("PMU must have shape [batch, cpu, 1, features]")
        if self.pebs.shape[-1] == 0 or self.pmu.shape[-1] == 0:
            raise ValueError("PEBS and PMU need at least one feature")
        expected_masks = (
            (self.pt_available, (batch, cpus, 16)),
            (self.pebs_available, (batch, cpus, 16)),
            (self.pmu_available, (batch, cpus, 1)),
        )
        if any(tensor.shape != shape for tensor, shape in expected_masks):
            raise ValueError("availability shapes must match their modality tokens")
        tokens = 33
        if self.time_bounds.shape != (batch, cpus, tokens, 2):
            raise ValueError("time bounds must have shape [batch, cpu, 33, 2]")
        if self.timing_quality.shape != (batch, cpus, tokens):
            raise ValueError("timing quality must have shape [batch, cpu, 33]")
        available = self.token_availability
        values = (
            (self.pt, self.pt_available),
            (self.pebs, self.pebs_available),
            (self.pmu, self.pmu_available),
        )
        if any(not bool(torch.isfinite(value[mask]).all()) for value, mask in values):
            raise ValueError("available modality features must be finite")
        if not bool(torch.isfinite(self.time_bounds[available]).all()):
            raise ValueError("available token time bounds must be finite")
        if bool((self.time_bounds[..., 1][available] < self.time_bounds[..., 0][available]).any()):
            raise ValueError("time bounds must not run backwards")
        quality = self.timing_quality[available]
        if not bool(torch.isfinite(quality).all()) or bool(((quality < 0) | (quality > 1)).any()):
            raise ValueError("available timing quality must be finite and in [0, 1]")
        if bool((available.sum(dim=2) == 0).any()):
            raise ValueError("every CPU slot must contain at least one available token")

    @property
    def batch_size(self) -> int:
        return self.pt.shape[0]

    @property
    def cpu_count(self) -> int:
        return self.pt.shape[1]

    @property
    def token_availability(self) -> torch.Tensor:
        return torch.cat(
            (self.pt_available, self.pebs_available, self.pmu_available), dim=2
        )

    def index_select(self, indices: torch.Tensor) -> "HardwareMultimodalBatch":
        return HardwareMultimodalBatch(*(
            tensor.index_select(0, indices.to(tensor.device))
            for tensor in (
                self.pt,
                self.pebs,
                self.pmu,
                self.pt_available,
                self.pebs_available,
                self.pmu_available,
                self.time_bounds,
                self.timing_quality,
            )
        ))

    def to(self, device: str | torch.device) -> "HardwareMultimodalBatch":
        return HardwareMultimodalBatch(*(
            tensor.to(device)
            for tensor in (
                self.pt,
                self.pebs,
                self.pmu,
                self.pt_available,
                self.pebs_available,
                self.pmu_available,
                self.time_bounds,
                self.timing_quality,
            )
        ))


@dataclass(frozen=True)
class MaskedTokens:
    """Tokens hidden from the encoder but retained as reconstruction targets."""

    pt: torch.Tensor
    pebs: torch.Tensor
    pmu: torch.Tensor

    def __post_init__(self) -> None:
        if any(tensor.dtype != torch.bool or tensor.ndim != 3
               for tensor in (self.pt, self.pebs, self.pmu)):
            raise ValueError("masked-token tensors must be three-dimensional booleans")
        if self.pt.shape != self.pebs.shape or self.pmu.shape != (*self.pt.shape[:2], 1):
            raise ValueError("masked-token shapes do not match the fixed modalities")
        if self.pebs.device != self.pt.device or self.pmu.device != self.pt.device:
            raise ValueError("masked-token tensors must be on one device")


@dataclass(frozen=True)
class MultimodalPrediction:
    pt: torch.Tensor
    pebs: torch.Tensor
    pmu: torch.Tensor


def _encoder(config: MultimodalConfig, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=config.model_dimensions,
        nhead=config.attention_heads,
        dim_feedforward=config.feedforward_dimensions,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class MaskedHardwareModel(nn.Module):
    """Masked reconstruction without CPU-number or cross-CPU-order features."""

    def __init__(self, config: MultimodalConfig) -> None:
        super().__init__()
        self.config = config
        dimensions = config.model_dimensions
        self.pt_input = nn.Linear(256, dimensions)
        self.pebs_input = nn.Linear(config.pebs_features, dimensions)
        self.pmu_input = nn.Linear(config.pmu_features, dimensions)
        self.modality_embedding = nn.Embedding(3, dimensions)
        self.position_embedding = nn.Embedding(config.tokens_per_cpu, dimensions)
        self.time_input = nn.Linear(6, dimensions)
        self.mask_embedding = nn.Parameter(torch.empty(3, dimensions))
        self.cpu_summary = nn.Parameter(torch.empty(1, 1, dimensions))
        self.local_encoder = _encoder(config, config.local_layers)
        self.cross_cpu_encoder = _encoder(config, config.cross_cpu_layers)
        self.output_norm = nn.LayerNorm(dimensions)
        self.pt_output = nn.Linear(dimensions, 256)
        self.pebs_output = nn.Linear(dimensions, config.pebs_features)
        self.pmu_output = nn.Linear(dimensions, config.pmu_features)
        self.register_buffer("pt_mean", torch.zeros(256))
        self.register_buffer("pt_scale", torch.ones(256))
        self.register_buffer("pebs_mean", torch.zeros(config.pebs_features))
        self.register_buffer("pebs_scale", torch.ones(config.pebs_features))
        self.register_buffer("pmu_mean", torch.zeros(config.pmu_features))
        self.register_buffer("pmu_scale", torch.ones(config.pmu_features))
        self.register_buffer("normalization_fitted", torch.tensor(False))
        nn.init.normal_(self.mask_embedding, std=0.02)
        nn.init.normal_(self.cpu_summary, std=0.02)

    def _check_batch(self, batch: HardwareMultimodalBatch) -> None:
        if batch.pebs.shape[-1] != self.config.pebs_features:
            raise ValueError("PEBS feature width differs from the model")
        if batch.pmu.shape[-1] != self.config.pmu_features:
            raise ValueError("PMU feature width differs from the model")
        if batch.pt.shape[2] != self.config.segments:
            raise ValueError("PT segments differ from the model")
        if batch.pt.device != self.pt_mean.device:
            raise ValueError("batch and model must be on one device")

    @staticmethod
    def _statistics(values: torch.Tensor, available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected = values.reshape(-1, values.shape[-1])[available.reshape(-1)]
        if selected.shape[0] == 0:
            return torch.zeros(values.shape[-1], device=values.device), torch.ones(
                values.shape[-1], device=values.device
            )
        mean = selected.mean(0)
        scale = selected.std(0, correction=0).clamp_min(1e-4)
        return mean, scale

    @torch.no_grad()
    def fit_normalization(self, batch: HardwareMultimodalBatch) -> None:
        """Fit feature scaling once from the training partition."""
        self._check_batch(batch)
        for values, available, mean_buffer, scale_buffer in (
            (batch.pt, batch.pt_available, self.pt_mean, self.pt_scale),
            (batch.pebs, batch.pebs_available, self.pebs_mean, self.pebs_scale),
            (batch.pmu, batch.pmu_available, self.pmu_mean, self.pmu_scale),
        ):
            mean, scale = self._statistics(values, available)
            mean_buffer.copy_(mean)
            scale_buffer.copy_(scale)
        self.normalization_fitted.fill_(True)

    def standardized_targets(self, batch: HardwareMultimodalBatch) -> MultimodalPrediction:
        self._check_batch(batch)
        if not bool(self.normalization_fitted):
            raise RuntimeError("fit normalization on the training partition first")
        return MultimodalPrediction(
            (batch.pt - self.pt_mean) / self.pt_scale,
            (batch.pebs - self.pebs_mean) / self.pebs_scale,
            (batch.pmu - self.pmu_mean) / self.pmu_scale,
        )

    @staticmethod
    def _time_features(batch: HardwareMultimodalBatch) -> torch.Tensor:
        available = batch.token_availability
        lower = batch.time_bounds[..., 0]
        upper = batch.time_bounds[..., 1]
        infinity = torch.full_like(lower, torch.inf)
        origin = torch.where(available, lower, infinity).amin(dim=(1, 2), keepdim=True)
        end = torch.where(available, upper, -infinity).amax(dim=(1, 2), keepdim=True)
        duration = (end - origin).clamp_min(1e-6)
        center = ((lower + upper) * 0.5 - origin) / duration
        width = (upper - lower) / duration
        center = torch.where(available, center, torch.zeros_like(center))
        width = torch.where(available, width, torch.zeros_like(width))
        quality = torch.where(
            available, batch.timing_quality, torch.zeros_like(batch.timing_quality)
        )
        angle = center * (2.0 * torch.pi)
        return torch.stack(
            (angle.sin(), angle.cos(), (2.0 * angle).sin(), (2.0 * angle).cos(),
             width, quality),
            dim=-1,
        )

    def forward(
        self,
        batch: HardwareMultimodalBatch,
        masked: MaskedTokens,
    ) -> MultimodalPrediction:
        self._check_batch(batch)
        if masked.pt.shape != batch.pt_available.shape:
            raise ValueError("PT masks differ from the batch")
        if masked.pebs.shape != batch.pebs_available.shape:
            raise ValueError("PEBS masks differ from the batch")
        if masked.pmu.shape != batch.pmu_available.shape:
            raise ValueError("PMU masks differ from the batch")
        if masked.pt.device != batch.pt.device:
            raise ValueError("masks and batch must be on one device")
        if bool((masked.pt & ~batch.pt_available).any()
                or (masked.pebs & ~batch.pebs_available).any()
                or (masked.pmu & ~batch.pmu_available).any()):
            raise ValueError("unavailable tokens cannot be reconstruction targets")

        targets = self.standardized_targets(batch)
        projected = (
            self.pt_input(torch.where(
                batch.pt_available[..., None], targets.pt, torch.zeros_like(targets.pt)
            )),
            self.pebs_input(torch.where(
                batch.pebs_available[..., None], targets.pebs, torch.zeros_like(targets.pebs)
            )),
            self.pmu_input(torch.where(
                batch.pmu_available[..., None], targets.pmu, torch.zeros_like(targets.pmu)
            )),
        )
        hidden = []
        for modality, (tokens, token_mask) in enumerate(zip(
            projected, (masked.pt, masked.pebs, masked.pmu)
        )):
            replacement = self.mask_embedding[modality].view(1, 1, 1, -1)
            hidden.append(torch.where(token_mask[..., None], replacement, tokens))
        token_values = torch.cat(hidden, dim=2)
        batch_size, cpus, token_count, dimensions = token_values.shape
        modality_ids = torch.cat((
            torch.zeros(self.config.segments, dtype=torch.long, device=token_values.device),
            torch.ones(self.config.segments, dtype=torch.long, device=token_values.device),
            torch.full((1,), 2, dtype=torch.long, device=token_values.device),
        ))
        positions = torch.arange(token_count, device=token_values.device)
        token_values = (
            token_values
            + self.modality_embedding(modality_ids).view(1, 1, token_count, dimensions)
            + self.position_embedding(positions).view(1, 1, token_count, dimensions)
            + self.time_input(self._time_features(batch))
        )
        summary = self.cpu_summary.expand(batch_size, cpus, -1, -1)
        local_input = torch.cat((summary, token_values), dim=2).reshape(
            batch_size * cpus, token_count + 1, dimensions
        )
        padding = torch.cat((
            torch.zeros((batch_size, cpus, 1), dtype=torch.bool, device=token_values.device),
            ~batch.token_availability,
        ), dim=2).reshape(batch_size * cpus, token_count + 1)
        local = self.local_encoder(local_input, src_key_padding_mask=padding)
        cpu_summaries = local[:, 0].reshape(batch_size, cpus, dimensions)
        cpu_context = self.cross_cpu_encoder(cpu_summaries)
        tokens = local[:, 1:].reshape(batch_size, cpus, token_count, dimensions)
        tokens = self.output_norm(tokens + cpu_context[:, :, None, :])
        pt_end = self.config.segments
        pebs_end = 2 * self.config.segments
        return MultimodalPrediction(
            self.pt_output(tokens[:, :, :pt_end]),
            self.pebs_output(tokens[:, :, pt_end:pebs_end]),
            self.pmu_output(tokens[:, :, pebs_end:]),
        )


def make_training_masks(
    batch: HardwareMultimodalBatch,
    *,
    generator: torch.Generator,
    masked_fraction: float = 0.3,
    maximum_span: int = 4,
    whole_modality_probability: float = 0.3,
) -> MaskedTokens:
    """Create contiguous spans and occasional whole-modality masks."""
    if not 0 < masked_fraction <= 1 or maximum_span <= 0:
        raise ValueError("mask fraction and maximum span must be positive")
    if not 0 <= whole_modality_probability <= 1:
        raise ValueError("whole-modality probability must be in [0, 1]")
    available = [
        batch.pt_available.cpu(), batch.pebs_available.cpu(), batch.pmu_available.cpu()
    ]
    masks = [torch.zeros_like(modality) for modality in available]
    for row in range(batch.batch_size):
        whole = bool(torch.rand((), generator=generator) < whole_modality_probability)
        selected = int(torch.randint(3, (), generator=generator)) if whole else -1
        for modality, modality_available in enumerate(available):
            if modality == selected:
                masks[modality][row] = modality_available[row]
                continue
            for cpu in range(batch.cpu_count):
                valid_count = int(modality_available[row, cpu].sum())
                if valid_count == 0:
                    continue
                target = max(1, round(valid_count * masked_fraction))
                attempts = 0
                while int(masks[modality][row, cpu].sum()) < target and attempts < 32:
                    span = int(torch.randint(1, maximum_span + 1, (), generator=generator))
                    width = modality_available.shape[2]
                    start = int(torch.randint(width, (), generator=generator))
                    stop = min(width, start + span)
                    masks[modality][row, cpu, start:stop] |= modality_available[
                        row, cpu, start:stop
                    ]
                    attempts += 1
                if not bool(masks[modality][row, cpu].any()):
                    first = int(torch.nonzero(modality_available[row, cpu])[0])
                    masks[modality][row, cpu, first] = True
    return MaskedTokens(*(mask.to(batch.pt.device) for mask in masks))


def masked_reconstruction_loss(
    model: MaskedHardwareModel,
    batch: HardwareMultimodalBatch,
    masked: MaskedTokens,
) -> torch.Tensor:
    prediction = model(batch, masked)
    target = model.standardized_targets(batch)
    losses = []
    for predicted, expected, selected in (
        (prediction.pt, target.pt, masked.pt),
        (prediction.pebs, target.pebs, masked.pebs),
        (prediction.pmu, target.pmu, masked.pmu),
    ):
        per_token = F.smooth_l1_loss(predicted, expected, reduction="none").mean(-1)
        if bool(selected.any()):
            losses.append(per_token[selected].mean())
    if not losses:
        raise ValueError("at least one available token must be masked")
    return torch.stack(losses).mean()


def train_masked_model(
    model: MaskedHardwareModel,
    training: HardwareMultimodalBatch,
    *,
    steps: int = 50,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    masked_fraction: float = 0.3,
    maximum_span: int = 4,
    whole_modality_probability: float = 0.3,
    seed: int = 0,
) -> tuple[float, ...]:
    """Fit the small fixture model without a training framework."""
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("training arguments must be positive")
    if not bool(model.normalization_fitted):
        model.fit_normalization(training)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        count = min(batch_size, training.batch_size)
        indices = torch.randperm(training.batch_size, generator=generator)[:count]
        batch = training.index_select(indices)
        masked = make_training_masks(
            batch,
            generator=generator,
            masked_fraction=masked_fraction,
            maximum_span=maximum_span,
            whole_modality_probability=whole_modality_probability,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = masked_reconstruction_loss(model, batch, masked)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return tuple(losses)


def freeze_multimodal_model(model: MaskedHardwareModel) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _top_token_error(error: torch.Tensor, available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat_error = error.flatten(1)
    flat_available = available.flatten(1)
    ordered = flat_error.masked_fill(~flat_available, -torch.inf).sort(1, descending=True).values
    counts = flat_available.sum(1)
    take_count = counts.clamp(max=2)
    ranks = torch.arange(ordered.shape[1], device=ordered.device)[None, :]
    take = ranks < take_count[:, None]
    value = torch.where(take, ordered, torch.zeros_like(ordered)).sum(1) / take_count.clamp_min(1)
    return value, counts > 0


@torch.no_grad()
def multimodal_anomaly_score(
    model: MaskedHardwareModel,
    batch: HardwareMultimodalBatch,
) -> torch.Tensor:
    """Score every available token once with deterministic complementary masks."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("freeze the model before anomaly scoring")
    targets = model.standardized_targets(batch)
    errors = {
        "pt": torch.zeros_like(batch.pt_available, dtype=torch.float32),
        "pebs": torch.zeros_like(batch.pebs_available, dtype=torch.float32),
        "pmu": torch.zeros_like(batch.pmu_available, dtype=torch.float32),
    }
    empty_pt = torch.zeros_like(batch.pt_available)
    empty_pebs = torch.zeros_like(batch.pebs_available)
    empty_pmu = torch.zeros_like(batch.pmu_available)
    plans = []
    for name, available in (("pt", batch.pt_available), ("pebs", batch.pebs_available)):
        positions = torch.arange(available.shape[2], device=available.device)[None, None, :]
        for parity in (0, 1):
            selected = available & (positions % 2 == parity)
            masks = {"pt": empty_pt, "pebs": empty_pebs, "pmu": empty_pmu}
            masks[name] = selected
            plans.append((name, MaskedTokens(masks["pt"], masks["pebs"], masks["pmu"])))
        masks = {"pt": empty_pt, "pebs": empty_pebs, "pmu": empty_pmu}
        masks[name] = available
        plans.append((name, MaskedTokens(masks["pt"], masks["pebs"], masks["pmu"])))
    plans.append(("pmu", MaskedTokens(empty_pt, empty_pebs, batch.pmu_available)))
    for name, masked in plans:
        selected = getattr(masked, name)
        if not bool(selected.any()):
            continue
        prediction = model(batch, masked)
        per_token = F.smooth_l1_loss(
            getattr(prediction, name), getattr(targets, name), reduction="none"
        ).mean(-1)
        errors[name][selected] = torch.maximum(
            errors[name][selected], per_token[selected]
        )
    modality_scores = []
    modality_present = []
    for name, available in (
        ("pt", batch.pt_available),
        ("pebs", batch.pebs_available),
        ("pmu", batch.pmu_available),
    ):
        score, present = _top_token_error(errors[name], available)
        modality_scores.append(score)
        modality_present.append(present)
    scores = torch.stack(modality_scores, 1)
    present = torch.stack(modality_present, 1)
    result = torch.where(present, scores, torch.zeros_like(scores)).sum(1) / present.sum(1)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("multimodal scoring produced a non-finite value")
    return result


def calibrate_multimodal_threshold(
    model: MaskedHardwareModel,
    calibration: HardwareMultimodalBatch,
    *,
    reviews_per_million: float = 1_000.0,
) -> float:
    if not 0 < reviews_per_million < 1_000_000:
        raise ValueError("reviews per million must be between zero and one million")
    scores = multimodal_anomaly_score(model, calibration)
    quantile = 1.0 - reviews_per_million / 1_000_000.0
    return float(torch.quantile(scores, quantile, interpolation="higher"))


def save_frozen_multimodal_model(
    path: str | Path,
    model: MaskedHardwareModel,
    threshold: float,
) -> None:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("only a frozen model can be saved for inference")
    if not torch.isfinite(torch.tensor(threshold)):
        raise ValueError("threshold must be finite")
    torch.save({
        "config": asdict(model.config),
        "state": model.state_dict(),
        "threshold": float(threshold),
    }, path)


def load_frozen_multimodal_model(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MaskedHardwareModel, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    try:
        model = MaskedHardwareModel(MultimodalConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["state"])
        threshold = float(checkpoint["threshold"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("invalid frozen multimodal checkpoint") from error
    if not bool(model.normalization_fitted) or not torch.isfinite(torch.tensor(threshold)):
        raise ValueError("invalid frozen multimodal checkpoint state")
    freeze_multimodal_model(model)
    return model, threshold
