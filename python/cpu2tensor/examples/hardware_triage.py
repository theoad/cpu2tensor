# SPDX-License-Identifier: AGPL-3.0-only
"""Frozen anomaly triage directly from undecoded hardware-trace bytes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

try:
    from cpu2tensor import _native
except ImportError:  # The pure-Python source tree remains directly testable.
    _native = None


@dataclass(frozen=True)
class RawTraceSketchConfig:
    """A small order-sensitive sketch computed without decoding trace packets."""

    segments: int = 8
    pair_bins: int = 1024

    def __post_init__(self) -> None:
        if self.segments <= 0:
            raise ValueError("segments must be positive")
        if self.pair_bins <= 0 or self.pair_bins & (self.pair_bins - 1):
            raise ValueError("pair_bins must be a positive power of two")

    @property
    def feature_dimensions(self) -> int:
        return self.segments * 256 + self.pair_bins + 1


def pack_raw_traces(traces: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate owned one-dimensional byte tensors and return prefix offsets."""
    rows = tuple(traces)
    if not rows or any(row.dtype != torch.uint8 or row.ndim != 1 for row in rows):
        raise ValueError("traces must be nonempty one-dimensional uint8 tensors")
    if any(row.numel() == 0 for row in rows):
        raise ValueError("hardware traces must not be empty")
    device = rows[0].device
    if any(row.device != device for row in rows):
        raise ValueError("all traces must be on one device")
    lengths = torch.tensor([row.numel() for row in rows], dtype=torch.int64, device=device)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), lengths.cumsum(0)))
    return torch.cat(rows), offsets


def raw_trace_sketch(
    trace_bytes: torch.Tensor,
    offsets: torch.Tensor,
    config: RawTraceSketchConfig = RawTraceSketchConfig(),
) -> torch.Tensor:
    """Map concatenated raw bytes to segment and adjacent-byte frequencies.

    The operation parses no Intel PT, ETL, instruction, address, or branch packet.
    Segment histograms retain coarse order and hashed adjacent-byte pairs retain
    local order. One batched pass handles every execution in ``offsets``.
    """
    if trace_bytes.dtype != torch.uint8 or trace_bytes.ndim != 1:
        raise ValueError("trace_bytes must be a one-dimensional uint8 tensor")
    if offsets.dtype != torch.int64 or offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("offsets must be a one-dimensional int64 prefix sum")
    if offsets.device != trace_bytes.device:
        raise ValueError("trace bytes and offsets must be on one device")
    if int(offsets[0]) != 0 or int(offsets[-1]) != trace_bytes.numel():
        raise ValueError("offsets must cover every trace byte exactly")
    lengths = offsets[1:] - offsets[:-1]
    if bool((lengths <= 0).any() or (offsets[1:] < offsets[:-1]).any()):
        raise ValueError("offsets must describe nonempty traces in order")

    if _native is not None and trace_bytes.device.type == "cpu" and trace_bytes.is_contiguous():
        result = _native.raw_trace_sketch(
            trace_bytes.numpy(),
            offsets.contiguous().numpy(),
            config.segments,
            config.pair_bins,
        )
        return torch.frombuffer(result, dtype=torch.float32).reshape(
            lengths.numel(), config.feature_dimensions
        )

    batch = lengths.numel()
    device = trace_bytes.device
    trace_index = torch.repeat_interleave(torch.arange(batch, device=device), lengths)
    repeated_starts = torch.repeat_interleave(offsets[:-1], lengths)
    positions = torch.arange(trace_bytes.numel(), device=device) - repeated_starts
    repeated_lengths = torch.repeat_interleave(lengths, lengths)
    segment = torch.minimum(
        positions * config.segments // repeated_lengths,
        torch.tensor(config.segments - 1, device=device),
    )

    feature_dimensions = config.feature_dimensions
    byte_dimensions = config.segments * 256
    flat_bins = trace_index * feature_dimensions + segment * 256 + trace_bytes.to(torch.int64)
    features = torch.bincount(
        flat_bins,
        minlength=batch * feature_dimensions,
    ).reshape(batch, feature_dimensions).to(torch.float32)

    segment_bins = trace_index * config.segments + segment
    segment_counts = torch.bincount(
        segment_bins,
        minlength=batch * config.segments,
    ).reshape(batch, config.segments).clamp_min(1).to(torch.float32)
    features[:, :byte_dimensions] /= segment_counts.repeat_interleave(256, dim=1)

    pair_positions = positions + 1 < repeated_lengths
    current = trace_bytes[pair_positions].to(torch.int64)
    pair_indices = torch.nonzero(pair_positions, as_tuple=False).flatten()
    next_bytes = trace_bytes[pair_indices + 1].to(torch.int64)
    pair_key = (current << 8) | next_bytes
    pair_hash = (pair_key ^ (pair_key >> 7) ^ (pair_key >> 3)) & (config.pair_bins - 1)
    pair_trace = trace_index[pair_positions]
    pair_bins = pair_trace * feature_dimensions + byte_dimensions + pair_hash
    features += torch.bincount(
        pair_bins,
        minlength=batch * feature_dimensions,
    ).reshape(batch, feature_dimensions).to(torch.float32)
    features[:, byte_dimensions:-1] /= (lengths - 1).clamp_min(1).to(torch.float32)[:, None]
    features[:, -1] = torch.log1p(lengths.to(torch.float32)) / 16.0
    return features


class FrozenRawTracePCA:
    """A fitted linear autoencoder with no online updates or decoder inference."""

    def __init__(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
        components: torch.Tensor,
        config: RawTraceSketchConfig,
    ) -> None:
        self.mean = mean
        self.scale = scale
        self.components = components
        self.config = config

    def _standardize(
        self,
        features: torch.Tensor,
        config: RawTraceSketchConfig,
    ) -> torch.Tensor:
        if features.dtype != torch.float32 or features.ndim != 2:
            raise ValueError("features must be a two-dimensional float32 tensor")
        if config != self.config:
            raise ValueError("features use a different raw trace sketch configuration")
        if features.shape[1] != self.mean.numel() or features.device != self.mean.device:
            raise ValueError("features differ from the fitted device or dimensions")
        return (features - self.mean) / self.scale

    def latent(
        self,
        features: torch.Tensor,
        *,
        config: RawTraceSketchConfig,
    ) -> torch.Tensor:
        return self._standardize(features, config) @ self.components

    def anomaly_score(
        self,
        features: torch.Tensor,
        *,
        config: RawTraceSketchConfig,
    ) -> torch.Tensor:
        standardized = self._standardize(features, config)
        latent = standardized @ self.components
        residual_energy = standardized.square().sum(1) - latent.square().sum(1)
        return residual_energy.clamp_min(0) / standardized.shape[1]


def fit_raw_trace_pca(
    features: torch.Tensor,
    *,
    config: RawTraceSketchConfig = RawTraceSketchConfig(),
    latent_dimensions: int = 8,
    minimum_scale: float = 1e-4,
) -> FrozenRawTracePCA:
    """Fit a standardized linear autoencoder once on benign trace features."""
    if features.dtype != torch.float32 or features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must contain at least two float32 rows")
    if features.shape[1] != config.feature_dimensions:
        raise ValueError("features do not match the raw trace sketch contract")
    if not 0 < latent_dimensions < min(features.shape) or minimum_scale <= 0:
        raise ValueError("latent dimensions and minimum scale must fit the dataset")
    mean = features.mean(0)
    scale = features.std(0, correction=1).clamp_min(minimum_scale)
    standardized = (features - mean) / scale
    _, _, right = torch.linalg.svd(standardized, full_matrices=False)
    components = right[:latent_dimensions].transpose(0, 1).contiguous()
    return FrozenRawTracePCA(mean, scale, components, config)
