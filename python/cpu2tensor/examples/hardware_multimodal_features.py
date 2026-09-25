# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic bridge from perf capture batches to multimodal model tensors.

Raw PT bytes retain byte order at sixteen contiguous log-count histogram segments.
PEBS timestamps use their CLOCK_MONOTONIC_RAW clock, while PT and PMU receive the
full capture envelope.  In particular, this module never assigns timestamps to
PT byte positions or imposes an event order between lanes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import time
from typing import Any, Sequence

import torch

from cpu2tensor.examples.hardware_multimodal import (
    HardwareMultimodalBatch as ModelHardwareMultimodalBatch,
)
from cpu2tensor.hardware import (
    HardwareBatch,
    HardwareCaptureEnvelope,
    HardwareCounterBatch,
    HardwareMultimodalBatch as CaptureHardwareMultimodalBatch,
    HardwareSourceStatus,
)


MULTIMODAL_FEATURE_SCHEMA = "cpu2tensor-hardware-multimodal-features-v2"
SEGMENTS = 16
BYTE_VALUES = 256
DATA_SOURCE_BINS = 16
PEBS_FEATURES = (
    # Count/latency and exact-IP summaries precede a normalized hash sketch of
    # the untouched data-source bit pattern and relocation-invariant offsets.
    "log_sample_count",
    "mean_log_latency",
    "maximum_log_latency",
    "exact_ip_fraction",
    *(f"raw_data_source_{index:02d}" for index in range(DATA_SOURCE_BINS)),
    "mean_log_address_offset",
    "std_log_address_offset",
    "mean_log_ip_offset",
    "std_log_ip_offset",
)
PMU_FEATURES = (
    "instructions_per_second",
    "cycles_per_second",
    "reference_cycles_per_second",
    "instructions_per_cycle",
)

_MODALITIES = ("intel_pt", "memory_loads", "counters")
_COUNTERS = ("instructions", "cycles", "ref_cycles")
_UINT64_MASK = (1 << 64) - 1
_CAPTURE_FIELDS = (
    "source",
    "tid",
    "cpu",
    "envelope",
    "status",
    "pt",
    "pebs",
    "counters",
)
_ENVELOPE_FIELDS = (
    "clock",
    "arm_before_ns",
    "arm_after_ns",
    "stop_before_ns",
    "stop_after_ns",
)
_STATUS_FIELDS = (
    "signal",
    "requested",
    "available",
    "lost",
    "time_enabled_ns",
    "time_running_ns",
)
_COUNTER_FIELDS = (
    "source",
    "tid",
    "cpu",
    "names",
    "values",
    "time_enabled_ns",
    "time_running_ns",
    "available",
    "lost",
)
_HARDWARE_BATCH_FIELDS = (
    "source",
    "signal",
    "ip",
    "pid",
    "tid",
    "time",
    "cpu",
    "period",
    "address",
    "weight",
    "data_source",
    "exact_ip",
    "trace_bytes",
    "perf_records",
)


class HardwareFeatureError(ValueError):
    """A capture cannot be represented without weakening the feature contract."""


@dataclass(frozen=True)
class HardwareLaneReport:
    """Acquisition facts retained outside the permutation-invariant model.

    ``migration_verified`` means at least one PEBS sample was present and every
    sampled CPU agreed. Perf provides no evidence about unsampled intervals.
    """

    tid: int
    observed_cpu: int | None
    migration_verified: bool
    pebs_samples: int


@dataclass(frozen=True)
class HardwareCaptureFeatures:
    """One fixed model row and its non-model acquisition report."""

    batch: ModelHardwareMultimodalBatch
    lanes: tuple[HardwareLaneReport, ...]


def _require_schema(
    value: Any, expected_type: type[Any], expected_fields: tuple[str, ...]
) -> None:
    if not isinstance(value, expected_type):
        raise HardwareFeatureError(f"expected {expected_type.__name__} capture schema")
    if tuple(field.name for field in fields(value)) != expected_fields:
        raise HardwareFeatureError(f"{expected_type.__name__} capture schema changed")


def _validate_envelope(envelope: HardwareCaptureEnvelope) -> tuple[int, int, int, int]:
    _require_schema(envelope, HardwareCaptureEnvelope, _ENVELOPE_FIELDS)
    if envelope.clock != "CLOCK_MONOTONIC_RAW":
        raise HardwareFeatureError("capture clock must be CLOCK_MONOTONIC_RAW")
    points = (
        envelope.arm_before_ns,
        envelope.arm_after_ns,
        envelope.stop_before_ns,
        envelope.stop_after_ns,
    )
    if any(not isinstance(point, int) or isinstance(point, bool) for point in points):
        raise HardwareFeatureError(
            "capture envelope values must be integer nanoseconds"
        )
    if not points[0] <= points[1] < points[2] <= points[3]:
        raise HardwareFeatureError("capture envelope is empty or runs backwards")
    return points


def _validate_status(batch: CaptureHardwareMultimodalBatch) -> dict[str, bool]:
    if not isinstance(batch.status, tuple) or len(batch.status) != len(_MODALITIES):
        raise HardwareFeatureError("capture status schema changed")
    actual = {
        "intel_pt": batch.pt is not None,
        "memory_loads": batch.pebs is not None,
        "counters": batch.counters is not None,
    }
    for expected_signal, status in zip(_MODALITIES, batch.status):
        _require_schema(status, HardwareSourceStatus, _STATUS_FIELDS)
        if status.signal != expected_signal:
            raise HardwareFeatureError("capture status order or signal schema changed")
        if any(
            type(value) is not bool
            for value in (status.requested, status.available, status.lost)
        ):
            raise HardwareFeatureError("capture availability fields must be boolean")
        if status.lost:
            raise HardwareFeatureError(f"capture reported lost {status.signal} data")
        if status.available != actual[status.signal]:
            raise HardwareFeatureError(
                f"{status.signal} status disagrees with its payload"
            )
        if status.available and not status.requested:
            raise HardwareFeatureError(
                f"unrequested {status.signal} payload is present"
            )
        if status.requested and not status.available:
            raise HardwareFeatureError(
                f"requested {status.signal} source is unavailable"
            )
        timed = status.signal in ("memory_loads", "counters")
        if status.available and timed:
            enabled = status.time_enabled_ns
            running = status.time_running_ns
            if (
                not isinstance(enabled, int)
                or isinstance(enabled, bool)
                or not isinstance(running, int)
                or isinstance(running, bool)
                or enabled <= 0
                or running != enabled
            ):
                raise HardwareFeatureError(
                    f"{status.signal} source is empty or multiplexed"
                )
        elif status.time_enabled_ns is not None or status.time_running_ns is not None:
            raise HardwareFeatureError(
                f"{status.signal} has unexpected scheduling time"
            )
        if status.signal == "counters" and batch.counters is not None:
            if (
                status.time_enabled_ns != batch.counters.time_enabled_ns
                or status.time_running_ns != batch.counters.time_running_ns
            ):
                raise HardwareFeatureError(
                    "counter status timing disagrees with its payload"
                )
    return actual


def _validate_cpu_tensor(
    tensor: torch.Tensor, name: str, *, dtype: torch.dtype
) -> None:
    if tensor.device.type != "cpu" or tensor.dtype != dtype or tensor.ndim != 1:
        raise HardwareFeatureError(
            f"{name} must be a one-dimensional CPU {dtype} tensor"
        )


def _validate_hardware_batch(
    batch: HardwareBatch,
    *,
    signal: str,
    source: int,
) -> int:
    _require_schema(batch, HardwareBatch, _HARDWARE_BATCH_FIELDS)
    if batch.source != source or batch.signal != signal:
        raise HardwareFeatureError(f"{signal} source identity or signal changed")
    columns = (
        batch.ip,
        batch.pid,
        batch.tid,
        batch.time,
        batch.cpu,
        batch.period,
        batch.address,
        batch.weight,
        batch.data_source,
        batch.exact_ip,
    )
    for name, tensor in zip(_HARDWARE_BATCH_FIELDS[2:12], columns):
        _validate_cpu_tensor(tensor, f"{signal}.{name}", dtype=torch.int64)
    _validate_cpu_tensor(batch.trace_bytes, f"{signal}.trace_bytes", dtype=torch.uint8)
    if batch.perf_records is not None:
        _validate_cpu_tensor(
            batch.perf_records, f"{signal}.perf_records", dtype=torch.uint8
        )
    rows = batch.ip.numel()
    if any(tensor.numel() != rows for tensor in columns):
        raise HardwareFeatureError(f"{signal} columns have different lengths")
    return rows


def _pt_features(batch: HardwareBatch) -> tuple[torch.Tensor, torch.Tensor]:
    rows = _validate_hardware_batch(batch, signal="intel_pt", source=batch.source)
    if rows != 0:
        raise HardwareFeatureError("undecoded PT must not contain sample rows")
    trace = batch.trace_bytes
    result = torch.zeros((SEGMENTS, BYTE_VALUES), dtype=torch.float32)
    available = torch.zeros(SEGMENTS, dtype=torch.bool)
    length = trace.numel()
    for segment in range(SEGMENTS):
        start = length * segment // SEGMENTS
        stop = length * (segment + 1) // SEGMENTS
        if stop > start:
            available[segment] = True
            # PT volume varies by orders of magnitude across lawful workloads.
            # Compress counts before training so trace length cannot dominate
            # byte-distribution and cross-modal residuals.
            result[segment] = torch.log1p(torch.bincount(
                trace[start:stop].to(torch.int64), minlength=BYTE_VALUES
            ).to(torch.float32))
    return result, available


def _unsigned(value: int) -> int:
    return value & _UINT64_MASK


def _data_source_bin(value: int) -> int:
    # SplitMix64 finalizer: deterministic across Python and torch versions.
    mixed = _unsigned(value)
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9 & _UINT64_MASK
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EB & _UINT64_MASK
    return (mixed ^ (mixed >> 31)) % DATA_SOURCE_BINS


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _pebs_features(
    batch: HardwareBatch,
    *,
    tid: int,
    outer_start_ns: int,
    outer_stop_ns: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
    rows = _validate_hardware_batch(batch, signal="memory_loads", source=tid)
    if batch.trace_bytes.numel() != 0:
        raise HardwareFeatureError("PEBS sample batches must not contain AUX bytes")
    if rows == 0:
        return (
            torch.zeros((SEGMENTS, len(PEBS_FEATURES)), dtype=torch.float32),
            torch.zeros(SEGMENTS, dtype=torch.bool),
            torch.full((SEGMENTS, 2), torch.nan, dtype=torch.float32),
            None,
        )
    if any(sample_tid != tid for sample_tid in batch.tid.tolist()):
        raise HardwareFeatureError("PEBS sample escaped its thread lane")
    cpus = set(batch.cpu.tolist())
    if any(cpu < 0 for cpu in cpus) or len(cpus) != 1:
        raise HardwareFeatureError("PEBS observed thread migration")
    if any(period <= 0 for period in batch.period.tolist()):
        raise HardwareFeatureError("PEBS sample period must be positive")
    if any(value not in (0, 1) for value in batch.exact_ip.tolist()):
        raise HardwareFeatureError("PEBS exact-IP values must be zero or one")
    times = batch.time.tolist()
    if any(time < outer_start_ns or time > outer_stop_ns for time in times):
        raise HardwareFeatureError("PEBS timestamp escaped the capture envelope")

    duration_ns = outer_stop_ns - outer_start_ns
    addresses = [_unsigned(value) for value in batch.address.tolist()]
    ips = [_unsigned(value) for value in batch.ip.tolist()]
    address_anchor = min(addresses)
    ip_anchor = min(ips)
    buckets: list[list[int]] = [[] for _ in range(SEGMENTS)]
    for row, timestamp in enumerate(times):
        segment = min(
            SEGMENTS - 1, (timestamp - outer_start_ns) * SEGMENTS // duration_ns
        )
        buckets[segment].append(row)

    features = torch.zeros((SEGMENTS, len(PEBS_FEATURES)), dtype=torch.float32)
    available = torch.zeros(SEGMENTS, dtype=torch.bool)
    time_bounds = torch.full((SEGMENTS, 2), torch.nan, dtype=torch.float32)
    weights = [_unsigned(value) for value in batch.weight.tolist()]
    data_sources = batch.data_source.tolist()
    exact_ip = batch.exact_ip.tolist()
    for segment, selected in enumerate(buckets):
        if not selected:
            continue
        available[segment] = True
        selected_times = [times[row] - outer_start_ns for row in selected]
        time_bounds[segment] = torch.tensor(
            (
                min(selected_times) / 1_000_000_000.0,
                max(selected_times) / 1_000_000_000.0,
            )
        )
        count = len(selected)
        latency = [math.log1p(weights[row]) for row in selected]
        address_offsets = [
            math.log1p(addresses[row] - address_anchor) for row in selected
        ]
        ip_offsets = [math.log1p(ips[row] - ip_anchor) for row in selected]
        address_mean, address_std = _mean_std(address_offsets)
        ip_mean, ip_std = _mean_std(ip_offsets)
        sketch = [0.0] * DATA_SOURCE_BINS
        for row in selected:
            sketch[_data_source_bin(data_sources[row])] += 1.0 / count
        features[segment] = torch.tensor(
            (
                math.log1p(count),
                sum(latency) / count,
                max(latency),
                sum(exact_ip[row] for row in selected) / count,
                *sketch,
                address_mean,
                address_std,
                ip_mean,
                ip_std,
            ),
            dtype=torch.float32,
        )
    return features, available, time_bounds, next(iter(cpus))


def _pmu_features(
    batch: HardwareCounterBatch,
    *,
    tid: int,
    cpu: int,
) -> torch.Tensor:
    _require_schema(batch, HardwareCounterBatch, _COUNTER_FIELDS)
    if batch.source != tid or batch.tid != tid or batch.cpu != cpu:
        raise HardwareFeatureError("PMU source identity changed")
    if batch.names != _COUNTERS:
        raise HardwareFeatureError("PMU counter schema changed")
    _validate_cpu_tensor(batch.values, "counters.values", dtype=torch.int64)
    if batch.values.numel() != len(_COUNTERS) or bool((batch.values < 0).any()):
        raise HardwareFeatureError("PMU counters must be three nonnegative deltas")
    if batch.lost or not batch.available:
        raise HardwareFeatureError("PMU counter payload is unavailable or lost")
    if (
        batch.time_enabled_ns <= 0
        or batch.time_running_ns <= 0
        or batch.time_enabled_ns != batch.time_running_ns
    ):
        raise HardwareFeatureError("PMU counter group is empty or multiplexed")
    instructions, cycles, reference_cycles = batch.values.tolist()
    seconds = batch.time_running_ns / 1_000_000_000.0
    ipc = instructions / cycles if cycles else 0.0
    values = (
        instructions / seconds,
        cycles / seconds,
        reference_cycles / seconds,
        ipc,
    )
    if any(not math.isfinite(value) for value in values):
        raise HardwareFeatureError("PMU features are not finite")
    return torch.tensor(values, dtype=torch.float32).view(1, -1)


def featurize_hardware_capture(
    batches: Sequence[CaptureHardwareMultimodalBatch],
    *,
    phase_costs_ns: dict[str, int] | None = None,
) -> HardwareCaptureFeatures:
    """Convert one ``PerfMultimodalCapture.stop`` result to a model batch.

    The tensor result has batch size one. Lanes sort only by TID. CPU identity is
    report metadata rather than a semantic order or model feature.
    """
    if not batches:
        raise HardwareFeatureError("a capture must contain at least one thread lane")

    envelope_values: tuple[int, int, int, int] | None = None
    seen_tids: set[int] = set()
    lanes = []
    for batch in batches:
        _require_schema(batch, CaptureHardwareMultimodalBatch, _CAPTURE_FIELDS)
        if batch.source != batch.tid or batch.tid <= 0 or batch.cpu != -1:
            raise HardwareFeatureError("capture thread identity changed")
        if batch.tid in seen_tids:
            raise HardwareFeatureError("capture repeated a thread lane")
        seen_tids.add(batch.tid)
        envelope = _validate_envelope(batch.envelope)
        if envelope_values is None:
            envelope_values = envelope
        elif envelope != envelope_values:
            raise HardwareFeatureError("thread lanes use different capture envelopes")
        actual = _validate_status(batch)
        outer_start, arm_after, stop_before, outer_stop = envelope

        pt = torch.zeros((SEGMENTS, BYTE_VALUES), dtype=torch.float32)
        pt_available = torch.zeros(SEGMENTS, dtype=torch.bool)
        if actual["intel_pt"]:
            assert batch.pt is not None
            if batch.pt.source != batch.tid:
                raise HardwareFeatureError("PT source escaped its thread lane")
            started_ns = time.perf_counter_ns()
            pt, pt_available = _pt_features(batch.pt)
            if phase_costs_ns is not None:
                phase_costs_ns["pt_histogram"] = (
                    phase_costs_ns.get("pt_histogram", 0)
                    + time.perf_counter_ns() - started_ns
                )

        pebs = torch.zeros((SEGMENTS, len(PEBS_FEATURES)), dtype=torch.float32)
        pebs_available = torch.zeros(SEGMENTS, dtype=torch.bool)
        pebs_time_bounds = torch.full((SEGMENTS, 2), torch.nan, dtype=torch.float32)
        observed_cpu = None
        if actual["memory_loads"]:
            assert batch.pebs is not None
            started_ns = time.perf_counter_ns()
            pebs, pebs_available, pebs_time_bounds, observed_cpu = _pebs_features(
                batch.pebs,
                tid=batch.tid,
                outer_start_ns=outer_start,
                outer_stop_ns=outer_stop,
            )
            if phase_costs_ns is not None:
                phase_costs_ns["pebs_pmu_features"] = (
                    phase_costs_ns.get("pebs_pmu_features", 0)
                    + time.perf_counter_ns() - started_ns
                )
        if batch.cpu >= 0 and observed_cpu is not None and batch.cpu != observed_cpu:
            raise HardwareFeatureError("PEBS CPU disagrees with its capture lane")

        pmu = torch.zeros((1, len(PMU_FEATURES)), dtype=torch.float32)
        pmu_available = torch.zeros(1, dtype=torch.bool)
        if actual["counters"]:
            assert batch.counters is not None
            started_ns = time.perf_counter_ns()
            pmu = _pmu_features(batch.counters, tid=batch.tid, cpu=batch.cpu)
            if phase_costs_ns is not None:
                phase_costs_ns["pebs_pmu_features"] = (
                    phase_costs_ns.get("pebs_pmu_features", 0)
                    + time.perf_counter_ns() - started_ns
                )
            pmu_available.fill_(True)

        if not bool(pt_available.any() or pebs_available.any() or pmu_available.any()):
            raise HardwareFeatureError("thread lane contains no model observation")

        time_bounds = torch.full((2 * SEGMENTS + 1, 2), torch.nan, dtype=torch.float32)
        timing_quality = torch.full((2 * SEGMENTS + 1,), torch.nan, dtype=torch.float32)
        outer_seconds = (outer_stop - outer_start) / 1_000_000_000.0
        if bool(pt_available.any()):
            # All PT segments carry the same bound: byte order is not a clock.
            time_bounds[:SEGMENTS] = torch.tensor((0.0, outer_seconds))
            timing_quality[:SEGMENTS] = 0.0
        for segment in torch.nonzero(pebs_available).flatten().tolist():
            time_bounds[SEGMENTS + segment] = pebs_time_bounds[segment]
            timing_quality[SEGMENTS + segment] = 1.0
        if bool(pmu_available[0]):
            time_bounds[-1] = torch.tensor((0.0, outer_seconds))
            stable = stop_before - arm_after
            duration_ns = outer_stop - outer_start
            timing_quality[-1] = stable / duration_ns
        pebs_samples = 0 if batch.pebs is None else batch.pebs.ip.numel()
        lanes.append(
            (
                batch.tid,
                pt,
                pebs,
                pmu,
                pt_available,
                pebs_available,
                pmu_available,
                time_bounds,
                timing_quality,
                HardwareLaneReport(
                    tid=batch.tid,
                    observed_cpu=observed_cpu,
                    migration_verified=observed_cpu is not None,
                    pebs_samples=pebs_samples,
                ),
            )
        )

    lanes.sort(key=lambda lane: lane[0])
    columns = [
        torch.stack([lane[index] for lane in lanes]).unsqueeze(0)
        for index in range(1, 9)
    ]
    return HardwareCaptureFeatures(
        batch=ModelHardwareMultimodalBatch(*columns),
        lanes=tuple(lane[9] for lane in lanes),
    )


def featurize_hardware_captures(
    captures: Sequence[Sequence[CaptureHardwareMultimodalBatch]],
) -> tuple[ModelHardwareMultimodalBatch, tuple[tuple[HardwareLaneReport, ...], ...]]:
    """Stack complete executions after independently sorting their lanes."""
    if not captures:
        raise HardwareFeatureError("at least one capture is required")
    rows = [featurize_hardware_capture(capture) for capture in captures]
    lane_count = rows[0].batch.cpu_count
    if any(row.batch.cpu_count != lane_count for row in rows):
        raise HardwareFeatureError(
            "capture thread-lane count changed between executions"
        )
    columns = [
        torch.cat([getattr(row.batch, name) for row in rows], dim=0)
        for name in (
            "pt",
            "pebs",
            "pmu",
            "pt_available",
            "pebs_available",
            "pmu_available",
            "time_bounds",
            "timing_quality",
        )
    ]
    return ModelHardwareMultimodalBatch(*columns), tuple(row.lanes for row in rows)
