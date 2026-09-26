# SPDX-License-Identifier: AGPL-3.0-only
"""Offline, non-model diagnostics for retained hardware capture shards.

PT byte positions have an order but no trusted timestamp here. PEBS samples have
timestamps, but a sampled store is not a complete memory-write stream. This
module deliberately does not assign PT bytes to PEBS time buckets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import time
from typing import Any

import torch


_PAGE_MASK = 4095
_JOINT_BINS = 2048
_OLD_JOINT_BINS = 32
_SEGMENTS = 16
_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class CaptureProbe:
    """Reproducible descriptors; vectors are counts, not learned features."""

    pt_bytes: int
    pt_histogram: torch.Tensor
    pt_bigrams: torch.Tensor
    pebs_samples: int
    pebs_exact_fraction: float
    pebs_time_span_fraction: float
    pebs_joint_counts: torch.Tensor
    pebs_unique_pairs: int
    pebs_old_joint_collisions: int
    pebs_new_joint_collisions: int
    duration_ns: int


def _mix64(value: int) -> int:
    value &= _UINT64_MASK
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _UINT64_MASK
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _UINT64_MASK
    return value ^ (value >> 31)


def _pt_descriptors(trace: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if trace.ndim != 1 or trace.dtype != torch.uint8:
        raise ValueError("raw PT must be a one-dimensional uint8 tensor")
    values = trace.to(torch.int64)
    histogram = torch.bincount(values, minlength=256)
    if values.numel() < 2:
        return histogram, torch.zeros(65536, dtype=torch.int64)
    # Exclude the fifteen boundaries between the original feature segments.
    # Their adjacent bytes were never represented together by that feature.
    keep = torch.ones(values.numel() - 1, dtype=torch.bool)
    for segment in range(1, _SEGMENTS):
        boundary = values.numel() * segment // _SEGMENTS
        if 0 < boundary < values.numel():
            keep[boundary - 1] = False
    pairs = values[:-1][keep] * 256 + values[1:][keep]
    return histogram, torch.bincount(pairs, minlength=65536)


def _pebs_descriptors(pebs: dict[str, Any], start: int, stop: int) -> tuple[Any, ...]:
    times = pebs["time"].tolist()
    ips = pebs["ip"].tolist()
    addresses = pebs["address"].tolist()
    exact = pebs["exact_ip"].tolist()
    cpus = pebs["cpu"].tolist()
    if not (len(times) == len(ips) == len(addresses) == len(exact) == len(cpus)):
        raise ValueError("PEBS columns have inconsistent lengths")
    if any(time < start or time > stop for time in times):
        raise ValueError("PEBS time escaped capture envelope")
    if len(set(cpus)) > 1:
        raise ValueError("PEBS lane migrated across sampled CPUs")
    if any(flag not in (0, 1) for flag in exact):
        raise ValueError("invalid exact-IP flag")
    if not times:
        return 0, math.nan, math.nan, torch.zeros(_JOINT_BINS, dtype=torch.int64), 0, 0, 0

    # Exact IPs are comparable only within this boot. The proposed vector uses
    # the IP page offset, which survives uniform relocation but aliases pages.
    pairs = {(ip & _UINT64_MASK, address & _PAGE_MASK) for ip, address in zip(ips, addresses)}
    anchor = min(ip & _UINT64_MASK for ip in ips)
    old_bins = {
        _mix64(((ip - anchor) & _UINT64_MASK) ^ (address << 32)) % _OLD_JOINT_BINS
        for ip, address in pairs
    }
    new_bins = {
        _mix64((ip & _PAGE_MASK) ^ (address << 32)) % _JOINT_BINS
        for ip, address in pairs
    }
    counts = torch.bincount(torch.tensor([
        _mix64((ip & _PAGE_MASK) ^ ((address & _PAGE_MASK) << 32)) % _JOINT_BINS
        for ip, address in zip(ips, addresses)
    ], dtype=torch.int64), minlength=_JOINT_BINS)
    return (
        len(times),
        sum(exact) / len(times),
        (max(times) - min(times)) / (stop - start),
        counts,
        len(pairs),
        len(pairs) - len(old_bins),
        len(pairs) - len(new_bins),
    )


def probe_raw_capture(raw: dict[str, Any]) -> CaptureProbe:
    """Describe one complete single-lane raw shard without modifying it."""
    batches = raw.get("batches")
    if not isinstance(batches, list) or len(batches) != 1:
        raise ValueError("probe requires exactly one preserved source lane")
    batch = batches[0]
    status = batch["status"]
    if [item["signal"] for item in status] != ["intel_pt", "memory_stores", "counters"]:
        raise ValueError("probe requires PT, precise stores, and PMU")
    if any(item["lost"] or not item["available"] for item in status):
        raise ValueError("a required modality is lost or unavailable")
    for item in status[1:]:
        if item["time_enabled_ns"] != item["time_running_ns"]:
            raise ValueError("PEBS or PMU was multiplexed")
    envelope = batch["envelope"]
    if envelope["clock"] != "CLOCK_MONOTONIC_RAW":
        raise ValueError("unknown capture clock")
    start, armed, stopping, stop = (
        envelope["arm_before_ns"], envelope["arm_after_ns"],
        envelope["stop_before_ns"], envelope["stop_after_ns"],
    )
    if not start <= armed < stopping <= stop:
        raise ValueError("invalid capture envelope")
    pt = batch["pt"]
    pebs = batch["pebs"]
    if pt is None or pebs is None or batch["counters"] is None:
        raise ValueError("capture payload is incomplete")
    histogram, bigrams = _pt_descriptors(pt["trace_bytes"])
    count, exact, span, joint, unique, old_collisions, new_collisions = (
        _pebs_descriptors(pebs, start, stop)
    )
    return CaptureProbe(
        pt_bytes=pt["trace_bytes"].numel(),
        pt_histogram=histogram,
        pt_bigrams=bigrams,
        pebs_samples=count,
        pebs_exact_fraction=exact,
        pebs_time_span_fraction=span,
        pebs_joint_counts=joint,
        pebs_unique_pairs=unique,
        pebs_old_joint_collisions=old_collisions,
        pebs_new_joint_collisions=new_collisions,
        duration_ns=stop - start,
    )


def _similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.dot(left, right).item()
    denominator = left.norm().item() * right.norm().item()
    return numerator / denominator if denominator else math.nan


def summarize_retained_audit(artifact: Path) -> dict[str, Any]:
    """Hash-check and summarize the preregistered raw audit selection only."""
    manifest_path = artifact / "capture-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = [entry for entry in manifest["entries"] if entry["raw_retained"]]
    if not entries:
        raise ValueError("manifest has no retained raw audit rows")
    records: list[tuple[dict[str, Any], CaptureProbe]] = []
    started = time.perf_counter()
    reduction_seconds = 0.0
    for entry in entries:
        raw_path = artifact / entry["raw_path"]
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != entry["raw_sha256"]:
            raise ValueError(f"raw shard hash mismatch: {entry['execution_id']}")
        raw = torch.load(raw_path, map_location="cpu", weights_only=True)
        reducing = time.perf_counter()
        records.append((entry, probe_raw_capture(raw)))
        reduction_seconds += time.perf_counter() - reducing
    elapsed = time.perf_counter() - started

    by_family_loops: dict[tuple[str, int], list[CaptureProbe]] = {}
    for entry, probe in records:
        by_family_loops.setdefault((entry["family"], entry["loops"]), []).append(probe)
    within: dict[str, list[float]] = {"pt_histogram": [], "pt_bigrams": [], "pebs_joint": []}
    early_late: dict[str, list[float]] = {name: [] for name in within}
    cross_intensity: dict[str, list[float]] = {name: [] for name in within}
    across: dict[str, list[float]] = {name: [] for name in within}
    vectors = {
        "pt_histogram": "pt_histogram",
        "pt_bigrams": "pt_bigrams",
        "pebs_joint": "pebs_joint_counts",
    }
    for probes in by_family_loops.values():
        if len(probes) < 2:
            continue
        for name, attribute in vectors.items():
            within[name].append(_similarity(
                getattr(probes[0], attribute).float(),
                getattr(probes[1], attribute).float(),
            ))
            early_late[name].append(_similarity(
                getattr(probes[0], attribute).float(),
                getattr(probes[-1], attribute).float(),
            ))
    for family in sorted({key[0] for key in by_family_loops}):
        intensities = sorted(key for key in by_family_loops if key[0] == family)
        if len(intensities) < 2:
            continue
        low = by_family_loops[intensities[0]][0]
        high = by_family_loops[intensities[-1]][0]
        for name, attribute in vectors.items():
            cross_intensity[name].append(_similarity(
                getattr(low, attribute).float(),
                getattr(high, attribute).float(),
            ))
    for key, probes in by_family_loops.items():
        alternatives = [other for other in by_family_loops
                        if other[1] == key[1] and other[0] != key[0]]
        if not alternatives:
            continue
        other = by_family_loops[sorted(alternatives)[0]][0]
        for name, attribute in vectors.items():
            across[name].append(_similarity(
                getattr(probes[0], attribute).float(),
                getattr(other, attribute).float(),
            ))
    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "retained_rows": len(records),
        "audit_partitions": {partition: sum(entry["partition"] == partition for entry, _ in records)
                             for partition in sorted({entry["partition"] for entry, _ in records})},
        "pt_bytes": sum(probe.pt_bytes for _, probe in records),
        "pebs_samples": sum(probe.pebs_samples for _, probe in records),
        "exact_ip_fraction_median": statistics.median(
            probe.pebs_exact_fraction for _, probe in records),
        "pebs_time_span_fraction_median": statistics.median(
            probe.pebs_time_span_fraction for _, probe in records),
        "exact_site_address_pairs_median": statistics.median(
            probe.pebs_unique_pairs for _, probe in records),
        "pt_bigram_occupied_median": statistics.median(
            int(torch.count_nonzero(probe.pt_bigrams)) for _, probe in records),
        "old_32_bin_joint_collisions_median": statistics.median(
            probe.pebs_old_joint_collisions for _, probe in records),
        "new_2048_bin_joint_collisions_median": statistics.median(
            probe.pebs_new_joint_collisions for _, probe in records),
        "within_same_family_and_loops_cosine_median": {
            name: statistics.median(values) for name, values in within.items()},
        "early_late_same_family_and_loops_cosine_median": {
            name: statistics.median(values) for name, values in early_late.items()},
        "within_family_cross_intensity_cosine_median": {
            name: statistics.median(values) for name, values in cross_intensity.items()},
        "across_family_same_loops_cosine_median": {
            name: statistics.median(values) for name, values in across.items()},
        "pair_counts": {"within": len(within["pt_histogram"]),
                        "across": len(across["pt_histogram"])},
        "elapsed_seconds": elapsed,
        "verified_rows_per_second": len(records) / elapsed,
        "reduction_only_seconds": reduction_seconds,
        "reduction_only_rows_per_second": len(records) / reduction_seconds,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def summarize_frozen_canary(artifact: Path) -> dict[str, Any]:
    """Post-choice readout only; labels do not fit features or a threshold."""
    report_bytes = (artifact / "report.json").read_bytes()
    report = json.loads(report_bytes)
    rows: dict[tuple[int, str], CaptureProbe] = {}
    for row in report["rows"]:
        key = (row["anonymous_pair"], row["arm"])
        if key in rows or key[1] not in ("effect", "neutral"):
            raise ValueError("canary pairing or arm is invalid")
        raw_path = artifact / row["raw_path"]
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != row["raw_sha256"]:
            raise ValueError("canary raw custody hash mismatch")
        rows[key] = probe_raw_capture(torch.load(
            raw_path, map_location="cpu", weights_only=True))
    pair_ids = sorted({key[0] for key in rows})
    if len(rows) != 2 * len(pair_ids):
        raise ValueError("canary is missing a paired arm")
    descriptors = {
        "pt_histogram": "pt_histogram",
        "pt_bigrams": "pt_bigrams",
        "pebs_joint": "pebs_joint_counts",
    }
    paired: dict[str, float] = {}
    adjacent: dict[str, float] = {}
    for name, attribute in descriptors.items():
        paired[name] = statistics.median(_similarity(
            getattr(rows[(pair_id, "effect")], attribute).float(),
            getattr(rows[(pair_id, "neutral")], attribute).float(),
        ) for pair_id in pair_ids)
        adjacent[name] = statistics.median(
            _similarity(
                getattr(rows[(pair_ids[index], arm)], attribute).float(),
                getattr(rows[(pair_ids[index + 1], arm)], attribute).float(),
            )
            for arm in ("effect", "neutral")
            for index in range(len(pair_ids) - 1)
        )
    arm_medians = {
        arm: {
            "pt_bytes": statistics.median(rows[(pair_id, arm)].pt_bytes for pair_id in pair_ids),
            "pebs_samples": statistics.median(rows[(pair_id, arm)].pebs_samples for pair_id in pair_ids),
            "pebs_unique_pairs": statistics.median(
                rows[(pair_id, arm)].pebs_unique_pairs for pair_id in pair_ids),
            "duration_ns": statistics.median(rows[(pair_id, arm)].duration_ns for pair_id in pair_ids),
        }
        for arm in ("effect", "neutral")
    }
    return {
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "paired_executions": len(pair_ids),
        "paired_effect_neutral_cosine_median": paired,
        "adjacent_same_arm_cosine_median": adjacent,
        "arm_medians": arm_medians,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--canary", action="store_true",
                        help="read frozen, archived paired canary evidence")
    args = parser.parse_args()
    summary = (summarize_frozen_canary(args.artifact) if args.canary else
               summarize_retained_audit(args.artifact))
    print(json.dumps(summary, indent=2, sort_keys=True))
