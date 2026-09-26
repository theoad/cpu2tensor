# SPDX-License-Identifier: AGPL-3.0-only
"""Offline per-lane raw-PT/PEBS grammar probe; no branch or PT-time decode.

PSB signatures divide raw bytes into ordinal spans. PEBS samples keep their
own CLOCK_MONOTONIC_RAW timestamps. A PT span has no assigned PEBS timestamp.
The fixed-size sketches below are diagnostics, not a production model schema.
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

import numpy as np
import torch

from cpu2tensor.examples.hardware_nuisance_sideband_probe_r1 import (
    core_text_offset,
    kernel_text_bounds,
    pt_options,
    psb_offsets,
)


_PT_ORDINAL_BINS = 16
_PT_BIGRAM_BINS = 128
_PEBS_TIME_BINS = 16
_PEBS_SITE_BINS = 256
_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class PtSpan:
    start: int
    stop: int


@dataclass(frozen=True)
class PebsStore:
    time_ns: int
    raw_ip: int
    core_text_offset: int | None
    raw_virtual_address: int
    address_page_offset: int
    data_source: int
    weight: int
    period: int
    exact_ip: bool
    cpu: int


@dataclass(frozen=True)
class LaneGrammar:
    tid: int
    pt_spans: tuple[PtSpan, ...]
    pebs_stores: tuple[PebsStore, ...]
    pt_ordinal_bigram: np.ndarray
    pebs_time_site_address: np.ndarray
    pt_bytes: int
    pt_psb_count: int
    pt_span_count: int
    pt_prefix_bytes_before_first_psb: int
    pebs_sample_count: int
    pebs_core_site_count: int
    pebs_exact_ip_count: int
    pebs_nonzero_address_count: int
    pebs_timestamp_inversions: int
    pebs_timestamp_ties: int
    sampled_cpus: tuple[int, ...]
    envelope_duration_ns: int


def _mix64(value: int) -> int:
    value &= _UINT64_MASK
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _UINT64_MASK
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _UINT64_MASK
    return value ^ (value >> 31)


def _pt_grammar(trace: torch.Tensor) -> tuple[tuple[PtSpan, ...], np.ndarray, int, int]:
    if trace.ndim != 1 or trace.dtype != torch.uint8:
        raise ValueError("PT AUX must be a one-dimensional uint8 tensor")
    raw = trace.numpy().tobytes()
    markers = psb_offsets(raw)
    starts = sorted({0, *markers})
    spans = tuple(PtSpan(start, stop) for start, stop in
                  zip(starts, [*starts[1:], len(raw)]) if stop > start)
    sketch = np.zeros((_PT_ORDINAL_BINS, _PT_BIGRAM_BINS), dtype=np.float32)
    if spans:
        # Equal numbers of candidate PSB spans per ordinal bin; byte length is
        # deliberately not treated as elapsed time.
        for ordinal in range(_PT_ORDINAL_BINS):
            first = len(spans) * ordinal // _PT_ORDINAL_BINS
            last = len(spans) * (ordinal + 1) // _PT_ORDINAL_BINS
            if last <= first:
                continue
            chunk = np.frombuffer(raw, dtype=np.uint8,
                                  count=spans[last - 1].stop - spans[first].start,
                                  offset=spans[first].start).astype(np.uint16)
            if chunk.size < 2:
                continue
            pairs = (chunk[:-1] << 8) | chunk[1:]
            bins = ((pairs.astype(np.uint32) * 0x9E37) ^
                    (pairs.astype(np.uint32) >> 7)) & (_PT_BIGRAM_BINS - 1)
            counts = np.bincount(bins, minlength=_PT_BIGRAM_BINS)
            sketch[ordinal] = counts / counts.sum()
    prefix = markers[0] if markers else len(raw)
    return spans, sketch, len(markers), prefix


def _pebs_grammar(pebs: dict[str, Any], bounds: tuple[int, int],
                  start_ns: int, stop_ns: int, tid: int) -> tuple[Any, ...]:
    fields = ("time", "ip", "address", "data_source", "weight", "period",
              "exact_ip", "cpu", "tid")
    columns = {name: pebs[name].tolist() for name in fields}
    if len({len(values) for values in columns.values()}) != 1:
        raise ValueError("PEBS columns have inconsistent lengths")
    stores: list[PebsStore] = []
    sketch = np.zeros((_PEBS_TIME_BINS, _PEBS_SITE_BINS), dtype=np.float32)
    exact_count = 0
    nonzero_count = 0
    core_count = 0
    for row in range(len(columns["time"])):
        timestamp = columns["time"][row]
        if not start_ns <= timestamp <= stop_ns:
            raise ValueError("PEBS timestamp escaped its capture envelope")
        if columns["tid"][row] != tid:
            raise ValueError("PEBS record escaped its lane")
        exact = columns["exact_ip"][row]
        if exact not in (0, 1):
            raise ValueError("invalid PEBS exact-IP flag")
        if columns["period"][row] <= 0:
            raise ValueError("invalid PEBS sample period")
        exact_count += exact
        ip = columns["ip"][row] & _UINT64_MASK
        address = columns["address"][row] & _UINT64_MASK
        nonzero_count += address != 0
        site = core_text_offset(ip, bounds)
        core_count += site is not None
        stores.append(PebsStore(
            time_ns=timestamp,
            raw_ip=ip,
            core_text_offset=site,
            raw_virtual_address=address,
            address_page_offset=address & 4095,
            data_source=columns["data_source"][row] & _UINT64_MASK,
            weight=columns["weight"][row] & _UINT64_MASK,
            period=columns["period"][row],
            exact_ip=bool(exact),
            cpu=columns["cpu"][row],
        ))
        bucket = min(_PEBS_TIME_BINS - 1,
                     (timestamp - start_ns) * _PEBS_TIME_BINS // (stop_ns - start_ns))
        # Keep a core/non-core tag; non-core raw IPs are exact-boot-only.
        tag = 0 if site is not None else 1
        site_identity = site if site is not None else ip
        mixed = _mix64(site_identity ^ ((address & 4095) << 32) ^
                       (columns["data_source"][row] << 1) ^ tag)
        sketch[bucket, mixed % _PEBS_SITE_BINS] += 1.0
    counts = sketch.sum(axis=1, keepdims=True)
    np.divide(sketch, counts, out=sketch, where=counts != 0)
    times = [store.time_ns for store in stores]
    inversions = sum(right < left for left, right in zip(times, times[1:]))
    ties = sum(right == left for left, right in zip(times, times[1:]))
    return tuple(stores), sketch, core_count, exact_count, nonzero_count, inversions, ties


def grammar_for_raw(raw: dict[str, Any], bounds: tuple[int, int]) -> tuple[LaneGrammar, ...]:
    lanes = []
    for batch in raw["batches"]:
        tid = batch["tid"]
        if batch["source"] != tid or batch["pt"] is None or batch["pebs"] is None:
            raise ValueError("source lane is incomplete or misattributed")
        if batch["pt"]["source"] != tid or batch["pebs"]["source"] != tid:
            raise ValueError("sensor payload escaped its source lane")
        statuses = batch["status"]
        if [item["signal"] for item in statuses] != ["intel_pt", "memory_stores", "counters"]:
            raise ValueError("unexpected sensor status")
        if any(item["lost"] or not item["available"] for item in statuses):
            raise ValueError("sensor loss or unavailability")
        if any(item["time_enabled_ns"] != item["time_running_ns"] for item in statuses[1:]):
            raise ValueError("PEBS or PMU multiplexing")
        envelope = batch["envelope"]
        if envelope["clock"] != "CLOCK_MONOTONIC_RAW":
            raise ValueError("unknown PEBS clock")
        start_ns = envelope["arm_before_ns"]
        stop_ns = envelope["stop_after_ns"]
        if start_ns >= stop_ns:
            raise ValueError("empty capture envelope")
        spans, pt_sketch, marker_count, prefix = _pt_grammar(batch["pt"]["trace_bytes"])
        (stores, pebs_sketch, core_count, exact_count, nonzero_count,
         inversions, ties) = _pebs_grammar(
            batch["pebs"], bounds, start_ns, stop_ns, tid)
        lanes.append(LaneGrammar(
            tid=tid, pt_spans=spans, pebs_stores=stores,
            pt_ordinal_bigram=pt_sketch,
            pebs_time_site_address=pebs_sketch,
            pt_bytes=batch["pt"]["trace_bytes"].numel(),
            pt_psb_count=marker_count,
            pt_span_count=len(spans),
            pt_prefix_bytes_before_first_psb=prefix,
            pebs_sample_count=len(stores),
            pebs_core_site_count=core_count,
            pebs_exact_ip_count=exact_count,
            pebs_nonzero_address_count=nonzero_count,
            pebs_timestamp_inversions=inversions,
            pebs_timestamp_ties=ties,
            sampled_cpus=tuple(sorted({store.cpu for store in stores})),
            envelope_duration_ns=stop_ns - start_ns,
        ))
    if len({lane.tid for lane in lanes}) != len(lanes):
        raise ValueError("duplicate lane TID")
    return tuple(lanes)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    flat_left = left.ravel().astype(np.float64)
    flat_right = right.ravel().astype(np.float64)
    denominator = np.linalg.norm(flat_left) * np.linalg.norm(flat_right)
    return float(np.dot(flat_left, flat_right) / denominator) if denominator else math.nan


def _pair_medians(rows: list[tuple[dict[str, Any], LaneGrammar]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[LaneGrammar]] = {}
    for entry, lane in rows:
        groups.setdefault((entry["family"], entry["loops"]), []).append(lane)
    vectors = ("pt_ordinal_bigram", "pebs_time_site_address")
    comparisons = {kind: {name: [] for name in vectors} for kind in
                   ("same_family_intensity", "cross_intensity", "cross_family_same_intensity")}
    for key, lanes in groups.items():
        if len(lanes) >= 2:
            for name in vectors:
                comparisons["same_family_intensity"][name].append(
                    _cosine(getattr(lanes[0], name), getattr(lanes[1], name)))
        alternatives = sorted(other for other in groups if other[1] == key[1]
                              and other[0] != key[0])
        if alternatives:
            for name in vectors:
                comparisons["cross_family_same_intensity"][name].append(
                    _cosine(getattr(lanes[0], name),
                            getattr(groups[alternatives[0]][0], name)))
    for family in sorted({key[0] for key in groups}):
        intensities = sorted(key for key in groups if key[0] == family)
        if len(intensities) >= 2:
            for name in vectors:
                comparisons["cross_intensity"][name].append(
                    _cosine(getattr(groups[intensities[0]][0], name),
                            getattr(groups[intensities[-1]][0], name)))
    return {
        kind: {"pairs": len(values[vectors[0]]),
               "cosine_median": {name: statistics.median([value for value in values[name]
                                                           if math.isfinite(value)])
                                  for name in vectors}}
        for kind, values in comparisons.items()
    }


def audit_raw_grammar(artifact: Path, *, verify_manifest: bool = True) -> dict[str, Any]:
    """Reduce one shard at a time; retain only compact diagnostic sketches."""
    manifest_bytes = (artifact / "capture-manifest.json").read_bytes() if verify_manifest else b""
    manifest = json.loads(manifest_bytes) if verify_manifest else None
    entries = ([entry for entry in manifest["entries"] if entry["raw_retained"]]
               if manifest is not None else None)
    paths = ([artifact / entry["raw_path"] for entry in entries] if entries is not None
             else sorted((artifact / "raw").glob("*.pt")))
    paths.sort()
    expected = ({entry["raw_path"]: entry for entry in entries} if entries else {})
    if not paths:
        raise ValueError("no retained raw shards")
    decode_cache: dict[str, tuple[dict[str, Any], tuple[int, int]]] = {}
    rows: list[tuple[dict[str, Any], LaneGrammar]] = []
    started = time.perf_counter()
    reduction_seconds = 0.0
    for path in paths:
        file_bytes = path.read_bytes()
        if expected:
            entry = expected[str(path.relative_to(artifact))]
            if hashlib.sha256(file_bytes).hexdigest() != entry["raw_sha256"]:
                raise ValueError(f"raw custody hash mismatch: {path.name}")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        reference = raw["decode_sideband"]["kernel_decode_state"]
        state_id = reference["kernel_state_sha256"]
        if state_id not in decode_cache:
            state_path = artifact / reference["path"]
            if hashlib.sha256(state_path.read_bytes()).hexdigest() != reference["sha256"]:
                raise ValueError("decode-state custody hash mismatch")
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            if state["kernel_state_sha256"] != state_id:
                raise ValueError("decode-state identity mismatch")
            bounds = kernel_text_bounds(state["kernel_symbols"].numpy().tobytes())
            decode_cache[state_id] = (state, bounds)
        state, bounds = decode_cache[state_id]
        options = pt_options(raw["decode_sideband"]["pt_attribute"].numpy().tobytes())
        if options.packet_timing_requested:
            raise ValueError("this offline grammar assumes untimed raw PT")
        reducing = time.perf_counter()
        lanes = grammar_for_raw(raw, bounds)
        reduction_seconds += time.perf_counter() - reducing
        if len(lanes) != 1:
            raise ValueError("the audit comparison requires one lane per shard")
        metadata = entry if expected else {**raw["execution"], "loops": raw["loops"]}
        # Do not retain raw PT, PEBS token objects, or decoded sideband per row.
        lane = lanes[0]
        rows.append((metadata, LaneGrammar(
            tid=lane.tid, pt_spans=(), pebs_stores=(),
            pt_ordinal_bigram=lane.pt_ordinal_bigram,
            pebs_time_site_address=lane.pebs_time_site_address,
            pt_bytes=lane.pt_bytes, pt_psb_count=lane.pt_psb_count,
            pt_span_count=lane.pt_span_count, pebs_sample_count=lane.pebs_sample_count,
            pt_prefix_bytes_before_first_psb=lane.pt_prefix_bytes_before_first_psb,
            pebs_core_site_count=lane.pebs_core_site_count,
            pebs_exact_ip_count=lane.pebs_exact_ip_count,
            pebs_nonzero_address_count=lane.pebs_nonzero_address_count,
            pebs_timestamp_inversions=lane.pebs_timestamp_inversions,
            pebs_timestamp_ties=lane.pebs_timestamp_ties,
            sampled_cpus=lane.sampled_cpus,
            envelope_duration_ns=lane.envelope_duration_ns,
        )))
        del raw, file_bytes, lanes, lane
    elapsed = time.perf_counter() - started
    source_rows = [lane for _, lane in rows]
    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest() if manifest is not None else None,
        "raw_hashes_verified": bool(expected),
        "rows": len(rows),
        "boot_decode_states": sorted(decode_cache),
        "pt_bytes": sum(lane.pt_bytes for lane in source_rows),
        "pt_psb_candidates": sum(lane.pt_psb_count for lane in source_rows),
        "pt_spans_median": statistics.median(lane.pt_span_count for lane in source_rows),
        "pt_rows_starting_at_psb": sum(lane.pt_prefix_bytes_before_first_psb == 0
                                       for lane in source_rows),
        "pebs_samples": sum(lane.pebs_sample_count for lane in source_rows),
        "pebs_exact_ip": sum(lane.pebs_exact_ip_count for lane in source_rows),
        "pebs_core_sites": sum(lane.pebs_core_site_count for lane in source_rows),
        "pebs_nonzero_addresses": sum(lane.pebs_nonzero_address_count for lane in source_rows),
        "pebs_timestamp_inversions": sum(lane.pebs_timestamp_inversions for lane in source_rows),
        "pebs_timestamp_ties": sum(lane.pebs_timestamp_ties for lane in source_rows),
        "sampled_cpus": sorted({cpu for lane in source_rows for cpu in lane.sampled_cpus}),
        "similarity": _pair_medians(rows),
        "elapsed_seconds": elapsed,
        "rows_per_second_hash_load_reduce": len(rows) / elapsed,
        "reduction_seconds": reduction_seconds,
        "rows_per_second_reduction": len(rows) / reduction_seconds,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--stream-benchmark", action="store_true",
                        help="skip large manifest parse; raw hashes must be verified separately")
    args = parser.parse_args()
    print(json.dumps(audit_raw_grammar(
        args.artifact, verify_manifest=not args.stream_benchmark),
        indent=2, sort_keys=True))
