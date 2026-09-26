# SPDX-License-Identifier: AGPL-3.0-only
"""Read-only exact-session metadata and PT synchronization inventory.

This probe does not decode control flow, derive PT timestamps from byte offsets,
or merge lanes into a total order. The 16-byte PSB signature is a candidate
packet synchronization boundary, not a clock tick or executed instruction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import statistics
import struct
from typing import Any

import torch


_PSB = bytes((0x02, 0x82)) * 8
_UINT64_MASK = (1 << 64) - 1
_PT_CONTROL = 1 << 0
_PT_CYC = 1 << 1
_PT_MTC = 1 << 9
_PT_TSC = 1 << 10
_PT_BRANCH = 1 << 13


@dataclass(frozen=True)
class PtOptions:
    config: int
    branch_requested: bool
    tsc_requested: bool
    mtc_requested: bool
    cyc_requested: bool

    @property
    def packet_timing_requested(self) -> bool:
        return self.tsc_requested or self.mtc_requested or self.cyc_requested


@dataclass(frozen=True)
class LaneGrammar:
    """Separate PT ordinal spans, PEBS sample clock, and PMU interval."""

    tid: int
    pt_bytes: int
    pt_psb_offsets: tuple[int, ...]
    pt_perf_record_bytes: int
    pebs_samples: int
    pebs_first_ns: int | None
    pebs_last_ns: int | None
    pebs_timestamp_inversions: int
    pebs_timestamp_ties: int
    pebs_exact_ip: int
    pebs_nonzero_address: int
    pebs_nonzero_weight: int
    pebs_unique_virtual_pages: int
    pebs_unique_data_sources: int
    pebs_data_source_values: tuple[int, ...]
    pebs_data_source_counts: tuple[tuple[int, int], ...]
    pebs_weight_median: float | None
    sampled_cpus: tuple[int, ...]
    pmu_time_enabled_ns: int
    pmu_time_running_ns: int


def _owned_bytes(value: torch.Tensor) -> bytes:
    if value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError("sideband and PT bytes must be one-dimensional uint8")
    return bytes(value.tolist())


def pt_options(attribute: bytes) -> PtOptions:
    """Read perf_event_attr header/config, not Linux perf's default CLI terms."""
    if len(attribute) < 16:
        raise ValueError("saved perf_event_attr is truncated")
    _event_type, size, config = struct.unpack_from("<IIQ", attribute)
    if size < 16 or size > len(attribute):
        raise ValueError("saved perf_event_attr size is inconsistent")
    if not config & _PT_CONTROL or not config & _PT_BRANCH:
        raise ValueError("saved event was not branch-enabled Intel PT")
    return PtOptions(
        config=config,
        branch_requested=bool(config & _PT_BRANCH),
        tsc_requested=bool(config & _PT_TSC),
        mtc_requested=bool(config & _PT_MTC),
        cyc_requested=bool(config & _PT_CYC),
    )


def kernel_text_bounds(symbols: bytes) -> tuple[int, int]:
    """Return runtime core-text bounds from the exact-boot kallsyms snapshot."""
    found: dict[str, int] = {}
    for line in symbols.decode("ascii", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] in ("_text", "_etext"):
            if parts[2] in found:
                raise ValueError(f"duplicate {parts[2]} symbol")
            found[parts[2]] = int(parts[0], 16)
    if set(found) != {"_text", "_etext"} or not 0 < found["_text"] < found["_etext"]:
        raise ValueError("exact-boot core text bounds are missing or invalid")
    return found["_text"], found["_etext"]


def kernel_slide(runtime_text: int, linktime_text: int | None) -> int | None:
    """A numeric slide requires a verified link-time anchor not in these shards."""
    return None if linktime_text is None else runtime_text - linktime_text


def core_text_offset(ip: int, bounds: tuple[int, int]) -> int | None:
    unsigned_ip = ip & _UINT64_MASK
    return unsigned_ip - bounds[0] if bounds[0] <= unsigned_ip < bounds[1] else None


def psb_offsets(trace: bytes) -> tuple[int, ...]:
    offsets: list[int] = []
    search_from = 0
    while (offset := trace.find(_PSB, search_from)) >= 0:
        offsets.append(offset)
        search_from = offset + len(_PSB)
    return tuple(offsets)


def lane_grammar(batch: dict[str, Any]) -> LaneGrammar:
    """Report each source independently; equal PEBS times are not ordered."""
    tid = batch["tid"]
    if batch["source"] != tid:
        raise ValueError("lane source and TID differ")
    statuses = batch["status"]
    if [item["signal"] for item in statuses] != ["intel_pt", "memory_stores", "counters"]:
        raise ValueError("expected PT, store PEBS, and PMU status")
    if any(item["lost"] or not item["available"] for item in statuses):
        raise ValueError("lane has lost or unavailable sensor data")
    for item in statuses[1:]:
        if item["time_enabled_ns"] != item["time_running_ns"]:
            raise ValueError("PEBS or PMU lane is multiplexed")
    envelope = batch["envelope"]
    if envelope["clock"] != "CLOCK_MONOTONIC_RAW":
        raise ValueError("unknown sample clock")
    start, stop = envelope["arm_before_ns"], envelope["stop_after_ns"]
    if start >= stop:
        raise ValueError("empty capture envelope")
    pt, pebs, counters = batch["pt"], batch["pebs"], batch["counters"]
    if pt is None or pebs is None or counters is None:
        raise ValueError("lane payload is incomplete")
    trace = _owned_bytes(pt["trace_bytes"])
    perf_records = pt.get("perf_records")
    times = pebs["time"].tolist()
    ips = pebs["ip"].tolist()
    addresses = pebs["address"].tolist()
    data_sources = pebs["data_source"].tolist()
    weights = pebs["weight"].tolist()
    exact = pebs["exact_ip"].tolist()
    cpus = pebs["cpu"].tolist()
    if len({len(times), len(ips), len(addresses), len(data_sources),
            len(weights), len(exact), len(cpus)}) != 1:
        raise ValueError("PEBS columns have different lengths")
    if any(time < start or time > stop for time in times):
        raise ValueError("PEBS timestamp escaped its lane envelope")
    if any(sample_tid != tid for sample_tid in pebs["tid"].tolist()):
        raise ValueError("PEBS sample escaped its lane")
    if any(value not in (0, 1) for value in exact):
        raise ValueError("invalid exact-IP flag")
    return LaneGrammar(
        tid=tid,
        pt_bytes=len(trace),
        pt_psb_offsets=psb_offsets(trace),
        pt_perf_record_bytes=0 if perf_records is None else perf_records.numel(),
        pebs_samples=len(times),
        pebs_first_ns=min(times) if times else None,
        pebs_last_ns=max(times) if times else None,
        pebs_timestamp_inversions=sum(right < left for left, right in zip(times, times[1:])),
        pebs_timestamp_ties=sum(right == left for left, right in zip(times, times[1:])),
        pebs_exact_ip=sum(exact),
        pebs_nonzero_address=sum((address & _UINT64_MASK) != 0 for address in addresses),
        pebs_nonzero_weight=sum((weight & _UINT64_MASK) != 0 for weight in weights),
        pebs_unique_virtual_pages=len({(address & _UINT64_MASK) >> 12 for address in addresses}),
        pebs_unique_data_sources=len(set(data_sources)),
        pebs_data_source_values=tuple(sorted(set(data_sources))),
        pebs_data_source_counts=tuple(sorted(Counter(data_sources).items())),
        pebs_weight_median=statistics.median(weights) if weights else None,
        sampled_cpus=tuple(sorted(set(cpus))),
        pmu_time_enabled_ns=statuses[2]["time_enabled_ns"],
        pmu_time_running_ns=statuses[2]["time_running_ns"],
    )


def inspect_raw_shard(raw: dict[str, Any], decode_state: dict[str, Any],
                      *, runtime_bounds: tuple[int, int] | None = None) -> dict[str, Any]:
    sideband = raw["decode_sideband"]
    if sideband["clock"] != "CLOCK_MONOTONIC_RAW":
        raise ValueError("sideband clock differs from sample clock")
    if decode_state["schema"] != "cpu2tensor-kernel-decode-state-v1":
        raise ValueError("unknown kernel decode state")
    if (sideband["kernel_decode_state"]["kernel_state_sha256"]
            != decode_state["kernel_state_sha256"]):
        raise ValueError("raw shard uses another exact-boot decode state")
    bounds = (runtime_bounds if runtime_bounds is not None else
              kernel_text_bounds(_owned_bytes(decode_state["kernel_symbols"])))
    options = pt_options(_owned_bytes(sideband["pt_attribute"]))
    lanes = tuple(lane_grammar(batch) for batch in raw["batches"])
    if len({lane.tid for lane in lanes}) != len(lanes):
        raise ValueError("duplicate source lane")
    core_text_samples = sum(
        core_text_offset(ip, bounds) is not None
        for batch in raw["batches"] for ip in batch["pebs"]["ip"].tolist()
    )
    return {
        "runtime_text_start": bounds[0],
        "runtime_text_end": bounds[1],
        "numeric_kernel_slide": kernel_slide(bounds[0], None),
        "pt_options": asdict(options),
        "pt_packet_timing_requested": options.packet_timing_requested,
        "process_maps_bytes": sideband["process_maps"].numel(),
        "kernel_modules_bytes": sideband["kernel_modules"].numel(),
        "module_build_ids_bytes": decode_state["module_build_ids_json"].numel(),
        "kernel_symbols_bytes": decode_state["kernel_symbols"].numel(),
        "core_text_pebs_samples": core_text_samples,
        "lanes": [asdict(lane) for lane in lanes],
    }


def inspect_retained_audit(artifact: Path) -> dict[str, Any]:
    """Verify raw custody and inventory every preregistered raw audit shard."""
    manifest_bytes = (artifact / "capture-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = [entry for entry in manifest["entries"] if entry["raw_retained"]]
    if not entries:
        raise ValueError("no raw audit shards were retained")
    state_infos = manifest["kernel_decode_states"]
    if not state_infos:
        raise ValueError("no exact-session kernel decode state was retained")
    decode_states = {}
    bounds_by_state = {}
    state_infos_by_id = {}
    for state_info in state_infos:
        state_id = state_info["kernel_state_sha256"]
        if state_id in decode_states:
            raise ValueError("duplicate exact-session decode-state identity")
        state_path = artifact / state_info["path"]
        if hashlib.sha256(state_path.read_bytes()).hexdigest() != state_info["sha256"]:
            raise ValueError("kernel decode state custody hash mismatch")
        decode_states[state_id] = torch.load(
            state_path, map_location="cpu", weights_only=True)
        state_infos_by_id[state_id] = state_info
        bounds_by_state[state_id] = kernel_text_bounds(
            _owned_bytes(decode_states[state_id]["kernel_symbols"]))
    observations = []
    for entry in entries:
        raw_path = artifact / entry["raw_path"]
        raw_bytes = raw_path.read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != entry["raw_sha256"]:
            raise ValueError(f"raw shard custody hash mismatch: {entry['execution_id']}")
        raw = torch.load(raw_path, map_location="cpu", weights_only=True)
        state_reference = raw["decode_sideband"]["kernel_decode_state"]
        state_id = state_reference["kernel_state_sha256"]
        try:
            decode_state = decode_states[state_id]
        except KeyError as error:
            raise ValueError("raw shard references an unlisted decode state") from error
        if state_reference != state_infos_by_id[state_id]:
            raise ValueError("raw shard decode-state reference differs from manifest")
        observations.append(inspect_raw_shard(
            raw, decode_state, runtime_bounds=bounds_by_state[state_id]))
    options = observations[0]["pt_options"]
    bounds = (observations[0]["runtime_text_start"], observations[0]["runtime_text_end"])
    if any(row["pt_options"] != options or
           (row["runtime_text_start"], row["runtime_text_end"]) != bounds
           for row in observations):
        raise ValueError("retained rows disagree on PT config or kernel text")
    lanes = [lane for row in observations for lane in row["lanes"]]
    data_source_counts = Counter()
    for lane in lanes:
        data_source_counts.update(dict(lane["pebs_data_source_counts"]))
    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "raw_shards_verified": len(entries),
        "kernel_state_sha256s": sorted(decode_states),
        "runtime_text_start_hex": hex(bounds[0]),
        "runtime_text_end_hex": hex(bounds[1]),
        "numeric_kernel_slide": None,
        "pt_options": options,
        "pt_packet_timing_requested": observations[0]["pt_packet_timing_requested"],
        "total_source_lanes": len(lanes),
        "lane_count_per_shard": sorted({len(row["lanes"]) for row in observations}),
        "sampled_cpus": sorted({cpu for lane in lanes for cpu in lane["sampled_cpus"]}),
        "pt_bytes": sum(lane["pt_bytes"] for lane in lanes),
        "psb_candidate_markers": sum(len(lane["pt_psb_offsets"]) for lane in lanes),
        "pt_perf_record_bytes": sum(lane["pt_perf_record_bytes"] for lane in lanes),
        "pebs_samples": sum(lane["pebs_samples"] for lane in lanes),
        "pebs_exact_ip": sum(lane["pebs_exact_ip"] for lane in lanes),
        "pebs_nonzero_address": sum(lane["pebs_nonzero_address"] for lane in lanes),
        "pebs_nonzero_weight": sum(lane["pebs_nonzero_weight"] for lane in lanes),
        "pebs_data_source_values": sorted({value for lane in lanes
                                            for value in lane["pebs_data_source_values"]}),
        "pebs_data_source_counts": {
            hex(value): count for value, count in sorted(data_source_counts.items())},
        "pebs_timestamp_inversions": sum(lane["pebs_timestamp_inversions"] for lane in lanes),
        "pebs_timestamp_ties": sum(lane["pebs_timestamp_ties"] for lane in lanes),
        "core_text_pebs_samples": sum(row["core_text_pebs_samples"] for row in observations),
        "pebs_unique_virtual_pages_median": statistics.median(
            lane["pebs_unique_virtual_pages"] for lane in lanes),
        "pebs_unique_data_sources_median": statistics.median(
            lane["pebs_unique_data_sources"] for lane in lanes),
        "pebs_weight_median_of_lane_medians": statistics.median(
            lane["pebs_weight_median"] for lane in lanes if lane["pebs_weight_median"] is not None),
        "process_maps_bytes_median": statistics.median(
            row["process_maps_bytes"] for row in observations),
        "kernel_modules_bytes_median": statistics.median(
            row["kernel_modules_bytes"] for row in observations),
        "module_build_ids_bytes": observations[0]["module_build_ids_bytes"],
        "kernel_symbols_bytes": observations[0]["kernel_symbols_bytes"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_retained_audit(args.artifact), indent=2, sort_keys=True))
