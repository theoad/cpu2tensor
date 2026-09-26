# SPDX-License-Identifier: AGPL-3.0-only
"""PT timing options and independent lane grammar require no real perf capture."""

import struct

import pytest
import torch

from cpu2tensor.examples.hardware_nuisance_sideband_probe_r1 import (
    core_text_offset,
    inspect_raw_shard,
    kernel_slide,
    kernel_text_bounds,
    pt_options,
)


def _bytes(value: bytes) -> torch.Tensor:
    return torch.tensor(list(value), dtype=torch.uint8)


def _lane(tid: int, cpu: int, timestamp: int) -> dict:
    psb = bytes((2, 130)) * 8
    return {
        "source": tid,
        "tid": tid,
        "envelope": {"clock": "CLOCK_MONOTONIC_RAW", "arm_before_ns": 10,
                     "stop_after_ns": 100},
        "status": [
            {"signal": "intel_pt", "available": True, "lost": False},
            {"signal": "memory_stores", "available": True, "lost": False,
             "time_enabled_ns": 80, "time_running_ns": 80},
            {"signal": "counters", "available": True, "lost": False,
             "time_enabled_ns": 80, "time_running_ns": 80},
        ],
        # A naked 0x19 byte is not proof of a TSC packet.
        "pt": {"trace_bytes": _bytes(psb + b"\x19" + psb),
               "perf_records": _bytes(b"\x00" * 8)},
        "pebs": {
            "time": torch.tensor([timestamp, timestamp], dtype=torch.int64),
            "tid": torch.tensor([tid, tid], dtype=torch.int64),
            "cpu": torch.tensor([cpu, cpu], dtype=torch.int64),
            "ip": torch.tensor([0x1010, 0x1020], dtype=torch.int64),
            "address": torch.tensor([0x2010, 0x3020], dtype=torch.int64),
            "data_source": torch.tensor([7, 7], dtype=torch.int64),
            "weight": torch.tensor([0, 0], dtype=torch.int64),
            "exact_ip": torch.tensor([1, 1], dtype=torch.int64),
        },
        "counters": {},
    }


def _sideband() -> tuple[dict, dict]:
    raw = {
        "decode_sideband": {
            "clock": "CLOCK_MONOTONIC_RAW",
            "kernel_decode_state": {"kernel_state_sha256": "same-boot"},
            "pt_attribute": _bytes(struct.pack("<IIQ", 9, 16, 0x2001)),
            "process_maps": _bytes(b"maps"),
            "kernel_modules": _bytes(b"modules"),
        },
        # The order is intentionally not a cross-lane timestamp order.
        "batches": [_lane(22, 3, 20), _lane(11, 2, 90)],
    }
    state = {
        "schema": "cpu2tensor-kernel-decode-state-v1",
        "kernel_state_sha256": "same-boot",
        "kernel_symbols": _bytes(b"0000000000001000 T _text\n0000000000002000 T _etext\n"),
        "module_build_ids_json": _bytes(b"{}"),
    }
    return raw, state


def test_runtime_anchor_not_a_numeric_slide() -> None:
    raw, state = _sideband()
    bounds = kernel_text_bounds(bytes(state["kernel_symbols"].tolist()))
    assert bounds == (0x1000, 0x2000)
    assert kernel_slide(bounds[0], None) is None
    assert kernel_slide(bounds[0], 0x800) == 0x800
    assert core_text_offset(0x1010, bounds) == 0x10
    assert core_text_offset(0x2020, bounds) is None
    assert inspect_raw_shard(raw, state)["numeric_kernel_slide"] is None


def test_pt_sync_is_not_a_clock_or_cross_lane_order() -> None:
    raw, state = _sideband()
    result = inspect_raw_shard(raw, state)
    assert pt_options(bytes(raw["decode_sideband"]["pt_attribute"].tolist())).config == 0x2001
    assert result["pt_packet_timing_requested"] is False
    assert [lane["tid"] for lane in result["lanes"]] == [22, 11]
    assert [lane["pt_psb_offsets"] for lane in result["lanes"]] == [(0, 17), (0, 17)]
    assert [lane["pebs_first_ns"] for lane in result["lanes"]] == [20, 90]
    assert [lane["sampled_cpus"] for lane in result["lanes"]] == [(3,), (2,)]
    assert all(lane["pebs_timestamp_ties"] == 1 for lane in result["lanes"])
    assert result["core_text_pebs_samples"] == 4


def test_rejects_multiplexing_and_bad_lane_identity() -> None:
    raw, state = _sideband()
    raw["batches"][0]["status"][1]["time_running_ns"] = 40
    with pytest.raises(ValueError, match="multiplexed"):
        inspect_raw_shard(raw, state)
    raw, state = _sideband()
    raw["batches"][1]["pebs"]["tid"][0] = 22
    with pytest.raises(ValueError, match="escaped"):
        inspect_raw_shard(raw, state)
