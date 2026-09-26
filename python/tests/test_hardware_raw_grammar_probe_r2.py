# SPDX-License-Identifier: AGPL-3.0-only
"""Raw grammar preserves lane order without manufacturing PT timestamps."""

import numpy as np
import pytest
import torch

from cpu2tensor.examples.hardware_raw_grammar_probe_r2 import grammar_for_raw


def _lane(tid: int, cpu: int, trace: bytes, timestamp: int) -> dict:
    return {
        "source": tid,
        "tid": tid,
        "envelope": {"clock": "CLOCK_MONOTONIC_RAW", "arm_before_ns": 100,
                     "stop_after_ns": 200},
        "status": [
            {"signal": "intel_pt", "available": True, "lost": False},
            {"signal": "memory_stores", "available": True, "lost": False,
             "time_enabled_ns": 50, "time_running_ns": 50},
            {"signal": "counters", "available": True, "lost": False,
             "time_enabled_ns": 50, "time_running_ns": 50},
        ],
        "pt": {"source": tid, "trace_bytes": torch.tensor(list(trace), dtype=torch.uint8)},
        "pebs": {
            "source": tid,
            "time": torch.tensor([timestamp, timestamp + 1]),
            "ip": torch.tensor([0x1010, 0x3010]),
            "address": torch.tensor([0x2015, 0x3F16]),
            "data_source": torch.tensor([7, 8]),
            "weight": torch.tensor([0, 0]),
            "period": torch.tensor([1000, 1000]),
            "exact_ip": torch.tensor([1, 1]),
            "cpu": torch.tensor([cpu, cpu]),
            "tid": torch.tensor([tid, tid]),
        },
    }


def test_psb_spans_and_pebs_sites_remain_separate() -> None:
    psb = bytes((2, 130)) * 8
    trace = b"x" + psb + b"\x01\x02\x03" + psb + b"\x04\x05"
    lane = grammar_for_raw({"batches": [_lane(11, 2, trace, 130)]}, (0x1000, 0x2000))[0]
    assert [(span.start, span.stop) for span in lane.pt_spans] == [
        (0, 1), (1, 20), (20, len(trace))]
    assert lane.pt_psb_count == 2
    assert lane.pt_prefix_bytes_before_first_psb == 1
    assert lane.pebs_stores[0].core_text_offset == 0x10
    assert lane.pebs_stores[1].core_text_offset is None
    assert lane.pebs_stores[0].raw_virtual_address == 0x2015
    assert lane.pebs_stores[0].address_page_offset == 0x15
    assert lane.pebs_stores[0].time_ns == 130
    assert lane.pt_ordinal_bigram.shape == (16, 128)
    assert lane.pebs_time_site_address.shape == (16, 256)


def test_raw_byte_order_and_lanes_are_not_merged() -> None:
    psb = bytes((2, 130)) * 8
    first = _lane(22, 3, psb + bytes([1, 2, 3]) * 8, 110)
    second = _lane(11, 2, psb + bytes([3, 2, 1]) * 8, 180)
    lanes = grammar_for_raw({"batches": [first, second]}, (0x1000, 0x2000))
    assert [lane.tid for lane in lanes] == [22, 11]
    assert [lane.sampled_cpus for lane in lanes] == [(3,), (2,)]
    assert [lane.pebs_stores[0].time_ns for lane in lanes] == [110, 180]
    assert not np.array_equal(lanes[0].pt_ordinal_bigram,
                              lanes[1].pt_ordinal_bigram)


def test_rejects_missing_or_multiplexed_signal() -> None:
    raw = {"batches": [_lane(11, 2, bytes((2, 130)) * 8, 150)]}
    raw["batches"][0]["status"][1]["time_running_ns"] = 40
    with pytest.raises(ValueError, match="multiplexing"):
        grammar_for_raw(raw, (0x1000, 0x2000))
    raw = {"batches": [_lane(11, 2, bytes((2, 130)) * 8, 150)]}
    raw["batches"][0]["pt"]["source"] = 99
    with pytest.raises(ValueError, match="escaped"):
        grammar_for_raw(raw, (0x1000, 0x2000))
