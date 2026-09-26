# SPDX-License-Identifier: AGPL-3.0-only
"""Offline descriptor checks do not assume PT has a timestamp."""

import pytest
import torch

from cpu2tensor.examples.hardware_invariant_capture_probe import probe_raw_capture


def _raw(trace: list[int], *, lost: bool = False) -> dict:
    return {
        "batches": [{
            "envelope": {
                "clock": "CLOCK_MONOTONIC_RAW",
                "arm_before_ns": 100,
                "arm_after_ns": 110,
                "stop_before_ns": 190,
                "stop_after_ns": 200,
            },
            "status": [
                {"signal": "intel_pt", "available": True, "lost": lost},
                {"signal": "memory_stores", "available": True, "lost": False,
                 "time_enabled_ns": 80, "time_running_ns": 80},
                {"signal": "counters", "available": True, "lost": False,
                 "time_enabled_ns": 80, "time_running_ns": 80},
            ],
            "pt": {"trace_bytes": torch.tensor(trace, dtype=torch.uint8)},
            "pebs": {
                "time": torch.tensor([120, 150, 180]),
                "ip": torch.tensor([0x1001, 0x1002, 0x1001]),
                "address": torch.tensor([0x2001, 0x2002, 0x2001]),
                "exact_ip": torch.tensor([1, 1, 1]),
                "cpu": torch.tensor([2, 2, 2]),
            },
            "counters": {},
        }],
    }


def test_pt_order_survives_histogram_collision() -> None:
    original = probe_raw_capture(_raw([1, 2, 3] * 16))
    changed = probe_raw_capture(_raw([3, 2, 1] * 16))
    assert torch.equal(original.pt_histogram, changed.pt_histogram)
    assert not torch.equal(original.pt_bigrams, changed.pt_bigrams)
    assert original.pebs_unique_pairs == 2
    assert original.pebs_time_span_fraction == pytest.approx(0.6)


def test_rejects_loss_and_multiplexing() -> None:
    with pytest.raises(ValueError, match="lost"):
        probe_raw_capture(_raw([1, 2], lost=True))
    raw = _raw([1, 2])
    raw["batches"][0]["status"][1]["time_running_ns"] = 40
    with pytest.raises(ValueError, match="multiplexed"):
        probe_raw_capture(raw)
