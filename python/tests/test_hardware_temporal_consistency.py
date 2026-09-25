# SPDX-License-Identifier: AGPL-3.0-only
"""Temporal consistency stays per-lane and uses coherent raw interventions."""

from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from cpu2tensor.examples.hardware_multimodal import HardwareMultimodalBatch as ModelBatch
from cpu2tensor.examples.hardware_multimodal import MultimodalConfig
from cpu2tensor.examples.hardware_multimodal_experiment import (
    PlannedExecution,
    raw_capture_payload,
)
from cpu2tensor.examples.hardware_multimodal_features import featurize_hardware_capture
from cpu2tensor.examples.hardware_temporal_consistency import (
    TemporalConsistencyModel,
    captures_from_raw_payload,
    cyclic_permutation,
    refeature_with_pebs_time_permutation,
    temporal_consistency_anomaly_score,
    train_temporal_consistency,
)
from cpu2tensor.hardware import (
    HardwareBatch,
    HardwareCaptureEnvelope,
    HardwareCounterBatch,
    HardwareMultimodalBatch,
    HardwareSourceStatus,
)


def _capture(tid: int = 77) -> HardwareMultimodalBatch:
    count = 16
    times = torch.arange(count, dtype=torch.int64) * 100 + 50
    empty_i64 = torch.empty(0, dtype=torch.int64)
    empty_i32 = torch.empty(0, dtype=torch.int64)
    empty_bool = torch.empty(0, dtype=torch.int64)
    pt = HardwareBatch(
        tid, "intel_pt", empty_i64, empty_i32, empty_i32, empty_i64,
        empty_i32, empty_i64, empty_i64, empty_i64, empty_i64, empty_bool,
        torch.arange(256, dtype=torch.uint8),
    )
    pebs = HardwareBatch(
        tid, "memory_loads", torch.arange(count, dtype=torch.int64) + 1,
        torch.full((count,), tid, dtype=torch.int64),
        torch.full((count,), tid, dtype=torch.int64), times,
        torch.full((count,), 2, dtype=torch.int64),
        torch.full((count,), 10_000, dtype=torch.int64),
        torch.arange(count, dtype=torch.int64) * 4096 + 1,
        torch.arange(count, dtype=torch.int64) + 1,
        torch.arange(count, dtype=torch.int64),
        torch.ones(count, dtype=torch.int64), torch.empty(0, dtype=torch.uint8),
    )
    counters = HardwareCounterBatch(
        tid, tid, -1, ("instructions", "cycles", "ref_cycles"),
        torch.tensor([100, 200, 150], dtype=torch.int64),
        1_000, 1_000, True, False,
    )
    return HardwareMultimodalBatch(
        tid, tid, -1,
        HardwareCaptureEnvelope("CLOCK_MONOTONIC_RAW", 0, 10, 1590, 1600),
        (
            HardwareSourceStatus("intel_pt", True, True, False),
            HardwareSourceStatus("memory_loads", True, True, False, 1_000, 1_000),
            HardwareSourceStatus("counters", True, True, False, 1_000, 1_000),
        ),
        pt, pebs, counters,
    )


def _synthetic(rows: int = 32, cpus: int = 1) -> ModelBatch:
    generator = torch.Generator().manual_seed(9)
    position = torch.arange(16, dtype=torch.float32).view(1, 1, 16, 1)
    phase = position * (2 * torch.pi / 16)
    pt = torch.randn((rows, cpus, 16, 256), generator=generator) * 0.02
    pt[..., 0:1] += phase.sin()
    pt[..., 1:2] += phase.cos()
    pebs = torch.randn((rows, cpus, 16, 24), generator=generator) * 0.02
    pebs[..., 0:1] += phase.sin()
    pebs[..., 1:2] += phase.cos()
    pebs[..., 2:3] += (2 * phase).sin()
    pmu = torch.randn((rows, cpus, 1, 4), generator=generator) * 0.02
    available = torch.ones((rows, cpus, 16), dtype=torch.bool)
    pmu_available = torch.ones((rows, cpus, 1), dtype=torch.bool)
    edges = torch.linspace(0.0, 1.0, 17)
    pebs_bounds = torch.stack((edges[:-1], edges[1:]), 1)
    pt_bounds = torch.tensor((0.0, 1.0)).expand(16, 2)
    bounds = torch.cat((pt_bounds, pebs_bounds, torch.tensor([[0.0, 1.0]])))
    bounds = bounds.view(1, 1, 33, 2).expand(rows, cpus, -1, -1).clone()
    quality = torch.cat((torch.zeros(16), torch.ones(16), torch.ones(1)))
    quality = quality.view(1, 1, 33).expand(rows, cpus, -1).clone()
    return ModelBatch(
        pt, pebs, pmu, available, available.clone(), pmu_available, bounds, quality
    )


def _roll_pebs(batch: ModelBatch, shift: int) -> ModelBatch:
    return replace(batch, pebs=batch.pebs.roll(shift, 2))


def _permute_lanes(batch: ModelBatch, order: torch.Tensor) -> ModelBatch:
    return ModelBatch(*(
        tensor.index_select(1, order) for tensor in (
            batch.pt, batch.pebs, batch.pmu, batch.pt_available,
            batch.pebs_available, batch.pmu_available, batch.time_bounds,
            batch.timing_quality,
        )
    ))


class HardwareTemporalConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_raw_timestamp_shift_refeaturizes_all_modalities_coherently(self) -> None:
        capture = _capture()
        payload = raw_capture_payload(
            (capture,), execution=PlannedExecution("x", "getpid", 0, "training"),
            loops=1, stdout=b"ok", elapsed_ns=1,
        )
        rebuilt = captures_from_raw_payload(payload)
        clean = featurize_hardware_capture(rebuilt).batch
        shifted = refeature_with_pebs_time_permutation(
            rebuilt, cyclic_permutation(1)
        )

        torch.testing.assert_close(shifted.pt, clean.pt)
        torch.testing.assert_close(shifted.pmu, clean.pmu)
        torch.testing.assert_close(shifted.pebs[:, :, 1], clean.pebs[:, :, 0])
        torch.testing.assert_close(shifted.pebs[:, :, 0], clean.pebs[:, :, 15])
        self.assertTrue(bool(shifted.pebs_available.all()))
        centers = shifted.time_bounds[0, 0, 16:32].mean(1)
        self.assertTrue(bool((centers[1:] > centers[:-1]).all()))
        # Custody data is immutable; the first raw timestamp did not move.
        self.assertEqual(int(rebuilt[0].pebs.time[0]), 50)

    def test_execution_energy_is_invariant_to_lane_permutation(self) -> None:
        batch = _synthetic(rows=4, cpus=2)
        model = TemporalConsistencyModel(MultimodalConfig(
            pebs_features=24, pmu_features=4, model_dimensions=16,
            attention_heads=4, feedforward_dimensions=32,
            local_layers=1, cross_cpu_layers=1,
        ))
        model.base.fit_normalization(batch)
        order = torch.tensor([1, 0])
        permuted = _permute_lanes(batch, order)

        torch.testing.assert_close(
            model.execution_energy(permuted), model.execution_energy(batch),
            rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(
            model.lane_energy(permuted), model.lane_energy(batch).index_select(1, order),
            rtol=1e-5, atol=1e-6,
        )

    def test_unseen_shift_receives_higher_frozen_score(self) -> None:
        batch = _synthetic()
        model = TemporalConsistencyModel(MultimodalConfig(
            pebs_features=24, pmu_features=4, model_dimensions=16,
            attention_heads=4, feedforward_dimensions=32,
            local_layers=1, cross_cpu_layers=1,
        ))
        history = train_temporal_consistency(
            model, batch, (_roll_pebs(batch, 1), _roll_pebs(batch, 4)),
            steps=100, batch_size=16, learning_rate=1e-3, weight_decay=0.0,
            seed=4,
        )
        clean = temporal_consistency_anomaly_score(model, batch)
        unseen = temporal_consistency_anomaly_score(model, _roll_pebs(batch, 2))

        self.assertLess(history[-1]["ranking"], history[0]["ranking"])
        self.assertGreater(float(unseen.mean()), 1.5 * float(clean.mean()))


if __name__ == "__main__":
    unittest.main()
