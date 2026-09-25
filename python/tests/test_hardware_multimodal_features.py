# SPDX-License-Identifier: AGPL-3.0-only
"""Real capture fixtures map to deterministic, uncertainty-aware model tensors."""

from dataclasses import replace
import io
import math
import unittest

import torch

from cpu2tensor.examples.hardware_multimodal_features import (
    DATA_SOURCE_BINS,
    PEBS_FEATURES,
    PMU_FEATURES,
    HardwareFeatureError,
    featurize_hardware_capture,
    featurize_hardware_captures,
)
from cpu2tensor.hardware import (
    HardwareBatch,
    HardwareCaptureEnvelope,
    HardwareCounterBatch,
    HardwareMultimodalBatch,
    HardwareSourceStatus,
)


def _empty() -> torch.Tensor:
    return torch.empty(0, dtype=torch.int64)


def _pt(tid: int, trace: bytes) -> HardwareBatch:
    empty = _empty()
    return HardwareBatch(
        tid,
        "intel_pt",
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        torch.tensor(list(trace), dtype=torch.uint8),
    )


def _pebs(
    tid: int,
    *,
    times: tuple[int, ...],
    cpus: tuple[int, ...],
    address_shift: int = 0,
) -> HardwareBatch:
    count = len(times)
    return HardwareBatch(
        source=tid,
        signal="memory_loads",
        ip=torch.tensor(
            [0x401000 + address_shift + 0x10 * row for row in range(count)],
            dtype=torch.int64,
        ),
        pid=torch.full((count,), 900, dtype=torch.int64),
        tid=torch.full((count,), tid, dtype=torch.int64),
        time=torch.tensor(times, dtype=torch.int64),
        cpu=torch.tensor(cpus, dtype=torch.int64),
        period=torch.full((count,), 10_000, dtype=torch.int64),
        address=torch.tensor(
            [0x70001000 + address_shift + 0x40 * row for row in range(count)],
            dtype=torch.int64,
        ),
        weight=torch.tensor([7 + row for row in range(count)], dtype=torch.int64),
        data_source=torch.tensor(
            [0x681 + 17 * row for row in range(count)], dtype=torch.int64
        ),
        exact_ip=torch.tensor([row % 2 for row in range(count)], dtype=torch.int64),
        trace_bytes=torch.empty(0, dtype=torch.uint8),
    )


def _counters(tid: int) -> HardwareCounterBatch:
    return HardwareCounterBatch(
        source=tid,
        tid=tid,
        cpu=-1,
        names=("instructions", "cycles", "ref_cycles"),
        values=torch.tensor([200, 100, 50], dtype=torch.int64),
        time_enabled_ns=100,
        time_running_ns=100,
        available=True,
        lost=False,
    )


def _statuses(
    *,
    pt: bool = True,
    pebs: bool = True,
    pmu: bool = True,
) -> tuple[HardwareSourceStatus, ...]:
    return tuple(
        HardwareSourceStatus(
            signal,
            available,
            available,
            False,
            100 if available and signal in ("memory_loads", "counters") else None,
            100 if available and signal in ("memory_loads", "counters") else None,
        )
        for signal, available in zip(
            ("intel_pt", "memory_loads", "counters"), (pt, pebs, pmu)
        )
    )


_ENVELOPE = HardwareCaptureEnvelope(
    "CLOCK_MONOTONIC_RAW",
    arm_before_ns=1_000_000_000,
    arm_after_ns=1_000_001_000,
    stop_before_ns=1_000_017_000,
    stop_after_ns=1_000_020_000,
)


def _lane(
    tid: int,
    *,
    cpu: int,
    trace: bytes | None = bytes(range(32)),
    times: tuple[int, ...] = (1_000_002_000, 1_000_002_200, 1_000_015_000),
    address_shift: int = 0,
    counters: HardwareCounterBatch | None = None,
) -> HardwareMultimodalBatch:
    pebs = _pebs(
        tid,
        times=times,
        cpus=(cpu,) * len(times),
        address_shift=address_shift,
    )
    return HardwareMultimodalBatch(
        source=tid,
        tid=tid,
        cpu=-1,
        envelope=_ENVELOPE,
        status=_statuses(pt=trace is not None, pebs=True, pmu=True),
        pt=None if trace is None else _pt(tid, trace),
        pebs=pebs,
        counters=_counters(tid) if counters is None else counters,
    )


def _assert_batches_equal(
    case: unittest.TestCase,
    left,
    right,
) -> None:
    for name in (
        "pt",
        "pebs",
        "pmu",
        "pt_available",
        "pebs_available",
        "pmu_available",
        "time_bounds",
        "timing_quality",
    ):
        torch.testing.assert_close(
            getattr(left, name), getattr(right, name), rtol=0, atol=0, equal_nan=True
        )


class HardwareMultimodalFeatureTests(unittest.TestCase):
    def test_instruction_sketch_keeps_relative_sites_but_not_uniform_relocation(self) -> None:
        lane = _lane(10, cpu=3)
        assert lane.pebs is not None
        shifted = replace(lane, pebs=replace(lane.pebs, ip=lane.pebs.ip + 0x100000))
        changed_ips = lane.pebs.ip.clone()
        changed_ips[-1] += 0x100000
        changed = replace(lane, pebs=replace(lane.pebs, ip=changed_ips))
        original = featurize_hardware_capture((lane,)).batch.pebs
        relocated = featurize_hardware_capture((shifted,)).batch.pebs
        modified = featurize_hardware_capture((changed,)).batch.pebs
        torch.testing.assert_close(original, relocated)
        self.assertFalse(torch.equal(original[..., 24:88], modified[..., 24:88]))

    def test_precise_store_lane_preserves_signal_and_timing(self) -> None:
        lane = _lane(10, cpu=3)
        assert lane.pebs is not None
        status = list(lane.status)
        status[1] = replace(status[1], signal="memory_stores")
        stores = replace(
            lane,
            status=tuple(status),
            pebs=replace(lane.pebs, signal="memory_stores"),
        )
        result = featurize_hardware_capture((stores,))
        self.assertEqual(result.batch.pebs.shape, (1, 1, 16, len(PEBS_FEATURES)))
        self.assertEqual(result.lanes[0].pebs_samples, 3)
        self.assertEqual(int(result.batch.pebs_available.sum()), 2)

    def test_real_capture_fixture_round_trips_to_fixed_model_contract(self) -> None:
        raw = (_lane(20, cpu=7, trace=bytes([9]) * 32), _lane(10, cpu=3))
        file = io.BytesIO()
        torch.save(raw, file)
        file.seek(0)
        restored = torch.load(file, weights_only=False)

        result = featurize_hardware_capture(restored)
        repeated = featurize_hardware_capture(tuple(reversed(raw)))

        self.assertEqual(result.batch.pt.shape, (1, 2, 16, 256))
        self.assertEqual(result.batch.pebs.shape, (1, 2, 16, len(PEBS_FEATURES)))
        self.assertEqual(result.batch.pmu.shape, (1, 2, 1, len(PMU_FEATURES)))
        self.assertEqual(tuple(lane.tid for lane in result.lanes), (10, 20))
        self.assertEqual(tuple(lane.observed_cpu for lane in result.lanes), (3, 7))
        self.assertTrue(all(lane.migration_verified for lane in result.lanes))
        self.assertEqual(tuple(lane.pebs_samples for lane in result.lanes), (3, 3))
        _assert_batches_equal(self, result.batch, repeated.batch)
        self.assertEqual(result.lanes, repeated.lanes)

        # TID 10 is first even though input order and CPU order say otherwise.
        for segment in range(16):
            self.assertAlmostEqual(
                float(result.batch.pt[0, 0, segment, 2 * segment]), math.log1p(1)
            )
            self.assertAlmostEqual(
                float(result.batch.pt[0, 0, segment, 2 * segment + 1]), math.log1p(1)
            )
        torch.testing.assert_close(
            result.batch.pt[0, 1, :, 9], torch.full((16,), math.log1p(2))
        )
        self.assertTrue(bool(result.batch.pt_available.all()))
        self.assertEqual(int(result.batch.pebs_available[0, 0].sum()), 2)
        populated = torch.nonzero(result.batch.pebs_available[0, 0]).flatten().tolist()
        first = populated[0]
        self.assertAlmostEqual(
            float(result.batch.pebs[0, 0, first, 0]), math.log1p(2), places=6
        )
        self.assertAlmostEqual(
            float(result.batch.pebs[0, 0, first, 4 : 4 + DATA_SOURCE_BINS].sum()),
            1.0,
            places=6,
        )
        torch.testing.assert_close(
            result.batch.pmu[0, 0, 0],
            torch.tensor([2.0e9, 1.0e9, 5.0e8, 2.0]),
        )

        outer = torch.tensor([0.0, 20_000 / 1.0e9])
        torch.testing.assert_close(
            result.batch.time_bounds[0, 0, :16], outer.expand(16, 2)
        )
        self.assertTrue(bool((result.batch.timing_quality[0, 0, :16] == 0).all()))
        torch.testing.assert_close(
            result.batch.time_bounds[0, 0, 16 + first],
            torch.tensor([2_000 / 1.0e9, 2_200 / 1.0e9]),
        )
        self.assertEqual(float(result.batch.timing_quality[0, 0, 16 + first]), 1.0)

    def test_phase_timing_sink_does_not_change_features(self) -> None:
        raw = (_lane(10, cpu=3), _lane(20, cpu=7))
        expected = featurize_hardware_capture(raw)
        phases: dict[str, int] = {}

        observed = featurize_hardware_capture(raw, phase_costs_ns=phases)

        _assert_batches_equal(self, observed.batch, expected.batch)
        self.assertEqual(observed.lanes, expected.lanes)
        self.assertEqual(set(phases), {"pt_histogram", "pebs_pmu_features"})
        self.assertGreater(phases["pt_histogram"], 0)
        self.assertGreater(phases["pebs_pmu_features"], 0)

    def test_absolute_time_address_cpu_and_input_order_do_not_enter_model(self) -> None:
        lane = _lane(10, cpu=3)
        expected = featurize_hardware_capture((lane,))
        time_shift = 987_654_321
        pebs = lane.pebs
        assert pebs is not None
        shifted_pebs = replace(pebs, time=pebs.time + time_shift)
        shifted_envelope = HardwareCaptureEnvelope(
            _ENVELOPE.clock,
            _ENVELOPE.arm_before_ns + time_shift,
            _ENVELOPE.arm_after_ns + time_shift,
            _ENVELOPE.stop_before_ns + time_shift,
            _ENVELOPE.stop_after_ns + time_shift,
        )
        relocated = _pebs(
            10,
            times=tuple(int(value) for value in shifted_pebs.time.tolist()),
            cpus=(103, 103, 103),
            address_shift=0x40000000,
        )
        changed = replace(lane, envelope=shifted_envelope, pebs=relocated)
        observed = featurize_hardware_capture((changed,))

        _assert_batches_equal(self, expected.batch, observed.batch)
        self.assertEqual(expected.lanes[0].observed_cpu, 3)
        self.assertEqual(observed.lanes[0].observed_cpu, 103)

    def test_empty_pt_segments_and_sparse_pebs_are_explicitly_unavailable(self) -> None:
        empty_pebs = _pebs(10, times=(), cpus=())
        lane = replace(_lane(10, cpu=3, trace=b"x"), pebs=empty_pebs)
        result = featurize_hardware_capture((lane,))

        self.assertEqual(int(result.batch.pt_available.sum()), 1)
        self.assertEqual(int(result.batch.pebs_available.sum()), 0)
        self.assertTrue(bool(torch.isnan(result.batch.time_bounds[0, 0, 16:32]).all()))
        self.assertEqual(result.lanes[0].observed_cpu, None)
        self.assertFalse(result.lanes[0].migration_verified)
        self.assertEqual(result.lanes[0].pebs_samples, 0)

        no_pt = replace(lane, pt=_pt(10, b""))
        no_pt_result = featurize_hardware_capture((no_pt,))
        self.assertEqual(int(no_pt_result.batch.pt_available.sum()), 0)
        self.assertTrue(
            bool(torch.isnan(no_pt_result.batch.time_bounds[0, 0, :16]).all())
        )

    def test_capture_stack_preserves_rows_and_rejects_topology_change(self) -> None:
        batch, reports = featurize_hardware_captures(
            (
                (_lane(10, cpu=3),),
                (_lane(30, cpu=7, address_shift=0x100000),),
            )
        )
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(tuple(row[0].tid for row in reports), (10, 30))
        with self.assertRaisesRegex(HardwareFeatureError, "lane count"):
            featurize_hardware_captures(
                ((_lane(10, cpu=3),), (_lane(20, cpu=4), _lane(30, cpu=5)))
            )

    def test_loss_migration_multiplexing_and_schema_mismatch_fail_closed(self) -> None:
        lane = _lane(10, cpu=3)
        pebs = lane.pebs
        counters = lane.counters
        assert pebs is not None
        assert counters is not None
        migrating = replace(
            lane,
            pebs=replace(pebs, cpu=torch.tensor([3, 4, 3], dtype=torch.int64)),
        )
        statuses = list(lane.status)
        statuses[0] = replace(statuses[0], lost=True)
        lost = replace(lane, status=tuple(statuses))
        statuses = list(lane.status)
        statuses[1] = replace(statuses[1], time_running_ns=99)
        pebs_multiplexed = replace(lane, status=tuple(statuses))
        statuses = list(lane.status)
        statuses[2] = replace(statuses[2], time_running_ns=99)
        multiplexed = replace(
            lane,
            status=tuple(statuses),
            counters=replace(counters, time_running_ns=99),
        )
        wrong_counters = replace(
            lane,
            counters=replace(counters, names=("cycles", "instructions", "ref_cycles")),
        )
        wrong_clock = replace(lane, envelope=replace(_ENVELOPE, clock="MONOTONIC"))
        invalid = (
            (migrating, "migration"),
            (lost, "lost"),
            (pebs_multiplexed, "memory_loads source is empty or multiplexed"),
            (multiplexed, "counters source is empty or multiplexed"),
            (wrong_counters, "schema"),
            (wrong_clock, "CLOCK_MONOTONIC_RAW"),
        )
        for capture, message in invalid:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(HardwareFeatureError, message),
            ):
                featurize_hardware_capture((capture,))

    def test_cross_lane_modality_swaps_fail_source_attribution(self) -> None:
        first = _lane(10, cpu=3)
        second = _lane(20, cpu=7)
        for field in ("pt", "pebs", "counters"):
            swapped = replace(first, **{field: getattr(second, field)})
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(HardwareFeatureError, "source"),
            ):
                featurize_hardware_capture((swapped,))


if __name__ == "__main__":
    unittest.main()
