# SPDX-License-Identifier: AGPL-3.0-only
"""Kernel-only experiment keeps custody, split, and capture scope explicit."""

from __future__ import annotations

import argparse
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from cpu2tensor.examples import hardware_multimodal_experiment as experiment
from cpu2tensor.examples.hardware_multimodal import HardwareMultimodalBatch as ModelBatch
from cpu2tensor.hardware import (
    HardwareBatch,
    HardwareCaptureEnvelope,
    HardwareCounterBatch,
    HardwareMultimodalBatch,
    HardwareSourceStatus,
)


def _hardware(signal: str, *, trace: bytes = b"") -> HardwareBatch:
    rows = 1 if signal == "memory_loads" else 0
    return HardwareBatch(
        source=77,
        signal=signal,
        ip=torch.ones(rows, dtype=torch.int64),
        pid=torch.full((rows,), 77, dtype=torch.int32),
        tid=torch.full((rows,), 77, dtype=torch.int32),
        time=torch.full((rows,), 15, dtype=torch.int64),
        cpu=torch.full((rows,), 2, dtype=torch.int32),
        period=torch.full((rows,), 10_000, dtype=torch.int64),
        address=torch.ones(rows, dtype=torch.int64),
        weight=torch.ones(rows, dtype=torch.int64),
        data_source=torch.ones(rows, dtype=torch.int64),
        exact_ip=torch.ones(rows, dtype=torch.bool),
        trace_bytes=torch.tensor(list(trace), dtype=torch.uint8),
    )


def _capture_batch() -> HardwareMultimodalBatch:
    return HardwareMultimodalBatch(
        source=77,
        tid=77,
        cpu=-1,
        envelope=HardwareCaptureEnvelope("CLOCK_MONOTONIC_RAW", 10, 11, 19, 20),
        status=(
            HardwareSourceStatus("intel_pt", True, True, False),
            HardwareSourceStatus("memory_loads", True, True, False, 8, 8),
            HardwareSourceStatus("counters", True, True, False, 8, 8),
        ),
        pt=_hardware("intel_pt", trace=b"\x01\x02"),
        pebs=_hardware("memory_loads"),
        counters=HardwareCounterBatch(
            77, 77, -1, ("instructions", "cycles", "ref_cycles"),
            torch.tensor([3, 4, 5], dtype=torch.int64), 8, 8, True, False,
        ),
    )


def _model_row(value: float) -> ModelBatch:
    pt = torch.full((1, 1, 16, 256), value, dtype=torch.float32)
    pebs = torch.full((1, 1, 16, 24), value * 0.5, dtype=torch.float32)
    pmu = torch.full((1, 1, 1, 4), value * 0.25, dtype=torch.float32)
    pt_available = torch.ones((1, 1, 16), dtype=torch.bool)
    pebs_available = torch.ones_like(pt_available)
    pmu_available = torch.ones((1, 1, 1), dtype=torch.bool)
    edges = torch.linspace(10.0, 20.0, 17)
    segments = torch.stack((edges[:-1], edges[1:]), 1)
    time_bounds = torch.cat((segments, segments, torch.tensor([[10.0, 20.0]])))
    return ModelBatch(
        pt, pebs, pmu, pt_available, pebs_available, pmu_available,
        time_bounds.view(1, 1, 33, 2), torch.ones((1, 1, 33)),
    )


class _Process:
    pid = 77
    returncode = 0

    def __init__(self) -> None:
        self.stdout = BytesIO(b"READY\n")

    def communicate(self, value: bytes | None = None, timeout: float | None = None):
        del value, timeout
        return b"42\n", b""

    def kill(self) -> None:
        pass


class _Capture:
    config = None

    def __init__(self, config) -> None:
        type(self).config = config

    def __enter__(self):
        return self

    def __exit__(self, *unused) -> None:
        pass

    def stop(self):
        return (_capture_batch(),)


class KernelMultimodalExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_plan_holds_out_whole_families_and_whole_executions(self) -> None:
        families = ("getpid", "mmap", "openat", "yield")
        first, heldout = experiment.make_plan(
            families, repetitions=7, training_rows=3, calibration_rows=2,
            heldout_family_count=1, seed=19,
        )
        second, second_heldout = experiment.make_plan(
            families, repetitions=7, training_rows=3, calibration_rows=2,
            heldout_family_count=1, seed=19,
        )

        self.assertEqual(first, second)
        self.assertEqual(heldout, second_heldout)
        self.assertEqual(len({row.execution_id for row in first}), len(first))
        for family in heldout:
            self.assertEqual(
                {row.partition for row in first if row.family == family},
                {"heldout_family"},
            )
        for family in set(families) - set(heldout):
            partitions = [row.partition for row in first if row.family == family]
            self.assertEqual(partitions.count("training"), 3)
            self.assertEqual(partitions.count("calibration"), 2)
            self.assertEqual(partitions.count("familiar_validation"), 2)

    def test_capture_is_kernel_only_pinned_raw_pt_and_period_10000(self) -> None:
        execution = experiment.PlannedExecution("mmap-00000", "mmap", 0, "training")
        with mock.patch.object(experiment.subprocess, "Popen", return_value=_Process()) as popen, \
                mock.patch.object(experiment, "PerfMultimodalCapture", _Capture):
            batches, output, elapsed, counts = experiment.capture_execution(
                Path("/tmp/workload"), execution, loops=5_000, target_cpu=2,
                data_pages=64, aux_pages=2048, timeout=1.0,
            )

        self.assertEqual(popen.call_args.args[0][:3], ("taskset", "-c", "2"))
        self.assertEqual(_Capture.config.scope, "process_kernel")
        self.assertEqual(_Capture.config.pebs_period, 10_000)
        self.assertEqual(_Capture.config.modalities, experiment.MODALITIES)
        self.assertEqual(output, b"42\n")
        self.assertGreater(elapsed, 0)
        self.assertEqual(counts["pt_bytes"], 2)
        self.assertEqual(counts["pebs_exact_ip"], 1)
        self.assertEqual(counts["pebs_nonzero_address"], 1)
        self.assertEqual(len(batches), 1)

    def test_capture_rejects_unqualified_pebs_rows_and_lanes(self) -> None:
        batch = _capture_batch()
        assert batch.pebs is not None
        invalid = (
            replace(batch.pebs, exact_ip=torch.tensor([False])),
            replace(batch.pebs, address=torch.tensor([0], dtype=torch.int64)),
            replace(batch.pebs, cpu=torch.tensor([3], dtype=torch.int32)),
            replace(
                batch.pebs,
                ip=torch.empty(0, dtype=torch.int64),
                exact_ip=torch.empty(0, dtype=torch.bool),
                address=torch.empty(0, dtype=torch.int64),
                cpu=torch.empty(0, dtype=torch.int32),
            ),
        )
        for pebs in invalid:
            with self.subTest(pebs=pebs):
                with self.assertRaisesRegex(
                    experiment.HardwareCaptureError, "PEBS|pilot"
                ):
                    experiment._validate_capture((replace(batch, pebs=pebs),), 77, 2)

        lane = argparse.Namespace(
            tid=77, observed_cpu=3, migration_verified=True, pebs_samples=1
        )
        with self.assertRaisesRegex(
            experiment.HardwareCaptureError, "migration evidence"
        ):
            experiment._validate_lanes((lane,), 2)

    def test_swaps_and_misalignment_preserve_sparse_availability(self) -> None:
        batch = experiment._concatenate((_model_row(1.0), _model_row(2.0)))
        pebs_available = torch.zeros_like(batch.pebs_available)
        pebs_available[0, 0, (0, 2)] = True
        pebs_available[1, 0, (1, 3)] = True
        pebs = batch.pebs.masked_fill(~pebs_available[..., None], torch.nan)
        time_bounds = batch.time_bounds.clone()
        timing_quality = batch.timing_quality.clone()
        time_bounds[:, :, 16:32] = time_bounds[:, :, 16:32].masked_fill(
            ~pebs_available[..., None], torch.nan
        )
        timing_quality[:, :, 16:32] = timing_quality[:, :, 16:32].masked_fill(
            ~pebs_available, torch.nan
        )
        batch = experiment._replace(
            batch, pebs=pebs, pebs_available=pebs_available,
            time_bounds=time_bounds, timing_quality=timing_quality,
        )
        swapped = experiment.modality_swap(batch, "pebs")
        misaligned = experiment.timestamp_misalignment(batch)

        torch.testing.assert_close(swapped.pt, batch.pt)
        torch.testing.assert_close(swapped.pebs, batch.pebs.roll(1, 0), equal_nan=True)
        torch.testing.assert_close(
            swapped.time_bounds[:, :, 16:32],
            batch.time_bounds[:, :, 16:32].roll(1, 0),
            equal_nan=True,
        )
        torch.testing.assert_close(misaligned.pebs, batch.pebs, equal_nan=True)
        expected = batch.time_bounds.clone()
        expected[0, 0, (16, 18)] = batch.time_bounds[0, 0, (18, 16)]
        expected[1, 0, (17, 19)] = batch.time_bounds[1, 0, (19, 17)]
        torch.testing.assert_close(
            misaligned.time_bounds,
            expected,
            equal_nan=True,
        )
        torch.testing.assert_close(misaligned.time_bounds[:, :, :16], batch.time_bounds[:, :, :16])

    def test_train_only_verifies_custody_and_emits_both_frozen_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            entries = []
            layout = (
                ("getpid", "training"), ("getpid", "calibration"),
                ("getpid", "familiar_validation"),
                ("mmap", "training"), ("mmap", "calibration"),
                ("mmap", "familiar_validation"),
                ("yield", "heldout_family"), ("yield", "heldout_family"),
            )
            for index, (family, partition) in enumerate(layout):
                execution_id = f"{family}-{index:05d}"
                raw_path = artifact / "raw" / f"{execution_id}.pt"
                raw_hash = experiment._atomic_torch_save(raw_path, {"raw": index})
                derived_path = artifact / "derived" / f"{execution_id}.pt"
                execution = {
                    "execution_id": execution_id,
                    "family": family,
                    "repetition": index,
                    "partition": partition,
                }
                derived_hash = experiment._atomic_torch_save(derived_path, {
                    "schema": experiment.DERIVED_SCHEMA,
                    "execution": execution,
                    "raw_sha256": raw_hash,
                    "batch": experiment._model_payload(_model_row(float(index + 1))),
                    "lanes": [],
                })
                entries.append({
                    **execution,
                    "raw_path": str(raw_path.relative_to(artifact)),
                    "raw_sha256": raw_hash,
                    "derived_path": str(derived_path.relative_to(artifact)),
                    "derived_sha256": derived_hash,
                    "elapsed_ns": 1_000_000,
                })
            manifest = {
                "schema": experiment.SCHEMA,
                "identity_sha256": "identity",
                "event": {
                    "scope": experiment.SCOPE,
                    "modalities": list(experiment.MODALITIES),
                    "intel_pt_representation": "raw_aux_bytes_no_decode",
                    "pebs_period": experiment.PEBS_PERIOD,
                },
                "split": {
                    "families": ["getpid", "mmap", "yield"],
                    "heldout_families": ["yield"],
                },
                "collection": {"loss_count": 0},
                "entries": entries,
            }
            manifest["manifest_content_sha256"] = experiment._json_hash(manifest)
            experiment._atomic_json(artifact / "capture-manifest.json", manifest)
            args = argparse.Namespace(
                artifact=artifact,
                device="cpu",
                cpu_threads=1,
                model_seed=5,
                model_dimensions=16,
                attention_heads=4,
                feedforward_dimensions=32,
                local_layers=1,
                cross_cpu_layers=1,
                pt_pca_dimensions=2,
                steps=1,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=0.0,
                reviews_per_million=100_000.0,
                whole_modality_probability=0.5,
            )

            report = experiment.train_and_evaluate(args)

            self.assertEqual(set(report["models"]), {"fused", "span-only"})
            self.assertEqual(set(report["baselines"]), {"marginal", "pt_only_pca"})
            self.assertTrue(report["baselines"]["pt_only_pca"][
                "checkpoint_reload_bit_exact"
            ])
            self.assertTrue(report["models"]["fused"]["checkpoint_reload_bit_exact"])
            self.assertTrue(report["models"]["span-only"]["checkpoint_reload_bit_exact"])
            self.assertEqual(report["models"]["span-only"]["whole_modality_probability"], 0.0)
            self.assertIn("worst_family", report["models"]["fused"])
            self.assertIn("timestamp_misalignment", report["models"]["fused"]["sensitivity"])

            (artifact / entries[0]["raw_path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "raw custody hash mismatch"):
                experiment.load_dataset(artifact)


if __name__ == "__main__":
    unittest.main()
