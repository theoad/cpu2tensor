# SPDX-License-Identifier: AGPL-3.0-only
"""Raw hardware trace triage uses bytes directly and freezes after fitting."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from cpu2tensor.examples.hardware_triage import (
    FrozenRawTracePCA,
    RawTraceSketchConfig,
    fit_raw_trace_pca,
    load_frozen_raw_trace_pca,
    pack_raw_traces,
    raw_trace_sketch,
)
from cpu2tensor.examples.hardware_triage_benchmark import (
    HOLDOUT_FAMILIES,
    SKETCH_CONFIG,
    _scores_per_million,
    _split_rows,
    leave_one_family_out,
)


class HardwareTriageTests(unittest.TestCase):
    def test_raw_sketch_is_batched_order_sensitive_and_normalized(self) -> None:
        config = RawTraceSketchConfig(segments=2, pair_bins=8, include_length=True)
        raw, offsets = pack_raw_traces(
            (torch.tensor([1, 2, 1, 2], dtype=torch.uint8),
             torch.tensor([2, 1, 2, 1], dtype=torch.uint8))
        )
        features = raw_trace_sketch(raw, offsets, config)
        with mock.patch("cpu2tensor.examples.hardware_triage._native", None):
            reference = raw_trace_sketch(raw, offsets, config)

        self.assertEqual(features.shape, (2, config.feature_dimensions))
        self.assertTrue(torch.allclose(features, reference))
        self.assertTrue(torch.allclose(features[:, :512].sum(1), torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.allclose(features[:, 512:-1].sum(1), torch.ones(2)))
        self.assertFalse(torch.equal(features[0], features[1]))

    def test_raw_sketch_rejects_ambiguous_ownership_and_offsets(self) -> None:
        with self.assertRaises(ValueError):
            RawTraceSketchConfig(include_length=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            pack_raw_traces(())
        with self.assertRaises(ValueError):
            pack_raw_traces((torch.tensor([], dtype=torch.uint8),))
        raw = torch.tensor([1, 2], dtype=torch.uint8)
        for offsets in (
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([0, 0, 2], dtype=torch.int64),
            torch.tensor([0, 3], dtype=torch.int64),
        ):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                raw_trace_sketch(raw, offsets)

    def test_native_sketch_matches_short_trace_fallback(self) -> None:
        config = RawTraceSketchConfig(segments=8, pair_bins=16)
        for length in range(1, config.segments):
            with self.subTest(length=length):
                raw = torch.arange(length, dtype=torch.uint8)
                offsets = torch.tensor([0, length], dtype=torch.int64)
                features = raw_trace_sketch(raw, offsets, config)
                with mock.patch("cpu2tensor.examples.hardware_triage._native", None):
                    reference = raw_trace_sketch(raw, offsets, config)

                self.assertTrue(torch.isfinite(features).all())
                self.assertTrue(torch.equal(features, reference))

    def test_native_sketch_can_omit_pairs_and_length(self) -> None:
        config = RawTraceSketchConfig(segments=4, pair_bins=0, include_length=False)
        raw = torch.tensor([1, 2, 3], dtype=torch.uint8)
        offsets = torch.tensor([0, 3], dtype=torch.int64)
        features = raw_trace_sketch(raw, offsets, config)
        with mock.patch("cpu2tensor.examples.hardware_triage._native", None):
            reference = raw_trace_sketch(raw, offsets, config)

        self.assertEqual(features.shape, (1, 1024))
        self.assertTrue(torch.equal(features, reference))

    def test_model_fits_once_scores_and_stays_frozen(self) -> None:
        traces = []
        for index in range(24):
            row = torch.tensor(([2, 130, 2, 130] * 64), dtype=torch.uint8)
            row[index % row.numel()] ^= index % 3
            traces.append(row)
        raw, offsets = pack_raw_traces(traces)
        features = raw_trace_sketch(raw, offsets)
        model = fit_raw_trace_pca(
            features,
            latent_dimensions=8,
        )

        self.assertIsInstance(model, FrozenRawTracePCA)
        scores = model.anomaly_score(features, config=RawTraceSketchConfig())
        latent = model.latent(features, config=RawTraceSketchConfig())
        self.assertEqual(scores.shape, (24,))
        self.assertEqual(latent.shape, (24, 8))
        self.assertTrue(torch.isfinite(scores).all())

    def test_model_rejects_same_width_from_different_sketch_config(self) -> None:
        fitted_config = RawTraceSketchConfig(segments=4, pair_bins=0, include_length=False)
        incompatible_config = RawTraceSketchConfig(
            segments=3,
            pair_bins=256,
            include_length=False,
        )
        features = torch.rand(24, fitted_config.feature_dimensions, dtype=torch.float32)
        model = fit_raw_trace_pca(features, config=fitted_config, latent_dimensions=8)

        with self.assertRaises(ValueError):
            model.anomaly_score(features, config=incompatible_config)

    def test_frozen_checkpoint_round_trip_preserves_inference(self) -> None:
        config = RawTraceSketchConfig(segments=2, pair_bins=0, include_length=False)
        features = torch.rand(24, config.feature_dimensions, dtype=torch.float32)
        model = fit_raw_trace_pca(features, config=config, latent_dimensions=4)
        expected = model.anomaly_score(features, config=config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triage.pt"
            torch.save(
                {
                    "mean": model.mean,
                    "scale": model.scale,
                    "components": model.components,
                    "sketch": {
                        "segments": config.segments,
                        "pair_bins": config.pair_bins,
                        "include_length": config.include_length,
                    },
                    "threshold": 0.25,
                },
                path,
            )
            restored, threshold = load_frozen_raw_trace_pca(path)

        self.assertEqual(threshold, 0.25)
        self.assertTrue(
            torch.equal(expected, restored.anomaly_score(features, config=config))
        )

    def test_benchmark_split_hides_families_and_keeps_calibration_independent(self) -> None:
        families = ("getpid", "fstat", *HOLDOUT_FAMILIES)
        rows = {
            family: torch.full((7, 3), index, dtype=torch.float32)
            for index, family in enumerate(families)
        }
        train, calibration, validation = _split_rows(
            rows,
            training_rows=2,
            calibration_rows=3,
        )

        self.assertEqual(train.shape, (4, 3))
        self.assertEqual(calibration.shape, (6, 3))
        self.assertEqual(validation.shape, (4, 3))
        self.assertEqual(set(train[:, 0].tolist()), {0.0, 1.0})

    def test_reviews_per_million_uses_strict_threshold(self) -> None:
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(_scores_per_million(scores, 2.0), 500_000.0)

    def test_leave_one_family_out_scores_every_family(self) -> None:
        generator = torch.Generator().manual_seed(17)
        rows = {
            family: torch.rand(
                6,
                SKETCH_CONFIG.feature_dimensions,
                generator=generator,
            )
            for family in ("first", "second", "third")
        }
        results = leave_one_family_out(
            rows,
            training_rows=2,
            calibration_rows=2,
            latent_dimensions=1,
            reviews_per_million=100_000.0,
        )

        self.assertEqual(set(results), set(rows))
        self.assertTrue(
            all("holdout_reviews_per_million" in result for result in results.values())
        )


if __name__ == "__main__":
    unittest.main()
