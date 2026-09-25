# SPDX-License-Identifier: AGPL-3.0-only
"""Masked hardware model preserves explicit modality and CPU semantics."""

from pathlib import Path
import tempfile
import unittest

import torch

from cpu2tensor.examples.hardware_multimodal import (
    HardwareMultimodalBatch,
    MaskedHardwareModel,
    MultimodalConfig,
    calibrate_multimodal_threshold,
    freeze_multimodal_model,
    load_frozen_multimodal_model,
    make_training_masks,
    multimodal_anomaly_score,
    save_frozen_multimodal_model,
    train_masked_model,
)


def _fixture(rows: int = 16, cpus: int = 2) -> HardwareMultimodalBatch:
    generator = torch.Generator().manual_seed(41)
    signatures = torch.randn((rows, 1, 1, 1), generator=generator) * 0.8
    cpu_effect = torch.randn((rows, cpus, 1, 1), generator=generator) * 0.2
    phase = torch.linspace(-1.0, 1.0, 16).view(1, 1, 16, 1)
    latent = signatures + cpu_effect + phase
    pt_weights = torch.linspace(0.8, 1.2, 256).view(1, 1, 1, 256)
    pebs_weights = torch.tensor([0.7, 0.9, 1.1, 1.3]).view(1, 1, 1, 4)
    pt = latent * pt_weights + 0.02 * torch.randn(
        (rows, cpus, 16, 256), generator=generator
    )
    pebs = latent * pebs_weights + 0.02 * torch.randn(
        (rows, cpus, 16, 4), generator=generator
    )
    pmu_latent = latent.mean(2, keepdim=True)
    pmu = pmu_latent * torch.tensor([0.8, 1.0, 1.2]).view(1, 1, 1, 3)
    pt_available = torch.ones((rows, cpus, 16), dtype=torch.bool)
    pebs_available = torch.ones_like(pt_available)
    pmu_available = torch.ones((rows, cpus, 1), dtype=torch.bool)

    edges = torch.linspace(10.0, 11.0, 17)
    segment_bounds = torch.stack((edges[:-1], edges[1:]), dim=1)
    pmu_bounds = torch.tensor([[10.0, 11.0]])
    per_cpu_bounds = torch.cat((segment_bounds, segment_bounds, pmu_bounds), dim=0)
    time_bounds = per_cpu_bounds.view(1, 1, 33, 2).expand(rows, cpus, -1, -1).clone()
    timing_quality = torch.cat((
        torch.full((16,), 0.25), torch.ones(16), torch.full((1,), 0.5)
    )).view(1, 1, 33).expand(rows, cpus, -1).clone()
    return HardwareMultimodalBatch(
        pt,
        pebs,
        pmu,
        pt_available,
        pebs_available,
        pmu_available,
        time_bounds,
        timing_quality,
    )


def _model(batch: HardwareMultimodalBatch) -> MaskedHardwareModel:
    torch.manual_seed(7)
    return MaskedHardwareModel(MultimodalConfig(
        pebs_features=batch.pebs.shape[-1],
        pmu_features=batch.pmu.shape[-1],
        model_dimensions=32,
        attention_heads=4,
        feedforward_dimensions=64,
        local_layers=1,
        cross_cpu_layers=1,
    ))


def _replace(
    batch: HardwareMultimodalBatch,
    **changes: torch.Tensor,
) -> HardwareMultimodalBatch:
    values = {
        "pt": batch.pt,
        "pebs": batch.pebs,
        "pmu": batch.pmu,
        "pt_available": batch.pt_available,
        "pebs_available": batch.pebs_available,
        "pmu_available": batch.pmu_available,
        "time_bounds": batch.time_bounds,
        "timing_quality": batch.timing_quality,
    }
    values.update(changes)
    return HardwareMultimodalBatch(**values)


class HardwareMultimodalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_masks_include_spans_and_one_whole_modality(self) -> None:
        batch = _fixture(rows=4, cpus=1)
        masks = make_training_masks(
            batch,
            generator=torch.Generator().manual_seed(3),
            whole_modality_probability=1.0,
        )
        for row in range(batch.batch_size):
            whole = sum((
                torch.equal(masks.pt[row], batch.pt_available[row]),
                torch.equal(masks.pebs[row], batch.pebs_available[row]),
                torch.equal(masks.pmu[row], batch.pmu_available[row]),
            ))
            self.assertGreaterEqual(whole, 1)
            self.assertTrue(bool(masks.pt[row].any() or masks.pebs[row].any()
                                 or masks.pmu[row].any()))

    def test_missing_modality_is_explicit_and_scores_finitely(self) -> None:
        batch = _fixture(rows=5, cpus=2)
        model = _model(batch)
        model.fit_normalization(batch)
        freeze_multimodal_model(model)
        pebs = torch.full_like(batch.pebs, torch.nan)
        pebs_available = torch.zeros_like(batch.pebs_available)
        time_bounds = batch.time_bounds.clone()
        timing_quality = batch.timing_quality.clone()
        time_bounds[:, :, 16:32] = torch.nan
        timing_quality[:, :, 16:32] = torch.nan
        missing = _replace(
            batch,
            pebs=pebs,
            pebs_available=pebs_available,
            time_bounds=time_bounds,
            timing_quality=timing_quality,
        )

        scores = multimodal_anomaly_score(model, missing)

        self.assertEqual(scores.shape, (5,))
        self.assertTrue(bool(torch.isfinite(scores).all()))

    def test_cpu_permutation_and_global_time_shift_preserve_inference(self) -> None:
        batch = _fixture(rows=6, cpus=3)
        model = _model(batch)
        model.fit_normalization(batch)
        freeze_multimodal_model(model)
        expected = multimodal_anomaly_score(model, batch)
        permutation = torch.tensor([2, 0, 1])
        permuted = HardwareMultimodalBatch(*(
            tensor.index_select(1, permutation)
            for tensor in (
                batch.pt,
                batch.pebs,
                batch.pmu,
                batch.pt_available,
                batch.pebs_available,
                batch.pmu_available,
                batch.time_bounds,
                batch.timing_quality,
            )
        ))
        shifted = _replace(batch, time_bounds=batch.time_bounds + 37.0)

        torch.testing.assert_close(
            multimodal_anomaly_score(model, permuted), expected, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            multimodal_anomaly_score(model, shifted), expected, rtol=1e-5, atol=1e-6
        )

    def test_training_swap_sensitivity_freeze_and_checkpoint(self) -> None:
        training = _fixture(rows=24, cpus=1)
        model = _model(training)
        losses = train_masked_model(
            model,
            training,
            steps=120,
            batch_size=24,
            learning_rate=2e-3,
            weight_decay=0.0,
            seed=9,
        )
        freeze_multimodal_model(model)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        baseline = multimodal_anomaly_score(model, training)
        swapped = _replace(training, pebs=training.pebs.roll(1, 0))
        swapped_scores = multimodal_anomaly_score(model, swapped)
        threshold = calibrate_multimodal_threshold(
            model, training, reviews_per_million=100_000.0
        )

        self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in losses))
        self.assertGreater(float(swapped_scores.mean()), float(baseline.mean()) * 1.2)
        self.assertTrue(torch.isfinite(torch.tensor(threshold)))
        self.assertTrue(all(torch.equal(before[name], value)
                            for name, value in model.state_dict().items()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multimodal.pt"
            save_frozen_multimodal_model(path, model, threshold)
            restored, restored_threshold = load_frozen_multimodal_model(path)
            restored_scores = multimodal_anomaly_score(restored, training)

        self.assertEqual(restored_threshold, threshold)
        torch.testing.assert_close(restored_scores, baseline, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
