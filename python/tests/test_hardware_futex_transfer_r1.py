# SPDX-License-Identifier: AGPL-3.0-only
import torch

from cpu2tensor.examples.hardware_futex_transfer_r1 import auc, direction_probe
from cpu2tensor.examples.hardware_multimodal import HardwareMultimodalBatch


def batch(values: list[float]) -> HardwareMultimodalBatch:
    count = len(values)
    pt = torch.zeros(count, 1, 16, 256)
    pebs = torch.zeros(count, 1, 16, 140)
    pmu = torch.zeros(count, 1, 1, 4)
    for index, value in enumerate(values):
        pt[index, 0, 0, 0] = value
        pebs[index, 0, 0, 0] = value
        pmu[index, 0, 0, 0] = value
    available = torch.ones(count, 1, 16, dtype=torch.bool)
    pmu_available = torch.ones(count, 1, 1, dtype=torch.bool)
    bounds = torch.zeros(count, 1, 33, 2)
    for index, value in enumerate(values):
        bounds[index, 0, :, 0] = value
        bounds[index, 0, :, 1] = value + 1
    return HardwareMultimodalBatch(
        pt, pebs, pmu, available, available, pmu_available,
        bounds, torch.ones(count, 1, 33),
    )


def test_auc_accounts_for_ties() -> None:
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert auc(labels, torch.tensor([0.0, 1.0, 1.0, 2.0])) == 0.875


def test_direction_probe_fits_only_training_scale_and_direction() -> None:
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    report = direction_probe(
        batch([0.0, 0.2, 1.0, 1.2]), labels,
        batch([10.0, 10.2, 11.0, 11.2]), labels,
    )
    for result in report.values():
        assert result["training_auroc"] == 1.0
        assert result["evaluation_auroc"] == 1.0
