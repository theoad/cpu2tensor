# SPDX-License-Identifier: AGPL-3.0-only
import pytest
import torch

from cpu2tensor.examples.hardware_latent_geometry_r4 import (
    effective_rank,
    nearest_centroid_accuracy,
)


def test_effective_rank_recovers_orthogonal_coordinates() -> None:
    assert effective_rank(torch.eye(4)) == pytest.approx(3.0, rel=1e-5)


def test_nearest_centroid_recovers_separated_classes() -> None:
    training = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = ["left", "left", "right", "right"]
    evaluation = torch.tensor([[-3.0, 0.0], [3.0, 0.0]])
    assert nearest_centroid_accuracy(
        training, labels, evaluation, ["left", "right"]
    ) == 1.0


def test_nearest_centroid_rejects_unseen_class() -> None:
    with pytest.raises(ValueError, match="unseen class"):
        nearest_centroid_accuracy(
            torch.tensor([[0.0], [1.0]]), ["a", "a"],
            torch.tensor([[2.0]]), ["b"],
        )
