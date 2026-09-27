# SPDX-License-Identifier: AGPL-3.0-only
import torch

from cpu2tensor.examples.hardware_fewshot_transfer_r1 import deterministic_examples


def test_deterministic_examples_are_balanced_stable_and_seeded() -> None:
    identifiers = [f"row-{index}" for index in range(12)]
    labels = torch.tensor([0.0] * 6 + [1.0] * 6)
    first = deterministic_examples(identifiers, labels, 3, 7)
    second = deterministic_examples(identifiers, labels, 3, 7)
    other = deterministic_examples(identifiers, labels, 3, 8)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert labels[first].tolist().count(0.0) == 3
    assert labels[first].tolist().count(1.0) == 3
