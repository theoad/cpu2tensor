# SPDX-License-Identifier: AGPL-3.0-only
import pytest
import torch

from cpu2tensor.examples.hardware_objective_ablation_r3 import (
    contrastive_loss,
    objective_decision,
    vicreg_loss,
)


def test_representation_losses_are_finite_and_differentiable() -> None:
    left = torch.randn(8, 16, requires_grad=True)
    right = torch.randn(8, 16, requires_grad=True)
    loss = contrastive_loss(left, right) + vicreg_loss(left, right)
    assert torch.isfinite(loss)
    loss.backward()
    assert left.grad is not None
    assert right.grad is not None


def test_contrastive_loss_prefers_matching_views() -> None:
    torch.manual_seed(7)
    left = torch.randn(8, 16)
    right = left + 0.01 * torch.randn(8, 16)
    assert contrastive_loss(left, right) < contrastive_loss(left, right.roll(1, 0))


def _runs(candidate_auc: float, candidate_loss: float):
    runs = []
    for objective in ("reconstruction", "contrastive", "vicreg"):
        for seed in (3901, 3902, 3903):
            auc = 0.1 if objective == "reconstruction" else candidate_auc
            loss = 0.08 if objective == "reconstruction" else candidate_loss
            runs.append({
                "objective": objective,
                "pretraining_seed": seed,
                "heldout_masked_loss": loss,
                "transfer": {"median_read_auroc": auc},
            })
    return runs


def test_objective_decision_promotes_a_transfer_gain_that_keeps_grammar() -> None:
    decision = objective_decision(_runs(0.8, 0.085))
    assert decision["passes_preregistered_gate"]
    assert decision["promoted_objectives"] == ["contrastive", "vicreg"]


@pytest.mark.parametrize("auc,loss", [(0.7, 0.08), (0.8, 0.09)])
def test_objective_decision_rejects_weak_transfer_or_lost_grammar(
    auc: float, loss: float
) -> None:
    assert not objective_decision(_runs(auc, loss))["passes_preregistered_gate"]
