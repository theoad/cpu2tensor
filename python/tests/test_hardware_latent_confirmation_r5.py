# SPDX-License-Identifier: AGPL-3.0-only
from cpu2tensor.examples.hardware_latent_confirmation_r5 import confirmation_decision


def _runs(vicreg=(0.96, 0.61, 0.97, 0.64)):
    runs = []
    for objective, metrics in (
        ("reconstruction", (0.91, 0.46, 0.99, 0.54)),
        ("vicreg", vicreg),
    ):
        for seed in (3901, 3902, 3903):
            runs.append({
                "objective": objective,
                "pretraining_seed": seed,
                "family_accuracy": metrics[0],
                "intensity_accuracy": metrics[1],
                "stratum_accuracy": metrics[2],
                "view_identity_top1": metrics[3],
            })
    return runs


def test_confirmation_accepts_reproduced_vicreg_geometry() -> None:
    assert confirmation_decision(_runs())["passes_preregistered_gate"]


def test_confirmation_rejects_weak_view_identity() -> None:
    assert not confirmation_decision(
        _runs((0.96, 0.61, 0.97, 0.57))
    )["passes_preregistered_gate"]


def test_confirmation_rejects_missing_matrix_row() -> None:
    try:
        confirmation_decision(_runs()[:-1])
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete matrix was accepted")
