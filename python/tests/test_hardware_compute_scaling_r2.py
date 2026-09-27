# SPDX-License-Identifier: AGPL-3.0-only
from cpu2tensor.examples.hardware_compute_scaling_r2 import scaling_decision


def _runs(compute_improves: bool = True, data_improves: bool = True):
    runs = []
    for width in (128, 256):
        for rows in (255, 1020, 2040):
            data_factor = (rows / 255) ** (-0.02 if data_improves else 0.02)
            for steps in (80, 160, 320):
                compute_factor = (
                    (steps / 80) ** (-0.05 if compute_improves else 0.05)
                )
                runs.append({
                    "model_dimensions": width,
                    "training_rows": rows,
                    "optimizer_steps": steps,
                    "heldout_masked_loss": 0.1 * data_factor * compute_factor,
                })
    return runs


def test_scaling_decision_accepts_coherent_compute_and_data_curves() -> None:
    decision = scaling_decision(_runs())
    assert decision["compute_improvements"] == 6
    assert decision["compute_scaling"]
    assert decision["data_scaling_emerges"]
    assert decision["passes_preregistered_gate"]


def test_scaling_decision_rejects_regressing_compute() -> None:
    decision = scaling_decision(_runs(compute_improves=False))
    assert decision["compute_improvements"] == 0
    assert not decision["compute_scaling"]
    assert not decision["passes_preregistered_gate"]


def test_scaling_decision_rejects_regressing_data() -> None:
    decision = scaling_decision(_runs(data_improves=False))
    assert not decision["data_scaling_emerges"]
    assert not decision["passes_preregistered_gate"]
