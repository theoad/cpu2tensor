# SPDX-License-Identifier: AGPL-3.0-only
"""Focused checks for the offline conditional PEBS research score."""

import torch

from cpu2tensor.examples.hardware_anomaly_pretraining_r1 import (
    FEATURE_COUNT, PEBS_END, PT_END, fit_conditional, score, standardize,
    threshold_report,
)


def test_conditional_score_uses_pt_to_predict_pebs_without_target_leakage():
    generator = torch.Generator().manual_seed(7)
    training = torch.randn((400, FEATURE_COUNT), generator=generator)
    training[:, PT_END:PEBS_END] = (
        0.7 * training[:, :PEBS_END - PT_END] +
        0.02 * torch.randn((400, PEBS_END - PT_END), generator=generator)
    )
    weights, residual_scale, _ = fit_conditional(training)
    ordinary = training[:32].clone()
    changed = ordinary.clone()
    changed[:, PT_END:PEBS_END] += 2.0
    assert weights.shape == (PT_END + FEATURE_COUNT - PEBS_END + 1,
                             PEBS_END - PT_END)
    assert bool((score(changed, weights, residual_scale) >
                 score(ordinary, weights, residual_scale)).all())
    assert float(score(ordinary, weights, residual_scale).mean()) < 1.0


def test_standardization_uses_training_only():
    train = torch.ones((2, FEATURE_COUNT))
    other = torch.full((2, FEATURE_COUNT), 9.0)
    center, scale, values = standardize(train, [train, other])
    assert torch.equal(center, train[0])
    assert bool((scale == 0.05).all())
    assert bool((values[0] == 0).all())
    assert bool((values[1] == 8).all())


def test_nuisance_conditioning_uses_explicit_values_and_checks_alignment():
    generator = torch.Generator().manual_seed(13)
    training = torch.randn((400, FEATURE_COUNT), generator=generator)
    loops = torch.linspace(-1.0, 1.0, 400).reshape(-1, 1)
    training[:, PT_END:PEBS_END] = 2 * loops
    weights, residual_scale, _ = fit_conditional(training, loops)
    correct = score(training[:20], weights, residual_scale, nuisance=loops[:20])
    wrong = score(training[:20], weights, residual_scale, nuisance=loops[-20:])
    assert float(correct.mean()) < float(wrong.mean())
    try:
        score(training[:20], weights, residual_scale, nuisance=loops[:19])
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned nuisance rows must fail")


def test_threshold_uses_calibration_only_and_strict_greater():
    scores = [torch.tensor([1., 2., 3.]), torch.tensor([2., 3., 4.]),
              torch.tensor([2., 4.])]
    partition = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3], dtype=torch.uint8)
    entries = ([{"family": "a"}] * 6 + [{"family": "b"}] * 2)
    result = threshold_report(scores, partition, entries, 0.1)
    assert result["threshold"] == 2.0
    assert result["calibration_alerts"] == 1
    assert result["familiar_validation"]["alerts"] == 2
    assert result["heldout_family"]["by_family"]["b"]["alerts"] == 1
