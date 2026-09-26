# SPDX-License-Identifier: AGPL-3.0-only
"""Transfer learning preserves frozen weights and faithful token attribution."""
from dataclasses import asdict
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
import torch

from cpu2tensor.examples.hardware_multimodal import HardwareMultimodalBatch, MaskedHardwareModel, MultimodalConfig
from cpu2tensor.examples.hardware_transfer_data import balanced_rows, checked_path, combine
from cpu2tensor.examples.hardware_transfer_experiment import (
    build_detector, check_health, evaluate_preserved, main, metrics, no_masks, pair_loss, score, train_arm,
)


def fixture() -> HardwareMultimodalBatch:
    generator = torch.Generator().manual_seed(23)
    return HardwareMultimodalBatch(
        pt=torch.randn(4, 2, 16, 256, generator=generator),
        pebs=torch.randn(4, 2, 16, 4, generator=generator),
        pmu=torch.randn(4, 2, 1, 3, generator=generator),
        pt_available=torch.ones(4, 2, 16, dtype=torch.bool),
        pebs_available=torch.cat((torch.zeros(4, 1, 16, dtype=torch.bool),
                                  torch.ones(4, 1, 16, dtype=torch.bool)), dim=1),
        pmu_available=torch.ones(4, 2, 1, dtype=torch.bool),
        time_bounds=torch.tensor([0.0, 1.0]).expand(4, 2, 33, 2).clone(),
        timing_quality=torch.ones(4, 2, 33),
    )


def checkpoint(batch: HardwareMultimodalBatch) -> dict:
    model = MaskedHardwareModel(MultimodalConfig(
        pebs_features=4, pmu_features=3, model_dimensions=16,
        attention_heads=2, feedforward_dimensions=32, local_layers=1, cross_cpu_layers=1,
    ))
    model.fit_normalization(batch)
    return {"config": asdict(model.config), "state": model.state_dict()}


def test_encoder_readout_preserves_reconstruction_and_exact_attribution():
    torch.set_num_threads(2)
    batch = fixture()
    detector = build_detector(checkpoint(batch), "frozen", 4)
    detector.eval()
    with torch.no_grad():
        tokens = detector.encoder.encode(batch, no_masks(batch))
        prediction = detector.encoder(batch, no_masks(batch))
        assert torch.equal(prediction.pt, detector.encoder.pt_output(tokens[:, :, :16]))
        contributions = detector.contributions(batch)
        assert bool((contributions[:, 0, 16:32] == 0).all())
        assert torch.allclose(detector(batch), contributions.sum((1, 2)) + detector.head.bias[0], atol=1e-6)


def test_scratch_uses_same_scaling_but_different_weights_and_same_head():
    batch = fixture()
    pretrained = checkpoint(batch)
    scratch = build_detector(pretrained, "scratch", 3)
    frozen = build_detector(pretrained, "frozen", 3)
    assert torch.equal(scratch.encoder.pebs_mean, frozen.encoder.pebs_mean)
    assert torch.equal(scratch.head.weight, frozen.head.weight)
    assert not torch.equal(scratch.encoder.pt_input.weight, frozen.encoder.pt_input.weight)
    assert all(not parameter.requires_grad for parameter in frozen.encoder.parameters())


def test_training_updates_only_the_frozen_head():
    torch.set_num_threads(2)
    batch = fixture()
    payload = {name: getattr(batch, name) for name in batch.__dataclass_fields__}
    data = {"pairs": {"effect": payload, "neutral": payload}, "benign": payload}
    with tempfile.TemporaryDirectory() as directory, patch(
            "cpu2tensor.examples.hardware_transfer_experiment.package_temperature", return_value=50):
        result = train_arm(checkpoint(batch), data, "frozen", 3, 2,
                           float("inf"), Path(directory))
    assert result["steps"] == 2
    assert result["encoder_changed"] is False
    assert result["trainable_parameters"] == 49


def test_metrics_calibrate_without_effect_labels_and_report_ties():
    values = {"effect": torch.tensor([2.0, 1.0]), "neutral": torch.tensor([1.0, 1.0]),
              "calibration": torch.tensor([0.0, 0.5, 1.0]),
              "familiar_validation": torch.tensor([0.5])}
    result = metrics(values)
    values["effect"] = torch.tensor([100.0, 200.0])
    changed = metrics(values)
    assert result["auc"] == 0.75
    assert result["thresholds"]["zero_calibration_alerts"]["threshold"] == 1.0
    assert (changed["thresholds"]["zero_calibration_alerts"]["threshold"] ==
            result["thresholds"]["zero_calibration_alerts"]["threshold"])
    assert not result["thresholds"]["zero_calibration_alerts"]["certifies_operational_fpr"]


def test_pair_loss_favors_effect_and_hard_negative_separation():
    assert pair_loss(torch.tensor([2.0]), torch.tensor([-2.0]), torch.tensor([-2.0])) < pair_loss(
        torch.tensor([-2.0]), torch.tensor([2.0]), torch.tensor([2.0]))


def test_selection_is_balanced_and_partition_disjoint():
    rows = [{"partition": partition, "family": family, "loops": loops, "execution_id": index}
            for index, (partition, family, loops) in enumerate(
                (partition, family, loops) for partition in ("training", "calibration")
                for family in ("getpid", "pipe") for loops in (5, 10) for _ in range(5))]
    training = balanced_rows(rows, "training", 8)
    calibration = balanced_rows(rows, "calibration", 8)
    assert {row["execution_id"] for row in training}.isdisjoint(
        row["execution_id"] for row in calibration)
    assert len({(row["family"], row["loops"]) for row in training}) == 4


def test_custody_rejects_path_escape_before_loading():
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(ValueError, match="path/hash"):
            checked_path(Path(directory), "../not-owned.pt", "0" * 64)


def test_combine_rejects_empty_partition():
    with pytest.raises(ValueError, match="empty"):
        combine([])


def test_mask_rng_does_not_change_training_row_schedule():
    torch.set_num_threads(2)
    batch = fixture()
    pretrained = checkpoint(batch)
    payload = {name: getattr(batch, name) for name in batch.__dataclass_fields__}
    data = {"pairs": {"effect": payload, "neutral": payload}, "benign": payload}
    with tempfile.TemporaryDirectory() as directory, patch(
            "cpu2tensor.examples.hardware_transfer_experiment.package_temperature", return_value=50):
        results = [train_arm(pretrained, data, arm, 11, 3, float("inf"), Path(directory))
                   for arm in ("scratch", "frozen", "finetuned")]
    assert len({row["row_schedule_sha256"] for row in results}) == 1


def test_thermal_guard_stops_before_more_training():
    with patch("cpu2tensor.examples.hardware_transfer_experiment.package_temperature", return_value=81):
        with pytest.raises(RuntimeError, match="temperature"):
            check_health(float("inf"))


def test_thermal_guard_requires_sensor():
    with patch("cpu2tensor.examples.hardware_transfer_experiment.package_temperature", return_value=None):
        with pytest.raises(RuntimeError, match="sensor required"):
            check_health(float("inf"))


def test_off_host_scoring_cannot_bypass_linux_thermal_guard():
    batch = fixture()
    payload = {name: getattr(batch, name) for name in batch.__dataclass_fields__}
    with patch("cpu2tensor.examples.hardware_transfer_experiment.platform.system", return_value="Linux"):
        with pytest.raises(RuntimeError, match="Mac coordinator"):
            score(build_detector(checkpoint(batch), "frozen", 4), payload, physical_host=False)


def test_partial_scoring_flags_never_start_training():
    arguments = ["transfer", "data", "pretrained", "baseline", "output", "--plan-sha256", "0",
                 "--checkpoint-sha256", "scratch=0", "frozen=0", "finetuned=0"]
    with patch("sys.argv", arguments), patch(
            "cpu2tensor.examples.hardware_transfer_experiment.run") as training:
        with pytest.raises(SystemExit) as failure:
            main()
        assert failure.value.code == 2
        training.assert_not_called()


def test_preserved_scoring_cannot_write_inside_immutable_inputs():
    with tempfile.TemporaryDirectory() as directory, patch(
            "cpu2tensor.examples.hardware_transfer_experiment.platform.system", return_value="Darwin"):
        root = Path(directory)
        with pytest.raises(ValueError, match="outside preserved"):
            evaluate_preserved(root, root, root / "baseline", root / "new-output", "0",
                               {arm: "0" for arm in ("scratch", "frozen", "finetuned")}, 1729)


def test_rejected_existing_cli_output_does_not_overwrite_custody(tmp_path):
    original = tmp_path / "failure.json"
    original.write_text("preserved evidence")
    arguments = ["transfer", str(tmp_path), "pretrained", "baseline", str(tmp_path),
                 "--plan-sha256", "0", "--score-checkpoints", str(tmp_path),
                 "--checkpoint-sha256", "scratch=0", "frozen=0", "finetuned=0"]
    with patch("sys.argv", arguments):
        with pytest.raises(SystemExit) as failure:
            main()
        assert failure.value.code == 2
    assert original.read_text() == "preserved evidence"
