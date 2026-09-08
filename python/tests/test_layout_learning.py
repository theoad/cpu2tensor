# SPDX-License-Identifier: AGPL-3.0-only
"""Mathematical fixtures for relocation learning, not live QEMU evidence."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from cpu2tensor.examples.learn_layout import (
    PathModel, address_features, encode, fit_vocabulary, run, train_one, validate_dataset, wrong_layout,
)


def fixture_dataset() -> dict:
    # Independent synthetic branch traces, with equal labels at every location.
    pc, layout, labels, train = [], [], [], []
    for location in range(6):
        base = (1 << 54) + location * 4096
        for label in range(4):
            for _ in range(8):
                pc.append([base, base + 4 + label * 16, base + 128, 0])
                layout.append([base, base + 256, base])
                labels.append(label)
                train.append(location < 4)
    addresses = torch.tensor(pc, dtype=torch.int64)
    return {"format": 1, "pc": addresses, "mask": addresses != 0,
            "layout": torch.tensor(layout, dtype=torch.int64),
            "labels": torch.tensor(labels), "train": torch.tensor(train)}


class LayoutLearningTests(unittest.TestCase):
    def test_integer_subtraction_preserves_low_bits_above_float_precision(self) -> None:
        data = fixture_dataset()
        values = address_features(data["pc"], data["layout"], True)
        self.assertEqual(values[0, :3].tolist(), [0, 4, 128])
        self.assertEqual(values[8, :3].tolist(), [0, 20, 128])
        vocabulary = fit_vocabulary(values, data["mask"], data["train"])
        self.assertEqual(vocabulary.tolist(), [0, 4, 20, 36, 52, 128])
        tokens = encode(values, vocabulary)
        self.assertTrue(torch.equal(tokens[0], tokens[32]))

    def test_vocabulary_never_learns_heldout_addresses(self) -> None:
        data = fixture_dataset()
        vocabulary = fit_vocabulary(data["pc"], data["mask"], data["train"])
        tokens = encode(data["pc"], vocabulary)
        self.assertTrue(bool((tokens[~data["train"]] == 0).all()))
        self.assertTrue(bool((tokens[data["train"]][data["mask"][data["train"]]] > 0).all()))

    def test_wrong_metadata_has_no_fixed_points(self) -> None:
        data = fixture_dataset()
        original = data["layout"][~data["train"]]
        wrong = wrong_layout(original)
        self.assertTrue(bool((wrong[:, 0] != original[:, 0]).all()))
        self.assertTrue(torch.equal(original.unique(dim=0), wrong.unique(dim=0)))

    def test_invalid_and_leaking_datasets_are_rejected(self) -> None:
        for change, message in (
            (lambda d: d["train"].__setitem__(0, False), "disjoint"),
            (lambda d: d["labels"].__setitem__(0, 1), "equally"),
            (lambda d: d["pc"].__setitem__((0, 0), 1), "text span"),
            (lambda d: d["mask"].__setitem__(0, False), "nonempty"),
            (lambda d: d.__setitem__("pc", d["pc"].double()), "int64"),
        ):
            data = fixture_dataset()
            change(data)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_dataset(data)

    def test_heldout_labels_cannot_change_training(self) -> None:
        original = validate_dataset(fixture_dataset())
        changed = {name: value.clone() for name, value in original.items()}
        changed["labels"][~changed["train"]] = (changed["labels"][~changed["train"]] + 1) % 4
        for mode in ("raw", "normalized", "metadata_only", "shuffled_labels"):
            _, first = train_one(original, mode, seed=7, epochs=3)
            _, second = train_one(changed, mode, seed=7, epochs=3)
            with self.subTest(mode=mode):
                for name in first["state_dict"]:
                    self.assertTrue(torch.equal(first["state_dict"][name], second["state_dict"][name]))

    def test_nonfinite_logits_and_gradients_stop_before_results(self) -> None:
        data = validate_dataset(fixture_dataset())
        for failure in ("logits", "gradient"):
            model = PathModel(128)
            if failure == "logits":
                with torch.no_grad():
                    model.classifier.bias.fill_(float("nan"))
            else:
                model.classifier.bias.register_hook(lambda gradient: gradient * float("inf"))
            with self.subTest(failure=failure), mock.patch(
                "cpu2tensor.examples.learn_layout.PathModel", return_value=model,
            ), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                with self.assertRaisesRegex(RuntimeError, "Nonfinite " + failure):
                    run({"format": 1, **data}, output, seeds=(7,), epochs=1)
                self.assertEqual(list(output.iterdir()), [])

    def test_learning_controls_and_checkpoint_roundtrip(self) -> None:
        data = fixture_dataset()
        with tempfile.TemporaryDirectory() as directory:
            report = run(data, Path(directory), seeds=(7,), epochs=40)
            by_mode = {result["mode"]: result for result in report["runs"]}
            normal = by_mode["normalized"]
            self.assertEqual(normal["history"][-1]["heldout_accuracy"], 1.0)
            self.assertLess(normal["history"][-1]["train_loss"], normal["history"][0]["train_loss"])
            self.assertEqual(normal["wrong_metadata_accuracy"], 0.25)
            for name in ("raw", "metadata_only"):
                self.assertEqual(by_mode[name]["history"][-1]["heldout_accuracy"], 0.25)
            self.assertEqual(by_mode["raw"]["history"][-1]["train_accuracy"], 1.0)
            shuffled = by_mode["shuffled_labels"]["history"][-1]
            self.assertGreater(shuffled["train_loss"], 1.2)
            self.assertLess(shuffled["heldout_correct_probability"], 0.4)
            self.assertTrue(all(result["checkpoint_reload"] for result in report["runs"]))
            self.assertEqual(len({result["parameter_capacity"] for result in report["runs"]}), 1)
            self.assertEqual(json.loads((Path(directory) / "metrics.json").read_text()), report)
            checkpoint = torch.load(Path(directory) / "normalized-7.pt", weights_only=True)
            self.assertEqual(checkpoint["vocabulary"].tolist(), [0, 4, 20, 36, 52, 128])


if __name__ == "__main__":
    unittest.main()
