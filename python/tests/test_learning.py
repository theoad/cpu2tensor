# SPDX-License-Identifier: AGPL-3.0-only
"""Check sample boundaries, label isolation and a complete small learning run."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from cpu2tensor import Batch
from cpu2tensor.examples.learn_trace import (
    FEATURE_NAMES, count_samples, make_inputs, parse_symbols, save_dataset, split_indices, train,
    validate_dataset,
)


def fixture_dataset(per_class: int = 30) -> dict:
    inputs, labels = make_inputs(per_class)
    features = torch.zeros(labels.numel(), 10)
    features.scatter_add_(1, inputs.to(torch.int64), torch.ones(labels.numel(), 16))
    return {"format": 1, "features": features, "labels": labels,
            "inputs": inputs, "feature_names": FEATURE_NAMES}


class LearningTests(unittest.TestCase):
    def test_nm_symbols_require_distinct_code_entries(self) -> None:
        lines = [f"{0x400000 + digit * 32:016x} T digit_{digit}" for digit in range(10)]
        text = "\n".join(["                 U puts", *lines, "0000000000401000 T sample_end"])
        symbols = parse_symbols(text)
        self.assertEqual(symbols["digit_9"], 0x400120)
        self.assertEqual(symbols["sample_end"], 0x401000)
        self.assertEqual(len(parse_symbols("\n".join(lines))), 10)
        with self.assertRaisesRegex(ValueError, "Missing digit"):
            parse_symbols("\n".join(lines[:-1]))
        with self.assertRaisesRegex(ValueError, "one code symbol"):
            parse_symbols(text + "\n0000000000400000 T digit_0")
        with self.assertRaisesRegex(ValueError, "distinct addresses"):
            parse_symbols(text.replace("0000000000401000", "0000000000400000"))

    def test_chunk_boundaries_and_unrelated_runtime_blocks(self) -> None:
        entries = list(range(100, 110))
        first = [100] * 10 + [101] * 6
        second = [109] * 9 + [102] * 7
        addresses = [999, *first, 200, 999, *second, 200, 999]
        # Every chunk width must produce the same samples, including a marker
        # at the first or last position and a sample crossing several batches.
        for width in (1, 2, 7, 17, 128):
            with self.subTest(width=width):
                batches = [Batch(0, offset, torch.tensor(addresses[offset:offset + width]))
                           for offset in range(0, len(addresses), width)]
                result = count_samples(batches, entries, 200)
                self.assertEqual(result.tolist(), [[10, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                                                   [0, 0, 7, 0, 0, 0, 0, 0, 0, 9]])

    def test_missing_marker_and_multiple_sources_are_rejected(self) -> None:
        entries = list(range(100, 110))
        for values in ([100] * 16, [100] * 15 + [200], [200]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                count_samples([Batch(0, 0, torch.tensor(values))], entries, 200)
        with self.assertRaisesRegex(ValueError, "single-vCPU"):
            count_samples([Batch(0, 0, torch.tensor([100] * 16)),
                           Batch(1, 0, torch.tensor([200]))], entries, 200)

    def test_independent_input_oracle_and_duplicate_rejection(self) -> None:
        dataset = fixture_dataset()
        features, labels, inputs = validate_dataset(dataset)
        self.assertEqual(features.shape, (300, 10))
        self.assertEqual(torch.bincount(labels).tolist(), [30] * 10)
        features[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "independent input histogram"):
            validate_dataset(dataset)
        features[0, 0] -= 1
        inputs[0] = inputs[1]
        with self.assertRaisesRegex(ValueError, "Duplicate input"):
            validate_dataset(dataset)

    def test_split_is_reproducible_disjoint_and_balanced(self) -> None:
        _, labels = make_inputs()
        training, testing = split_indices(labels)
        again = split_indices(labels)
        self.assertTrue(torch.equal(training, again[0]))
        self.assertTrue(torch.equal(testing, again[1]))
        self.assertFalse(set(training.tolist()) & set(testing.tolist()))
        self.assertEqual(sorted(torch.cat((training, testing)).tolist()), list(range(300)))
        self.assertEqual(torch.bincount(labels[training]).tolist(), [24] * 10)
        self.assertEqual(torch.bincount(labels[testing]).tolist(), [6] * 10)

    def test_training_and_checkpoint_on_controlled_features(self) -> None:
        # This is a mathematical fixture, not evidence of live QEMU capture.
        dataset = fixture_dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_dataset(path / "dataset.pt", dataset["features"], dataset["labels"],
                         dataset["inputs"], host="unit-test fixture", capture="synthetic counts")
            metrics = train(path / "dataset.pt", path / "result")
            self.assertTrue(metrics["passed"])
            self.assertGreaterEqual(metrics["test_accuracy"], 0.9)
            self.assertLess(metrics["shuffled_label_accuracy"], 0.3)
            self.assertLess(metrics["shuffled_trace_accuracy"], 0.3)
            checkpoint = torch.load(path / "result/model.pt", weights_only=True)
            self.assertEqual(checkpoint["feature_names"], FEATURE_NAMES)
            self.assertEqual(checkpoint["state_dict"]["weight"].shape, (10, 10))
            restored = torch.nn.Linear(10, 10)
            restored.load_state_dict(checkpoint["state_dict"])
            test_rows = checkpoint["test_rows"]
            with torch.no_grad():
                predictions = restored(dataset["features"][test_rows] / checkpoint["input_scale"]).argmax(1)
            accuracy = float((predictions == dataset["labels"][test_rows]).float().mean())
            self.assertEqual(accuracy, metrics["test_accuracy"])
            self.assertEqual(json.loads((path / "result/metrics.json").read_text()), metrics)


if __name__ == "__main__":
    unittest.main()
