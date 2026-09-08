# SPDX-License-Identifier: AGPL-3.0-only
"""Check label-independent marker extraction across arbitrary batch boundaries."""

import unittest
from unittest import mock

import torch

from cpu2tensor import Batch, ExecutableLayout
from cpu2tensor.examples.collect_layout import collect, collect_samples, dataset_from_runs


def layout_batch(base: int = 1000) -> Batch:
    return Batch(None, None, torch.empty(0, dtype=torch.int64),
                 layout=ExecutableLayout(torch.tensor([base, base + 200, base])))


def blocks(values: list[int], source: int = 0) -> Batch:
    return Batch(source, 0, torch.tensor(values, dtype=torch.int64))


class LayoutCollectionTests(unittest.TestCase):
    def test_marker_at_any_chunk_boundary_keeps_previous_block(self) -> None:
        addresses = [1000, 1010, 1100, 1004, 1020, 1100, 1008]
        for width in (1, 2, 3, 7):
            batches = [layout_batch()] + [blocks(addresses[index:index + width])
                                          for index in range(0, len(addresses), width)]
            with self.subTest(width=width):
                result = collect_samples(batches, 100)
                self.assertEqual(result["pc"], [[1010], [1020]])
                self.assertEqual(result["full_block_count"], 7)
                self.assertEqual(result["metadata_frames"], 1)

    def test_missing_duplicate_late_layout_and_invalid_blocks_fail(self) -> None:
        cases = [
            ([blocks([1010, 1100])], "before block"),
            ([layout_batch(), layout_batch()], "exactly one"),
            ([layout_batch(), blocks([1010, 1100]), layout_batch()], "exactly one"),
            ([layout_batch(), blocks([1010]), blocks([1100], source=1)], "one vCPU"),
            ([layout_batch(), blocks([999, 1100])], "text span"),
            ([layout_batch(), blocks([1100])], "preceding block"),
            ([layout_batch(), blocks([1010, 1100, 1100])], "preceding block"),
            ([layout_batch(), blocks([1010])], "sample markers"),
        ]
        for batches, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                collect_samples(batches, 100)

    def test_labels_are_attached_after_complete_trace_and_do_not_select_features(self) -> None:
        with mock.patch("cpu2tensor.examples.collect_layout.Pool") as pool:
            stream = pool.return_value.__enter__.return_value
            stream.read.return_value = [layout_batch(), blocks([1010, 1100, 1020, 1100])]
            first = collect("tcp://fixture:1", b"01", 100)
            second = collect("tcp://fixture:1", b"32", 100)
            self.assertEqual(first["pc"], second["pc"])
            self.assertEqual(first["labels"], [0, 1])
            self.assertEqual(second["labels"], [3, 2])
            with self.assertRaisesRegex(ValueError, "number of complete"):
                collect("tcp://fixture:1", b"0", 100)
            with self.assertRaisesRegex(ValueError, "ASCII digits"):
                collect("tcp://fixture:1", b"0\n", 100)

    def test_transport_failure_is_never_a_complete_capture(self) -> None:
        def interrupted():
            yield layout_batch()
            yield blocks([1010, 1100])
            raise RuntimeError("Incomplete trace")
        with self.assertRaisesRegex(RuntimeError, "Incomplete trace"):
            collect_samples(interrupted(), 100)

    def test_json_runs_preserve_disjoint_balanced_splits(self) -> None:
        runs = []
        for index in range(4):
            base = 1000 + index * 1000
            runs.append({"layout": [base, base + 200, base],
                         "pc": [[base + 10 + label * 4] for label in range(4)],
                         "labels": [0, 1, 2, 3], "train": index < 2})
        dataset = dataset_from_runs({"format": 1, "runs": runs})
        self.assertEqual(dataset["pc"].shape, (16, 1))
        self.assertEqual(dataset["train"].sum().item(), 8)
        self.assertTrue(bool(dataset["mask"].all()))
        runs[0]["pc"][0][0] = 1010.5
        with self.assertRaisesRegex(ValueError, "exact integers"):
            dataset_from_runs({"format": 1, "runs": runs})
        runs[0]["pc"][0][0] = 1010
        runs[3]["layout"] = runs[0]["layout"]
        runs[3]["pc"] = runs[0]["pc"]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            dataset_from_runs({"format": 1, "runs": runs})


if __name__ == "__main__":
    unittest.main()
