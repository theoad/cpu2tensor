# SPDX-License-Identifier: AGPL-3.0-only
"""Check source boundaries, bounded minibatches and stream completion in training."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from cpu2tensor import Batch
from cpu2tensor.examples.pretrain_kernel import (
    BlockPairs, PairBatches, HeldOutBatches, evaluate, make_model, train, worker_batches,
)
from test_consumer import frame, worker


def trace(addresses: tuple[int, ...]) -> bytes:
    parts = [frame(1, detail=1)]
    for offset in range(0, len(addresses), 256):
        parts.append(frame(2, sequence=offset, addresses=addresses[offset:offset + 256]))
    parts.extend((frame(3, sequence=len(addresses)), frame(4)))
    return b"".join(parts)


class PretrainingTests(unittest.TestCase):
    def test_pairs_never_cross_workers_or_sources(self) -> None:
        pairs = BlockPairs(256)
        make = lambda source, values: Batch(source, 0, torch.tensor(values, dtype=torch.int64))
        self.assertEqual(pairs.add(0, make(0, [4, 8])).tolist(), [[1, 2]])
        self.assertEqual(pairs.add(0, make(1, [80])).tolist(), [])
        self.assertEqual(pairs.add(1, make(0, [120])).tolist(), [])
        self.assertEqual(pairs.add(0, make(0, [12])).tolist(), [[2, 3]])
        self.assertEqual(pairs.add(0, make(1, [84])).tolist(), [[20, 21]])
        self.assertEqual(pairs.add(1, make(0, [124])).tolist(), [[30, 31]])
        pairs.end(0)
        self.assertEqual(pairs.add(0, make(0, [16])).tolist(), [])
        high = make(0, [-1, -(1 << 63)])
        original = high.addresses.clone()
        self.assertEqual(pairs.add(2, high).shape, (1, 2))
        self.assertTrue(torch.equal(original, high.addresses), "Feature hashing changed raw addresses")

    def test_partial_minibatches_need_no_padding_and_own_storage(self) -> None:
        batches = PairBatches(4)
        pairs = torch.arange(22).reshape(11, 2)
        result = list(batches.add(pairs[:3])) + list(batches.add(pairs[3:]))
        result.append(batches.finish())
        self.assertEqual([len(batch) for batch in result], [4, 4, 3])
        self.assertTrue(torch.equal(torch.cat(result), pairs))
        list(batches.add(torch.zeros((8, 2), dtype=torch.int64)))
        self.assertTrue(torch.equal(torch.cat(result), pairs), "Minibatch storage was reused")

    def test_test_reservoir_is_bounded_and_reproducible(self) -> None:
        first, second = HeldOutBatches(3, 17), HeldOutBatches(3, 17)
        for index in range(100):
            batch = torch.full((4, 2), index, dtype=torch.int64)
            first.add(batch)
            second.add(batch)
        self.assertEqual(first.seen, 100)
        self.assertEqual(len(first.batches), 3)
        self.assertTrue(torch.equal(torch.stack(first.batches), torch.stack(second.batches)))

    def test_early_consumer_exit_cancels_reader(self) -> None:
        data = frame(1, detail=1) + frame(2, addresses=(4, 8))
        with worker(data, wait_for_close=True) as endpoint:
            with worker_batches([endpoint], queue_batches=1, timeout=1) as batches:
                index, batch = next(batches)
                self.assertEqual(index, 0)
                self.assertEqual(batch.addresses.tolist(), [4, 8])
        # Fixture cleanup confirms the socket was closed and its thread stopped.

    def test_incomplete_worker_is_not_a_successful_end_marker(self) -> None:
        data = frame(1, detail=1) + frame(2, addresses=(4, 8))
        with worker(data) as endpoint:
            with worker_batches([endpoint], timeout=1) as batches:
                with self.assertRaisesRegex(RuntimeError, "failed before completion"):
                    list(batches)

    def test_budget_drains_both_runs_and_checkpoint_reloads(self) -> None:
        training = tuple((index % 4 + 1) * 4 for index in range(3000))
        testing = tuple((index % 4 + 1) * 4 for index in range(1500))
        with tempfile.TemporaryDirectory() as directory:
            with worker(trace(training), fragment=4096) as train_endpoint:
                with worker(trace(testing), fragment=4096) as test_endpoint:
                    result = train([train_endpoint], [test_endpoint], directory,
                                   vocabulary=16, batch_size=128, max_updates=3,
                                   queue_batches=1, test_batches=3, capture_host="TCP fixtures")
            self.assertTrue(result["all_traces_complete"])
            self.assertEqual(result["workers_completed"], 2)
            self.assertEqual(result["updates"], 3)
            self.assertEqual(result["observed_train_pairs"], 2999)
            self.assertEqual(result["observed_test_pairs"], 1499)
            self.assertEqual(result["unused_training_pairs"], 2999 - 384)
            self.assertLessEqual(result["test_minibatches_retained"], 3)
            saved = torch.load(Path(directory) / "model.pt", weights_only=True)
            model = make_model(saved["vocabulary"], 0)
            model.load_state_dict(saved["state_dict"])
            held_out = torch.load(Path(directory) / "test-pairs.pt", weights_only=True)
            self.assertEqual(evaluate(model, held_out, "cpu"), result["final_test"])
            self.assertEqual(json.loads((Path(directory) / "metrics.json").read_text()), result)

    def test_late_capture_failure_after_update_budget_still_fails(self) -> None:
        addresses = tuple((index % 4 + 1) * 4 for index in range(3000))
        incomplete = trace(addresses)[:-32]  # All data and source end, but no completion.
        with tempfile.TemporaryDirectory() as directory:
            with worker(incomplete, fragment=4096) as train_endpoint:
                with worker(trace(addresses), fragment=4096) as test_endpoint:
                    with self.assertRaisesRegex(RuntimeError, "failed before completion"):
                        train([train_endpoint], [test_endpoint], directory,
                              vocabulary=16, batch_size=128, max_updates=1, queue_batches=1)
            self.assertFalse((Path(directory) / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
