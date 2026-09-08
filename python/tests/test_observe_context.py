# SPDX-License-Identifier: AGPL-3.0-only
"""CR3 follows per-source event order and keeps only compact caller state."""

from contextlib import redirect_stdout
import gc
import io
import json
import unittest
from unittest import mock

import torch

from cpu2tensor import AddressContext, Batch
from cpu2tensor._batching import merge_batches
from cpu2tensor._device import to_device
from cpu2tensor.examples.observe_context import cr3_for_blocks, main


def batch(first, blocks=(), changes=(), *, explicit=True, source=0, worker=0, known=63):
    positions = torch.tensor(blocks, dtype=torch.int64)
    context = None
    if changes:
        sequences, values = zip(*changes)
        zeros = torch.zeros(len(changes), dtype=torch.int64)
        context = AddressContext(
            pc=torch.full_like(zeros, 0x1000), cr0=zeros, cr3=torch.tensor(values),
            cr4=zeros, efer=zeros, cs_base=zeros, mode=torch.full_like(zeros, 64),
            known=torch.full_like(zeros, known),
            sequences=torch.tensor(sequences) if explicit else None,
        )
    return Batch(source, first, torch.full_like(positions, 0x1000), context=context,
                 worker=worker, block_sequences=positions if explicit else None)


class ObserveContextTests(unittest.TestCase):
    def check_device(self, device):
        def move(value):
            return to_device(value, torch.device(device))

        # Identical PCs occur on both sides of two context changes.
        value = move(batch(0, (1, 2, 4, 6, 8), ((0, -(1 << 63)), (3, -1), (7, 0x2000))))
        labels, state = cr3_for_blocks(value)
        self.assertEqual(labels.dtype, torch.int64)
        self.assertEqual(labels.device.type, device)
        self.assertEqual(labels.cpu().tolist(), [-(1 << 63), -(1 << 63), -1, -1, 0x2000])
        self.assertEqual(state.cpu().tolist(), [7, 0x2000])

        # Collation adds explicit positions to otherwise dense legacy frames.
        legacy = [batch(0, changes=((0, 0x3000),), explicit=False),
                  batch(1, (1, 2), explicit=False),
                  batch(3, changes=((3, 0x4000),), explicit=False),
                  batch(4, (4,), explicit=False)]
        labels, state = cr3_for_blocks(move(merge_batches(legacy)))
        self.assertEqual(labels.cpu().tolist(), [0x3000, 0x3000, 0x4000])
        labels, state = cr3_for_blocks(move(batch(5, (5, 6), explicit=False)), state)
        self.assertEqual(labels.cpu().tolist(), [0x4000, 0x4000])

        # A change after the final block becomes the next batch's context.
        labels, state = cr3_for_blocks(move(batch(7, (7,), ((8, 0x5000),))), state)
        self.assertEqual(labels.cpu().tolist(), [0x4000])
        labels, state = cr3_for_blocks(move(batch(9, (9,))), state)
        self.assertEqual(labels.cpu().tolist(), [0x5000])

        # Caller-owned state distinguishes both worker and vCPU, never PC alone.
        states = {}
        for worker, source, raw in ((0, 0, 0x1000), (0, 1, 0x2000), (1, 0, -1)):
            key = worker, source
            _, states[key] = cr3_for_blocks(move(batch(0, changes=((0, raw),),
                                                       worker=worker, source=source, explicit=False)))
        for key, expected in (((0, 0), 0x1000), ((0, 1), 0x2000), ((1, 0), -1)):
            labels, states[key] = cr3_for_blocks(move(batch(1, (1,), worker=key[0], source=key[1])), states[key])
            self.assertEqual(labels.cpu().tolist(), [expected])

        # The state owns sixteen logical bytes instead of the packed batch page.
        value = move(batch(0, tuple(range(1, 2000)), ((0, -1),)))
        _, state = cr3_for_blocks(value)
        self.assertEqual(state.untyped_storage().nbytes(), 16)
        self.assertNotEqual(state.untyped_storage().data_ptr(), value.context.cr3.untyped_storage().data_ptr())
        value.context.cr3.zero_()
        del value
        gc.collect()
        self.assertEqual(state.cpu().tolist(), [0, -1])

        for value in (batch(0, (0,)), batch(0, (0, 2), ((1, 0x1000),)),
                      batch(0, (0,), ((0, 0x1000),))):
            with self.assertRaisesRegex(ValueError, "preceding CR3"):
                cr3_for_blocks(move(value))
        with self.assertRaisesRegex(ValueError, "known CR3"):
            cr3_for_blocks(move(batch(0, (1,), ((0, 0x1000),), known=61)))
        # Upstream's known=15 fallback includes the CR3 bit too.
        labels, _ = cr3_for_blocks(move(batch(0, (1,), ((0, -1),), known=15)))
        self.assertEqual(labels.cpu().tolist(), [-1])

    def test_cpu_sequences_values_and_owned_state(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_sequences_values_and_owned_state(self):
        self.check_device("mps")

    def test_cli_keeps_per_source_state_and_counts_after_completion(self):
        observations = [batch(0, changes=((0, 0x1000),), explicit=False),
                        batch(0, changes=((0, 0x2000),), source=1, explicit=False),
                        batch(1, (1, 2), explicit=False),
                        batch(1, (1,), source=1, explicit=False)]
        pool = mock.MagicMock()
        pool.__enter__.return_value = pool
        pool.read.return_value = iter(observations)
        output = io.StringIO()
        with mock.patch("sys.argv", ["observe-context", "--endpoint", "tcp://fixture:1", "--batch-bytes", "64"]), \
                mock.patch("cpu2tensor.examples.observe_context.Pool", return_value=pool) as make_pool, \
                redirect_stdout(output):
            main()
        make_pool.assert_called_once_with(["tcp://fixture:1"], device="cpu", batch_bytes=64)
        self.assertEqual(json.loads(output.getvalue()),
                         {"workers": 1, "sources": 2, "blocks": 3, "context_rows": 2, "device": "cpu"})
        pool.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
