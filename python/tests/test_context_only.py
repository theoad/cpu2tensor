# SPDX-License-Identifier: AGPL-3.0-only
"""Paging context can accompany blocks without register or memory capture."""

import gc
import struct
import unittest

import torch
from cpu2tensor import Pool
from test_address_context import context
from test_consumer import frame, worker
from test_mixed import run
from test_signals import signal_frame


# x86 system capture with address context, deliberately no register/memory bits.
CONTEXT_ONLY = 2 | (1 << 12) | (1 << 16)
MIXED = 1 << 18


def context_capture(mixed):
    hello = frame(1, detail=CONTEXT_ONLY | (MIXED if mixed else 0))
    if mixed:
        first = (run(12, context(0x1000)) + run(2, struct.pack("<Q", 0x4000))
                 + run(12, context(0x3000)) + run(2, struct.pack("<2Q", 0x4004, 0x4008), 2))
        other = run(12, context(0x9000)) + run(2, struct.pack("<2Q", 0x4000, 0x4004), 2)
        bodies = [
            signal_frame(14, first, count=5),
            signal_frame(14, other, source=1, count=3),
            signal_frame(14, run(2, struct.pack("<Q", 0x4008)), source=1, sequence=3),
            frame(3, source=1, sequence=4),
            signal_frame(14, run(12, context(0x5000)), sequence=5),
        ]
    else:
        bodies = [
            signal_frame(12, context(0x1000)),
            signal_frame(12, context(0x9000), source=1),
            frame(2, addresses=(0x4000,), sequence=1),
            frame(2, addresses=(0x4000, 0x4004), source=1, sequence=1),
            signal_frame(12, context(0x3000), sequence=2),
            frame(2, addresses=(0x4004, 0x4008), sequence=3),
            frame(2, addresses=(0x4008,), source=1, sequence=3),
            frame(3, source=1, sequence=4),
            signal_frame(12, context(0x5000), sequence=5),
        ]
    return hello + b"".join(bodies) + frame(3, sequence=6) + frame(4)


class ContextOnlyTests(unittest.TestCase):
    def check_capture(self, *, mixed, device, batch_bytes):
        retained = []
        batches = []
        with worker(context_capture(mixed), fragment=5) as endpoint:
            with Pool([endpoint], device=device, batch_bytes=batch_bytes) as pool:
                for batch in pool.read():
                    batches.append(batch)
                    self.assertIsNone(batch.registers)
                    self.assertIsNone(batch.memory)
                    self.assertIsNone(batch.layout)
                    self.assertEqual(batch.worker, 0)
                    tensors = [batch.addresses]
                    if batch.block_sequences is not None:
                        tensors.append(batch.block_sequences)
                    if batch.context is not None:
                        tensors.extend(getattr(batch.context, field) for field in
                                       ("pc", "cr0", "cr3", "cr4", "efer", "cs_base", "mode", "known"))
                        if batch.context.sequences is not None:
                            tensors.append(batch.context.sequences)
                    for tensor in tensors:
                        self.assertEqual(tensor.device.type, device)
                        retained.append((tensor, tensor.cpu().clone()))
        del pool
        gc.collect()
        for tensor, saved in retained:
            self.assertTrue(torch.equal(tensor.cpu(), saved))

        if batch_bytes:
            # Each source's final partial group must flush at source_end even
            # though neither source reaches the byte target.
            self.assertEqual([(batch.source, batch.first_sequence) for batch in batches], [(1, 0), (0, 0)])
        elif mixed:
            self.assertEqual(batches[0].context.sequences.cpu().tolist(), [0, 2])
            self.assertEqual(batches[0].block_sequences.cpu().tolist(), [1, 3, 4])

        events = {0: [], 1: []}
        for batch in batches:
            columns = [("block", batch.addresses, batch.block_sequences)]
            if batch.context is not None:
                columns.append(("context", batch.context.cr3, batch.context.sequences))
                self.assertEqual(batch.context.known.cpu().tolist(), [63] * batch.context.pc.numel())
                self.assertEqual(batch.context.mode.cpu().tolist(), [32] * batch.context.pc.numel())
            for kind, values, sequences in columns:
                if not values.numel():
                    continue
                if mixed or batch_bytes:
                    self.assertIsNotNone(sequences)
                    positions = sequences.cpu().tolist()
                else:
                    self.assertIsNone(sequences)
                    positions = range(batch.first_sequence, batch.first_sequence + values.numel())
                events[batch.source].extend(zip(positions, [kind] * values.numel(), values.cpu().tolist()))

        expected = {
            0: [(0, "context", 0x1000), (1, "block", 0x4000), (2, "context", 0x3000),
                (3, "block", 0x4004), (4, "block", 0x4008), (5, "context", 0x5000)],
            1: [(0, "context", 0x9000), (1, "block", 0x4000), (2, "block", 0x4004), (3, "block", 0x4008)],
        }
        associations = {}
        for source, rows in events.items():
            rows.sort()
            self.assertEqual(rows, expected[source])
            current_root = None
            associations[source] = []
            for _, kind, value in rows:
                if kind == "context":
                    current_root = value
                else:
                    self.assertIsNotNone(current_root)
                    associations[source].append((value, current_root))
        self.assertEqual(associations, {
            0: [(0x4000, 0x1000), (0x4004, 0x3000), (0x4008, 0x3000)],
            1: [(0x4000, 0x9000), (0x4004, 0x9000), (0x4008, 0x9000)],
        })

    def test_legacy_cpu(self):
        for batch_bytes in (0, 65536):
            with self.subTest(batch_bytes=batch_bytes):
                self.check_capture(mixed=False, device="cpu", batch_bytes=batch_bytes)

    def test_mixed_cpu(self):
        for batch_bytes in (0, 65536):
            with self.subTest(batch_bytes=batch_bytes):
                self.check_capture(mixed=True, device="cpu", batch_bytes=batch_bytes)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_legacy_mps(self):
        for batch_bytes in (0, 65536):
            with self.subTest(batch_bytes=batch_bytes):
                self.check_capture(mixed=False, device="mps", batch_bytes=batch_bytes)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mixed_mps(self):
        for batch_bytes in (0, 65536):
            with self.subTest(batch_bytes=batch_bytes):
                self.check_capture(mixed=True, device="mps", batch_bytes=batch_bytes)

    def test_each_source_needs_initial_context(self):
        for mixed in (False, True):
            for other_source_has_context in (False, True):
                for batch_bytes in (0, 65536):
                    with self.subTest(mixed=mixed, other_source=other_source_has_context, batch_bytes=batch_bytes):
                        data = frame(1, detail=CONTEXT_ONLY | (MIXED if mixed else 0))
                        if other_source_has_context:
                            data += signal_frame(12, context(0x1000), source=1)
                        if mixed:
                            # A later context in this same mixed frame cannot
                            # retroactively describe the preceding block.
                            body = run(2, struct.pack("<Q", 0x4000)) + run(12, context(0x3000))
                            data += signal_frame(14, body, count=2)
                        else:
                            data += frame(2, addresses=(0x4000,))
                        with worker(data) as endpoint, Pool([endpoint], batch_bytes=batch_bytes) as pool:
                            with self.assertRaisesRegex(ValueError, "initial address context"):
                                list(pool.read())

    def test_unsealed_context_tail_is_incomplete(self):
        data = frame(1, detail=CONTEXT_ONLY) + signal_frame(12, context(0x1000))
        with worker(data) as endpoint, Pool([endpoint], batch_bytes=65536) as pool:
            with self.assertRaisesRegex(RuntimeError, "Incomplete trace"):
                list(pool.read())


if __name__ == "__main__":
    unittest.main()
