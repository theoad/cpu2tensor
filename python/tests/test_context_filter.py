# SPDX-License-Identifier: AGPL-3.0-only
"""A filtered two-vCPU stream keeps target transactions across migration."""

import struct
import unittest

from cpu2tensor import Pool
from test_address_context import context
from test_consumer import frame, worker
from test_signals import memory_row, signal_frame


FEATURES = ((1 << 8) | (1 << 10) | (1 << 12) | (1 << 15) |
            (1 << 16) | (1 << 21))


def current(frame_bytes):
    """Use wire v3 while shared compatibility fixtures keep their v2 default."""
    return frame_bytes[:4] + struct.pack("<H", 3) + frame_bytes[6:]


def summary(*, policy=1, latch=2, source=0, gate=0x81001000, cr3=0x12000,
            kept=2, dropped=2, matching=2, foreign=2, unknown=0):
    payload = struct.pack("<IIIIQQQQQQQQ", policy, latch, source, 0, gate, cr3,
                          cr3 & ~0xfff,
                          kept, dropped, matching, foreign, unknown)
    return current(signal_frame(18, payload))


def rich_access(context_sequence, physical, value):
    return (memory_row(address=0x8ffe, size=4)
            + struct.pack("<QIIQ", physical, 2, 3, context_sequence)
            + value + bytes(16 - len(value)))


class ContextFilterTests(unittest.TestCase):
    def test_target_values_survive_vcpu_migration_and_background_is_absent(self):
        target = 0x12000
        background = 0x34000
        value0 = bytes.fromhex("44332211")
        value1 = bytes.fromhex("88776655")
        data = b"".join((
            current(frame(1, detail=2 | FEATURES)),
            current(signal_frame(12, context(target), source=0)),
            current(frame(2, source=0, sequence=1, addresses=(0x81001000,))),
            current(signal_frame(8, rich_access(0, 0x20ffe, value0), source=0, sequence=2)),
            current(signal_frame(12, context(background), source=0, sequence=3)),
            current(frame(2, source=0, sequence=4, addresses=(0x81002000,))),
            current(signal_frame(12, context(background), source=1)),
            current(frame(2, source=1, sequence=1, addresses=(0x81003000,))),
            current(signal_frame(12, context(target), source=1, sequence=2)),
            current(frame(2, source=1, sequence=3, addresses=(0x81004000,))),
            current(signal_frame(8, rich_access(2, 0x30ffe, value1),
                                 source=1, sequence=4)),
            current(frame(3, source=0, sequence=5)),
            current(frame(3, source=1, sequence=5)),
            summary(),
            current(frame(4)),
        ))
        with worker(data) as endpoint, Pool([endpoint], batch_bytes=4096) as pool:
            batches = list(pool.read())

        memory = [(batch.source, batch.memory.values.tolist()) for batch in batches
                  if batch.memory is not None]
        self.assertEqual(memory, [
            (0, [list(value0 + bytes(12))]),
            (1, [list(value1 + bytes(12))]),
        ])
        self.assertEqual(
            [(batch.source, batch.addresses.tolist()) for batch in batches
             if batch.addresses.numel()],
            [(0, [0x81001000, 0x81002000]), (1, [0x81003000, 0x81004000])],
        )
        metadata = batches[-1]
        self.assertIsNone(metadata.source)
        self.assertEqual(metadata.context_filter.policy, "drop")
        self.assertTrue(metadata.context_filter.latch_known)
        self.assertEqual((metadata.context_filter.gate_source, metadata.context_filter.gate_pc,
                          metadata.context_filter.cr3), (0, 0x81001000, target))
        self.assertEqual(metadata.context_filter.paging_root, target)
        self.assertEqual((metadata.context_filter.kept, metadata.context_filter.dropped,
                          metadata.context_filter.matching, metadata.context_filter.foreign,
                          metadata.context_filter.unknown), (2, 2, 2, 2, 0))

    def test_missing_or_inconsistent_summary_is_rejected(self):
        cases = (
            b"",
            summary(kept=3),
            summary(policy=2, dropped=1, kept=4),
        )
        for tail in cases:
            with self.subTest(tail=tail[:32]):
                data = (current(frame(1, detail=2 | FEATURES)) + current(frame(3))
                        + tail + current(frame(4)))
                with worker(data) as endpoint, Pool([endpoint]) as pool, self.assertRaises(ValueError):
                    list(pool.read())


if __name__ == "__main__":
    unittest.main()
