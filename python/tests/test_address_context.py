# SPDX-License-Identifier: AGPL-3.0-only
"""Context references and physical coverage survive validation and tensor ownership."""
import gc
import struct
import unittest

import torch
from cpu2tensor import Pool
from test_consumer import frame, worker
from test_signals import signal_frame, finish, memory_row, schema_row, register_row

FEATURES = 2 | (1 << 8) | (1 << 12) | (1 << 15) | (1 << 16)


def context(root=0x1000, mode=32, known=63):
    return struct.pack('<8Q', 0x7c00, 0x80000011, root, 0, 0, 0, mode, known)


def access(sequence=0, physical=0x20ffe, mapped=2, flags=3, values=False):
    data = memory_row(address=0x8ffe, size=4)
    data += struct.pack('<QIIQ', physical, mapped, flags, sequence)
    return data + (bytes.fromhex('44332211') + bytes(12) if values else b'')


class ContextTests(unittest.TestCase):
    def test_legacy_system_register_capture_is_not_raw_state(self):
        data = frame(1, detail=2 | (1 << 9) | (1 << 12))
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaisesRegex(ValueError, 'requires address context; update the worker'):
                list(pool.read())

    def check_owned(self, device):
        data = b''.join((frame(1, detail=FEATURES | (1 << 10)),
            signal_frame(12, context()),
            signal_frame(8, access(values=True), sequence=1),
            signal_frame(12, context(0x4000), sequence=2),
            signal_frame(8, access(sequence=2, physical=0x40ffe, flags=1, values=True), sequence=3),
            finish(4)))
        with worker(data, fragment=7) as endpoint, Pool([endpoint], device=device) as pool:
            batches = list(pool.read())
        gc.collect()
        self.assertEqual(batches[0].context.cr3.cpu().tolist(), [0x1000])
        self.assertEqual(batches[2].context.cr3.cpu().tolist(), [0x4000])
        self.assertEqual(batches[0].context.known.cpu().tolist(), [63])
        first, last = batches[1].memory, batches[3].memory
        self.assertEqual(first.physical_addresses.cpu().tolist(), [0x20ffe])
        self.assertEqual(first.mapped_sizes.cpu().tolist(), [2])
        self.assertEqual(first.sizes.cpu().tolist(), [4])
        self.assertEqual(last.context_sequences.cpu().tolist(), [2])
        self.assertEqual(last.mapping_flags.cpu().tolist(), [1])
        self.assertEqual(first.values.cpu().tolist(), [list(bytes.fromhex('44332211') + bytes(12))])
        model = torch.nn.Linear(16, 1, device=device)
        model(first.values.float()).sum().backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all().item())

    def test_cpu_context_and_partial_mapping(self):
        self.check_owned('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_context_and_partial_mapping(self):
        self.check_owned('mps')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA remains unverified without hardware')
    def test_cuda_context_and_partial_mapping(self):
        self.check_owned('cuda')

    def test_unknowns_are_not_known_zero(self):
        data = frame(1, detail=FEATURES) + signal_frame(12, context(mode=0, known=15))
        data += signal_frame(8, access(physical=0, mapped=0, flags=0), sequence=1) + finish(2)
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            batches = list(pool.read())
        self.assertEqual(batches[0].context.known.tolist(), [15])
        self.assertEqual(batches[1].memory.mapping_flags.tolist(), [0])

    def test_rejects_ambiguous_context_and_mapping(self):
        cases = {
            'future context': signal_frame(8, access(sequence=1), sequence=1),
            'cross page claimed contiguous': signal_frame(8, access(mapped=4), sequence=1),
            'known zero missing validity': signal_frame(8, access(flags=0), sequence=1),
            'io without known kind': signal_frame(8, access(flags=5), sequence=1),
            'physical wrap': signal_frame(8, access(physical=2**64-1), sequence=1),
            'unknown mode with known flag': signal_frame(12, context(mode=0), sequence=1),
        }
        for name, invalid in cases.items():
            with self.subTest(name=name):
                data = frame(1, detail=FEATURES) + signal_frame(12, context()) + invalid + finish(2)
                with worker(data) as endpoint, Pool([endpoint]) as pool, self.assertRaises(ValueError):
                    list(pool.read())

    def test_context_cannot_leak_between_sources_or_go_stale(self):
        invalids = [signal_frame(8, access(), source=1),
                    signal_frame(12, context(0x4000), sequence=1)
                    + signal_frame(8, access(sequence=0), sequence=2)]
        for invalid in invalids:
            with self.subTest(invalid=invalid[:32]):
                data = frame(1, detail=FEATURES) + signal_frame(12, context()) + invalid
                with worker(data) as endpoint, Pool([endpoint]) as pool, self.assertRaises(ValueError):
                    list(pool.read())

    def test_raw_registers_and_blocks_require_context(self):
        for body in (frame(2, addresses=(0x7c00,)),
                     signal_frame(6, schema_row(name=b'rax')) + signal_frame(7, register_row(bytes(8)))):
            data = frame(1, detail=FEATURES | (1 << 9)) + body
            with worker(data) as endpoint, Pool([endpoint]) as pool, self.assertRaises(ValueError):
                list(pool.read())
