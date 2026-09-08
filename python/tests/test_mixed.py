# SPDX-License-Identifier: AGPL-3.0-only
"""Mixed columns keep their original sequence and independent storage."""
import gc
import struct
import unittest
import torch
from cpu2tensor import Pool
from test_consumer import frame, worker
from test_signals import schema_row, signal_frame, register_row
from test_address_context import FEATURES, context, access


def run(kind, body, count=1):
    return struct.pack('<HHI', kind, count, len(body)) + body


def capture():
    body = (run(12, context()) + run(7, register_row(bytes.fromhex('efcdab8967452301'))) +
            run(2, struct.pack('<Q', 0x7c00)) + run(8, access(values=True)) +
            run(12, context(0x4000)) + run(7, register_row(bytes(8), flags=1)) +
            run(2, struct.pack('<2Q', 0x7c04, 0x7c08), 2) +
            run(8, access(sequence=4, physical=0x40ffe, values=True)))
    return (frame(1, detail=FEATURES | (1 << 9) | (1 << 10) | (1 << 18)) +
            signal_frame(6, schema_row(name=b'rax')) + signal_frame(14, body, count=9) +
            frame(3, sequence=9) + frame(4))


class MixedTests(unittest.TestCase):
    def check_owned(self, device):
        with worker(capture(), fragment=3) as endpoint, Pool([endpoint], device=device) as pool:
            batches = list(pool.read())
        gc.collect()
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(batch.block_sequences.cpu().tolist(), [2, 6, 7])
        self.assertEqual(batch.registers.sequences.cpu().tolist(), [1, 5])
        self.assertEqual(batch.memory.sequences.cpu().tolist(), [3, 8])
        self.assertEqual(batch.context.sequences.cpu().tolist(), [0, 4])
        self.assertEqual(batch.memory.context_sequences.cpu().tolist(), [0, 4])
        self.assertEqual(batch.context.cr3.cpu().tolist(), [0x1000, 0x4000])
        self.assertEqual(batch.addresses.cpu().tolist(), [0x7c00, 0x7c04, 0x7c08])
        self.assertEqual(batch.registers.values.cpu().tolist()[0], list(bytes.fromhex('efcdab8967452301')))
        model = torch.nn.Linear(16, 2, device=device)
        model(batch.memory.values.float()).sum().backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all().item())

    def test_cpu(self): self.check_owned('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps(self): self.check_owned('mps')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA remains unvalidated')
    def test_cuda(self): self.check_owned('cuda')

    def test_invalid_runs_fail_in_native_decoder(self):
        bodies = [
            (run(14, bytes(8)), 1),
            (run(2, bytes(8)), 2),
            (run(2, bytes(8)) + b'X', 1),
            (struct.pack('<HHI', 2, 1, 99) + bytes(8), 1),
            (run(8, access()), 1),  # No address context or complete register baseline.
            (run(2, bytes(8), count=0), 1),
        ]
        for body, count in bodies:
            data = frame(1, detail=1 | (1 << 18)) + signal_frame(14, body, count=count)
            with self.subTest(body=body), worker(data) as endpoint, Pool([endpoint]) as pool:
                with self.assertRaises(ValueError): list(pool.read())

    def test_mixed_requires_advertised_feature(self):
        data = frame(1, detail=1) + signal_frame(14, run(2, bytes(8)))
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaisesRegex(ValueError, 'Mixed capture'): list(pool.read())
