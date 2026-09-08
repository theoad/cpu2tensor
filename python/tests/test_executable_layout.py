# SPDX-License-Identifier: AGPL-3.0-only
"""Worker metadata has owned storage and cannot masquerade as CPU progress."""
import gc
import struct
import unittest

import torch
from cpu2tensor import Pool
from cpu2tensor.examples.benchmark_capture import TraceCounter
from test_consumer import frame, worker
from test_signals import schema_row, signal_frame, register_row

FEATURE = 1 << 17


def layout(start=0x400000, end=0x401000, entry=0x400100, **fields):
    return signal_frame(13, struct.pack('<3Q', start, end, entry), **fields)


class LayoutTests(unittest.TestCase):
    def check_owned(self, device):
        data = (frame(1, detail=1 | FEATURE) + layout() +
                frame(2, addresses=(0x400100,)) + frame(3, sequence=1) + frame(4))
        with worker(data, fragment=1) as endpoint, Pool([endpoint], device=device) as pool:
            batches = list(pool.read())
        gc.collect()
        first = batches[0]
        self.assertIsNone(first.source)
        self.assertIsNone(first.first_sequence)
        self.assertEqual(first.addresses.numel(), 0)
        self.assertEqual(first.layout.values.cpu().tolist(), [0x400000, 0x401000, 0x400100])
        self.assertEqual(first.layout.code_start.item(), 0x400000)
        self.assertEqual(first.layout.code_end.item(), 0x401000)
        self.assertEqual(first.layout.initial_entry.item(), 0x400100)
        self.assertEqual((batches[1].source, batches[1].first_sequence), (0, 0))
        relative = batches[1].addresses - first.layout.code_start
        self.assertEqual(relative.cpu().tolist(), [0x100])
        model = torch.nn.Embedding(512, 4, device=device)
        model(relative).sum().backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all().item())

    def test_cpu_owned_layout(self): self.check_owned('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_owned_layout(self): self.check_owned('mps')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unvalidated without hardware')
    def test_cuda_owned_layout(self): self.check_owned('cuda')

    def test_schema_can_precede_layout_and_entry_can_be_interpreter(self):
        data = (frame(1, detail=1 | FEATURE | (1 << 9)) + signal_frame(6, schema_row()) +
                layout(entry=0x800000) + signal_frame(7, register_row(bytes(8))) +
                frame(3, sequence=1) + frame(4))
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            self.assertEqual(list(pool.read())[0].layout.initial_entry.item(), 0x800000)

    def test_rejects_missing_duplicate_unadvertised_or_misattributed_layout(self):
        cases = [
            frame(1, detail=1) + layout(),
            frame(1, detail=1 | FEATURE) + layout() + layout(),
            frame(1, detail=1 | FEATURE) + frame(2, addresses=(1,)),
            frame(1, detail=1 | FEATURE) + frame(4),
            frame(1, detail=2 | FEATURE | (1 << 12)),
        ]
        for invalid in (layout(start=9, end=9), layout(source=1), layout(sequence=1)):
            cases.append(frame(1, detail=1 | FEATURE) + invalid)
        for data in cases:
            with self.subTest(data=data), worker(data) as endpoint, Pool([endpoint]) as pool:
                with self.assertRaises(ValueError): list(pool.read())

    def test_benchmark_counts_metadata_without_advancing_cpu(self):
        data = (frame(1, detail=1 | FEATURE) + layout() +
                frame(2, source=1, addresses=(8,)) + frame(3, source=1, sequence=1) + frame(4))
        counter = TraceCounter()
        counter.feed(data)
        counter.finish()
        self.assertEqual(counter.result()['events'], {'blocks': 1})
        self.assertEqual(counter.result()['sources'], 1)
        self.assertEqual(counter.result()['frames'], 5)
