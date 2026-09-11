# SPDX-License-Identifier: AGPL-3.0-only
"""CPU collation preserves source progress while bounding pending decoded bytes."""

import gc
import struct
import threading
import unittest
from unittest import mock
import weakref
from types import SimpleNamespace

import torch

from cpu2tensor import Batch, ExecutableLayout, Pool
from cpu2tensor._batching import (
    BatchCollator, MAX_BATCH_BYTES, merge_batches, optional_column, tensor_bytes,
)
from test_address_context import FEATURES, access, context
from test_consumer import frame, worker
from test_mixed import run
from test_multiworker import worker as controlled_worker
from test_signals import MEMORY, REGISTERS, VALUES, memory_row, register_row, schema_row, signal_frame


def block_batch(source, sequence, *, count=1, worker_id=0):
    return Batch(source, sequence, torch.arange(sequence, sequence + count), worker=worker_id)


def rich_trace():
    baseline = signal_frame(7, register_row(bytes(range(10))), sequence=0)
    other = signal_frame(7, register_row(bytes(range(16)), register=1), sequence=1)
    mixed = run(2, struct.pack("<2Q", 0x1000, 0x1004), 2)
    mixed += run(8, memory_row(size=4, value=b"\x01\x02\x03\x04" + bytes(12)))
    mixed += run(7, register_row(bytes([9] * 10), flags=1))
    return (frame(1, detail=1 | REGISTERS | MEMORY | VALUES | (1 << 18))
            + signal_frame(6, schema_row(width=10, name=b"wide") + schema_row(register=1, width=16, name=b"vector"), count=2)
            + baseline + other + signal_frame(14, mixed, count=4, sequence=2)
            + frame(2, sequence=6, addresses=(0x1008,))
            + frame(3, sequence=7) + frame(4))


class BatchingTests(unittest.TestCase):
    def test_invalid_collation_inputs_are_rejected_before_retention(self):
        for size in (0, MAX_BATCH_BYTES + 1):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "between 1 and 4 MiB"):
                BatchCollator(size)
        collator = BatchCollator(64)
        with self.assertRaisesRegex(ValueError, "source and sequence"):
            collator.add(Batch(None, None, torch.tensor([1])))
        with self.assertRaisesRegex(ValueError, "inconsistent optional column"):
            optional_column(
                [SimpleNamespace(values=None), SimpleNamespace(values=torch.tensor([1]))],
                "values",
            )

    def test_budget_flushes_oldest_source_and_handles_one_oversized_frame(self):
        collator = BatchCollator(32)
        for source in range(7):
            self.assertEqual(collator.add(block_batch(source, 0)), [])
        flushed = collator.add(block_batch(7, 0))
        self.assertEqual([batch.source for batch in flushed], [0])
        self.assertEqual(collator.pending_bytes, 56)
        oversized = block_batch(8, 0, count=20)
        self.assertGreater(tensor_bytes(oversized), 2 * collator.target_bytes)
        self.assertEqual([batch.source for batch in collator.add(oversized)], [8])
        self.assertEqual(collator.pending_bytes, 56)
        self.assertEqual([batch.source for batch in collator.finish()], list(range(1, 8)))
        self.assertEqual(collator.pending_bytes, 0)

    def test_source_target_is_separate_from_other_workers_and_sources(self):
        collator = BatchCollator(16)
        self.assertEqual(collator.add(block_batch(0, 0, worker_id=0)), [])
        self.assertEqual(collator.add(block_batch(0, 0, worker_id=1)), [])
        flushed = collator.add(block_batch(0, 1, worker_id=0))
        self.assertEqual([(batch.worker, batch.source) for batch in flushed], [(0, 0)])
        self.assertEqual(flushed[0].block_sequences.tolist(), [0, 1])
        self.assertEqual(collator.end_source(0, worker=1)[0].worker, 1)
        self.assertEqual(collator.finish(), [])
        with self.assertRaisesRegex(ValueError, "different workers"):
            merge_batches([block_batch(0, 0), block_batch(1, 1)])

    def test_tiny_frames_flush_one_source_before_python_objects_accumulate(self):
        collator = BatchCollator(MAX_BATCH_BYTES)
        for sequence in range(255):
            self.assertEqual(collator.add(block_batch(0, sequence)), [])
        self.assertEqual(collator.pending_frames, 255)
        first = collator.add(block_batch(0, 255))[0]
        self.assertEqual(first.addresses.tolist(), list(range(256)))
        self.assertEqual(first.block_sequences.tolist(), list(range(256)))
        self.assertEqual((collator.pending_frames, collator.pending_bytes), (0, 0))
        for sequence in range(256, 512):
            ready = collator.add(block_batch(0, sequence))
        self.assertEqual(ready[0].block_sequences.tolist(), list(range(256, 512)))
        self.assertEqual(first.addresses.tolist(), list(range(256)))
        self.assertEqual((collator.pending_frames, collator.pending_bytes), (0, 0))
        self.assertEqual(collator.finish(), [])

    def test_total_tiny_frames_flush_oldest_source_at_the_worker_cap(self):
        collator = BatchCollator(MAX_BATCH_BYTES)
        for index in range(511):
            self.assertEqual(collator.add(block_batch(index % 4, index // 4)), [])
        self.assertEqual(collator.pending_frames, 511)
        flushed = collator.add(block_batch(3, 127))
        self.assertEqual([batch.source for batch in flushed], [0])
        self.assertEqual(flushed[0].block_sequences.tolist(), list(range(128)))
        self.assertEqual((collator.pending_frames, collator.pending_bytes), (384, 384 * 8))
        ended = collator.end_source(2)
        self.assertEqual(ended[0].block_sequences.tolist(), list(range(128)))
        self.assertEqual(collator.pending_frames, 256)
        tails = collator.finish()
        self.assertEqual([batch.source for batch in tails], [1, 3])
        self.assertTrue(all(batch.block_sequences.tolist() == list(range(128)) for batch in tails))
        self.assertEqual((collator.pending_frames, collator.pending_bytes), (0, 0))
        self.assertEqual(flushed[0].addresses.tolist(), list(range(128)))

    def test_zero_row_cpu_batch_is_rejected_without_pending_objects(self):
        collator = BatchCollator(MAX_BATCH_BYTES)
        with self.assertRaisesRegex(ValueError, "empty CPU batch"):
            collator.add(Batch(0, 0, torch.empty(0, dtype=torch.int64)))
        self.assertEqual((collator.pending_frames, collator.pending_bytes), (0, 0))
        self.assertEqual(collator.finish(), [])

    def test_layout_is_immediate_and_source_tail_does_not_wait_for_an_idle_source(self):
        collator = BatchCollator(64)
        collator.add(block_batch(1, 0))
        layout = Batch(None, None, torch.empty(0, dtype=torch.int64),
                       layout=ExecutableLayout(torch.tensor([1, 2, 1])))
        self.assertEqual(collator.add(layout), [layout])
        self.assertEqual(collator.pending_bytes, 8)
        collator.add(block_batch(2, 0))
        self.assertEqual(collator.end_source(2)[0].source, 2)
        self.assertEqual(collator.pending_bytes, 8)
        self.assertEqual(collator.finish()[0].source, 1)

    def check_rich(self, device):
        with worker(rich_trace(), fragment=7) as endpoint, Pool([endpoint], device=device, batch_bytes=65536) as pool:
            batches = list(pool.read())
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(batch.first_sequence, 0)
        self.assertEqual(batch.addresses.cpu().tolist(), [0x1000, 0x1004, 0x1008])
        self.assertEqual(batch.block_sequences.cpu().tolist(), [2, 3, 6])
        self.assertEqual(batch.registers.sequences.cpu().tolist(), [0, 1, 5])
        self.assertEqual(batch.memory.sequences.cpu().tolist(), [4])
        self.assertEqual(batch.registers.widths.cpu().tolist(), [10, 16, 10])
        self.assertEqual(batch.registers.names, {0: "wide", 1: "vector"})
        self.assertEqual(batch.registers.values.cpu().tolist(), [list(range(10)) + [0] * 6, list(range(16)), [9] * 10 + [0] * 6])
        self.assertEqual(batch.memory.values.cpu().tolist(), [[1, 2, 3, 4] + [0] * 12])
        self.assertEqual(batch.memory.values.device.type, device)
        retained = batch.registers.values
        del batches, batch, pool
        gc.collect()
        model = torch.nn.Linear(16, 1, device=device)
        model(retained.float()).sum().backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all().item())

    def test_cpu_mixed_and_legacy_sequences_with_variable_register_widths(self):
        self.check_rich("cpu")

    def test_system_context_and_physical_references_keep_original_sequences(self):
        data = (frame(1, detail=FEATURES | REGISTERS | VALUES)
                + signal_frame(6, schema_row(name=b"rax"))
                + signal_frame(12, context())
                + signal_frame(7, register_row(bytes(8)), sequence=1)
                + signal_frame(8, access(values=True), sequence=2)
                + signal_frame(12, context(0x4000), sequence=3)
                + signal_frame(8, access(sequence=3, physical=0x40ffe, values=True), sequence=4)
                + frame(2, sequence=5, addresses=(0x7c00,))
                + frame(3, sequence=6) + frame(4))
        with worker(data) as endpoint, Pool([endpoint], batch_bytes=65536) as pool:
            batches = list(pool.read())
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(batch.context.sequences.tolist(), [0, 3])
        self.assertEqual(batch.context.cr3.tolist(), [0x1000, 0x4000])
        self.assertEqual(batch.memory.sequences.tolist(), [2, 4])
        self.assertEqual(batch.memory.context_sequences.tolist(), [0, 3])
        self.assertEqual(batch.memory.physical_addresses.tolist(), [0x20ffe, 0x40ffe])
        self.assertEqual(batch.memory.mapped_sizes.tolist(), [2, 2])
        self.assertEqual(batch.registers.sequences.tolist(), [1])
        self.assertEqual(batch.block_sequences.tolist(), [5])

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_mixed_and_legacy_sequences_with_variable_register_widths(self):
        self.check_rich("mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_mixed_and_legacy_sequences_with_variable_register_widths(self):
        self.check_rich("cuda")

    def test_source_end_flushes_before_connection_completion(self):
        release = threading.Event()

        def serve(connection):
            connection.sendall(frame(1, detail=1) + frame(2, source=1, addresses=(91,))
                               + frame(2, source=2, addresses=(92,)) + frame(3, source=1, sequence=1))
            if not release.wait(5):
                raise AssertionError("Ended source was not delivered before Complete")
            connection.sendall(frame(3, source=2, sequence=1) + frame(4))

        with controlled_worker(serve) as endpoint, Pool([endpoint], batch_bytes=65536) as pool:
            try:
                batches = pool.read()
                first = next(batches)
                self.assertEqual((first.source, first.addresses.item()), (1, 91))
                release.set()
                rest = list(batches)
                self.assertEqual([(batch.source, batch.addresses.item()) for batch in rest], [(2, 92)])
            finally:
                release.set()

    def test_nonzero_exit_follows_a_flushed_tail(self):
        data = frame(1, detail=1) + frame(2, addresses=(77,)) + frame(3, sequence=1) + frame(4, detail=7)
        with worker(data) as endpoint, Pool([endpoint], batch_bytes=65536) as pool:
            batches = pool.read()
            self.assertEqual(next(batches).addresses.tolist(), [77])
            with self.assertRaisesRegex(RuntimeError, "exited with code 7"):
                next(batches)

    def test_incomplete_source_cannot_turn_pending_data_into_success(self):
        for ending in (b"", frame(5, detail=1)):
            data = frame(1, detail=1) + frame(2, addresses=(1,)) + ending
            with self.subTest(ending=ending), worker(data) as endpoint, Pool([endpoint], batch_bytes=65536) as pool:
                with self.assertRaisesRegex(RuntimeError, "Incomplete trace"):
                    list(pool.read())

    def check_public_multiworker(self, device):
        with worker(rich_trace()) as first, worker(rich_trace()) as second, \
                Pool([first, second], device=device, batch_bytes=65536) as pool:
            batches = sorted(pool.read(), key=lambda batch: batch.worker)
        self.assertEqual([(batch.worker, batch.source) for batch in batches], [(0, 0), (1, 0)])
        for batch in batches:
            self.assertEqual(batch.block_sequences.cpu().tolist(), [2, 3, 6])
            self.assertEqual(batch.registers.sequences.cpu().tolist(), [0, 1, 5])
            self.assertEqual(batch.memory.values.device.type, device)

    def test_public_multiworker_cpu_collation(self):
        self.check_public_multiworker("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_public_multiworker_mps_collation(self):
        self.check_public_multiworker("mps")

    def test_configuration_is_rejected_before_connections(self):
        with mock.patch("cpu2tensor.pool.socket.socket", side_effect=AssertionError("Unexpected connection")):
            for value in (-1, MAX_BATCH_BYTES + 1, 1.5, None):
                for endpoints in (["tcp://127.0.0.1:1"], ["tcp://127.0.0.1:1", "tcp://127.0.0.1:2"]):
                    with self.subTest(value=value, endpoints=endpoints), self.assertRaisesRegex(ValueError, "batch_bytes"):
                        Pool(endpoints, batch_bytes=value)

    def test_flushed_cpu_storage_does_not_retain_input_batches(self):
        collator = BatchCollator(16)
        first = block_batch(0, 0)
        storage = weakref.ref(first.addresses)
        collator.add(first)
        del first
        merged = collator.add(block_batch(0, 1))[0]
        gc.collect()
        self.assertIsNone(storage())
        self.assertEqual(merged.addresses.tolist(), [0, 1])
        collator.add(block_batch(0, 2))
        collator.add(block_batch(0, 3))
        self.assertEqual(merged.addresses.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
