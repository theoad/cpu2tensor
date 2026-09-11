# SPDX-License-Identifier: AGPL-3.0-only
"""Packed uploads preserve raw bits, alignment, metadata and retained views."""

from dataclasses import fields, replace
import gc
from types import MappingProxyType
import unittest
from unittest import mock

import torch

from cpu2tensor import (
    AddressContext, Batch, BlockTransitions, ExecutableLayout, MemoryAccesses,
    RegisterChanges, TransitionWindow,
)
from cpu2tensor._device import to_device


def columns(batch):
    result = [("addresses", batch.addresses)]
    if batch.block_sequences is not None:
        result.append(("block_sequences", batch.block_sequences))
    for name in ("registers", "memory", "context", "layout", "transitions"):
        table = getattr(batch, name)
        if table is not None:
            result.extend((f"{name}.{field.name}", value) for field in fields(table)
                          if isinstance(value := getattr(table, field.name), torch.Tensor))
    return result


def rich_batch():
    addresses = torch.tensor([-(1 << 63), -1, 0, (1 << 63) - 1], dtype=torch.int64)
    registers = RegisterChanges(
        pc=addresses[:3], ids=torch.arange(3), widths=torch.tensor([10, 8, 1]),
        flags=torch.tensor([256, 1, 1]),
        # Thirty strided bytes force padding before the following int64 column.
        values=torch.arange(60, dtype=torch.uint8).reshape(3, 20)[:, ::2],
        names=MappingProxyType({0: "st0", 1: "rax", 2: "small"}), sequences=torch.tensor([1, 2, 3]),
    )
    memory = MemoryAccesses(
        pc=addresses[:3], addresses=addresses[1:], sizes=torch.tensor([1, 4, 16]),
        flags=torch.tensor([0, 1, 3]), values=torch.arange(48, dtype=torch.uint8).reshape(3, 16),
        physical_addresses=addresses[:3], mapped_sizes=torch.tensor([1, 2, 16]),
        mapping_flags=torch.tensor([1, 3, 7]), context_sequences=torch.tensor([0, 0, 6]),
        sequences=torch.tensor([5, 7, 8]),
    )
    context = AddressContext(
        pc=addresses[:2], cr0=torch.tensor([1, 1]), cr3=torch.tensor([0x1000, 0x4000]),
        cr4=torch.tensor([0, 0]), efer=torch.tensor([0, 0]), cs_base=torch.tensor([0, 0]),
        mode=torch.tensor([32, 64]), known=torch.tensor([63, 63]), sequences=torch.tensor([0, 6]),
    )
    transitions = BlockTransitions(
        9, from_addresses=addresses[:3], destinations=addresses[1:],
        counts=torch.tensor([11, 12, 13]),
    )
    return Batch(3, 0, addresses, registers, memory, context, worker=4,
                 block_sequences=torch.tensor([4, 9, 10, 11]), transitions=transitions)


class DeviceTransferTests(unittest.TestCase):
    def test_cpu_keeps_the_original_columns_without_staging(self):
        batch = rich_batch()
        with mock.patch("cpu2tensor._device.torch.zeros", side_effect=AssertionError("CPU staging")), \
                mock.patch.object(torch.Tensor, "to", side_effect=AssertionError("CPU copy")):
            moved = to_device(batch, torch.device("cpu"))
        self.assertIs(moved, batch)
        self.assertIs(moved.registers.values, batch.registers.values)
        self.assertFalse(moved.registers.values.is_contiguous())

    def test_packed_upload_control_flow_without_accelerator_hardware(self):
        """Check packing in CPU memory; this does not validate an accelerator."""
        uploads = []

        def transfer(tensor, device, *, non_blocking):
            uploads.append((device.type, non_blocking))
            return tensor.clone()

        with mock.patch.object(torch.Tensor, "to", transfer), \
                mock.patch("cpu2tensor._device.torch.mps.synchronize") as synchronize:
            moved = to_device(rich_batch(), torch.device("mps"))
        self.assertEqual(uploads, [("mps", False)])
        synchronize.assert_called_once_with()
        self.assertEqual(
            {name: value.tolist() for name, value in columns(moved)},
            {name: value.tolist() for name, value in columns(rich_batch())},
        )

    def test_block_upload_control_flow_without_accelerator_hardware(self):
        batch = Batch(0, 0, torch.tensor([1, 2, 3], dtype=torch.int64))
        with mock.patch.object(torch.Tensor, "to", return_value=batch.addresses.clone()), \
                mock.patch("cpu2tensor._device.torch.mps.synchronize") as synchronize:
            moved = to_device(batch, torch.device("mps"))
        self.assertTrue(torch.equal(moved.addresses, batch.addresses))
        synchronize.assert_called_once_with()

    def test_upload_rejects_non_cpu_input_columns(self):
        block = Batch(0, 0, torch.empty(1, dtype=torch.int64, device="meta"))
        with self.assertRaisesRegex(ValueError, "requires CPU input"):
            to_device(block, torch.device("mps"))
        rich = replace(
            rich_batch(),
            block_sequences=torch.empty(4, dtype=torch.int64, device="meta"),
        )
        with self.assertRaisesRegex(ValueError, "requires CPU input"):
            to_device(rich, torch.device("mps"))

    def check_all_columns(self, device):
        batch = rich_batch()
        expected = {name: value.clone() for name, value in columns(batch)}
        moved = to_device(batch, torch.device(device))
        self.assertEqual((moved.worker, moved.source, moved.first_sequence), (4, 3, 0))
        self.assertIs(moved.registers.names, batch.registers.names)
        self.assertIsNone(moved.layout)
        storage = moved.addresses.untyped_storage().data_ptr()
        for name, value in columns(moved):
            with self.subTest(column=name):
                self.assertEqual(value.device.type, device)
                self.assertEqual(value.dtype, expected[name].dtype)
                self.assertEqual(value.shape, expected[name].shape)
                self.assertTrue(torch.equal(value.cpu(), expected[name]))
                self.assertEqual(value.untyped_storage().data_ptr(), storage)
                self.assertEqual(value.data_ptr() % value.element_size(), 0)
                self.assertFalse(value.requires_grad)
        model = torch.nn.Linear(16, 1, device=device)
        model(moved.memory.values.float()).sum().backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all().item())

        held = moved.registers.values
        batch.registers.values.zero_()
        del moved, batch
        gc.collect()
        for _ in range(8):
            newer = to_device(rich_batch(), torch.device(device))
            self.assertNotEqual(newer.registers.values.untyped_storage().data_ptr(), held.untyped_storage().data_ptr())
        self.assertTrue(torch.equal(held.cpu(), expected["registers.values"]))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_all_mps_columns_and_retained_10_byte_register_values(self):
        self.check_all_columns("mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_all_cuda_columns_and_retained_10_byte_register_values(self):
        self.check_all_columns("cuda")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_uses_one_upload_and_one_synchronization(self):
        original = torch.Tensor.to
        uploads = []

        def transfer(tensor, *arguments, **keywords):
            uploads.append((tensor.dtype, tensor.device.type, arguments, keywords))
            return original(tensor, *arguments, **keywords)

        with mock.patch.object(torch.Tensor, "to", transfer), \
                mock.patch("cpu2tensor._device.torch.mps.synchronize", wraps=torch.mps.synchronize) as sync:
            moved = to_device(rich_batch(), torch.device("mps"))
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0][:2], (torch.uint8, "cpu"))
        self.assertEqual(uploads[0][2], (torch.device("mps"),))
        self.assertEqual(uploads[0][3], {"non_blocking": False})
        sync.assert_called_once_with()
        self.assertEqual(moved.memory.values.shape, (3, 16))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_empty_columns_layout_and_absent_values_on_mps(self):
        empty = Batch(0, 0, torch.empty(0, dtype=torch.int64))
        moved = to_device(empty, torch.device("mps"))
        self.assertEqual((moved.addresses.device.type, moved.addresses.numel()), ("mps", 0))
        self.assertIsNone(moved.block_sequences)
        summary = Batch(
            None, None, torch.empty(0, dtype=torch.int64),
            transition_window=TransitionWindow(1, "ended", 2, 8, 0, 0, 0),
        )
        moved_summary = to_device(summary, torch.device("mps"))
        self.assertEqual(moved_summary.addresses.device.type, "mps")
        self.assertEqual(moved_summary.transition_window, summary.transition_window)
        layout = Batch(None, None, empty.addresses,
                       layout=ExecutableLayout(torch.tensor([0x400000, 0x401000, 0x400010])), worker=2)
        moved = to_device(layout, torch.device("mps"))
        self.assertEqual((moved.worker, moved.source, moved.first_sequence), (2, None, None))
        self.assertTrue(torch.equal(moved.layout.values.cpu(), layout.layout.values))
        batch = rich_batch()
        batch = replace(batch, memory=replace(batch.memory, values=None))
        self.assertIsNone(to_device(batch, torch.device("mps")).memory.values)


if __name__ == "__main__":
    unittest.main()
