# SPDX-License-Identifier: AGPL-3.0-only
"""Replay actual captures of system_memory_guest.S through the public client."""
import os
from pathlib import Path
import unittest

from cpu2tensor import Pool
from cpu2tensor.examples.observe_context import cr3_for_blocks
from test_consumer import worker


@unittest.skipUnless(os.environ.get('CPU2TENSOR_STATE_CAPTURES'), 'Set the directory of actual system paging captures')
class SystemStateCaptureTests(unittest.TestCase):
    def check_profile(self, profile):
        data = (Path(os.environ['CPU2TENSOR_STATE_CAPTURES']) / (profile + '.trace')).read_bytes()
        contexts = {}
        registers = {}
        modes = set()
        observations = []
        raw_compatibility = False
        names = set()
        with worker(data, fragment=65536) as endpoint, Pool([endpoint]) as pool:
            for batch in pool.read():
                if batch.context is not None:
                    context = batch.context
                    contexts[batch.source] = (batch.first_sequence, context.cr0.item(), context.cr3.item(),
                        context.cr4.item(), context.efer.item(), context.mode.item())
                    self.assertEqual(context.known.item(), 63)
                    modes.add(context.mode.item())
                if batch.registers is not None:
                    rows = batch.registers
                    names.update(rows.names.values())
                    for register, width, value in zip(rows.ids.tolist(), rows.widths.tolist(), rows.values.tolist()):
                        registers[rows.names[register]] = int.from_bytes(bytes(value[:width]), 'little')
                if batch.addresses.numel() and contexts[batch.source][-1] == 32:
                    raw_compatibility |= registers.get('rax') == 0x1122334455667788 and registers.get('r8') == 0x8877665544332211
                if batch.memory is not None:
                    rows = batch.memory
                    for i, address in enumerate(rows.addresses.tolist()):
                        if address not in (0x8ffe, 0xaffe, 0xb030) or not 0x7c00 <= rows.pc[i].item() < 0x7e00:
                            continue
                        ctx = contexts[batch.source]
                        self.assertEqual(rows.context_sequences[i].item(), ctx[0])
                        value = int.from_bytes(bytes(rows.values[i, :rows.sizes[i]].tolist()), 'little')
                        observations.append((address, rows.sizes[i].item(), value, rows.physical_addresses[i].item(),
                            rows.mapped_sizes[i].item(), rows.mapping_flags[i].item(), ctx[1:5]))
        self.assertEqual(modes, {16, 32, 64})
        self.assertTrue(raw_compatibility)
        self.assertFalse(names & {'eflags', 'ftag', 'fiseg', 'fioff', 'foseg', 'fooff', 'fop'})
        if profile == 'selected': self.assertEqual(names, {'rax', 'r8', 'rip'})
        expected = [
            (0x8ffe, 4, 0x11223344, 0x20ffe, 2, 3, (0x80000011, 0x1000, 0, 0)),
            (0x8ffe, 4, 0x11223344, 0x20ffe, 2, 3, (0x80000011, 0x1000, 0, 0)),
            (0xaffe, 2, 0x3344, 0x20ffe, 2, 3, (0x80000011, 0x1000, 0, 0)),
            (0x8ffe, 4, 0x55667788, 0x40ffe, 2, 3, (0x80000011, 0x4000, 0, 0)),
            (0x8ffe, 4, 0x55667788, 0x40ffe, 2, 3, (0x80000011, 0x4000, 0, 0)),
            (0x8ffe, 4, 0x11223344, 0x20ffe, 2, 3, (0x80000011, 0x1000, 0, 0)),
            (0x8ffe, 4, 0x11227788, 0x40ffe, 2, 3, (0x80000011, 0x1000, 0, 0)),
            (0x8ffe, 4, 0x11227788, 0x40ffe, 2, 3, (0x80000011, 0x1000, 0x80, 0)),
            (0x8ffe, 4, 0x11227788, 0x40ffe, 2, 3, (0x80000011, 0x1000, 0x80, 0x800)),
            (0xb030, 4, 0x50014, 0xfee00030, 4, 7, (0x80000011, 0x1000, 0x80, 0x800)),
        ]
        self.assertEqual(observations, expected)

    def test_all_registers_and_mapped_memory(self): self.check_profile('all')
    def test_selected_registers_and_mapped_memory(self): self.check_profile('selected')

    def test_unmodified_public_memory_marks_unknown_fields(self):
        data = (Path(os.environ['CPU2TENSOR_STATE_CAPTURES']) / 'public-memory.trace').read_bytes()
        contexts = {}
        roots = set()
        found_apic = False
        with worker(data, fragment=65536) as endpoint, Pool([endpoint]) as pool:
            for batch in pool.read():
                self.assertIsNone(batch.registers)
                if batch.context is not None:
                    self.assertEqual(batch.context.known.tolist(), [15])
                    self.assertEqual(batch.context.mode.tolist(), [0])
                    contexts[batch.source] = batch.first_sequence
                    roots.add(batch.context.cr3.item())
                if batch.memory is not None:
                    rows = batch.memory
                    self.assertTrue(bool((rows.context_sequences == contexts[batch.source]).all()))
                    self.assertFalse(bool((rows.mapping_flags & 2).any()))
                    selected = (rows.addresses == 0xb030) & (rows.pc >= 0x7c00) & (rows.pc < 0x7e00)
                    if bool(selected.any()):
                        found_apic = True
                        self.assertEqual(rows.physical_addresses[selected].tolist(), [0xfee00030])
                        self.assertEqual(rows.mapping_flags[selected].tolist(), [1])
        self.assertTrue(found_apic)
        self.assertTrue({0x1000, 0x4000} <= roots)

    def check_context_only(self, name, known):
        data = (Path(os.environ['CPU2TENSOR_STATE_CAPTURES']) / (name + '.trace')).read_bytes()
        roots = []
        modes = set()
        current = None
        block_roots = set()
        count = 0
        state = None
        with worker(data, fragment=65536) as endpoint, Pool([endpoint]) as pool:
            for batch in pool.read():
                labels, state = cr3_for_blocks(batch, state)
                self.assertIsNone(batch.registers)
                self.assertIsNone(batch.memory)
                if batch.context is not None:
                    table = batch.context
                    self.assertEqual(table.known.tolist(), [known])
                    current = table.cr3.item()
                    roots.append(current)
                    if known & 32:
                        modes.add(table.mode.item())
                if batch.addresses.numel():
                    self.assertIsNotNone(current)
                    self.assertEqual(set(labels.tolist()), {current})
                    block_roots.add(current)
                    count += batch.addresses.numel()
        self.assertGreater(count, 0)
        # This independent boot fixture installs root0x1000, switches to0x4000,
        # and returns to0x1000. Each root must govern following block entries.
        self.assertTrue({0x1000, 0x4000} <= block_roots)
        second = roots.index(0x4000)
        self.assertIn(0x1000, roots[:second])
        self.assertIn(0x1000, roots[second + 1:])
        self.assertEqual(modes, {16, 32, 64} if known == 63 else set())

    def test_context_only_paging_oracle(self):
        self.check_context_only('context-only', 63)

    def test_public_context_only_marks_unknown_mode(self):
        self.check_context_only('public-context-only', 15)

    def test_missing_hook_reports_dependency(self):
        directory = Path(os.environ['CPU2TENSOR_STATE_CAPTURES'])
        self.assertEqual((directory / 'missing-hook.trace').read_bytes(), b'')
        self.assertIn('exact x86 system registers require', (directory / 'missing-hook.stderr').read_text())
