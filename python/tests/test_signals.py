# SPDX-License-Identifier: AGPL-3.0-only
"""Signal frames exercise native validation and owned tensor columns together."""

import gc
import struct
import unittest
import weakref

import torch

from cpu2tensor import Pool
from test_consumer import frame, worker


MEMORY = 1 << 8
REGISTERS = 1 << 9
VALUES = 1 << 10
BASELINE = 1 << 8


def signal_frame(kind, payload, *, count=1, source=0, sequence=0):
    return struct.pack("<IHHIIQQ", 0x31543243, 2, kind, source, count,
                       sequence, len(payload)) + payload


def schema_row(register=0, width=8, name=b"x0"):
    return struct.pack("<II64s", register, width, name)


def register_row(value, *, pc=0x1000, register=0, flags=BASELINE | 1, width=None):
    return struct.pack("<QIHH", pc, register, len(value) if width is None else width,
                       flags) + value


def memory_row(*, pc=0x1000, address=0x2000, size=8, flags=0, value=None):
    return struct.pack("<QQII", pc, address, size, flags) + (value or b"")


def finish(sequence, *, source=0):
    return frame(3, source=source, sequence=sequence) + frame(4)


class SignalTests(unittest.TestCase):
    def test_mixed_signals_share_the_source_sequence(self):
        data = b"".join((
            frame(1, detail=1 | REGISTERS | MEMORY),
            signal_frame(6, schema_row()),
            signal_frame(7, register_row(bytes(8))),
            frame(2, sequence=1, addresses=(0x1000,)),
            signal_frame(8, memory_row(flags=1), sequence=2),
            signal_frame(7, register_row(b"\x05" + bytes(7), pc=0x1004, flags=2), sequence=3),
            finish(4),
        ))
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            batches = list(pool.read())
        self.assertEqual([batch.first_sequence for batch in batches], [0, 1, 2, 3])
        self.assertEqual([batch.addresses.tolist() for batch in batches], [[], [0x1000], [], []])
        self.assertEqual(batches[0].registers.names, {0: "x0"})
        self.assertEqual(batches[0].registers.flags.tolist(), [BASELINE | 1])
        self.assertEqual(batches[0].registers.values.tolist(), [[0] * 8])
        self.assertEqual(batches[3].registers.values.tolist(), [[5] + [0] * 7])
        self.assertEqual(batches[3].registers.flags.tolist(), [2])
        self.assertEqual(batches[2].memory.addresses.tolist(), [0x2000])
        self.assertEqual(batches[2].memory.sizes.tolist(), [8])
        self.assertEqual(batches[2].memory.flags.tolist(), [1])
        self.assertIsNone(batches[2].memory.values)
        self.assertIsNone(batches[1].registers)
        self.assertIsNone(batches[1].memory)
        with self.assertRaises(TypeError):
            batches[0].registers.names[0] = "changed"

    def test_source_schemas_are_separate_and_can_span_frames(self):
        data = b"".join((
            frame(1, detail=1 | REGISTERS),
            signal_frame(6, schema_row(name=b"x0")),
            signal_frame(6, schema_row(register=1, width=4, name=b"cpsr")),
            signal_frame(6, schema_row(name=b"other"), source=1),
            signal_frame(7, register_row(bytes(8))),
            signal_frame(7, register_row(bytes(4), register=1), sequence=1),
            signal_frame(7, register_row(bytes(8)), source=1),
            frame(3, sequence=2), frame(3, source=1, sequence=1), frame(4),
        ))
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            batches = list(pool.read())
        self.assertEqual(batches[0].registers.names, {0: "x0", 1: "cpsr"})
        self.assertIs(batches[0].registers.names, batches[1].registers.names)
        self.assertEqual(batches[2].registers.names, {0: "other"})

    def check_retained_columns(self, device):
        wide_value = bytes(range(32))
        transaction = bytes(range(240, 256))
        schema = schema_row(width=32, name=b"wide") + schema_row(register=1, width=4, name=b"narrow")
        rows = (register_row(wide_value, pc=2**64 - 1)
                + register_row(bytes.fromhex("89abcdef"), register=1))
        data = b"".join((
            frame(1, detail=1 | MEMORY | REGISTERS | VALUES),
            signal_frame(6, schema, count=2),
            signal_frame(7, rows, count=2),
            signal_frame(8, memory_row(pc=2**63, address=2**64 - 1, size=16,
                                       flags=3, value=transaction), sequence=2),
        ))
        for sequence in range(3, 103):
            data += signal_frame(7, register_row(bytes([sequence]) * 32, flags=1), sequence=sequence)
        data += finish(103)
        with worker(data, fragment=4096) as endpoint, Pool([endpoint], device=device) as pool:
            batches = pool.read()
            registers = next(batches).registers
            memory = next(batches).memory
            for batch in batches:
                self.assertEqual(batch.registers.ids.numel(), 1)
        del batches, pool, batch
        gc.collect()
        self.assertEqual(registers.pc.cpu().tolist(), [-1, 0x1000])
        self.assertEqual(registers.ids.cpu().tolist(), [0, 1])
        self.assertEqual(registers.widths.cpu().tolist(), [32, 4])
        self.assertEqual(registers.values.dtype, torch.uint8)
        self.assertEqual(registers.values.device.type, device)
        self.assertEqual(registers.values.cpu().tolist(),
                         [list(wide_value), list(bytes.fromhex("89abcdef")) + [0] * 28])
        self.assertEqual(memory.pc.cpu().tolist(), [-(2**63)])
        self.assertEqual(memory.addresses.cpu().tolist(), [-1])
        self.assertEqual(memory.values.cpu().tolist(), [list(transaction)])
        self.assertEqual(memory.values.device.type, device)
        # A consumer can use values on the device without lossy address conversion.
        model = torch.nn.Linear(16, 1, bias=False, device=device)
        result = model(memory.values.to(torch.float32) / 255)
        result.sum().backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all().item())

    def test_retained_cpu_columns(self):
        self.check_retained_columns("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable on this host")
    def test_retained_mps_columns(self):
        self.check_retained_columns("mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable; backend remains unverified")
    def test_retained_cuda_columns(self):
        self.check_retained_columns("cuda")

    def test_consumed_columns_are_released(self):
        data = b"".join((
            frame(1, detail=1 | MEMORY),
            signal_frame(8, memory_row()),
            frame(2, sequence=1, addresses=(8,)),
            finish(2),
        ))
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            batches = pool.read()
            first = next(batches)
            storage = weakref.ref(first.memory.addresses)
            del first
            next(batches)
            gc.collect()
            self.assertIsNone(storage(), "Pool kept a consumed signal tensor alive")
            self.assertEqual(list(batches), [])

    def test_register_validation(self):
        hello = frame(1, detail=1 | REGISTERS)
        schema = signal_frame(6, schema_row())
        good_row = register_row(bytes(8))
        cases = {
            "unknown register": register_row(bytes(8), register=1),
            "changed width": register_row(bytes(4)),
            "zero width": register_row(b""),
            "missing baseline": register_row(bytes(8), flags=1),
            "unknown checkpoint": register_row(bytes(8), flags=BASELINE | 4),
            "unknown flags": register_row(bytes(8), flags=BASELINE | 1 | (1 << 9)),
            "truncated value": good_row[:-1],
            "trailing bytes": good_row + b"\0",
        }
        for name, row in cases.items():
            data = hello + schema + signal_frame(7, row) + finish(1)
            with self.subTest(name=name), worker(data) as endpoint, Pool([endpoint]) as pool:
                with self.assertRaises(ValueError):
                    list(pool.read())
        # A baseline belongs to the first observation of that register only.
        data = (hello + schema + signal_frame(7, good_row)
                + signal_frame(7, good_row, sequence=1) + finish(2))
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaises(ValueError):
                list(pool.read())

    def test_schema_validation(self):
        hello = frame(1, detail=1 | REGISTERS)
        cases = {
            "empty name": schema_row(name=b""),
            "missing terminator": schema_row(name=b"x" * 64),
            "non-ascii name": schema_row(name=b"\xff"),
            "dirty padding": schema_row(name=b"x\0y"),
            "zero width": schema_row(width=0),
            "too wide": schema_row(width=257),
            "id outside limit": schema_row(register=512),
            "truncated row": schema_row()[:-1],
        }
        for name, row in cases.items():
            data = hello + signal_frame(6, row) + finish(0)
            with self.subTest(name=name), worker(data) as endpoint, Pool([endpoint]) as pool:
                with self.assertRaises(ValueError):
                    list(pool.read())
        for row in (schema_row(), schema_row(register=1)):
            data = hello + signal_frame(6, schema_row()) + signal_frame(6, row) + finish(0)
            with self.subTest(duplicate=row), worker(data) as endpoint, Pool([endpoint]) as pool:
                with self.assertRaises(ValueError):
                    list(pool.read())
        data = (hello + signal_frame(6, schema_row()) + frame(2, addresses=(1,))
                + signal_frame(6, schema_row(register=1, name=b"x1")) + finish(1))
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaises(ValueError):
                list(pool.read())

    def test_memory_validation(self):
        cases = {
            "zero size": (MEMORY, memory_row(size=0)),
            "non power of two": (MEMORY, memory_row(size=3)),
            "unknown flags": (MEMORY, memory_row(flags=4)),
            "truncated prefix": (MEMORY, memory_row()[:-1]),
            "unexpected values": (MEMORY, memory_row(value=bytes(16))),
            "missing values": (MEMORY | VALUES, memory_row()),
            "too wide for values": (MEMORY | VALUES, memory_row(size=32, value=bytes(16))),
            "nonzero value padding": (MEMORY | VALUES, memory_row(size=1, value=bytes([1]) * 16)),
        }
        for name, (features, row) in cases.items():
            data = frame(1, detail=1 | features) + signal_frame(8, row) + finish(1)
            with self.subTest(name=name), worker(data) as endpoint, Pool([endpoint]) as pool:
                with self.assertRaises(ValueError):
                    list(pool.read())

    def test_register_capture_requires_schema_before_any_data(self):
        for kind, event in (
            ("block", frame(2, addresses=(0x1000,))),
            ("memory", signal_frame(8, memory_row())),
        ):
            data = frame(1, detail=1 | REGISTERS | MEMORY) + event + finish(1)
            with self.subTest(kind=kind), worker(data) as endpoint, Pool([endpoint]) as pool:
                # Reject the event before yielding it, not just at SourceEnd.
                with self.assertRaises(ValueError):
                    next(pool.read())

    def test_source_end_requires_every_declared_register_baseline(self):
        schema = schema_row() + schema_row(register=1, name=b"x1")
        data = b"".join((
            frame(1, detail=1 | REGISTERS),
            signal_frame(6, schema, count=2),
            signal_frame(7, register_row(bytes(8))),
            finish(1),
        ))
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            batches = pool.read()
            self.assertEqual(next(batches).registers.ids.tolist(), [0])
            with self.assertRaises(ValueError):
                next(batches)

    def test_empty_sources_do_not_require_register_samples(self):
        for schema in (b"", signal_frame(6, schema_row())):
            data = frame(1, detail=1 | REGISTERS) + schema + finish(0)
            with self.subTest(has_schema=bool(schema)), worker(data) as endpoint, Pool([endpoint]) as pool:
                self.assertEqual(list(pool.read()), [])

    def test_features_and_signal_sequences_are_validated(self):
        hello = frame(1, detail=1 | REGISTERS | MEMORY)
        schema = signal_frame(6, schema_row())
        cases = {
            "unknown hello flag": frame(1, detail=1 | (1 << 11)),
            "values without memory": frame(1, detail=1 | VALUES),
            "disabled memory": frame(1, detail=1) + signal_frame(8, memory_row()),
            "disabled registers": frame(1, detail=1) + schema,
            "register data without schema": hello + signal_frame(7, register_row(bytes(8))),
            "signal gap": hello + schema + signal_frame(8, memory_row(), sequence=1),
            "schema sequence": hello + signal_frame(6, schema_row(), sequence=1),
            "source end ignores signals": (frame(1, detail=1 | MEMORY)
                                           + signal_frame(8, memory_row()) + finish(0)),
            "zero rows": hello + signal_frame(8, b"", count=0),
            "row count mismatch": hello + signal_frame(8, memory_row(), count=2),
        }
        for name, data in cases.items():
            with self.subTest(name=name), worker(data) as endpoint, Pool([endpoint]) as pool:
                with self.assertRaises(ValueError):
                    list(pool.read())

    @unittest.skipIf(torch.cuda.is_available(), "Requires a host without CUDA")
    def test_unavailable_cuda_reports_error(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            Pool(["tcp://localhost:1"], device="cuda")


if __name__ == "__main__":
    unittest.main()
