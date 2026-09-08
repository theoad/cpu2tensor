# SPDX-License-Identifier: AGPL-3.0-only
"""Real TCP framing with controlled readiness and bounded reader queues."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
import socket
import struct
import threading
import unittest

import torch

from cpu2tensor import Pool
from cpu2tensor._multipool import EndpointReaders
from test_address_context import FEATURES, access, context
from test_executable_layout import FEATURE, layout
from test_signals import MEMORY, REGISTERS, VALUES, memory_row, register_row, schema_row, signal_frame
from test_mixed import capture as mixed_capture


def frame(kind: int, *, source: int = 0, sequence: int = 0,
          detail: int = 0, addresses: tuple[int, ...] = ()) -> bytes:
    return (struct.pack("<IHHIIQQ", 0x31543243, 2, kind, source,
                        len(addresses), sequence, detail)
            + struct.pack(f"<{len(addresses)}Q", *addresses))


def trace(address: int, *, source: int = 0, count: int = 1) -> bytes:
    return (frame(1, detail=1)
            + b"".join(frame(2, source=source, sequence=index,
                             addresses=(address + index,)) for index in range(count))
            + frame(3, source=source, sequence=count) + frame(4))


@contextmanager
def worker(serve: Callable[[socket.socket], None]) -> Iterator[str]:
    errors: list[BaseException] = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)

        def run() -> None:
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(5)
                    serve(connection)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            yield f"tcp://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            thread.join(6)
            if thread.is_alive():
                raise AssertionError("Fixture worker did not stop")
            if errors:
                raise AssertionError("Fixture worker failed") from errors[0]


class MultiworkerTests(unittest.TestCase):
    def test_ready_worker_does_not_wait_for_first_endpoint(self) -> None:
        slow_connected = threading.Event()
        release_slow = threading.Event()

        def slow(connection: socket.socket) -> None:
            slow_connected.set()
            if not release_slow.wait(5):
                raise AssertionError("Slow worker was never released")
            connection.sendall(trace(100, source=3))

        with worker(slow) as first, worker(lambda sock: sock.sendall(trace(200))) as second:
            readers = EndpointReaders([first, second], "cpu", 5)
            batches = readers.read()
            try:
                ready = next(batches)
                self.assertTrue(slow_connected.wait(5))
                self.assertEqual((ready.worker, ready.source, ready.first_sequence), (1, 0, 0))
                self.assertEqual(ready.addresses.tolist(), [200])
                release_slow.set()
                rest = list(batches)
                self.assertEqual([(batch.worker, batch.source) for batch in rest], [(0, 3)])
            finally:
                release_slow.set()
                readers.close()

    def test_empty_worker_does_not_end_other_workers(self) -> None:
        empty = frame(1, detail=1) + frame(3) + frame(4)
        with worker(lambda sock: sock.sendall(empty)) as first, \
                worker(lambda sock: sock.sendall(trace(30, count=4))) as second:
            readers = EndpointReaders([first, second], "cpu", 5)
            batches = list(readers.read())
        self.assertEqual([batch.worker for batch in batches], [1] * 4)
        self.assertEqual([batch.first_sequence for batch in batches], [0, 1, 2, 3])
        self.assertTrue(all(not thread.is_alive() for thread in readers._threads))

    def check_retained_batches(self, device: str) -> None:
        with worker(lambda sock: sock.sendall(trace(40, source=2, count=12))) as first, \
                worker(lambda sock: sock.sendall(trace(80, source=2, count=12))) as second:
            with Pool([first, second], device=device, timeout=5) as pool:
                retained = list(pool.read())
        for identity, base in ((0, 40), (1, 80)):
            batches = [batch for batch in retained if batch.worker == identity]
            self.assertEqual([batch.source for batch in batches], [2] * 12)
            self.assertEqual([batch.first_sequence for batch in batches], list(range(12)))
            self.assertEqual([batch.addresses.item() for batch in batches], list(range(base, base + 12)))
            self.assertTrue(all(batch.addresses.device.type == device for batch in batches))
        features = torch.cat([batch.addresses for batch in retained]).to(torch.float32)
        weight = torch.tensor(0.5, device=device, requires_grad=True)
        (weight * features).mean().backward()
        self.assertTrue(torch.isfinite(weight.grad).item())

    def test_retained_batches_keep_worker_source_sequence_and_storage(self) -> None:
        self.check_retained_batches("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable on this host")
    def test_retained_batches_on_mps(self) -> None:
        self.check_retained_batches("mps")

    def check_nested_columns(self, device: str) -> None:
        system = (frame(1, detail=FEATURES | REGISTERS | VALUES)
                  + signal_frame(6, schema_row(name=b"rax"))
                  + signal_frame(12, context())
                  + signal_frame(7, register_row(bytes(8)), sequence=1)
                  + signal_frame(8, access(values=True), sequence=2)
                  + frame(3, sequence=3) + frame(4))
        process = (frame(1, detail=1 | FEATURE | REGISTERS | MEMORY | VALUES)
                   + signal_frame(6, schema_row(name=b"x0")) + layout()
                   + signal_frame(7, register_row(bytes(8)))
                   + signal_frame(8, memory_row(value=bytes(16)), sequence=1)
                   + frame(3, sequence=2) + frame(4))
        with worker(lambda sock: sock.sendall(system)) as first, \
                worker(lambda sock: sock.sendall(process)) as second:
            with Pool([first, second], device=device, timeout=5) as pool:
                batches = list(pool.read())
        seen = set()
        for batch in batches:
            self.assertEqual(batch.addresses.device.type, device)
            for name in ("registers", "memory", "context", "layout"):
                columns = getattr(batch, name)
                if columns is not None:
                    seen.add(name)
                    for field in fields(columns):
                        value = getattr(columns, field.name)
                        if isinstance(value, torch.Tensor):
                            self.assertEqual(value.device.type, device)
        self.assertEqual(seen, {"registers", "memory", "context", "layout"})
        system_batches = [batch for batch in batches if batch.worker == 0]
        self.assertEqual(system_batches[0].context.cr3.item(), 0x1000)
        self.assertEqual(system_batches[1].registers.names, {0: "rax"})
        self.assertEqual(system_batches[2].memory.context_sequences.item(), 0)
        self.assertEqual(system_batches[2].memory.physical_addresses.item(), 0x20ffe)
        metadata = next(batch for batch in batches if batch.layout is not None)
        self.assertEqual((metadata.worker, metadata.source, metadata.first_sequence), (1, None, None))
        self.assertEqual(metadata.layout.code_start.item(), 0x400000)

    def test_nested_columns_and_worker_metadata_on_cpu(self) -> None:
        self.check_nested_columns("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable on this host")
    def test_nested_columns_and_worker_metadata_on_mps(self) -> None:
        self.check_nested_columns("mps")

    def check_public_mixed_batches(self, device: str) -> None:
        with worker(lambda sock: sock.sendall(mixed_capture())) as first, \
                worker(lambda sock: sock.sendall(mixed_capture())) as second:
            with Pool([first, second], device=device, timeout=5) as pool:
                batches = sorted(pool.read(), key=lambda batch: batch.worker)
        self.assertEqual([(batch.worker, batch.source) for batch in batches], [(0, 0), (1, 0)])
        for batch in batches:
            self.assertEqual(batch.block_sequences.cpu().tolist(), [2, 6, 7])
            self.assertEqual(batch.registers.sequences.cpu().tolist(), [1, 5])
            self.assertEqual(batch.memory.sequences.cpu().tolist(), [3, 8])
            self.assertEqual(batch.context.sequences.cpu().tolist(), [0, 4])
            self.assertEqual(batch.memory.context_sequences.cpu().tolist(), [0, 4])
            self.assertEqual(batch.context.cr3.cpu().tolist(), [0x1000, 0x4000])
            self.assertEqual(batch.block_sequences.device.type, device)
            for name in ("registers", "memory", "context"):
                columns = getattr(batch, name)
                for field in fields(columns):
                    value = getattr(columns, field.name)
                    if isinstance(value, torch.Tensor):
                        self.assertEqual(value.device.type, device)
        values = torch.cat([batch.memory.values for batch in batches]).float()
        model = torch.nn.Linear(16, 1, device=device)
        model(values).sum().backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all().item())

    def test_public_pool_mixed_sequences_and_retention_on_cpu(self) -> None:
        self.check_public_mixed_batches("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable on this host")
    def test_public_pool_mixed_sequences_and_retention_on_mps(self) -> None:
        self.check_public_mixed_batches("mps")

    def test_failure_is_reported_even_when_its_batch_slot_is_full(self) -> None:
        release_failure = threading.Event()

        def failing(connection: socket.socket) -> None:
            if not release_failure.wait(5):
                raise AssertionError("Failure worker was never released")
            connection.sendall(frame(1, detail=1)
                               + frame(2, addresses=(9,)) + frame(5, detail=1))

        with worker(lambda sock: sock.sendall(trace(1, count=20))) as first, \
                worker(failing) as second:
            readers = EndpointReaders([first, second], "cpu", 5)
            batches = readers.read()
            try:
                self.assertEqual(next(batches).worker, 0)
                release_failure.set()
                with readers._ready:
                    self.assertTrue(readers._ready.wait_for(lambda: readers._failure is not None, 5))
                    self.assertIsNotNone(readers._batches[1])
                with self.assertRaisesRegex(RuntimeError, "Worker 1 .*capture failed") as raised:
                    next(batches)
                self.assertIsInstance(raised.exception.__cause__, RuntimeError)
                self.assertTrue(all(not thread.is_alive() for thread in readers._threads))
            finally:
                release_failure.set()
                readers.close()

    def test_early_close_wakes_full_queue_and_blocked_socket(self) -> None:
        quiet_connected = threading.Event()
        quiet_closed = threading.Event()

        def quiet(connection: socket.socket) -> None:
            connection.sendall(frame(1, detail=1))
            quiet_connected.set()
            if connection.recv(1) == b"":
                quiet_closed.set()

        with worker(lambda sock: sock.sendall(trace(1, count=20))) as first, worker(quiet) as second:
            readers = EndpointReaders([first, second], "cpu", 5)
            batches = readers.read()
            next(batches)
            self.assertTrue(quiet_connected.wait(5))
            with readers._ready:
                self.assertTrue(readers._ready.wait_for(lambda: readers._batches[0] is not None, 5))
            readers.close()
            self.assertTrue(quiet_closed.wait(5))
            self.assertTrue(all(not thread.is_alive() for thread in readers._threads))
            self.assertEqual(list(batches), [])

    def test_closing_iterator_closes_all_readers(self) -> None:
        def waiting(connection: socket.socket) -> None:
            connection.sendall(frame(1, detail=1) + frame(2, addresses=(7,)))
            connection.recv(1)

        with worker(waiting) as first, worker(waiting) as second:
            readers = EndpointReaders([first, second], "cpu", 5)
            batches = readers.read()
            next(batches)
            batches.close()
            self.assertTrue(readers._closed)
            self.assertTrue(all(not thread.is_alive() for thread in readers._threads))

    def test_round_robin_serves_every_ready_worker_before_returning_to_first(self) -> None:
        with worker(lambda sock: sock.sendall(trace(10, count=20))) as first, \
                worker(lambda sock: sock.sendall(trace(30, count=20))) as second:
            readers = EndpointReaders([first, second], "cpu", 5)
            batches = readers.read()
            try:
                initial = next(batches)
                with readers._ready:
                    self.assertTrue(readers._ready.wait_for(
                        lambda: all(batch is not None for batch in readers._batches), 5))
                self.assertEqual(next(batches).worker, 1 - initial.worker)
                self.assertEqual(next(batches).worker, initial.worker)
            finally:
                batches.close()

    def test_read_once_and_close_before_start(self) -> None:
        readers = EndpointReaders(["tcp://127.0.0.1:1", "tcp://127.0.0.1:2"], "cpu", 1)
        batches = readers.read()
        with self.assertRaisesRegex(RuntimeError, "only be called once"):
            readers.read()
        readers.close()
        with self.assertRaisesRegex(RuntimeError, "closed before reading"):
            next(batches)
        self.assertEqual(readers._threads, [])
        with self.assertRaisesRegex(RuntimeError, "Pool is closed"):
            readers.read()

    def test_invalid_later_endpoint_opens_no_connections(self) -> None:
        with self.assertRaisesRegex(ValueError, "tcp://host:port"):
            EndpointReaders(["tcp://127.0.0.1:1", "bad"], "cpu", 1)
