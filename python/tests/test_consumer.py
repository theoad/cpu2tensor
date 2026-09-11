# SPDX-License-Identifier: AGPL-3.0-only
"""Small TCP peers exercise the actual native decoder and tensor consumer."""

from collections.abc import Iterator
from contextlib import contextmanager
import gc
import socket
import struct
import threading
import unittest
from unittest import mock
import weakref

import torch

from cpu2tensor import Pool


def frame(
    kind: int,
    *,
    version: int = 2,
    source: int = 0,
    sequence: int = 0,
    detail: int = 0,
    addresses: tuple[int, ...] = (),
) -> bytes:
    header = struct.pack("<IHHIIQQ", 0x31543243, version, kind, source, len(addresses), sequence, detail)
    return header + struct.pack(f"<{len(addresses)}Q", *addresses)


@contextmanager
def worker(data: bytes, *, fragment: int = 7, wait_for_close: bool = False) -> Iterator[str]:
    """Publish a finite fixture; never leave a server waiting after a failed test."""
    errors: list[BaseException] = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(5)

        def serve() -> None:
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(5)
                    for offset in range(0, len(data), fragment):
                        connection.sendall(data[offset:offset + fragment])
                    if wait_for_close and connection.recv(1) != b"":
                        raise AssertionError("Observation client sent unexpected data")
            except (BrokenPipeError, ConnectionResetError):
                pass  # Invalid frames and early cancellation close the client.
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            yield f"tcp://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            thread.join(timeout=6)
            if thread.is_alive():
                raise AssertionError("Fixture server did not stop")
            if errors:
                raise AssertionError("Fixture server failed") from errors[0]


class ConsumerTests(unittest.TestCase):
    def test_full_batch_then_final_partial_without_retention(self) -> None:
        data = (frame(1, detail=1) + frame(2, addresses=tuple(range(256)))
                + frame(2, sequence=256, addresses=(256,)) + frame(3, sequence=257) + frame(4))
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            batches = pool.read()
            first = next(batches)
            self.assertEqual(first.addresses.tolist(), list(range(256)))
            storage = weakref.ref(first.addresses)
            del first
            final = next(batches)
            gc.collect()
            self.assertIsNone(storage(), "Pool kept an already consumed tensor alive")
            self.assertEqual(final.first_sequence, 256)
            self.assertEqual(final.addresses.tolist(), [256])
            self.assertEqual(list(batches), [])

    def test_fragmented_partial_batches_and_independent_sources(self) -> None:
        data = b"".join((
            frame(1, detail=1),
            frame(2, source=1, addresses=(100, 104)),
            frame(2, source=0, addresses=(200,)),
            frame(2, source=1, sequence=2, addresses=(108,)),
            frame(3, source=0, sequence=1),
            frame(3, source=1, sequence=3),
            frame(4),
        ))
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            batches = list(pool.read())
        self.assertEqual([(batch.source, batch.first_sequence) for batch in batches],
                         [(1, 0), (0, 0), (1, 2)])
        self.assertEqual([batch.addresses.tolist() for batch in batches], [[100, 104], [200], [108]])

    def check_retained_storage(self, device: str) -> None:
        addresses = (0, (1 << 63) - 1, 1 << 63, (1 << 64) - 1)
        data = frame(1, detail=1) + frame(2, addresses=addresses)
        for sequence in range(4, 304):
            data += frame(2, sequence=sequence, addresses=(sequence,))
        data += frame(3, sequence=304) + frame(4)
        with worker(data, fragment=4096) as endpoint, Pool([endpoint], device=device) as pool:
            batches = pool.read()
            first = next(batches)
            for batch in batches:
                self.assertEqual(batch.addresses.numel(), 1)
        del pool, batches, batch
        gc.collect()
        self.assertEqual(first.addresses.dtype, torch.int64)
        self.assertEqual(first.addresses.device.type, device)
        self.assertEqual(first.addresses.cpu().tolist(), [0, (1 << 63) - 1, -(1 << 63), -1])
        features = (first.addresses & 255).to(torch.float32) / 255
        model = torch.nn.Linear(4, 1, bias=False, device=device)
        with torch.no_grad():
            model.weight.fill_(1)
        result = model(features)
        result.sum().backward()
        self.assertEqual(result.item(), 2)
        self.assertEqual(model.weight.grad.cpu().tolist(), [[0, 1, 0, 1]])

    def test_retained_cpu_storage_and_model(self) -> None:
        self.check_retained_storage("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable on this host")
    def test_retained_mps_storage_and_model(self) -> None:
        self.check_retained_storage("mps")

    def test_invalid_frames(self) -> None:
        bad_version = bytearray(frame(1, detail=1))
        struct.pack_into("<H", bad_version, 4, 1)
        oversized = bytearray(frame(2, addresses=(1,)))
        struct.pack_into("<I", oversized, 12, 257)
        cases = {
            "version": bytes(bad_version),
            "length": frame(1, detail=1) + bytes(oversized),
            "sequence gap": frame(1, detail=1) + frame(2, sequence=1, addresses=(1,)),
            "repeated entry": frame(1, detail=1) + frame(2, addresses=(1,)) + frame(2, addresses=(1,)),
            "missing source end": frame(1, detail=1) + frame(2, addresses=(1,)) + frame(4),
            "duplicate source end": frame(1, detail=1) + frame(3) + frame(3),
            "missing hello": frame(2, addresses=(1,)),
        }
        for name, data in cases.items():
            with self.subTest(name=name), worker(data) as endpoint, Pool([endpoint]) as pool:
                with self.assertRaises(ValueError):
                    list(pool.read())

    def test_disconnect_is_incomplete(self) -> None:
        complete = frame(1, detail=1) + frame(2, addresses=(1,)) + frame(3, sequence=1)
        for data in (b"", complete, complete[:40], complete + frame(4)[:10]):
            with self.subTest(length=len(data)), worker(data) as endpoint, Pool([endpoint]) as pool:
                with self.assertRaisesRegex(RuntimeError, "Incomplete trace"):
                    list(pool.read())

    def test_nonzero_exit_delivers_preceding_batch(self) -> None:
        data = frame(1, detail=1) + frame(2, addresses=(4,)) + frame(3, sequence=1) + frame(4, detail=7)
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            batches = pool.read()
            self.assertEqual(next(batches).addresses.item(), 4)
            with self.assertRaisesRegex(RuntimeError, "exited with code 7"):
                next(batches)

    def test_worker_errors(self) -> None:
        for detail in range(1, 5):
            with self.subTest(detail=detail), worker(frame(1, detail=1) + frame(5, detail=detail)) as endpoint:
                with Pool([endpoint]) as pool, self.assertRaisesRegex(RuntimeError, "Incomplete trace"):
                    list(pool.read())

    def test_read_once_and_close(self) -> None:
        data = frame(1, detail=1) + frame(2, addresses=(4,))
        with worker(data, wait_for_close=True) as endpoint, Pool([endpoint]) as pool:
            batches = pool.read()
            with self.assertRaisesRegex(RuntimeError, "only be called once"):
                pool.read()
            first = next(batches)
            pool.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pool.read()
            batches.close()
        self.assertEqual(first.addresses.item(), 4)

    def test_waiting_for_data_times_out_and_closes_connection(self) -> None:
        with worker(frame(1, detail=1), wait_for_close=True) as endpoint:
            with Pool([endpoint], timeout=0.1) as pool:
                with self.assertRaisesRegex(TimeoutError, "timed out before completion"):
                    list(pool.read())

    def test_reject_unsupported_configuration(self) -> None:
        for endpoints in ([], ["tcp://host:1", "tcp://host:1"], "tcp://host:1", ["ssh://host:1"], ["tcp://host:1/path"]):
            with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                Pool(endpoints)
        with self.assertRaises(ValueError):
            Pool(["tcp://host:1"], device="meta")
        with self.assertRaises(ValueError):
            Pool(["tcp://host:1"], timeout=0)
        with mock.patch("cpu2tensor.pool.torch.backends.mps.is_available", return_value=False), \
                self.assertRaisesRegex(RuntimeError, "MPS is not available"):
            Pool(["tcp://host:1"], device="mps")

    def test_enter_rejects_a_closed_pool(self) -> None:
        pool = Pool(["tcp://host:1"])
        pool.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.__enter__()

    def test_close_interrupts_a_published_connection_attempt(self) -> None:
        connecting = threading.Event()
        interrupted = threading.Event()
        finished = threading.Event()
        errors = []

        def connect(address) -> None:
            self.assertEqual(address, ("127.0.0.1", 12345))
            connecting.set()
            if not interrupted.wait(5):
                raise AssertionError("Connection attempt was not interrupted")
            raise OSError("Connection was closed")

        connection = mock.Mock(spec=socket.socket)
        connection.connect.side_effect = connect
        connection.shutdown.side_effect = lambda _: interrupted.set()
        connection.close.side_effect = interrupted.set
        pool = Pool(["tcp://127.0.0.1:12345"])

        def read() -> None:
            try:
                list(pool.read())
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        with mock.patch("cpu2tensor.pool.socket.socket", return_value=connection):
            reader = threading.Thread(target=read)
            reader.start()
            try:
                self.assertTrue(connecting.wait(2))
                pool.close()
                self.assertTrue(finished.wait(2), "close left the connection attempt blocked")
            finally:
                interrupted.set()
                reader.join(5)
        self.assertFalse(reader.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("closed while connecting", str(errors[0]))
        connection.shutdown.assert_called_once_with(socket.SHUT_RDWR)

    def test_failed_address_candidate_does_not_prevent_a_later_connection(self) -> None:
        data = frame(1, detail=1) + frame(2, addresses=(91,)) + frame(3, sequence=1) + frame(4)
        original_socket = socket.socket

        def without_ipv6(family=socket.AF_INET, *arguments, **keywords):
            if family == socket.AF_INET6:
                raise OSError("IPv6 is unavailable")
            return original_socket(family, *arguments, **keywords)

        with worker(data) as endpoint:
            port = int(endpoint.rsplit(":", 1)[1])
            addresses = [
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", port, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", port)),
            ]
            with mock.patch("cpu2tensor.pool.socket.getaddrinfo", return_value=addresses), \
                    mock.patch("cpu2tensor.pool.socket.socket", side_effect=without_ipv6), \
                    Pool([endpoint]) as pool:
                self.assertEqual([batch.addresses.item() for batch in pool.read()], [91])


if __name__ == "__main__":
    unittest.main()
