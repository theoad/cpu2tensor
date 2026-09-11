# SPDX-License-Identifier: AGPL-3.0-only
"""Independent context-only Pools decode concurrently without semantic loss."""

import struct
import socket
import threading
import unittest

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from cpu2tensor import Pool, _native
from concurrent_decode import context_only_capture, expected_digest, replay


_HEADER = struct.Struct("<IHHIIQQ")


def _frames(capture: bytes) -> list[bytes]:
    result = []
    offset = 0
    while offset < len(capture):
        size = _native.payload_size(capture[offset:offset + _HEADER.size])
        end = offset + _HEADER.size + size
        result.append(capture[offset:end])
        offset = end
    return result


@contextmanager
def _partial_successor_worker(frames: list[bytes]):
    release = threading.Event()
    errors = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(3)

        def serve() -> None:
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.sendall(frames[0] + frames[1] + frames[2][:_HEADER.size - 1])
                    if not release.wait(3):
                        raise AssertionError("Client did not release the partial frame")
                    connection.sendall(frames[2][_HEADER.size - 1:] + b"".join(frames[3:]))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            yield f"tcp://127.0.0.1:{listener.getsockname()[1]}", release
        finally:
            release.set()
            thread.join(timeout=4)
            if thread.is_alive() or errors:
                raise AssertionError("Partial-frame worker did not stop cleanly")


class ConcurrentDecodeTests(unittest.TestCase):
    def test_partial_successor_does_not_delay_a_ready_group(self) -> None:
        frames = _frames(context_only_capture(0, 1))
        with _partial_successor_worker(frames) as (endpoint, release):
            with Pool([endpoint], timeout=2) as pool, ThreadPoolExecutor(max_workers=1) as executor:
                batches = pool.read()
                ready = executor.submit(next, batches)
                try:
                    first = ready.result(timeout=0.5)
                except TimeoutError:
                    release.set()
                    ready.result(timeout=2)
                    self.fail("A partial successor delayed the complete first batch")
                self.assertEqual(first.source, 0)
                self.assertEqual(first.block_sequences.tolist(), list(range(256))[1:])
                release.set()
                self.assertEqual(len(list(batches)), 1)

    def test_native_group_publishes_one_owned_result_per_source(self) -> None:
        frames = _frames(context_only_capture(3, 8))
        stream = _native.new_stream()
        _native.decode(stream, frames[0])
        decoded, error = _native.decode_context_frames(stream, frames[1:17])
        del frames

        self.assertIsNone(error)
        self.assertEqual([item[1] for item in decoded], [0, 1])
        for _, source, count, sequence, _, payload in decoded:
            self.assertEqual(count, 8 * 256)
            self.assertEqual(sequence, 0)
            block_sequences = list(struct.iter_unpack("<Q", payload["block_sequences"]))
            context_sequences = list(struct.iter_unpack("<Q", payload["context"]["sequences"]))
            self.assertEqual(len(block_sequences), 8 * 256 - 1)
            self.assertEqual(context_sequences, [(0,)])
            self.assertEqual(
                sorted(value[0] for value in block_sequences + context_sequences),
                list(range(8 * 256)),
            )
            first_address = struct.unpack_from("<Q", payload["blocks"])[0]
            self.assertEqual(first_address, 3_000_000_001 + source * 100_000_000)

    def test_native_group_returns_rows_before_a_sequence_error(self) -> None:
        frames = _frames(context_only_capture(0, 2))
        bad = bytearray(frames[3])
        struct.pack_into("<Q", bad, 16, 512)
        stream = _native.new_stream()
        _native.decode(stream, frames[0])
        decoded, error = _native.decode_context_frames(stream, [frames[1], frames[2], bad])

        self.assertIn("source", error.lower())
        self.assertEqual([item[1] for item in decoded], [0, 1])
        self.assertEqual([item[2] for item in decoded], [256, 256])

    def test_pool_yields_accepted_groups_before_a_sequence_error(self) -> None:
        frames = _frames(context_only_capture(0, 2))
        bad = bytearray(frames[3])
        struct.pack_into("<Q", bad, 16, 512)
        data = frames[0] + frames[1] + frames[2] + bad
        from test_consumer import worker
        with worker(data, fragment=len(data)) as endpoint, Pool([endpoint]) as pool:
            batches = pool.read()
            self.assertEqual({next(batches).source, next(batches).source}, {0, 1})
            with self.assertRaisesRegex(ValueError, "source"):
                next(batches)

    def test_pool_yields_a_group_before_a_bad_buffered_header(self) -> None:
        frames = _frames(context_only_capture(0, 1))
        bad = bytearray(frames[-1])
        struct.pack_into("<H", bad, 4, 99)
        data = frames[0] + frames[1] + bad
        from test_consumer import worker
        with worker(data, fragment=len(data)) as endpoint, Pool([endpoint]) as pool:
            batches = pool.read()
            self.assertEqual(next(batches).source, 0)
            with self.assertRaisesRegex(ValueError, "Unsupported trace version"):
                next(batches)

    def test_one_four_and_sixteen_independent_pools(self) -> None:
        for worker_count in (1, 4, 16):
            with self.subTest(worker_count=worker_count):
                results, errors, elapsed = replay(worker_count, 8)
                self.assertLess(elapsed, 15)
                self.assertEqual(errors, [])
                self.assertEqual(
                    sorted(results, key=lambda result: result.identity),
                    [expected_digest(identity, 8) for identity in range(worker_count)],
                )

    def test_incomplete_stream_does_not_hide_other_results(self) -> None:
        results, errors, elapsed = replay(4, 4, incomplete=2)
        self.assertLess(elapsed, 15)
        self.assertEqual(
            sorted(results, key=lambda result: result.identity),
            [expected_digest(identity, 4) for identity in (0, 1, 3)],
        )
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("Incomplete trace", str(errors[0]))


if __name__ == "__main__":
    unittest.main()
