# SPDX-License-Identifier: AGPL-3.0-only
"""Terminal outcomes keep worker reasons separate from socket symptoms."""

from collections.abc import Iterator
from contextlib import contextmanager
import socket
import struct
import threading
import unittest

from cpu2tensor import (
    BoundaryProgress, Pool, TerminalReason, TraceConnectionError,
    TraceTerminalError, TransportEnd,
)
from test_consumer import frame, worker


HELLO = 1 << 0
DATA = 1 << 1
START_CONFIGURED = 1 << 2
START_OBSERVED = 1 << 3
STOP_CONFIGURED = 1 << 4
STOP_OBSERVED = 1 << 5


def report(flags: int, *, version: int = 1, reason: int = 1) -> bytes:
    return frame(17, version=3, detail=version | (reason << 8) | (flags << 16))


def hello(features: int = 1) -> bytes:
    return frame(1, version=3, detail=features)


def observations(address: int) -> bytes:
    return frame(2, version=3, addresses=(address,))


@contextmanager
def reset_worker(data: bytes) -> Iterator[str]:
    errors: list[BaseException] = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)

        def serve() -> None:
            try:
                connection, _ = listener.accept()
                connection.sendall(data)
                connection.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0),
                )
                connection.close()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            yield f"tcp://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            thread.join(timeout=5)
            if thread.is_alive():
                raise AssertionError("Reset fixture did not stop")
            if errors:
                raise AssertionError("Reset fixture failed") from errors[0]


class TerminalOutcomeTests(unittest.TestCase):
    def test_deadline_can_arrive_before_hello(self) -> None:
        with worker(report(0)) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaises(TraceTerminalError) as raised:
                list(pool.read())
        outcome = raised.exception.outcome
        self.assertFalse(outcome.hello_received)
        self.assertFalse(outcome.hello_reported)
        self.assertEqual(outcome.reason, TerminalReason.MAX_RUN_DEADLINE)

    def test_deadline_before_data_reports_clean_transport(self) -> None:
        data = hello() + report(HELLO)
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaises(TraceTerminalError) as raised:
                list(pool.read())
            outcome = raised.exception.outcome
            self.assertIsInstance(raised.exception, RuntimeError)
            self.assertEqual(outcome.reason, TerminalReason.MAX_RUN_DEADLINE)
            self.assertEqual((outcome.endpoint, outcome.worker, outcome.source), (endpoint, 0, None))
            self.assertEqual((outcome.hello_received, outcome.data_received), (True, False))
            self.assertEqual((outcome.hello_reported, outcome.data_reported), (True, False))
            self.assertEqual(outcome.start, BoundaryProgress.NOT_CONFIGURED)
            self.assertEqual(outcome.stop, BoundaryProgress.NOT_CONFIGURED)
            self.assertEqual(outcome.transport, TransportEnd.CLEAN)
            self.assertFalse(outcome.complete)
            self.assertEqual(pool.outcomes, (outcome,))

    def test_deadline_after_data_has_same_reason_after_reset(self) -> None:
        stream = hello() + observations(9) + report(HELLO | DATA)
        with reset_worker(stream) as endpoint, Pool([endpoint]) as pool:
            batches = pool.read()
            self.assertEqual(next(batches).addresses.tolist(), [9])
            with self.assertRaises(TraceConnectionError) as raised:
                next(batches)
        outcome = raised.exception.outcome
        self.assertIsInstance(raised.exception, ConnectionError)
        self.assertIsInstance(raised.exception, TraceTerminalError)
        self.assertEqual(outcome.reason, TerminalReason.MAX_RUN_DEADLINE)
        self.assertTrue(outcome.data_received)
        self.assertTrue(outcome.data_reported)
        self.assertEqual(outcome.transport, TransportEnd.RESET)

    def test_deadline_after_start_before_stop_reports_boundaries(self) -> None:
        features = 1 | (1 << 14) | (1 << 19)
        flags = HELLO | DATA | START_CONFIGURED | START_OBSERVED | STOP_CONFIGURED
        stream = hello(features) + observations(3) + report(flags)
        with worker(stream) as endpoint, Pool([endpoint]) as pool:
            batches = pool.read()
            next(batches)
            with self.assertRaises(TraceTerminalError) as raised:
                next(batches)
        self.assertEqual(raised.exception.outcome.start, BoundaryProgress.OBSERVED)
        self.assertEqual(raised.exception.outcome.stop, BoundaryProgress.NOT_OBSERVED)

    def test_report_can_confirm_both_boundaries(self) -> None:
        features = 1 | (1 << 14) | (1 << 19)
        flags = (HELLO | DATA | START_CONFIGURED | START_OBSERVED
                 | STOP_CONFIGURED | STOP_OBSERVED)
        stream = hello(features) + observations(3) + report(flags)
        with worker(stream) as endpoint, Pool([endpoint]) as pool:
            batches = pool.read()
            next(batches)
            with self.assertRaises(TraceTerminalError) as raised:
                next(batches)
        self.assertEqual(raised.exception.outcome.start, BoundaryProgress.OBSERVED)
        self.assertEqual(raised.exception.outcome.stop, BoundaryProgress.OBSERVED)

    def test_legacy_eof_and_reset_remain_unknown_disconnections(self) -> None:
        legacy_hello = frame(1, detail=1)
        with worker(legacy_hello) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaises(TraceTerminalError) as clean:
                list(pool.read())
        self.assertEqual(clean.exception.outcome.reason, TerminalReason.UNKNOWN_DISCONNECTION)
        self.assertEqual(clean.exception.outcome.transport, TransportEnd.CLEAN)
        self.assertIsNone(clean.exception.outcome.hello_reported)
        self.assertEqual(clean.exception.outcome.start, BoundaryProgress.UNKNOWN)

        with reset_worker(legacy_hello) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaises(TraceConnectionError) as reset:
                list(pool.read())
        self.assertEqual(reset.exception.outcome.reason, TerminalReason.UNKNOWN_DISCONNECTION)
        self.assertEqual(reset.exception.outcome.transport, TransportEnd.RESET)

    def test_partial_and_malformed_reports_fail_closed(self) -> None:
        current_hello = hello()
        with worker(current_hello + report(HELLO)[:11]) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaises(TraceTerminalError) as truncated:
                list(pool.read())
        self.assertEqual(truncated.exception.outcome.reason, TerminalReason.TRUNCATED_STREAM)

        with worker(current_hello + report(DATA)) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaises(ValueError):
                list(pool.read())

    def test_report_must_match_received_progress_and_hello(self) -> None:
        with worker(report(HELLO)) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaisesRegex(ValueError, "contradicts received"):
                list(pool.read())

        features = 1 | (1 << 14)
        with worker(hello(features) + report(HELLO)) as endpoint, Pool([endpoint]) as pool:
            with self.assertRaisesRegex(ValueError, "contradicts Hello"):
                list(pool.read())

    def test_success_records_outcome_without_an_exception(self) -> None:
        data = frame(1, detail=1) + frame(3) + frame(4)
        with worker(data) as endpoint, Pool([endpoint]) as pool:
            self.assertEqual(list(pool.read()), [])
            outcome = pool.outcomes[0]
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.reason, TerminalReason.COMPLETE)
        self.assertTrue(outcome.complete)
        self.assertEqual(outcome.transport, TransportEnd.NOT_OBSERVED)

    def test_multi_endpoint_failure_keeps_identity_and_type(self) -> None:
        complete = frame(1, detail=1) + frame(3) + frame(4)
        failed = hello() + report(HELLO)
        with worker(complete) as first, worker(failed) as second, Pool([first, second]) as pool:
            with self.assertRaises(TraceTerminalError) as raised:
                list(pool.read())
            outcome = raised.exception.outcome
            self.assertEqual((outcome.worker, outcome.endpoint), (1, second))
            self.assertIn(f"Worker 1 ({second})", str(raised.exception))
            outcomes = pool.outcomes
        self.assertEqual((outcomes[0].worker, outcomes[0].endpoint), (0, first))
        self.assertEqual(outcomes[1], outcome)

    def test_multi_endpoint_success_records_each_identity(self) -> None:
        complete = frame(1, detail=1) + frame(3) + frame(4)
        with worker(complete) as first, worker(complete) as second, Pool([first, second]) as pool:
            self.assertEqual(list(pool.read()), [])
            outcomes = pool.outcomes
        self.assertEqual(
            [(item.reason, item.worker, item.endpoint) for item in outcomes],
            [
                (TerminalReason.COMPLETE, 0, first),
                (TerminalReason.COMPLETE, 1, second),
            ],
        )


if __name__ == "__main__":
    unittest.main()
