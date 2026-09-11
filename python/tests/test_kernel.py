# SPDX-License-Identifier: AGPL-3.0-only
"""Kernel control fixtures use real sockets, decoding, and tensor ownership."""

import importlib.util
import json
import socket
import struct
import unittest

import numpy as np
import torch

from cpu2tensor.batch import Batch
from cpu2tensor.examples.learn_kernel import COMMANDS, SourceBlockFeatures, run, verify_result
from cpu2tensor.kernel import KernelEnv
from test_consumer import frame, worker
from test_signals import MEMORY, VALUES, REGISTERS, memory_row, register_row, schema_row, signal_frame
from test_stdio import interactive_worker, receive


FEATURES = (1 << 12) | (1 << 13)
WINDOWS = 1 << 20


def event(value: dict) -> bytes:
    payload = json.dumps(value).encode()
    return frame(11, detail=len(payload)) + payload


def ready(step: int) -> bytes:
    return event({"event": "ready", "step": step}) + frame(10, detail=127)


def transition_frame(source: int, window: int,
                     rows: tuple[tuple[int, int, int], ...]) -> bytes:
    payload = b"".join(struct.pack("<QQQ", *row) for row in rows)
    return (struct.pack("<IHHIIQQ", 0x31543243, 2, 15, source, len(rows),
                        window, len(payload)) + payload)


def window_frame(window: int, status: int, *, sources: int, capacity: int,
                 distinct: int, observed: int, overflow: int = 0) -> bytes:
    payload = struct.pack("<IIIIQQQ", status, sources, capacity, 0,
                          distinct, observed, overflow)
    return (struct.pack("<IHHIIQQ", 0x31543243, 2, 16, 0, 1,
                        window, len(payload)) + payload)


def read_action(connection: socket.socket) -> bytes:
    length = struct.unpack("<I", receive(connection, 4))[0]
    return receive(connection, length)


def result_for(action: int, step: int) -> dict:
    # Constants were independently calculated from the documented recurrence.
    results = (
        {"action": "getpid", "value": 1},
        {"action": "memory", "seed": 17, "bytes": 256, "checksum": 31717},
        {"action": "pipe", "seed": 17, "bytes": 64, "checksum": 6863},
        {"action": "parallel", "seed": 17, "bytes": 256, "cpu0": 0, "cpu1": 1,
         "checksum0": 31717, "checksum1": 31310},
    )
    return {"event": "result", "step": step, **results[action]}


class KernelTests(unittest.TestCase):
    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_window_summary_empty_tensor_uses_configured_device(self) -> None:
        def serve(connection: socket.socket) -> None:
            connection.sendall(frame(1, detail=2 | FEATURES | WINDOWS) + ready(0))
            self.assertEqual(read_action(connection), COMMANDS[4])
            connection.sendall(window_frame(1, 1, sources=1, capacity=8,
                                            distinct=0, observed=0)
                               + frame(3, source=0)
                               + event({"event": "complete", "ok": True}) + frame(4))

        with interactive_worker(serve) as endpoint, KernelEnv(endpoint, device="mps") as env:
            self.assertEqual(list(env.reset()), [])
            batches = list(env.step(COMMANDS[4]))
        summary = next(batch for batch in batches if batch.transition_window is not None)
        self.assertEqual(summary.addresses.device.type, "mps")

    def test_reduced_window_matches_raw_per_source_transitions(self) -> None:
        def serve(connection: socket.socket) -> None:
            connection.sendall(frame(1, detail=2 | FEATURES | WINDOWS) + ready(0))
            self.assertEqual(read_action(connection), COMMANDS[0])
            connection.sendall(
                frame(2, source=0, addresses=(10, 20, 10, 20))
                + frame(2, source=1, addresses=(90, 91))
                + transition_frame(0, 1, ((10, 20, 2), (20, 10, 1)))
                + transition_frame(1, 1, ((90, 91, 1),))
                + window_frame(1, 1, sources=2, capacity=8,
                               distinct=3, observed=4)
                + event(result_for(0, 0)) + ready(1)
            )
            self.assertEqual(read_action(connection), COMMANDS[4])
            connection.sendall(window_frame(2, 1, sources=2, capacity=8,
                                            distinct=0, observed=0)
                               + frame(3, source=0, sequence=4)
                               + frame(3, source=1, sequence=2)
                               + event({"event": "complete", "steps": 1, "ok": True})
                               + frame(4))

        with interactive_worker(serve) as endpoint, KernelEnv(endpoint) as env:
            self.assertEqual(list(env.reset()), [])
            batches = list(env.step(COMMANDS[0]))
            raw: dict[int, list[int]] = {}
            reduced: dict[tuple[int, int, int], int] = {}
            summary = None
            for batch in batches:
                if batch.addresses.numel():
                    raw.setdefault(batch.source, []).extend(batch.addresses.tolist())
                if batch.transitions is not None:
                    for source, destination, count in zip(
                        batch.transitions.from_addresses.tolist(),
                        batch.transitions.destinations.tolist(),
                        batch.transitions.counts.tolist(), strict=True,
                    ):
                        reduced[(batch.source, source, destination)] = count
                if batch.transition_window is not None:
                    summary = batch.transition_window
            expected: dict[tuple[int, int, int], int] = {}
            for source, addresses in raw.items():
                for previous, current in zip(addresses, addresses[1:]):
                    key = (source, previous, current)
                    expected[key] = expected.get(key, 0) + 1
            self.assertEqual(reduced, expected)
            self.assertIsNotNone(summary)
            self.assertEqual(summary.status, "ended")
            self.assertTrue(summary.complete)
            self.assertEqual((summary.distinct, summary.observed, summary.overflow),
                             (3, 4, 0))
            list(env.step(COMMANDS[4]))

    def test_incomplete_and_overflow_window_statuses_are_public(self) -> None:
        def serve(connection: socket.socket) -> None:
            connection.sendall(frame(1, detail=2 | FEATURES | WINDOWS) + ready(0))
            self.assertEqual(read_action(connection), COMMANDS[0])
            connection.sendall(window_frame(1, 3, sources=2, capacity=8,
                                            distinct=0, observed=0) + ready(1))
            self.assertEqual(read_action(connection), COMMANDS[1])
            connection.sendall(window_frame(2, 1, sources=2, capacity=8,
                                            distinct=0, observed=3, overflow=3) + ready(2))
            self.assertEqual(read_action(connection), COMMANDS[4])
            connection.sendall(window_frame(3, 2, sources=2, capacity=8,
                                            distinct=0, observed=0)
                               + frame(3, source=0) + frame(3, source=1)
                               + event({"event": "complete", "ok": True}) + frame(4))

        with interactive_worker(serve) as endpoint, KernelEnv(endpoint) as env:
            self.assertEqual(list(env.reset()), [])
            windows = [batch.transition_window for batch in env.step(COMMANDS[0])
                       if batch.transition_window is not None]
            windows.extend(batch.transition_window for batch in env.step(COMMANDS[1])
                           if batch.transition_window is not None)
            list(env.step(COMMANDS[4]))
        self.assertEqual([window.status for window in windows],
                         ["incomplete", "ended"])
        self.assertEqual(windows[1].overflow, 3)
        self.assertFalse(windows[1].complete)

    def test_boundary_requires_complete_baselines_on_each_source(self) -> None:
        data = (frame(1, detail=2 | FEATURES | REGISTERS | (1 << 16))
                + signal_frame(6, schema_row() + schema_row(register=1, name=b"x1"), count=2, source=1)
                + signal_frame(12, struct.pack('<8Q', 0x1000, 0, 0, 0, 0, 0, 64, 63), source=1)
                + signal_frame(7, register_row(bytes(8)), source=1, sequence=1)
                + frame(10, detail=127))
        with worker(data) as endpoint, KernelEnv(endpoint) as env:
            with self.assertRaisesRegex(ValueError, "complete source register baselines"):
                list(env.reset())

    def test_two_sources_signals_and_control_preserve_sequences_and_storage(self) -> None:
        def serve(connection: socket.socket) -> None:
            first = (frame(1, detail=2 | FEATURES | MEMORY | VALUES)
                     + frame(2, source=1, addresses=(2**64 - 1,))
                     + frame(2, source=0, addresses=(0x1000, 0x1004)) + ready(0))
            for offset in range(0, len(first), 5):
                connection.sendall(first[offset:offset + 5])
            self.assertEqual(read_action(connection), COMMANDS[1])
            connection.sendall(signal_frame(8, memory_row(address=2**63, size=1,
                                                           value=b"\xfe" + bytes(15)),
                                                source=1, sequence=1)
                               + frame(2, source=0, sequence=2, addresses=(0x2000,))
                               + event(result_for(1, 0)) + ready(1))
            self.assertEqual(read_action(connection), COMMANDS[4])
            connection.sendall(frame(3, source=0, sequence=3)
                               + event({"event": "complete", "steps": 1, "ok": True})
                               + frame(3, source=1, sequence=2) + frame(4))

        with interactive_worker(serve) as endpoint, KernelEnv(endpoint) as env:
            before = list(env.reset())
            self.assertEqual([(b.source, b.first_sequence) for b in before], [(1, 0), (0, 0)])
            self.assertEqual(env.max_action_bytes, 127)
            self.assertEqual(env.event, {"event": "ready", "step": 0})
            self.assertIsNone(env.result)
            after = list(env.step(COMMANDS[1]))
            result = env.result
            self.assertEqual([(b.source, b.first_sequence) for b in after], [(1, 1), (0, 2)])
            self.assertEqual(after[0].memory.addresses.tolist(), [-(2**63)])
            self.assertEqual(after[0].memory.values.tolist(), [[254] + [0] * 15])
            self.assertEqual(before[0].addresses.tolist(), [-1])
            self.assertEqual(env.event, {"event": "ready", "step": 1})
            self.assertEqual(result, result_for(1, 0))
            pending = env.step(COMMANDS[4])
            self.assertIsNone(env.result)
            self.assertIsNone(env.event)
            list(pending)
            self.assertEqual(result, result_for(1, 0))
            self.assertEqual(env.exit_code, 0)
            self.assertEqual(env.event["event"], "complete")

    def test_invalid_action_keeps_request_available(self) -> None:
        def serve(connection: socket.socket) -> None:
            connection.sendall(frame(1, detail=2 | FEATURES) + ready(0))
            self.assertEqual(read_action(connection), COMMANDS[0])
            connection.sendall(frame(4))

        with interactive_worker(serve) as endpoint, KernelEnv(endpoint) as env:
            list(env.reset())
            for action in (b"", b"getpid", b"getpid\nquit\n", b"\x00\n", b"a" * 127 + b"\n"):
                with self.subTest(action=action), self.assertRaises(ValueError):
                    env.step(action)
                self.assertTrue(env.needs_input)
            with self.assertRaises(TypeError):
                env.step("getpid\n")
            list(env.step(COMMANDS[0]))

    def test_invalid_control_metadata_and_missing_source_tail_fail(self) -> None:
        cases = (
            frame(10, source=1, detail=127), frame(10, sequence=1, detail=127),
            frame(10, detail=128), frame(11, detail=1025),
            frame(2, source=1, addresses=(1,)) + frame(2, source=1, sequence=2, addresses=(2,)),
            frame(2, source=1, addresses=(1,)) + frame(4),
            frame(11, detail=2) + b"[]", frame(11, detail=2) + b"??",
            event({"step": 0}),
        )
        for tail in cases:
            with self.subTest(tail=tail), worker(frame(1, detail=2 | FEATURES) + tail) as endpoint:
                with KernelEnv(endpoint) as env, self.assertRaises(ValueError):
                    list(env.reset())
                self.assertIsNone(env.exit_code)

    def test_non_kernel_worker_and_invalid_features_fail(self) -> None:
        for detail in (2, 2 | (1 << 11), 2 | (1 << 13), 2 | FEATURES | (1 << 11)):
            with self.subTest(detail=detail), worker(frame(1, detail=detail)) as endpoint:
                with KernelEnv(endpoint) as env, self.assertRaises((RuntimeError, ValueError)):
                    list(env.reset())

    def test_capture_failure_disconnect_and_timeout_are_not_completion(self) -> None:
        for tail in (b"", frame(5, detail=1)):
            with self.subTest(tail=tail), worker(frame(1, detail=2 | FEATURES) + tail) as endpoint:
                with KernelEnv(endpoint) as env, self.assertRaisesRegex(RuntimeError, "Incomplete trace"):
                    list(env.reset())
                self.assertIsNone(env.exit_code)
        with worker(frame(1, detail=2 | FEATURES), wait_for_close=True) as endpoint:
            with KernelEnv(endpoint, timeout=0.1) as env, self.assertRaises(TimeoutError):
                list(env.reset())
            self.assertIsNone(env.exit_code)

    def test_reset_cancels_old_run_without_stale_iterator_cancelling_new_run(self) -> None:
        def first(connection: socket.socket) -> None:
            connection.sendall(frame(1, detail=2 | FEATURES) + frame(2, addresses=(1,)) + ready(0))
            self.assertEqual(connection.recv(1), b"")

        def second(connection: socket.socket) -> None:
            connection.sendall(frame(1, detail=2 | FEATURES) + ready(0))
            self.assertEqual(read_action(connection), COMMANDS[4])
            connection.sendall(frame(4))

        with interactive_worker(first, second) as endpoint, KernelEnv(endpoint) as env:
            old = env.reset()
            retained = next(old)
            current = env.reset()
            with self.assertRaisesRegex(RuntimeError, "cancelled run"):
                next(old)
            list(current)
            list(env.step(COMMANDS[4]))
            self.assertEqual(retained.addresses.tolist(), [1])
            self.assertEqual(env.exit_code, 0)

    def test_features_keep_cpu_rows_and_retained_batches_independent(self) -> None:
        reducer = SourceBlockFeatures(sources=2, bins=8)
        address = torch.tensor([4, 4, 8, -1], dtype=torch.int64)
        original = address.clone()
        features = reducer([Batch(1, 0, address), Batch(0, 0, torch.tensor([4]))])
        self.assertEqual(reducer.counts.sum(dim=1).tolist(), [1, 4])
        self.assertTrue(torch.equal(address, original))
        retained = features.copy()
        next_features = reducer([Batch(0, 1, torch.tensor([8]))])
        np.testing.assert_array_equal(features, retained)
        self.assertFalse(np.shares_memory(features, next_features))
        self.assertTrue(np.isfinite(features).all())
        with self.assertRaisesRegex(ValueError, "CPU exceeds"):
            reducer([Batch(2, 0, torch.tensor([4]))])

    def test_result_oracle_rejects_wrong_values_and_fake_parallel_cpu(self) -> None:
        for action in range(4):
            verify_result(action, result_for(action, 0))
        with self.assertRaisesRegex(ValueError, "Incorrect"):
            verify_result(1, {**result_for(1, 0), "checksum": 1})
        with self.assertRaisesRegex(ValueError, "distinct"):
            verify_result(3, {**result_for(3, 0), "cpu1": 0})
        with self.assertRaises(ValueError):
            verify_result(0, None)

    @unittest.skipUnless(importlib.util.find_spec("gymnasium"), "Gymnasium is optional")
    def test_gym_example_checks_workloads_updates_policy_and_completes(self) -> None:
        observed_actions = []

        def serve(connection: socket.socket) -> None:
            sequence = [1, 1]
            connection.sendall(frame(1, detail=2 | FEATURES)
                               + frame(2, source=0, addresses=(4,))
                               + frame(2, source=1, addresses=(8,)) + ready(0))
            for step in range(8):
                action = COMMANDS.index(read_action(connection))
                observed_actions.append(action)
                if action == 4:
                    connection.sendall(event({"event": "complete", "steps": step, "ok": True})
                                       + frame(3, source=0, sequence=sequence[0])
                                       + frame(3, source=1, sequence=sequence[1]) + frame(4))
                    return
                source = step % 2
                connection.sendall(frame(2, source=source, sequence=sequence[source],
                                         addresses=(4 * (step + 3),) * (action + 1))
                                   + event(result_for(action, step)) + ready(step + 1))
                sequence[source] += action + 1
            raise AssertionError("Example did not send quit")

        with interactive_worker(serve) as endpoint:
            metrics, checkpoint = run(endpoint, bins=16, policy_steps=3)
        self.assertEqual(observed_actions[:4], [0, 1, 2, 3])
        self.assertEqual(observed_actions[-1], 4)
        self.assertEqual(len(metrics["policy_updates"]), 3)
        self.assertTrue(metrics["complete"])
        self.assertTrue(metrics["parameters_changed"])
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in checkpoint["model_state"].values()))
        self.assertGreater(metrics["policy_updates"][0]["reward"], 0)


if __name__ == "__main__":
    unittest.main()
