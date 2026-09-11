# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise interactive boundaries through the real native frame decoder."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import importlib.util
import socket
import struct
import threading
import unittest
from unittest import mock

import torch

from cpu2tensor.stdio import StdioEnv
from test_consumer import frame


_FEATURE_STDIO = 1 << 11


def receive(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        part = connection.recv(size - len(result))
        if not part:
            raise AssertionError("Client disconnected before sending its full action")
        result.extend(part)
    return bytes(result)


@contextmanager
def interactive_worker(*runs: Callable[[socket.socket], None]) -> Iterator[str]:
    errors: list[BaseException] = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(5)

        def serve() -> None:
            try:
                for run in runs:
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(5)
                        try:
                            run(connection)
                        except (BrokenPipeError, ConnectionResetError):
                            pass  # Cancelling an unread stream can send a TCP reset.
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            yield f"tcp://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            thread.join(timeout=6)
            if thread.is_alive():
                raise AssertionError("Interactive fixture server did not stop")
            if errors:
                raise AssertionError("Interactive fixture server failed") from errors[0]


def prompt(connection: socket.socket, *, address: int = 0x1000) -> None:
    data = (frame(1, detail=1 | _FEATURE_STDIO) + frame(2, addresses=(address,))
            + frame(9, sequence=1, detail=2))
    for offset in range(0, len(data), 7):
        connection.sendall(data[offset:offset + 7])


def episode(connection: socket.socket, *, expected: bytes = b"3\n", exit_code: int = 0) -> None:
    prompt(connection)
    length = struct.unpack("<I", receive(connection, 4))[0]
    if receive(connection, length) != expected:
        raise AssertionError("Worker received the wrong action")
    connection.sendall(frame(2, sequence=1, addresses=(0x2000,))
                       + frame(3, sequence=2) + frame(4, detail=exit_code))


class StdioTests(unittest.TestCase):
    def test_streaming_reset_step_and_retained_tensor(self) -> None:
        with interactive_worker(episode) as endpoint, StdioEnv(endpoint) as env:
            before = list(env.reset())
            self.assertTrue(env.needs_input)
            self.assertEqual(env.max_action_bytes, 2)
            self.assertIsNone(env.exit_code)
            after = list(env.step(b"3\n"))
            self.assertEqual(after[0].addresses.tolist(), [0x2000])
            self.assertEqual(after[0].first_sequence, 1)
            self.assertEqual(before[0].addresses.tolist(), [0x1000])
            self.assertFalse(env.needs_input)
            self.assertEqual(env.exit_code, 0)

    def test_nonzero_target_exit_is_an_episode_result(self) -> None:
        def lose(connection: socket.socket) -> None:
            episode(connection, expected=b"9\n", exit_code=1)

        with interactive_worker(lose) as endpoint, StdioEnv(endpoint) as env:
            list(env.reset())
            list(env.step(b"9\n"))
            self.assertEqual(env.exit_code, 1)

    def test_multiple_input_requests_preserve_event_sequence(self) -> None:
        def two_inputs(connection: socket.socket) -> None:
            prompt(connection)
            self.assertEqual(receive(connection, 6), b"\x02\0\0\0a\n")
            connection.sendall(frame(2, sequence=1, addresses=(0x2000,))
                               + frame(9, sequence=2, detail=2))
            self.assertEqual(receive(connection, 6), b"\x02\0\0\0b\n")
            connection.sendall(frame(3, sequence=2) + frame(4))

        with interactive_worker(two_inputs) as endpoint, StdioEnv(endpoint) as env:
            list(env.reset())
            middle = list(env.step(b"a\n"))
            self.assertEqual(middle[0].first_sequence, 1)
            self.assertTrue(env.needs_input)
            self.assertEqual(list(env.step(b"b\n")), [])
            self.assertEqual(env.exit_code, 0)

    def test_reset_restarts_a_paused_target(self) -> None:
        def cancelled(connection: socket.socket) -> None:
            prompt(connection)
            self.assertEqual(connection.recv(1), b"")

        with interactive_worker(cancelled, episode) as endpoint, StdioEnv(endpoint) as env:
            list(env.reset())
            list(env.reset())
            list(env.step(b"3\n"))
            self.assertEqual(env.exit_code, 0)

    def test_reset_invalidates_old_iterator_without_cancelling_new_run(self) -> None:
        def cancelled(connection: socket.socket) -> None:
            prompt(connection)
            self.assertEqual(connection.recv(1), b"")

        with interactive_worker(cancelled, episode) as endpoint, StdioEnv(endpoint) as env:
            old = env.reset()
            next(old)
            current = env.reset()
            with self.assertRaisesRegex(RuntimeError, "cancelled run"):
                next(old)
            list(current)
            list(env.step(b"3\n"))
            self.assertEqual(env.exit_code, 0)

    def test_invalid_actions_leave_request_available(self) -> None:
        with interactive_worker(episode) as endpoint, StdioEnv(endpoint) as env:
            with self.assertRaisesRegex(RuntimeError, "reset"):
                env.step(b"3\n")
            observations = env.reset()
            next(observations)
            with self.assertRaisesRegex(RuntimeError, "Consume all"):
                env.step(b"3\n")
            list(observations)
            with self.assertRaises(TypeError):
                env.step("3\n")  # type: ignore[arg-type]
            for action in (b"", b"123"):
                with self.assertRaises(ValueError):
                    env.step(action)
            self.assertTrue(env.needs_input)
            list(env.step(b"3\n"))

    def test_observation_worker_is_rejected(self) -> None:
        def observe_only(connection: socket.socket) -> None:
            connection.sendall(frame(1, detail=1) + frame(4))

        with interactive_worker(observe_only) as endpoint, StdioEnv(endpoint) as env:
            with self.assertRaisesRegex(RuntimeError, "interactive worker"):
                list(env.reset())

    def test_disconnect_is_not_episode_completion(self) -> None:
        def disconnect(connection: socket.socket) -> None:
            connection.sendall(frame(1, detail=1 | _FEATURE_STDIO))

        with interactive_worker(disconnect) as endpoint, StdioEnv(endpoint) as env:
            with self.assertRaisesRegex(RuntimeError, "Incomplete trace"):
                list(env.reset())
            self.assertIsNone(env.exit_code)

    def test_capture_error_is_not_client_failure_reward(self) -> None:
        def failed(connection: socket.socket) -> None:
            connection.sendall(frame(1, detail=1 | _FEATURE_STDIO) + frame(5, detail=1))

        with interactive_worker(failed) as endpoint, StdioEnv(endpoint) as env:
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                list(env.reset())
            self.assertIsNone(env.exit_code)

    def test_closed_environment_cannot_restart(self) -> None:
        env = StdioEnv("tcp://127.0.0.1:9000")
        env.close()
        env.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            env.__enter__()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            env.reset()

    def test_reset_reports_connection_failure(self) -> None:
        with StdioEnv("tcp://127.0.0.1:9000") as env, \
                mock.patch("cpu2tensor.stdio.socket.create_connection", side_effect=OSError("offline")), \
                self.assertRaisesRegex(ConnectionError, "connection failed"):
            env.reset()

    @unittest.skipUnless(importlib.util.find_spec("gymnasium"), "Gymnasium is optional")
    def test_gym_reduction_client_reward_and_termination(self) -> None:
        import gymnasium as gym

        def observe(batches: Iterator) -> int:
            return sum(batch.addresses.numel() for batch in batches)

        with interactive_worker(episode) as endpoint, StdioEnv(endpoint) as env:
            wrapper = env.as_gym(
                observe=observe,
                encode=lambda action: f"{action}\n".encode(),
                reward=lambda stream: 1.0 if stream.exit_code == 0 else 0.0,
                observation_space=gym.spaces.Discrete(10),
                action_space=gym.spaces.Discrete(10),
            )
            observation, info = wrapper.reset()
            self.assertEqual(observation, 1)
            self.assertTrue(info["needs_input"])
            result = wrapper.step(3)
            self.assertEqual(result, (1, 1.0, True, False, {"needs_input": False, "exit_code": 0}))
            with self.assertRaisesRegex(RuntimeError, "ended"):
                wrapper.step(3)

    @unittest.skipUnless(importlib.util.find_spec("gymnasium"), "Gymnasium is optional")
    def test_gym_reducer_must_drain_observations(self) -> None:
        import gymnasium as gym

        def cancelled(connection: socket.socket) -> None:
            prompt(connection)
            self.assertEqual(connection.recv(1), b"")

        with interactive_worker(cancelled) as endpoint, StdioEnv(endpoint) as env:
            wrapper = env.as_gym(
                observe=lambda batches: torch.zeros(1),
                encode=lambda action: b"3\n",
                reward=lambda stream: 0.0,
                observation_space=gym.spaces.Discrete(10),
                action_space=gym.spaces.Discrete(10),
            )
            with self.assertRaisesRegex(RuntimeError, "consume every batch"):
                wrapper.reset()

    @unittest.skipUnless(importlib.util.find_spec("gymnasium"), "Gymnasium is optional")
    def test_gym_client_truncation_cancels_waiting_target(self) -> None:
        import gymnasium as gym

        def two_requests(connection: socket.socket) -> None:
            prompt(connection)
            self.assertEqual(receive(connection, 6), b"\x02\0\0\x003\n")
            connection.sendall(frame(9, sequence=1, detail=2))
            self.assertEqual(connection.recv(1), b"")

        with interactive_worker(two_requests) as endpoint, StdioEnv(endpoint) as env:
            wrapper = env.as_gym(
                observe=lambda batches: sum(batch.addresses.numel() for batch in batches),
                encode=lambda action: b"3\n",
                reward=lambda stream: -0.25,
                episode_end=lambda stream: (False, True),
                observation_space=gym.spaces.Discrete(10),
                action_space=gym.spaces.Discrete(10),
            )
            wrapper.reset()
            observation, reward, terminated, truncated, _ = wrapper.step(3)
            self.assertEqual((observation, reward, terminated, truncated), (0, -0.25, False, True))
            self.assertFalse(env.needs_input)

    @unittest.skipUnless(importlib.util.find_spec("gymnasium"), "Gymnasium is optional")
    def test_gym_rejects_seed_it_cannot_forward_to_target(self) -> None:
        import gymnasium as gym

        with StdioEnv("tcp://127.0.0.1:9000") as env:
            wrapper = env.as_gym(
                observe=lambda batches: 0,
                encode=lambda action: b"3\n",
                reward=lambda stream: 0.0,
                observation_space=gym.spaces.Discrete(10),
                action_space=gym.spaces.Discrete(10),
            )
            with self.assertRaisesRegex(ValueError, "no target seed"):
                wrapper.reset(seed=19)


if __name__ == "__main__":
    unittest.main()
