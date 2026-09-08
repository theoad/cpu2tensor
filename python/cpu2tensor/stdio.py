# SPDX-License-Identifier: AGPL-3.0-only
"""Read observations and send bytes at a target's stdin requests."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
import socket
import struct
from types import MappingProxyType, TracebackType
from typing import Any, cast

from cpu2tensor import _native
from cpu2tensor.batch import Batch
from cpu2tensor.pool import Pool


_HEADER_BYTES = 32
_HELLO = 1
_BLOCKS = 2
_COMPLETE = 4
_ERROR = 5
_REGISTER_SCHEMA = 6
_REGISTERS = 7
_MEMORY = 8
_INPUT_REQUEST = 9
_FEATURE_STDIO = 1 << 11
_FAILURES = {
    1: "capture failed",
    2: "target was killed",
    3: "target behavior is not supported",
    4: "worker transport failed",
}


class StdioEnv:
    """A synchronous trace stream paused at actual stdin read calls.

    Fully consume each reset or step iterator before taking another action.
    Reduction happens in the client as batches arrive; this class never keeps
    a history of observations. Reset closes the previous run and reconnects to
    an operator-started interactive worker, which must permit another episode.
    """

    _feature = _FEATURE_STDIO
    _request_kind = _INPUT_REQUEST

    def __init__(
        self,
        endpoint: str,
        *,
        device: str = "cpu",
        timeout: float = 30.0,
    ) -> None:
        # Pool owns the existing endpoint validation and tensor conversion.
        self._converter = Pool([endpoint], device=device, timeout=timeout)
        self._connection: socket.socket | None = None
        self._stream: Any = None
        self._names: dict[int, Mapping[int, str]] = {}
        self._reading = False
        self._closed = False
        self._run = 0
        self.needs_input = False
        self.max_action_bytes = 0
        self.exit_code: int | None = None

    def __enter__(self) -> "StdioEnv":
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _cancel_run(self) -> None:
        self._run += 1
        connection = self._connection
        self._connection = None
        self._reading = False
        self.needs_input = False
        self.max_action_bytes = 0
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def close(self) -> None:
        """Cancel the current run and release its connection."""
        self._closed = True
        self._cancel_run()
        self._converter.close()

    def reset(self) -> Iterator[Batch]:
        """Restart, then yield observations until input is needed or the run ends."""
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        self._cancel_run()
        self.exit_code = None
        self._clear_events()
        self._stream = _native.new_stream()
        self._names = {}
        try:
            self._connection = socket.create_connection(
                self._converter._address, timeout=self._converter._timeout
            )
        except TimeoutError as error:
            raise TimeoutError(f"{type(self).__name__} connection timed out while starting a run") from error
        except OSError as error:
            raise ConnectionError(f"{type(self).__name__} connection failed while starting a run") from error
        self._reading = True
        return self._read(self._run, first=True)

    def step(self, action: bytes) -> Iterator[Batch]:
        """Send bytes, then yield observations until the next input request or exit.

        Bytes are passed unchanged. Include any newline the target expects.
        A nonzero target exit code is an episode result, not a capture failure.
        """
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        if self._reading:
            raise RuntimeError("Consume all observations before taking an action")
        if not self.needs_input or self._connection is None:
            raise RuntimeError("The target is not waiting for input; call reset first")
        if not isinstance(action, bytes):
            raise TypeError("Stdio actions must be bytes")
        if not 1 <= len(action) <= self.max_action_bytes:
            raise ValueError(f"Action must contain 1 to {self.max_action_bytes} bytes")
        self._clear_events()
        self.needs_input = False
        self.max_action_bytes = 0
        try:
            self._connection.sendall(struct.pack("<I", len(action)) + action)
        except TimeoutError as error:
            self._cancel_run()
            raise TimeoutError(f"{type(self).__name__} connection timed out while sending an action") from error
        except OSError as error:
            self._cancel_run()
            raise ConnectionError(f"{type(self).__name__} connection failed while sending an action") from error
        self._reading = True
        return self._read(self._run, first=False)

    def _read(self, run: int, *, first: bool) -> Iterator[Batch]:
        reached_boundary = False
        try:
            if run != self._run or self._connection is None:
                raise RuntimeError("This observation iterator belongs to a cancelled run")
            connection = self._connection
            while True:
                if run != self._run:
                    raise RuntimeError("This observation iterator belongs to a cancelled run")
                header = self._converter._receive(connection, _HEADER_BYTES)
                payload_size = _native.payload_size(header)
                frame = header + self._converter._receive(connection, payload_size)
                kind, source, count, sequence, detail, payload = _native.decode(self._stream, frame)
                if first:
                    if kind != _HELLO or not detail & self._feature:
                        raise RuntimeError(f"{type(self).__name__} requires a matching interactive worker")
                    first = False
                if kind == _REGISTER_SCHEMA:
                    update = cast(dict[int, str], payload)
                    self._names[source] = MappingProxyType({**self._names.get(source, {}), **update})
                elif kind in (_BLOCKS, _REGISTERS, _MEMORY, 12, 13, 14):
                    yield self._converter._batch(
                        kind, source, count, sequence, payload, self._names.get(source)
                    )
                elif kind == self._request_kind:
                    self.needs_input = True
                    self.max_action_bytes = detail
                    reached_boundary = True
                    return
                elif kind == _COMPLETE:
                    self.exit_code = detail
                    return
                elif kind == _ERROR:
                    raise RuntimeError(f"Incomplete trace: {_FAILURES[detail]}")
                elif kind == 11:
                    self._guest_event(cast(bytearray, payload))
        except TimeoutError as error:
            raise TimeoutError(f"{type(self).__name__} connection timed out before an input request or exit") from error
        except OSError as error:
            raise ConnectionError(f"{type(self).__name__} connection failed before an input request or exit") from error
        finally:
            if run == self._run:
                self._reading = False
                if not reached_boundary:
                    self._cancel_run()

    def _clear_events(self) -> None:
        pass

    def _guest_event(self, payload: bytearray) -> None:
        raise RuntimeError("Unexpected guest adapter event")

    def _info(self) -> dict[str, Any]:
        return {"needs_input": self.needs_input, "exit_code": self.exit_code}

    def as_gym(
        self,
        *,
        observe: Callable[[Iterator[Batch]], Any],
        encode: Callable[[Any], bytes],
        reward: Callable[["StdioEnv"], float],
        observation_space: Any,
        action_space: Any,
        episode_end: Callable[["StdioEnv"], tuple[bool, bool]] | None = None,
    ) -> Any:
        """Wrap the stream with Gymnasium's reset and step return conventions.

        The client supplies spaces, a streaming observation reducer, action
        encoding, and reward. By default target exit terminates an episode;
        episode_end can supply the client's termination and truncation rules.
        Transport/capture failures remain exceptions, never successful episodes.
        """
        try:
            import gymnasium as gym
        except ImportError as error:
            raise ImportError("Install cpu2tensor[gym] to use the Gymnasium wrapper") from error
        stream = self

        class StdioGymEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self) -> None:
                self.observation_space = observation_space
                self.action_space = action_space
                self._ended = True

            def _observe(self, batches: Iterator[Batch]) -> Any:
                try:
                    observation = observe(batches)
                    if stream._reading:
                        raise RuntimeError("The observation reducer must consume every batch")
                    return observation
                except BaseException:
                    stream._cancel_run()
                    raise

            def _info(self) -> dict[str, Any]:
                return stream._info()

            def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
                if seed is not None or options:
                    raise ValueError("This worker has no target seed or reset options adapter")
                super().reset(seed=seed)
                observation = self._observe(stream.reset())
                self._ended = stream.exit_code is not None
                return observation, self._info()

            def step(self, action: Any) -> tuple:
                if self._ended:
                    raise RuntimeError("The episode has ended; call reset before step")
                if not self.action_space.contains(action):
                    raise ValueError("Action is outside the client action space")
                observation = self._observe(stream.step(encode(action)))
                try:
                    result = float(reward(stream))
                    terminated, truncated = ((stream.exit_code is not None, False)
                                             if episode_end is None else episode_end(stream))
                    if stream.exit_code is not None and not (terminated or truncated):
                        raise ValueError("An exited target must terminate or truncate its episode")
                except BaseException:
                    stream._cancel_run()
                    raise
                self._ended = bool(terminated or truncated)
                info = self._info()
                if self._ended:
                    stream._cancel_run()
                return observation, result, bool(terminated), bool(truncated), info

            def close(self) -> None:
                stream.close()

        return StdioGymEnv()
