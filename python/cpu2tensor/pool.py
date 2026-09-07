# SPDX-License-Identifier: AGPL-3.0-only
"""Read bounded trace batches from an operator-started worker."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import socket
import threading
from types import MappingProxyType, TracebackType
from typing import cast
from urllib.parse import urlsplit

import torch

from cpu2tensor import _native
from cpu2tensor.batch import Batch, MemoryAccesses, RegisterChanges


_HEADER_BYTES = 32
_BLOCKS = 2
_COMPLETE = 4
_ERROR = 5
_REGISTER_SCHEMA = 6
_REGISTERS = 7
_MEMORY = 8
_FAILURES = {
    1: "capture failed",
    2: "target was killed",
    3: "target behavior is not supported",
    4: "worker transport failed",
}


class Pool:
    """A synchronous stream from one worker endpoint.

    A pool currently accepts one ``tcp://host:port`` endpoint. Connecting
    starts the worker's prescribed target run. Read once, preferably inside a
    ``with`` block so an early loop exit closes the connection and cancels the run.
    The timeout applies to socket operations, including waiting for a batch.
    """

    def __init__(
        self,
        endpoints: Sequence[str],
        *,
        device: str = "cpu",
        timeout: float = 30.0,
    ) -> None:
        if isinstance(endpoints, str) or len(endpoints) != 1:
            raise ValueError("Pool currently requires a list with exactly one endpoint")
        endpoint = urlsplit(endpoints[0])
        if (
            endpoint.scheme != "tcp"
            or not endpoint.hostname
            or endpoint.port is None
            or not 1 <= endpoint.port <= 65535
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.path
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("Worker endpoint must be tcp://host:port")
        if timeout <= 0:
            raise ValueError("Socket timeout must be positive")
        self._address = (endpoint.hostname, endpoint.port)
        self._device = torch.device(device)
        if self._device.type not in ("cpu", "mps", "cuda"):
            raise ValueError("Pool supports CPU, MPS, and CUDA devices")
        if self._device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available in this Python environment")
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this Python environment")
        self._timeout = timeout
        self._socket: socket.socket | None = None
        self._started = False
        self._closed = False
        self._state_lock = threading.Lock()

    def __enter__(self) -> "Pool":
        if self._closed:
            raise RuntimeError("Pool is closed; create a new pool for another run")
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the connection; closing before completion cancels the run."""
        with self._state_lock:
            self._closed = True
            connection = self._socket
            self._socket = None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # A peer that has already closed still needs local cleanup.
            connection.close()

    def read(self) -> Iterator[Batch]:
        """Yield batches once, receiving only when the caller asks for the next."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Pool is closed; create a new pool for another run")
            if self._started:
                raise RuntimeError("Pool.read() can only be called once per run")
            self._started = True
        return self._read()

    def _receive(self, connection: socket.socket, size: int) -> bytearray:
        data = bytearray(size)
        view = memoryview(data)
        received = 0
        while received < size:
            count = connection.recv_into(view[received:])
            if count == 0:
                raise RuntimeError("Incomplete trace: worker disconnected before completion")
            received += count
        return data

    def _read(self) -> Iterator[Batch]:
        try:
            if self._closed:
                raise RuntimeError("Pool was closed before reading started")
            connection = socket.create_connection(self._address, timeout=self._timeout)
            with self._state_lock:
                if self._closed:
                    connection.close()
                    raise RuntimeError("Pool was closed while connecting")
                self._socket = connection
            stream = _native.new_stream()
            names: dict[int, Mapping[int, str]] = {}
            while True:
                header = self._receive(connection, _HEADER_BYTES)
                payload_bytes = _native.payload_size(header)
                frame = header + self._receive(connection, payload_bytes)
                kind, source, count, sequence, detail, payload = _native.decode(stream, frame)
                if kind == 1 and detail & 2048:
                    raise ValueError("Use StdioEnv for an interactive worker")
                if kind == _REGISTER_SCHEMA:
                    # Schema updates are cold-path work. Retained batches see an
                    # immutable snapshot, not a dictionary changed by later reads.
                    update = cast(dict[int, str], payload)
                    names[source] = MappingProxyType({**names.get(source, {}), **update})
                elif kind in (_BLOCKS, _REGISTERS, _MEMORY):
                    yield self._batch(kind, source, count, sequence, payload, names.get(source))
                elif kind == _COMPLETE:
                    if detail != 0:
                        raise RuntimeError(f"Target exited with code {detail}")
                    return
                elif kind == _ERROR:
                    raise RuntimeError(f"Incomplete trace: {_FAILURES[detail]}")
        except TimeoutError as error:
            raise TimeoutError("Trace connection timed out before completion") from error
        except OSError as error:
            raise ConnectionError("Trace connection failed before completion") from error
        finally:
            self.close()

    def _tensor(self, data: bytearray, dtype: torch.dtype) -> torch.Tensor:
        # Each decoded column has fresh storage. frombuffer keeps it alive, and
        # synchronous device copies finish before the batch can leave this call.
        return torch.frombuffer(data, dtype=dtype).to(self._device, non_blocking=False)

    def _batch(
        self,
        kind: int,
        source: int,
        count: int,
        sequence: int,
        payload: bytearray | dict[int, str] | _native.RegisterColumns | _native.MemoryColumns,
        names: Mapping[int, str] | None,
    ) -> Batch:
        registers = None
        memory = None
        if kind == _BLOCKS:
            addresses = self._tensor(cast(bytearray, payload), torch.int64)
        else:
            addresses = torch.empty(0, dtype=torch.int64, device=self._device)
            if kind == _REGISTERS:
                columns = cast("_native.RegisterColumns", payload)
                assert names is not None  # The native decoder requires a schema.
                registers = RegisterChanges(
                    pc=self._tensor(columns["pc"], torch.int64),
                    ids=self._tensor(columns["ids"], torch.int64),
                    widths=self._tensor(columns["widths"], torch.int64),
                    flags=self._tensor(columns["flags"], torch.int64),
                    values=self._tensor(columns["values"], torch.uint8).reshape(
                        count, columns["value_width"]
                    ),
                    names=names,
                )
            else:
                accesses = cast("_native.MemoryColumns", payload)
                memory = MemoryAccesses(
                    pc=self._tensor(accesses["pc"], torch.int64),
                    addresses=self._tensor(accesses["addresses"], torch.int64),
                    sizes=self._tensor(accesses["sizes"], torch.int64),
                    flags=self._tensor(accesses["flags"], torch.int64),
                    values=(None if accesses["values"] is None else
                            self._tensor(accesses["values"], torch.uint8).reshape(count, 16)),
                )
        if self._device.type == "mps":
            torch.mps.synchronize()
        return Batch(source, sequence, addresses, registers, memory)
