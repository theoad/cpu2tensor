# SPDX-License-Identifier: AGPL-3.0-only
"""Read bounded trace batches from an operator-started worker."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import socket
import struct
import threading
from types import MappingProxyType, TracebackType
from typing import cast
from urllib.parse import urlsplit

import torch

from cpu2tensor import _native
from cpu2tensor._batching import BatchCollator, MAX_BATCH_BYTES
from cpu2tensor._device import to_device
from cpu2tensor.batch import (
    AddressContext, Batch, BlockTransitions, ExecutableLayout, MemoryAccesses,
    RegisterChanges, TransitionWindow,
)


_HEADER_BYTES = 32
_BLOCKS = 2
_COMPLETE = 4
_ERROR = 5
_REGISTER_SCHEMA = 6
_REGISTERS = 7
_MEMORY = 8
_ADDRESS_CONTEXT = 12
_EXECUTABLE_LAYOUT = 13
_MIXED = 14
_BLOCK_TRANSITIONS = 15
_TRANSITION_WINDOW = 16
_WINDOW = struct.Struct("<IIIIQQQ")
_WINDOW_STATUS = {1: "ended", 2: "aborted", 3: "incomplete"}
_FAILURES = {
    1: "capture failed",
    2: "target was killed",
    3: "target behavior is not supported",
    4: "worker transport failed",
}


class Pool:
    """A synchronous stream from one or more worker endpoints.

    Each ``tcp://host:port`` endpoint runs the worker's prescribed target.
    ``batch.worker`` is its index in the endpoint list; source and sequence
    remain local to that worker. Multiple workers progress independently through
    bounded reader queues. Read once, preferably inside a ``with`` block so an
    early loop exit closes every connection and cancels unfinished runs.

    The timeout applies to socket operations, including connection attempts and
    waiting for batches. Closing interrupts published sockets. Hostname lookup
    uses the operating system resolver, which Python cannot interrupt or bound
    with this socket timeout; close may wait for an in-progress DNS lookup.

    ``batch_bytes=0`` preserves worker frame boundaries. A positive value groups
    CPU columns per source before upload, adding latency and CPU concatenation
    copies. Pending decoded columns are limited to twice this target plus one
    incoming frame; merged outputs, sequence/padding columns and Python objects
    need additional memory. Source ends flush immediately. Retaining a device
    column retains its whole shared batch upload. Incomplete traces still raise.
    """

    def __init__(
        self,
        endpoints: Sequence[str],
        *,
        device: str = "cpu",
        timeout: float = 30.0,
        batch_bytes: int = 0,
    ) -> None:
        self._readers = None
        self._closed = False
        if not isinstance(batch_bytes, int) or not 0 <= batch_bytes <= MAX_BATCH_BYTES:
            raise ValueError("batch_bytes must be an integer between 0 and 4 MiB")
        self._batch_bytes = batch_bytes
        if isinstance(endpoints, str) or not endpoints:
            raise ValueError("Pool requires a nonempty list of endpoints")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("Pool endpoints must be distinct")
        if len(endpoints) > 1:
            from cpu2tensor._multipool import EndpointReaders
            self._readers = EndpointReaders(endpoints, device, timeout, batch_bytes=batch_bytes)
            return
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
        self._column_device = torch.device("cpu") if batch_bytes else self._device
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
        if self._readers is not None:
            self._closed = True
            self._readers.close()
            return
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
        if self._readers is not None:
            return self._readers.read()
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

    def _connect(self) -> socket.socket:
        # Resolve first, then expose each socket before connect can block. This
        # lets close cancel a connection attempt as well as an established read.
        addresses = socket.getaddrinfo(*self._address, type=socket.SOCK_STREAM)
        last_error: OSError | None = None
        for family, kind, protocol, _, address in addresses:
            connection: socket.socket | None = None
            try:
                connection = socket.socket(family, kind, protocol)
                connection.settimeout(self._timeout)
                with self._state_lock:
                    if self._closed:
                        raise RuntimeError("Pool was closed while connecting")
                    self._socket = connection
                connection.connect(address)
                with self._state_lock:
                    if self._closed:
                        raise RuntimeError("Pool was closed while connecting")
                return connection
            except BaseException as error:
                with self._state_lock:
                    if self._socket is connection:
                        self._socket = None
                    closed = self._closed
                if connection is not None:
                    connection.close()
                if closed:
                    raise RuntimeError("Pool was closed while connecting") from error
                if not isinstance(error, OSError):
                    raise
                last_error = error
        if last_error is not None:
            raise last_error
        raise OSError("Worker address resolved to no stream sockets")

    def _read(self) -> Iterator[Batch]:
        try:
            if self._closed:
                raise RuntimeError("Pool was closed before reading started")
            connection = self._connect()
            stream = _native.new_stream()
            names: dict[int, Mapping[int, str]] = {}
            collator = BatchCollator(self._batch_bytes) if self._batch_bytes else None
            while True:
                header = self._receive(connection, _HEADER_BYTES)
                payload_bytes = _native.payload_size(header)
                frame = header + self._receive(connection, payload_bytes)
                kind, source, count, sequence, detail, payload = _native.decode(stream, frame)
                if kind == 1 and detail & 2048:
                    raise ValueError("Use StdioEnv for an interactive worker")
                if kind == 1 and detail & (1 << 13):
                    raise ValueError("Use KernelEnv for an interactive system worker")
                if kind == _REGISTER_SCHEMA:
                    # Schema updates are cold-path work. Retained batches see an
                    # immutable snapshot, not a dictionary changed by later reads.
                    update = cast(dict[int, str], payload)
                    names[source] = MappingProxyType({**names.get(source, {}), **update})
                elif kind in (_BLOCKS, _REGISTERS, _MEMORY, _ADDRESS_CONTEXT,
                              _EXECUTABLE_LAYOUT, _MIXED, _BLOCK_TRANSITIONS,
                              _TRANSITION_WINDOW):
                    batch = self._batch(kind, source, count, sequence, payload, names.get(source))
                    if collator is None:
                        yield batch
                    else:
                        for ready in collator.add(batch):
                            yield to_device(ready, self._device)
                            del ready
                    del batch
                elif kind == 3 and collator is not None:
                    for ready in collator.end_source(source):
                        yield to_device(ready, self._device)
                        del ready
                elif kind == _COMPLETE:
                    if collator is not None:
                        for ready in collator.finish():
                            yield to_device(ready, self._device)
                            del ready
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
        # CPU tensors retain their decoded column storage without a copy. Once
        # all columns exist, accelerator batches use one shared staging upload.
        return torch.frombuffer(data, dtype=dtype)

    def _batch(
        self,
        kind: int,
        source: int,
        count: int,
        sequence: int,
        payload: (
            bytearray | dict[int, str] | _native.RegisterColumns |
            _native.MemoryColumns | _native.TransitionColumns
        ),
        names: Mapping[int, str] | None,
    ) -> Batch:
        registers = None
        memory = None
        context = None
        if kind == _BLOCK_TRANSITIONS:
            columns = cast("_native.TransitionColumns", payload)
            transitions = BlockTransitions(
                sequence,
                **{name: self._tensor(data, torch.int64) for name, data in columns.items()},
            )
            return to_device(Batch(source, None, torch.empty(0, dtype=torch.int64),
                                   transitions=transitions), self._column_device)
        if kind == _TRANSITION_WINDOW:
            status, sources, capacity, reserved, distinct, observed, overflow = (
                _WINDOW.unpack(cast(bytearray, payload))
            )
            assert reserved == 0
            window = TransitionWindow(sequence, _WINDOW_STATUS[status], sources, capacity,
                                      distinct, observed, overflow)
            return to_device(
                Batch(None, None, torch.empty(0, dtype=torch.int64),
                      transition_window=window),
                self._column_device,
            )
        if kind == _MIXED:
            columns = cast(dict, payload)
            addresses = (self._tensor(columns['blocks'], torch.int64) if 'blocks' in columns else
                         torch.empty(0, dtype=torch.int64, device="cpu"))
            block_sequences = (self._tensor(columns['block_sequences'], torch.int64)
                               if 'blocks' in columns else None)
            if 'context' in columns:
                context = AddressContext(**{name: self._tensor(data, torch.int64)
                                            for name, data in columns['context'].items()})
            if 'registers' in columns:
                data = columns['registers']
                registers = RegisterChanges(
                    **{name: self._tensor(data[name], torch.int64)
                       for name in ('pc', 'ids', 'widths', 'flags', 'sequences')},
                    values=self._tensor(data['values'], torch.uint8).reshape(-1, data['value_width']),
                    names=names,
                )
            if 'memory' in columns:
                data = columns['memory']
                memory = MemoryAccesses(
                    **{name: None if value is None else self._tensor(value, torch.int64)
                       for name, value in data.items() if name not in ('values', 'value_width')},
                    values=None if data['values'] is None else self._tensor(data['values'], torch.uint8).reshape(-1, 16),
                )
            return to_device(Batch(source, sequence, addresses, registers, memory, context,
                                   block_sequences=block_sequences), self._column_device)
        if kind == _EXECUTABLE_LAYOUT:
            layout = ExecutableLayout(self._tensor(cast(bytearray, payload), torch.int64))
            return to_device(Batch(None, None, torch.empty(0, dtype=torch.int64, device="cpu"), layout=layout), self._column_device)
        if kind == _BLOCKS:
            addresses = self._tensor(cast(bytearray, payload), torch.int64)
        else:
            addresses = torch.empty(0, dtype=torch.int64, device="cpu")
            if kind == _ADDRESS_CONTEXT:
                context = AddressContext(**{
                    name: self._tensor(data, torch.int64)
                    for name, data in cast(dict[str, bytearray], payload).items()
                })
            elif kind == _REGISTERS:
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
                    physical_addresses=(None if accesses["physical_addresses"] is None else
                                        self._tensor(accesses["physical_addresses"], torch.int64)),
                    mapped_sizes=(None if accesses["mapped_sizes"] is None else
                                  self._tensor(accesses["mapped_sizes"], torch.int64)),
                    mapping_flags=(None if accesses["mapping_flags"] is None else
                                   self._tensor(accesses["mapping_flags"], torch.int64)),
                    context_sequences=(None if accesses["context_sequences"] is None else
                                       self._tensor(accesses["context_sequences"], torch.int64)),
                    values=(None if accesses["values"] is None else
                            self._tensor(accesses["values"], torch.uint8).reshape(count, 16)),
                )
        return to_device(Batch(source, sequence, addresses, registers, memory, context), self._column_device)
