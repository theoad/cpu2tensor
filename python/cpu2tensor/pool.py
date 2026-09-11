# SPDX-License-Identifier: AGPL-3.0-only
"""Read bounded trace batches from an operator-started worker."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
import select
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
from cpu2tensor.terminal import (
    BoundaryProgress, TerminalOutcome, TerminalReason, TraceConnectionError,
    TraceTerminalError, TraceTimeoutError, TransportEnd,
)


_HEADER_BYTES = 32
_CONTEXT_GROUP_FRAMES = 32
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
_TERMINAL_REPORT = 17
_TERMINAL_REPORT_VERSION = 1
_TERMINAL_HELLO = 1 << 0
_TERMINAL_DATA = 1 << 1
_TERMINAL_START_CONFIGURED = 1 << 2
_TERMINAL_START_OBSERVED = 1 << 3
_TERMINAL_STOP_CONFIGURED = 1 << 4
_TERMINAL_STOP_OBSERVED = 1 << 5
_WINDOW_FEATURE = 1 << 14
_MEMORY_FEATURE = 1 << 8
_REGISTERS_FEATURE = 1 << 9
_CONTEXT_FEATURE = 1 << 16
_MIXED_FEATURE = 1 << 18
_STOP_FEATURE = 1 << 19
_DATA_KINDS = {
    _BLOCKS, _REGISTERS, _MEMORY, _ADDRESS_CONTEXT, _EXECUTABLE_LAYOUT,
    _MIXED, _BLOCK_TRANSITIONS, _TRANSITION_WINDOW,
}
_WINDOW = struct.Struct("<IIIIQQQ")
_WINDOW_STATUS = {1: "ended", 2: "aborted", 3: "incomplete"}
_FAILURES = {
    1: "capture failed",
    2: "target was killed",
    3: "target behavior is not supported",
    4: "worker transport failed",
}
_FAILURE_REASONS = {
    1: TerminalReason.CAPTURE_FAILURE,
    2: TerminalReason.TARGET_KILLED,
    3: TerminalReason.UNSUPPORTED_TARGET,
    4: TerminalReason.WORKER_TRANSPORT_FAILURE,
}


class _CleanEnd(Exception):
    def __init__(self, truncated: bool) -> None:
        self.truncated = truncated


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

    ``batch_bytes=0`` normally preserves worker frame boundaries. Context-only
    mixed capture opportunistically combines at most 32 complete buffered frames
    per source before making tensors. A positive value groups CPU columns further
    before upload, adding latency and CPU concatenation copies. Pending decoded
    columns are limited to twice this target plus one bounded native group of at
    most 32 frames; merged outputs, sequence/padding columns and Python objects
    need additional memory.
    Source ends flush immediately. Retaining a device column retains its whole
    shared batch upload. Incomplete traces still raise.
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
        self._endpoints = tuple(endpoints)
        self._outcome: TerminalOutcome | None = None
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
        self._endpoint = endpoints[0]
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

    @property
    def outcomes(self) -> tuple[TerminalOutcome | None, ...]:
        """The outcome of each endpoint, or ``None`` while it is still running."""
        if self._readers is not None:
            return self._readers.outcomes
        return (self._outcome,)

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

    def _receive(
        self,
        connection: socket.socket,
        size: int,
        *,
        classify_end: bool = False,
        frame_started: bool = False,
    ) -> bytearray:
        data = bytearray(size)
        view = memoryview(data)
        received = 0
        while received < size:
            count = connection.recv_into(view[received:])
            if count == 0:
                if classify_end:
                    raise _CleanEnd(frame_started or received != 0)
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

    def _receive_frame(self, connection: socket.socket) -> bytearray:
        header = self._receive(connection, _HEADER_BYTES, classify_end=True)
        payload_bytes = _native.payload_size(header)
        return header + self._receive(
            connection, payload_bytes, classify_end=True, frame_started=True,
        )

    @staticmethod
    def _complete_frame_is_ready(connection: socket.socket) -> bool:
        # MSG_PEEK leaves the next frame in the socket. A readable socket may
        # hold only one byte, so inspect both the header and its complete bounded
        # payload before read-ahead. The caller remains the socket's only reader.
        if not select.select((connection,), (), (), 0)[0]:
            return False
        header = connection.recv(_HEADER_BYTES, socket.MSG_PEEK)
        if len(header) < _HEADER_BYTES:
            return False
        frame_bytes = _HEADER_BYTES + _native.payload_size(header)
        return len(connection.recv(frame_bytes, socket.MSG_PEEK)) == frame_bytes

    def _read(self) -> Iterator[Batch]:
        hello_received = False
        data_received = False
        start_configured: bool | None = None
        stop_configured: bool | None = None
        connected = False
        try:
            if self._closed:
                raise RuntimeError("Pool was closed before reading started")
            connection = self._connect()
            connected = True
            stream = _native.new_stream()
            names: dict[int, Mapping[int, str]] = {}
            collator = BatchCollator(self._batch_bytes) if self._batch_bytes else None
            context_grouping = False
            pending_frame: bytearray | None = None
            while True:
                frame = pending_frame if pending_frame is not None else self._receive_frame(connection)
                pending_frame = None
                deferred_error: BaseException | None = None
                validation_error: str | None = None
                frame_kind = struct.unpack_from("<H", frame, 6)[0]
                if context_grouping and frame_kind == _MIXED:
                    frames = [frame]
                    while len(frames) < _CONTEXT_GROUP_FRAMES:
                        try:
                            if not self._complete_frame_is_ready(connection):
                                break
                            following = self._receive_frame(connection)
                        except (OSError, RuntimeError, ValueError, _CleanEnd) as error:
                            deferred_error = error
                            break
                        if struct.unpack_from("<H", following, 6)[0] != _MIXED:
                            pending_frame = following
                            break
                        frames.append(following)
                    decoded, validation_error = _native.decode_context_frames(stream, frames)
                else:
                    decoded = [_native.decode(stream, frame)]

                for kind, source, count, sequence, detail, payload in decoded:
                    if kind == 1:
                        hello_received = True
                        start_configured = bool(detail & _WINDOW_FEATURE)
                        stop_configured = bool(detail & _STOP_FEATURE)
                        context_grouping = bool(detail & _CONTEXT_FEATURE and
                                                detail & _MIXED_FEATURE and
                                                not detail & (_MEMORY_FEATURE | _REGISTERS_FEATURE))
                    elif kind in _DATA_KINDS:
                        data_received = True
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
                        self._outcome = self._make_outcome(
                            TerminalReason.COMPLETE if detail == 0 else TerminalReason.TARGET_EXIT,
                            hello_received, data_received,
                            start=self._completed_boundary(start_configured),
                            stop=self._completed_boundary(stop_configured),
                            complete=True,
                        )
                        if detail != 0:
                            raise TraceTerminalError(f"Target exited with code {detail}", self._outcome)
                        return
                    elif kind == _ERROR:
                        self._outcome = self._make_outcome(
                            _FAILURE_REASONS[detail], hello_received, data_received,
                        )
                        raise TraceTerminalError(
                            f"Incomplete trace: {_FAILURES[detail]}", self._outcome,
                        )
                    elif kind == _TERMINAL_REPORT:
                        self._raise_reported_terminal(
                            connection, detail, hello_received, data_received,
                            start_configured, stop_configured,
                        )
                if validation_error is not None:
                    raise ValueError(validation_error)
                if deferred_error is not None:
                    raise deferred_error
        except _CleanEnd as error:
            reason = (TerminalReason.TRUNCATED_STREAM if error.truncated
                      else TerminalReason.UNKNOWN_DISCONNECTION)
            self._outcome = self._make_outcome(
                reason, hello_received, data_received,
                transport=TransportEnd.CLEAN,
            )
            raise TraceTerminalError(
                ("Incomplete trace: worker disconnected inside a frame" if error.truncated
                 else "Incomplete trace: worker disconnected before completion"),
                self._outcome,
            ) from error
        except TraceTerminalError:
            raise
        except TimeoutError as error:
            self._outcome = self._make_outcome(
                TerminalReason.CLIENT_TIMEOUT, hello_received, data_received,
                transport=TransportEnd.TIMEOUT,
            )
            raise TraceTimeoutError(
                "Trace connection timed out before completion", self._outcome,
            ) from error
        except OSError as error:
            transport = TransportEnd.RESET if isinstance(error, ConnectionResetError) else TransportEnd.ERROR
            self._outcome = self._make_outcome(
                (TerminalReason.UNKNOWN_DISCONNECTION if connected
                 else TerminalReason.CONNECTION_FAILURE),
                hello_received, data_received,
                transport=transport,
            )
            raise TraceConnectionError(
                "Trace connection failed before completion", self._outcome,
            ) from error
        finally:
            self.close()

    def _make_outcome(
        self,
        reason: TerminalReason,
        hello_received: bool,
        data_received: bool,
        *,
        hello_reported: bool | None = None,
        data_reported: bool | None = None,
        start: BoundaryProgress = BoundaryProgress.UNKNOWN,
        stop: BoundaryProgress = BoundaryProgress.UNKNOWN,
        transport: TransportEnd = TransportEnd.NOT_OBSERVED,
        complete: bool = False,
    ) -> TerminalOutcome:
        return TerminalOutcome(
            reason=reason, endpoint=self._endpoint, worker=0, source=None,
            hello_received=hello_received, data_received=data_received,
            hello_reported=hello_reported, data_reported=data_reported,
            start=start, stop=stop, transport=transport, complete=complete,
        )

    @staticmethod
    def _completed_boundary(configured: bool | None) -> BoundaryProgress:
        if configured is None:
            return BoundaryProgress.UNKNOWN
        return BoundaryProgress.OBSERVED if configured else BoundaryProgress.NOT_CONFIGURED

    @staticmethod
    def _reported_boundary(flags: int, configured: int, observed: int) -> BoundaryProgress:
        if not flags & configured:
            return BoundaryProgress.NOT_CONFIGURED
        return BoundaryProgress.OBSERVED if flags & observed else BoundaryProgress.NOT_OBSERVED

    def _raise_reported_terminal(
        self,
        connection: socket.socket,
        detail: int,
        hello_received: bool,
        data_received: bool,
        start_configured: bool | None,
        stop_configured: bool | None,
    ) -> None:
        version = detail & 255
        reason_code = (detail >> 8) & 255
        flags = (detail >> 16) & 255
        if version != _TERMINAL_REPORT_VERSION or reason_code != 1:
            raise ValueError("Unknown terminal report version or reason")
        hello_reported = bool(flags & _TERMINAL_HELLO)
        data_reported = bool(flags & _TERMINAL_DATA)
        if hello_received != hello_reported or data_received != data_reported:
            raise ValueError("Terminal report contradicts received trace progress")
        if hello_received and (
            start_configured != bool(flags & _TERMINAL_START_CONFIGURED)
            or stop_configured != bool(flags & _TERMINAL_STOP_CONFIGURED)
        ):
            raise ValueError("Terminal report contradicts Hello boundary configuration")
        outcome = self._make_outcome(
            TerminalReason.MAX_RUN_DEADLINE, hello_received, data_received,
            hello_reported=hello_reported, data_reported=data_reported,
            start=self._reported_boundary(
                flags, _TERMINAL_START_CONFIGURED, _TERMINAL_START_OBSERVED,
            ),
            stop=self._reported_boundary(
                flags, _TERMINAL_STOP_CONFIGURED, _TERMINAL_STOP_OBSERVED,
            ),
        )
        try:
            trailing = connection.recv(1)
        except TimeoutError as error:
            self._outcome = replace(outcome, transport=TransportEnd.TIMEOUT)
            raise TraceTimeoutError(
                "Incomplete trace: target exceeded its max-run deadline", self._outcome,
            ) from error
        except OSError as error:
            transport = TransportEnd.RESET if isinstance(error, ConnectionResetError) else TransportEnd.ERROR
            self._outcome = replace(outcome, transport=transport)
            raise TraceConnectionError(
                "Incomplete trace: target exceeded its max-run deadline", self._outcome,
            ) from error
        if trailing:
            raise ValueError("Trace contains data after its terminal report")
        self._outcome = replace(outcome, transport=TransportEnd.CLEAN)
        raise TraceTerminalError(
            "Incomplete trace: target exceeded its max-run deadline", self._outcome,
        )

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
