# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded Linux hardware captures with no file writes during measurement."""

from __future__ import annotations

from array import array
import ctypes
from dataclasses import dataclass
import mmap
import os
from pathlib import Path
import platform
import struct
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    import fcntl
except ImportError:  # Windows uses a separate ETW backend.
    fcntl = None


_SAMPLE = 9
_LOST = 2
_AUX = 11
_LOST_SAMPLES = 13
_SAMPLE_FIELDS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 7) | (1 << 8)
_MEMORY_FIELDS = _SAMPLE_FIELDS | (1 << 14) | (1 << 15)
_HEADER = struct.Struct("<IHH")
_SAMPLE_ROW = struct.Struct("<QIIQQIIQ")
_ENABLE = 0x2400
_DISABLE = 0x2401
_RESET = 0x2403
_DATA_HEADER_OFFSET = 1024
_AUX_BAD_FLAGS = 0x0F  # truncated, overwritten, partial, or collided


class HardwareCaptureError(RuntimeError):
    """A hardware source is unavailable or its capture cannot be trusted."""


class HardwareTraceLost(HardwareCaptureError):
    """A perf buffer overflowed; the result is not a complete capture."""


@dataclass(frozen=True)
class HardwareConfig:
    """One host Linux capture. Kernel mode samples the host kernel on selected CPUs.

    ``process`` observes threads of ``pid`` that exist when capture starts.
    Attach before releasing a stopped target to avoid a startup gap. ``kernel``
    samples all tasks executing kernel code on the selected host CPUs; it does
    not imply that a virtual machine's guest kernel is visible to the host PMU.
    """

    scope: Literal["process", "kernel"]
    signal: Literal["cycles", "instructions", "memory_loads", "intel_pt"] = "cycles"
    pid: int | None = None
    cpus: tuple[int, ...] | None = None
    period: int = 100_000
    data_pages: int = 64
    aux_pages: int = 64

    def __post_init__(self) -> None:
        if self.scope not in ("process", "kernel"):
            raise ValueError("scope must be process or kernel")
        if self.signal not in ("cycles", "instructions", "memory_loads", "intel_pt"):
            raise ValueError("Unknown hardware signal")
        if self.scope == "process" and (self.pid is None or self.pid <= 0 or self.cpus is not None):
            raise ValueError("Process capture needs a positive pid and no CPU list")
        if self.scope == "kernel" and (self.pid is not None or not self.cpus):
            raise ValueError("Kernel capture needs a nonempty CPU list and no pid")
        if self.cpus is not None and (len(set(self.cpus)) != len(self.cpus) or
                                      any(cpu < 0 for cpu in self.cpus)):
            raise ValueError("CPU numbers must be distinct and nonnegative")
        if self.period <= 0:
            raise ValueError("period must be positive")
        for name, pages in (("data_pages", self.data_pages), ("aux_pages", self.aux_pages)):
            if pages <= 0 or pages & (pages - 1):
                raise ValueError(f"{name} must be a positive power of two")


@dataclass(frozen=True)
class HardwareBatch:
    """Owned host samples or undecoded PT packets from one perf event.

    Every tensor is CPU resident. ``ip`` is sampled, not a complete instruction
    trace. ``address``, ``weight`` and ``data_source`` are meaningful only for
    ``memory_loads``. ``exact_ip`` reports the kernel's exact-IP sample flag.
    ``trace_bytes`` contains PT packets and is empty for sampled PMU events.
    """

    source: int
    signal: str
    ip: torch.Tensor
    pid: torch.Tensor
    tid: torch.Tensor
    time: torch.Tensor
    cpu: torch.Tensor
    period: torch.Tensor
    address: torch.Tensor
    weight: torch.Tensor
    data_source: torch.Tensor
    exact_ip: torch.Tensor
    trace_bytes: torch.Tensor


class _PerfAttr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32), ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64), ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64), ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64), ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32), ("config1", ctypes.c_uint64),
        ("config2", ctypes.c_uint64), ("branch_sample_type", ctypes.c_uint64),
        ("sample_regs_user", ctypes.c_uint64), ("sample_stack_user", ctypes.c_uint32),
        ("clockid", ctypes.c_int32), ("sample_regs_intr", ctypes.c_uint64),
        ("aux_watermark", ctypes.c_uint32), ("sample_max_stack", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16), ("aux_sample_size", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
    ]


def _syscall_number() -> int:
    numbers = {"x86_64": 298, "aarch64": 241}
    try:
        return numbers[platform.machine().lower()]
    except KeyError as error:
        raise HardwareCaptureError("perf capture supports Linux x86-64 and AArch64") from error


def _source_value(path: str) -> int:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError) as error:
        raise HardwareCaptureError(f"Required perf source is unavailable: {path}") from error


def _attribute(config: HardwareConfig) -> _PerfAttr:
    attr = _PerfAttr()
    attr.size = ctypes.sizeof(_PerfAttr)
    attr.sample_period = config.period
    attr.sample_type = _MEMORY_FIELDS if config.signal == "memory_loads" else _SAMPLE_FIELDS
    attr.flags = 1 | (1 << 6)  # disabled; exclude hypervisor
    if config.scope == "process":
        # A per-thread event with cpu=-1 cannot mmap a ring when inherit is set.
        attr.flags |= 1 << 5  # exclude kernel
    else:
        attr.flags |= 1 << 4  # exclude user code
    if config.signal in ("cycles", "instructions"):
        attr.type = 0  # PERF_TYPE_HARDWARE
        attr.config = 0 if config.signal == "cycles" else 1
    elif config.signal == "memory_loads":
        root = "/sys/bus/event_source/devices/cpu"
        attr.type = _source_value(root + "/type")
        # Intel's mem-loads alias must keep its event/umask/ldlat contract.
        # Other PMUs need their own reviewed encoding and decoder.
        try:
            alias = Path(root + "/events/mem-loads").read_text().strip()
        except OSError as error:
            raise HardwareCaptureError("Precise memory-load PMU is unavailable") from error
        if not alias.startswith("event=0xcd,umask=0x1,ldlat="):
            raise HardwareCaptureError("Unsupported precise memory-load PMU encoding")
        attr.config = 0x1CD
        attr.config1 = int(alias.split("ldlat=", 1)[1], 0)
        attr.flags |= 2 << 15  # request zero-skid precise IP (PEBS on supported Intel CPUs)
    else:
        root = "/sys/bus/event_source/devices/intel_pt"
        attr.type = _source_value(root + "/type")
        try:
            pt_format = Path(root + "/format/pt").read_text().strip()
        except OSError as error:
            raise HardwareCaptureError("Intel PT is unavailable") from error
        if pt_format != "config:0":
            raise HardwareCaptureError("Unsupported Intel PT enable bit")
        attr.config = 1
        attr.sample_type = 0
        attr.sample_period = 0
    return attr


def _open_event(attr: _PerfAttr, tid: int, cpu: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(_syscall_number(), ctypes.byref(attr), tid, cpu, -1, 0)
    if fd < 0:
        code = ctypes.get_errno()
        raise HardwareCaptureError(
            f"perf_event_open failed for tid={tid}, cpu={cpu}: {os.strerror(code)}"
        )
    return int(fd)


def _ring_bytes(mapping: mmap.mmap, head: int, tail: int, offset: int, size: int) -> bytes:
    if head < tail or head - tail > size:
        raise HardwareTraceLost("Perf ring wrapped before capture stopped")
    start = tail % size
    count = head - tail
    first = min(count, size - start)
    return mapping[offset + start:offset + start + first] + mapping[offset:offset + count - first]


def _tensor(values: array | bytearray) -> torch.Tensor:
    import torch

    if not values:
        return torch.empty(0, dtype=torch.uint8 if isinstance(values, bytearray) else torch.int64)
    return torch.frombuffer(values, dtype=torch.uint8 if isinstance(values, bytearray) else torch.int64)


def _decode_records(data: bytes, signal: str, source: int, trace: bytes) -> HardwareBatch:
    names = ("ip", "pid", "tid", "time", "cpu", "period", "address", "weight", "data_source", "exact_ip")
    columns = {name: array("q") for name in names}
    position = 0
    aux_position = 0
    while position < len(data):
        if len(data) - position < _HEADER.size:
            raise HardwareTraceLost("Perf ring ended inside a record header")
        kind, misc, size = _HEADER.unpack_from(data, position)
        if size < _HEADER.size or size > len(data) - position:
            raise HardwareTraceLost("Perf ring ended inside a record")
        record = data[position + _HEADER.size:position + size]
        if kind == _LOST or kind == _LOST_SAMPLES:
            raise HardwareTraceLost("Perf reported lost samples")
        if kind == _AUX:
            if len(record) < 24 or struct.unpack_from("<Q", record, 16)[0] & _AUX_BAD_FLAGS:
                raise HardwareTraceLost("Intel PT reported incomplete AUX data")
            offset, length = struct.unpack_from("<QQ", record)
            if signal != "intel_pt" or offset != aux_position or length == 0 or offset + length > len(trace):
                raise HardwareTraceLost("Intel PT AUX records do not cover the captured bytes")
            aux_position += length
        if kind == _SAMPLE:
            if signal == "intel_pt":
                raise HardwareCaptureError("Unexpected sample in an Intel PT stream")
            expected = _SAMPLE_ROW.size + (16 if signal == "memory_loads" else 0)
            if len(record) != expected:
                raise HardwareCaptureError("Unexpected perf sample layout")
            ip, pid, tid, time, address, cpu, _, period = _SAMPLE_ROW.unpack_from(record)
            weight, data_source = struct.unpack_from("<QQ", record, _SAMPLE_ROW.size) if signal == "memory_loads" else (0, 0)
            row = (ip, pid, tid, time, cpu, period, address, weight, data_source, bool(misc & (1 << 14)))
            for name, value in zip(names, row):
                columns[name].append(value if value < 1 << 63 else value - (1 << 64))
        position += size
    if signal == "intel_pt" and aux_position != len(trace):
        raise HardwareTraceLost("Intel PT bytes have no matching AUX record")
    return HardwareBatch(source, signal, *(_tensor(columns[name]) for name in names), _tensor(bytearray(trace)))


class PerfCapture:
    """A stopped, finite hardware capture; buffers are read only after stop().

    No capture data is written to a file. Hardware buffers cannot apply lossless
    backpressure, so overflow or a PT gap raises instead of returning a complete
    looking batch. The caller owns target lifetime and chooses the capture window.
    """

    def __init__(self, config: HardwareConfig) -> None:
        if platform.system() != "Linux":
            raise HardwareCaptureError("perf capture requires Linux")
        self.config = config
        self._events: list[tuple[int, int, mmap.mmap, mmap.mmap | None]] = []
        self._running = False
        self._stopped = False

    def __enter__(self) -> "PerfCapture":
        if self._running or self._stopped:
            raise RuntimeError("Create a new PerfCapture for each run")
        attr = _attribute(self.config)
        page = mmap.PAGESIZE
        targets = ((tid, -1) for tid in sorted(int(name) for name in os.listdir(f"/proc/{self.config.pid}/task"))) if self.config.scope == "process" else ((-1, cpu) for cpu in self.config.cpus or ())
        try:
            for tid, cpu in targets:
                fd = _open_event(attr, tid, cpu)
                try:
                    data = mmap.mmap(fd, page * (1 + self.config.data_pages),
                                     flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
                    aux = None
                    try:
                        if self.config.signal == "intel_pt":
                            aux_offset = page * (1 + self.config.data_pages)
                            aux_size = page * self.config.aux_pages
                            struct.pack_into("<QQ", data, _DATA_HEADER_OFFSET + 48, aux_offset, aux_size)
                            aux = mmap.mmap(fd, aux_size, flags=mmap.MAP_SHARED,
                                            prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=aux_offset)
                        self._events.append((fd, cpu if cpu >= 0 else tid, data, aux))
                    except BaseException:
                        data.close()
                        raise
                except BaseException:
                    os.close(fd)
                    raise
            for fd, _, _, _ in self._events:
                fcntl.ioctl(fd, _RESET, 0)
                fcntl.ioctl(fd, _ENABLE, 0)
            self._running = True
            return self
        except BaseException:
            self.close()
            raise

    def stop(self) -> tuple[HardwareBatch, ...]:
        if not self._running or self._stopped:
            raise RuntimeError("Capture must be running and stopped only once")
        self._stopped = True
        for fd, _, _, _ in self._events:
            fcntl.ioctl(fd, _DISABLE, 0)
        self._running = False
        batches = []
        for _, source, data, aux in self._events:
            head, tail, offset, size, aux_head, aux_tail, aux_offset, aux_size = struct.unpack_from(
                "<8Q", data, _DATA_HEADER_OFFSET,
            )
            if offset < mmap.PAGESIZE or size != self.config.data_pages * mmap.PAGESIZE:
                raise HardwareCaptureError("Unexpected perf ring layout")
            records = _ring_bytes(data, head, tail, offset, size)
            trace = b""
            if aux is not None:
                if aux_offset != mmap.PAGESIZE * (1 + self.config.data_pages) or aux_size != len(aux):
                    raise HardwareCaptureError("Unexpected perf AUX layout")
                trace = _ring_bytes(aux, aux_head, aux_tail, 0, aux_size)
            batches.append(_decode_records(records, self.config.signal, source, trace))
        return tuple(batches)

    def close(self) -> None:
        for fd, _, data, aux in self._events:
            try:
                fcntl.ioctl(fd, _DISABLE, 0)
            except OSError:
                pass
            if aux is not None:
                aux.close()
            data.close()
            os.close(fd)
        self._events.clear()
        self._running = False

    def __exit__(self, *_: object) -> None:
        self.close()
