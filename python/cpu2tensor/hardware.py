# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded Linux hardware captures with no file writes during measurement."""

from __future__ import annotations

from array import array
import ctypes
from dataclasses import dataclass
import hashlib
import mmap
import os
from pathlib import Path
import platform
import struct
import time
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
_ID = 0x80082407
_GROUP_FLAG = 1
_FORMAT_TOTAL_TIME_ENABLED = 1 << 0
_FORMAT_TOTAL_TIME_RUNNING = 1 << 1
_FORMAT_ID = 1 << 2
_FORMAT_GROUP = 1 << 3
_GROUP_READ_FORMAT = (
    _FORMAT_TOTAL_TIME_ENABLED | _FORMAT_TOTAL_TIME_RUNNING | _FORMAT_ID | _FORMAT_GROUP
)
_CLOCK_MONOTONIC_RAW = 4
_DATA_HEADER_OFFSET = 1024
_AUX_BAD_FLAGS = 0x0F  # truncated, overwritten, partial, or collided
_COUNTER_SIGNALS = ("instructions", "cycles", "ref_cycles")
_MODALITIES = ("intel_pt", "memory_loads", "counters")
_PEBS_SIGNALS = ("memory_loads", "memory_stores")
_KERNEL_DECODE_CACHE: tuple[bytes, bytes, bytes, bytes, str] | None = None
_SIDEBAND_SAMPLE_FIELDS = (1 << 1) | (1 << 2) | (1 << 7)  # TID, time, CPU.
_PT_SIDEBAND_FLAGS = (
    (1 << 8)   # mmap
    | (1 << 9)  # comm
    | (1 << 13)  # task
    | (1 << 17)  # mmap_data
    | (1 << 18)  # sample_id_all
    | (1 << 23)  # mmap2
    | (1 << 24)  # comm_exec
    | (1 << 25)  # use_clockid
    | (1 << 26)  # context_switch
    | (1 << 28)  # namespaces
    | (1 << 29)  # ksymbol
    | (1 << 30)  # bpf_event
    | (1 << 33)  # text_poke
    | (1 << 34)  # build_id
)


class HardwareCaptureError(RuntimeError):
    """A hardware source is unavailable or its capture cannot be trusted."""


class HardwareTraceLost(HardwareCaptureError):
    """A perf buffer overflowed; the result is not a complete capture."""


@dataclass(frozen=True)
class HardwareConfig:
    """One host Linux capture. Kernel mode samples the host kernel on selected CPUs.

    ``process`` observes user execution of threads that exist when capture starts;
    ``process_kernel`` observes only their kernel execution. Attach before
    releasing a stopped target to avoid a startup gap. ``kernel`` samples all
    tasks executing kernel code on the selected host CPUs; it does not imply that
    a virtual machine's guest kernel is visible to the host PMU.
    """

    scope: Literal["process", "process_kernel", "kernel"]
    signal: Literal[
        "cycles", "instructions", "memory_loads", "memory_stores", "intel_pt"
    ] = "cycles"
    pid: int | None = None
    cpus: tuple[int, ...] | None = None
    period: int = 100_000
    data_pages: int = 64
    aux_pages: int = 128

    def __post_init__(self) -> None:
        if self.scope not in ("process", "process_kernel", "kernel"):
            raise ValueError("scope must be process, process_kernel, or kernel")
        if self.signal not in (
            "cycles", "instructions", "memory_loads", "memory_stores", "intel_pt"
        ):
            raise ValueError("Unknown hardware signal")
        if self.scope in ("process", "process_kernel") and (
            self.pid is None or self.pid <= 0 or self.cpus is not None
        ):
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
    ``memory_loads`` and ``memory_stores``. ``exact_ip`` reports the kernel's
    exact-IP sample flag.
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
    perf_records: torch.Tensor | None = None


@dataclass(frozen=True)
class HardwareDecodeSideband:
    """Static decode state sampled after opening, but before arming, PT events.

    ``perf_records`` on each :class:`HardwareBatch` carries changes emitted while
    the window runs.  These snapshots cover mappings that already existed when
    the event was opened.  They belong to the exact capture session; a later
    replay is not an acceptable substitute.
    """

    clock: str
    captured_before_arm_ns: int
    process_maps: bytes
    kernel_modules: bytes
    kernel_symbols: bytes
    module_build_ids_json: bytes
    pt_attribute: bytes
    kernel_state_sha256: str


@dataclass(frozen=True)
class HardwareMultimodalConfig:
    """One finite process capture with independently owned hardware sources.

    The target must be stopped at its READY gate before entering the capture.
    ``modalities`` exists so matched perturbation arms can use the same seam;
    the default requests simultaneous PT, PEBS memory loads, and non-sampling
    boundary counter reads. New threads created after entry are not followed.
    """

    scope: Literal["process", "process_kernel"]
    pid: int
    modalities: tuple[str, ...] = _MODALITIES
    pebs_period: int = 100_000
    # Exact PT sideband includes mmap2 records.  A 256 KiB ring overflowed on
    # the qualified 5,000-mmap fixture; 4 MiB retained all 5,001 records.
    data_pages: int = 1024
    aux_pages: int = 2048

    def __post_init__(self) -> None:
        if self.scope not in ("process", "process_kernel"):
            raise ValueError("Multimodal capture needs process or process_kernel scope")
        if self.pid <= 0:
            raise ValueError("Multimodal capture needs a positive pid")
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be nonempty and distinct")
        if any(modality not in (*_MODALITIES, "memory_stores")
               for modality in self.modalities):
            raise ValueError("Unknown multimodal hardware source")
        if sum(signal in self.modalities for signal in _PEBS_SIGNALS) > 1:
            raise ValueError("one precise memory event is supported per capture")
        if self.pebs_period <= 0:
            raise ValueError("pebs_period must be positive")
        for name, pages in (("data_pages", self.data_pages), ("aux_pages", self.aux_pages)):
            if pages <= 0 or pages & (pages - 1):
                raise ValueError(f"{name} must be a positive power of two")


@dataclass(frozen=True)
class HardwareCaptureEnvelope:
    """Clock brackets around sequential source enable and disable operations.

    These bounds do not timestamp PT packets or impose an order across CPUs.
    """

    clock: str
    arm_before_ns: int
    arm_after_ns: int
    stop_before_ns: int
    stop_after_ns: int


@dataclass(frozen=True)
class HardwareSourceStatus:
    """Availability and loss state for one requested source."""

    signal: str
    requested: bool
    available: bool
    lost: bool
    time_enabled_ns: int | None = None
    time_running_ns: int | None = None


@dataclass(frozen=True)
class HardwareCounterBatch:
    """Non-sampling PMU deltas over the complete armed target interval."""

    source: int
    tid: int
    cpu: int
    names: tuple[str, ...]
    values: torch.Tensor
    time_enabled_ns: int
    time_running_ns: int
    available: bool
    lost: bool


@dataclass(frozen=True)
class HardwareMultimodalBatch:
    """PT, PEBS, and PMU observations for one explicitly identified thread.

    ``cpu`` is ``-1`` for a task-following perf event. PEBS rows retain the CPU
    reported for each sample. No CPU is inferred for PT bytes or counter deltas.
    """

    source: int
    tid: int
    cpu: int
    envelope: HardwareCaptureEnvelope
    status: tuple[HardwareSourceStatus, ...]
    pt: HardwareBatch | None
    pebs: HardwareBatch | None
    counters: HardwareCounterBatch | None


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
    memory_signal = config.signal in ("memory_loads", "memory_stores")
    attr.sample_type = _MEMORY_FIELDS if memory_signal else _SAMPLE_FIELDS
    attr.flags = 1 | (1 << 6)  # disabled; exclude hypervisor
    if config.scope == "process":
        # A per-thread event with cpu=-1 cannot mmap a ring when inherit is set.
        attr.flags |= 1 << 5  # exclude kernel
    elif config.scope in ("process_kernel", "kernel"):
        attr.flags |= 1 << 4  # exclude user code
    if config.signal in ("cycles", "instructions"):
        attr.type = 0  # PERF_TYPE_HARDWARE
        attr.config = 0 if config.signal == "cycles" else 1
    elif memory_signal:
        root = "/sys/bus/event_source/devices/cpu"
        attr.type = _source_value(root + "/type")
        alias_name = "mem-loads" if config.signal == "memory_loads" else "mem-stores"
        try:
            alias = Path(root + f"/events/{alias_name}").read_text().strip()
        except OSError as error:
            raise HardwareCaptureError(
                f"Precise {config.signal.replace('_', ' ')} PMU is unavailable"
            ) from error
        if config.signal == "memory_loads":
            # Intel's alias must keep its event/umask/ldlat contract.
            if not alias.startswith("event=0xcd,umask=0x1,ldlat="):
                raise HardwareCaptureError("Unsupported precise memory-load PMU encoding")
            attr.config = 0x1CD
            attr.config1 = int(alias.split("ldlat=", 1)[1], 0)
        else:
            if alias != "event=0xd0,umask=0x82":
                raise HardwareCaptureError("Unsupported precise memory-store PMU encoding")
            attr.config = 0x82D0
        # Fail instead of silently time-sharing the precise event with the PMU
        # boundary group. The read-format times are checked after disable.
        attr.flags |= (2 << 15) | (1 << 2)
        attr.read_format = _FORMAT_TOTAL_TIME_ENABLED | _FORMAT_TOTAL_TIME_RUNNING
    else:
        root = "/sys/bus/event_source/devices/intel_pt"
        attr.type = _source_value(root + "/type")
        try:
            pt_format = Path(root + "/format/pt").read_text().strip()
            branch_format = Path(root + "/format/branch").read_text().strip()
        except OSError as error:
            raise HardwareCaptureError("Intel PT is unavailable") from error
        if pt_format != "config:0" or branch_format != "config:13":
            raise HardwareCaptureError("Unsupported Intel PT branch encoding")
        # Bit 0 opts into explicit control, so branch packets must be enabled
        # separately with bit 13. Without it, a nonempty AUX buffer can still
        # contain only context packets and falsely look like a branch trace.
        attr.config = 1 | (1 << 13)
        # Preserve the exact sideband needed to interpret the AUX bytes.  The
        # kernel reports changes during the armed interval in the data ring;
        # existing mappings are snapshotted after the fd opens and before arm.
        attr.sample_type = _SIDEBAND_SAMPLE_FIELDS
        attr.sample_period = 0
        attr.flags |= _PT_SIDEBAND_FLAGS
        attr.clockid = _CLOCK_MONOTONIC_RAW
    if config.signal != "intel_pt":
        # A single explicit clock lets PEBS and generic samples share an
        # envelope without pretending raw PT packets carry timestamps.
        attr.flags |= 1 << 25  # use_clockid
        attr.clockid = _CLOCK_MONOTONIC_RAW
    return attr


def _counter_attribute(scope: str, signal: str, *, leader: bool) -> _PerfAttr:
    configs = {"cycles": 0, "instructions": 1, "ref_cycles": 9}
    try:
        event = configs[signal]
    except KeyError as error:
        raise ValueError("Unknown boundary counter") from error
    attr = _PerfAttr()
    attr.type = 0  # PERF_TYPE_HARDWARE
    attr.size = ctypes.sizeof(_PerfAttr)
    attr.config = event
    attr.read_format = _GROUP_READ_FORMAT
    attr.flags = 1 | (1 << 6)  # disabled; exclude hypervisor
    if leader:
        # A pinned leader fails instead of silently time-sharing this group.
        attr.flags |= 1 << 2
    if scope == "process":
        attr.flags |= 1 << 5
    elif scope == "process_kernel":
        attr.flags |= 1 << 4
    else:
        raise ValueError("Boundary counters need process or process_kernel scope")
    return attr


def _open_event(attr: _PerfAttr, tid: int, cpu: int, group_fd: int = -1) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(_syscall_number(), ctypes.byref(attr), tid, cpu, group_fd, 0)
    if fd < 0:
        code = ctypes.get_errno()
        raise HardwareCaptureError(
            f"perf_event_open failed for tid={tid}, cpu={cpu}: {os.strerror(code)}"
        )
    return int(fd)


def _event_id(fd: int) -> int:
    identifier = array("Q", (0,))
    try:
        fcntl.ioctl(fd, _ID, identifier, True)
    except OSError as error:
        raise HardwareCaptureError("Cannot identify a PMU group member") from error
    if identifier[0] == 0:
        raise HardwareCaptureError("Perf returned an invalid PMU event identifier")
    return int(identifier[0])


def _read_counter_group(fd: int, identifiers: tuple[int, ...]) -> tuple[tuple[int, ...], int, int]:
    expected = 24 + 16 * len(identifiers)
    try:
        payload = os.read(fd, expected)
    except OSError as error:
        raise HardwareCaptureError("Cannot read the PMU counter group") from error
    if len(payload) != expected:
        # A pinned group that cannot be scheduled can return EOF.
        raise HardwareCaptureError("Pinned PMU counter group did not produce a complete read")
    count, enabled, running = struct.unpack_from("<QQQ", payload)
    if count != len(identifiers):
        raise HardwareCaptureError("PMU counter group member count changed")
    observed: dict[int, int] = {}
    position = 24
    for _ in identifiers:
        value, identifier = struct.unpack_from("<QQ", payload, position)
        position += 16
        if identifier in observed:
            raise HardwareCaptureError("PMU counter group repeated an event identifier")
        observed[identifier] = value
    if set(observed) != set(identifiers):
        raise HardwareCaptureError("PMU counter group identity changed")
    return tuple(observed[identifier] for identifier in identifiers), enabled, running


def _read_sampled_event(fd: int) -> tuple[int, int, int]:
    """Read one sampled event's count and scheduling times after disable."""
    try:
        payload = os.read(fd, 24)
    except OSError as error:
        raise HardwareCaptureError("Cannot read the PEBS scheduling status") from error
    if len(payload) != 24:
        raise HardwareCaptureError("Pinned PEBS event did not produce a complete read")
    value, enabled, running = struct.unpack("<QQQ", payload)
    if enabled == 0 or running == 0:
        raise HardwareCaptureError("Pinned PEBS event did not run")
    if running != enabled:
        raise HardwareCaptureError(
            f"PEBS event was multiplexed: enabled={enabled}, running={running}"
        )
    return value, enabled, running


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


def _decode_records(
    data: bytes,
    signal: str,
    source: int,
    trace: bytes,
    *,
    trace_offset: int = 0,
) -> HardwareBatch:
    names = ("ip", "pid", "tid", "time", "cpu", "period", "address", "weight", "data_source", "exact_ip")
    columns = {name: array("q") for name in names}
    position = 0
    aux_position = trace_offset
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
            if (signal != "intel_pt" or offset != aux_position or length == 0 or
                    offset + length > trace_offset + len(trace)):
                raise HardwareTraceLost("Intel PT AUX records do not cover the captured bytes")
            aux_position += length
        if kind == _SAMPLE:
            if signal == "intel_pt":
                raise HardwareCaptureError("Unexpected sample in an Intel PT stream")
            memory_signal = signal in ("memory_loads", "memory_stores")
            expected = _SAMPLE_ROW.size + (16 if memory_signal else 0)
            if len(record) != expected:
                raise HardwareCaptureError("Unexpected perf sample layout")
            ip, pid, tid, time, address, cpu, _, period = _SAMPLE_ROW.unpack_from(record)
            weight, data_source = (
                struct.unpack_from("<QQ", record, _SAMPLE_ROW.size)
                if memory_signal else (0, 0)
            )
            row = (ip, pid, tid, time, cpu, period, address, weight, data_source, bool(misc & (1 << 14)))
            for name, value in zip(names, row):
                columns[name].append(value if value < 1 << 63 else value - (1 << 64))
        position += size
    if signal == "intel_pt" and aux_position != trace_offset + len(trace):
        raise HardwareTraceLost(
            "Intel PT bytes have no matching AUX record: "
            f"covered={aux_position - trace_offset}, bytes={len(trace)}, "
            f"perf_records={len(data)}"
        )
    return HardwareBatch(
        source,
        signal,
        *(_tensor(columns[name]) for name in names),
        _tensor(bytearray(trace)),
        _tensor(bytearray(data)),
    )


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
        if self.config.scope in ("process", "process_kernel"):
            try:
                tids = sorted(int(name) for name in os.listdir(f"/proc/{self.config.pid}/task"))
            except OSError as error:
                raise HardwareCaptureError(f"Cannot enumerate process {self.config.pid} threads") from error
            targets = ((tid, -1) for tid in tids)
        else:
            targets = ((-1, cpu) for cpu in self.config.cpus or ())
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


@dataclass
class _MultimodalSource:
    tid: int
    cpu: int = -1
    pt_fd: int = -1
    pt_data: mmap.mmap | None = None
    pt_aux: mmap.mmap | None = None
    pebs_fd: int = -1
    pebs_data: mmap.mmap | None = None
    pebs_count: int = 0
    pebs_time_enabled_ns: int = 0
    pebs_time_running_ns: int = 0
    counter_fds: tuple[int, ...] = ()
    counter_ids: tuple[int, ...] = ()
    counter_start: tuple[tuple[int, ...], int, int] | None = None


def _map_data(fd: int, pages: int) -> mmap.mmap:
    return mmap.mmap(
        fd,
        mmap.PAGESIZE * (1 + pages),
        flags=mmap.MAP_SHARED,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )


def _map_aux(fd: int, data: mmap.mmap, data_pages: int, aux_pages: int) -> mmap.mmap:
    offset = mmap.PAGESIZE * (1 + data_pages)
    size = mmap.PAGESIZE * aux_pages
    struct.pack_into("<QQ", data, _DATA_HEADER_OFFSET + 48, offset, size)
    return mmap.mmap(
        fd,
        size,
        flags=mmap.MAP_SHARED,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
        offset=offset,
    )


def _read_sideband_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise HardwareCaptureError(
            f"Cannot retain required PT decode sideband: {path}"
        ) from error


def _module_build_ids() -> bytes:
    """Return stable JSON bytes for every loaded module build-ID note."""
    import json

    modules = {}
    try:
        roots = sorted(Path("/sys/module").iterdir())
    except OSError as error:
        raise HardwareCaptureError("Cannot enumerate loaded kernel modules") from error
    for root in roots:
        # sysfs also exposes global module parameters as regular files below
        # /sys/module (for example ``compression`` on Ubuntu).
        if not root.is_dir():
            continue
        note = root / "notes" / ".note.gnu.build-id"
        try:
            payload = note.read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as error:
            raise HardwareCaptureError(
                f"Cannot read kernel module build ID: {note}"
            ) from error
        modules[root.name] = payload.hex()
    return json.dumps(modules, sort_keys=True, separators=(",", ":")).encode()


def _module_decode_identity(modules: bytes) -> bytes:
    """Identify module mappings without volatile reference counts or list order."""
    mappings = []
    for line in modules.splitlines():
        fields = line.split()
        if len(fields) != 6:
            raise HardwareCaptureError("Cannot parse required kernel module mapping")
        mappings.append(b" ".join((fields[0], fields[1], fields[3], fields[4], fields[5])))
    return b"\n".join(sorted(mappings))


def _kernel_decode_state() -> tuple[bytes, bytes, bytes, str]:
    """Reuse symbols while the boot's address-bearing module map is stable."""
    global _KERNEL_DECODE_CACHE

    boot_id = _read_sideband_file(Path("/proc/sys/kernel/random/boot_id"))
    modules = _read_sideband_file(Path("/proc/modules"))
    module_identity = _module_decode_identity(modules)
    if _KERNEL_DECODE_CACHE is not None:
        cached_boot, cached_modules, symbols, build_ids, identity = _KERNEL_DECODE_CACHE
        if boot_id == cached_boot and module_identity == cached_modules:
            return modules, symbols, build_ids, identity
    symbols = _read_sideband_file(Path("/proc/kallsyms"))
    build_ids = _module_build_ids()
    digest = hashlib.sha256()
    for value in (boot_id, module_identity, symbols, build_ids):
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
    identity = digest.hexdigest()
    _KERNEL_DECODE_CACHE = (boot_id, module_identity, symbols, build_ids, identity)
    return modules, symbols, build_ids, identity


def _capture_decode_sideband(pid: int, pt_attribute: _PerfAttr) -> HardwareDecodeSideband:
    """Snapshot pre-existing decode state inside the open perf session."""
    before = time.clock_gettime_ns(_CLOCK_MONOTONIC_RAW)
    modules, symbols, build_ids, identity = _kernel_decode_state()
    return HardwareDecodeSideband(
        clock="CLOCK_MONOTONIC_RAW",
        captured_before_arm_ns=before,
        process_maps=_read_sideband_file(Path(f"/proc/{pid}/maps")),
        kernel_modules=modules,
        kernel_symbols=symbols,
        module_build_ids_json=build_ids,
        pt_attribute=ctypes.string_at(
            ctypes.addressof(pt_attribute), ctypes.sizeof(pt_attribute)
        ),
        kernel_state_sha256=identity,
    )


def _read_mapped_batch(
    data: mmap.mmap,
    aux: mmap.mmap | None,
    *,
    data_pages: int,
    signal: str,
    source: int,
    consume: bool = False,
) -> HardwareBatch:
    head, tail, offset, size, aux_head, aux_tail, aux_offset, aux_size = struct.unpack_from(
        "<8Q", data, _DATA_HEADER_OFFSET,
    )
    if offset < mmap.PAGESIZE or size != data_pages * mmap.PAGESIZE:
        raise HardwareCaptureError("Unexpected perf ring layout")
    records = _ring_bytes(data, head, tail, offset, size)
    trace = b""
    if aux is not None:
        if aux_offset != mmap.PAGESIZE * (1 + data_pages) or aux_size != len(aux):
            raise HardwareCaptureError("Unexpected perf AUX layout")
        trace = _ring_bytes(aux, aux_head, aux_tail, 0, aux_size)
    batch = _decode_records(
        records,
        signal,
        source,
        trace,
        trace_offset=aux_tail if aux is not None else 0,
    )
    if consume:
        # Publish ownership only after a complete decode. A failed decode leaves
        # both tails untouched so callers can preserve the corrupt window.
        struct.pack_into("<Q", data, _DATA_HEADER_OFFSET + 8, head)
        if aux is not None:
            struct.pack_into("<Q", data, _DATA_HEADER_OFFSET + 40, aux_head)
    return batch


def _counter_batch(
    source: _MultimodalSource,
    final: tuple[tuple[int, ...], int, int],
) -> HardwareCounterBatch:
    if source.counter_start is None:
        raise HardwareCaptureError("PMU counter group has no boundary baseline")
    start_values, start_enabled, start_running = source.counter_start
    final_values, final_enabled, final_running = final
    if (final_enabled < start_enabled or final_running < start_running or
            any(end < start for start, end in zip(start_values, final_values))):
        raise HardwareCaptureError("PMU counter group moved backwards")
    enabled = final_enabled - start_enabled
    running = final_running - start_running
    if enabled == 0 or running == 0:
        raise HardwareCaptureError("PMU counter group did not run")
    if running != enabled:
        raise HardwareCaptureError(
            f"PMU counter group was multiplexed: enabled={enabled}, running={running}"
        )
    values = tuple(end - start for start, end in zip(start_values, final_values))
    return HardwareCounterBatch(
        source=source.tid,
        tid=source.tid,
        cpu=source.cpu,
        names=_COUNTER_SIGNALS,
        values=_tensor(array("q", values)),
        time_enabled_ns=enabled,
        time_running_ns=running,
        available=True,
        lost=False,
    )


class PerfMultimodalCapture:
    """Finite simultaneous PT, PEBS, and boundary-counter capture.

    Enter only after every target thread is stopped at a READY boundary. The
    source descriptors are armed sequentially inside one CLOCK_MONOTONIC_RAW
    bracket, after which the caller may release the target. ``stop`` is likewise
    bracketed. The envelope is source-control timing, not PT packet timing.
    """

    def __init__(self, config: HardwareMultimodalConfig) -> None:
        if platform.system() != "Linux":
            raise HardwareCaptureError("perf capture requires Linux")
        self.config = config
        self._sources: list[_MultimodalSource] = []
        self._running = False
        self._stopped = False
        self._arm_before_ns = 0
        self._arm_after_ns = 0
        self._decode_sideband: HardwareDecodeSideband | None = None

    @property
    def decode_sideband(self) -> HardwareDecodeSideband:
        if self._decode_sideband is None:
            raise RuntimeError("PT decode sideband is unavailable before capture entry")
        return self._decode_sideband

    def _hardware_config(self, signal: str) -> HardwareConfig:
        return HardwareConfig(
            self.config.scope,
            pid=self.config.pid,
            signal=signal,
            period=self.config.pebs_period,
            data_pages=self.config.data_pages,
            aux_pages=self.config.aux_pages,
        )

    def _open_source(self, tid: int) -> _MultimodalSource:
        source = _MultimodalSource(tid=tid)
        self._sources.append(source)
        if "intel_pt" in self.config.modalities:
            source.pt_fd = _open_event(_attribute(self._hardware_config("intel_pt")), tid, -1)
            source.pt_data = _map_data(source.pt_fd, self.config.data_pages)
            try:
                source.pt_aux = _map_aux(
                    source.pt_fd, source.pt_data, self.config.data_pages, self.config.aux_pages
                )
            except BaseException:
                source.pt_data.close()
                source.pt_data = None
                raise
        pebs_signal = next(
            (signal for signal in _PEBS_SIGNALS if signal in self.config.modalities),
            None,
        )
        if pebs_signal is not None:
            source.pebs_fd = _open_event(
                _attribute(self._hardware_config(pebs_signal)), tid, -1
            )
            source.pebs_data = _map_data(source.pebs_fd, self.config.data_pages)
        if "counters" in self.config.modalities:
            fds = []
            leader = -1
            for index, signal in enumerate(_COUNTER_SIGNALS):
                fd = _open_event(
                    _counter_attribute(self.config.scope, signal, leader=index == 0),
                    tid,
                    -1,
                    leader,
                )
                if index == 0:
                    leader = fd
                fds.append(fd)
                source.counter_fds = tuple(fds)
            source.counter_ids = tuple(_event_id(fd) for fd in fds)
        return source

    def _open_sources(self) -> None:
        if self._sources:
            raise RuntimeError("Perf sources are already open")
        try:
            tids = sorted(int(name) for name in os.listdir(f"/proc/{self.config.pid}/task"))
        except OSError as error:
            raise HardwareCaptureError(
                f"Cannot enumerate process {self.config.pid} threads"
            ) from error
        for tid in tids:
            self._open_source(tid)

    def _assert_empty_rings(self) -> None:
        for source in self._sources:
            for data, has_aux in ((source.pt_data, True), (source.pebs_data, False)):
                if data is None:
                    continue
                head, tail, _, _, aux_head, aux_tail, _, _ = struct.unpack_from(
                    "<8Q", data, _DATA_HEADER_OFFSET,
                )
                if head != tail or (has_aux and aux_head != aux_tail):
                    raise HardwareCaptureError(
                        f"perf ring for thread {source.tid} was not completely consumed"
                    )

    def _start_window(self, *, require_empty: bool) -> None:
        if self._running:
            raise RuntimeError("Capture window is already running")
        if require_empty:
            self._assert_empty_rings()
        self._arm_before_ns = time.clock_gettime_ns(_CLOCK_MONOTONIC_RAW)
        for source in self._sources:
            if source.pt_fd >= 0:
                fcntl.ioctl(source.pt_fd, _RESET, 0)
                fcntl.ioctl(source.pt_fd, _ENABLE, 0)
            if source.pebs_fd >= 0:
                fcntl.ioctl(source.pebs_fd, _RESET, 0)
                fcntl.ioctl(source.pebs_fd, _ENABLE, 0)
            if source.counter_fds:
                leader = source.counter_fds[0]
                fcntl.ioctl(leader, _RESET, _GROUP_FLAG)
                fcntl.ioctl(leader, _ENABLE, _GROUP_FLAG)
        self._arm_after_ns = time.clock_gettime_ns(_CLOCK_MONOTONIC_RAW)
        for source in self._sources:
            if source.counter_fds:
                source.counter_start = _read_counter_group(
                    source.counter_fds[0], source.counter_ids
                )
        self._running = True

    def __enter__(self) -> "PerfMultimodalCapture":
        if self._running or self._stopped:
            raise RuntimeError("Create a new PerfMultimodalCapture for each run")
        try:
            self._open_sources()
            if "intel_pt" in self.config.modalities:
                self._decode_sideband = _capture_decode_sideband(
                    self.config.pid, _attribute(self._hardware_config("intel_pt"))
                )
            self._start_window(require_empty=False)
            return self
        except BaseException:
            self.close()
            raise

    def stop(self) -> tuple[HardwareMultimodalBatch, ...]:
        if not self._running or self._stopped:
            raise RuntimeError("Capture must be running and stopped only once")
        self._stopped = True
        return self._stop_window(consume=False)

    def _stop_window(self, *, consume: bool) -> tuple[HardwareMultimodalBatch, ...]:
        stop_before_ns = time.clock_gettime_ns(_CLOCK_MONOTONIC_RAW)
        final_counters: dict[int, tuple[tuple[int, ...], int, int]] = {}
        try:
            for source in self._sources:
                if source.counter_fds:
                    final_counters[source.tid] = _read_counter_group(
                        source.counter_fds[0], source.counter_ids
                    )
        finally:
            for source in self._sources:
                if source.counter_fds:
                    fcntl.ioctl(source.counter_fds[0], _DISABLE, _GROUP_FLAG)
                if source.pebs_fd >= 0:
                    fcntl.ioctl(source.pebs_fd, _DISABLE, 0)
                    (source.pebs_count, source.pebs_time_enabled_ns,
                     source.pebs_time_running_ns) = _read_sampled_event(source.pebs_fd)
                if source.pt_fd >= 0:
                    fcntl.ioctl(source.pt_fd, _DISABLE, 0)
            self._running = False
        stop_after_ns = time.clock_gettime_ns(_CLOCK_MONOTONIC_RAW)
        envelope = HardwareCaptureEnvelope(
            clock="CLOCK_MONOTONIC_RAW",
            arm_before_ns=self._arm_before_ns,
            arm_after_ns=self._arm_after_ns,
            stop_before_ns=stop_before_ns,
            stop_after_ns=stop_after_ns,
        )
        batches = []
        for source in self._sources:
            pt = None
            pebs = None
            counters = None
            if source.pt_data is not None:
                pt = _read_mapped_batch(
                    source.pt_data,
                    source.pt_aux,
                    data_pages=self.config.data_pages,
                    signal="intel_pt",
                    source=source.tid,
                    consume=consume,
                )
            if source.pebs_data is not None:
                pebs = _read_mapped_batch(
                    source.pebs_data,
                    None,
                    data_pages=self.config.data_pages,
                    signal=next(
                        signal for signal in _PEBS_SIGNALS
                        if signal in self.config.modalities
                    ),
                    source=source.tid,
                    consume=consume,
                )
            if source.counter_fds:
                counters = _counter_batch(source, final_counters[source.tid])
            pebs_signal = next(
                (signal for signal in _PEBS_SIGNALS if signal in self.config.modalities),
                "memory_loads",
            )
            status = tuple(
                HardwareSourceStatus(
                    signal=modality,
                    requested=modality in self.config.modalities,
                    available=(pt is not None if modality == "intel_pt" else
                               pebs is not None if modality == pebs_signal else
                               counters is not None),
                    lost=False,
                    time_enabled_ns=(
                        source.pebs_time_enabled_ns
                        if modality == pebs_signal and pebs is not None
                        else counters.time_enabled_ns
                        if modality == "counters" and counters is not None
                        else None
                    ),
                    time_running_ns=(
                        source.pebs_time_running_ns
                        if modality == pebs_signal and pebs is not None
                        else counters.time_running_ns
                        if modality == "counters" and counters is not None
                        else None
                    ),
                )
                for modality in ("intel_pt", pebs_signal, "counters")
            )
            batches.append(
                HardwareMultimodalBatch(
                    source=source.tid,
                    tid=source.tid,
                    cpu=source.cpu,
                    envelope=envelope,
                    status=status,
                    pt=pt,
                    pebs=pebs,
                    counters=counters,
                )
            )
        return tuple(batches)

    def close(self) -> None:
        for source in self._sources:
            for fd in source.counter_fds:
                try:
                    fcntl.ioctl(fd, _DISABLE, _GROUP_FLAG)
                except OSError:
                    pass
                os.close(fd)
            for fd in (source.pebs_fd, source.pt_fd):
                if fd >= 0:
                    try:
                        fcntl.ioctl(fd, _DISABLE, 0)
                    except OSError:
                        pass
            if source.pebs_data is not None:
                source.pebs_data.close()
            if source.pt_aux is not None:
                source.pt_aux.close()
            if source.pt_data is not None:
                source.pt_data.close()
            for fd in (source.pebs_fd, source.pt_fd):
                if fd >= 0:
                    os.close(fd)
        self._sources.clear()
        self._running = False

    def __exit__(self, *_: object) -> None:
        self.close()


class PerfMultimodalSession(PerfMultimodalCapture):
    """Long-lived perf ownership with loss-checked, individually owned windows.

    The target thread set is frozen when the session opens. ``start`` and
    ``stop`` reset and disable the same descriptors, while ``stop`` advances the
    data and AUX tails only after a complete decode. This removes event-open and
    mmap work from the hot path without weakening per-window custody.
    """

    def __init__(self, config: HardwareMultimodalConfig) -> None:
        super().__init__(config)
        self._entered = False

    def __enter__(self) -> "PerfMultimodalSession":
        if self._entered or self._sources:
            raise RuntimeError("PerfMultimodalSession is already open")
        try:
            self._open_sources()
            self._assert_empty_rings()
            if "intel_pt" in self.config.modalities:
                self._decode_sideband = _capture_decode_sideband(
                    self.config.pid, _attribute(self._hardware_config("intel_pt"))
                )
            self._entered = True
            return self
        except BaseException:
            self.close()
            raise

    def start(self) -> "PerfMultimodalSession":
        if not self._entered:
            raise RuntimeError("Open the session before starting a window")
        self._start_window(require_empty=True)
        return self

    def stop(self) -> tuple[HardwareMultimodalBatch, ...]:
        if not self._entered or not self._running:
            raise RuntimeError("A session window must be running before stop")
        return self._stop_window(consume=True)

    def close(self) -> None:
        super().close()
        self._entered = False
