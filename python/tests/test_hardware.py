# SPDX-License-Identifier: AGPL-3.0-only
"""Hardware record semantics and bounded in-memory capture ownership."""

from array import array
import ctypes
import mmap
import os
import struct
import unittest
from unittest import mock

import torch

from cpu2tensor.hardware import (
    HardwareBatch, HardwareCaptureError, HardwareConfig, HardwareTraceLost,
    PerfCapture, _PerfAttr, _attribute, _decode_records, _ring_bytes,
    _open_event, _source_value, _syscall_number, _tensor,
)


class FakeMapping(bytearray):
    def close(self) -> None:
        self.closed = True


def record(kind: int, payload: bytes = b"", *, misc: int = 0) -> bytes:
    return struct.pack("<IHH", kind, misc, 8 + len(payload)) + payload


class HardwareTests(unittest.TestCase):
    def test_config_rejects_ambiguous_scope_and_unbounded_sizes(self) -> None:
        invalid = (
            dict(scope="other"), dict(scope="process"), dict(scope="process", pid=-1),
            dict(scope="process", pid=1, cpus=(0,)), dict(scope="kernel"),
            dict(scope="kernel", cpus=(0,), pid=2), dict(scope="kernel", cpus=(0, 0)),
            dict(scope="kernel", cpus=(-1,)), dict(scope="kernel", cpus=(0,), period=0),
            dict(scope="kernel", cpus=(0,), data_pages=3),
            dict(scope="kernel", cpus=(0,), aux_pages=0),
            dict(scope="kernel", cpus=(0,), signal="not-a-signal"),
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                HardwareConfig(**options)

    def test_sample_config_is_hardware_only_with_correct_privilege_scope(self) -> None:
        process = _attribute(HardwareConfig("process", pid=7, signal="instructions"))
        kernel = _attribute(HardwareConfig("kernel", cpus=(0,), signal="cycles"))
        self.assertEqual((process.type, process.config, process.sample_period), (0, 1, 100_000))
        self.assertEqual((kernel.type, kernel.config), (0, 0))
        self.assertEqual(process.size, ctypes.sizeof(_PerfAttr))
        self.assertEqual(process.flags & ((1 << 1) | (1 << 5) | (1 << 6)),
                         (1 << 5) | (1 << 6))
        self.assertEqual(kernel.flags & ((1 << 4) | (1 << 6)), (1 << 4) | (1 << 6))
        self.assertFalse(kernel.flags & (1 << 5))

    def test_vendor_sources_are_probed_and_precise_memory_is_required(self) -> None:
        def read_text(path, *args, **kwargs):
            name = str(path)
            if name.endswith("/type"):
                return "11"
            if name.endswith("/events/mem-loads"):
                return "event=0xcd,umask=0x1,ldlat=3"
            if name.endswith("/format/pt"):
                return "config:0"
            raise AssertionError(name)

        with mock.patch("pathlib.Path.read_text", read_text):
            memory = _attribute(HardwareConfig("kernel", cpus=(0,), signal="memory_loads"))
            pt = _attribute(HardwareConfig("process", pid=7, signal="intel_pt"))
        self.assertEqual((memory.type, memory.config, memory.config1), (11, 0x1CD, 3))
        self.assertEqual((memory.flags >> 15) & 3, 2)
        self.assertEqual((pt.type, pt.config, pt.sample_type, pt.sample_period), (11, 1, 0, 0))
        with mock.patch("pathlib.Path.read_text", return_value="not-an-int"):
            with self.assertRaises(HardwareCaptureError):
                _source_value("missing")
        with mock.patch("pathlib.Path.read_text", side_effect=OSError("missing")):
            with self.assertRaises(HardwareCaptureError):
                _source_value("missing")
            with self.assertRaises(HardwareCaptureError):
                _attribute(HardwareConfig("kernel", cpus=(0,), signal="memory_loads"))
            with self.assertRaises(HardwareCaptureError):
                _attribute(HardwareConfig("kernel", cpus=(0,), signal="intel_pt"))
        with mock.patch("pathlib.Path.read_text", side_effect=["4", "other"]):
            with self.assertRaises(HardwareCaptureError):
                _attribute(HardwareConfig("kernel", cpus=(0,), signal="memory_loads"))
        with mock.patch("pathlib.Path.read_text", side_effect=["11", "config:2"]):
            with self.assertRaises(HardwareCaptureError):
                _attribute(HardwareConfig("kernel", cpus=(0,), signal="intel_pt"))
        for signal in ("memory_loads", "intel_pt"):
            with mock.patch("pathlib.Path.read_text", side_effect=["11", OSError("absent")]):
                with self.subTest(signal=signal), self.assertRaises(HardwareCaptureError):
                    _attribute(HardwareConfig("kernel", cpus=(0,), signal=signal))

    def test_supported_syscalls_and_wrong_os(self) -> None:
        for machine, number in (("x86_64", 298), ("aarch64", 241)):
            with mock.patch("cpu2tensor.hardware.platform.machine", return_value=machine):
                self.assertEqual(_syscall_number(), number)
        with mock.patch("cpu2tensor.hardware.platform.machine", return_value="riscv64"):
            with self.assertRaises(HardwareCaptureError):
                _syscall_number()
        with mock.patch("cpu2tensor.hardware.platform.system", return_value="Darwin"):
            with self.assertRaises(HardwareCaptureError):
                PerfCapture(HardwareConfig("kernel", cpus=(0,)))

    def test_wrap_and_overflow_are_explicit(self) -> None:
        mapping = bytearray(b"abcd1234")
        self.assertEqual(_ring_bytes(mapping, 10, 6, 0, 8), b"34ab")
        for head, tail in ((5, 6), (10, 1)):
            with self.assertRaises(HardwareTraceLost):
                _ring_bytes(mapping, head, tail, 0, 8)

    def test_perf_open_reports_permission_and_returns_fd(self) -> None:
        syscall = mock.Mock(return_value=17)
        with mock.patch("cpu2tensor.hardware.ctypes.CDLL", return_value=mock.Mock(syscall=syscall)), \
                mock.patch("cpu2tensor.hardware.platform.machine", return_value="x86_64"):
            self.assertEqual(_open_event(_PerfAttr(), 3, -1), 17)
            syscall.return_value = -1
            with mock.patch("cpu2tensor.hardware.ctypes.get_errno", return_value=1):
                with self.assertRaisesRegex(HardwareCaptureError, "Operation not permitted"):
                    _open_event(_PerfAttr(), 3, -1)

    def test_exact_sample_columns_and_owned_storage(self) -> None:
        sample = struct.pack("<QIIQQIIQQQ", 0xFFFFFFFFFFFFFFFF, 3, 4, 17,
                             0xABC, 2, 0, 100, 77, 9)
        data = record(9, sample, misc=1 << 14)
        result = _decode_records(data, "memory_loads", 2, b"")
        self.assertIsInstance(result, HardwareBatch)
        self.assertEqual(result.source, 2)
        self.assertEqual(result.ip.tolist(), [-1])
        self.assertEqual(result.pid.tolist(), [3])
        self.assertEqual(result.tid.tolist(), [4])
        self.assertEqual(result.time.tolist(), [17])
        self.assertEqual(result.address.tolist(), [0xABC])
        self.assertEqual(result.cpu.tolist(), [2])
        self.assertEqual(result.period.tolist(), [100])
        self.assertEqual(result.weight.tolist(), [77])
        self.assertEqual(result.data_source.tolist(), [9])
        self.assertEqual(result.exact_ip.tolist(), [1])
        self.assertEqual(result.trace_bytes.numel(), 0)
        self.assertEqual(_tensor(array("q")).dtype, torch.int64)
        self.assertEqual(_tensor(bytearray()).dtype, torch.uint8)
        self.assertEqual(_decode_records(record(9, sample[:48]), "cycles", 0, b"").ip.tolist(), [-1])

    def test_pt_packets_are_retained_and_gaps_rejected(self) -> None:
        result = _decode_records(record(11, struct.pack("<QQQ", 0, 4, 0)), "intel_pt", 0, b"PT!!")
        self.assertEqual(result.trace_bytes.tolist(), [80, 84, 33, 33])
        self.assertEqual(result.ip.numel(), 0)
        for data in (
            b"123", record(9, b"bad"), record(2, struct.pack("<QQ", 1, 2)),
            record(13, struct.pack("<Q", 1)), record(11, b"short"),
            record(11, struct.pack("<QQQ", 0, 4, 1)),
            record(11, struct.pack("<QQQ", 1, 4, 0)),
            record(11, struct.pack("<QQQ", 0, 3, 0)),
            record(9, struct.pack("<QIIQQIIQ", 1, 2, 3, 4, 5, 6, 0, 7)),
        ):
            with self.subTest(data=data), self.assertRaises(HardwareCaptureError):
                _decode_records(data, "intel_pt", 0, b"")
        with self.assertRaises(HardwareTraceLost):
            _decode_records(struct.pack("<IHH", 9, 0, 100), "cycles", 0, b"")
        with self.assertRaises(HardwareCaptureError):
            _decode_records(record(9, b"bad"), "cycles", 0, b"")
        with self.assertRaises(HardwareTraceLost):
            _decode_records(b"", "intel_pt", 0, b"orphaned")

    def test_capture_disables_before_reading_and_never_opens_a_file(self) -> None:
        page = mmap.PAGESIZE
        config = HardwareConfig("kernel", cpus=(2,), data_pages=1)
        raw = record(9, struct.pack("<QIIQQIIQ", 0x123, 7, 8, 10, 0, 2, 0, 50))
        data = FakeMapping(page * 2)
        struct.pack_into("<8Q", data, 1024, len(raw), 0, page, page, 0, 0, 0, 0)
        data[page:page + len(raw)] = raw
        calls = []

        def ioctl(fd, command, arg):
            calls.append(command)

        with mock.patch("cpu2tensor.hardware.platform.system", return_value="Linux"):
            capture = PerfCapture(config)
        with self.assertRaises(RuntimeError):
            capture.stop()
        with mock.patch("cpu2tensor.hardware._open_event", return_value=11), \
                mock.patch("cpu2tensor.hardware.mmap.mmap", return_value=data), \
                mock.patch("cpu2tensor.hardware.fcntl.ioctl", side_effect=ioctl), \
                mock.patch("cpu2tensor.hardware.os.close"):
            with capture:
                result = capture.stop()
                with self.assertRaises(RuntimeError):
                    capture.stop()
        self.assertEqual(result[0].ip.tolist(), [0x123])
        self.assertEqual(result[0].source, 2)
        self.assertEqual(calls[:3], [0x2403, 0x2400, 0x2401])
        self.assertTrue(data.closed)

    def test_process_pt_mapping_and_owned_aux_bytes(self) -> None:
        page = mmap.PAGESIZE
        config = HardwareConfig("process", pid=43, signal="intel_pt", data_pages=1, aux_pages=1)
        raw = record(11, struct.pack("<QQQ", 0, 4, 0))
        data = FakeMapping(page * 2)
        aux = FakeMapping(page)
        aux[:4] = b"PT!!"
        struct.pack_into("<8Q", data, 1024, len(raw), 0, page, page, 4, 0, page * 2, page)
        data[page:page + len(raw)] = raw
        with mock.patch("cpu2tensor.hardware.platform.system", return_value="Linux"):
            capture = PerfCapture(config)
        with mock.patch("cpu2tensor.hardware._attribute", return_value=_PerfAttr()), \
                mock.patch("cpu2tensor.hardware.os.listdir", return_value=["43"]), \
                mock.patch("cpu2tensor.hardware._open_event", return_value=11), \
                mock.patch("cpu2tensor.hardware.mmap.mmap", side_effect=[data, aux]), \
                mock.patch("cpu2tensor.hardware.fcntl.ioctl"), \
                mock.patch("cpu2tensor.hardware.os.close"):
            with capture:
                batch = capture.stop()[0]
        self.assertEqual(batch.trace_bytes.tolist(), list(b"PT!!"))
        self.assertEqual(batch.source, 43)
        self.assertTrue(data.closed)
        self.assertTrue(aux.closed)
        with self.assertRaises(RuntimeError):
            capture.__enter__()

    def test_mapping_failure_cleans_up_fd_and_mapping(self) -> None:
        page = mmap.PAGESIZE
        data = FakeMapping(page * 2)
        with mock.patch("cpu2tensor.hardware.platform.system", return_value="Linux"):
            capture = PerfCapture(HardwareConfig("kernel", cpus=(0,), signal="intel_pt", data_pages=1))
        with mock.patch("cpu2tensor.hardware._attribute", return_value=_PerfAttr()), \
                mock.patch("cpu2tensor.hardware._open_event", return_value=11), \
                mock.patch("cpu2tensor.hardware.mmap.mmap", side_effect=[data, OSError("aux failed")]), \
                mock.patch("cpu2tensor.hardware.os.close") as close:
            with self.assertRaisesRegex(OSError, "aux failed"):
                capture.__enter__()
        self.assertTrue(data.closed)
        close.assert_called_once_with(11)


if __name__ == "__main__":
    unittest.main()
