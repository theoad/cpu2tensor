# SPDX-License-Identifier: AGPL-3.0-only
"""Hardware record semantics and bounded in-memory capture ownership."""

from array import array
import ctypes
import mmap
import struct
import unittest
from unittest import mock

import torch

from cpu2tensor.hardware import (
    HardwareBatch, HardwareCaptureError, HardwareConfig, HardwareDecodeSideband,
    HardwareMultimodalConfig,
    HardwareTraceLost, PerfCapture, PerfMultimodalCapture, PerfMultimodalSession,
    _PerfAttr, _attribute,
    _MultimodalSource, _counter_attribute, _counter_batch, _decode_records,
    _event_id, _read_counter_group, _ring_bytes,
    _open_event, _read_sampled_event, _source_value, _syscall_number, _tensor,
)


class FakeMapping(bytearray):
    def close(self) -> None:
        self.closed = True


def record(kind: int, payload: bytes = b"", *, misc: int = 0) -> bytes:
    return struct.pack("<IHH", kind, misc, 8 + len(payload)) + payload


class HardwareTests(unittest.TestCase):
    def test_multimodal_config_requires_a_ready_process_and_known_sources(self) -> None:
        config = HardwareMultimodalConfig("process_kernel", 7)
        self.assertEqual(
            config.modalities,
            ("intel_pt", "memory_loads", "counters"),
        )
        invalid = (
            dict(scope="kernel", pid=7),
            dict(scope="process", pid=0),
            dict(scope="process", pid=7, modalities=()),
            dict(scope="process", pid=7, modalities=("counters", "counters")),
            dict(scope="process", pid=7, modalities=("unknown",)),
            dict(scope="process", pid=7, pebs_period=0),
            dict(scope="process", pid=7, data_pages=3),
            dict(scope="process", pid=7, aux_pages=0),
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                HardwareMultimodalConfig(**options)

    def test_config_rejects_ambiguous_scope_and_unbounded_sizes(self) -> None:
        invalid = (
            dict(scope="other"), dict(scope="process"), dict(scope="process", pid=-1),
            dict(scope="process", pid=1, cpus=(0,)), dict(scope="kernel"),
            dict(scope="process_kernel"), dict(scope="process_kernel", pid=1, cpus=(0,)),
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
        process_kernel = _attribute(
            HardwareConfig("process_kernel", pid=7, signal="instructions")
        )
        kernel = _attribute(HardwareConfig("kernel", cpus=(0,), signal="cycles"))
        self.assertEqual((process.type, process.config, process.sample_period), (0, 1, 100_000))
        self.assertEqual((kernel.type, kernel.config), (0, 0))
        self.assertEqual(process.size, ctypes.sizeof(_PerfAttr))
        self.assertEqual(process.flags & ((1 << 1) | (1 << 5) | (1 << 6)),
                         (1 << 5) | (1 << 6))
        self.assertEqual(
            process_kernel.flags & ((1 << 4) | (1 << 5) | (1 << 6)),
            (1 << 4) | (1 << 6),
        )
        self.assertEqual(kernel.flags & ((1 << 4) | (1 << 6)), (1 << 4) | (1 << 6))
        self.assertFalse(kernel.flags & (1 << 5))
        for attr in (process, process_kernel, kernel):
            self.assertTrue(attr.flags & (1 << 25))
            self.assertEqual(attr.clockid, 4)

    def test_boundary_counters_are_pinned_group_reads_not_samples(self) -> None:
        leader = _counter_attribute("process_kernel", "instructions", leader=True)
        member = _counter_attribute("process_kernel", "cycles", leader=False)
        reference = _counter_attribute("process", "ref_cycles", leader=False)
        self.assertEqual((leader.type, leader.config, leader.sample_period), (0, 1, 0))
        self.assertEqual((member.type, member.config), (0, 0))
        self.assertEqual(reference.config, 9)
        self.assertEqual(leader.read_format, 15)
        self.assertTrue(leader.flags & (1 << 2))
        self.assertFalse(member.flags & (1 << 2))
        self.assertTrue(leader.flags & (1 << 4))
        self.assertTrue(reference.flags & (1 << 5))
        with self.assertRaises(ValueError):
            _counter_attribute("kernel", "cycles", leader=True)
        with self.assertRaises(ValueError):
            _counter_attribute("process", "branches", leader=True)

    def test_vendor_sources_are_probed_and_precise_memory_is_required(self) -> None:
        def read_text(path, *args, **kwargs):
            name = str(path)
            if name.endswith("/type"):
                return "11"
            if name.endswith("/events/mem-loads"):
                return "event=0xcd,umask=0x1,ldlat=3"
            if name.endswith("/events/mem-stores"):
                return "event=0xd0,umask=0x82"
            if name.endswith("/format/pt"):
                return "config:0"
            if name.endswith("/format/branch"):
                return "config:13"
            raise AssertionError(name)

        with mock.patch("pathlib.Path.read_text", read_text):
            memory = _attribute(HardwareConfig("kernel", cpus=(0,), signal="memory_loads"))
            stores = _attribute(HardwareConfig("kernel", cpus=(0,), signal="memory_stores"))
            pt = _attribute(HardwareConfig("process", pid=7, signal="intel_pt"))
        self.assertEqual((memory.type, memory.config, memory.config1), (11, 0x1CD, 3))
        self.assertEqual((stores.type, stores.config, stores.config1), (11, 0x82D0, 0))
        self.assertEqual((memory.flags >> 15) & 3, 2)
        self.assertTrue(memory.flags & (1 << 2))
        self.assertEqual(memory.read_format, 3)
        self.assertEqual((pt.type, pt.config, pt.sample_type, pt.sample_period), (11, 8193, 134, 0))
        for bit in (8, 9, 13, 17, 18, 23, 24, 25, 26, 28, 29, 30, 33, 34):
            self.assertTrue(pt.flags & (1 << bit))
        self.assertEqual(pt.clockid, 4)
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
        with mock.patch("pathlib.Path.read_text", side_effect=["4", "other"]):
            with self.assertRaises(HardwareCaptureError):
                _attribute(HardwareConfig("kernel", cpus=(0,), signal="memory_stores"))
        with mock.patch("pathlib.Path.read_text", side_effect=["11", "config:2", "config:13"]):
            with self.assertRaises(HardwareCaptureError):
                _attribute(HardwareConfig("kernel", cpus=(0,), signal="intel_pt"))
        with mock.patch("pathlib.Path.read_text", side_effect=["11", "config:0", "config:12"]):
            with self.assertRaises(HardwareCaptureError):
                _attribute(HardwareConfig("kernel", cpus=(0,), signal="intel_pt"))
        for signal in ("memory_loads", "memory_stores", "intel_pt"):
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
            with self.assertRaises(HardwareCaptureError):
                PerfMultimodalCapture(HardwareMultimodalConfig("process", 7))

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
            self.assertEqual(_open_event(_PerfAttr(), 3, 2, 9), 17)
            syscall.return_value = -1
            with mock.patch("cpu2tensor.hardware.ctypes.get_errno", return_value=1):
                with self.assertRaisesRegex(HardwareCaptureError, "Operation not permitted"):
                    _open_event(_PerfAttr(), 3, -1)

    def test_counter_group_reads_are_identity_checked(self) -> None:
        payload = struct.pack("<QQQQQQQ", 2, 100, 100, 11, 31, 17, 29)
        with mock.patch("cpu2tensor.hardware.os.read", return_value=payload):
            values, enabled, running = _read_counter_group(7, (29, 31))
        self.assertEqual((values, enabled, running), ((17, 11), 100, 100))
        for bad in (
            b"",
            struct.pack("<QQQQQQQ", 1, 100, 100, 11, 31, 17, 29),
            struct.pack("<QQQQQQQ", 2, 100, 100, 11, 31, 17, 31),
            struct.pack("<QQQQQQQ", 2, 100, 100, 11, 41, 17, 29),
        ):
            with mock.patch("cpu2tensor.hardware.os.read", return_value=bad):
                with self.subTest(payload=bad), self.assertRaises(HardwareCaptureError):
                    _read_counter_group(7, (29, 31))
        with mock.patch("cpu2tensor.hardware.os.read", side_effect=OSError("gone")):
            with self.assertRaisesRegex(HardwareCaptureError, "read the PMU"):
                _read_counter_group(7, (29, 31))

    def test_sampled_event_scheduling_is_complete_and_not_multiplexed(self) -> None:
        with mock.patch(
            "cpu2tensor.hardware.os.read",
            return_value=struct.pack("<QQQ", 17, 100, 100),
        ):
            self.assertEqual(_read_sampled_event(7), (17, 100, 100))
        for payload, message in (
            (b"", "complete"),
            (struct.pack("<QQQ", 0, 0, 0), "did not run"),
            (struct.pack("<QQQ", 17, 100, 90), "multiplexed"),
        ):
            with mock.patch("cpu2tensor.hardware.os.read", return_value=payload), \
                    self.subTest(payload=payload), \
                    self.assertRaisesRegex(HardwareCaptureError, message):
                _read_sampled_event(7)
        with mock.patch("cpu2tensor.hardware.os.read", side_effect=OSError("gone")):
            with self.assertRaisesRegex(HardwareCaptureError, "PEBS scheduling"):
                _read_sampled_event(7)

    def test_counter_event_identity_is_explicit(self) -> None:
        def identified(fd, command, identifier, mutate):
            identifier[0] = 19

        with mock.patch("cpu2tensor.hardware.fcntl.ioctl", side_effect=identified):
            self.assertEqual(_event_id(7), 19)
        with mock.patch("cpu2tensor.hardware.fcntl.ioctl", side_effect=OSError("gone")):
            with self.assertRaisesRegex(HardwareCaptureError, "identify"):
                _event_id(7)
        with mock.patch("cpu2tensor.hardware.fcntl.ioctl"):
            with self.assertRaisesRegex(HardwareCaptureError, "invalid"):
                _event_id(7)

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
        self.assertEqual(result.perf_records.tolist(), list(data))
        self.assertEqual(_tensor(array("q")).dtype, torch.int64)
        self.assertEqual(_tensor(bytearray()).dtype, torch.uint8)
        stores = _decode_records(data, "memory_stores", 2, b"")
        self.assertEqual(stores.address.tolist(), [0xABC])
        self.assertEqual(stores.weight.tolist(), [77])
        self.assertEqual(stores.data_source.tolist(), [9])
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
        with mock.patch("cpu2tensor.hardware._attribute", return_value=_PerfAttr()), \
                mock.patch("cpu2tensor.hardware.os.listdir", side_effect=OSError("gone")):
            with mock.patch("cpu2tensor.hardware.platform.system", return_value="Linux"):
                missing = PerfCapture(config)
            with self.assertRaisesRegex(HardwareCaptureError, "enumerate process 43"):
                missing.__enter__()

    def test_multimodal_capture_owns_all_sources_and_one_clock_envelope(self) -> None:
        page = mmap.PAGESIZE
        pt_record = record(11, struct.pack("<QQQ", 0, 4, 0))
        pt_data = FakeMapping(page * 2)
        pt_aux = FakeMapping(page)
        pt_aux[:4] = b"PT!!"
        struct.pack_into(
            "<8Q", pt_data, 1024,
            len(pt_record), 0, page, page, 4, 0, page * 2, page,
        )
        pt_data[page:page + len(pt_record)] = pt_record

        sample = struct.pack("<QIIQQIIQQQ", 0x123, 43, 43, 17, 0xABC, 2, 0, 100, 9, 7)
        pebs_record = record(9, sample, misc=1 << 14)
        pebs_data = FakeMapping(page * 2)
        struct.pack_into(
            "<8Q", pebs_data, 1024,
            len(pebs_record), 0, page, page, 0, 0, 0, 0,
        )
        pebs_data[page:page + len(pebs_record)] = pebs_record

        def counter_read(enabled, running, values):
            fields = [3, enabled, running]
            for value, identifier in zip(values, (101, 102, 103)):
                fields.extend((value, identifier))
            return struct.pack("<9Q", *fields)

        counter_reads = [
            counter_read(0, 0, (0, 0, 0)),
            counter_read(50, 50, (11, 13, 17)),
            struct.pack("<QQQ", 101, 50, 50),
        ]
        config = HardwareMultimodalConfig(
            "process_kernel", 43, data_pages=1, aux_pages=1,
        )
        calls = []

        def ioctl(fd, command, arg=0, mutate=False):
            calls.append((fd, command, arg))

        with mock.patch("cpu2tensor.hardware.platform.system", return_value="Linux"):
            capture = PerfMultimodalCapture(config)
        with mock.patch("cpu2tensor.hardware.os.listdir", return_value=["43"]), \
                mock.patch("cpu2tensor.hardware._attribute", return_value=_PerfAttr()), \
                mock.patch("cpu2tensor.hardware._open_event", side_effect=[10, 11, 12, 13, 14]), \
                mock.patch("cpu2tensor.hardware._event_id", side_effect=[101, 102, 103]), \
                mock.patch("cpu2tensor.hardware.mmap.mmap",
                           side_effect=[pt_data, pt_aux, pebs_data]), \
                mock.patch("cpu2tensor.hardware.time.clock_gettime_ns",
                           side_effect=[100, 110, 200, 210]), \
                mock.patch("cpu2tensor.hardware.os.read", side_effect=counter_reads), \
                mock.patch("cpu2tensor.hardware.fcntl.ioctl", side_effect=ioctl), \
                mock.patch("cpu2tensor.hardware.os.close") as close, \
                mock.patch(
                    "cpu2tensor.hardware._capture_decode_sideband",
                    return_value=HardwareDecodeSideband(
                        "CLOCK_MONOTONIC_RAW", 90, b"maps", b"modules",
                        b"symbols", b"{}", b"attr", "state",
                    ),
                ):
            with capture:
                batches = capture.stop()
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual((batch.source, batch.tid, batch.cpu), (43, 43, -1))
        self.assertEqual(
            (batch.envelope.clock, batch.envelope.arm_before_ns,
             batch.envelope.arm_after_ns, batch.envelope.stop_before_ns,
             batch.envelope.stop_after_ns),
            ("CLOCK_MONOTONIC_RAW", 100, 110, 200, 210),
        )
        self.assertEqual(batch.pt.trace_bytes.tolist(), list(b"PT!!"))
        self.assertEqual(batch.pebs.address.tolist(), [0xABC])
        self.assertEqual(batch.pebs.cpu.tolist(), [2])
        self.assertEqual(batch.counters.names, ("instructions", "cycles", "ref_cycles"))
        self.assertEqual(batch.counters.values.tolist(), [11, 13, 17])
        self.assertEqual(
            (batch.counters.time_enabled_ns, batch.counters.time_running_ns),
            (50, 50),
        )
        self.assertTrue(all(row.requested and row.available and not row.lost
                            for row in batch.status))
        pebs_status = next(row for row in batch.status if row.signal == "memory_loads")
        self.assertEqual(
            (pebs_status.time_enabled_ns, pebs_status.time_running_ns),
            (50, 50),
        )
        self.assertIn((12, 0x2400, 1), calls)
        self.assertIn((12, 0x2401, 1), calls)
        self.assertTrue(pt_data.closed)
        self.assertTrue(pt_aux.closed)
        self.assertTrue(pebs_data.closed)
        self.assertEqual(close.call_count, 5)
        with self.assertRaises(RuntimeError):
            capture.__enter__()
        with self.assertRaises(RuntimeError):
            capture.stop()

        with mock.patch("cpu2tensor.hardware.platform.system", return_value="Linux"):
            missing = PerfMultimodalCapture(config)
        with mock.patch("cpu2tensor.hardware.os.listdir", side_effect=OSError("gone")):
            with self.assertRaisesRegex(HardwareCaptureError, "enumerate process 43"):
                missing.__enter__()

    def test_multimodal_session_reuses_sources_and_consumes_each_window(self) -> None:
        page = mmap.PAGESIZE
        pt_record_size = len(record(11, struct.pack("<QQQ", 0, 4, 0)))
        pebs_payload = struct.pack(
            "<QIIQQIIQQQ", 0x123, 43, 43, 17, 0xABC, 2, 0, 100, 9, 7,
        )
        pebs_record = record(9, pebs_payload, misc=1 << 14)
        pt_data = FakeMapping(page * 2)
        pt_aux = FakeMapping(page)
        pebs_data = FakeMapping(page * 2)
        struct.pack_into("<8Q", pt_data, 1024, 0, 0, page, page, 0, 0, page * 2, page)
        struct.pack_into("<8Q", pebs_data, 1024, 0, 0, page, page, 0, 0, 0, 0)

        def counter_read(enabled, values):
            fields = [3, enabled, enabled]
            for value, identifier in zip(values, (101, 102, 103)):
                fields.extend((value, identifier))
            return struct.pack("<9Q", *fields)

        reads = []
        for _ in range(2):
            reads.extend((
                counter_read(0, (0, 0, 0)),
                counter_read(50, (11, 13, 17)),
                struct.pack("<QQQ", 1, 50, 50),
            ))
        config = HardwareMultimodalConfig(
            "process_kernel", 43, data_pages=1, aux_pages=1,
        )
        with mock.patch("cpu2tensor.hardware.platform.system", return_value="Linux"):
            session = PerfMultimodalSession(config)
        with mock.patch("cpu2tensor.hardware.os.listdir", return_value=["43"]), \
                mock.patch("cpu2tensor.hardware._attribute", return_value=_PerfAttr()), \
                mock.patch("cpu2tensor.hardware._open_event", side_effect=[10, 11, 12, 13, 14]), \
                mock.patch("cpu2tensor.hardware._event_id", side_effect=[101, 102, 103]), \
                mock.patch("cpu2tensor.hardware.mmap.mmap",
                           side_effect=[pt_data, pt_aux, pebs_data]), \
                mock.patch("cpu2tensor.hardware.time.clock_gettime_ns",
                           side_effect=range(100, 180, 10)), \
                mock.patch("cpu2tensor.hardware.os.read", side_effect=reads), \
                mock.patch("cpu2tensor.hardware.fcntl.ioctl"), \
                mock.patch("cpu2tensor.hardware.os.close") as close, \
                mock.patch(
                    "cpu2tensor.hardware._capture_decode_sideband",
                    return_value=HardwareDecodeSideband(
                        "CLOCK_MONOTONIC_RAW", 90, b"maps", b"modules",
                        b"symbols", b"{}", b"attr", "state",
                    ),
                ):
            with session:
                results = []
                for window in range(2):
                    session.start()
                    aux_offset = window * 4
                    record_offset = window * pt_record_size
                    pt_record = record(11, struct.pack("<QQQ", aux_offset, 4, 0))
                    pt_data[page + record_offset:page + record_offset + len(pt_record)] = pt_record
                    pt_aux[aux_offset:aux_offset + 4] = b"PT!!"
                    struct.pack_into("<Q", pt_data, 1024, record_offset + len(pt_record))
                    struct.pack_into("<Q", pt_data, 1024 + 32, aux_offset + 4)
                    pebs_offset = window * len(pebs_record)
                    pebs_data[page + pebs_offset:page + pebs_offset + len(pebs_record)] = pebs_record
                    struct.pack_into("<Q", pebs_data, 1024, pebs_offset + len(pebs_record))
                    results.append(session.stop()[0])
        self.assertEqual([row.pt.trace_bytes.tolist() for row in results], [
            list(b"PT!!"), list(b"PT!!"),
        ])
        self.assertEqual([row.pebs.address.tolist() for row in results], [[0xABC], [0xABC]])
        self.assertEqual(struct.unpack_from("<Q", pt_data, 1024 + 8)[0], 2 * pt_record_size)
        self.assertEqual(struct.unpack_from("<Q", pt_data, 1024 + 40)[0], 8)
        self.assertEqual(struct.unpack_from("<Q", pebs_data, 1024 + 8)[0], 2 * len(pebs_record))
        self.assertEqual(close.call_count, 5)
        with self.assertRaises(RuntimeError):
            session.start()

    def test_counter_delta_rejects_multiplexing_and_backward_values(self) -> None:
        source = _MultimodalSource(
            tid=7,
            counter_start=((10, 20, 30), 100, 100),
        )
        with self.assertRaisesRegex(HardwareCaptureError, "multiplexed"):
            _counter_batch(source, ((11, 21, 31), 200, 190))
        with self.assertRaisesRegex(HardwareCaptureError, "moved backwards"):
            _counter_batch(source, ((9, 21, 31), 200, 200))
        with self.assertRaisesRegex(HardwareCaptureError, "did not run"):
            _counter_batch(source, ((11, 21, 31), 100, 100))
        with self.assertRaisesRegex(HardwareCaptureError, "no boundary baseline"):
            _counter_batch(_MultimodalSource(tid=8), ((1, 2, 3), 10, 10))

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
