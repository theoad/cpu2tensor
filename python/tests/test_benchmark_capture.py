# SPDX-License-Identifier: AGPL-3.0-only
"""The timing fixture checks the runner, not QEMU performance or register values."""

import json
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile
import unittest

from cpu2tensor.examples.benchmark_capture import (
    MIXED_FEATURE, TraceCounter, benchmark, file_contains, run_case, summarize, validate_manifest,
)


def frame(kind, source=0, count=0, sequence=0, detail=0, payload=b""):
    return struct.pack("<IHHIIQQ", 0x31543243, 2, kind, source, count, sequence, detail) + payload


def capture():
    return (frame(1, detail=2) + frame(2, count=2, payload=struct.pack("<QQ", 11, 22)) +
            frame(3, sequence=2) + frame(4))


def mixed(runs, *, source=0, sequence=0, count=None):
    payload = b"".join(struct.pack("<HHI", kind, rows, len(data)) + data
                       for kind, rows, data in runs)
    return frame(14, source=source, sequence=sequence,
                 count=sum(rows for _, rows, _ in runs) if count is None else count,
                 detail=len(payload), payload=payload)


def command(source, trace=False):
    return {"name": "capture" if trace else "vanilla", "signals": "fixture",
            "argv": [sys.executable, "-c", source, *( ["{trace_fd}"] if trace else [])]}


class TraceCounterTests(unittest.TestCase):
    def test_fragmented_stream_preserves_counts_without_rows(self):
        stream = capture()
        counter = TraceCounter()
        for begin in range(0, len(stream), 7):
            counter.feed(stream[begin:begin + 7])
        counter.finish()
        self.assertEqual(counter.result(), {"bytes": len(stream), "frames": 4,
                                           "run_headers": 0, "frame_header_bytes": 128,
                                           "run_header_bytes": 0, "framing_bytes": 128,
                                           "events": {"blocks": 2}, "sources": 1, "complete": True})

    def test_mixed_signal_runs_count_events_but_only_one_outer_frame(self):
        features = 2 | MIXED_FEATURE | (1 << 8) | (1 << 9) | (1 << 12) | (1 << 15) | (1 << 16)
        runs = [(12, 1, bytes(64)), (7, 2, bytes(48)),
                (8, 2, bytes(96)), (2, 3, bytes(24))]
        stream = frame(1, detail=features) + mixed(runs) + frame(3, sequence=8) + frame(4)
        counter = TraceCounter()
        for offset in range(0, len(stream), 7):
            counter.feed(stream[offset:offset + 7])
        counter.finish()
        result = counter.result()
        self.assertEqual(result["events"], {"address_context": 1, "registers": 2, "memory": 2, "blocks": 3})
        self.assertEqual(result["frames"], 4)
        self.assertEqual(result["run_headers"], 4)
        self.assertEqual(result["frame_header_bytes"], 128)
        self.assertEqual(result["run_header_bytes"], 32)
        self.assertEqual(result["framing_bytes"], 160)
        self.assertEqual(result["bytes"], len(stream))

    def test_mixed_and_ordinary_frames_share_each_source_sequence(self):
        block = [(2, 1, bytes(8))]
        stream = (frame(1, detail=2 | MIXED_FEATURE)
                  + mixed(block, source=1) + mixed(block)
                  + frame(2, source=1, count=1, sequence=1, payload=bytes(8))
                  + mixed(block, sequence=1)
                  + frame(3, source=1, sequence=2) + frame(3, sequence=2) + frame(4))
        counter = TraceCounter()
        counter.feed(stream)
        counter.finish()
        self.assertEqual(counter.result()["sources"], 2)
        self.assertEqual(counter.result()["events"], {"blocks": 4})
        self.assertEqual(counter.result()["frames"], 8)
        self.assertEqual(counter.result()["run_headers"], 3)

    def test_mixed_invalid_framing_is_rejected(self):
        valid = mixed([(2, 1, bytes(8))])
        invalid = {
            "unadvertised": frame(1, detail=2) + valid,
            "nested": mixed([(14, 1, bytes(16))]),
            "control": mixed([(4, 1, bytes(8))]),
            "empty run": mixed([(2, 0, bytes(8))], count=1),
            "outer count too small": mixed([(2, 2, bytes(16))], count=1),
            "outer count too large": mixed([(2, 1, bytes(8))], count=2),
            "bad block bytes": mixed([(2, 1, bytes(9))]),
            "bad register bytes": mixed([(7, 1, bytes(16))]),
            "bad memory bytes": mixed([(8, 1, bytes(25))]),
            "bad context count": mixed([(12, 2, bytes(128))]),
            "truncated run payload": frame(14, count=1, detail=16,
                                            payload=struct.pack("<HHI", 2, 1, 16) + bytes(8)),
            "partial run header": frame(14, count=1, detail=20,
                                         payload=struct.pack("<HHI", 2, 1, 8) + bytes(12)),
            "oversized count": frame(14, count=257, detail=16),
            "oversized bytes": frame(14, count=1, detail=4065),
            "sequence gap": mixed([(2, 1, bytes(8))], sequence=1),
            "after source end": frame(3) + valid,
        }
        for name, data in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                counter = TraceCounter()
                counter.feed((b"" if name == "unadvertised" else frame(1, detail=2 | MIXED_FEATURE)) + data)

    def test_sequence_gap_unended_source_partial_and_failure_are_rejected(self):
        invalid = [
            (frame(1, detail=2) + frame(2, count=1, sequence=3, payload=b"\0" * 8), "missing"),
            (frame(1, detail=2) + frame(2, count=1, payload=b"\0" * 8) + frame(4), "every source"),
            (capture()[:-1], "partial frame"),
            (capture()[:-32], "without Complete"),
            (frame(1, detail=2) + frame(5, detail=1), "trace failure"),
            (capture() + frame(4), "after Complete"),
        ]
        for data, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                counter = TraceCounter()
                counter.feed(data)
                counter.finish()

    def test_many_sources_do_not_require_global_order(self):
        counter = TraceCounter()
        counter.feed(frame(1, detail=2) +
                     frame(2, source=1, count=1, payload=b"\0" * 8) +
                     frame(2, count=1, payload=b"\0" * 8) +
                     frame(3, sequence=1) + frame(3, source=1, sequence=1) + frame(4))
        counter.finish()
        self.assertEqual(counter.result()["sources"], 2)

    def test_oversized_header_fails_before_payload_allocation(self):
        with self.assertRaises(ValueError):
            TraceCounter().feed(frame(8, count=1, detail=2**63))


class RunnerTests(unittest.TestCase):
    def run_fixture(self, case, timeout=5):
        with tempfile.TemporaryDirectory() as name:
            return run_case(case, Path(name), Path(name), timeout)

    def test_noisy_stdout_and_stderr_cannot_fill_pipes(self):
        source = ("import os,sys; "
                  "os.write(1,b'o'*1048576); os.write(2,b'e'*1048576); "
                  f"data={capture()!r}; fd=int(sys.argv[1]); "
                  "[os.write(fd,data[i:i+7]) for i in range(0,len(data),7)]; "
                  "os.write(1,b'workload succeeded')")
        case = command(source, trace=True)
        case["success_stdout"] = "workload succeeded"
        result = self.run_fixture(case)
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["stdout_bytes"], 1048576 + len("workload succeeded"))
        self.assertEqual(result["stderr_bytes"], 1048576)
        self.assertEqual(result["trace"]["events"], {"blocks": 2})
        self.assertGreater(result["peak_rss_bytes"], 0)

    def test_sealed_trace_with_failed_process_is_not_success(self):
        result = self.run_fixture(command(
            f"import os,sys; os.write(int(sys.argv[1]),{capture()!r}); sys.exit(7)", True))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 7)
        self.assertTrue(result["trace"]["complete"])

    def test_process_success_with_partial_trace_is_not_success(self):
        result = self.run_fixture(command(
            f"import os,sys; os.write(int(sys.argv[1]),{capture()[:-32]!r})", True))
        self.assertEqual(result["status"], "failed")
        self.assertIn("without Complete", result["error"])

    def test_missing_workload_success_marker_is_failure(self):
        case = command("print('panic')")
        case["success_stdout"] = "all workloads complete"
        self.assertEqual(self.run_fixture(case)["status"], "failed")

    def test_timeout_kills_and_reaps_child(self):
        result = self.run_fixture(command("import time; time.sleep(60)"), timeout=0.05)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["exit_code"], -9)
        self.assertLess(result["elapsed_seconds"], 5)

    @unittest.skipUnless(hasattr(os, "fork"), "POSIX process group fixture")
    def test_descendant_holding_trace_open_is_bounded(self):
        source = ("import os,sys,time; "
                  f"os.write(int(sys.argv[1]),{capture()!r}); "
                  "pid=os.fork(); time.sleep(60) if pid == 0 else None")
        result = self.run_fixture(command(source, True), timeout=0.2)
        self.assertEqual(result["status"], "timeout")
        self.assertLess(result["elapsed_seconds"], 5)

    def test_missing_command_is_failure_with_artifacts(self):
        case = command("")
        case["argv"] = ["/nonexistent-cpu2tensor-fixture"]
        self.assertEqual(self.run_fixture(case)["status"], "failed")

    def test_success_text_can_cross_a_read_boundary(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "console"
            path.write_bytes(b"x" * 65534 + b"success")
            self.assertTrue(file_contains(path, b"success"))


class ReportTests(unittest.TestCase):
    def test_medians_and_failure_do_not_cherry_pick_repeats(self):
        runs = [{"case": "vanilla", "status": "ok", "elapsed_seconds": value} for value in (1, 3, 2)]
        runs += [{"case": "rich", "status": "ok", "elapsed_seconds": value,
                  "trace": {"events": {"blocks": 100, "memory": 300}, "bytes": 4000,
                            "frames": 20, "run_header_bytes": 160}}
                 for value in (4, 8, 6)]
        result = summarize(runs, "vanilla")
        self.assertEqual(result["rich"]["slowdown_vs_baseline"], 3)
        self.assertAlmostEqual(result["rich"]["median_events_per_second"], 400 / 6)
        self.assertEqual(result["rich"]["median_framing_fraction"], (20 * 32 + 160) / 4000)
        runs[-1]["status"] = "timeout"
        self.assertNotIn("median_seconds", summarize(runs, "vanilla")["rich"])
        runs[0]["status"] = "failed"
        self.assertNotIn("slowdown_vs_baseline", summarize(runs, "vanilla")["vanilla"])

    def test_manifest_repeats_artifacts_and_literal_argv(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = {"host_name": "fixture", "host_isa": platform.machine(), "guest_isa": "fixture",
                        "workload": "echo a literal argument", "baseline": "vanilla", "repeats": 2,
                        "cases": [{"name": "vanilla", "signals": "none",
                                   "argv": [sys.executable, "-c", "import sys; print(sys.argv[1])", "$(false)"]}]}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            result = benchmark(path, root / "results")
            self.assertEqual(len(result["runs"]), 2)
            self.assertEqual(result["summary"]["vanilla"]["successful_runs"], 2)
            self.assertEqual((root / "results/run-000/stdout.log").read_text(), "$(false)\n")
            self.assertTrue((root / "results/results.json").is_file())
            with self.assertRaises(FileExistsError):
                benchmark(path, root / "results")

    def test_invalid_manifest_fails_before_launch(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            validate_manifest([])
        manifest = {"host_name": "fixture", "host_isa": "wrong-isa", "guest_isa": "fixture",
                    "workload": "fixture", "baseline": "vanilla", "cases": []}
        with self.assertRaisesRegex(ValueError, "cross-ISA"):
            validate_manifest(manifest)
        manifest.update(host_isa=platform.machine(), cases=[command("print('ok')")], timeout_seconds=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
