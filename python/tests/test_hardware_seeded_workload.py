# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic target input and exact invocation custody fixtures."""

from dataclasses import replace
import hashlib
from io import BytesIO
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from cpu2tensor.examples import hardware_multimodal_experiment as experiment
from cpu2tensor.hardware import HardwareDecodeSideband


class HardwareSeededWorkloadTests(unittest.TestCase):
    def test_seed_is_repeatable_split_independent_and_changes_with_dataset_seed(self) -> None:
        execution = experiment.PlannedExecution("pipe-00007", "pipe", 7, "training")
        other_split = replace(execution, partition="calibration")
        first = experiment.planned_input_seed(execution, seed=41)
        self.assertEqual(first, experiment.planned_input_seed(execution, seed=41))
        self.assertEqual(first, experiment.planned_input_seed(other_split, seed=41))
        self.assertNotEqual(first, experiment.planned_input_seed(execution, seed=42))
        self.assertEqual(experiment.input_frame(first), b"s" + first.to_bytes(8, "little"))
        self.assertEqual(experiment.input_frame(None), b"x")
        with self.assertRaises(ValueError):
            experiment.input_frame(1 << 64)

    def test_raw_payload_retains_exact_target_invocation(self) -> None:
        execution = experiment.PlannedExecution("pipe-00007", "pipe", 7, "training")
        sideband = HardwareDecodeSideband(
            "CLOCK_MONOTONIC_RAW", 9, b"maps", b"modules", b"symbols",
            b"{}", b"attr", "state",
        )
        seed = 0x0102030405060708
        stdin_bytes = experiment.input_frame(seed)
        argv = ("/tmp/hardware_kernel_workload", "pipe", "8")
        payload = experiment.raw_capture_payload(
            (), decode_sideband=sideband,
            kernel_decode_state={"kernel_state_sha256": "state"},
            execution=execution, loops=8, stdout=b"36\n", elapsed_ns=10,
            invocation_argv=argv, invocation_stdin=stdin_bytes, input_seed=seed,
        )
        invocation = payload["invocation"]
        self.assertEqual(invocation["argv"], list(argv))
        self.assertEqual(bytes(invocation["stdin"].tolist()), stdin_bytes)
        self.assertEqual(invocation["stdin_sha256"], hashlib.sha256(stdin_bytes).hexdigest())
        self.assertEqual(invocation["input_seed"], seed)

    def test_capture_sends_and_records_the_same_frame(self) -> None:
        execution = experiment.PlannedExecution("pipe-00007", "pipe", 7, "training")
        sideband = HardwareDecodeSideband(
            "CLOCK_MONOTONIC_RAW", 9, b"maps", b"modules", b"symbols",
            b"{}", b"attr", "state",
        )

        class Process:
            pid = 77
            returncode = 0

            def __init__(self):
                self.stdout = BytesIO(b"READY\n")
                self.received = None

            def communicate(self, value=None, timeout=None):
                del timeout
                self.received = value
                return b"36\n", b""

            def kill(self):
                raise AssertionError("healthy fixture must not be killed")

        class Capture:
            def __init__(self, unused_config):
                self.decode_sideband = sideband

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                pass

            def stop(self):
                return ()

        for seed in (None, 0x0102030405060708):
            process = Process()
            with self.subTest(seed=seed), \
                    mock.patch.object(experiment.subprocess, "Popen", return_value=process), \
                    mock.patch.object(experiment, "PerfMultimodalCapture", Capture), \
                    mock.patch.object(experiment.os, "sched_getaffinity", create=True, return_value={2}), \
                    mock.patch.object(experiment, "_validate_capture", return_value={}):
                captured = experiment.capture_execution(
                    Path("/tmp/hardware_kernel_workload"), execution,
                    loops=8, target_cpu=2, data_pages=64, aux_pages=2048,
                    timeout=1.0, input_seed=seed,
                )
            self.assertEqual(process.received, experiment.input_frame(seed))
            self.assertEqual(captured.invocation_stdin, process.received)
            self.assertEqual(captured.input_seed, seed)
            self.assertEqual(
                captured.invocation_argv,
                ("/tmp/hardware_kernel_workload", "pipe", "8"),
            )

    @unittest.skipUnless(platform.system() == "Linux" and shutil.which("cc"),
                         "native gated target requires Linux and a C compiler")
    def test_native_legacy_seeded_repeat_and_short_frame(self) -> None:
        source = Path(__file__).resolve().parents[2] / "native/examples/hardware_kernel_workload.c"
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "hardware_kernel_workload"
            subprocess.run(
                ("cc", "-std=c17", "-D_GNU_SOURCE", "-Wall", "-Wextra",
                 "-Wpedantic", "-Werror", str(source), "-o", str(binary)),
                check=True,
            )

            def run(stdin_bytes: bytes) -> tuple[int, bytes]:
                process = subprocess.Popen(
                    (str(binary), "pipe", "8"), stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                assert process.stdout is not None
                self.assertEqual(process.stdout.readline(), b"READY\n")
                output, error = process.communicate(stdin_bytes, timeout=5)
                self.assertEqual(error, b"")
                return process.returncode, output

            self.assertEqual(run(b"x"), (0, b"960\n"))
            first = experiment.input_frame(0x0102030405060708)
            second = experiment.input_frame(0x1111111111111111)
            self.assertEqual(run(first), (0, b"36\n"))
            self.assertEqual(run(first), (0, b"36\n"))
            self.assertEqual(run(second), (0, b"136\n"))
            self.assertEqual(run(b"s\x01"), (3, b""))


if __name__ == "__main__":
    unittest.main()
