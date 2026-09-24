# SPDX-License-Identifier: AGPL-3.0-only
"""Windows WPR profile and memory-mode lifecycle contracts."""

import subprocess
import unittest
from unittest import mock
from xml.etree import ElementTree

from cpu2tensor import HardwareCaptureError, WprCapture, WprConfig
from cpu2tensor.windows_hardware import _profile, _run_wpr


class WprTests(unittest.TestCase):
    def test_config_and_profile_contain_only_memory_capture_controls(self) -> None:
        for options in (
            dict(signal="unknown"), dict(code_mode="Both"),
            dict(buffer_kb=0), dict(buffers=0), dict(period=0),
            dict(signal="cycles", code_mode="Kernel"),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                WprConfig(**options)
        for signal, expected in (("cycles", "TotalCycles"),
                                 ("instructions", "InstructionRetired"),
                                 ("processor_trace", "ProcessorTrace")):
            with self.subTest(signal=signal):
                mode = "Kernel" if signal == "processor_trace" else "UserKernel"
                root = ElementTree.fromstring(_profile(WprConfig(signal=signal, code_mode=mode)))
                profiles = root.findall("./Profiles/Profile")
                self.assertEqual({item.attrib["LoggingMode"] for item in profiles},
                                 {"Memory", "File"})
                self.assertIsNotNone(root.find("./Profiles/HardwareCounter/" +
                                           ("ProcessorTrace" if signal == "processor_trace" else "SampledCounters")))
                self.assertIn(expected, ElementTree.tostring(root).decode())
                self.assertEqual(root.find("./Profiles/SystemCollector/BufferSize").attrib["Value"], "1024")
                self.assertEqual(root.find("./Profiles/SystemCollector/Buffers").attrib["Value"], "128")

    def test_stop_exports_only_after_capture_and_reports_unknown_completeness(self) -> None:
        commands = []

        def run(*arguments):
            commands.append(arguments)
            if arguments[0] == "-stop":
                from pathlib import Path
                Path(arguments[1]).write_bytes(b"etl")

        with mock.patch("cpu2tensor.windows_hardware.platform.system", return_value="Windows"), \
                mock.patch("cpu2tensor.windows_hardware._run_wpr", side_effect=run):
            with WprCapture(WprConfig(signal="processor_trace")) as capture:
                self.assertEqual(len(commands), 1)
                self.assertEqual(commands[0][0], "-start")
                self.assertNotIn("-filemode", commands[0])
                batch = capture.stop()
                with self.assertRaises(RuntimeError):
                    capture.stop()
        self.assertEqual(commands[-1][0], "-stop")
        self.assertEqual(batch.etl_bytes.tolist(), list(b"etl"))
        self.assertFalse(batch.complete)
        with self.assertRaises(RuntimeError):
            capture.__enter__()

    def test_errors_and_cancel(self) -> None:
        with mock.patch("cpu2tensor.windows_hardware.platform.system", return_value="Darwin"):
            with self.assertRaises(HardwareCaptureError):
                WprCapture(WprConfig())
        with mock.patch("cpu2tensor.windows_hardware.subprocess.run", side_effect=OSError("missing")):
            with self.assertRaises(HardwareCaptureError):
                _run_wpr("-start", "CPU")
        failure = subprocess.CompletedProcess(["wpr"], 1, stdout="no PMU", stderr="")
        with mock.patch("cpu2tensor.windows_hardware.subprocess.run", return_value=failure):
            with self.assertRaisesRegex(HardwareCaptureError, "no PMU"):
                _run_wpr("-start", "CPU")
        commands = []
        with mock.patch("cpu2tensor.windows_hardware.platform.system", return_value="Windows"), \
                mock.patch("cpu2tensor.windows_hardware._run_wpr", side_effect=lambda *args: commands.append(args)):
            with WprCapture(WprConfig()):
                pass
        self.assertEqual([command[0] for command in commands], ["-start", "-cancel"])
        with mock.patch("cpu2tensor.windows_hardware.platform.system", return_value="Windows"), \
                mock.patch("cpu2tensor.windows_hardware._run_wpr", side_effect=HardwareCaptureError("start failed")):
            with self.assertRaisesRegex(HardwareCaptureError, "start failed"):
                WprCapture(WprConfig()).__enter__()


if __name__ == "__main__":
    unittest.main()
