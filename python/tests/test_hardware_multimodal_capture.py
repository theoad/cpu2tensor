# SPDX-License-Identifier: AGPL-3.0-only

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from cpu2tensor.examples import hardware_multimodal_capture as capture


class HardwareMultimodalCaptureTests(unittest.TestCase):
    def test_sha256_and_subject_bind_the_workload(self) -> None:
        payload = b"fixed hardware workload\n"
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "workload"
            binary.write_bytes(payload)
            with mock.patch.object(capture.platform, "node", return_value="host"), \
                    mock.patch.object(capture.platform, "machine", return_value="x86_64"), \
                    mock.patch.object(capture.platform, "release", return_value="kernel"), \
                    mock.patch.object(capture.platform, "version", return_value="version"), \
                    mock.patch.object(capture, "_cpu_model", return_value="cpu"), \
                    mock.patch.object(capture, "_read", return_value="identity"), \
                    mock.patch.object(capture, "_optional_sha256", return_value="optional"):
                subject = capture._subject(binary)

        expected = hashlib.sha256(payload).hexdigest()
        self.assertEqual(capture._sha256.__name__, "_sha256")
        self.assertEqual(subject["workload_sha256"], expected)
        self.assertEqual(subject["host"], "host")
        self.assertEqual(subject["machine"], "x86_64")
        self.assertEqual(subject["cpu_model"], "cpu")
        self.assertEqual(subject["boot_id"], "identity")
        self.assertEqual(subject["kernel_btf_sha256"], "optional")
        self.assertEqual(len(subject["capture_module_sha256"]), 64)
        self.assertEqual(len(subject["hardware_module_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
