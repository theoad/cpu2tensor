# SPDX-License-Identifier: AGPL-3.0-only
"""Offline Mac custody checks; synthetic evidence and no hardware access."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from cpu2tensor.examples import hardware_seeded_mac_custody_r2 as custody


class HardwareSeededMacCustodyR2Tests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, str, str, str]:
        spool = root / "copied-spool"
        artifact = spool / "smoke"
        for name in ("raw", "derived", "decode"):
            (artifact / name).mkdir(parents=True, exist_ok=True)
        (spool / "collector.log").write_text("sealed\n")
        (artifact / "raw/e.pt").write_bytes(b"raw")
        (artifact / "derived/e.pt").write_bytes(b"derived")
        (artifact / "decode/kernel.pt").write_bytes(b"decode")
        (artifact / "execution-plan.json").write_text(json.dumps({"kind": "smoke"}))
        (spool / "smoke-plan.json").write_text(json.dumps({"kind": "smoke"}))
        bundle_files = {
            name: custody.guard.sha256(Path(module.__file__)) for name, module in (
                ("python/cpu2tensor/examples/hardware_multimodal_experiment.py", custody.collector),
                ("python/cpu2tensor/examples/hardware_seeded_capture_plan_r2.py", custody.plans),
                ("python/cpu2tensor/examples/hardware_seeded_capture_supervisor_r2.py", custody.guard),
            )
        }
        bundle_hash = custody._digest(bundle_files)
        (spool / "source-bundle.json").write_text(json.dumps({
            "schema": "cpu2tensor-seeded-source-bundle-r2",
            "files": bundle_files, "digest_sha256": bundle_hash,
        }))
        binary_hash = "b" * 64
        manifest = {
            "identity_sha256": "a" * 64,
            "subject": {
                "boot_id": custody.guard.EXPECTED_BOOT,
                "kernel_release": custody.guard.EXPECTED_KERNEL,
                "intel_pstate_no_turbo": "1",
                "target_scaling_max_freq_khz": custody.guard.CAPTURE_FREQ_KHZ,
                "target_cpu": 2, "controller_cpu": 3,
            },
            "event": {"pebs_event": "memory_loads", "pebs_period": 10_000},
            "build": {"workload_sha256": binary_hash},
            "entries": [{"raw_path": "raw/e.pt", "derived_path": "derived/e.pt"}],
            "kernel_decode_states": [{"path": "decode/kernel.pt",
                                      "sha256": custody.guard.sha256(artifact / "decode/kernel.pt")}],
        }
        (artifact / "capture-manifest.json").write_text(json.dumps(manifest))
        return spool, custody.guard.sha256(artifact / "capture-manifest.json"), bundle_hash, binary_hash

    def test_ack_is_outside_read_only_copy_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool, manifest_hash, bundle_hash, binary_hash = self._fixture(root)
            with mock.patch.object(custody.plans, "load_plan", return_value={"kind": "smoke"}), \
                    mock.patch.object(custody.guard, "verify_smoke_artifact") as verified:
                ack = custody.verify_copy(
                    spool, expected_manifest_sha256=manifest_hash,
                    expected_source_bundle_sha256=bundle_hash,
                    expected_binary_sha256=binary_hash,
                )
            verified.assert_called_once()
            self.assertEqual(ack["retained_raw_count"], 51)
            output = root / "ack.json"
            custody.write_ack(output, spool, ack)
            self.assertEqual(json.loads(output.read_text()), ack)
            with self.assertRaisesRegex(ValueError, "new"):
                custody.write_ack(output, spool, ack)
            with self.assertRaisesRegex(ValueError, "outside"):
                custody.write_ack(spool / "ack.json", spool, ack)

    def test_symlink_incomplete_and_tampered_copy_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool, manifest_hash, bundle_hash, binary_hash = self._fixture(root)
            arguments = dict(expected_manifest_sha256=manifest_hash,
                             expected_source_bundle_sha256=bundle_hash,
                             expected_binary_sha256=binary_hash)
            link = root / "shortcut"
            link.symlink_to(spool)
            with self.assertRaisesRegex(ValueError, "symlink"):
                custody.verify_copy(link, **arguments)
            (spool / "smoke/raw/e.pt").unlink()
            with mock.patch.object(custody.plans, "load_plan", return_value={"kind": "smoke"}), \
                    mock.patch.object(custody.guard, "verify_smoke_artifact"):
                with self.assertRaisesRegex(ValueError, "missing sealed evidence"):
                    custody.verify_copy(spool, **arguments)
            (spool / "smoke/raw/e.pt").write_bytes(b"raw")
            (spool / "smoke/capture-manifest.json").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "host SHA-256"):
                custody.verify_copy(spool, **arguments)


if __name__ == "__main__":
    unittest.main()
