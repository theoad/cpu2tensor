# SPDX-License-Identifier: AGPL-3.0-only

import argparse
import json
from pathlib import Path
import tempfile
from unittest import mock

from cpu2tensor.examples import hardware_foundation_policy as policy


def test_policy_is_restored_after_bounded_command() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        no_turbo = root / "no_turbo"
        maximum = root / "maximum"
        receipt = root / "receipt.json"
        no_turbo.write_text("0\n")
        maximum.write_text("4900000\n")
        args = argparse.Namespace(
            cpu=2, no_turbo="1", maximum_khz="1800000", receipt=receipt,
            publish_artifact=None, command=["/bin/sh", "-c", "exit 0"],
        )
        with mock.patch.object(policy, "NO_TURBO", no_turbo), \
                mock.patch.object(policy, "_maximum", return_value=maximum), \
                mock.patch.object(policy.os, "geteuid", return_value=0):
            assert policy.run(args) == 0
        assert no_turbo.read_text() == "0\n"
        assert maximum.read_text() == "4900000\n"
        evidence = json.loads(receipt.read_text())
        assert evidence["restored"] is True
        assert evidence["return_code"] == 0


def test_policy_publishes_deferred_manifest_only_after_restore() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        no_turbo = root / "no_turbo"
        maximum = root / "maximum"
        receipt = root / "receipt.json"
        artifact = root / "artifact"
        artifact.mkdir()
        (artifact / "capture-manifest.pending.json").write_text(
            '{"schema":"fixture","manifest_content_sha256":"old"}\n'
        )
        no_turbo.write_text("0\n")
        maximum.write_text("4900000\n")
        args = argparse.Namespace(
            cpu=2, no_turbo="1", maximum_khz="1800000", receipt=receipt,
            publish_artifact=artifact, command=["/bin/sh", "-c", "exit 0"],
        )
        with mock.patch.object(policy, "NO_TURBO", no_turbo), \
                mock.patch.object(policy, "_maximum", return_value=maximum), \
                mock.patch.object(policy.os, "geteuid", return_value=0):
            assert policy.run(args) == 0
        assert not (artifact / "capture-manifest.pending.json").exists()
        manifest = json.loads((artifact / "capture-manifest.json").read_text())
        assert manifest["cpu_policy_receipt"]["content"]["restored"] is True
        assert manifest["manifest_content_sha256"] != "old"
