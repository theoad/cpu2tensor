# SPDX-License-Identifier: AGPL-3.0-only
"""Run the client-owned kernel action guest as an ordinary Linux process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from cpu2tensor.examples.custom_kernel_actions import verify_result


BUILD = Path(os.environ["CPU2TENSOR_CI_BUILD"])


def test_custom_guest_runs_two_matched_actions_after_one_start() -> None:
    target = BUILD / "kernel_custom_actions"
    process = subprocess.run(
        [target, "--check"],
        input="open plain 4\nopen cloexec 4\nquit\n",
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    rows = [json.loads(line[4:]) for line in process.stdout.splitlines()
            if line.startswith("C2T ")]
    assert rows[0] == {"event": "start", "adapter": "open-sequence-v1"}
    assert [row for row in rows if row["event"] == "ready"] == [
        {"event": "ready", "step": 0},
        {"event": "ready", "step": 1},
        {"event": "ready", "step": 2},
    ]
    results = [row for row in rows if row["event"] == "result"]
    assert len(results) == 2
    verify_result("plain", 4, results[0])
    verify_result("cloexec", 4, results[1])
    assert rows[-1] == {"event": "complete", "steps": 2, "ok": True}

    symbols = subprocess.run(
        ["nm", "-g", "--defined-only", target], capture_output=True, text=True,
        check=True, timeout=5
    ).stdout.splitlines()
    names = ("cpu2tensor_action_begin", "cpu2tensor_action_end",
             "cpu2tensor_action_abort")
    addresses = {
        fields[2]: int(fields[0], 16)
        for line in symbols
        if len(fields := line.split()) == 3 and fields[2] in names
    }
    assert set(addresses) == set(names)
    assert len(set(addresses.values())) == 3
