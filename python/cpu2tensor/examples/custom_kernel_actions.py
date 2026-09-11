# SPDX-License-Identifier: AGPL-3.0-only
"""Run two client-defined syscall-sequence actions in one kernel episode."""

from __future__ import annotations

import argparse
from typing import Any

from cpu2tensor import KernelEnv


# The guest is Linux even when this controller runs on macOS.
LINUX_O_CLOEXEC = 0x80000


def open_action(variant: str, repetitions: int) -> bytes:
    """Encode one bounded action from this example's client-owned grammar."""
    if variant not in ("plain", "cloexec"):
        raise ValueError("variant must be plain or cloexec")
    if not 1 <= repetitions <= 64:
        raise ValueError("repetitions must be from 1 through 64")
    return f"open {variant} {repetitions}\n".encode("ascii")


def verify_result(variant: str, repetitions: int,
                  result: dict[str, Any] | None) -> None:
    """Reject a response that does not name the exact executed action."""
    expected_flags = 0 if variant == "plain" else LINUX_O_CLOEXEC
    expected = {
        "event": "result",
        "action": "open_sequence",
        "variant": variant,
        "repetitions": repetitions,
        "syscalls": repetitions * 3,
        "open_flags": expected_flags,
    }
    if result is None or any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"guest result does not match {expected}")


def run(endpoint: str, repetitions: int) -> list[dict[str, int | str]]:
    """Drain two matched actions and their explicit transition windows."""
    summaries: list[dict[str, int | str]] = []
    with KernelEnv(endpoint, timeout=120) as env:
        list(env.reset())
        if env.event != {"event": "ready", "step": 0}:
            raise RuntimeError(f"guest did not publish its initial ready event: {env.event}")
        for step, variant in enumerate(("plain", "cloexec")):
            blocks = 0
            windows = []
            for batch in env.step(open_action(variant, repetitions)):
                blocks += batch.addresses.numel()
                if batch.transition_window is not None:
                    windows.append(batch.transition_window)
            verify_result(variant, repetitions, env.result)
            if env.event != {"event": "ready", "step": step + 1}:
                raise RuntimeError(f"guest did not publish ready after action {step}: {env.event}")
            if len(windows) != 1 or not windows[0].complete:
                raise RuntimeError(f"action {step} has no complete trace window")
            summaries.append({"variant": variant, "repetitions": repetitions,
                              "blocks": blocks, "transitions": windows[0].observed})
        list(env.step(b"quit\n"))
        if env.exit_code != 0 or env.event != {"event": "complete", "steps": 2,
                                               "ok": True}:
            raise RuntimeError(f"guest did not complete successfully: {env.event}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--repetitions", type=int, default=8)
    args = parser.parse_args()
    for summary in run(args.endpoint, args.repetitions):
        print(summary)


if __name__ == "__main__":
    main()
