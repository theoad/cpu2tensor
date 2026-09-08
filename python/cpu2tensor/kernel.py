# SPDX-License-Identifier: AGPL-3.0-only
"""Consume a system trace and act while its guest is paused."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from cpu2tensor.batch import Batch
from cpu2tensor.stdio import StdioEnv


class KernelEnv(StdioEnv):
    """A kernel side adapter with the same streaming reset/step interface.

    A reset or step iterator ends only after QEMU has stopped every vCPU and
    the plugin has drained their captured tails, or after the target exits.
    Actions are one ASCII command line for the guest's fixed syscall adapter.
    The most recent result and event are bounded metadata, not trace history.
    Rewards and episode limits remain client decisions through ``as_gym``.
    """

    _feature = 1 << 13
    _request_kind = 10

    def __init__(self, endpoint: str, *, device: str = "cpu", timeout: float = 30.0) -> None:
        super().__init__(endpoint, device=device, timeout=timeout)
        self.event: dict[str, Any] | None = None
        self.result: dict[str, Any] | None = None

    def _clear_events(self) -> None:
        self.event = None
        self.result = None

    def _guest_event(self, payload: bytearray) -> None:
        event = json.loads(payload)
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise ValueError("Invalid guest adapter event")
        self.event = event
        if event["event"] == "result":
            self.result = event

    def step(self, action: bytes) -> Iterator[Batch]:
        """Send a single command such as ``b'memory 17 4096\\n'``."""
        if not isinstance(action, bytes):
            raise TypeError("Kernel actions must be bytes")
        if not action.endswith(b"\n") or action.count(b"\n") != 1 or any(
            value < 32 or value > 126 for value in action[:-1]
        ):
            raise ValueError("Kernel actions need one printable ASCII command and a newline")
        return super().step(action)

    def _info(self) -> dict[str, Any]:
        return {**super()._info(), "event": self.event, "result": self.result}
