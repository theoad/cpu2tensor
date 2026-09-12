# SPDX-License-Identifier: AGPL-3.0-only
"""Public metadata for observing complete worker wire frames."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class WireFrame:
    """One complete frame borrowed for the duration of a wire observer call."""

    worker: int
    endpoint: str
    version: int
    kind: int
    source: int
    count: int
    sequence: int
    detail: int
    data: memoryview


WireObserver = Callable[[WireFrame], None]
