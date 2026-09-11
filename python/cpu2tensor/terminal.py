# SPDX-License-Identifier: AGPL-3.0-only
"""Structured facts about a finished or interrupted trace."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TerminalReason(str, Enum):
    """Why a trace stopped, without depending on its socket symptom."""

    COMPLETE = "complete"
    TARGET_EXIT = "target_exit"
    MAX_RUN_DEADLINE = "max_run_deadline"
    CAPTURE_FAILURE = "capture_failure"
    TARGET_KILLED = "target_killed"
    UNSUPPORTED_TARGET = "unsupported_target"
    WORKER_TRANSPORT_FAILURE = "worker_transport_failure"
    UNKNOWN_DISCONNECTION = "unknown_disconnection"
    TRUNCATED_STREAM = "truncated_stream"
    CLIENT_TIMEOUT = "client_timeout"
    CONNECTION_FAILURE = "connection_failure"


class BoundaryProgress(str, Enum):
    """Worker-reported progress for an optional capture boundary."""

    UNKNOWN = "unknown"
    NOT_CONFIGURED = "not_configured"
    NOT_OBSERVED = "not_observed"
    OBSERVED = "observed"


class TransportEnd(str, Enum):
    """How this client observed the transport after the last valid frame."""

    CLEAN = "clean"
    RESET = "reset"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_OBSERVED = "not_observed"


@dataclass(frozen=True)
class TerminalOutcome:
    """Terminal trace facts, separated into client and worker observations."""

    reason: TerminalReason
    endpoint: str
    worker: int
    source: None
    hello_received: bool
    data_received: bool
    hello_reported: bool | None
    data_reported: bool | None
    start: BoundaryProgress
    stop: BoundaryProgress
    transport: TransportEnd
    complete: bool


class TraceTerminalError(RuntimeError):
    """A trace ended without a successful target outcome."""

    def __init__(self, message: str, outcome: TerminalOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


class TraceConnectionError(ConnectionError, TraceTerminalError):
    """A terminal trace error that remains catchable as ConnectionError."""

    def __init__(self, message: str, outcome: TerminalOutcome) -> None:
        TraceTerminalError.__init__(self, message, outcome)


class TraceTimeoutError(TimeoutError, TraceTerminalError):
    """A terminal trace error that remains catchable as TimeoutError."""

    def __init__(self, message: str, outcome: TerminalOutcome) -> None:
        TraceTerminalError.__init__(self, message, outcome)
