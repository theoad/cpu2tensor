# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic context-only streams for concurrent Pool checks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import struct
import time

from cpu2tensor import Pool
from test_consumer import frame, worker


_CONTEXT_ONLY_MIXED = 2 | (1 << 12) | (1 << 16) | (1 << 18)
_ROWS_PER_FRAME = 256


def _run(kind: int, rows: bytes, count: int) -> bytes:
    return struct.pack("<HHI", kind, count, len(rows)) + rows


def _context(root: int) -> bytes:
    return struct.pack("<8Q", 0x1000, 1, root, 2, 3, 4, 64, 63)


def _mixed_frame(source: int, sequence: int, body: bytes) -> bytes:
    return struct.pack(
        "<IHHIIQQ", 0x31543243, 2, 14, source, _ROWS_PER_FRAME, sequence, len(body)
    ) + body


def context_only_capture(identity: int, frames_per_source: int, *, complete: bool = True) -> bytes:
    """Build two exact source streams resembling mixed kernel block capture."""
    parts = [frame(1, detail=_CONTEXT_ONLY_MIXED)]
    for frame_index in range(frames_per_source):
        for source in range(2):
            sequence = frame_index * _ROWS_PER_FRAME
            base = identity * 1_000_000_000 + source * 100_000_000
            if frame_index == 0:
                addresses = struct.pack(
                    f"<{_ROWS_PER_FRAME - 1}Q",
                    *(base + item for item in range(1, _ROWS_PER_FRAME)),
                )
                body = _run(12, _context(base + 0x8000), 1) + _run(
                    2, addresses, _ROWS_PER_FRAME - 1
                )
            else:
                addresses = struct.pack(
                    f"<{_ROWS_PER_FRAME}Q",
                    *(base + sequence + item for item in range(_ROWS_PER_FRAME)),
                )
                body = _run(2, addresses, _ROWS_PER_FRAME)
            parts.append(_mixed_frame(source, sequence, body))
    end = frames_per_source * _ROWS_PER_FRAME
    parts.extend((frame(3, source=0, sequence=end), frame(3, source=1, sequence=end)))
    if complete:
        parts.append(frame(4))
    return b"".join(parts)


@dataclass(frozen=True)
class Digest:
    identity: int
    block_rows: int
    context_rows: int
    address_sum: int


def consume(endpoint: str, identity: int, frames_per_source: int) -> Digest:
    next_sequence = [0, 0]
    block_rows = 0
    context_rows = 0
    address_sum = 0
    retained = []
    with Pool([endpoint], timeout=15) as pool:
        for batch in pool.read():
            sequences = []
            if batch.addresses.numel():
                block_sequences = batch.block_sequences.tolist()
                addresses = batch.addresses.tolist()
                sequences.extend(block_sequences)
                block_rows += len(addresses)
                address_sum += sum(addresses)
                retained.append(batch.addresses)
            if batch.context is not None:
                context_sequences = batch.context.sequences.tolist()
                sequences.extend(context_sequences)
                context_rows += len(context_sequences)
                retained.append(batch.context.cr3)
            expected = list(range(next_sequence[batch.source], next_sequence[batch.source] + len(sequences)))
            if sorted(sequences) != expected:
                raise AssertionError(f"source {batch.source} sequence mismatch")
            next_sequence[batch.source] += len(sequences)
    if next_sequence != [frames_per_source * _ROWS_PER_FRAME] * 2:
        raise AssertionError(f"source tails differ: {next_sequence}")
    if any(tensor.numel() == 0 for tensor in retained):
        raise AssertionError("retained tensor lost its storage")
    return Digest(identity, block_rows, context_rows, address_sum)


def replay(worker_count: int, frames_per_source: int, *, incomplete: int | None = None):
    """Read independent Pools concurrently and return results plus wall time."""
    with ExitStack() as stack:
        captures = [
            context_only_capture(identity, frames_per_source, complete=identity != incomplete)
            for identity in range(worker_count)
        ]
        endpoints = [
            stack.enter_context(worker(data, fragment=len(data)))
            for data in captures
        ]
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(consume, endpoint, identity, frames_per_source)
                for identity, endpoint in enumerate(endpoints)
            ]
            results = []
            errors = []
            for future in futures:
                try:
                    results.append(future.result(timeout=30))
                except BaseException as error:
                    errors.append(error)
        return results, errors, time.perf_counter() - started


def expected_digest(identity: int, frames_per_source: int) -> Digest:
    per_source = frames_per_source * _ROWS_PER_FRAME
    blocks = 2 * (per_source - 1)
    total = 0
    for source in range(2):
        base = identity * 1_000_000_000 + source * 100_000_000
        total += (per_source - 1) * base + per_source * (per_source - 1) // 2
    return Digest(identity, blocks, 2, total)
