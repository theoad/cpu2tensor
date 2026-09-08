# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded CPU collation before a batch is uploaded to the learner device."""

from dataclasses import dataclass, fields, replace

import torch

from cpu2tensor.batch import Batch


MAX_BATCH_BYTES = 4 * 1024 * 1024
MAX_SOURCE_FRAMES = 256
MAX_PENDING_FRAMES = 512


def tensor_bytes(batch: Batch) -> int:
    total = 0
    for owner in (batch, batch.registers, batch.memory, batch.context, batch.layout):
        if owner is not None:
            for field in fields(owner):
                value = getattr(owner, field.name)
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
    return total


def concatenate(values):
    return values[0] if len(values) == 1 else torch.cat(values)


def optional_column(tables, name):
    values = [getattr(table, name) for table in tables]
    if any(value is None for value in values):
        if not all(value is None for value in values):
            raise ValueError(f"Cannot collate inconsistent optional column {name}")
        return None
    return concatenate(values)


def row_sequences(batch, table=None):
    values = batch.addresses if table is None else table.pc
    sequence = batch.block_sequences if table is None else table.sequences
    if sequence is not None:
        return sequence
    # Legacy batches contain one populated table. Mixed batches already carry
    # original row positions, which must never be replaced with a dense range.
    return torch.arange(batch.first_sequence, batch.first_sequence + values.numel(), dtype=torch.int64, device="cpu")


def merge_batches(batches: list[Batch]) -> Batch:
    first = batches[0]
    if any((batch.worker, batch.source) != (first.worker, first.source) for batch in batches):
        raise ValueError("Cannot collate different workers or CPU sources")
    blocks = [batch for batch in batches if batch.addresses.numel()]
    changes = {
        "addresses": concatenate([batch.addresses for batch in blocks]) if blocks else first.addresses,
        "block_sequences": concatenate([row_sequences(batch) for batch in blocks]) if blocks else None,
    }
    for name in ("registers", "memory", "context"):
        selected = [(batch, getattr(batch, name)) for batch in batches if getattr(batch, name) is not None]
        if not selected:
            changes[name] = None
            continue
        tables = [table for _, table in selected]
        columns = {}
        for field in fields(tables[-1]):
            if field.name in ("names", "sequences") or (name == "registers" and field.name == "values"):
                continue
            columns[field.name] = optional_column(tables, field.name)
        columns["sequences"] = concatenate([row_sequences(batch, table) for batch, table in selected])
        if name == "registers":
            width = max(table.values.shape[1] for table in tables)
            if all(table.values.shape[1] == width for table in tables):
                columns["values"] = concatenate([table.values for table in tables])
            else:
                values = torch.zeros((sum(table.values.shape[0] for table in tables), width), dtype=torch.uint8, device="cpu")
                offset = 0
                for table in tables:
                    count, own_width = table.values.shape
                    values[offset:offset + count, :own_width].copy_(table.values)
                    offset += count
                columns["values"] = values
        # The latest table owns the latest immutable register-name mapping.
        changes[name] = replace(tables[-1], **columns)
    return replace(first, **changes)


@dataclass
class PendingSource:
    batches: list[Batch]
    size: int = 0


class BatchCollator:
    """Target bytes per source, with byte and frame-count limits on pending work.

    Adding one incoming batch can temporarily exceed the budget by that batch's
    size. Concatenation and generated sequence/padding columns need additional
    output storage. Byte accounting excludes Python objects and allocator caches.
    Sources flush at 256 frames and workers flush oldest sources at 512 pending
    frames, so tiny batches cannot accumulate unbounded Python objects.
    """

    def __init__(self, target_bytes: int):
        if not 1 <= target_bytes <= MAX_BATCH_BYTES:
            raise ValueError("Collation target must be between 1 and 4 MiB")
        self.target_bytes = target_bytes
        self.pending_bytes = 0
        self.pending_frames = 0
        self._sources: dict[tuple[int, int], PendingSource] = {}

    def _flush(self, key) -> Batch:
        pending = self._sources.pop(key)
        self.pending_bytes -= pending.size
        self.pending_frames -= len(pending.batches)
        return merge_batches(pending.batches)

    def add(self, batch: Batch) -> list[Batch]:
        if batch.layout is not None:
            return [batch]
        if batch.source is None or batch.first_sequence is None:
            raise ValueError("CPU observations need a source and sequence")
        if not batch.addresses.numel() and not any(
            table is not None and table.pc.numel()
            for table in (batch.registers, batch.memory, batch.context)
        ):
            raise ValueError("Cannot collate an empty CPU batch")
        key = (batch.worker, batch.source)
        pending = self._sources.get(key)
        if pending is None:
            pending = PendingSource([])
            self._sources[key] = pending
        size = tensor_bytes(batch)
        pending.batches.append(batch)
        pending.size += size
        self.pending_bytes += size
        self.pending_frames += 1
        ready = []
        if pending.size >= self.target_bytes or len(pending.batches) >= MAX_SOURCE_FRAMES:
            ready.append(self._flush(key))
        while self._sources and (self.pending_bytes >= 2 * self.target_bytes or
                                 self.pending_frames >= MAX_PENDING_FRAMES):
            ready.append(self._flush(next(iter(self._sources))))
        return ready

    def end_source(self, source: int, worker: int = 0) -> list[Batch]:
        key = (worker, source)
        return [self._flush(key)] if key in self._sources else []

    def finish(self) -> list[Batch]:
        return [self._flush(key) for key in tuple(self._sources)]
