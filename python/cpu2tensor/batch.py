# SPDX-License-Identifier: AGPL-3.0-only
"""Owned tensor columns for consecutive events from one CPU."""

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class RegisterChanges:
    """Register values at checkpoints, with names from this CPU's schema.

    Values contain little-endian bytes, padded with zeros to the widest row.
    Widths select each row's actual bytes. Flags identify the checkpoint in the
    low byte (1 block entry, 2 syscall entry, 3 CPU exit); bit 8 marks a baseline.
    These samples are not a record of every register write between checkpoints.
    """

    pc: Tensor
    ids: Tensor
    widths: Tensor
    flags: Tensor
    values: Tensor
    names: Mapping[int, str]


@dataclass(frozen=True)
class MemoryAccesses:
    """Successful emulated transactions, excluding syscall copies and DMA.

    Flags use bit 0 for stores and bit 1 for big-endian guest accesses. Optional
    values contain little-endian significance bytes, zero-padded to 16 bytes.
    """

    pc: Tensor
    addresses: Tensor
    sizes: Tensor
    flags: Tensor
    values: Tensor | None


@dataclass(frozen=True)
class Batch:
    """One signal's consecutive events, with independently owned tensor storage.

    Addresses and PCs keep their raw 64 bits in signed int64 tensor storage.
    A negative address represents its unsigned value modulo 2**64. Source order
    is meaningful; arrival order between different sources is not memory order.
    Retaining a batch keeps its storage alive independently of the pool.
    ``addresses`` is empty for register and memory batches. A row's source
    sequence is ``first_sequence`` plus its index in the populated table.
    """

    source: int
    first_sequence: int
    addresses: Tensor
    registers: RegisterChanges | None = None
    memory: MemoryAccesses | None = None
