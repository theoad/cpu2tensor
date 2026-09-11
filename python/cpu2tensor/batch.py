# SPDX-License-Identifier: AGPL-3.0-only
"""Owned tensor columns for consecutive events from one CPU."""

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ExecutableLayout:
    """Initial user-process layout, owned by the worker rather than a vCPU.

    ``values`` holds code_start, code_end (exclusive), initial_entry in that
    order. The code span is QEMU's nominal executable segment span, not an ELF
    load bias or a complete mapping table. A dynamic program's initial entry
    can belong to its interpreter. Later loads/remaps are not represented.
    """

    values: Tensor

    @property
    def code_start(self) -> Tensor:
        return self.values[0]

    @property
    def code_end(self) -> Tensor:
        return self.values[1]

    @property
    def initial_entry(self) -> Tensor:
        return self.values[2]


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
    sequences: Tensor | None = None


@dataclass(frozen=True)
class AddressContext:
    """Ordered x86 paging/execution context, scoped to one worker and vCPU.

    ``known`` has bits for cr0, cr3, cr4, efer, cs_base and mode respectively.
    Unknown fields contain zero, not a measured zero. Mode is 16, 32 or 64.
    CR3 is raw paging state, not a process ID or a stable address-space ID.
    """

    pc: Tensor
    cr0: Tensor
    cr3: Tensor
    cr4: Tensor
    efer: Tensor
    cs_base: Tensor
    mode: Tensor
    known: Tensor
    sequences: Tensor | None = None


@dataclass(frozen=True)
class MemoryAccesses:
    """Successful emulated transactions, excluding syscall copies and DMA.

    Flags use bit 0 for stores and bit 1 for big-endian guest accesses. Optional
    values contain little-endian significance bytes, zero-padded to 16 bytes.
    System mappings describe only a physical prefix of each transaction; a
    cross-page tail can be unknown. Mapping flags: bit 0 physical prefix known,
    bit 1 QEMU I/O-dispatch flag known, bit 2 I/O dispatch. The flag applies to
    the first address only, not every byte of the prefix. Neither a direct path
    nor an unknown flag proves RAM: subpage wrappers and ROMD affect dispatch. Context sequences reference preceding AddressContext event sequences
    on this same source; the decoder verifies each reference.
    """

    pc: Tensor
    addresses: Tensor
    sizes: Tensor
    flags: Tensor
    values: Tensor | None
    physical_addresses: Tensor | None = None
    mapped_sizes: Tensor | None = None
    mapping_flags: Tensor | None = None
    context_sequences: Tensor | None = None
    sequences: Tensor | None = None


@dataclass(frozen=True)
class BlockTransitions:
    """Bounded adjacent block counts from one vCPU and one action window.

    The first block observed on a vCPU has no incoming transition. Window
    boundaries reset that predecessor, so rows never connect separate actions.
    """

    window: int
    from_addresses: Tensor
    destinations: Tensor
    counts: Tensor


@dataclass(frozen=True)
class TransitionWindow:
    """Final status for one guest-declared action window.

    Status is ``ended``, ``aborted``, or ``incomplete``. An ended window is a
    complete reduced observation only when ``overflow`` is zero. Overflow counts
    transitions that could not fit the fixed per-vCPU table.
    """

    id: int
    status: str
    sources: int
    capacity_per_source: int
    distinct: int
    observed: int
    overflow: int

    @property
    def complete(self) -> bool:
        return self.status == "ended" and self.overflow == 0


@dataclass(frozen=True)
class ObservationTransitions:
    """Bounded adjacent-block counts from one vCPU's complete run.

    Rows are aggregate keys rather than a contiguous occurrence stream. Use the
    per-source ``ObservationSummary`` for the exact processed transition count.
    """

    from_addresses: Tensor
    destinations: Tensor
    counts: Tensor


@dataclass(frozen=True)
class ObservationContext:
    """A bounded prefix of context changes with exact block positions.

    ``block_positions`` indexes every processed block from zero on this vCPU.
    These rows have their own retained-record order and are not raw block events.
    Availability bits and address-space semantics match ``AddressContext``.
    """

    block_positions: Tensor
    pc: Tensor
    cr0: Tensor
    cr3: Tensor
    cr4: Tensor
    efer: Tensor
    cs_base: Tensor
    mode: Tensor
    known: Tensor


@dataclass(frozen=True)
class ObservationSummary:
    """Exact per-vCPU totals for a completed bounded observation run.

    Pool publishes this only after the worker's Complete frame. ``exact`` says
    whether every distinct transition and context change fit; it does not claim
    that aggregate rows retain raw occurrence order.
    """

    sources: int
    transition_capacity: int
    context_capacity: int
    blocks: int
    transitions: int
    distinct_transitions: int
    transition_overflow: int
    context_changes: int
    retained_contexts: int
    context_overflow: int
    upstream_complete: bool = True

    @property
    def exact(self) -> bool:
        return self.upstream_complete and self.transition_overflow == 0 and self.context_overflow == 0


@dataclass(frozen=True)
class Batch:
    """Events from one CPU, with independently owned tensor storage.

    Addresses and PCs keep their raw 64 bits in signed int64 tensor storage.
    A negative address represents its unsigned value modulo 2**64. Source order
    is meaningful; arrival order between different sources is not memory order.
    Retaining a batch keeps its storage alive independently of the pool.
    Legacy batches populate one table; a row's source sequence is
    ``first_sequence`` plus its index. Mixed batches can populate several tables:
    use ``block_sequences`` and each table's ``sequences`` for the original event
    positions. ``first_sequence`` is the first event across all populated tables.
    ``worker`` is the endpoint's index in the Pool, stable for that Pool's run.
    Use (worker, source) together when maintaining per-CPU state.
    An opt-in ``layout`` batch describes the worker's initial executable instead:
    ``source`` and ``first_sequence`` are None and no CPU sequence is consumed.
    Transition batches are per source and window. Final transition-window metadata
    is worker-wide and also consumes no CPU sequence.
    Observation reduction uses separate context/transition tables with no raw
    occurrence sequence. Its per-source summary appears only after successful
    upstream completion; context rows carry their original block positions.
    """

    source: int | None
    first_sequence: int | None
    addresses: Tensor
    registers: RegisterChanges | None = None
    memory: MemoryAccesses | None = None
    context: AddressContext | None = None
    layout: ExecutableLayout | None = None
    worker: int = 0
    block_sequences: Tensor | None = None
    transitions: BlockTransitions | None = None
    transition_window: TransitionWindow | None = None
    observation_transitions: ObservationTransitions | None = None
    observation_context: ObservationContext | None = None
    observation_summary: ObservationSummary | None = None
