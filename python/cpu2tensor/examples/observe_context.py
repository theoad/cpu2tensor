# SPDX-License-Identifier: AGPL-3.0-only
"""Associate block observations with raw CR3 using per-CPU event order."""

import argparse
import json

import torch

from cpu2tensor import Batch, Pool


def _positions(batch: Batch, positions: torch.Tensor | None, count: int) -> torch.Tensor:
    if positions is not None:
        return positions
    if batch.first_sequence is None:
        raise ValueError("CPU observations need event sequences")
    return torch.arange(batch.first_sequence, batch.first_sequence + count,
                        dtype=torch.int64, device=batch.addresses.device)


def cr3_for_blocks(batch: Batch, previous: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return each block's preceding raw CR3 and a compact state for the next batch.

    Pass consecutive batches from one (worker, source). The returned state holds
    [last context sequence, raw CR3] in its own two-element int64 allocation.
    Keep separate states for separate sources. Raw CR3 retains all 64 bits in
    signed int64 storage; it is not a process ID or a stable address-space ID.
    """
    if previous is not None and (previous.shape != (2,) or previous.dtype != torch.int64
                                 or previous.device != batch.addresses.device):
        raise ValueError("Previous context must be two int64 values on the batch device")
    context = batch.context
    sequences = torch.empty(0, dtype=torch.int64, device=batch.addresses.device)
    values = sequences
    latest = previous
    if context is not None and context.pc.numel():
        if not bool(((context.known & 2) != 0).all().item()):
            raise ValueError("Address context does not provide a known CR3 value")
        sequences = _positions(batch, context.sequences, context.pc.numel())
        values = context.cr3
        # stack copies these scalars; retaining state does not retain the batch.
        latest = torch.stack((sequences[-1], values[-1]))
    if not batch.addresses.numel():
        return torch.empty_like(batch.addresses), latest
    if previous is not None:
        sequences = torch.cat((previous[:1], sequences))
        values = torch.cat((previous[1:], values))
    if not sequences.numel():
        raise ValueError("Blocks need a preceding CR3 context")
    blocks = _positions(batch, batch.block_sequences, batch.addresses.numel())
    # A context event must precede the block, even when their PCs are identical.
    indices = torch.searchsorted(sequences, blocks, right=False) - 1
    if bool((indices < 0).any().item()):
        raise ValueError("Blocks need a preceding CR3 context")
    return values[indices], latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-bytes", type=int, default=65536)
    args = parser.parse_args()
    states = {}
    blocks = contexts = 0
    with Pool(args.endpoint, device=args.device, batch_bytes=args.batch_bytes) as pool:
        for batch in pool.read():
            if batch.source is None:
                continue  # Worker metadata has no CPU context.
            key = (batch.worker, batch.source)
            cr3, state = cr3_for_blocks(batch, states.get(key))
            if state is not None:
                states[key] = state
            blocks += cr3.numel()
            contexts += 0 if batch.context is None else batch.context.pc.numel()
    print(json.dumps({"workers": len(args.endpoint), "sources": len(states),
                      "blocks": blocks, "context_rows": contexts, "device": args.device}))


if __name__ == "__main__":
    main()
