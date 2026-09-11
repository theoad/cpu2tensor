# SPDX-License-Identifier: AGPL-3.0-only
"""One owned accelerator upload for the tensor columns in a Batch."""

from dataclasses import fields, replace

import torch

from cpu2tensor.batch import Batch


def to_device(batch: Batch, device: torch.device) -> Batch:
    """Keep CPU columns zero-copy; pack accelerator columns into one allocation.

    Accelerator transfer adds one CPU staging copy. Typed slices share the
    uploaded storage, which is never reused for another batch. Retaining any
    returned tensor therefore keeps its bytes alive. Input columns are capture
    data; this operation does not preserve a gradient graph for those inputs.
    """
    if device.type == "cpu":
        return batch
    columns = [(None, "addresses", batch.addresses)]
    if batch.block_sequences is not None:
        columns.append((None, "block_sequences", batch.block_sequences))
    for name in ("registers", "memory", "context", "layout", "transitions"):
        table = getattr(batch, name)
        if table is not None:
            for field in fields(table):
                value = getattr(table, field.name)
                if isinstance(value, torch.Tensor):
                    columns.append((name, field.name, value))

    if len(columns) == 1:
        # Block-only capture already needs one transfer. Do not add a staging copy.
        if batch.addresses.device.type != "cpu":
            raise ValueError("Batch upload requires CPU input columns")
        result = replace(batch, addresses=batch.addresses.to(device, non_blocking=False))
        if device.type == "mps":
            torch.mps.synchronize()
        return result

    packed = []
    total_bytes = 0
    for table, name, tensor in columns:
        if tensor.device.type != "cpu":
            raise ValueError("Batch upload requires CPU input columns")
        alignment = tensor.element_size()
        offset = (total_bytes + alignment - 1) // alignment * alignment
        size = tensor.numel() * alignment
        packed.append((table, name, tensor, offset, size))
        total_bytes = offset + size

    with torch.no_grad():
        # Zero initialization also defines alignment padding. Copy directly into
        # typed views: non-contiguous inputs need no intermediate allocation.
        staging = torch.zeros(total_bytes, dtype=torch.uint8, device="cpu")
        for _, _, tensor, offset, size in packed:
            staging.narrow(0, offset, size).view(tensor.dtype).view(tensor.shape).copy_(tensor)
        uploaded = staging.to(device, non_blocking=False)
        changes = {}
        tables = {}
        for table, name, tensor, offset, size in packed:
            value = uploaded.narrow(0, offset, size).view(tensor.dtype).view(tensor.shape)
            if table is None:
                changes[name] = value
            else:
                tables.setdefault(table, {})[name] = value
        for table, values in tables.items():
            changes[table] = replace(getattr(batch, table), **values)
        result = replace(batch, **changes)
    if device.type == "mps":
        torch.mps.synchronize()
    return result
