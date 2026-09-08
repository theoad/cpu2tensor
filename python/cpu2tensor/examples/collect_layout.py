# SPDX-License-Identifier: AGPL-3.0-only
"""Collect path-ending block addresses and one executable layout per worker."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path

import torch

from cpu2tensor import Batch, Pool
from cpu2tensor.examples.learn_layout import validate_dataset


def collect_samples(batches: Iterable[Batch], sample_end_offset: int) -> dict:
    """Extract the block before each marker, without receiving inputs or labels."""
    if sample_end_offset < 0:
        raise ValueError("The sample marker offset must be nonnegative")
    layout = None
    source = None
    previous = None
    samples: list[list[int]] = []
    full_block_count = 0
    for batch in batches:
        if batch.layout is not None:
            if layout is not None or full_block_count:
                raise ValueError("Expected exactly one layout before all block data")
            if (batch.source is not None or batch.first_sequence is not None or batch.addresses.numel()
                    or batch.registers is not None or batch.memory is not None or batch.context is not None):
                raise ValueError("Executable layout must be a separate worker-wide batch")
            values = batch.layout.values.cpu()
            if values.dtype != torch.int64 or values.shape != (3,):
                raise ValueError("Expected three int64 executable layout fields")
            layout = values.tolist()
            start, end, entry = layout
            if not (0 <= start <= entry < end and start + sample_end_offset < end):
                raise ValueError("Static executable layout or sample marker lies outside the text span")
            continue
        if layout is None:
            raise ValueError("The worker must emit executable layout before block data")
        if batch.registers is not None or batch.memory is not None or batch.context is not None:
            raise ValueError("Use block-only capture for the layout tutorial")
        if batch.source is None or batch.first_sequence is None:
            raise ValueError("Block batches need a CPU source and sequence")
        if source is not None and batch.source != source:
            raise ValueError("The layout tutorial requires one vCPU")
        source = batch.source
        addresses = batch.addresses.cpu()
        if addresses.dtype != torch.int64 or addresses.ndim != 1:
            raise ValueError("Expected one int64 block address column")
        if not bool(((addresses >= layout[0]) & (addresses < layout[1])).all()):
            raise ValueError("Every block must lie in the executable text span")
        full_block_count += addresses.numel()
        marker = layout[0] + sample_end_offset
        for pc in addresses.tolist():
            if pc == marker:
                if previous is None:
                    raise ValueError("A sample marker must have a preceding block")
                samples.append([previous])
                previous = None
            else:
                previous = pc
    if layout is None or not samples:
        raise ValueError("A complete capture needs executable layout and sample markers")
    return {"layout": layout, "pc": samples, "full_block_count": full_block_count,
            "metadata_frames": 1}


def collect(endpoint: str, input_bytes: bytes, sample_end_offset: int,
            *, train: bool = True, timeout: float = 30.0) -> dict:
    """Read a complete prescribed-input run; labels never select trace features.

    The operator starts the worker with the same input bytes. This function does
    not send target input, start workers, or manage connections through SSH.
    """
    with Pool([endpoint], timeout=timeout) as pool:
        result = collect_samples(pool.read(), sample_end_offset)
    if not input_bytes or any(value < ord("0") or value > ord("3") for value in input_bytes):
        raise ValueError("Target input must contain only ASCII digits 0 through 3")
    labels = [value - ord("0") for value in input_bytes]
    if len(result["pc"]) != len(labels):
        raise ValueError("The number of complete sample markers must match the supplied input")
    return {**result, "labels": labels, "train": train}


def dataset_from_runs(capture: dict) -> dict:
    """Pad JSON runs into the learner's tensor schema, keeping the supplied split."""
    if capture.get("format") != 1 or not isinstance(capture.get("runs"), list):
        raise ValueError("Expected capture format 1 with a list of worker runs")
    pc, layouts, labels, train = [], [], [], []
    for run in capture["runs"]:
        if not isinstance(run.get("train"), bool) or len(run["pc"]) != len(run["labels"]):
            raise ValueError("Each run needs a Boolean split and one label per sample")
        numbers = [*run["layout"], *run["labels"], *(value for row in run["pc"] for value in row)]
        if any(type(value) is not int for value in numbers) or len(run["layout"]) != 3:
            raise ValueError("Captured addresses and labels must be exact integers, with three layout fields")
        pc.extend(run["pc"])
        layouts.extend([run["layout"]] * len(run["pc"]))
        labels.extend(run["labels"])
        train.extend([run["train"]] * len(run["pc"]))
    if not pc or any(not row for row in pc):
        raise ValueError("Worker runs must contain nonempty samples")
    addresses = torch.zeros(len(pc), max(map(len, pc)), dtype=torch.int64)
    mask = torch.zeros_like(addresses, dtype=torch.bool)
    for index, row in enumerate(pc):
        addresses[index, :len(row)] = torch.tensor(row, dtype=torch.int64)
        mask[index, :len(row)] = True
    dataset = {"format": 1, "pc": addresses, "mask": mask,
               "layout": torch.tensor(layouts, dtype=torch.int64),
               "labels": torch.tensor(labels, dtype=torch.int64),
               "train": torch.tensor(train, dtype=torch.bool)}
    validate_dataset(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sample-end-offset", required=True, type=lambda value: int(value, 0))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    args = parser.parse_args()
    result = collect(args.endpoint, args.input.read_bytes(), args.sample_end_offset, train=args.split == "train")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
