# SPDX-License-Identifier: AGPL-3.0-only
"""Consume prescribed execution from an already listening worker."""

import argparse

import torch

from cpu2tensor import Pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", help="Worker address, for example tcp://localhost:9000")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    arguments = parser.parse_args()

    entries = 0
    register_changes = 0
    memory_accesses = 0
    # Derive a small feature on the selected device. Raw addresses remain int64;
    # floating point is suitable for the derived feature, not the full address.
    total = torch.zeros((), device=arguments.device)
    with Pool([arguments.endpoint], device=arguments.device) as pool:
        for batch in pool.read():
            features = (batch.addresses & 255).to(torch.float32) / 255
            total += features.sum()
            entries += batch.addresses.numel()
            if batch.registers is not None:
                register_changes += batch.registers.ids.numel()
            if batch.memory is not None:
                memory_accesses += batch.memory.addresses.numel()
    print(f"Read {entries} block entries, {register_changes} register changes, "
          f"{memory_accesses} memory accesses; feature sum: {total.item():.3f}")


if __name__ == "__main__":
    main()
