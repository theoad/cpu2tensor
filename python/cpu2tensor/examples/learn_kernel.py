# SPDX-License-Identifier: AGPL-3.0-only
"""Verify a benign kernel adapter, then update a small command policy."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cpu2tensor.batch import Batch
from cpu2tensor.kernel import KernelEnv


COMMANDS = (b"getpid\n", b"memory 17 256\n", b"pipe 17 64\n",
            b"parallel 17 256\n", b"quit\n")


def checksum(seed: int, size: int) -> int:
    """Calculate the guest's byte sum independently of its returned value."""
    total = 0
    for _ in range(size):
        seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
        total += seed >> 24
    return total


def verify_result(action: int, result: dict[str, Any] | None) -> None:
    """A wrong guest result is an error, never a successful training sample."""
    if not 0 <= action < len(COMMANDS) - 1:
        raise ValueError("Expected a workload action")
    if result is None or result.get("event") != "result":
        raise ValueError("The guest did not return a workload result")
    fields = COMMANDS[action].decode().split()
    expected: dict[str, Any] = {"action": fields[0]}
    if action == 0:
        expected["value"] = 1  # The adapter runs as the guest's init process.
    else:
        seed, size = int(fields[1]), int(fields[2])
        expected.update(seed=seed, bytes=size)
        if action == 3:
            expected.update(checksum0=checksum(seed, size), checksum1=checksum(seed + 1, size))
            cpus = (result.get("cpu0"), result.get("cpu1"))
            if any(type(cpu) is not int or cpu < 0 for cpu in cpus) or cpus[0] == cpus[1]:
                raise ValueError("Parallel work did not report two distinct guest CPUs")
        else:
            expected["checksum"] = checksum(seed, size)
    if any(type(result.get(key)) is not type(value) or result[key] != value
           for key, value in expected.items()):
        raise ValueError(f"Incorrect guest result for {fields[0]}")


class SourceBlockFeatures:
    """Reduce every chunk into fixed, separate CPU rows without keeping a trace."""

    def __init__(self, sources: int = 2, bins: int = 64) -> None:
        if sources < 1 or bins < 2 or bins & (bins - 1):
            raise ValueError("Use positive sources and a power-of-two bin count")
        self.sources = sources
        self.bins = bins
        self.counts = torch.zeros((sources, bins), dtype=torch.int64)
        self.seen = torch.zeros((sources, bins), dtype=torch.bool)

    @property
    def size(self) -> int:
        return self.sources * (self.bins + 1)

    def __call__(self, batches: Iterable[Batch]) -> np.ndarray:
        counts = torch.zeros_like(self.counts)
        for batch in batches:
            if not 0 <= batch.source < self.sources:
                raise ValueError("Trace CPU exceeds --sources; configure every guest vCPU")
            addresses = batch.addresses
            if addresses.device.type != "cpu":
                raise ValueError("Reduce CPU trace batches before moving features to the model")
            if addresses.numel():
                # Hashing is intentionally lossy. The original int64 bits stay intact.
                tokens = ((addresses >> 2) ^ (addresses >> 12)) & (self.bins - 1)
                counts[batch.source] += torch.bincount(tokens, minlength=self.bins)
        self.counts = counts
        totals = counts.sum(dim=1, keepdim=True).to(torch.float32)
        frequencies = counts.to(torch.float32) / totals.clamp_min(1)
        features = torch.cat((frequencies, torch.log1p(totals) / 16), dim=1)
        return features.flatten().numpy()

    def coverage_reward(self, block_budget: int) -> float:
        present = self.counts > 0
        new = int((present & ~self.seen).sum())
        self.seen |= present
        # A small example objective: discover hashed blocks while limiting work.
        return new / self.seen.numel() - min(int(self.counts.sum()) / block_budget, 1.0) * 0.01


class CommandClient:
    """Keep action encoding, result checks, and reward in ordinary client code."""

    def __init__(self, features: SourceBlockFeatures, block_budget: int = 100000) -> None:
        if block_budget < 1:
            raise ValueError("The block budget must be positive")
        self.features = features
        self.block_budget = block_budget
        self.action = 0

    def encode(self, action: int) -> bytes:
        self.action = int(action)
        return COMMANDS[self.action]

    def reward(self, env: KernelEnv) -> float:
        if self.action == len(COMMANDS) - 1:
            if (env.exit_code != 0 or env.event is None or env.event.get("event") != "complete"
                    or env.event.get("ok") is not True):
                raise ValueError("The guest did not report successful completion")
            return 0.0
        verify_result(self.action, env.result)
        if not env.needs_input or env.exit_code is not None:
            raise ValueError("The guest did not return to its command boundary")
        return self.features.coverage_reward(self.block_budget)


def policy_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    observation: np.ndarray,
    gym_env: Any,
    device: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """One on-policy update; the action uses only the preceding trace features."""
    inputs = torch.from_numpy(observation).to(device)
    distribution = torch.distributions.Categorical(logits=model(inputs))
    action = distribution.sample()
    next_observation, reward, ended, truncated, _ = gym_env.step(int(action.item()))
    if ended or truncated:
        raise ValueError("The guest ended before the explicit quit action")
    loss = -distribution.log_prob(action) * reward - 0.01 * distribution.entropy()
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("The policy loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
           for parameter in model.parameters()):
        raise ValueError("The policy gradient is not finite")
    optimizer.step()
    return next_observation, {"action": int(action.item()), "reward": reward, "loss": loss.item()}


def run(
    endpoint: str, *, device: str = "cpu", sources: int = 2, bins: int = 64,
    policy_steps: int = 4, seed: int = 7, timeout: float = 120.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import gymnasium as gym

    if not 0 <= policy_steps <= 1000:
        raise ValueError("Choose from 0 through 1000 policy steps")
    torch.manual_seed(seed)
    features = SourceBlockFeatures(sources, bins)
    client = CommandClient(features)
    model = torch.nn.Linear(features.size, len(COMMANDS) - 1, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checked_results = []
    updates = []
    with KernelEnv(endpoint, timeout=timeout) as env:
        wrapper = env.as_gym(
            observe=features, encode=client.encode, reward=client.reward,
            observation_space=gym.spaces.Box(0, np.inf, shape=(features.size,), dtype=np.float32),
            action_space=gym.spaces.Discrete(len(COMMANDS)),
        )
        observation, _ = wrapper.reset()
        for action in range(len(COMMANDS) - 1):
            observation, _, _, _, info = wrapper.step(action)
            checked_results.append(dict(info["result"]))
        # The calibration establishes adapter correctness. Start the learning
        # objective afresh, then keep coverage for this one guest trajectory.
        features.seen.zero_()
        for _ in range(policy_steps):
            observation, update = policy_update(model, optimizer, observation, wrapper, device)
            updates.append(update)
        _, _, ended, truncated, _ = wrapper.step(len(COMMANDS) - 1)
        if not ended or truncated:
            raise ValueError("Explicit quit did not complete the episode")
    final = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    if any(not bool(value.isfinite().all()) for value in final.values()):
        raise ValueError("The updated model is not finite")
    changed = any(not torch.equal(initial[name], value) for name, value in final.items())
    if policy_steps and not changed:
        raise ValueError("The policy parameters did not change")
    metrics = {
        "device": device, "seed": seed, "sources": sources, "bins": bins,
        "checked_results": checked_results, "policy_updates": updates,
        "parameters_changed": changed, "complete": True,
        "scope": "Command adapter and finite model update; no convergence or performance claim",
    }
    checkpoint = {"initial_state": initial, "model_state": final, "metrics": metrics}
    return metrics, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--sources", type=int, default=2)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--policy-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path,
                        default=Path.home() / ".cache/cpu2tensor/kernel-gym-check")
    args = parser.parse_args()
    metrics, checkpoint = run(args.endpoint, device=args.device, sources=args.sources,
                              bins=args.bins, policy_steps=args.policy_steps, seed=args.seed,
                              timeout=args.timeout)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output / "policy.pt")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
