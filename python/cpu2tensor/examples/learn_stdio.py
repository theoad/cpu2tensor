# SPDX-License-Identifier: AGPL-3.0-only
"""Learn a legal digit response from a paused target's execution trace."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path

import torch
from torch import nn

from cpu2tensor.batch import Batch
from cpu2tensor.stdio import StdioEnv


def observe(batches: Iterable[Batch], entries: torch.Tensor) -> torch.Tensor:
    """Keep ten counts, regardless of how much execution leads to the request."""
    counts = torch.zeros(10, dtype=torch.int64, device=entries.device)
    for batch in batches:
        counts += (batch.addresses[:, None] == entries[None, :]).sum(dim=0)
    if int(counts.sum().item()) != 1:
        raise RuntimeError("Expected exactly one digit cue before the stdin request")
    return counts.to(torch.float32)


def finish(env: StdioEnv, action: int) -> float:
    """The example defines reward from its target's documented exit codes."""
    for _ in env.step(f"{action}\n".encode("ascii")):
        pass
    if env.exit_code not in (0, 1):
        raise RuntimeError("Expected one stdin decision followed by target exit 0 or 1")
    return float(env.exit_code == 0)


def imitation(
    env: StdioEnv, policy: nn.Module, entries: torch.Tensor, *, limit: int
) -> dict:
    """Collect one real trace of each cue, then fit a tiny supervised baseline."""
    examples: dict[int, torch.Tensor] = {}
    episodes = 0
    while len(examples) != 10 and episodes < limit:
        observation = observe(env.reset(), entries)
        label = int(observation.argmax().item())
        examples[label] = observation
        finish(env, label)
        episodes += 1
    if len(examples) != 10:
        raise RuntimeError("Not all ten random cues appeared; increase --episodes")
    features = torch.stack([examples[label] for label in range(10)])
    labels = torch.arange(10, device=entries.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
    losses = []
    for _ in range(80):
        loss = nn.functional.cross_entropy(policy(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    if not losses[-1] < losses[0] * 0.1:
        raise RuntimeError("The supervised loss did not fall as expected")
    return {"training_episodes": episodes, "first_loss": losses[0], "last_loss": losses[-1]}


def reinforce(
    env: StdioEnv, policy: nn.Module, entries: torch.Tensor, *, limit: int
) -> dict:
    """Learn only from sampled actions and success rewards, without cue labels."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
    baseline = 0.0
    rewards: list[float] = []
    for episode in range(limit):
        observation = observe(env.reset(), entries)
        distribution = torch.distributions.Categorical(logits=policy(observation))
        action = distribution.sample()
        reward = finish(env, int(action.item()))
        loss = -(reward - baseline) * distribution.log_prob(action)
        loss -= 0.01 * distribution.entropy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        baseline = 0.95 * baseline + 0.05 * reward
        rewards.append(reward)
        if (episode + 1) % 50 == 0:
            print(json.dumps({"episode": episode + 1, "recent_reward": sum(rewards[-50:]) / 50}))
    return {"training_episodes": limit, "training_reward": sum(rewards) / len(rewards)}


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    symbols = json.loads(args.symbols.expanduser().read_text())
    addresses = [int(symbols[f"digit_{digit}"]) for digit in range(10)]
    if len(set(addresses)) != 10:
        raise ValueError("Digit symbols must have distinct entry addresses")
    entries = torch.tensor(addresses, dtype=torch.int64, device=args.device)
    policy = nn.Linear(10, 10).to(args.device)
    # Equal initial logits make the ten-way chance baseline explicit.
    with torch.no_grad():
        policy.weight.zero_()
        policy.bias.zero_()
    with StdioEnv(args.endpoint, device=args.device) as env:
        train = imitation if args.method == "imitation" else reinforce
        result = train(env, policy, entries, limit=args.episodes)
        successes = 0.0
        evaluated_cues = [0] * 10
        policy.eval()
        with torch.no_grad():
            for _ in range(args.evaluate):
                observation = observe(env.reset(), entries)
                evaluated_cues[int(observation.argmax().item())] += 1
                action = int(policy(observation).argmax().item())
                successes += finish(env, action)
    success_rate = successes / args.evaluate
    result.update({"method": args.method, "device": args.device, "chance_success": 0.1,
                   "evaluation_episodes": args.evaluate, "evaluation_success": success_rate,
                   "evaluation_cue_counts": evaluated_cues, "seed": args.seed,
                   "target_seeded": False})
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if success_rate < args.min_success:
        raise RuntimeError(f"Evaluation success {success_rate:.3f} is below {args.min_success:.3f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint")
    parser.add_argument("--symbols", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--method", choices=("imitation", "reinforce"), default="imitation")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Maximum collection episodes, or REINFORCE training episodes")
    parser.add_argument("--evaluate", type=int, default=40)
    parser.add_argument("--min-success", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=19, help="Seed PyTorch; the target uses OS randomness")
    args = parser.parse_args()
    if args.episodes < 1 or args.evaluate < 1 or not 0 <= args.min_success <= 1:
        parser.error("Episode counts must be positive and minimum success must be between 0 and 1")
    run(args)


if __name__ == "__main__":
    main()
