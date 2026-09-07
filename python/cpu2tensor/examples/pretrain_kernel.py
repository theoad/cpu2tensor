# SPDX-License-Identifier: AGPL-3.0-only
"""Train a small next-block model on operator-started observation workers."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import queue
import threading
import time

import torch

from cpu2tensor import Batch, Pool


class BlockPairs:
    """Keep one previous token per worker/vCPU, never a global predecessor."""

    def __init__(self, vocabulary: int) -> None:
        if vocabulary < 16 or vocabulary > 4096 or vocabulary & (vocabulary - 1):
            raise ValueError("Vocabulary must be a power of two between 16 and 4096")
        self.vocabulary = vocabulary
        self._previous: dict[tuple[int, int], int] = {}

    def add(self, worker: int, batch: Batch) -> torch.Tensor:
        addresses = batch.addresses.cpu()
        if not addresses.numel():
            return torch.empty((0, 2), dtype=torch.int64)
        # Hashing is example feature preparation. The original Batch keeps every
        # raw address bit; this deliberately small vocabulary merges addresses.
        tokens = ((addresses >> 2) ^ (addresses >> 12)) & (self.vocabulary - 1)
        key = (worker, batch.source)
        previous = self._previous.get(key)
        self._previous[key] = int(tokens[-1])
        if previous is None:
            return torch.stack((tokens[:-1], tokens[1:]), dim=1)
        before = torch.cat((torch.tensor([previous], dtype=torch.int64), tokens[:-1]))
        return torch.stack((before, tokens), dim=1)

    def end(self, worker: int) -> None:
        for key in list(self._previous):
            if key[0] == worker:
                del self._previous[key]


class PairBatches:
    """Make fixed-size minibatches; the final partial batch needs no padding."""

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("Minibatch size must be positive")
        self._size = size
        self._data = torch.empty((size, 2), dtype=torch.int64)
        self._used = 0

    def add(self, pairs: torch.Tensor) -> Iterator[torch.Tensor]:
        offset = 0
        while offset < pairs.shape[0]:
            count = min(self._size - self._used, pairs.shape[0] - offset)
            self._data[self._used:self._used + count].copy_(pairs[offset:offset + count])
            self._used += count
            offset += count
            if self._used == self._size:
                full = self._data
                # A yielded batch owns its storage even if the caller retains it.
                self._data = torch.empty((self._size, 2), dtype=torch.int64)
                self._used = 0
                yield full

    def finish(self) -> torch.Tensor:
        partial = self._data[:self._used].clone()
        self._used = 0
        return partial


class HeldOutBatches:
    """Keep a bounded uniform reservoir of completed test minibatches."""

    def __init__(self, limit: int, seed: int) -> None:
        if limit < 1:
            raise ValueError("Keep at least one test minibatch")
        self._limit = limit
        self._random = torch.Generator().manual_seed(seed)
        self.seen = 0
        self.batches: list[torch.Tensor] = []

    def add(self, batch: torch.Tensor) -> None:
        if not batch.shape[0]:
            return
        self.seen += 1
        if len(self.batches) < self._limit:
            self.batches.append(batch)
        else:
            selected = int(torch.randint(self.seen, (), generator=self._random))
            if selected < self._limit:
                self.batches[selected] = batch


@contextmanager
def worker_batches(
    endpoints: Sequence[str], *, queue_batches: int = 8, timeout: float = 30,
) -> Iterator[Iterator[tuple[int, Batch | None]]]:
    """Read independent ordinary Pools through one bounded example-local queue.

    None marks successful completion of one worker. Leaving this context early
    cancels remaining runs; a cancelled stream is never reported as complete.
    """
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("Provide distinct worker endpoints")
    if queue_batches < 1:
        raise ValueError("Queue capacity must be positive")
    readers = [Pool([endpoint], timeout=timeout) for endpoint in endpoints]
    pending: queue.Queue[tuple[int, Batch | BaseException | None]] = queue.Queue(queue_batches)
    stopped = threading.Event()

    def publish(item: tuple[int, Batch | BaseException | None]) -> None:
        while not stopped.is_set():
            try:
                pending.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def read(index: int, pool: Pool) -> None:
        try:
            with pool:
                for batch in pool.read():
                    if stopped.is_set():
                        return
                    if batch.addresses.numel():
                        publish((index, batch))
                publish((index, None))
        except BaseException as error:
            publish((index, error))

    def receive() -> Iterator[tuple[int, Batch | None]]:
        remaining = len(readers)
        while remaining:
            index, item = pending.get()
            if isinstance(item, BaseException):
                raise RuntimeError(f"Worker {endpoints[index]} failed before completion") from item
            if item is None:
                remaining -= 1
            yield index, item

    threads = [threading.Thread(target=read, args=(index, pool), daemon=True)
               for index, pool in enumerate(readers)]
    started = []
    try:
        for thread in threads:
            thread.start()
            started.append(thread)
        yield receive()
    finally:
        stopped.set()
        for pool in readers:
            pool.close()
        deadline = time.monotonic() + timeout + 1
        for thread in started:
            thread.join(max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in started):
            raise RuntimeError("A worker reader did not stop after cancellation")


def make_model(vocabulary: int, seed: int) -> torch.nn.Sequential:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Embedding(vocabulary, 32), torch.nn.Linear(32, vocabulary))


def evaluate(model: torch.nn.Module, batches: Sequence[torch.Tensor], device: str) -> dict:
    total_loss = 0.0
    correct = 0
    pairs = 0
    with torch.no_grad():
        for batch in batches:
            data = batch.to(device)
            logits = model(data[:, 0])
            total_loss += float(torch.nn.functional.cross_entropy(logits, data[:, 1], reduction="sum").cpu())
            correct += int((logits.argmax(1) == data[:, 1]).sum().cpu())
            pairs += data.shape[0]
    if not pairs:
        raise ValueError("Test runs produced no same-source next-block pairs")
    return {"loss": total_loss / pairs, "accuracy": correct / pairs, "pairs": pairs}


def train(
    train_endpoints: Sequence[str], test_endpoints: Sequence[str], output: str | Path, *,
    device: str = "cpu", vocabulary: int = 256, batch_size: int = 1024,
    max_updates: int = 100, test_batches: int = 16, queue_batches: int = 8,
    timeout: float = 30, seed: int = 23, capture_host: str = "operator supplied",
) -> dict:
    """Train from bounded streams, then compare initial/final models on held-out runs."""
    if not train_endpoints or not test_endpoints:
        raise ValueError("Provide separate training and test worker runs")
    if max_updates < 1:
        raise ValueError("Maximum updates must be positive")
    selected = torch.device(device)
    if selected.type not in ("cpu", "mps", "cuda"):
        raise ValueError("Choose CPU, MPS or CUDA")
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    pairs = BlockPairs(vocabulary)
    training = PairBatches(batch_size)
    testing = PairBatches(batch_size)
    held_out = HeldOutBatches(test_batches, seed + 1)
    initial = make_model(vocabulary, seed)
    model = make_model(vocabulary, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses: list[float] = []
    trained_pairs = 0
    observed_train_pairs = 0
    observed_test_pairs = 0
    completed = 0
    endpoints = [*train_endpoints, *test_endpoints]

    def update(batch: torch.Tensor) -> None:
        nonlocal trained_pairs
        if not batch.shape[0] or len(losses) >= max_updates:
            return
        data = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(data[:, 0]), data[:, 1])
        loss.backward()
        if not bool(torch.isfinite(loss)) or not all(
            bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
        ):
            raise RuntimeError("Next-block training produced a non-finite loss or gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        trained_pairs += data.shape[0]

    with worker_batches(endpoints, queue_batches=queue_batches, timeout=timeout) as batches:
        for worker, batch in batches:
            if batch is None:
                pairs.end(worker)
                completed += 1
                continue
            observed = pairs.add(worker, batch)
            if worker < len(train_endpoints):
                observed_train_pairs += observed.shape[0]
                if len(losses) < max_updates:
                    for part in training.add(observed):
                        update(part)
            else:
                observed_test_pairs += observed.shape[0]
                for part in testing.add(observed):
                    held_out.add(part)
    # Every worker was consumed through its validated completion, even after the
    # update budget. Full completion is not inferred from the final data batch.
    update(training.finish())
    held_out.add(testing.finish())
    if not losses:
        raise ValueError("Training runs produced no same-source next-block pairs")
    before = evaluate(initial, held_out.batches, "cpu")
    after = evaluate(model, held_out.batches, device)
    changed = any(not torch.equal(old, new.detach().cpu())
                  for old, new in zip(initial.parameters(), model.parameters()))
    if not changed or not all(torch.isfinite(torch.tensor(value)) for value in (before["loss"], after["loss"])):
        raise RuntimeError("Next-block training did not produce finite updated weights")
    metrics = {
        "device": str(selected), "torch_version": str(torch.__version__), "seed": seed,
        "capture_host": capture_host, "workers_completed": completed,
        "train_workers": len(train_endpoints), "test_workers": len(test_endpoints),
        "vocabulary": vocabulary, "batch_size": batch_size, "queue_batches": queue_batches,
        "updates": len(losses), "max_updates": max_updates, "trained_pairs": trained_pairs,
        "observed_train_pairs": observed_train_pairs, "observed_test_pairs": observed_test_pairs,
        "unused_training_pairs": observed_train_pairs - trained_pairs,
        "test_minibatches_seen": held_out.seen, "test_minibatches_retained": len(held_out.batches),
        "initial_test": before, "final_test": after, "weights_changed": changed,
        "all_traces_complete": completed == len(endpoints),
    }
    directory = Path(output).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "vocabulary": vocabulary, "metrics": metrics}, directory / "model.pt")
    torch.save(held_out.batches, directory / "test-pairs.pt")
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (directory / "loss.json").write_text(json.dumps(losses) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="append", required=True, help="Training endpoint; repeat for more workers")
    parser.add_argument("--test", action="append", required=True, help="Independent test run endpoint; repeat as needed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--vocabulary", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-updates", type=int, default=100)
    parser.add_argument("--test-batches", type=int, default=16)
    parser.add_argument("--queue-batches", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--capture-host", default="operator supplied")
    args = parser.parse_args()
    print(json.dumps(train(args.train, args.test, args.output, device=args.device,
                           vocabulary=args.vocabulary, batch_size=args.batch_size,
                           max_updates=args.max_updates, test_batches=args.test_batches,
                           queue_batches=args.queue_batches, timeout=args.timeout,
                           seed=args.seed, capture_host=args.capture_host), indent=2))


if __name__ == "__main__":
    main()
