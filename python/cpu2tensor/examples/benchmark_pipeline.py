# SPDX-License-Identifier: AGPL-3.0-only
"""Measure complete endpoint-to-tensor runs and a small reconstruction workload."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import fields
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import time

import torch

from cpu2tensor import Batch, Pool


FEATURE_GROUPS = ("blocks", "register_bytes", "memory_addresses", "memory_sizes",
                  "memory_values", "context", "layout")
HISTOGRAM_BINS = 16
FEATURE_COUNT = len(FEATURE_GROUPS) * HISTOGRAM_BINS
ROW_KINDS = ("blocks", "registers", "memory", "context", "layout")


def row_counts(batch: Batch) -> dict[str, int]:
    return {
        "blocks": batch.addresses.numel(),
        "registers": 0 if batch.registers is None else batch.registers.pc.numel(),
        "memory": 0 if batch.memory is None else batch.memory.pc.numel(),
        "context": 0 if batch.context is None else batch.context.pc.numel(),
        "layout": int(batch.layout is not None),
    }


def tensor_bytes(batch: Batch) -> int:
    """Logical column bytes; excludes queues, allocator caches and Python objects."""
    total = 0
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    for name in ("registers", "memory", "context", "layout"):
        columns = getattr(batch, name)
        if columns is not None:
            for field in fields(columns):
                value = getattr(columns, field.name)
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
    return total


def histogram(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Sixteen coarse low-byte bins, deliberately lossy and cheap to modify."""
    bins = ((values.reshape(-1).to(torch.int64) & 255) >> 4)
    weights = (torch.ones_like(bins, dtype=torch.float32) if mask is None else
               mask.reshape(-1).to(torch.float32))
    result = torch.zeros(HISTOGRAM_BINS, device=values.device, dtype=torch.float32)
    result.scatter_add_(0, bins, weights)
    return result / weights.sum().clamp_min(1)


def summarize(batch: Batch) -> torch.Tensor:
    """Consume every populated table with tensor operations, without row loops.

    Byte histograms exclude register padding and unused memory-value bytes.
    Memory sizes use log2 buckets so the usual 1/2/4/8/16-byte accesses differ.
    Histograms are a throughput workload, not a lossless encoding or an ASLR fix.
    All work stays on the device chosen for the full incoming Batch.
    """
    empty = torch.empty(0, dtype=torch.int64, device=batch.addresses.device)
    registers = batch.registers
    memory = batch.memory
    register_values = empty if registers is None else registers.values
    memory_values = empty if memory is None or memory.values is None else memory.values
    register_mask = None if registers is None else (
        torch.arange(registers.values.shape[1], device=empty.device)[None, :] < registers.widths[:, None]
    )
    memory_mask = None if memory is None or memory.values is None else (
        torch.arange(memory.values.shape[1], device=empty.device)[None, :] < memory.sizes[:, None]
    )
    context = empty if batch.context is None else torch.stack([
        getattr(batch.context, name)
        for name in ("pc", "cr0", "cr3", "cr4", "efer", "cs_base", "mode", "known")
    ]).reshape(-1)
    groups = (
        histogram(batch.addresses),
        histogram(register_values, register_mask),
        histogram(empty if memory is None else memory.addresses),
        histogram(empty if memory is None else
                  (memory.sizes.float().log2().clamp(max=15).to(torch.int64) << 4)),
        histogram(memory_values, memory_mask),
        histogram(context),
        histogram(empty if batch.layout is None else batch.layout.values),
    )
    return torch.cat(groups)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def allocated_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory()
    return None


def make_model(device: torch.device, seed: int) -> torch.nn.Sequential:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(FEATURE_COUNT, 64, dtype=torch.float32), torch.nn.ReLU(),
                               torch.nn.Linear(64, FEATURE_COUNT, dtype=torch.float32)).to(device)


def _write_failure(output: str | Path, error: BaseException, metrics: dict | None = None) -> None:
    """Keep partial counters when a runner rejects an otherwise complete trace."""
    directory = Path(output).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    failure_path = directory / "failure.json"
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
    failure.update(metrics or {})
    failure.update(all_traces_complete=False, error_type=type(error).__name__, error=str(error))
    (directory / "metrics.json").unlink(missing_ok=True)
    failure_path.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")


def benchmark(
    endpoints: Sequence[str], output: str | Path, *, device: str = "cpu", mode: str = "drain",
    group_size: int = 64, timeout: float = 30, seed: int = 17,
    capture_hosts: Sequence[str] = (), batch_bytes: int = 0, max_seconds: float | None = None,
    write_metrics: bool = True,
) -> dict:
    """Consume all workers through validated completion and write measured results.

    next_batch_seconds includes waiting, decoding and full rich-column upload;
    background readers can overlap work in multiendpoint runs. Device fences at
    phase boundaries make these wall-time measurements explicit, but inhibit
    overlap. They are not GPU utilization measurements or kernel timings.
    Internal runners use write_metrics=False until their extra checks pass.
    """
    if mode not in ("drain", "train"):
        raise ValueError("Mode must be drain or train")
    if not 1 <= group_size <= 4096:
        raise ValueError("Summary group size must be between 1 and 4096")
    if capture_hosts and len(capture_hosts) != len(endpoints):
        raise ValueError("Provide one capture host label per endpoint")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("Measurement budget must be positive")
    setup_started = time.perf_counter()
    selected = torch.device(device)
    pool = Pool(endpoints, device=device, timeout=timeout, **({"batch_bytes": batch_bytes} if batch_bytes else {}))
    directory = Path(output).expanduser()
    directory.mkdir(parents=True, exist_ok=False)
    model = make_model(selected, seed) if mode == "train" else None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if model is not None else None
    initial = None if model is None else {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    synchronize(selected)
    setup_seconds = time.perf_counter() - setup_started
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)

    rows = {name: 0 for name in ROW_KINDS}
    sources: dict[tuple[int, int | None], dict[str, int]] = {}
    pending: list[torch.Tensor] = []
    probe = None
    batches = 0
    updates = 0
    trained_summaries = 0
    loss_sum = 0.0
    first_loss = last_loss = None
    next_seconds = feature_seconds = model_seconds = accounting_seconds = 0.0
    maximum_batch_bytes = 0
    sampled_device_bytes = allocated_bytes(selected)

    def sample_memory() -> None:
        nonlocal sampled_device_bytes
        allocated = allocated_bytes(selected)
        if allocated is not None:
            sampled_device_bytes = max(sampled_device_bytes or 0, allocated)

    def update() -> None:
        nonlocal probe, updates, trained_summaries, loss_sum, first_loss, last_loss, model_seconds
        if not pending:
            return
        assert optimizer is not None and model is not None
        started = time.perf_counter()
        data = torch.stack(pending)
        pending.clear()
        if probe is None:
            probe = data[:1].clone()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(data), data)
        loss.backward()
        finite = [torch.isfinite(loss), *[
            torch.isfinite(parameter.grad).all() for parameter in model.parameters()
        ]]
        if not bool(torch.stack(finite).all().item()):
            raise RuntimeError("Reconstruction produced a non-finite loss or gradient")
        optimizer.step()
        if not bool(torch.stack([torch.isfinite(value).all() for value in model.parameters()]).all().item()):
            raise RuntimeError("Reconstruction produced non-finite weights")
        synchronize(selected)
        value = float(loss.detach().cpu())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        loss_sum += value * data.shape[0]
        updates += 1
        trained_summaries += data.shape[0]
        sample_memory()
        model_seconds += time.perf_counter() - started

    started = time.perf_counter()
    usage_started = resource.getrusage(resource.RUSAGE_SELF)
    try:
        with pool:
            iterator = pool.read()
            while True:
                if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                    raise TimeoutError("Measurement exceeded its total consumption budget")
                phase_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    next_seconds += time.perf_counter() - phase_started
                    break
                synchronize(selected)
                next_seconds += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                batches += 1
                counted = row_counts(batch)
                source = sources.setdefault((batch.worker, batch.source), {name: 0 for name in ROW_KINDS})
                for name, count in counted.items():
                    rows[name] += count
                    source[name] += count
                maximum_batch_bytes = max(maximum_batch_bytes, tensor_bytes(batch))
                sample_memory()
                accounting_seconds += time.perf_counter() - phase_started
                if model is not None:
                    phase_started = time.perf_counter()
                    pending.append(summarize(batch))
                    synchronize(selected)
                    sample_memory()
                    feature_seconds += time.perf_counter() - phase_started
                    if len(pending) == group_size:
                        update()
                del batch
            update()
        synchronize(selected)
        elapsed = time.perf_counter() - started
        usage_finished = resource.getrusage(resource.RUSAGE_SELF)
        peak_device_bytes = torch.cuda.max_memory_allocated(selected) if selected.type == "cuda" else None
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
        validation_started = time.perf_counter()
        changed = reloaded = None
        if model is not None:
            if not updates:
                raise ValueError("Training needs at least one observation batch")
            changed = any(not torch.equal(initial[name], value.detach().cpu()) for name, value in model.state_dict().items())
            if not changed:
                raise RuntimeError("Reconstruction did not change the model weights")
            checkpoint = directory / "model.pt"
            torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                        "feature_groups": list(FEATURE_GROUPS), "histogram_bins": HISTOGRAM_BINS}, checkpoint)
            restored = make_model(selected, seed)
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
            with torch.no_grad():
                torch.testing.assert_close(model(probe), restored(probe), rtol=0, atol=0)
            synchronize(selected)
            reloaded = True
        events = sum(rows[name] for name in ROW_KINDS if name != "layout")
        result = {
            "format": 1, "completed_at": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
            "python_version": platform.python_version(), "torch_version": str(torch.__version__),
            "cpu2tensor_version": version("cpu2tensor"), "device": str(selected),
            "device_name": torch.cuda.get_device_name(selected) if selected.type == "cuda" else selected.type,
            "cuda_runtime_version": torch.version.cuda,
            "endpoints": list(endpoints), "capture_hosts": list(capture_hosts), "mode": mode,
            "all_traces_complete": True, "workers_completed": len(endpoints), "batches": batches,
            "batch_bytes": batch_bytes, "max_seconds": max_seconds,
            "process_cpu_seconds": (usage_finished.ru_utime + usage_finished.ru_stime
                                    - usage_started.ru_utime - usage_started.ru_stime),
            "rows": rows, "events": events, "events_per_second": events / elapsed,
            "per_source": [{"worker": worker, "source": source, "rows": counts}
                           for (worker, source), counts in sorted(sources.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1]))],
            "setup_seconds": setup_seconds, "elapsed_seconds": elapsed,
            "next_batch_seconds": next_seconds, "feature_seconds": feature_seconds,
            "model_seconds": model_seconds, "accounting_seconds": accounting_seconds,
            "validation_seconds": time.perf_counter() - validation_started,
            "timing_synchronizes_device": selected.type != "cpu", "max_batch_tensor_bytes": maximum_batch_bytes,
            "host_peak_rss_bytes": peak_rss, "device_allocated_peak_bytes": peak_device_bytes,
            "device_allocated_observed_max_bytes": sampled_device_bytes,
            "memory_notes": "Host RSS is a process-lifetime high-water mark. CUDA peak covers the measured pipeline. MPS has sampled allocation only; transient peaks can be missed. Logical batch bytes exclude queues and allocator caches.",
            "seed": seed, "summary_group_size": group_size, "summary_features": FEATURE_COUNT,
            "feature_groups": list(FEATURE_GROUPS), "updates": updates, "trained_summaries": trained_summaries,
            "first_loss": first_loss, "last_loss": last_loss,
            "mean_loss": loss_sum / trained_summaries if trained_summaries else None,
            "weights_changed": changed, "checkpoint_reloaded": reloaded,
            "learning_scope": "Summary reconstruction exercises the pipeline; losses are not task learning or held-out accuracy.",
        }
        if write_metrics:
            (directory / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    except BaseException as error:
        failure = dict(all_traces_complete=False, error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=time.perf_counter() - started, batches=batches, rows=rows,
                       device=str(selected), endpoints=list(endpoints), batch_bytes=batch_bytes,
                       per_source=[dict(worker=worker, source=source, rows=counts)
                                   for (worker, source), counts in sources.items()],
                       max_seconds=max_seconds, updates=updates)
        _write_failure(directory, error, failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", required=True, help="Operator-started endpoint; repeat for more workers")
    parser.add_argument("--capture-host", action="append", default=[], help="Optional host label for each endpoint, in the same order")
    parser.add_argument("--output", type=Path, required=True, help="New output directory for metrics and an optional checkpoint")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("drain", "train"), default="drain")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--batch-bytes", type=int, default=0, help="CPU collation target; zero keeps individual frames")
    parser.add_argument("--max-seconds", type=float, help="Consumption budget checked between batches; socket timeout still applies")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = benchmark(args.endpoint, args.output, device=args.device, mode=args.mode,
                       group_size=args.group_size, timeout=args.timeout, seed=args.seed,
                       capture_hosts=args.capture_host, batch_bytes=args.batch_bytes, max_seconds=args.max_seconds)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
