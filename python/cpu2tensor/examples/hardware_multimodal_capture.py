# SPDX-License-Identifier: AGPL-3.0-only
"""Qualify simultaneous PT, PEBS, and PMU capture on a gated Linux target."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

from cpu2tensor.hardware import HardwareMultimodalConfig, PerfMultimodalCapture


ARMS = {
    "baseline": None,
    "pmu": ("counters",),
    "pt_pmu": ("intel_pt", "counters"),
    "pt_pebs_pmu": ("intel_pt", "memory_loads", "counters"),
}


def _run_target(args: argparse.Namespace, arm: str) -> dict[str, object]:
    process = subprocess.Popen(
        (
            "taskset", "-c", str(args.cpu), str(args.binary),
            args.family, str(args.loops),
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stdout.readline() != b"READY\n":
        process.kill()
        output, error = process.communicate()
        raise RuntimeError(f"target did not reach READY: stdout={output!r}, stderr={error!r}")
    modalities = ARMS[arm]
    batches = ()
    try:
        if modalities is None:
            started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            output, error = process.communicate(b"x", timeout=args.timeout)
            elapsed_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - started
        else:
            config = HardwareMultimodalConfig(
                "process_kernel",
                process.pid,
                modalities=modalities,
                pebs_period=args.pebs_period,
                data_pages=args.data_pages,
                aux_pages=args.aux_pages,
            )
            with PerfMultimodalCapture(config) as capture:
                started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                output, error = process.communicate(b"x", timeout=args.timeout)
                elapsed_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - started
                batches = capture.stop()
    except BaseException:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0 or error or len(batches) > 1:
        raise RuntimeError(
            f"target failed: status={process.returncode}, stdout={output!r}, "
            f"stderr={error!r}, batches={len(batches)}"
        )
    text = output.decode("ascii").strip()
    row: dict[str, object] = {
        "arm": arm,
        "elapsed_ns": elapsed_ns,
        "output": text,
        "pid": process.pid,
        "tid": process.pid,
        "cpu": args.cpu,
        "pt_bytes": 0,
        "pebs_samples": 0,
        "pebs_time_min_ns": None,
        "pebs_time_max_ns": None,
        "pebs_cpus": [],
        "pebs_tids": [],
        "pebs_exact_ip_all": False,
        "pebs_nonzero_addresses": 0,
        "pmu": {},
        "pmu_time_enabled_ns": 0,
        "pmu_time_running_ns": 0,
        "arm_skew_ns": 0,
        "stop_skew_ns": 0,
        "status": [],
        "envelope": None,
    }
    if batches:
        batch = batches[0]
        if batch.tid != process.pid or batch.cpu != -1:
            raise RuntimeError("perf returned an unexpected task-following source identity")
        row["pt_bytes"] = 0 if batch.pt is None else batch.pt.trace_bytes.numel()
        row["pebs_samples"] = 0 if batch.pebs is None else batch.pebs.ip.numel()
        if batch.pebs is not None and batch.pebs.ip.numel():
            row["pebs_time_min_ns"] = int(batch.pebs.time.min())
            row["pebs_time_max_ns"] = int(batch.pebs.time.max())
            row["pebs_cpus"] = sorted(set(batch.pebs.cpu.tolist()))
            row["pebs_tids"] = sorted(set(batch.pebs.tid.tolist()))
            row["pebs_exact_ip_all"] = bool(batch.pebs.exact_ip.all())
            row["pebs_nonzero_addresses"] = int((batch.pebs.address != 0).sum())
        if batch.counters is not None:
            row["pmu"] = dict(zip(batch.counters.names, batch.counters.values.tolist()))
            row["pmu_time_enabled_ns"] = batch.counters.time_enabled_ns
            row["pmu_time_running_ns"] = batch.counters.time_running_ns
        row["arm_skew_ns"] = (
            batch.envelope.arm_after_ns - batch.envelope.arm_before_ns
        )
        row["stop_skew_ns"] = (
            batch.envelope.stop_after_ns - batch.envelope.stop_before_ns
        )
        row["status"] = [status.__dict__ for status in batch.status]
        row["envelope"] = batch.envelope.__dict__
    return row


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.binary.is_file():
        raise RuntimeError(f"target is not a file: {args.binary}")
    available = os.cpu_count()
    if available is None or args.cpu < 0 or args.cpu >= available:
        raise RuntimeError(f"CPU {args.cpu} is not an online logical CPU")
    schedule = list(ARMS) * args.runs
    random.Random(args.seed).shuffle(schedule)
    rows = [_run_target(args, arm) for arm in schedule]
    outputs = {row["output"] for row in rows}
    if len(outputs) != 1:
        raise RuntimeError("instrumentation changed the target's exact output")
    tri_modal = [row for row in rows if row["arm"] == "pt_pebs_pmu"]
    if any(row["pt_bytes"] == 0 or row["pebs_samples"] == 0 for row in tri_modal):
        raise RuntimeError("a tri-modal arm did not produce both PT bytes and PEBS samples")
    for row in tri_modal:
        envelope = row["envelope"]
        if (row["pebs_cpus"] != [args.cpu] or row["pebs_tids"] != [row["tid"]] or
                row["pebs_time_min_ns"] < envelope["arm_before_ns"] or
                row["pebs_time_max_ns"] > envelope["stop_after_ns"]):
            raise RuntimeError("PEBS samples escaped their declared CPU, task, or time envelope")
    for row in rows:
        if row["arm"] == "baseline":
            continue
        statuses = [status for status in row["status"] if status["requested"]]
        if not statuses or any(not status["available"] or status["lost"] for status in statuses):
            raise RuntimeError("a requested hardware source was unavailable or lost")
        if row["pmu"] and row["pmu_time_enabled_ns"] != row["pmu_time_running_ns"]:
            raise RuntimeError("PMU group was multiplexed")
    elapsed = {
        arm: [int(row["elapsed_ns"]) for row in rows if row["arm"] == arm]
        for arm in ARMS
    }
    baseline = statistics.median(elapsed["baseline"])
    return {
        "schema": "cpu2tensor-hardware-multimodal-qualification-v1",
        "binary": str(args.binary.resolve()),
        "family": args.family,
        "loops": args.loops,
        "cpu": args.cpu,
        "seed": args.seed,
        "runs_per_arm": args.runs,
        "schedule": schedule,
        "output": next(iter(outputs)),
        "arms": {
            arm: {
                "median_elapsed_ns": statistics.median(values),
                "median_overhead_ratio": statistics.median(values) / baseline,
            }
            for arm, values in elapsed.items()
        },
        "maximum_arm_skew_ns": max(int(row["arm_skew_ns"]) for row in rows),
        "maximum_stop_skew_ns": max(int(row["stop_skew_ns"]) for row in rows),
        "rows": rows,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--binary", type=Path, required=True)
    result.add_argument("--cpu", type=int, default=2)
    result.add_argument("--family", default="mmap")
    result.add_argument("--loops", type=int, default=1000)
    result.add_argument("--runs", type=int, default=5)
    result.add_argument("--seed", type=int, default=20260925)
    result.add_argument("--pebs-period", type=int, default=10_000)
    result.add_argument("--data-pages", type=int, default=64)
    result.add_argument("--aux-pages", type=int, default=2048)
    result.add_argument("--timeout", type=float, default=30.0)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.loops <= 0 or args.runs <= 0 or args.timeout <= 0:
        raise SystemExit("loops, runs, and timeout must be positive")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
