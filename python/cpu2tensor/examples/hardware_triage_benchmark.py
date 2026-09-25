# SPDX-License-Identifier: AGPL-3.0-only
"""Pretrain and calibrate frozen raw-PT triage on benign kernel workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import torch

from cpu2tensor.examples.hardware_triage import (
    RawTraceSketchConfig,
    fit_raw_trace_pca,
    load_frozen_raw_trace_pca,
    pack_raw_traces,
    raw_trace_sketch,
)
from cpu2tensor.hardware import HardwareConfig, PerfCapture


WORKLOAD_LOOPS = {
    "getpid": 250,
    "fstat": 250,
    "futex": 250,
    "openat": 100,
    "pipe": 100,
    "mmap": 100,
    "eventfd": 100,
    "epoll": 100,
    "socketpair": 100,
    "getrandom": 250,
    "memfd": 100,
    "ioctl": 100,
    "dup": 250,
    "yield": 250,
    "uname": 250,
    "readlink": 100,
    "fork": 20,
}
HOLDOUT_FAMILIES = ("epoll", "memfd", "fork")
SKETCH_CONFIG = RawTraceSketchConfig(segments=4, pair_bins=0, include_length=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _capture(
    binary: Path,
    family: str,
    loops: int,
    *,
    capture_cpu: int,
) -> tuple[torch.Tensor, float]:
    process = subprocess.Popen(
        ("taskset", "-c", str(capture_cpu), str(binary), family, str(loops)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stdout.readline() != b"READY\n":
        process.kill()
        process.communicate()
        raise RuntimeError(f"{family} did not reach its capture gate")
    started = time.perf_counter()
    config = HardwareConfig(
        "process_kernel",
        pid=process.pid,
        signal="intel_pt",
        data_pages=64,
        aux_pages=2048,
    )
    try:
        with PerfCapture(config) as capture:
            output, error = process.communicate(b"x", timeout=30)
            batches = capture.stop()
    except BaseException:
        process.kill()
        process.communicate()
        raise
    elapsed = time.perf_counter() - started
    if process.returncode != 0 or error or len(batches) != 1:
        raise RuntimeError(
            f"{family} failed: status={process.returncode}, "
            f"stdout={output!r}, stderr={error!r}, batches={len(batches)}"
        )
    if batches[0].trace_bytes.numel() == 0:
        raise RuntimeError(f"{family} produced an empty Intel PT trace")
    return batches[0].trace_bytes, elapsed


def _split_rows(
    rows: dict[str, torch.Tensor],
    *,
    training_rows: int,
    calibration_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = []
    calibration = []
    validation = []
    for family, family_rows in rows.items():
        if family in HOLDOUT_FAMILIES:
            continue
        train.append(family_rows[:training_rows])
        calibration.append(family_rows[training_rows:training_rows + calibration_rows])
        validation.append(family_rows[training_rows + calibration_rows:])
    return torch.cat(train), torch.cat(calibration), torch.cat(validation)


def _scores_per_million(scores: torch.Tensor, threshold: float) -> float:
    return float((scores > threshold).sum()) * 1_000_000.0 / scores.numel()


def leave_one_family_out(
    rows: dict[str, torch.Tensor],
    *,
    training_rows: int,
    calibration_rows: int,
    latent_dimensions: int,
    reviews_per_million: float,
) -> dict[str, dict[str, float]]:
    """Measure whether the frozen baseline mistakes each unseen benign family for a bug."""
    results = {}
    quantile = 1.0 - reviews_per_million / 1_000_000.0
    for heldout_family, holdout in rows.items():
        known = [values for family, values in rows.items() if family != heldout_family]
        train = torch.cat([values[:training_rows] for values in known])
        calibration = torch.cat(
            [values[training_rows:training_rows + calibration_rows] for values in known]
        )
        validation = torch.cat(
            [values[training_rows + calibration_rows:] for values in known]
        )
        model = fit_raw_trace_pca(
            train,
            config=SKETCH_CONFIG,
            latent_dimensions=latent_dimensions,
        )
        calibration_scores = model.anomaly_score(calibration, config=SKETCH_CONFIG)
        validation_scores = model.anomaly_score(validation, config=SKETCH_CONFIG)
        holdout_scores = model.anomaly_score(holdout, config=SKETCH_CONFIG)
        threshold = float(
            torch.quantile(calibration_scores, quantile, interpolation="higher")
        )
        results[heldout_family] = {
            "threshold": threshold,
            "validation_reviews_per_million": _scores_per_million(
                validation_scores, threshold
            ),
            "holdout_reviews_per_million": _scores_per_million(holdout_scores, threshold),
            "holdout_score_median": float(holdout_scores.median()),
            "holdout_score_max": float(holdout_scores.max()),
        }
    return results


def run(args: argparse.Namespace) -> dict[str, object]:
    if platform.system() != "Linux":
        raise RuntimeError("The raw-PT benchmark requires Linux")
    binary = args.binary.resolve()
    if not binary.is_file():
        raise ValueError(f"Missing workload executable: {binary}")
    if args.repetitions <= args.training_rows + args.calibration_rows:
        raise ValueError("repetitions must leave independent validation rows")
    if not 0 < args.reviews_per_million < 1_000_000:
        raise ValueError("reviews-per-million must be between zero and one million")

    os.sched_setaffinity(0, {args.controller_cpu})
    torch.set_num_threads(1)
    order = [family for family in WORKLOAD_LOOPS for _ in range(args.repetitions)]
    random.Random(args.seed).shuffle(order)
    features = {family: [] for family in WORKLOAD_LOOPS}
    traces = {family: [] for family in WORKLOAD_LOOPS}
    trace_sizes = {family: [] for family in WORKLOAD_LOOPS}
    durations = {family: [] for family in WORKLOAD_LOOPS}

    for family, loops in WORKLOAD_LOOPS.items():
        _capture(binary, family, loops, capture_cpu=args.capture_cpu)
    collection_started = time.perf_counter()
    for family in order:
        trace, elapsed = _capture(
            binary,
            family,
            WORKLOAD_LOOPS[family],
            capture_cpu=args.capture_cpu,
        )
        raw, offsets = pack_raw_traces((trace,))
        features[family].append(raw_trace_sketch(raw, offsets, SKETCH_CONFIG)[0])
        traces[family].append(trace)
        trace_sizes[family].append(trace.numel())
        durations[family].append(elapsed)
    collection_seconds = time.perf_counter() - collection_started

    rows = {family: torch.stack(values) for family, values in features.items()}
    train, calibration, validation = _split_rows(
        rows,
        training_rows=args.training_rows,
        calibration_rows=args.calibration_rows,
    )
    model = fit_raw_trace_pca(
        train,
        config=SKETCH_CONFIG,
        latent_dimensions=args.latent_dimensions,
    )
    calibration_scores = model.anomaly_score(calibration, config=SKETCH_CONFIG)
    validation_scores = model.anomaly_score(validation, config=SKETCH_CONFIG)
    holdout = torch.cat([rows[family] for family in HOLDOUT_FAMILIES])
    holdout_scores = model.anomaly_score(holdout, config=SKETCH_CONFIG)
    quantile = 1.0 - args.reviews_per_million / 1_000_000.0
    threshold = float(torch.quantile(calibration_scores, quantile, interpolation="higher"))
    family_holdouts = leave_one_family_out(
        rows,
        training_rows=args.training_rows,
        calibration_rows=args.calibration_rows,
        latent_dimensions=args.latent_dimensions,
        reviews_per_million=args.reviews_per_million,
    )
    worst_holdout_rate = max(
        result["holdout_reviews_per_million"] for result in family_holdouts.values()
    )

    family_results = {}
    for family, family_features in rows.items():
        scores = model.anomaly_score(family_features, config=SKETCH_CONFIG)
        family_results[family] = {
            "trace_bytes_median": statistics.median(trace_sizes[family]),
            "capture_seconds_median": statistics.median(durations[family]),
            "score_median": float(scores.median()),
            "score_max": float(scores.max()),
            "reviews_per_million": _scores_per_million(scores, threshold),
        }

    subject = {
        "host": platform.node(),
        "kernel": platform.release(),
        "kernel_version": platform.version(),
        "boot_id": _read("/proc/sys/kernel/random/boot_id"),
        "microcode": _read("/sys/devices/system/cpu/cpu0/microcode/version"),
        "kernel_btf_sha256": _sha256(Path("/sys/kernel/btf/vmlinux")),
        "kernel_notes_sha256": _sha256(Path("/sys/kernel/notes")),
    }
    ordered_traces = [trace for family in WORKLOAD_LOOPS for trace in traces[family]]
    trace_bytes, trace_offsets = pack_raw_traces(ordered_traces)
    corpus = {
        "trace_bytes": trace_bytes,
        "trace_offsets": trace_offsets,
        "families": [family for family in WORKLOAD_LOOPS for _ in traces[family]],
        "features": torch.cat([rows[family] for family in WORKLOAD_LOOPS]),
        "sketch": {
            "segments": SKETCH_CONFIG.segments,
            "pair_bins": SKETCH_CONFIG.pair_bins,
            "include_length": SKETCH_CONFIG.include_length,
        },
        "subject": subject,
        "workloads": WORKLOAD_LOOPS,
        "seed": args.seed,
    }
    args.corpus.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corpus, args.corpus)
    corpus_sha256 = _sha256(args.corpus)
    checkpoint = {
        "mean": model.mean,
        "scale": model.scale,
        "components": model.components,
        "sketch": {
            "segments": SKETCH_CONFIG.segments,
            "pair_bins": SKETCH_CONFIG.pair_bins,
            "include_length": SKETCH_CONFIG.include_length,
        },
        "threshold": threshold,
        "reviews_per_million": args.reviews_per_million,
        "subject": subject,
        "workloads": WORKLOAD_LOOPS,
        "holdout_families": HOLDOUT_FAMILIES,
        "seed": args.seed,
        "corpus_sha256": corpus_sha256,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    restored, restored_threshold = load_frozen_raw_trace_pca(args.checkpoint)
    restored_scores = restored.anomaly_score(validation, config=SKETCH_CONFIG)
    if restored_threshold != threshold or not torch.equal(restored_scores, validation_scores):
        raise RuntimeError("Reloaded frozen triage checkpoint changed inference")
    report = {
        "decision": (
            "GO"
            if worst_holdout_rate <= args.holdout_limit
            else "NO-GO"
        ),
        "subject": subject,
        "workload_sha256": _sha256(binary),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_reload": True,
        "corpus": str(args.corpus),
        "corpus_sha256": corpus_sha256,
        "capture_cpu": args.capture_cpu,
        "controller_cpu": args.controller_cpu,
        "repetitions_per_family": args.repetitions,
        "training_rows_per_familiar_family": args.training_rows,
        "calibration_rows_per_familiar_family": args.calibration_rows,
        "validation_rows": validation.shape[0],
        "heldout_rows": holdout.shape[0],
        "review_budget_per_million": args.reviews_per_million,
        "holdout_limit_per_million": args.holdout_limit,
        "threshold": threshold,
        "validation_reviews_per_million": _scores_per_million(validation_scores, threshold),
        "heldout_reviews_per_million": _scores_per_million(holdout_scores, threshold),
        "worst_single_family_holdout_reviews_per_million": worst_holdout_rate,
        "leave_one_family_out": family_holdouts,
        "collection_seconds": collection_seconds,
        "capture_executions_per_second": len(order) / collection_seconds,
        "total_trace_bytes": sum(sum(values) for values in trace_sizes.values()),
        "family": family_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--capture-cpu", type=int, default=2)
    parser.add_argument("--controller-cpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--repetitions", type=int, default=512)
    parser.add_argument("--training-rows", type=int, default=64)
    parser.add_argument("--calibration-rows", type=int, default=128)
    parser.add_argument("--latent-dimensions", type=int, default=8)
    parser.add_argument("--reviews-per-million", type=float, default=1_000.0)
    parser.add_argument("--holdout-limit", type=float, default=100_000.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(_arguments()), indent=2, sort_keys=True))
