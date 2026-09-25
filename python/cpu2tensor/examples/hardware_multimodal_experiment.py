# SPDX-License-Identifier: AGPL-3.0-only
"""Collect, train, and evaluate the first kernel-only multimodal corpus.

Collection is deliberately limited to the gated benign syscall workload.  Every
accepted execution is a separate custody boundary: raw perf tensors, the exact
target output, derived model tensors, and their hashes are sealed before the next
execution starts.  Training can therefore happen later on another machine.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import time
from typing import Sequence

import torch

from cpu2tensor.examples.hardware_multimodal import (
    HardwareMultimodalBatch as ModelBatch,
    MaskedHardwareModel,
    MultimodalConfig,
    calibrate_multimodal_threshold,
    freeze_multimodal_model,
    load_frozen_multimodal_model,
    multimodal_anomaly_score,
    save_frozen_multimodal_model,
    train_masked_model,
)
from cpu2tensor.hardware import (
    HardwareBatch,
    HardwareCounterBatch,
    HardwareMultimodalBatch,
    HardwareMultimodalConfig,
    HardwareCaptureError,
    HardwareTraceLost,
    PerfMultimodalCapture,
)


SCHEMA = "cpu2tensor-kernel-multimodal-experiment-v1"
RAW_SCHEMA = "cpu2tensor-kernel-multimodal-raw-v1"
DERIVED_SCHEMA = "cpu2tensor-kernel-multimodal-derived-v1"
SCOPE = "process_kernel"
PEBS_PERIOD = 10_000
MODALITIES = ("intel_pt", "memory_loads", "counters")
COUNTER_NAMES = ("instructions", "cycles", "ref_cycles")

# These are the only accepted targets.  They are the gated benign families in
# hardware_kernel_workload, not process-scope canaries or vulnerability triggers.
WORKLOAD_LOOPS = {
    "getpid": 5_000,
    "fstat": 5_000,
    "futex": 5_000,
    "openat": 5_000,
    "pipe": 5_000,
    "mmap": 5_000,
    "eventfd": 5_000,
    "epoll": 5_000,
    "socketpair": 5_000,
    "getrandom": 5_000,
    "memfd": 5_000,
    "ioctl": 5_000,
    "dup": 5_000,
    "yield": 5_000,
    "uname": 5_000,
    "readlink": 5_000,
    "fork": 100,
}


@dataclass(frozen=True)
class PlannedExecution:
    execution_id: str
    family: str
    repetition: int
    partition: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _optional_sha256(path: str) -> str | None:
    try:
        return _sha256(Path(path))
    except OSError:
        return None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return _sha256(path)


def _cpu_model() -> str | None:
    text = _read("/proc/cpuinfo")
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def _git_identity(module: Path) -> dict[str, object]:
    root = module
    while root != root.parent and not (root / ".git").exists():
        root = root.parent
    if not (root / ".git").exists():
        return {"revision": None, "dirty_diff_sha256": None}
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=root, check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD"), cwd=root, check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "revision": revision,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _elf_build_id(binary: Path) -> str | None:
    try:
        notes = subprocess.run(
            ("readelf", "-n", str(binary)), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", notes)
    return None if match is None else match.group(1).lower()


def _module_identity(module: Path) -> dict[str, str]:
    return {"path": str(module.resolve()), "sha256": _sha256(module)}


def subject_manifest(binary: Path, *, target_cpu: int, controller_cpu: int,
                     data_pages: int, aux_pages: int) -> dict[str, object]:
    """Return the immutable build, boot, and event identity for collection."""
    runner = Path(__file__).resolve()
    package = runner.parents[1]
    hardware = package / "hardware.py"
    model = runner.with_name("hardware_multimodal.py")
    feature_module = runner.with_name("hardware_multimodal_features.py")
    build = {
        "workload_path": str(binary.resolve()),
        "workload_sha256": _sha256(binary),
        "workload_elf_build_id": _elf_build_id(binary),
        "runner": _module_identity(runner),
        "capture_module": _module_identity(hardware),
        "model_module": _module_identity(model),
        "feature_module": _module_identity(feature_module),
        "source": _git_identity(runner),
    }
    subject = {
        "host": platform.node(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "kernel_release": platform.release(),
        "kernel_version": platform.version(),
        "boot_id": _read("/proc/sys/kernel/random/boot_id"),
        "microcode": _read("/sys/devices/system/cpu/cpu0/microcode/version"),
        "kernel_btf_sha256": _optional_sha256("/sys/kernel/btf/vmlinux"),
        "kernel_notes_sha256": _optional_sha256("/sys/kernel/notes"),
        "kernel_cmdline_sha256": _optional_sha256("/proc/cmdline"),
        "perf_event_paranoid": _read("/proc/sys/kernel/perf_event_paranoid"),
        "target_cpu": target_cpu,
        "controller_cpu": controller_cpu,
    }
    event = {
        "scope": SCOPE,
        "modalities": list(MODALITIES),
        "intel_pt_representation": "raw_aux_bytes_no_decode",
        "pebs_event": "memory_loads",
        "pebs_period": PEBS_PERIOD,
        "boundary_counters": list(COUNTER_NAMES),
        "data_pages": data_pages,
        "aux_pages": aux_pages,
    }
    manifest = {"subject": subject, "build": build, "event": event}
    manifest["identity_sha256"] = _json_hash(manifest)
    return manifest


def make_plan(
    families: Sequence[str],
    *,
    repetitions: int,
    training_rows: int,
    calibration_rows: int,
    heldout_family_count: int,
    seed: int,
) -> tuple[tuple[PlannedExecution, ...], tuple[str, ...]]:
    """Randomize whole families and executions without splitting an execution."""
    families = tuple(families)
    if not families or len(set(families)) != len(families):
        raise ValueError("families must be nonempty and distinct")
    if any(family not in WORKLOAD_LOOPS for family in families):
        raise ValueError("only gated hardware_kernel_workload families are allowed")
    if repetitions <= training_rows + calibration_rows:
        raise ValueError("repetitions must leave familiar validation executions")
    if training_rows <= 0 or calibration_rows <= 0:
        raise ValueError("training and calibration row counts must be positive")
    if not 0 < heldout_family_count < len(families):
        raise ValueError("heldout family count must leave familiar families")

    generator = random.Random(seed)
    heldout = tuple(sorted(generator.sample(families, heldout_family_count)))
    plan = []
    for family in families:
        repetitions_for_family = list(range(repetitions))
        generator.shuffle(repetitions_for_family)
        for ordinal, repetition in enumerate(repetitions_for_family):
            if family in heldout:
                partition = "heldout_family"
            elif ordinal < training_rows:
                partition = "training"
            elif ordinal < training_rows + calibration_rows:
                partition = "calibration"
            else:
                partition = "familiar_validation"
            plan.append(PlannedExecution(
                execution_id=f"{family}-{repetition:05d}",
                family=family,
                repetition=repetition,
                partition=partition,
            ))
    generator.shuffle(plan)
    return tuple(plan), heldout


def _hardware_payload(batch: HardwareBatch | None) -> dict[str, object] | None:
    if batch is None:
        return None
    return {
        "source": batch.source,
        "signal": batch.signal,
        **{name: getattr(batch, name).cpu() for name in (
            "ip", "pid", "tid", "time", "cpu", "period", "address", "weight",
            "data_source", "exact_ip", "trace_bytes",
        )},
    }


def _counter_payload(batch: HardwareCounterBatch | None) -> dict[str, object] | None:
    if batch is None:
        return None
    return {
        "source": batch.source,
        "tid": batch.tid,
        "cpu": batch.cpu,
        "names": list(batch.names),
        "values": batch.values.cpu(),
        "time_enabled_ns": batch.time_enabled_ns,
        "time_running_ns": batch.time_running_ns,
        "available": batch.available,
        "lost": batch.lost,
    }


def raw_capture_payload(
    batches: Sequence[HardwareMultimodalBatch],
    *,
    execution: PlannedExecution,
    loops: int,
    stdout: bytes,
    elapsed_ns: int,
) -> dict[str, object]:
    return {
        "schema": RAW_SCHEMA,
        "execution": asdict(execution),
        "loops": loops,
        "stdout": torch.tensor(list(stdout), dtype=torch.uint8),
        "elapsed_ns": elapsed_ns,
        "batches": [{
            "source": batch.source,
            "tid": batch.tid,
            "cpu": batch.cpu,
            "envelope": asdict(batch.envelope),
            "status": [asdict(status) for status in batch.status],
            "pt": _hardware_payload(batch.pt),
            "pebs": _hardware_payload(batch.pebs),
            "counters": _counter_payload(batch.counters),
        } for batch in batches],
    }


def _model_payload(batch: ModelBatch) -> dict[str, torch.Tensor]:
    if batch.batch_size != 1:
        raise ValueError("one derived shard must contain exactly one execution")
    return {name: getattr(batch, name).cpu() for name in (
        "pt", "pebs", "pmu", "pt_available", "pebs_available", "pmu_available",
        "time_bounds", "timing_quality",
    )}


def _model_batch(payload: dict[str, torch.Tensor]) -> ModelBatch:
    return ModelBatch(**payload)


def _concatenate(batches: Sequence[ModelBatch]) -> ModelBatch:
    if not batches:
        raise ValueError("a dataset partition must not be empty")
    cpu_count = batches[0].cpu_count
    if any(batch.cpu_count != cpu_count for batch in batches):
        raise ValueError("all executions must have the same lane count")
    return ModelBatch(*(
        torch.cat([getattr(batch, name) for batch in batches], dim=0)
        for name in (
            "pt", "pebs", "pmu", "pt_available", "pebs_available", "pmu_available",
            "time_bounds", "timing_quality",
        )
    ))


def _feature_statistics(
    values: torch.Tensor, available: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = values.reshape(-1, values.shape[-1])[available.reshape(-1)]
    if selected.shape[0] == 0:
        return torch.zeros(values.shape[-1]), torch.ones(values.shape[-1])
    return selected.mean(0), selected.std(0, correction=0).clamp_min(1e-4)


def _top_two_score(error: torch.Tensor, available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat_error = error.flatten(1)
    flat_available = available.flatten(1)
    ordered = flat_error.masked_fill(~flat_available, -torch.inf).sort(
        1, descending=True
    ).values
    counts = flat_available.sum(1)
    ranks = torch.arange(ordered.shape[1])[None, :]
    selected = ranks < counts.clamp(max=2)[:, None]
    score = torch.where(selected, ordered, torch.zeros_like(ordered)).sum(1)
    score /= counts.clamp(min=1, max=2)
    return score, counts > 0


@dataclass(frozen=True)
class MarginalBaseline:
    """Independent per-modality feature means with no cross-modal learning."""

    pt_mean: torch.Tensor
    pt_scale: torch.Tensor
    pebs_mean: torch.Tensor
    pebs_scale: torch.Tensor
    pmu_mean: torch.Tensor
    pmu_scale: torch.Tensor

    @classmethod
    def fit(cls, training: ModelBatch) -> "MarginalBaseline":
        values = []
        for name in ("pt", "pebs", "pmu"):
            values.extend(_feature_statistics(
                getattr(training, name), getattr(training, f"{name}_available")
            ))
        return cls(*values)

    def score(self, batch: ModelBatch) -> torch.Tensor:
        scores = []
        present = []
        for name in ("pt", "pebs", "pmu"):
            values = getattr(batch, name)
            available = getattr(batch, f"{name}_available")
            mean = getattr(self, f"{name}_mean")
            scale = getattr(self, f"{name}_scale")
            standardized = torch.where(
                available[..., None], (values - mean) / scale,
                torch.zeros_like(values),
            )
            token_error = torch.nn.functional.smooth_l1_loss(
                standardized, torch.zeros_like(standardized), reduction="none"
            ).mean(-1)
            modality_score, modality_present = _top_two_score(token_error, available)
            scores.append(modality_score)
            present.append(modality_present)
        stacked = torch.stack(scores, 1)
        availability = torch.stack(present, 1)
        return torch.where(availability, stacked, torch.zeros_like(stacked)).sum(1) / (
            availability.sum(1).clamp_min(1)
        )

    def state(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PtPcaBaseline:
    """Learned PCA reconstruction over raw-PT histogram tensors only."""

    mean: torch.Tensor
    scale: torch.Tensor
    components: torch.Tensor

    @staticmethod
    def _matrix(batch: ModelBatch) -> tuple[torch.Tensor, torch.Tensor]:
        available = batch.pt_available[..., None].expand_as(batch.pt).flatten(1)
        return batch.pt.flatten(1), available

    @classmethod
    def fit(cls, training: ModelBatch, latent_dimensions: int) -> "PtPcaBaseline":
        if latent_dimensions <= 0:
            raise ValueError("PT PCA latent dimensions must be positive")
        values, available = cls._matrix(training)
        counts = available.sum(0)
        safe_counts = counts.clamp_min(1)
        mean = torch.where(
            counts > 0,
            torch.where(available, values, torch.zeros_like(values)).sum(0) / safe_counts,
            torch.zeros_like(values[0]),
        )
        centered = torch.where(available, values - mean, torch.zeros_like(values))
        variance = centered.square().sum(0) / safe_counts
        scale = torch.where(counts > 0, variance.sqrt().clamp_min(1e-4),
                            torch.ones_like(variance))
        standardized = centered / scale
        _, _, right = torch.linalg.svd(standardized, full_matrices=False)
        dimensions = min(latent_dimensions, right.shape[0])
        return cls(mean, scale, right[:dimensions].contiguous())

    def score(self, batch: ModelBatch) -> torch.Tensor:
        values, available = self._matrix(batch)
        standardized = torch.where(
            available, (values - self.mean) / self.scale, torch.zeros_like(values)
        )
        reconstruction = (standardized @ self.components.T) @ self.components
        error = (standardized - reconstruction).square()
        return torch.where(available, error, torch.zeros_like(error)).sum(1) / (
            available.sum(1).clamp_min(1)
        )

    def state(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _featurize_capture(
    batches: Sequence[HardwareMultimodalBatch],
) -> tuple[ModelBatch, tuple[object, ...]]:
    # Imported lazily so custody and train-only inspection remain usable on a
    # checkout made before the independently developed featurizer is integrated.
    from cpu2tensor.examples.hardware_multimodal_features import (
        featurize_hardware_capture,
    )
    features = featurize_hardware_capture(batches)
    return features.batch, features.lanes


def _validate_capture(
    batches: Sequence[HardwareMultimodalBatch], target_pid: int, target_cpu: int,
) -> dict[str, int]:
    if not batches or any(batch.tid != batch.source for batch in batches):
        raise HardwareCaptureError("capture returned an invalid source identity")
    if target_pid not in {batch.tid for batch in batches}:
        raise HardwareCaptureError("capture omitted the gated target thread")
    counts = {"pt_bytes": 0, "pebs_samples": 0, "pebs_exact_ip": 0,
              "pebs_nonzero_address": 0, "lost_sources": 0,
              "missing_sources": 0, "multiplexed_sources": 0}
    for batch in batches:
        requested = [status for status in batch.status if status.requested]
        counts["lost_sources"] += sum(status.lost for status in requested)
        counts["missing_sources"] += sum(not status.available for status in requested)
        if any(status.lost or not status.available for status in requested):
            raise HardwareCaptureError("requested hardware modality was lost or unavailable")
        for status in requested:
            if status.signal in ("memory_loads", "counters"):
                multiplexed = (
                    status.time_enabled_ns is None or status.time_enabled_ns <= 0 or
                    status.time_running_ns != status.time_enabled_ns
                )
                counts["multiplexed_sources"] += int(multiplexed)
                if multiplexed:
                    raise HardwareCaptureError(
                        f"{status.signal} source was empty or multiplexed"
                    )
        if batch.pt is not None:
            counts["pt_bytes"] += batch.pt.trace_bytes.numel()
        if batch.pebs is not None:
            samples = batch.pebs.ip.numel()
            counts["pebs_samples"] += samples
            counts["pebs_exact_ip"] += int(batch.pebs.exact_ip.sum())
            counts["pebs_nonzero_address"] += int((batch.pebs.address != 0).sum())
            if (samples and (not bool(batch.pebs.exact_ip.all()) or
                             not bool((batch.pebs.address != 0).all()) or
                             set(batch.pebs.cpu.tolist()) != {target_cpu})):
                raise HardwareCaptureError(
                    "PEBS samples must be exact-IP, nonzero-address, and target-CPU attributed"
                )
        if batch.counters is not None:
            multiplexed = batch.counters.time_enabled_ns != batch.counters.time_running_ns
            counts["multiplexed_sources"] += int(multiplexed)
            if multiplexed:
                raise HardwareCaptureError("boundary PMU counter group was multiplexed")
    if counts["pt_bytes"] == 0:
        raise HardwareCaptureError("raw Intel PT capture was empty")
    if counts["pebs_samples"] == 0:
        raise HardwareCaptureError("the pilot requires at least one PEBS sample per execution")
    return counts


def capture_execution(
    binary: Path,
    execution: PlannedExecution,
    *,
    loops: int,
    target_cpu: int,
    data_pages: int,
    aux_pages: int,
    timeout: float,
) -> tuple[tuple[HardwareMultimodalBatch, ...], bytes, int, dict[str, int]]:
    process = subprocess.Popen(
        ("taskset", "-c", str(target_cpu), str(binary), execution.family, str(loops)),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stdout.readline() != b"READY\n":
        process.kill()
        output, error = process.communicate()
        raise RuntimeError(
            f"{execution.execution_id} did not reach READY: stdout={output!r}, stderr={error!r}"
        )
    config = HardwareMultimodalConfig(
        SCOPE,
        process.pid,
        modalities=MODALITIES,
        pebs_period=PEBS_PERIOD,
        data_pages=data_pages,
        aux_pages=aux_pages,
    )
    try:
        with PerfMultimodalCapture(config) as capture:
            started_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            output, error = process.communicate(b"x", timeout=timeout)
            elapsed_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - started_ns
            batches = capture.stop()
    except BaseException:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0 or error:
        raise RuntimeError(
            f"{execution.execution_id} failed: status={process.returncode}, "
            f"stdout={output!r}, stderr={error!r}"
        )
    counts = _validate_capture(batches, process.pid, target_cpu)
    return batches, output, elapsed_ns, counts


def _lane_payload(lane: object) -> dict[str, object]:
    return {
        "tid": int(getattr(lane, "tid")),
        "observed_cpu": getattr(lane, "observed_cpu"),
        "migration_verified": bool(getattr(lane, "migration_verified")),
        "pebs_samples": int(getattr(lane, "pebs_samples")),
    }


def _validate_lanes(lanes: Sequence[object], target_cpu: int) -> None:
    if not lanes:
        raise HardwareCaptureError("featurizer returned no execution lanes")
    for lane in lanes:
        if (not bool(getattr(lane, "migration_verified")) or
                getattr(lane, "observed_cpu") != target_cpu or
                int(getattr(lane, "pebs_samples")) <= 0):
            raise HardwareCaptureError(
                "every admitted lane needs target-CPU PEBS migration evidence"
            )


def collect(args: argparse.Namespace) -> dict[str, object]:
    if platform.system() != "Linux":
        raise RuntimeError("kernel-only perf collection requires Linux")
    binary = args.binary.resolve()
    if not binary.is_file():
        raise ValueError(f"missing workload executable: {binary}")
    allowed_cpus = os.sched_getaffinity(0)
    if args.target_cpu == args.controller_cpu:
        raise ValueError("target and controller CPUs must be distinct")
    if args.target_cpu not in allowed_cpus or args.controller_cpu not in allowed_cpus:
        raise ValueError("target and controller CPUs must be in the allowed affinity set")
    os.sched_setaffinity(0, {args.controller_cpu})

    families = tuple(args.families)
    plan, heldout = make_plan(
        families,
        repetitions=args.repetitions,
        training_rows=args.training_rows,
        calibration_rows=args.calibration_rows,
        heldout_family_count=args.heldout_family_count,
        seed=args.seed,
    )
    identity = subject_manifest(
        binary,
        target_cpu=args.target_cpu,
        controller_cpu=args.controller_cpu,
        data_pages=args.data_pages,
        aux_pages=args.aux_pages,
    )
    artifact = args.artifact.resolve()
    if (artifact / "capture-manifest.json").exists():
        raise ValueError("artifact already contains a sealed capture manifest")

    entries = []
    collection_started = time.perf_counter()
    feature_seconds = 0.0
    total_pt_bytes = 0
    total_pebs_samples = 0
    rejected_capture_attempts = 0
    loss_rejections = 0
    missing_modalities = {"pt": 0, "pebs": 0, "pmu": 0}
    censored_tokens = {"pt": 0, "pebs": 0, "pmu": 0}
    for execution in plan:
        loops = WORKLOAD_LOOPS[execution.family] * args.loop_scale
        for attempt in range(args.capture_retries + 1):
            try:
                batches, output, elapsed_ns, counts = capture_execution(
                    binary, execution, loops=loops, target_cpu=args.target_cpu,
                    data_pages=args.data_pages, aux_pages=args.aux_pages,
                    timeout=args.timeout,
                )
                break
            except HardwareCaptureError as error:
                rejected_capture_attempts += 1
                loss_rejections += int(isinstance(error, HardwareTraceLost))
                if attempt == args.capture_retries:
                    raise
        feature_started = time.perf_counter()
        model_batch, lanes = _featurize_capture(batches)
        feature_seconds += time.perf_counter() - feature_started
        _validate_lanes(lanes, args.target_cpu)
        raw_path = artifact / "raw" / f"{execution.execution_id}.pt"
        raw_hash = _atomic_torch_save(raw_path, raw_capture_payload(
            batches, execution=execution, loops=loops, stdout=output,
            elapsed_ns=elapsed_ns,
        ))
        derived_path = artifact / "derived" / f"{execution.execution_id}.pt"
        derived_hash = _atomic_torch_save(derived_path, {
            "schema": DERIVED_SCHEMA,
            "execution": asdict(execution),
            "raw_sha256": raw_hash,
            "batch": _model_payload(model_batch),
            "lanes": [_lane_payload(lane) for lane in lanes],
        })
        availability = {
            "pt": model_batch.pt_available,
            "pebs": model_batch.pebs_available,
            "pmu": model_batch.pmu_available,
        }
        for name, available in availability.items():
            missing_modalities[name] += int(not bool(available.any()))
            censored_tokens[name] += int((~available).sum())
        total_pt_bytes += counts["pt_bytes"]
        total_pebs_samples += counts["pebs_samples"]
        entries.append({
            **asdict(execution),
            "loops": loops,
            "raw_path": str(raw_path.relative_to(artifact)),
            "raw_sha256": raw_hash,
            "derived_path": str(derived_path.relative_to(artifact)),
            "derived_sha256": derived_hash,
            "stdout_base64": base64.b64encode(output).decode("ascii"),
            "stdout_sha256": hashlib.sha256(output).hexdigest(),
            "elapsed_ns": elapsed_ns,
            "capture": counts,
            "lanes": [_lane_payload(lane) for lane in lanes],
        })
    collection_seconds = time.perf_counter() - collection_started
    manifest = {
        "schema": SCHEMA,
        **identity,
        "split": {
            "seed": args.seed,
            "families": list(families),
            "heldout_families": list(heldout),
            "repetitions": args.repetitions,
            "training_rows_per_familiar_family": args.training_rows,
            "calibration_rows_per_familiar_family": args.calibration_rows,
        },
        "collection": {
            "executions": len(entries),
            "seconds": collection_seconds,
            "executions_per_second": len(entries) / collection_seconds,
            "featurization_seconds": feature_seconds,
            "featurized_executions_per_second": len(entries) / feature_seconds,
            "total_pt_bytes": total_pt_bytes,
            "pt_bytes_per_second": total_pt_bytes / collection_seconds,
            "total_pebs_samples": total_pebs_samples,
            "loss_count": loss_rejections,
            "rejected_capture_attempts": rejected_capture_attempts,
            "missing_modality_executions": missing_modalities,
            "censored_tokens": censored_tokens,
        },
        "entries": entries,
    }
    manifest["manifest_content_sha256"] = _json_hash(manifest)
    _atomic_json(artifact / "capture-manifest.json", manifest)
    return manifest


def load_dataset(artifact: Path) -> tuple[dict[str, object], dict[str, ModelBatch]]:
    manifest_path = artifact / "capture-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported kernel multimodal manifest")
    event = manifest.get("event", {})
    if (event.get("scope") != SCOPE or event.get("pebs_period") != PEBS_PERIOD or
            tuple(event.get("modalities", ())) != MODALITIES or
            event.get("intel_pt_representation") != "raw_aux_bytes_no_decode"):
        raise ValueError("dataset is not the frozen kernel-only event contract")
    expected_content_hash = manifest.pop("manifest_content_sha256", None)
    if expected_content_hash != _json_hash(manifest):
        raise ValueError("capture manifest content hash does not match")
    manifest["manifest_content_sha256"] = expected_content_hash

    rows = {}
    for entry in manifest["entries"]:
        raw_path = artifact / entry["raw_path"]
        derived_path = artifact / entry["derived_path"]
        if _sha256(raw_path) != entry["raw_sha256"]:
            raise ValueError(f"raw custody hash mismatch for {entry['execution_id']}")
        if _sha256(derived_path) != entry["derived_sha256"]:
            raise ValueError(f"derived tensor hash mismatch for {entry['execution_id']}")
        payload = torch.load(derived_path, map_location="cpu", weights_only=True)
        if (payload.get("schema") != DERIVED_SCHEMA or
                payload.get("raw_sha256") != entry["raw_sha256"] or
                payload.get("execution") != {
                    name: entry[name] for name in (
                        "execution_id", "family", "repetition", "partition"
                    )
                }):
            raise ValueError(f"derived tensor identity mismatch for {entry['execution_id']}")
        rows[entry["execution_id"]] = _model_batch(payload["batch"])
    return manifest, rows


def _partition(
    manifest: dict[str, object], rows: dict[str, ModelBatch], name: str,
) -> ModelBatch:
    return _concatenate([
        rows[entry["execution_id"]]
        for entry in manifest["entries"] if entry["partition"] == name
    ])


def _replace(batch: ModelBatch, **changes: torch.Tensor) -> ModelBatch:
    values = {name: getattr(batch, name) for name in (
        "pt", "pebs", "pmu", "pt_available", "pebs_available", "pmu_available",
        "time_bounds", "timing_quality",
    )}
    values.update(changes)
    return ModelBatch(**values)


def modality_swap(batch: ModelBatch, modality: str) -> ModelBatch:
    if batch.batch_size < 2:
        raise ValueError("modality swap needs at least two executions")
    if modality not in ("pt", "pebs", "pmu"):
        raise ValueError("unknown modality")
    values = getattr(batch, modality).roll(1, 0)
    available = getattr(batch, f"{modality}_available").roll(1, 0)
    token_slice = {
        "pt": slice(0, 16),
        "pebs": slice(16, 32),
        "pmu": slice(32, 33),
    }[modality]
    bounds = batch.time_bounds.clone()
    quality = batch.timing_quality.clone()
    bounds[:, :, token_slice] = bounds[:, :, token_slice].roll(1, 0)
    quality[:, :, token_slice] = quality[:, :, token_slice].roll(1, 0)
    return _replace(batch, **{
        modality: values,
        f"{modality}_available": available,
        "time_bounds": bounds,
        "timing_quality": quality,
    })


def timestamp_misalignment(batch: ModelBatch) -> ModelBatch:
    if batch.batch_size < 2:
        raise ValueError("timestamp misalignment needs at least two executions")
    bounds = batch.time_bounds.clone()
    quality = batch.timing_quality.clone()
    # Token order is PT[0:16], PEBS[16:32], PMU[32]. Reverse only timestamps at
    # available PEBS positions within each execution/lane. This preserves sparse
    # masks and finite-time invariants while breaking value/time alignment.
    for row in range(batch.batch_size):
        for cpu in range(batch.cpu_count):
            positions = torch.nonzero(
                batch.pebs_available[row, cpu], as_tuple=False
            ).flatten()
            if positions.numel() < 2:
                continue
            tokens = positions + 16
            reversed_tokens = tokens.flip(0)
            bounds[row, cpu, tokens] = batch.time_bounds[row, cpu, reversed_tokens]
            quality[row, cpu, tokens] = batch.timing_quality[row, cpu, reversed_tokens]
    return _replace(batch, time_bounds=bounds, timing_quality=quality)


def _alerts_per_thousand(scores: torch.Tensor, threshold: float) -> float:
    return float((scores > threshold).sum()) * 1_000.0 / scores.numel()


def _score_summary(scores: torch.Tensor, threshold: float,
                   elapsed_ns: Sequence[int]) -> dict[str, float | int]:
    alerts = int((scores > threshold).sum())
    hours = sum(elapsed_ns) / 3_600_000_000_000
    return {
        "executions": scores.numel(),
        "alerts": alerts,
        "alerts_per_thousand": _alerts_per_thousand(scores, threshold),
        "alerts_per_hour": 0.0 if hours == 0 else alerts / hours,
        "score_mean": float(scores.mean()),
        "score_median": float(scores.median()),
        "score_max": float(scores.max()),
    }


def _score_sensitivity(score, clean: ModelBatch, threshold: float) -> dict[str, object]:
    clean_scores = score(clean)
    result = {
        "clean": {
            "score_mean": float(clean_scores.mean()),
            "alerts_per_thousand": _alerts_per_thousand(clean_scores, threshold),
        }
    }
    corruptions = {
        **{f"{name}_swap": modality_swap(clean, name) for name in ("pt", "pebs", "pmu")},
        "timestamp_misalignment": timestamp_misalignment(clean),
    }
    for name, corrupted in corruptions.items():
        scores = score(corrupted)
        clean_mean = float(clean_scores.mean())
        result[name] = {
            "score_mean": float(scores.mean()),
            "clean_score_mean": clean_mean,
            "mean_delta": float(scores.mean()) - clean_mean,
            "mean_ratio": float(scores.mean()) / max(clean_mean, 1e-12),
            "alerts_per_thousand": _alerts_per_thousand(scores, threshold),
        }
    return result


def _sensitivity(model: MaskedHardwareModel, clean: ModelBatch,
                 threshold: float) -> dict[str, object]:
    return _score_sensitivity(
        lambda batch: multimodal_anomaly_score(model, batch), clean, threshold
    )


def _availability(batch: ModelBatch) -> dict[str, object]:
    result = {}
    for name in ("pt", "pebs", "pmu"):
        available = getattr(batch, f"{name}_available")
        result[name] = {
            "missing_executions": int((~available.flatten(1).any(1)).sum()),
            "censored_tokens": int((~available).sum()),
            "available_tokens": int(available.sum()),
        }
    return result


def _entries_for(manifest: dict[str, object], partition: str,
                 family: str | None = None) -> list[dict[str, object]]:
    return [entry for entry in manifest["entries"]
            if entry["partition"] == partition and
            (family is None or entry["family"] == family)]


def _family_results(
    score,
    threshold: float,
    manifest: dict[str, object],
    rows: dict[str, ModelBatch],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    results = {}
    for family in manifest["split"]["families"]:
        entries = [entry for entry in manifest["entries"]
                   if entry["family"] == family and entry["partition"] in (
                       "familiar_validation", "heldout_family")]
        if not entries:
            continue
        batch = _concatenate([rows[entry["execution_id"]] for entry in entries])
        results[family] = {
            "partition": entries[0]["partition"],
            **_score_summary(score(batch), threshold,
                             [entry["elapsed_ns"] for entry in entries]),
        }
    worst = {}
    for partition, key in (
        ("familiar_validation", "familiar"),
        ("heldout_family", "heldout"),
    ):
        candidates = {
            family: result for family, result in results.items()
            if result["partition"] == partition
        }
        family = max(
            candidates,
            key=lambda candidate: candidates[candidate]["alerts_per_thousand"],
        )
        worst[key] = {"name": family, **candidates[family]}
    overall = max(
        results, key=lambda family: results[family]["alerts_per_thousand"]
    )
    return results, {"name": overall, **results[overall]}, worst


def _baseline_report(
    name: str,
    baseline: MarginalBaseline | PtPcaBaseline,
    *,
    calibration: ModelBatch,
    familiar: ModelBatch,
    heldout: ModelBatch,
    manifest: dict[str, object],
    rows: dict[str, ModelBatch],
    artifact: Path,
    reviews_per_million: float,
    fitting_seconds: float,
    training_executions: int,
) -> dict[str, object]:
    calibration_scores = baseline.score(calibration)
    quantile = 1.0 - reviews_per_million / 1_000_000.0
    threshold = float(torch.quantile(
        calibration_scores, quantile, interpolation="higher"
    ))
    checkpoint = artifact / f"{name}.pt"
    state = baseline.state()
    checkpoint_hash = _atomic_torch_save(checkpoint, {
        "schema": SCHEMA,
        "kind": name,
        "state": state,
        "threshold": threshold,
        "fitting_seconds": fitting_seconds,
        "fitting_executions_per_second": (
            training_executions / fitting_seconds
        ),
    })
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    baseline_type = MarginalBaseline if name == "marginal-baseline" else PtPcaBaseline
    reloaded = baseline_type(**restored["state"])
    deterministic = (
        restored["threshold"] == threshold and
        torch.equal(reloaded.score(calibration), calibration_scores)
    )
    if not deterministic:
        raise RuntimeError(f"{name} checkpoint reload changed frozen inference")
    started = time.perf_counter()
    familiar_scores = baseline.score(familiar)
    heldout_scores = baseline.score(heldout)
    scoring_seconds = time.perf_counter() - started
    family, worst, partition_worst = _family_results(
        baseline.score, threshold, manifest, rows
    )
    evaluation = _concatenate((familiar, heldout))
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_reload_bit_exact": deterministic,
        "threshold": threshold,
        "fitting_seconds": fitting_seconds,
        "fitting_executions_per_second": training_executions / fitting_seconds,
        "familiar": _score_summary(
            familiar_scores, threshold,
            [entry["elapsed_ns"] for entry in _entries_for(
                manifest, "familiar_validation")],
        ),
        "heldout": _score_summary(
            heldout_scores, threshold,
            [entry["elapsed_ns"] for entry in _entries_for(
                manifest, "heldout_family")],
        ),
        "family": family,
        "worst_family": worst,
        "worst_familiar_family": partition_worst["familiar"],
        "worst_heldout_family": partition_worst["heldout"],
        "sensitivity": _score_sensitivity(baseline.score, evaluation, threshold),
        "scoring_seconds": scoring_seconds,
        "scoring_executions_per_second": evaluation.batch_size / scoring_seconds,
    }


def _train_one(
    name: str,
    training: ModelBatch,
    calibration: ModelBatch,
    *,
    args: argparse.Namespace,
    artifact: Path,
    whole_modality_probability: float,
) -> tuple[MaskedHardwareModel, float, dict[str, object]]:
    torch.manual_seed(args.model_seed)
    model = MaskedHardwareModel(MultimodalConfig(
        pebs_features=training.pebs.shape[-1],
        pmu_features=training.pmu.shape[-1],
        model_dimensions=args.model_dimensions,
        attention_heads=args.attention_heads,
        feedforward_dimensions=args.feedforward_dimensions,
        local_layers=args.local_layers,
        cross_cpu_layers=args.cross_cpu_layers,
    )).to(args.device)
    started = time.perf_counter()
    losses = train_masked_model(
        model, training, steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        whole_modality_probability=whole_modality_probability,
        seed=args.model_seed,
    )
    training_seconds = time.perf_counter() - started
    freeze_multimodal_model(model)
    threshold = calibrate_multimodal_threshold(
        model, calibration, reviews_per_million=args.reviews_per_million,
    )
    checkpoint = artifact / f"{name}.pt"
    temporary = checkpoint.with_name(checkpoint.name + ".partial")
    save_frozen_multimodal_model(temporary, model, threshold)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(checkpoint)
    checkpoint_hash = _sha256(checkpoint)
    restored, restored_threshold = load_frozen_multimodal_model(
        checkpoint, device=args.device,
    )
    expected = multimodal_anomaly_score(model, calibration)
    observed = multimodal_anomaly_score(restored, calibration)
    deterministic = restored_threshold == threshold and torch.equal(expected, observed)
    if not deterministic:
        raise RuntimeError(f"{name} checkpoint reload changed frozen inference")
    return model, threshold, {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_reload_bit_exact": deterministic,
        "threshold": threshold,
        "whole_modality_probability": whole_modality_probability,
        "steps": args.steps,
        "training_seconds": training_seconds,
        "training_executions_per_second": (
            args.steps * min(args.batch_size, training.batch_size) / training_seconds
        ),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_minimum": min(losses),
    }


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    artifact = args.artifact.resolve()
    manifest, rows = load_dataset(artifact)
    device = _resolve_device(args.device)
    args.device = device
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    training_cpu = _partition(manifest, rows, "training")
    calibration_cpu = _partition(manifest, rows, "calibration")
    familiar_cpu = _partition(manifest, rows, "familiar_validation")
    heldout_cpu = _partition(manifest, rows, "heldout_family")
    evaluation_cpu = _concatenate((familiar_cpu, heldout_cpu))

    started = time.perf_counter()
    marginal = MarginalBaseline.fit(training_cpu)
    marginal_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pt_pca = PtPcaBaseline.fit(training_cpu, args.pt_pca_dimensions)
    pt_pca_seconds = time.perf_counter() - started
    baselines = {
        "marginal": _baseline_report(
            "marginal-baseline", marginal, calibration=calibration_cpu,
            familiar=familiar_cpu, heldout=heldout_cpu, manifest=manifest,
            rows=rows, artifact=artifact,
            reviews_per_million=args.reviews_per_million,
            fitting_seconds=marginal_seconds,
            training_executions=training_cpu.batch_size,
        ),
        "pt_only_pca": _baseline_report(
            "pt-only-pca-baseline", pt_pca, calibration=calibration_cpu,
            familiar=familiar_cpu, heldout=heldout_cpu, manifest=manifest,
            rows=rows, artifact=artifact,
            reviews_per_million=args.reviews_per_million,
            fitting_seconds=pt_pca_seconds,
            training_executions=training_cpu.batch_size,
        ),
    }

    training = training_cpu.to(device)
    calibration = calibration_cpu.to(device)
    familiar = familiar_cpu.to(device)
    heldout = heldout_cpu.to(device)
    evaluation = evaluation_cpu.to(device)

    report: dict[str, object] = {
        "schema": SCHEMA,
        "dataset_manifest_sha256": _sha256(artifact / "capture-manifest.json"),
        "dataset_identity_sha256": manifest["identity_sha256"],
        "device": str(device),
        "partitions": {
            "training": training.batch_size,
            "calibration": calibration.batch_size,
            "familiar_validation": familiar.batch_size,
            "heldout_family": heldout.batch_size,
        },
        "collection": manifest["collection"],
        "availability": _availability(evaluation_cpu),
        "calibration": {
            "pilot_only": True,
            "executions": calibration.batch_size,
            "reviews_per_million": args.reviews_per_million,
            "warning": (
                "This small pilot calibration cannot support a prospective operational "
                "alert-rate claim; use at least 100,000 independent calibration executions "
                "before claiming 1,000 reviews per million."
            ),
        },
        "baselines": baselines,
        "models": {},
    }
    for name, whole_probability in (("fused", args.whole_modality_probability),
                                    ("span-only", 0.0)):
        model, threshold, model_report = _train_one(
            name, training, calibration, args=args, artifact=artifact,
            whole_modality_probability=whole_probability,
        )
        scoring_started = time.perf_counter()
        familiar_scores = multimodal_anomaly_score(model, familiar)
        heldout_scores = multimodal_anomaly_score(model, heldout)
        scoring_seconds = time.perf_counter() - scoring_started
        model_report["familiar"] = _score_summary(
            familiar_scores, threshold,
            [entry["elapsed_ns"] for entry in _entries_for(
                manifest, "familiar_validation")],
        )
        model_report["heldout"] = _score_summary(
            heldout_scores, threshold,
            [entry["elapsed_ns"] for entry in _entries_for(
                manifest, "heldout_family")],
        )
        family_results = {}
        for family in manifest["split"]["families"]:
            entries = [entry for entry in manifest["entries"]
                       if entry["family"] == family and entry["partition"] in (
                           "familiar_validation", "heldout_family")]
            if not entries:
                continue
            batch = _concatenate([rows[entry["execution_id"]] for entry in entries]).to(device)
            scores = multimodal_anomaly_score(model, batch)
            family_results[family] = {
                "partition": entries[0]["partition"],
                **_score_summary(scores, threshold,
                                 [entry["elapsed_ns"] for entry in entries]),
            }
        worst_family = max(
            family_results,
            key=lambda family: family_results[family]["alerts_per_thousand"],
        )
        model_report["family"] = family_results
        model_report["worst_family"] = {
            "name": worst_family,
            **family_results[worst_family],
        }
        for partition, key in (
            ("familiar_validation", "worst_familiar_family"),
            ("heldout_family", "worst_heldout_family"),
        ):
            candidates = {
                family: result for family, result in family_results.items()
                if result["partition"] == partition
            }
            worst = max(
                candidates,
                key=lambda family: candidates[family]["alerts_per_thousand"],
            )
            model_report[key] = {"name": worst, **candidates[worst]}
        model_report["sensitivity"] = _sensitivity(model, evaluation, threshold)
        model_report["scoring_seconds"] = scoring_seconds
        model_report["scoring_executions_per_second"] = (
            evaluation.batch_size / scoring_seconds
        )
        report["models"][name] = model_report

    report_path = artifact / "report.json"
    _atomic_json(report_path, report)
    return report


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.collect_only and args.train_only:
        raise ValueError("collect-only and train-only are mutually exclusive")
    if (args.loop_scale <= 0 or args.repetitions <= 0 or args.timeout <= 0 or
            args.capture_retries < 0):
        raise ValueError("loop scale, repetitions, and timeout must be positive")
    if args.collect_only:
        return {"collection": collect(args)}
    if args.train_only:
        return {"training": train_and_evaluate(args)}
    collection = collect(args)
    training = train_and_evaluate(args)
    return {"collection": collection, "training": training}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("artifact", type=Path)
    result.add_argument("--binary", type=Path)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--collect-only", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    result.add_argument("--families", nargs="+", choices=tuple(WORKLOAD_LOOPS),
                        default=tuple(WORKLOAD_LOOPS))
    result.add_argument("--target-cpu", type=int, default=2)
    result.add_argument("--controller-cpu", type=int, default=3)
    result.add_argument("--data-pages", type=int, default=64)
    result.add_argument("--aux-pages", type=int, default=8192)
    result.add_argument("--timeout", type=float, default=30.0)
    result.add_argument("--capture-retries", type=int, default=2)
    result.add_argument("--loop-scale", type=int, default=1)
    result.add_argument("--seed", type=int, default=20260925)
    result.add_argument("--repetitions", type=int, default=12)
    result.add_argument("--training-rows", type=int, default=6)
    result.add_argument("--calibration-rows", type=int, default=3)
    result.add_argument("--heldout-family-count", type=int, default=3)
    result.add_argument("--device", default="auto")
    result.add_argument("--cpu-threads", type=int, default=1)
    result.add_argument("--model-seed", type=int, default=41)
    result.add_argument("--steps", type=int, default=100)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-2)
    result.add_argument("--whole-modality-probability", type=float, default=0.3)
    result.add_argument("--reviews-per-million", type=float, default=10_000.0)
    result.add_argument("--model-dimensions", type=int, default=64)
    result.add_argument("--attention-heads", type=int, default=4)
    result.add_argument("--feedforward-dimensions", type=int, default=128)
    result.add_argument("--local-layers", type=int, default=2)
    result.add_argument("--cross-cpu-layers", type=int, default=2)
    result.add_argument("--pt-pca-dimensions", type=int, default=8)
    return result


def main() -> None:
    args = parser().parse_args()
    if not args.train_only and args.binary is None:
        raise SystemExit("--binary is required for collection")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
