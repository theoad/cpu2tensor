# SPDX-License-Identifier: AGPL-3.0-only
"""Collect and shard a curated mixed user/kernel hardware corpus."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import grp
import hashlib
import json
import os
from pathlib import Path
import platform
import pwd
import re
import shutil
import subprocess
import tarfile
import time
from typing import Sequence

import torch

from cpu2tensor.hardware import (
    HardwareBatch,
    HardwareCounterBatch,
    HardwareDecodeSideband,
    HardwareMultimodalBatch,
    HardwareMultimodalConfig,
    HardwareCaptureError,
    PerfMultimodalCapture,
)


PLAN_SCHEMA = "cpu2tensor-hardware-foundation-plan-v1"
RAW_SCHEMA = "cpu2tensor-hardware-foundation-raw-v1"
SHARD_SCHEMA = "cpu2tensor-hardware-foundation-shard-v1"
MANIFEST_SCHEMA = "cpu2tensor-hardware-foundation-corpus-v1"
SCOPE = "process_user_kernel"
MODALITIES = ("intel_pt", "memory_loads", "counters")
PARTITIONS = {
    "training", "calibration", "familiar_validation",
    "heldout_application", "heldout_session",
}
NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PERF_RECORD_FORK = 7
HEADER_SIZE = 8


@dataclass(frozen=True)
class WorkloadRow:
    execution_id: str
    family: str
    application: str
    partition: str
    session: str
    input_seed: int
    argv: tuple[str, ...]
    stdin: bytes
    cwd: str
    input_paths: tuple[str, ...]
    expected_exit_code: int
    expected_stdout_sha256: str
    expected_stderr_sha256: str
    timeout_seconds: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_torch(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return _sha256(path)


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _optional_sha256(path: str) -> str | None:
    try:
        return _sha256(Path(path))
    except OSError:
        return None


def _cpu_model() -> str | None:
    text = _read_text("/proc/cpuinfo")
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def _cpu_identity() -> dict[str, object]:
    text = _read_text("/proc/cpuinfo") or ""
    first: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            break
        if ":" in line:
            key, value = line.split(":", 1)
            first[key.strip()] = value.strip()
    return {
        "vendor": first.get("vendor_id"),
        "family": first.get("cpu family"),
        "model": first.get("model"),
        "stepping": first.get("stepping"),
        "model_name": first.get("model name"),
        "cpuinfo_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _sysfs_snapshot(paths: Sequence[Path]) -> dict[str, str | None]:
    return {str(path): _read_text(str(path)) for path in paths}


def _subject_manifest(
    *, target_cpu: int, controller_cpu: int, target_user: str,
    data_pages: int, aux_pages: int, pebs_period: int, gate: Path,
) -> dict[str, object]:
    runner = Path(__file__).resolve()
    hardware = runner.parents[1] / "hardware.py"
    cpu_root = Path(f"/sys/devices/system/cpu/cpu{target_cpu}")
    cache_paths: list[Path] = []
    for index in sorted((cpu_root / "cache").glob("index*")):
        cache_paths.extend(index / name for name in (
            "level", "type", "size", "coherency_line_size",
            "number_of_sets", "ways_of_associativity", "shared_cpu_list",
        ))
    policy_paths = [
        Path("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        cpu_root / "cpufreq/scaling_driver",
        cpu_root / "cpufreq/scaling_governor",
        cpu_root / "cpufreq/scaling_min_freq",
        cpu_root / "cpufreq/scaling_max_freq",
        cpu_root / "topology/core_id",
        cpu_root / "topology/physical_package_id",
        cpu_root / "topology/thread_siblings_list",
        cpu_root / "topology/core_siblings_list",
    ]
    perf_paths = [
        Path("/sys/bus/event_source/devices/intel_pt/type"),
        Path("/sys/bus/event_source/devices/intel_pt/format/tsc"),
        Path("/sys/bus/event_source/devices/intel_pt/caps/psb_cyc"),
        Path("/sys/bus/event_source/devices/cpu/type"),
        Path("/sys/bus/event_source/devices/cpu/events/mem-loads"),
        Path("/proc/sys/kernel/perf_event_paranoid"),
    ]
    build = {
        "runner": _identity(str(runner)),
        "capture_module": _identity(str(hardware)),
        "gate": _identity(str(gate)),
    }
    subject = {
        "host": platform.node(),
        "machine": platform.machine(),
        "cpu": _cpu_identity(),
        "kernel_release": platform.release(),
        "kernel_version": platform.version(),
        "boot_id": _read_text("/proc/sys/kernel/random/boot_id"),
        "microcode": _read_text("/sys/devices/system/cpu/cpu0/microcode/version"),
        "kernel_btf_sha256": _optional_sha256("/sys/kernel/btf/vmlinux"),
        "kernel_notes_sha256": _optional_sha256("/sys/kernel/notes"),
        "kernel_cmdline_sha256": _optional_sha256("/proc/cmdline"),
        "cache_topology": _sysfs_snapshot(cache_paths),
        "cpu_policy": _sysfs_snapshot(policy_paths),
        "perf_sources": _sysfs_snapshot(perf_paths),
        "scope": SCOPE,
        "modalities": list(MODALITIES),
        "target_cpu": target_cpu,
        "controller_cpu": controller_cpu,
        "target_user": target_user,
        "data_pages": data_pages,
        "aux_pages": aux_pages,
        "pebs_period": pebs_period,
    }
    manifest = {"subject": subject, "build": build}
    manifest["identity_sha256"] = _json_hash(manifest)
    return manifest


def _decode_stdin(value: object) -> bytes:
    if not isinstance(value, str):
        raise ValueError("stdin_base64 must be a string")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ValueError("stdin_base64 is not canonical base64") from error


def load_plan(path: Path) -> tuple[dict[str, object], tuple[WorkloadRow, ...]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != PLAN_SCHEMA:
        raise ValueError("unknown foundation corpus plan schema")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise ValueError("plan needs a nonempty explicit environment")
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in environment.items()):
        raise ValueError("plan environment must contain only string pairs")
    values = payload.get("rows")
    if not isinstance(values, list) or not values:
        raise ValueError("plan needs at least one execution row")
    rows = []
    identities = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("plan rows must be objects")
        names = tuple(value.get(name) for name in (
            "execution_id", "family", "application", "partition", "session",
        ))
        if any(not isinstance(name, str) for name in names):
            raise ValueError("row identities must be strings")
        execution_id, family, application, partition, session = names
        if any(NAME.fullmatch(name) is None
               for name in (execution_id, family, application, session)):
            raise ValueError("row identities contain unsupported characters")
        if execution_id in identities:
            raise ValueError(f"duplicate execution ID: {execution_id}")
        identities.add(execution_id)
        if partition not in PARTITIONS:
            raise ValueError(f"unknown partition: {partition}")
        argv = value.get("argv")
        if (not isinstance(argv, list) or not argv or
                any(not isinstance(item, str) or "\0" in item for item in argv)):
            raise ValueError("argv must be a nonempty string list")
        executable = Path(argv[0])
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError(f"executable is not an absolute file: {argv[0]}")
        cwd = Path(value.get("cwd", "/"))
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ValueError(f"cwd is not an absolute directory: {cwd}")
        input_paths = value.get("input_paths", [])
        if (not isinstance(input_paths, list) or
                any(not isinstance(item, str) for item in input_paths)):
            raise ValueError("input_paths must be a string list")
        for item in input_paths:
            item_path = Path(item)
            if not item_path.is_absolute() or not item_path.is_file():
                raise ValueError(f"input is not an absolute file: {item}")
        expected = tuple(value.get(name) for name in (
            "expected_stdout_sha256", "expected_stderr_sha256",
        ))
        if any(not isinstance(item, str) or SHA256.fullmatch(item) is None
               for item in expected):
            raise ValueError("expected output hashes must be lowercase SHA-256")
        exit_code = value.get("expected_exit_code", 0)
        input_seed = value.get("input_seed")
        timeout = value.get("timeout_seconds", 30.0)
        if (not isinstance(exit_code, int) or not isinstance(input_seed, int) or
                input_seed < 0 or not isinstance(timeout, (int, float)) or
                timeout <= 0):
            raise ValueError("expected exit and timeout are invalid")
        rows.append(WorkloadRow(
            execution_id=execution_id,
            family=family,
            application=application,
            partition=partition,
            session=session,
            input_seed=input_seed,
            argv=tuple(argv),
            stdin=_decode_stdin(value.get("stdin_base64", "")),
            cwd=str(cwd),
            input_paths=tuple(input_paths),
            expected_exit_code=exit_code,
            expected_stdout_sha256=expected[0],
            expected_stderr_sha256=expected[1],
            timeout_seconds=float(timeout),
        ))
    training_apps = {row.application for row in rows if row.partition != "heldout_application"}
    heldout_apps = {row.application for row in rows if row.partition == "heldout_application"}
    overlap = training_apps & heldout_apps
    if overlap:
        raise ValueError(f"held-out applications leak into other splits: {sorted(overlap)}")
    return payload, tuple(rows)


def _hardware_payload(batch: HardwareBatch | None) -> dict[str, object] | None:
    if batch is None:
        return None
    names = (
        "ip", "pid", "tid", "time", "cpu", "period", "address", "weight",
        "data_source", "exact_ip", "trace_bytes",
    )
    return {
        "source": batch.source,
        "signal": batch.signal,
        **{name: getattr(batch, name).cpu() for name in names},
        "perf_records": (
            torch.empty(0, dtype=torch.uint8)
            if batch.perf_records is None else batch.perf_records.cpu()
        ),
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


def _owned_bytes(value: bytes) -> torch.Tensor:
    return torch.tensor(list(value), dtype=torch.uint8)


def _record_kinds(records: torch.Tensor) -> tuple[int, ...]:
    data = bytes(records.tolist())
    kinds = []
    position = 0
    while position < len(data):
        if len(data) - position < HEADER_SIZE:
            raise HardwareCaptureError("perf sideband ended inside a header")
        kind = int.from_bytes(data[position:position + 4], "little")
        size = int.from_bytes(data[position + 6:position + 8], "little")
        if size < HEADER_SIZE or size > len(data) - position:
            raise HardwareCaptureError("perf sideband ended inside a record")
        kinds.append(kind)
        position += size
    return tuple(kinds)


def _validate_capture(
    batches: Sequence[HardwareMultimodalBatch], pid: int, cpu: int,
) -> dict[str, int]:
    if len(batches) != 1 or batches[0].tid != pid:
        raise HardwareCaptureError("capture did not retain exactly the gated target thread")
    batch = batches[0]
    if batch.pt is None or batch.pt.trace_bytes.numel() == 0:
        raise HardwareCaptureError("mixed capture returned no PT bytes")
    if batch.pt.perf_records is None:
        raise HardwareCaptureError("mixed capture omitted PT sideband")
    kinds = _record_kinds(batch.pt.perf_records)
    if PERF_RECORD_FORK in kinds:
        raise HardwareCaptureError("target created an untraced thread or child")
    if batch.counters is None or not batch.counters.available or batch.counters.lost:
        raise HardwareCaptureError("boundary PMU counters are unavailable")
    if batch.counters.time_enabled_ns != batch.counters.time_running_ns:
        raise HardwareCaptureError("boundary PMU counters were multiplexed")
    pebs_samples = 0
    pebs_user = 0
    pebs_kernel = 0
    pebs_inexact = 0
    pebs_zero_address = 0
    pebs_usable = 0
    if batch.pebs is not None:
        pebs_samples = batch.pebs.ip.numel()
        if pebs_samples:
            if (set(batch.pebs.tid.tolist()) != {pid} or
                    set(batch.pebs.cpu.tolist()) != {cpu}):
                raise HardwareCaptureError("PEBS task or CPU attribution is invalid")
            # Keep raw samples exactly as emitted.  Inexact-IP and zero-address
            # rows are explicit missingness for derived views, not a reason to
            # retry into a biased distribution or discard the PT execution.
            pebs_inexact = int((~batch.pebs.exact_ip).sum())
            pebs_zero_address = int((batch.pebs.address == 0).sum())
            pebs_usable = int((
                batch.pebs.exact_ip & (batch.pebs.address != 0)
            ).sum())
            # Kernel canonical addresses have the sign bit set on x86-64.  The
            # tensor stores the same bits as signed int64, so avoid uint64 ops
            # that are not implemented by every PyTorch backend.
            pebs_kernel = int((batch.pebs.ip < 0).sum())
            pebs_user = pebs_samples - pebs_kernel
    requested = [status for status in batch.status if status.requested]
    if not requested or any(not status.available or status.lost for status in requested):
        raise HardwareCaptureError("a requested source is missing or lost")
    return {
        "pt_bytes": batch.pt.trace_bytes.numel(),
        "perf_records": len(kinds),
        "pebs_samples": pebs_samples,
        "pebs_user_samples": pebs_user,
        "pebs_kernel_samples": pebs_kernel,
        "pebs_inexact_samples": pebs_inexact,
        "pebs_zero_address_samples": pebs_zero_address,
        "pebs_usable_samples": pebs_usable,
        "instructions": int(batch.counters.values[0]),
        "cycles": int(batch.counters.values[1]),
        "ref_cycles": int(batch.counters.values[2]),
    }


class TarShardWriter:
    """Own one partial tar and publish only content-addressed closed shards."""

    def __init__(
        self, output: Path, target_bytes: int,
        owner: tuple[int, int] | None = None,
    ) -> None:
        self.output = output
        self.target_bytes = target_bytes
        self.owner = owner
        self.partial = output / "partial"
        self.ready = output / "ready"
        self.partial.mkdir(parents=True, exist_ok=True)
        self.ready.mkdir(parents=True, exist_ok=True)
        if owner is not None:
            for path in (output, self.partial, self.ready):
                os.chown(path, *owner)
        self._sequence = 0
        self._tar: tarfile.TarFile | None = None
        self._path: Path | None = None
        self._index: list[dict[str, object]] = []
        self.published: list[dict[str, object]] = []

    def _open(self) -> None:
        if self._tar is not None:
            return
        self._path = self.partial / f"shard-{self._sequence:05d}.tar.partial"
        self._tar = tarfile.open(self._path, "w", format=tarfile.PAX_FORMAT)
        self._index = []

    def add(self, raw: Path, metadata: dict[str, object]) -> None:
        self._open()
        assert self._tar is not None and self._path is not None
        member = f"{metadata['execution_id']}.pt"
        info = self._tar.gettarinfo(str(raw), arcname=member)
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        offset = self._tar.offset + 512
        with raw.open("rb") as stream:
            self._tar.addfile(info, stream)
        self._index.append({
            **metadata,
            "member": member,
            "offset": offset,
            "size": info.size,
            "sha256": _sha256(raw),
        })
        raw.unlink()
        if self._tar.offset >= self.target_bytes:
            self.close()

    def close(self) -> None:
        if self._tar is None or self._path is None:
            return
        self._tar.close()
        with self._path.open("rb") as stream:
            os.fsync(stream.fileno())
        digest = _sha256(self._path)
        tar_path = self.ready / f"{digest}.tar"
        self._path.replace(tar_path)
        index_path = self.ready / f"{digest}.index.jsonl"
        with index_path.open("w") as stream:
            for row in self._index:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = {
            "schema": SHARD_SCHEMA,
            "sequence": self._sequence,
            "sha256": digest,
            "bytes": tar_path.stat().st_size,
            "records": len(self._index),
            "tar": tar_path.name,
            "index": index_path.name,
            "index_sha256": _sha256(index_path),
        }
        _atomic_json(self.ready / f"{digest}.json", descriptor)
        if self.owner is not None:
            for path in (tar_path, index_path, self.ready / f"{digest}.json"):
                os.chown(path, *self.owner)
        self.published.append(descriptor)
        self._sequence += 1
        self._tar = None
        self._path = None
        self._index = []


def seal_kernel_decode_state(
    artifact: Path, sideband: HardwareDecodeSideband,
    owner: tuple[int, int] | None = None,
) -> dict[str, str]:
    """Seal boot-static state once; execution records keep only a reference."""
    path = artifact / "decode" / f"kernel-{sideband.kernel_state_sha256}.pt"
    if path.exists():
        sha256 = _sha256(path)
    else:
        sha256 = _atomic_torch(path, {
            "schema": "cpu2tensor-kernel-decode-state-v1",
            "kernel_state_sha256": sideband.kernel_state_sha256,
            "kernel_modules": _owned_bytes(sideband.kernel_modules),
            "kernel_symbols": _owned_bytes(sideband.kernel_symbols),
            "module_build_ids_json": _owned_bytes(sideband.module_build_ids_json),
        })
        if owner is not None:
            os.chown(path, *owner)
    return {
        "path": str(path.relative_to(artifact)),
        "sha256": sha256,
        "kernel_state_sha256": sideband.kernel_state_sha256,
    }


def _sideband_payload(
    sideband: HardwareDecodeSideband, kernel_decode_state: dict[str, str],
) -> dict[str, object]:
    if (kernel_decode_state["kernel_state_sha256"] !=
            sideband.kernel_state_sha256):
        raise HardwareCaptureError("kernel decode state changed during collection")
    return {
        "clock": sideband.clock,
        "captured_before_arm_ns": sideband.captured_before_arm_ns,
        "process_maps": _owned_bytes(sideband.process_maps),
        "pt_attribute": _owned_bytes(sideband.pt_attribute),
        "kernel_decode_state": dict(kernel_decode_state),
    }


def _identity(path: str) -> dict[str, object]:
    target = Path(path)
    return {"path": str(target.resolve()), "sha256": _sha256(target), "bytes": target.stat().st_size}


def seal_custody(
    artifact: Path, plan_path: Path, rows: Sequence[WorkloadRow], gate: Path,
    owner: tuple[int, int],
) -> dict[str, object]:
    """Preserve exact small inputs and binaries without duplicating source bytes."""
    root = artifact / "custody"
    objects = root / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    for path in (root, objects):
        os.chown(path, *owner)
    candidates: dict[str, set[str]] = {}
    roles: dict[str, set[str]] = {}
    runner = Path(__file__).resolve()
    hardware = runner.parents[1] / "hardware.py"
    for role, path in (
        ("collector", runner), ("capture_module", hardware), ("gate", gate),
    ):
        resolved = str(path.resolve())
        candidates.setdefault(resolved, set()).add(role)
    for row in rows:
        executable = str(Path(row.argv[0]).resolve())
        candidates.setdefault(executable, set()).add("executable")
        for path in row.input_paths:
            resolved = str(Path(path).resolve())
            candidates.setdefault(resolved, set()).add("input")
    members = []
    for original, member_roles in sorted(candidates.items()):
        source = Path(original)
        digest = _sha256(source)
        destination = objects / digest
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
            os.chown(destination, *owner)
        elif _sha256(destination) != digest:
            raise RuntimeError(f"custody object collision: {digest}")
        roles.setdefault(digest, set()).update(member_roles)
        members.append({
            "original_path": original,
            "sha256": digest,
            "bytes": source.stat().st_size,
            "roles": sorted(member_roles),
        })
    plan_copy = root / "plan.json"
    shutil.copyfile(plan_path, plan_copy)
    os.chown(plan_copy, *owner)
    inventory = root / "packages.txt"
    try:
        packages = subprocess.run(
            ("dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${Architecture}\n"),
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        packages = b""
    inventory.write_bytes(packages)
    os.chown(inventory, *owner)
    manifest = {
        "schema": "cpu2tensor-hardware-foundation-custody-v1",
        "plan": {"path": "plan.json", "sha256": _sha256(plan_copy),
                 "bytes": plan_copy.stat().st_size},
        "package_inventory": {
            "path": "packages.txt", "sha256": _sha256(inventory),
            "bytes": inventory.stat().st_size,
        },
        "members": members,
        "unique_objects": [
            {"path": f"objects/{digest}", "sha256": digest,
             "bytes": (objects / digest).stat().st_size,
             "roles": sorted(member_roles)}
            for digest, member_roles in sorted(roles.items())
        ],
    }
    _atomic_json(root / "manifest.json", manifest)
    os.chown(root / "manifest.json", *owner)
    return manifest


def collect_row(
    row: WorkloadRow, *, gate: Path, environment: dict[str, str],
    target_cpu: int, target_user: str, data_pages: int, aux_pages: int,
    pebs_period: int, artifact: Path, artifact_owner: tuple[int, int],
) -> tuple[dict[str, object], dict[str, int]]:
    account = pwd.getpwnam(target_user)
    groups = os.getgrouplist(target_user, account.pw_gid)
    command = ("taskset", "-c", str(target_cpu), str(gate), *row.argv)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=row.cwd,
        env=environment,
        user=account.pw_uid,
        group=account.pw_gid,
        extra_groups=groups,
    )
    if process.stdout is None or process.stdout.readline() != b"READY\n":
        process.kill()
        output, error = process.communicate()
        raise RuntimeError(f"{row.execution_id} did not reach gate: {output!r} {error!r}")
    if tuple(sorted(os.sched_getaffinity(process.pid))) != (target_cpu,):
        process.kill()
        process.communicate()
        raise HardwareCaptureError("target affinity differs from the exact requested CPU")
    config = HardwareMultimodalConfig(
        SCOPE, process.pid, modalities=MODALITIES,
        pebs_period=pebs_period, data_pages=data_pages, aux_pages=aux_pages,
    )
    try:
        with PerfMultimodalCapture(config) as capture:
            started_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            output, error = process.communicate(b"x" + row.stdin, timeout=row.timeout_seconds)
            elapsed_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - started_ns
            batches = capture.stop()
            sideband = capture.decode_sideband
    except BaseException:
        process.kill()
        process.communicate()
        raise
    stdout_hash = hashlib.sha256(output).hexdigest()
    stderr_hash = hashlib.sha256(error).hexdigest()
    if (process.returncode != row.expected_exit_code or
            stdout_hash != row.expected_stdout_sha256 or
            stderr_hash != row.expected_stderr_sha256):
        raise RuntimeError(
            f"{row.execution_id} output changed: status={process.returncode}, "
            f"stdout={stdout_hash}, stderr={stderr_hash}"
        )
    counts = _validate_capture(batches, process.pid, target_cpu)
    kernel_decode_state = seal_kernel_decode_state(
        artifact, sideband, artifact_owner
    )
    payload = {
        "schema": RAW_SCHEMA,
        "execution": {**asdict(row), "stdin": torch.tensor(list(row.stdin), dtype=torch.uint8)},
        "invocation": {
            "argv": list(row.argv),
            "cwd": row.cwd,
            "environment": dict(sorted(environment.items())),
            "executable": _identity(row.argv[0]),
            "inputs": [_identity(path) for path in row.input_paths],
        },
        "result": {
            "exit_code": process.returncode,
            "stdout_sha256": stdout_hash,
            "stdout_bytes": len(output),
            "stderr_sha256": stderr_hash,
            "stderr_bytes": len(error),
            "elapsed_ns": elapsed_ns,
        },
        "capture": counts,
        "decode_sideband": _sideband_payload(sideband, kernel_decode_state),
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
    return payload, counts


def collect(args: argparse.Namespace) -> dict[str, object]:
    if os.geteuid() != 0:
        raise RuntimeError("mixed user/kernel corpus collection requires root perf access")
    plan, rows = load_plan(args.plan.resolve())
    rows = tuple(row for row in rows if row.session == args.session)
    if not rows:
        raise ValueError(f"plan has no rows for session {args.session}")
    environment = dict(plan["environment"])
    artifact = args.artifact.resolve()
    if artifact.exists() and any(artifact.iterdir()):
        raise ValueError("artifact directory must be new or empty")
    artifact.mkdir(parents=True, exist_ok=True)
    account = pwd.getpwnam(args.target_user)
    artifact_owner = (account.pw_uid, account.pw_gid)
    os.chown(artifact, *artifact_owner)
    subject = _subject_manifest(
        target_cpu=args.target_cpu, controller_cpu=args.controller_cpu,
        target_user=args.target_user, data_pages=args.data_pages,
        aux_pages=args.aux_pages, pebs_period=args.pebs_period,
        gate=args.gate.resolve(),
    )
    custody = seal_custody(
        artifact, args.plan.resolve(), rows, args.gate.resolve(), artifact_owner
    )
    original_affinity = os.sched_getaffinity(0)
    writer = TarShardWriter(
        artifact / "shards", args.shard_bytes, artifact_owner
    )
    entries = []
    rejections: list[dict[str, object]] = []
    try:
        os.sched_setaffinity(0, {args.controller_cpu})
        for row in rows:
            temporary = artifact / "current" / f"{row.execution_id}.pt"
            try:
                payload, counts = collect_row(
                    row, gate=args.gate.resolve(), environment=environment,
                    target_cpu=args.target_cpu, target_user=args.target_user,
                    data_pages=args.data_pages, aux_pages=args.aux_pages,
                    pebs_period=args.pebs_period, artifact=artifact,
                    artifact_owner=artifact_owner,
                )
            except (HardwareCaptureError, RuntimeError, subprocess.TimeoutExpired) as error:
                rejection = {
                    "execution_id": row.execution_id,
                    "family": row.family,
                    "application": row.application,
                    "partition": row.partition,
                    "session": row.session,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                rejections.append(rejection)
                _atomic_json(artifact / "rejections.json", rejections)
                os.chown(artifact / "rejections.json", *artifact_owner)
                if args.allow_rejections:
                    continue
                raise
            raw_hash = _atomic_torch(temporary, payload)
            metadata = {
                "execution_id": row.execution_id,
                "family": row.family,
                "application": row.application,
                "partition": row.partition,
                "session": row.session,
                "raw_sha256": raw_hash,
                **counts,
            }
            writer.add(temporary, metadata)
            entries.append(metadata)
    finally:
        writer.close()
        os.sched_setaffinity(0, original_affinity)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "corpus_id": plan.get("corpus_id"),
        "plan_sha256": _sha256(args.plan.resolve()),
        "plan_content_sha256": _json_hash(plan),
        "subject": subject,
        "custody": {
            "path": "custody/manifest.json",
            "sha256": _sha256(artifact / "custody/manifest.json"),
            "content_sha256": _json_hash(custody),
        },
        "session": args.session,
        "executions": len(entries),
        "rejections": rejections,
        "pt_bytes": sum(int(row["pt_bytes"]) for row in entries),
        "pebs_samples": sum(int(row["pebs_samples"]) for row in entries),
        "entries": entries,
        "shards": writer.published,
        "decode_states": [
            {
                "path": str(path.relative_to(artifact)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted((artifact / "decode").glob("kernel-*.pt"))
        ],
    }
    manifest["manifest_content_sha256"] = _json_hash(manifest)
    _atomic_json(artifact / "capture-manifest.json", manifest)
    os.chown(artifact / "capture-manifest.json", *artifact_owner)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("artifact", type=Path)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--session", required=True)
    result.add_argument("--gate", type=Path, required=True)
    result.add_argument("--target-user", default="user")
    result.add_argument("--target-cpu", type=int, default=2)
    result.add_argument("--controller-cpu", type=int, default=3)
    result.add_argument("--pebs-period", type=int, default=10_000)
    result.add_argument("--data-pages", type=int, default=1024)
    result.add_argument("--aux-pages", type=int, default=8192)
    result.add_argument("--shard-bytes", type=int, default=256 * 1024 * 1024)
    result.add_argument("--allow-rejections", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.target_cpu == args.controller_cpu:
        raise SystemExit("target and controller CPUs must differ")
    if args.shard_bytes < 1024 * 1024:
        raise SystemExit("shards smaller than one MiB are unsupported")
    try:
        print(json.dumps(collect(args), sort_keys=True))
    except BaseException as error:
        artifact = args.artifact.resolve()
        if artifact.exists():
            _atomic_json(artifact / "capture-failed.json", {
                "schema": "cpu2tensor-hardware-foundation-failure-v1",
                "error_type": type(error).__name__,
                "error": str(error),
            })
        raise


if __name__ == "__main__":
    main()
