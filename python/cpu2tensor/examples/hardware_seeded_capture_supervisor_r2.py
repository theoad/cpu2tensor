# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed supervisor for one approved, bounded 51-row hardware smoke.

This module does not mount tmpfs. A separately approved, persistent 896 MiB
tmpfs mount and an independent systemd ExecStopPost restore are prerequisites.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import pwd
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

import torch

from cpu2tensor.examples import hardware_multimodal_experiment as collector
from cpu2tensor.examples import hardware_seeded_capture_plan_r2 as plans


GIB = 1024 ** 3
MIB = 1024 ** 2
SPOOL_CAP = 896 * MIB
MIN_START_FREE = 11 * GIB // 2
MIN_RUNNING_FREE = 5 * GIB
MAX_SECONDS = 1200
STOP_MILLIC = 80_000
START_MILLIC = 70_000
NO_TURBO = Path("/sys/devices/system/cpu/intel_pstate/no_turbo")
MAX_FREQ = Path("/sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq")
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
EXPECTED_BOOT = "31179aea-9a43-4c5d-8bf6-1205735e42c4"
EXPECTED_KERNEL = "5.13.0-30-generic"
CAPTURE_FREQ_KHZ = "1800000"
POLICY_STATE = Path("/run/cpu2tensor-seeded-r2-policy.json")
SERVICE_UNIT = "cpu2tensor-seeded-smoke-r2.service"
RESTORE_TIMER = "cpu2tensor-seeded-restore-r2.timer"
CPU2_PACKAGE = Path("/sys/devices/system/cpu/cpu2/topology/physical_package_id")
MEMINFO = Path("/proc/meminfo")
CGROUP = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROCESS_STATUS = Path("/proc/self/status")
MIN_START_MEM = 4 * GIB
MIN_RUNNING_MEM = 3 * GIB
MIN_START_CGROUP = 2 * GIB
MIN_RUNNING_CGROUP = GIB


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_manifest(repo: Path) -> dict[str, object]:
    """Bind a portable source-only deployment when Git metadata is absent."""
    files: dict[str, str] = {}
    for relative_root in ("python", "native"):
        root = repo / relative_root
        if not root.is_dir():
            raise ValueError("source bundle is incomplete")
        for path in root.rglob("*"):
            if "__pycache__" in path.parts or any(part.startswith(".") for part in path.relative_to(repo).parts):
                continue
            if path.is_symlink():
                raise ValueError("source bundle may not contain symlinks")
            if path.is_file() and (path.suffix in (
                    ".py", ".pyi", ".pyx", ".pxd", ".c", ".h", ".cc",
                    ".cpp", ".cxx", ".hpp",
            ) or
                                   path.name == "CMakeLists.txt"):
                files[path.relative_to(repo).as_posix()] = sha256(path)
    for name in ("pyproject.toml", "AGENTS.md"):
        path = repo / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"source bundle lacks {name}")
        files[name] = sha256(path)
    content = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "cpu2tensor-seeded-source-bundle-r2",
        "files": files,
        "digest_sha256": hashlib.sha256(content).hexdigest(),
    }


def _read(path: Path) -> str:
    return path.read_text().strip()


def _write(path: Path, value: str) -> None:
    path.write_text(value + "\n")


def package_temperature_millic(path: Path) -> int:
    match = re.fullmatch(r"temp([0-9]+)_input", path.name)
    if match is None or _read(path.parent / "name") != "coretemp":
        raise ValueError("temperature input is not a coretemp package sensor")
    if _read(path.parent / f"temp{match.group(1)}_label") != f"Package id {_read(CPU2_PACKAGE)}":
        raise ValueError("temperature input is not a package sensor")
    value = int(_read(path))
    if not 0 <= value <= 125_000:
        raise ValueError("invalid package temperature reading")
    return value


def _memory_headroom() -> tuple[int, int | None]:
    match = re.search(r"^MemAvailable:\s+(\d+) kB$", MEMINFO.read_text(), re.MULTILINE)
    if match is None:
        raise ValueError("MemAvailable is unavailable")
    available = int(match.group(1)) * 1024
    cgroups = [line.split(":", 2) for line in CGROUP.read_text().splitlines()]
    unified = [fields[2] for fields in cgroups if len(fields) == 3 and fields[:2] == ["0", ""]]
    legacy = [fields[2] for fields in cgroups if len(fields) == 3 and
              "memory" in fields[1].split(",")]
    if len(unified) == 1 and ".." not in Path(unified[0]).parts:
        directory = CGROUP_ROOT / unified[0].lstrip("/")
        limit, used = _read(directory / "memory.max"), int(_read(directory / "memory.current"))
    elif len(legacy) == 1 and ".." not in Path(legacy[0]).parts:
        directory = CGROUP_ROOT / "memory" / legacy[0].lstrip("/")
        limit, used = _read(directory / "memory.limit_in_bytes"), int(
            _read(directory / "memory.usage_in_bytes"))
    else:
        raise ValueError("cgroup memory accounting is unavailable")
    if limit == "max":
        return available, None
    if limit.isdecimal() and int(limit) >= 1 << 60:
        return available, None
    if not limit.isdecimal() or int(limit) < used:
        raise ValueError("invalid cgroup memory accounting")
    return available, int(limit) - used


def _check_memory(start: bool) -> None:
    available, cgroup = _memory_headroom()
    if available < (MIN_START_MEM if start else MIN_RUNNING_MEM):
        raise RuntimeError("MemAvailable fell below bounded capture reserve")
    if cgroup is not None and cgroup < (MIN_START_CGROUP if start else MIN_RUNNING_CGROUP):
        raise RuntimeError("cgroup memory headroom fell below bounded capture reserve")


def _require_service_context() -> None:
    if not os.environ.get("INVOCATION_ID"):
        raise ValueError("capture must run under the reviewed systemd unit")
    if not any(SERVICE_UNIT in line for line in CGROUP.read_text().splitlines()):
        raise ValueError("capture is outside the reviewed systemd cgroup")
    result = subprocess.run(
        ("systemctl", "is-active", "--quiet", RESTORE_TIMER), check=False,
        capture_output=True, timeout=5,
    )
    if result.returncode != 0:
        raise ValueError("independent restore timer is inactive")
    properties = subprocess.run(
        ("systemctl", "show", SERVICE_UNIT, "--property=ExecStopPost,RuntimeMaxUSec,KillMode,UMask,ReadOnlyPaths"),
        check=True, capture_output=True, text=True, timeout=5,
    ).stdout
    values = dict(line.split("=", 1) for line in properties.splitlines() if "=" in line)
    runtime = values.get("RuntimeMaxUSec", "")
    runtime_us = (int(runtime) if runtime.isdecimal() else
                  MAX_SECONDS * 1_000_000 if runtime in ("20min", "1200s") else 0)
    read_only = values.get("ReadOnlyPaths", "")
    if ("hardware_seeded_capture_supervisor_r2 restore" not in values.get("ExecStopPost", "") or
            values.get("KillMode") != "control-group" or
            values.get("UMask") not in ("0027", "27") or
            "/home/user/.cache/cpu2tensor/seeded-r2-source-20260926" not in read_only or
            "/home/user/.cache/cpu2tensor/seeded-r2-build-20260926" not in read_only or
            not 0 < runtime_us <= MAX_SECONDS * 1_000_000):
        raise ValueError("systemd hard stop or independent restore hook is missing")


def _check_spool_access(spool: Path) -> None:
    """Ensure the unprivileged Mac transfer user can read, never write, evidence."""
    user_gid = pwd.getpwnam("user").pw_gid
    root = spool.stat()
    if (root.st_uid != 0 or root.st_gid != user_gid or
            os.getegid() != user_gid or stat.S_IMODE(root.st_mode) != 0o750):
        raise ValueError("tmpfs ownership/mode must be root:user 0750")
    match = re.search(r"^Umask:\s+([0-7]{4})$", PROCESS_STATUS.read_text(), re.MULTILINE)
    if match is None or match.group(1) != "0027":
        raise ValueError("service umask must be 0027")
    for flag in ("-r", "-x"):
        result = subprocess.run(
            ("/usr/sbin/runuser", "-u", "user", "--", "/usr/bin/test", flag, str(spool)),
            check=False, capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            raise ValueError("unprivileged transfer user cannot read tmpfs")


def _verify_mac_readability(
    spool: Path, *, owner_uid: int = 0, user_gid: int | None = None,
) -> None:
    if user_gid is None:
        user_gid = pwd.getpwnam("user").pw_gid
    for path in spool.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("spool contains a symlink")
        metadata = path.stat()
        mode = stat.S_IMODE(metadata.st_mode)
        if (metadata.st_uid != owner_uid or metadata.st_gid != user_gid or
                mode & 0o027 or
                (path.is_dir() and (mode & 0o750) != 0o750) or
                (path.is_file() and mode & 0o040 == 0) or
                not (path.is_dir() or path.is_file())):
            raise RuntimeError("sealed spool is not read-only transferable by user")


def _verify_frozen_sources(args: argparse.Namespace, repo: Path) -> None:
    for path, expected in (
        (args.binary, args.binary_sha256),
        (repo / "python/cpu2tensor/examples/hardware_multimodal_experiment.py", args.runner_sha256),
        (repo / "python/cpu2tensor/examples/hardware_seeded_capture_plan_r2.py", args.planner_sha256),
        (Path(__file__).resolve(), args.supervisor_sha256),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("binary or source changed")
    if source_bundle_manifest(repo)["digest_sha256"] != args.source_bundle_sha256:
        raise ValueError("portable source bundle changed")


def _mount_fields(spool: Path) -> tuple[str, str, str]:
    result = subprocess.run(
        ("findmnt", "-n", "-o", "SOURCE,FSTYPE,TARGET", "--target", str(spool)),
        check=True, capture_output=True, text=True,
    )
    fields = result.stdout.strip().split()
    if len(fields) != 3:
        raise ValueError("ambiguous spool mount")
    return fields[0], fields[1], fields[2]


def _check_child_resolution(repo: Path) -> None:
    """Check the root child interpreter, not just this supervisor's imports."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo / "python")
    code = (
        "import inspect, json, cpu2tensor, torch; "
        "from cpu2tensor.examples import hardware_multimodal_experiment as runner; "
        "from cpu2tensor.examples import hardware_seeded_capture_plan_r2 as planner; "
        "from cpu2tensor.examples import hardware_seeded_capture_supervisor_r2 as guard; "
        "print(json.dumps({'package': cpu2tensor.__file__, 'torch': torch.__file__, "
        "'weights_only': 'weights_only' in inspect.signature(torch.load).parameters, "
        "'runner': runner.__file__, 'planner': planner.__file__, 'guard': guard.__file__}))"
    )
    result = subprocess.run(
        (sys.executable, "-c", code), cwd=repo, env=environment,
        check=True, capture_output=True, text=True, timeout=15,
    )
    paths = json.loads(result.stdout)
    expected = {
        "package": repo / "python/cpu2tensor/__init__.py",
        "runner": repo / "python/cpu2tensor/examples/hardware_multimodal_experiment.py",
        "planner": repo / "python/cpu2tensor/examples/hardware_seeded_capture_plan_r2.py",
        "guard": repo / "python/cpu2tensor/examples/hardware_seeded_capture_supervisor_r2.py",
    }
    if any(Path(paths.get(name, "")).resolve() != path for name, path in expected.items()):
        raise ValueError("root child Python resolves another cpu2tensor checkout")
    torch_path = paths.get("torch")
    if (not isinstance(torch_path, str) or not Path(torch_path).is_file() or
            paths.get("weights_only") is not True):
        raise ValueError("root child Python lacks the required torch loader")
    print(f"root child Python: {sys.executable}; torch: {torch_path}; cpu2tensor: {paths['package']}")


def preflight(args: argparse.Namespace) -> tuple[str, str]:
    """Read-only checks before the policy state file or sysfs is changed."""
    if type(args.cohort_seed) is not int or not 0 <= args.cohort_seed < 1 << 64:
        raise ValueError("cohort seed must be an unsigned 64-bit integer")
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise ValueError("approved Linux root service is required")
    _require_service_context()
    if _read(BOOT_ID) != EXPECTED_BOOT or platform.release() != EXPECTED_KERNEL:
        raise ValueError("boot or kernel changed")
    repo = Path(__file__).resolve().parents[3]
    if repo != args.source_root.resolve():
        raise ValueError("supervisor source checkout differs from reviewed path")
    runner = repo / "python/cpu2tensor/examples/hardware_multimodal_experiment.py"
    planner = repo / "python/cpu2tensor/examples/hardware_seeded_capture_plan_r2.py"
    if Path(collector.__file__).resolve() != runner or Path(plans.__file__).resolve() != planner:
        raise ValueError("Python resolved another checkout")
    _verify_frozen_sources(args, repo)
    _check_child_resolution(repo)
    spool = args.spool
    if spool.is_symlink() or not spool.is_dir() or not os.path.ismount(spool):
        raise ValueError("spool is not a dedicated mounted directory")
    source, filesystem, target = _mount_fields(spool)
    if source != "tmpfs" or filesystem != "tmpfs" or Path(target) != spool.resolve():
        raise ValueError("spool must be the exact dedicated tmpfs mount")
    filesystem_size = os.statvfs(spool).f_blocks * os.statvfs(spool).f_frsize
    if filesystem_size != SPOOL_CAP:
        raise ValueError("tmpfs must be exactly 896 MiB for the bounded pilot")
    _check_spool_access(spool)
    if any(spool.iterdir()):
        raise ValueError("smoke requires an empty dedicated tmpfs")
    if shutil.disk_usage("/").free < MIN_START_FREE:
        raise ValueError("root filesystem has less than 5.5 GiB free")
    _check_memory(start=True)
    temperature = package_temperature_millic(args.temperature_input)
    if temperature >= START_MILLIC:
        raise ValueError("package is too hot to start")
    no_turbo, maximum = _read(NO_TURBO), _read(MAX_FREQ)
    if no_turbo != "0" or not maximum.isdecimal():
        raise ValueError("unexpected pre-run CPU frequency policy")
    if POLICY_STATE.exists():
        raise ValueError("another policy restore state already exists")
    return no_turbo, maximum


def _seal_policy_state(path: Path, no_turbo: str, maximum: str) -> None:
    payload = json.dumps({"no_turbo": no_turbo, "maximum_khz": maximum}) + "\n"
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    # Publish only complete restore state before touching sysfs. Link is
    # fail-closed when another active service already owns the state path.
    os.link(temporary, path)
    temporary.unlink()


def restore_policy(path: Path) -> None:
    """Idempotent ExecStopPost entry point; leave state on a failed restore."""
    if not path.exists():
        return
    state = json.loads(path.read_text())
    if (type(state) is not dict or state.get("no_turbo") not in ("0", "1") or
            not isinstance(state.get("maximum_khz"), str) or
            not state["maximum_khz"].isdecimal()):
        raise ValueError("invalid saved policy state")
    _write(NO_TURBO, state["no_turbo"])
    _write(MAX_FREQ, state["maximum_khz"])
    if _read(MAX_FREQ) != state["maximum_khz"] or _read(NO_TURBO) != state["no_turbo"]:
        raise RuntimeError("CPU policy restore did not stick")
    path.unlink()


def restore_if_inactive() -> None:
    """Independent timer action; never restore during an active smoke/stop."""
    if not POLICY_STATE.exists():
        return
    state = subprocess.run(
        ("systemctl", "show", SERVICE_UNIT, "--property=ActiveState", "--value"),
        check=True, capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    if state in ("inactive", "failed"):
        restore_policy(POLICY_STATE)
    elif state not in ("active", "activating", "deactivating"):
        raise RuntimeError("unknown smoke service state; preserve restore state")


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The parent may have exited while a descendant still owns the process group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _interrupted(signum: int, unused_frame: object) -> None:
    raise InterruptedError(f"supervisor received signal {signum}")


def _verified_sensor_counts(
    raw: dict[str, object], derived: dict[str, object], *,
    pebs_signal: str = "memory_loads",
) -> dict[str, int]:
    """Recompute admitted sensor status from sealed payloads, not manifest claims."""
    counts = dict.fromkeys((
        "pt_bytes", "pebs_samples", "pebs_usable_samples", "pebs_censored_samples",
        "pebs_exact_ip", "pebs_nonzero_address", "lost_sources", "missing_sources",
        "multiplexed_sources",
    ), 0)
    batches = raw.get("batches")
    if not isinstance(batches, list) or not batches:
        raise RuntimeError("raw sensor batches are missing")
    for batch in batches:
        if batch.get("source") != batch.get("tid"):
            raise RuntimeError("raw sensor source identity is invalid")
        statuses = batch.get("status")
        if (not isinstance(statuses, list) or len(statuses) != 3 or
                {status.get("signal") for status in statuses} !=
                {"intel_pt", pebs_signal, "counters"}):
            raise RuntimeError("raw sensor status contract is incomplete")
        for status in statuses:
            if status.get("requested") is not True or status.get("available") is not True or status.get("lost"):
                raise RuntimeError("raw sensor reports loss or missing modality")
            if status["signal"] != "intel_pt" and (
                    type(status.get("time_enabled_ns")) is not int or
                    status["time_enabled_ns"] <= 0 or
                    status.get("time_running_ns") != status["time_enabled_ns"]):
                raise RuntimeError("raw sensor reports multiplexing")
        pt, pebs, pmu = (batch.get(name) for name in ("pt", "pebs", "counters"))
        if not all(isinstance(item, dict) for item in (pt, pebs, pmu)):
            raise RuntimeError("raw sensor payload is missing")
        if pt.get("signal") != "intel_pt" or pebs.get("signal") != pebs_signal:
            raise RuntimeError("raw sensor signal differs from admission")
        trace = pt.get("trace_bytes")
        ip, address, exact, cpu = (pebs.get(name) for name in ("ip", "address", "exact_ip", "cpu"))
        if (not all(isinstance(item, torch.Tensor) for item in (trace, ip, address, exact, cpu)) or
                not all(item.numel() == ip.numel() for item in (address, exact, cpu)) or
                (cpu.numel() and set(cpu.tolist()) != {2})):
            raise RuntimeError("raw PT/PEBS sample payload is invalid")
        if (pmu.get("available") is not True or pmu.get("lost") or
                type(pmu.get("time_enabled_ns")) is not int or
                pmu["time_enabled_ns"] <= 0 or
                pmu.get("time_running_ns") != pmu["time_enabled_ns"]):
            raise RuntimeError("raw PMU counter group is unavailable or multiplexed")
        usable = exact.bool() & (address != 0)
        counts["pt_bytes"] += trace.numel()
        counts["pebs_samples"] += ip.numel()
        counts["pebs_exact_ip"] += int(exact.bool().sum())
        counts["pebs_nonzero_address"] += int((address != 0).sum())
        counts["pebs_usable_samples"] += int(usable.sum())
        counts["pebs_censored_samples"] += ip.numel() - int(usable.sum())
    if counts["pt_bytes"] <= 0:
        raise RuntimeError("raw PT stream is empty")
    features = derived.get("batch")
    if not isinstance(features, dict):
        raise RuntimeError("derived sensor features are missing")
    for modality in ("pt", "pebs", "pmu"):
        availability = features.get(f"{modality}_available")
        if not isinstance(availability, torch.Tensor) or not bool(availability.any()):
            raise RuntimeError("derived sensor modality is unavailable")
    return counts


def verify_smoke_artifact(
    artifact: Path, identity_sha256: str, plan: dict[str, object],
) -> None:
    """Locally verify sealed files; Mac acknowledgement remains independent."""
    manifest = json.loads((artifact / "capture-manifest.json").read_text())
    content_hash = manifest.pop("manifest_content_sha256", None)
    if content_hash != collector._json_hash(manifest):
        raise RuntimeError("smoke manifest content hash mismatch")
    if manifest.get("identity_sha256") != identity_sha256:
        raise RuntimeError("smoke subject identity changed")
    split = manifest.get("split", {}).get("explicit_plan", {})
    sealed_plan_path = artifact / "execution-plan.json"
    if (split.get("kind") != "smoke" or
            split.get("sha256") != sha256(sealed_plan_path) or
            json.loads(sealed_plan_path.read_text()) != plan):
        raise RuntimeError("sealed smoke plan differs from preregistration")
    entries = manifest.get("entries", [])
    planned_ids = {row["execution_id"] for row in plan["rows"]}
    if (len(entries) != 51 or
            {row.get("execution_id") for row in entries} != planned_ids or
            sum(bool(row.get("raw_retained")) for row in entries) != 51 or
            any(row.get("admission", {}).get("attempt") != 1 for row in entries)):
        raise RuntimeError("smoke manifest failed 51/51 first-attempt gate")
    collection = manifest.get("collection", {})
    if (collection.get("loss_count") != 0 or
            collection.get("admission", {}).get("rejected_attempts") != 0 or
            collection.get("missing_modality_executions") !=
            {"pt": 0, "pebs": 0, "pmu": 0}):
        raise RuntimeError("smoke manifest records sensor loss or missing source")
    total_counts = {"pt_bytes": 0, "pebs_samples": 0,
                    "pebs_usable_samples": 0, "pebs_censored_samples": 0}
    planned_by_id = {row["execution_id"]: row for row in plan["rows"]}
    binary = manifest.get("build", {}).get("workload_path")
    if not isinstance(binary, str):
        raise RuntimeError("smoke manifest lacks workload path")
    for entry in entries:
        planned = planned_by_id[entry["execution_id"]]
        execution = {key: planned[key] for key in (
            "execution_id", "family", "repetition", "partition",
        )}
        if (any(entry.get(key) != value for key, value in execution.items()) or
                entry.get("loops") != planned["loops"]):
            raise RuntimeError("smoke row differs from exact execution plan")
        frame = collector.input_frame(planned["input_seed"])
        frame_hash = hashlib.sha256(frame).hexdigest()
        argv = [binary, planned["family"], str(planned["loops"])]
        invocation = entry.get("invocation", {})
        if (invocation.get("input_seed") != planned["input_seed"] or
                invocation.get("stdin_sha256") != frame_hash or
                invocation.get("argv") != argv):
            raise RuntimeError("smoke invocation differs from exact seed frame")
        for folder in ("raw", "derived"):
            relative = Path(entry[f"{folder}_path"])
            if (relative.is_absolute() or not relative.parts or
                    relative.parts[0] != folder or ".." in relative.parts or
                    not (artifact / relative).resolve().is_relative_to(artifact.resolve())):
                raise RuntimeError("unsafe smoke artifact path")
            if sha256(artifact / relative) != entry[f"{folder}_sha256"]:
                raise RuntimeError("smoke artifact hash mismatch")
        raw = torch.load(artifact / entry["raw_path"], map_location="cpu", weights_only=True)
        raw_invocation = raw.get("invocation", {})
        if (raw.get("schema") != collector.RAW_SCHEMA or
                raw.get("execution") != execution or
                raw.get("loops") != planned["loops"] or
                raw_invocation.get("argv") != argv or
                raw_invocation.get("input_seed") != planned["input_seed"] or
                raw_invocation.get("stdin_sha256") != frame_hash or
                bytes(raw_invocation["stdin"].tolist()) != frame):
            raise RuntimeError("raw smoke payload differs from exact seed frame")
        stdout = bytes(raw["stdout"].tolist())
        if (hashlib.sha256(stdout).hexdigest() != entry.get("stdout_sha256") or
                base64.b64encode(stdout).decode("ascii") != entry.get("stdout_base64")):
            raise RuntimeError("raw smoke output differs from the oracle custody")
        derived = torch.load(
            artifact / entry["derived_path"], map_location="cpu", weights_only=True,
        )
        if (derived.get("schema") != collector.DERIVED_SCHEMA or
                derived.get("execution") != execution or
                derived.get("raw_sha256") != entry["raw_sha256"]):
            raise RuntimeError("derived smoke payload differs from raw custody")
        counts = _verified_sensor_counts(raw, derived)
        if entry.get("capture") != counts:
            raise RuntimeError("manifest sensor counts differ from raw payload")
        for name in total_counts:
            total_counts[name] += counts[name]
    for key, value in (
        ("total_pt_bytes", total_counts["pt_bytes"]),
        ("total_pebs_samples", total_counts["pebs_samples"]),
        ("total_pebs_usable_samples", total_counts["pebs_usable_samples"]),
        ("total_pebs_censored_samples", total_counts["pebs_censored_samples"]),
    ):
        if collection.get(key) != value:
            raise RuntimeError("manifest sensor aggregate differs from raw payload")
    if (collection.get("executions") != 51 or
            collection.get("admission", {}).get("total_attempts") != 51):
        raise RuntimeError("manifest admission totals differ from sealed rows")


def run_smoke(args: argparse.Namespace) -> None:
    original_no_turbo, original_maximum = preflight(args)
    process: subprocess.Popen[bytes] | None = None
    state_sealed = False
    capture_sealed = False
    old_term = signal.signal(signal.SIGTERM, _interrupted)
    old_int = signal.signal(signal.SIGINT, _interrupted)
    try:
        _seal_policy_state(POLICY_STATE, original_no_turbo, original_maximum)
        state_sealed = True
        _write(NO_TURBO, "1")
        _write(MAX_FREQ, CAPTURE_FREQ_KHZ)
        if _read(NO_TURBO) != "1" or _read(MAX_FREQ) != CAPTURE_FREQ_KHZ:
            raise RuntimeError("capture CPU policy did not stick")
        if os.environ.get("C2T_SEEDED_REHEARSAL") == "1":
            # Same service/cgroup/preflight/policy path; deliberately no perf,
            # subject invocation, spool write, or perf collection.
            process = subprocess.Popen(("/bin/sleep", "30"), start_new_session=True)
            process.wait(timeout=35)
            return
        repo = Path(__file__).resolve().parents[3]
        bundle = source_bundle_manifest(repo)
        if bundle["digest_sha256"] != args.source_bundle_sha256:
            raise RuntimeError("source bundle changed after preflight")
        with (args.spool / "source-bundle.json").open("x") as stream:
            json.dump(bundle, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        identity = collector.subject_manifest(
            args.binary, target_cpu=2, controller_cpu=3,
            data_pages=1024, aux_pages=8192,
            pebs_signal="memory_loads", pebs_period=10_000,
        )
        plan = plans.make_plan("smoke", identity["identity_sha256"], args.cohort_seed)
        plan_path = args.spool / "smoke-plan.json"
        with plan_path.open("x") as stream:
            json.dump(plan, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        artifact = args.spool / "smoke"
        command = (
            sys.executable, "-m", "cpu2tensor.examples.hardware_multimodal_experiment",
            str(artifact), "--binary", str(args.binary), "--collect-only",
            "--execution-plan", str(plan_path), "--capture-retries", "0",
            "--target-cpu", "2", "--controller-cpu", "3",
            "--data-pages", "1024", "--aux-pages", "8192",
            "--pebs-signal", "memory_loads", "--pebs-period", "10000",
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repo / "python")
        if package_temperature_millic(args.temperature_input) >= START_MILLIC:
            raise RuntimeError("package heated above the start threshold")
        if shutil.disk_usage("/").free < MIN_START_FREE:
            raise RuntimeError("root free space fell below the start threshold")
        _check_memory(start=True)
        started = time.monotonic()
        with (args.spool / "collector.log").open("xb") as log:
            process = subprocess.Popen(
                command, cwd=repo, env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            while process.poll() is None:
                if time.monotonic() - started >= MAX_SECONDS:
                    raise TimeoutError("smoke exceeded 20 minutes")
                if package_temperature_millic(args.temperature_input) >= STOP_MILLIC:
                    raise RuntimeError("package reached 80 C stop threshold")
                if shutil.disk_usage("/").free < MIN_RUNNING_FREE:
                    raise RuntimeError("root filesystem fell below 5 GiB free")
                _check_memory(start=False)
                if os.statvfs(args.spool).f_bavail * os.statvfs(args.spool).f_frsize < 128 * MIB:
                    raise RuntimeError("tmpfs headroom fell below 128 MiB")
                if _read(NO_TURBO) != "1" or _read(MAX_FREQ) != CAPTURE_FREQ_KHZ:
                    raise RuntimeError("CPU frequency policy changed during capture")
                time.sleep(1)
            if process.wait() != 0:
                raise RuntimeError("collector failed; preserve tmpfs for diagnosis")
        _verify_frozen_sources(args, repo)
        verify_smoke_artifact(artifact, identity["identity_sha256"], plan)
        _verify_mac_readability(args.spool)
        capture_sealed = True
    finally:
        # A second TERM must not interrupt restoration; ExecStopPost remains a
        # fallback for SIGKILL or an unexpected interpreter failure.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if process is not None:
                terminate_group(process)
        finally:
            try:
                if state_sealed:
                    restore_policy(POLICY_STATE)
            finally:
                signal.signal(signal.SIGTERM, old_term)
                signal.signal(signal.SIGINT, old_int)
    if capture_sealed:
        print("SEALED LOCALLY; UNADMITTED until independent Mac hash acknowledgement")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("smoke")
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--binary-sha256", required=True)
    run.add_argument("--runner-sha256", required=True)
    run.add_argument("--planner-sha256", required=True)
    run.add_argument("--supervisor-sha256", required=True)
    run.add_argument("--source-bundle-sha256", required=True)
    run.add_argument("--spool", type=Path, required=True)
    run.add_argument("--temperature-input", type=Path, required=True)
    run.add_argument("--cohort-seed", type=int, default=2026092601)
    sub.add_parser("restore")
    sub.add_parser("restore-if-inactive")
    args = parser.parse_args()
    if args.command == "restore":
        restore_policy(POLICY_STATE)
    elif args.command == "restore-if-inactive":
        restore_if_inactive()
    else:
        run_smoke(args)


if __name__ == "__main__":
    main()
