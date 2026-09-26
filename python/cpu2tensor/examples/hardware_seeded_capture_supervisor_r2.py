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
import re
import shutil
import signal
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
    if not _read(path.parent / f"temp{match.group(1)}_label").startswith("Package id"):
        raise ValueError("temperature input is not a package sensor")
    value = int(_read(path))
    if not 0 <= value <= 125_000:
        raise ValueError("invalid package temperature reading")
    return value


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
    if _read(BOOT_ID) != EXPECTED_BOOT or platform.release() != EXPECTED_KERNEL:
        raise ValueError("boot or kernel changed")
    repo = Path(__file__).resolve().parents[3]
    if repo != args.source_root.resolve():
        raise ValueError("supervisor source checkout differs from reviewed path")
    runner = repo / "python/cpu2tensor/examples/hardware_multimodal_experiment.py"
    planner = repo / "python/cpu2tensor/examples/hardware_seeded_capture_plan_r2.py"
    supervisor = Path(__file__).resolve()
    if Path(collector.__file__).resolve() != runner or Path(plans.__file__).resolve() != planner:
        raise ValueError("Python resolved another checkout")
    if (sha256(runner) != args.runner_sha256 or
            sha256(planner) != args.planner_sha256 or
            sha256(supervisor) != args.supervisor_sha256):
        raise ValueError("collector, planner, or supervisor source changed")
    bundle = source_bundle_manifest(repo)
    if bundle["digest_sha256"] != args.source_bundle_sha256:
        raise ValueError("portable source bundle changed")
    _check_child_resolution(repo)
    if not args.binary.is_file() or sha256(args.binary) != args.binary_sha256:
        raise ValueError("workload binary changed")
    spool = args.spool
    if spool.is_symlink() or not spool.is_dir() or not os.path.ismount(spool):
        raise ValueError("spool is not a dedicated mounted directory")
    source, filesystem, target = _mount_fields(spool)
    if source != "tmpfs" or filesystem != "tmpfs" or Path(target) != spool.resolve():
        raise ValueError("spool must be the exact dedicated tmpfs mount")
    filesystem_size = os.statvfs(spool).f_blocks * os.statvfs(spool).f_frsize
    if filesystem_size != SPOOL_CAP:
        raise ValueError("tmpfs must be exactly 896 MiB for the bounded pilot")
    if any(spool.iterdir()):
        raise ValueError("smoke requires an empty dedicated tmpfs")
    if shutil.disk_usage("/").free < MIN_START_FREE:
        raise ValueError("root filesystem has less than 5.5 GiB free")
    temperature = package_temperature_millic(args.temperature_input)
    if temperature >= START_MILLIC:
        raise ValueError("package is too hot to start")
    no_turbo, maximum = _read(NO_TURBO), _read(MAX_FREQ)
    if no_turbo != "0" or not maximum.isdecimal():
        raise ValueError("unexpected pre-run CPU frequency policy")
    if args.state_path.exists():
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


def run_smoke(args: argparse.Namespace) -> None:
    original_no_turbo, original_maximum = preflight(args)
    process: subprocess.Popen[bytes] | None = None
    state_sealed = False
    capture_sealed = False
    old_term = signal.signal(signal.SIGTERM, _interrupted)
    old_int = signal.signal(signal.SIGINT, _interrupted)
    try:
        _seal_policy_state(args.state_path, original_no_turbo, original_maximum)
        state_sealed = True
        _write(NO_TURBO, "1")
        _write(MAX_FREQ, CAPTURE_FREQ_KHZ)
        if _read(NO_TURBO) != "1" or _read(MAX_FREQ) != CAPTURE_FREQ_KHZ:
            raise RuntimeError("capture CPU policy did not stick")
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
                if os.statvfs(args.spool).f_bavail * os.statvfs(args.spool).f_frsize < 128 * MIB:
                    raise RuntimeError("tmpfs headroom fell below 128 MiB")
                if _read(NO_TURBO) != "1" or _read(MAX_FREQ) != CAPTURE_FREQ_KHZ:
                    raise RuntimeError("CPU frequency policy changed during capture")
                time.sleep(1)
            if process.wait() != 0:
                raise RuntimeError("collector failed; preserve tmpfs for diagnosis")
        verify_smoke_artifact(artifact, identity["identity_sha256"], plan)
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
                    restore_policy(args.state_path)
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
    run.add_argument("--state-path", type=Path, default=Path("/run/cpu2tensor-seeded-r2-policy.json"))
    restore = sub.add_parser("restore")
    restore.add_argument("--state-path", type=Path, default=Path("/run/cpu2tensor-seeded-r2-policy.json"))
    args = parser.parse_args()
    if args.command == "restore":
        restore_policy(args.state_path)
    else:
        run_smoke(args)


if __name__ == "__main__":
    main()
