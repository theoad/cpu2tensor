# SPDX-License-Identifier: AGPL-3.0-only
"""Run one bounded collector under an exact, automatically restored CPU policy."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


NO_TURBO = Path("/sys/devices/system/cpu/intel_pstate/no_turbo")
LOCK_PATH = Path("/run/lock/cpu2tensor-foundation-capture.lock")


def _maximum(cpu: int) -> Path:
    return Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq")


def _read(path: Path) -> str:
    return path.read_text().strip()


def _write(path: Path, value: str) -> None:
    for _ in range(20):
        path.write_text(value + "\n")
        if _read(path) == value:
            return
        time.sleep(0.05)
    raise RuntimeError(f"CPU policy write did not stick: {path}")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _json_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def run(args: argparse.Namespace) -> int:
    if os.geteuid() != 0:
        raise RuntimeError("foundation policy wrapper requires root")
    if not args.command:
        raise ValueError("policy wrapper needs a collector command after --")
    lock_descriptor = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_descriptor)
        raise RuntimeError("another foundation collector owns the CPU policy") from error
    maximum = _maximum(args.cpu)
    original = {"no_turbo": _read(NO_TURBO), "maximum_khz": _read(maximum)}
    if original["no_turbo"] not in ("0", "1") or not original["maximum_khz"].isdigit():
        raise RuntimeError("unexpected initial CPU frequency policy")
    receipt = {
        "schema": "cpu2tensor-hardware-foundation-policy-v1",
        "cpu": args.cpu,
        "original": original,
        "requested": {"no_turbo": args.no_turbo,
                      "maximum_khz": args.maximum_khz},
        "started_unix_ns": time.time_ns(),
        "command": args.command,
        "restored": False,
    }
    _atomic_json(args.receipt, receipt)
    process: subprocess.Popen[bytes] | None = None
    interrupted = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    old_term = signal.signal(signal.SIGTERM, stop)
    old_int = signal.signal(signal.SIGINT, stop)
    try:
        _write(NO_TURBO, args.no_turbo)
        _write(maximum, args.maximum_khz)
        process = subprocess.Popen(args.command, start_new_session=True)
        return_code = process.wait()
        if interrupted and return_code == 0:
            return_code = 128 + signal.SIGTERM
        return return_code
    finally:
        # Re-enable turbo before restoring a maximum above the non-turbo
        # ceiling; intel_pstate otherwise clamps the write to base frequency.
        restore_error: BaseException | None = None
        try:
            _write(NO_TURBO, original["no_turbo"])
            _write(maximum, original["maximum_khz"])
        except BaseException as error:
            restore_error = error
        receipt["ended_unix_ns"] = time.time_ns()
        receipt["restored"] = (
            _read(NO_TURBO) == original["no_turbo"] and
            _read(maximum) == original["maximum_khz"]
        )
        receipt["return_code"] = None if process is None else process.poll()
        _atomic_json(args.receipt, receipt)
        if args.publish_artifact is not None:
            pending = args.publish_artifact / "capture-manifest.pending.json"
            published = args.publish_artifact / "capture-manifest.json"
            if (receipt["return_code"] == 0 and receipt["restored"] and
                    pending.exists()):
                manifest = json.loads(pending.read_text())
                manifest["cpu_policy_receipt"] = {
                    "content": receipt,
                    "content_sha256": _json_hash(receipt),
                }
                manifest["manifest_content_sha256"] = _json_hash({
                    key: value for key, value in manifest.items()
                    if key != "manifest_content_sha256"
                })
                _atomic_json(published, manifest)
                pending.unlink()
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        os.close(lock_descriptor)
        if restore_error is not None:
            raise restore_error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cpu", type=int, default=2)
    result.add_argument("--no-turbo", choices=("0", "1"), default="1")
    result.add_argument("--maximum-khz", default="1800000")
    result.add_argument("--receipt", type=Path, required=True)
    result.add_argument("--publish-artifact", type=Path)
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
