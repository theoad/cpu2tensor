# SPDX-License-Identifier: AGPL-3.0-only
"""One bounded, exclusive hardware autoresearch trial; no autonomous loop.

This immutable control plane snapshots the mutable trainer, enforces one Mac
GPU owner and a 300-second total deadline, hashes custody artifacts, and
appends trial outcomes.  The current benign-only evaluator cannot authorize
keep/discard, so a successful trial is report-only and never alters a champion.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid
from typing import Iterator


SCHEMA = "cpu2tensor-autoresearch-runner-v1"
MAX_TOTAL_SECONDS = 300.0
MAX_TRAIN_SECONDS = 240.0
EVALUATOR_MODULE = "cpu2tensor.examples.hardware_autoresearch_eval"
GPU_LOCK = Path.home() / ".cache/cpu2tensor/hardware-autoresearch.mac-mps.lock"


class TrialError(RuntimeError):
    """A trial did not complete its fixed integrity or time contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_ledger(path: Path, record: dict[str, object]) -> None:
    line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise OSError("short append-only ledger write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_gpu_lock(path: Path = GPU_LOCK) -> Iterator[int]:
    """Cooperating trials use one fixed host lock for the entire subprocess span."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TrialError(f"another autoresearch trial owns GPU lock {path}") from error
        yield descriptor
    finally:
        # Do not explicitly unlock: a child inherits this file description.
        # If the runner dies, the GPU lock must survive until that child exits.
        os.close(descriptor)


def recover_unfinished(ledger: Path) -> None:
    """Close abandoned starts after exclusive lock acquisition, without edits."""
    if not ledger.exists():
        return
    active: dict[str, dict[str, object]] = {}
    for line in ledger.read_text().splitlines():
        record = json.loads(line)
        trial_id = record.get("trial_id")
        if not isinstance(trial_id, str):
            raise TrialError("ledger has a record without trial_id")
        if record.get("event") == "start":
            if trial_id in active:
                raise TrialError("ledger has duplicate trial start")
            active[trial_id] = record
        elif record.get("event") == "outcome":
            if trial_id not in active:
                raise TrialError("ledger has an outcome without a start")
            del active[trial_id]
        else:
            raise TrialError("ledger contains an unknown event")
    for trial_id, start in active.items():
        append_ledger(ledger, {
            "schema": SCHEMA, "event": "outcome", "trial_id": trial_id,
            "status": "crash", "reason": "previous runner ended without outcome",
            "started_utc": start.get("started_utc"), "finished_utc": utc_now(),
            "recovered": True,
        })


def child_command(command: list[str], *, log_path: Path, deadline: float,
                  cwd: Path, environment: dict[str, str], lock_descriptor: int) -> None:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TrialError("total trial deadline expired before subprocess launch")
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=(lock_descriptor,),
        )
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TrialError(f"subprocess exceeded total deadline: {command[0]}") from error
        finally:
            log.flush()
            os.fsync(log.fileno())
    if process.returncode != 0:
        raise TrialError(f"subprocess exited {process.returncode}; inspect {log_path}")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TrialError(f"expected JSON object: {path}")
    return payload


def _peak_child_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def verify_v2_checkpoint_custody(
    checkpoints: list[Path], train_cache_sha256: str,
    train_identity: dict[str, object],
) -> None:
    """Bind each candidate checkpoint to the exact locked training bytes."""
    import torch

    expected = {
        "source_manifest_sha256": train_identity.get("source_manifest_sha256"),
        "source_cache_sha256": train_identity.get("source_cache_sha256"),
        "exposure_fields": ("pt_bytes", "elapsed_ns", "pebs_samples"),
        "train_cache_sha256": train_cache_sha256,
    }
    if (train_identity.get("train_cache_sha256") != train_cache_sha256 or
            train_identity.get("exposure_fields") != list(expected["exposure_fields"])):
        raise TrialError("v2 training report has wrong physical cache identity")
    for checkpoint, seed in zip(checkpoints, (1729, 1730)):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (not isinstance(payload, dict) or
                payload.get("schema") != "cpu2tensor-autoresearch-checkpoint-v2" or
                payload.get("seed") != seed or
                payload.get("train_cache_identity") != expected):
            raise TrialError(f"v2 checkpoint has wrong training-byte custody: {checkpoint}")


def target_site_packages(python: Path, deadline: float) -> list[str]:
    """Find the target interpreter's package roots without inheriting .pth hooks.

    The short normal-start probe asks only for paths.  Actual trainer/evaluator
    subprocesses use -S plus these paths, so a stale editable .pth finder is
    never installed into their import machinery.
    """
    code = (
        "import json,site,sys,sysconfig; "
        "paths=[*sys.path,site.getusersitepackages(),"
        "sysconfig.get_path('purelib'),sysconfig.get_path('platlib')]; "
        "print(json.dumps(paths))"
    )
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TrialError("total deadline expired before interpreter path probe")
    completed = subprocess.run(
        (str(python), "-c", code), capture_output=True, text=True,
        check=True, timeout=remaining,
    )
    discovered = json.loads(completed.stdout)
    if not isinstance(discovered, list):
        raise TrialError("interpreter site-path probe returned no list")
    paths = []
    for value in discovered:
        if (isinstance(value, str) and value and
                Path(value).name in ("site-packages", "dist-packages") and
                Path(value).is_dir() and value not in paths):
            paths.append(value)
    if not paths:
        raise TrialError("target interpreter has no site-package directory")
    return paths


def run_trial(
    *, trainer_source: Path, train_cache: Path, trial_root: Path,
    python: Path, train_seconds: float = MAX_TRAIN_SECONDS,
    max_seconds: float = MAX_TOTAL_SECONDS,
    evaluator_source: Path | None = None,
    eval_cache: Path | None = None,
    eval_sha256: str | None = None,
    champion: Path | None = None,
    lock_path: Path = GPU_LOCK,
) -> dict[str, object]:
    if not (0 < train_seconds <= MAX_TRAIN_SECONDS and
            train_seconds < max_seconds <= MAX_TOTAL_SECONDS):
        raise ValueError("require train<=240s and train<total<=300s")
    eval_inputs = (evaluator_source, eval_cache, eval_sha256)
    if any(value is not None for value in eval_inputs) and not all(
        value is not None for value in eval_inputs
    ):
        raise ValueError("evaluator source, cache, and expected hash are all required together")
    if trainer_source.name != "hardware_autoresearch_train.py":
        raise ValueError("runner accepts only the one mutable trainer file")
    if not python.exists() or not trainer_source.is_file() or not train_cache.is_file():
        raise ValueError("Python, trainer, and train-only cache must exist")
    if champion is not None and not champion.is_file():
        raise ValueError("champion snapshot source must be an existing file")
    if eval_cache is not None:
        if not evaluator_source.is_file() or not eval_cache.is_file():
            raise ValueError("evaluator source and cache must exist")
        expected_evaluator = trainer_source.parent / "hardware_autoresearch_eval.py"
        if evaluator_source.resolve() != expected_evaluator.resolve():
            raise ValueError("evaluator source must be the module invoked by this runner")
        if (not isinstance(eval_sha256, str) or len(eval_sha256) != 64 or
                any(character not in "0123456789abcdef" for character in eval_sha256)):
            raise ValueError("expected eval SHA-256 must be lowercase hex")

    trial_root.mkdir(parents=True, exist_ok=True)
    ledger = trial_root / "ledger.jsonl"
    with exclusive_gpu_lock(lock_path) as lock_descriptor:
        recover_unfinished(ledger)
        started = time.perf_counter()
        deadline = started + max_seconds
        trial_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        trial = trial_root / trial_id
        trial.mkdir(exist_ok=False)
        record: dict[str, object] = {
            "schema": SCHEMA, "event": "outcome", "trial_id": trial_id,
            "status": "crash", "reason": None,
            "started_utc": utc_now(), "trial_directory": str(trial),
            "host": platform.node(), "machine": platform.machine(),
            "max_total_seconds": max_seconds, "train_seconds_cap": train_seconds,
            "champion_mutated": False,
        }
        append_ledger(ledger, {
            "schema": SCHEMA, "event": "start", "trial_id": trial_id,
            "started_utc": record["started_utc"], "trial_directory": str(trial),
        })
        try:
            record["trainer_source_sha256"] = sha256(trainer_source)
            record["train_cache_sha256"] = sha256(train_cache)
            if eval_cache is not None:
                observed_eval_sha = sha256(eval_cache)
                if observed_eval_sha != eval_sha256:
                    raise TrialError("protected eval cache hash does not match expected SHA-256")
                record["eval_cache_sha256"] = observed_eval_sha
                record["evaluator_source_sha256"] = sha256(evaluator_source)
            if champion is not None:
                champion_hash = sha256(champion)
                snapshot = trial / "champion-before.pt"
                shutil.copyfile(champion, snapshot)
                if sha256(snapshot) != champion_hash:
                    raise TrialError("champion snapshot hash mismatch")
                record["champion_sha256_before"] = champion_hash
                record["champion_snapshot"] = str(snapshot)

            snapshot_source = trial / "hardware_autoresearch_train.py"
            shutil.copyfile(trainer_source, snapshot_source)
            if sha256(snapshot_source) != record["trainer_source_sha256"]:
                raise TrialError("trainer source changed during snapshot")
            environment = os.environ.copy()
            # Source-first package path is needed for the evaluator; running
            # with -S prevents the local stale editable finder from overriding it.
            source_python_root = trainer_source.parent.parent.parent
            sites = target_site_packages(python, deadline)
            environment["PYTHONPATH"] = os.pathsep.join((
                str(source_python_root), *sites,
            ))
            record["interpreter_site_packages"] = sites
            train_output = trial / "train"
            child_command([
                str(python), "-S", str(snapshot_source),
                "--train-cache", str(train_cache),
                "--output", str(train_output),
                "--train-seconds", str(train_seconds),
            ], log_path=trial / "train.log", deadline=deadline,
                cwd=trainer_source.parent.parent.parent.parent, environment=environment,
                lock_descriptor=lock_descriptor)
            train_report = train_output / "train.json"
            parsed_train = _read_json(train_report)
            trial_schema = parsed_train.get("schema")
            if (trial_schema not in ("cpu2tensor-autoresearch-trial-v1",
                                     "cpu2tensor-autoresearch-trial-v2") or
                    parsed_train.get("train_rows") != 56_000 or
                    parsed_train.get("feature_dimension") != 789):
                raise TrialError("trainer output has wrong schema or train-only shape")
            checkpoints = [train_output / f"seed-{seed}.pt" for seed in (1729, 1730)]
            if not all(path.is_file() for path in checkpoints):
                raise TrialError("trainer omitted a required seed checkpoint")
            if trial_schema == "cpu2tensor-autoresearch-trial-v2":
                if (parsed_train.get("train_cache_schema") !=
                        "cpu2tensor-autoresearch-train-v2" or
                        not isinstance(parsed_train.get("train_cache_identity"), dict)):
                    raise TrialError("v2 trainer report lacks train-only cache custody")
                verify_v2_checkpoint_custody(
                    checkpoints, record["train_cache_sha256"],
                    parsed_train["train_cache_identity"],
                )
            record["train_report_sha256"] = sha256(train_report)
            record["checkpoints_sha256"] = {
                path.name: sha256(path) for path in checkpoints
            }
            if eval_cache is not None:
                eval_output = trial / "evaluation"
                child_command([
                    str(python), "-S", "-m", EVALUATOR_MODULE,
                    "--eval-cache", str(eval_cache),
                    "--eval-sha256", eval_sha256,
                    "--checkpoint", str(checkpoints[0]),
                    "--checkpoint", str(checkpoints[1]),
                    "--output", str(eval_output),
                    "--device", "cpu",
                ], log_path=trial / "evaluation.log", deadline=deadline,
                    cwd=trainer_source.parent.parent.parent.parent, environment=environment,
                    lock_descriptor=lock_descriptor)
                evaluation_report = eval_output / "evaluation.json"
                scores = eval_output / "scores.pt"
                parsed_eval = _read_json(evaluation_report)
                if not scores.is_file() or not isinstance(
                    parsed_eval.get("keep_discard_authorized"), bool
                ):
                    raise TrialError("evaluator output lacks required scores or authorization state")
                record["evaluation_report_sha256"] = sha256(evaluation_report)
                record["scores_sha256"] = sha256(scores)
                record["evaluator_status"] = parsed_eval.get("status")
                record["evaluator_keep_discard_authorized"] = parsed_eval["keep_discard_authorized"]
                if trial_schema == "cpu2tensor-autoresearch-trial-v2":
                    source_identity = parsed_eval.get("source_identity")
                    if (not isinstance(source_identity, dict) or
                            source_identity.get("train_cache_sha256") !=
                            record["train_cache_sha256"] or
                            source_identity.get("source_manifest_sha256") !=
                            parsed_train["train_cache_identity"]["source_manifest_sha256"] or
                            source_identity.get("source_cache_sha256") !=
                            parsed_train["train_cache_identity"]["source_cache_sha256"]):
                        raise TrialError("v2 evaluator report has wrong training-byte custody")
            if sha256(trainer_source) != record["trainer_source_sha256"] or sha256(train_cache) != record["train_cache_sha256"]:
                raise TrialError("trainer source or train-only cache changed during trial")
            if eval_cache is not None and (
                sha256(eval_cache) != record["eval_cache_sha256"] or
                sha256(evaluator_source) != record["evaluator_source_sha256"]
            ):
                raise TrialError("evaluator source or protected cache changed during trial")
            if champion is not None and sha256(champion) != record["champion_sha256_before"]:
                raise TrialError("champion changed during report-only trial")
            if time.perf_counter() > deadline:
                raise TrialError("total trial deadline expired during integrity checks")
            # No locked separate-session effect/control metric exists yet.
            # A benign-only report or even an unexpected authorization bit
            # never grants promotion in this version of the runner.
            record["status"] = "report_only"
            record["reason"] = (
                "no locked separate-session physical effect/control metric; "
                "automatic keep/discard is disabled"
            )
        except Exception as error:
            record["status"] = "crash"
            record["reason"] = f"{type(error).__name__}: {error}"
        record["finished_utc"] = utc_now()
        record["total_wall_seconds"] = time.perf_counter() - started
        record["peak_child_rss_bytes"] = _peak_child_rss_bytes()
        (trial / "outcome.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        append_ledger(ledger, record)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-source", required=True, type=Path)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--trial-root", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--train-seconds", type=float, default=MAX_TRAIN_SECONDS)
    parser.add_argument("--max-seconds", type=float, default=MAX_TOTAL_SECONDS)
    parser.add_argument("--evaluator-source", type=Path)
    parser.add_argument("--eval-cache", type=Path)
    parser.add_argument("--eval-sha256")
    parser.add_argument("--champion", type=Path)
    args = parser.parse_args()
    result = run_trial(
        trainer_source=args.trainer_source, train_cache=args.train_cache,
        trial_root=args.trial_root, python=args.python,
        train_seconds=args.train_seconds, max_seconds=args.max_seconds,
        evaluator_source=args.evaluator_source, eval_cache=args.eval_cache,
        eval_sha256=args.eval_sha256, champion=args.champion,
    )
    print(json.dumps({
        "trial_id": result["trial_id"], "status": result["status"],
        "reason": result["reason"], "total_wall_seconds": result["total_wall_seconds"],
        "trial_directory": result["trial_directory"],
    }, sort_keys=True))
    if result["status"] == "crash":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
