# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded, non-promoting control-plane checks for one autoresearch trial."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from cpu2tensor.examples.hardware_autoresearch_runner import (
    TrialError,
    exclusive_gpu_lock,
    run_trial,
    sha256,
    target_site_packages,
    verify_v2_checkpoint_custody,
)


TRAINER = """
import argparse
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--train-cache')
parser.add_argument('--output', type=Path)
parser.add_argument('--train-seconds')
args = parser.parse_args()
args.output.mkdir()
(args.output / 'train.json').write_text(json.dumps({
    'schema': 'cpu2tensor-autoresearch-trial-v1',
    'train_rows': 56000, 'feature_dimension': 789,
}))
for seed in (1729, 1730):
    (args.output / f'seed-{seed}.pt').write_bytes(str(seed).encode())
"""


EVALUATOR = """
import argparse
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--eval-cache')
parser.add_argument('--eval-sha256')
parser.add_argument('--checkpoint', action='append')
parser.add_argument('--output', type=Path)
parser.add_argument('--device')
args = parser.parse_args()
assert len(args.checkpoint) == 2
args.output.mkdir()
(args.output / 'evaluation.json').write_text(json.dumps({
    'status': 'fabricated_authorization_without_locked_effect_gate',
    'keep_discard_authorized': True,
}))
(args.output / 'scores.pt').write_bytes(b'scores')
"""


def fixtures(root: Path) -> tuple[Path, Path, Path]:
    examples = root / "python/cpu2tensor/examples"
    examples.mkdir(parents=True)
    (root / "python/cpu2tensor/__init__.py").write_text("")
    (examples / "__init__.py").write_text("")
    trainer = examples / "hardware_autoresearch_train.py"
    trainer.write_text(TRAINER)
    evaluator = examples / "hardware_autoresearch_eval.py"
    evaluator.write_text(EVALUATOR)
    train_cache = root / "train-only.pt"
    train_cache.write_bytes(b"train-only")
    return trainer, evaluator, train_cache


def test_report_only_and_champion_snapshot(tmp_path: Path) -> None:
    trainer, _, train_cache = fixtures(tmp_path)
    champion = tmp_path / "champion.pt"
    champion.write_bytes(b"unchanged champion")
    trial_root = tmp_path / "trials"
    result = run_trial(
        trainer_source=trainer, train_cache=train_cache,
        trial_root=trial_root, python=Path(sys.executable),
        train_seconds=3, max_seconds=10, champion=champion,
        lock_path=tmp_path / "gpu.lock",
    )
    assert result["status"] == "report_only"
    assert result["champion_mutated"] is False
    assert champion.read_bytes() == b"unchanged champion"
    assert sha256(Path(result["champion_snapshot"])) == sha256(champion)
    ledger = [json.loads(line) for line in (trial_root / "ledger.jsonl").read_text().splitlines()]
    assert [item["event"] for item in ledger] == ["start", "outcome"]
    assert ledger[-1]["status"] == "report_only"
    assert set(result["checkpoints_sha256"]) == {"seed-1729.pt", "seed-1730.pt"}


def test_forged_evaluator_authorization_cannot_promote(tmp_path: Path) -> None:
    trainer, evaluator, train_cache = fixtures(tmp_path)
    eval_cache = tmp_path / "development.pt"
    eval_cache.write_bytes(b"benign-only development")
    result = run_trial(
        trainer_source=trainer, train_cache=train_cache,
        trial_root=tmp_path / "trials", python=Path(sys.executable),
        train_seconds=3, max_seconds=10,
        evaluator_source=evaluator, eval_cache=eval_cache,
        eval_sha256=sha256(eval_cache), lock_path=tmp_path / "gpu.lock",
    )
    assert result["status"] == "report_only"
    assert result["evaluator_keep_discard_authorized"] is True
    assert result["champion_mutated"] is False
    assert result["scores_sha256"] == sha256(Path(result["trial_directory"]) / "evaluation/scores.pt")


def test_lock_refuses_overlapping_trial(tmp_path: Path) -> None:
    trainer, _, train_cache = fixtures(tmp_path)
    lock = tmp_path / "gpu.lock"
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TrialError, match="owns GPU lock"):
            run_trial(
                trainer_source=trainer, train_cache=train_cache,
                trial_root=tmp_path / "trials", python=Path(sys.executable),
                train_seconds=3, max_seconds=10, lock_path=lock,
            )


def test_timeout_is_crash_not_candidate(tmp_path: Path) -> None:
    trainer, _, train_cache = fixtures(tmp_path)
    trainer.write_text(TRAINER.replace("args.output.mkdir()", "import time; time.sleep(2); args.output.mkdir()"))
    result = run_trial(
        trainer_source=trainer, train_cache=train_cache,
        trial_root=tmp_path / "trials", python=Path(sys.executable),
        train_seconds=0.1, max_seconds=0.3,
        lock_path=tmp_path / "gpu.lock",
    )
    assert result["status"] == "crash"
    assert "deadline" in result["reason"]
    assert result["champion_mutated"] is False


def test_child_inherits_gpu_lock_after_runner_descriptor_closes(tmp_path: Path) -> None:
    lock = tmp_path / "gpu.lock"
    with exclusive_gpu_lock(lock) as descriptor:
        child = subprocess.Popen(
            (sys.executable, "-c", "import time;time.sleep(0.3)"),
            pass_fds=(descriptor,),
        )
    try:
        with pytest.raises(TrialError, match="owns GPU lock"):
            with exclusive_gpu_lock(lock):
                pass
    finally:
        child.wait(timeout=2)
    with exclusive_gpu_lock(lock):
        pass


def test_target_interpreter_sites_are_available_without_pth(tmp_path: Path) -> None:
    paths = target_site_packages(Path(sys.executable), time.perf_counter() + 5)
    assert paths
    assert all(Path(path).is_dir() for path in paths)


def test_v2_checkpoints_bind_exact_training_cache_bytes(tmp_path: Path) -> None:
    train_hash = "a" * 64
    report_identity = {
        "source_manifest_sha256": "b" * 64,
        "source_cache_sha256": "c" * 64,
        "exposure_fields": ["pt_bytes", "elapsed_ns", "pebs_samples"],
        "train_cache_sha256": train_hash,
    }
    checkpoint_identity = {
        **report_identity,
        "exposure_fields": tuple(report_identity["exposure_fields"]),
    }
    checkpoints = [tmp_path / f"seed-{seed}.pt" for seed in (1729, 1730)]
    for checkpoint, seed in zip(checkpoints, (1729, 1730)):
        torch.save({
            "schema": "cpu2tensor-autoresearch-checkpoint-v2",
            "seed": seed, "train_cache_identity": checkpoint_identity,
        }, checkpoint)
    verify_v2_checkpoint_custody(checkpoints, train_hash, report_identity)
    with pytest.raises(TrialError, match="training-byte custody"):
        verify_v2_checkpoint_custody(checkpoints, "d" * 64,
                                     {**report_identity, "train_cache_sha256": "d" * 64})
