# SPDX-License-Identifier: AGPL-3.0-only
"""Diagnose which retained hardware modalities transfer across capture sessions.

This is a development diagnostic over the already reviewed matched pairs.  It
does not capture or execute a target, tune a detector, or qualify an operational
threshold.  Benign training rows alone define feature scaling.  The first
canary session defines one fixed mean-difference direction per modality; the
second session is then scored without refitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SCHEMA = "cpu2tensor-hardware-transfer-stability-v1"
MODALITIES = ("pt", "pebs", "pmu", "timing", "all")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def modality_vectors(batch: dict[str, torch.Tensor], modality: str) -> torch.Tensor:
    """Return one finite row per execution without using labels or scores."""
    if modality == "pt":
        values = (batch["pt"], batch["pt_available"].to(torch.float32))
    elif modality == "pebs":
        values = (batch["pebs"], batch["pebs_available"].to(torch.float32))
    elif modality == "pmu":
        values = (batch["pmu"], batch["pmu_available"].to(torch.float32))
    elif modality == "timing":
        values = (
            torch.nan_to_num(batch["time_bounds"]),
            torch.nan_to_num(batch["timing_quality"]),
        )
    elif modality == "all":
        return torch.cat(
            [modality_vectors(batch, name) for name in MODALITIES[:-1]], dim=1
        )
    else:
        raise ValueError(f"unknown modality: {modality}")
    flattened = torch.cat([value.flatten(1).to(torch.float64) for value in values], dim=1)
    if not bool(torch.isfinite(flattened).all()):
        raise ValueError(f"nonfinite {modality} representation")
    return flattened


def fit_scale(benign: dict[str, torch.Tensor], modality: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = modality_vectors(benign, modality)
    center = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False)
    active = scale > 1e-8
    if not bool(active.any()):
        raise ValueError(f"benign training data has no varying {modality} features")
    return center[active], scale[active], active


def normalize(
    batch: dict[str, torch.Tensor],
    modality: str,
    center: torch.Tensor,
    scale: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    # Clipping is fixed before the canary labels are opened.  It prevents a
    # single benign-near-constant coordinate from dominating this diagnosis.
    values = modality_vectors(batch, modality)[:, active]
    return ((values - center) / scale).clamp(-20.0, 20.0)


def auc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    comparison = positive[:, None] - negative[None, :]
    return float(((comparison > 0).to(torch.float64) +
                  0.5 * (comparison == 0).to(torch.float64)).mean())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    return float(torch.dot(left, right) / denominator) if denominator > 0 else 0.0


def diagnose_modality(
    benign: dict[str, torch.Tensor],
    train_pairs: dict[str, dict[str, torch.Tensor]],
    test_pairs: dict[str, dict[str, torch.Tensor]],
    modality: str,
) -> dict[str, object]:
    if set(train_pairs) != {"effect", "neutral"} or set(test_pairs) != {"effect", "neutral"}:
        raise ValueError("diagnosis requires exact effect/neutral arms")
    for name, pairs in (("train", train_pairs), ("test", test_pairs)):
        counts = {arm: int(batch["pt"].shape[0]) for arm, batch in pairs.items()}
        if not counts["effect"] or counts["effect"] != counts["neutral"]:
            raise ValueError(f"{name} effect/neutral arms must be nonempty and equal")
    center, scale, active = fit_scale(benign, modality)
    train = {arm: normalize(batch, modality, center, scale, active)
             for arm, batch in train_pairs.items()}
    test = {arm: normalize(batch, modality, center, scale, active)
            for arm, batch in test_pairs.items()}
    train_effect = train["effect"].mean(dim=0) - train["neutral"].mean(dim=0)
    test_effect = test["effect"].mean(dim=0) - test["neutral"].mean(dim=0)
    train_midpoint = (train["effect"].mean(dim=0) + train["neutral"].mean(dim=0)) / 2
    test_midpoint = (test["effect"].mean(dim=0) + test["neutral"].mean(dim=0)) / 2
    session_shift = test_midpoint - train_midpoint
    train_effect_length = train_effect.norm()
    if train_effect_length <= 1e-12:
        raise ValueError(f"zero {modality} training effect direction")
    direction = train_effect / train_effect_length
    scores = {
        split: {arm: values @ direction for arm, values in pairs.items()}
        for split, pairs in (("train", train), ("test", test))
    }
    session_norm = float(session_shift.norm())
    effect_norm = float(train_effect.norm())
    return {
        "dimensions": int(active.numel()),
        "active_dimensions": int(active.sum()),
        "train_effect_norm": effect_norm,
        "test_effect_norm": float(test_effect.norm()),
        "session_shift_norm": session_norm,
        "train_effect_to_session_shift": effect_norm / max(session_norm, 1e-12),
        "cross_session_effect_cosine": cosine(train_effect, test_effect),
        "session_shift_projection": float(torch.dot(session_shift, direction)),
        "train_auc": auc(scores["train"]["effect"], scores["train"]["neutral"]),
        "test_auc": auc(scores["test"]["effect"], scores["test"]["neutral"]),
        "train_paired_wins": int((scores["train"]["effect"] > scores["train"]["neutral"]).sum()),
        "test_paired_wins": int((scores["test"]["effect"] > scores["test"]["neutral"]).sum()),
        "pairs_per_session": int(scores["test"]["effect"].numel()),
    }


def run(data: Path, output: Path, plan_sha256: str) -> dict[str, object]:
    if output.exists():
        raise ValueError("use a fresh output path")
    if sha256(data / "plan.json") != plan_sha256:
        raise ValueError("plan hash mismatch")
    plan = json.loads((data / "plan.json").read_text())
    train_path, evaluation_path = data / "train.pt", data / "evaluation.pt"
    if sha256(train_path) != plan["train_sha256"] or sha256(evaluation_path) != plan["evaluation_sha256"]:
        raise ValueError("tensor cache hash mismatch")
    train = torch.load(train_path, map_location="cpu", weights_only=True)
    evaluation = torch.load(evaluation_path, map_location="cpu", weights_only=True)
    if train["schema"] != evaluation["schema"] or train["manifest_sha256"] != evaluation["manifest_sha256"]:
        raise ValueError("training/evaluation identity mismatch")
    result = {
        "schema": SCHEMA,
        "scope": "retained single-defect cross-session development diagnosis",
        "operational_go": False,
        "plan_sha256": plan_sha256,
        "train_sha256": plan["train_sha256"],
        "evaluation_sha256": plan["evaluation_sha256"],
        "modalities": {
            modality: diagnose_modality(
                train["benign"], train["pairs"], evaluation["pairs"], modality
            )
            for modality in MODALITIES
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(args.data, args.output, args.plan_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
