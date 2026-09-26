# SPDX-License-Identifier: AGPL-3.0-only
"""Short scratch/frozen/fine-tuned transfer comparison on retained traces.

All trainable architecture, losses, and optimizer choices live in this file.
Training loads only train.pt. Evaluation is opened after all three checkpoints
are saved; thresholds depend only on the separate benign calibration partition.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F

from cpu2tensor.examples.hardware_multimodal import (
    HardwareMultimodalBatch, MaskedHardwareModel, MaskedTokens, MultimodalConfig,
    make_training_masks, masked_reconstruction_loss,
)
from cpu2tensor.examples.hardware_transfer_data import SCHEMA, sha256
from cpu2tensor.examples.hardware_anomaly_exposure_r2 import (
    condition_features, make_design, normalize, score as density_score,
)


ARMS = ("scratch", "frozen", "finetuned")
HEAD_RATE = 1e-3
ENCODER_RATE = 1e-4
RECONSTRUCTION_WEIGHT = 0.05
PAIRS_PER_STEP = 8
BENIGN_PER_STEP = 8
SCORE_BATCH = 16
THREADS = 1
PRETRAINED_SHA256 = "77de8becc0a64ee89f9708c3b7531136d732dc4f3405c01d616d71d0645da02b"
BASELINE_SHA256 = "63d241e035c2ea969332b4ba5001868150d622713560a62dad37c3c7733bdab9"


def package_temperature() -> float | None:
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        if (zone / "type").read_text().strip() == "x86_pkg_temp":
            return int((zone / "temp").read_text()) / 1000.0
    return None


def required_temperature() -> float:
    temperature = package_temperature()
    if temperature is None or not 0 < temperature < 120:
        raise RuntimeError("valid CPU package temperature sensor required")
    return temperature


def check_health(deadline: float) -> None:
    """Bound host heat without changing the subject's CPU policy."""
    if time.perf_counter() >= deadline:
        raise TimeoutError("cycle deadline reached")
    temperature = required_temperature()
    if temperature >= 80:
        raise RuntimeError(f"CPU package temperature reached {temperature:.1f} C")
    # Offline training can wait for cooling; this never delays a captured target.
    if temperature >= 72:
        while temperature >= 68:
            time.sleep(0.1)
            if time.perf_counter() >= deadline:
                raise TimeoutError("cycle deadline reached during cooling")
            temperature = required_temperature()
            if temperature >= 80:
                raise RuntimeError("package temperature unavailable or unsafe during cooling")


def no_masks(batch: HardwareMultimodalBatch) -> MaskedTokens:
    return MaskedTokens(*(torch.zeros_like(value) for value in
                          (batch.pt_available, batch.pebs_available, batch.pmu_available)))


def concatenate(batches: list[HardwareMultimodalBatch]) -> HardwareMultimodalBatch:
    return HardwareMultimodalBatch(**{
        field.name: torch.cat([getattr(batch, field.name) for batch in batches])
        for field in fields(HardwareMultimodalBatch)
    })


def pooled_tokens(tokens: torch.Tensor, batch: HardwareMultimodalBatch) -> torch.Tensor:
    groups = ((slice(0, 16), batch.pt_available),
              (slice(16, 32), batch.pebs_available),
              (slice(32, 33), batch.pmu_available))
    pooled = []
    for interval, available in groups:
        count = available.sum((1, 2)).clamp_min(1)
        values = torch.where(available[..., None], tokens[:, :, interval], 0.0)
        pooled.append(values.sum((1, 2)) / count[:, None])
    return torch.cat(pooled, dim=-1)


class TransferDetector(nn.Module):
    def __init__(self, encoder: MaskedHardwareModel) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(3 * encoder.config.model_dimensions, 1)

    def forward(self, batch: HardwareMultimodalBatch) -> torch.Tensor:
        tokens = self.encoder.encode(batch, no_masks(batch))
        return self.head(pooled_tokens(tokens, batch)).squeeze(-1)

    def contributions(self, batch: HardwareMultimodalBatch) -> torch.Tensor:
        """Exact additive contribution of each observed token, excluding bias."""
        tokens = self.encoder.encode(batch, no_masks(batch))
        weights = self.head.weight.reshape(3, self.encoder.config.model_dimensions)
        contributions = []
        for index, (interval, available) in enumerate((
            (slice(0, 16), batch.pt_available),
            (slice(16, 32), batch.pebs_available),
            (slice(32, 33), batch.pmu_available),
        )):
            count = available.sum((1, 2)).clamp_min(1)
            contribution = (tokens[:, :, interval] * weights[index]).sum(-1)
            contributions.append(torch.where(available, contribution / count[:, None, None], 0.0))
        return torch.cat(contributions, dim=2)


def build_detector(pretrained: dict, arm: str, seed: int) -> TransferDetector:
    if arm not in ARMS:
        raise ValueError("unknown transfer arm")
    torch.manual_seed(seed)
    encoder = MaskedHardwareModel(MultimodalConfig(**pretrained["config"]))
    if arm == "scratch":
        # Identical pretraining-only scaling in every arm; only weights differ.
        for name, buffer in encoder.named_buffers():
            buffer.copy_(pretrained["state"][name])
    else:
        encoder.load_state_dict(pretrained["state"])
    for parameter in encoder.parameters():
        parameter.requires_grad_(arm != "frozen")
    # Identical initial head even though scratch encoder construction consumes RNG.
    torch.manual_seed(seed + 1)
    return TransferDetector(encoder)


def pair_loss(positive: torch.Tensor, negative: torch.Tensor,
              benign: torch.Tensor) -> torch.Tensor:
    return (F.softplus(negative - positive).mean() +
            0.25 * (F.softplus(-positive).mean() + F.softplus(negative).mean()) +
            0.25 * F.softplus(benign).mean())


def train_arm(pretrained: dict, data: dict, arm: str, seed: int,
              steps: int, deadline: float, output: Path) -> dict:
    detector = build_detector(pretrained, arm, seed)
    detector.train()
    positive = HardwareMultimodalBatch(**data["pairs"]["effect"])
    negative = HardwareMultimodalBatch(**data["pairs"]["neutral"])
    benign = HardwareMultimodalBatch(**data["benign"])
    optimizer = torch.optim.AdamW([
        {"params": detector.head.parameters(), "lr": HEAD_RATE},
        {"params": [parameter for parameter in detector.encoder.parameters()
                    if parameter.requires_grad], "lr": ENCODER_RATE},
    ], weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    mask_generator = torch.Generator().manual_seed(seed + 100_000)
    start = time.perf_counter()
    initial_encoder = {key: value.clone() for key, value in detector.encoder.state_dict().items()}
    losses = []
    row_schedule = hashlib.sha256()
    cached = None
    if arm == "frozen":
        detector.encoder.eval()
        with torch.no_grad():
            cached = []
            for batch in (positive, negative, benign):
                chunks = []
                for offset in range(0, batch.batch_size, SCORE_BATCH):
                    check_health(deadline)
                    selection = torch.arange(offset, min(offset + SCORE_BATCH, batch.batch_size))
                    part = batch.index_select(selection)
                    chunks.append(pooled_tokens(detector.encoder.encode(part, no_masks(part)), part))
                cached.append(torch.cat(chunks))
    for _ in range(steps):
        check_health(deadline)
        pairs = torch.randperm(positive.batch_size, generator=generator)[:PAIRS_PER_STEP]
        normals = torch.randperm(benign.batch_size, generator=generator)[:BENIGN_PER_STEP]
        row_schedule.update(pairs.numpy().tobytes())
        row_schedule.update(normals.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        if cached is not None:
            logits = [detector.head(cached[index][selection]).squeeze(-1)
                      for index, selection in enumerate((pairs, pairs, normals))]
            loss = pair_loss(*logits)
        else:
            background = benign.index_select(normals)
            combined = concatenate([positive.index_select(pairs),
                                    negative.index_select(pairs), background])
            logits = detector(combined)
            count = len(pairs)
            loss = pair_loss(logits[:count], logits[count:2 * count], logits[2 * count:])
            masks = make_training_masks(background, generator=mask_generator)
            loss = loss + RECONSTRUCTION_WEIGHT * masked_reconstruction_loss(
                detector.encoder, background, masks,
            )
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite transfer training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    changed = any(not torch.equal(value, detector.encoder.state_dict()[key])
                  for key, value in initial_encoder.items())
    if arm == "frozen" and changed:
        raise RuntimeError("frozen encoder was modified")
    detector.eval()
    checkpoint = output / f"{arm}.pt"
    torch.save({"schema": "cpu2tensor-hardware-transfer-checkpoint-v1",
                "config": pretrained["config"], "state": detector.state_dict(),
                "arm": arm, "seed": seed, "cpu_threads": torch.get_num_threads()}, checkpoint)
    return {"steps": steps, "loss_first": losses[0], "loss_last": losses[-1],
            "train_seconds": time.perf_counter() - start,
            "encoder_changed": changed, "checkpoint_sha256": sha256(checkpoint),
            "row_schedule_sha256": row_schedule.hexdigest(),
            "total_parameters": sum(parameter.numel() for parameter in detector.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in detector.parameters()
                                        if parameter.requires_grad)}


def load_detector(path: Path) -> TransferDetector:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    detector = TransferDetector(MaskedHardwareModel(MultimodalConfig(**payload["config"])))
    detector.load_state_dict(payload["state"])
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    return detector


@torch.no_grad()
def score(detector: TransferDetector, payload: dict,
          deadline: float = float("inf"), *, physical_host: bool = True) -> torch.Tensor:
    if not physical_host and platform.system() != "Darwin":
        raise RuntimeError("off-host scoring is restricted to the Mac coordinator")
    batch = HardwareMultimodalBatch(**payload)
    chunks = []
    for start in range(0, batch.batch_size, SCORE_BATCH):
        if physical_host:
            check_health(deadline)
        elif time.perf_counter() >= deadline:
            raise TimeoutError("off-host scoring deadline reached")
        selection = torch.arange(start, min(start + SCORE_BATCH, batch.batch_size))
        chunks.append(detector(batch.index_select(selection)))
    result = torch.cat(chunks)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite evaluation score")
    return result


def metrics(values: dict[str, torch.Tensor]) -> dict:
    positive, negative = values["effect"], values["neutral"]
    comparisons = positive[:, None] - negative[None, :]
    auc = float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean())
    result = {"auc": auc, "paired_wins": int((positive > negative).sum()),
              "pairs": len(positive), "thresholds": {}}
    calibration = values["calibration"]
    for name, tail in (("development_1pct", max(1, int(len(calibration) * 0.01))),
                       ("zero_calibration_alerts", 0)):
        threshold = float(torch.topk(calibration, tail + 1).values[-1])
        result["thresholds"][name] = {
            "threshold": threshold, "calibration_rows": len(calibration),
            "alerts": {key: int((value > threshold).sum()) for key, value in values.items()},
            "rows": {key: len(value) for key, value in values.items()},
            "certifies_operational_fpr": False,
        }
    return result


def exposure_scores(evaluation: dict, baseline_path: Path, manifest_hash: str) -> dict:
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if baseline["manifest_sha256"] != manifest_hash or baseline["variant"] != "volume_time":
        raise ValueError("baseline differs from preregistered exposure scorer")
    values = {}
    for key, payload in evaluation["baseline"].items():
        partition = torch.ones(len(payload["features"]), dtype=torch.uint8)
        pt_design, pebs_design, _ = make_design(payload["exposure"], partition, "volume_time", baseline)
        conditioned, _, _ = condition_features(payload["features"], pt_design, pebs_design, partition,
                                               baseline["pt_coefficients"], baseline["pebs_coefficients"])
        standardized, _, _ = normalize(conditioned, partition,
                                       baseline["feature_center"], baseline["feature_scale"])
        values[key] = density_score(standardized)
    return values


def run(data_root: Path, pretrained_path: Path, baseline_path: Path, output: Path,
        seed: int, steps: int, max_seconds: float, expected_plan_sha256: str,
        threads: int = THREADS) -> dict:
    if output.exists() or steps <= 0 or not 0 < max_seconds <= 300:
        raise ValueError("use fresh output, positive steps and a <=300-second cycle")
    if threads not in (1, 2):
        raise ValueError("short host cycles support one or two CPU threads")
    torch.set_num_threads(threads)
    start = time.perf_counter()
    deadline = start + max_seconds
    while required_temperature() > 70:
        check_health(deadline)
        time.sleep(0.1)
    if (sha256(data_root / "plan.json") != expected_plan_sha256 or
            sha256(pretrained_path) != PRETRAINED_SHA256 or
            sha256(baseline_path) != BASELINE_SHA256):
        raise ValueError("plan or archived checkpoint differs from the preregistered hashes")
    plan = json.loads((data_root / "plan.json").read_text())
    if sha256(data_root / "train.pt") != plan["train_sha256"]:
        raise ValueError("training cache hash mismatch")
    data = torch.load(data_root / "train.pt", map_location="cpu", weights_only=True)
    pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    for key in ("subject_identity_sha256", "event_identity_sha256"):
        if pretrained["metadata"][key] != data[key] or data[key] != plan[key]:
            raise ValueError("pretraining checkpoint and transfer subject differ")
    if data["schema"] != SCHEMA:
        raise ValueError("unsupported transfer data")
    output.mkdir(parents=True)
    report = {"schema": "cpu2tensor-hardware-transfer-result-v1",
              "scope": plan["scope"], "operational_go": False,
              "unseen_defect_evaluation": False, "independent_input_evaluation": False,
              "host": platform.node(), "machine": platform.machine(), "device": "cpu",
              "cpu_threads": threads, "seed": seed, "requested_steps": steps,
              "plan_sha256": sha256(data_root / "plan.json"),
              "pretrained_sha256": sha256(pretrained_path),
              "experiment_source_sha256": sha256(Path(__file__)), "arms": {}}
    report["encoder_source_sha256"] = sha256(Path(__file__).with_name("hardware_multimodal.py"))
    for arm in ARMS:
        report["arms"][arm] = train_arm(pretrained, data, arm, seed, steps, deadline, output)
        (output / "training.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"{arm}: training complete", flush=True)
    if len({row["row_schedule_sha256"] for row in report["arms"].values()}) != 1:
        raise RuntimeError("arms did not receive identical training row schedules")
    # Only now open held-out data. There is no early stopping against its scores.
    if sha256(data_root / "evaluation.pt") != plan["evaluation_sha256"]:
        raise ValueError("evaluation cache hash mismatch")
    evaluation = torch.load(data_root / "evaluation.pt", map_location="cpu", weights_only=True)
    for key in ("subject_identity_sha256", "event_identity_sha256", "schema"):
        if evaluation[key] != data[key]:
            raise ValueError("train/evaluation subject differs")
    scores = {}
    for arm in ARMS:
        if sha256(output / f"{arm}.pt") != report["arms"][arm]["checkpoint_sha256"]:
            raise ValueError("saved checkpoint changed before evaluation")
        detector = load_detector(output / f"{arm}.pt")
        score_start = time.perf_counter()
        values = {key: score(detector, value, deadline) for key, value in
                  {**evaluation["pairs"], **evaluation["benign"]}.items()}
        report["arms"][arm]["scoring_seconds"] = time.perf_counter() - score_start
        report["arms"][arm]["evaluation"] = metrics(values)
        training_positive = score(detector, data["pairs"]["effect"], deadline)
        training_negative = score(detector, data["pairs"]["neutral"], deadline)
        train_comparisons = training_positive[:, None] - training_negative[None, :]
        report["arms"][arm]["training_separation"] = {
            "auc": float(((train_comparisons > 0).float() +
                           0.5 * (train_comparisons == 0).float()).mean()),
            "paired_wins": int((training_positive > training_negative).sum()),
            "pairs": len(training_positive),
        }
        # Reload equality and faithful attribution on a held-out batch are checks,
        # never optimization signals.
        example = HardwareMultimodalBatch(**evaluation["pairs"]["effect"])
        with torch.no_grad():
            direct = detector(example)
            attributed = detector.contributions(example).sum((1, 2)) + detector.head.bias[0]
        if not torch.allclose(direct, attributed, atol=1e-5, rtol=1e-5):
            raise RuntimeError("token contribution does not reconstruct head score")
        contributions = detector.contributions(example).detach()
        scores[arm] = {"scores": values, "effect_token_contributions": contributions}
        print(f"{arm}: evaluation complete", flush=True)
    baseline_values = exposure_scores(evaluation, baseline_path, data["manifest_sha256"])
    report["baseline"] = {"checkpoint_sha256": sha256(baseline_path),
                          "evaluation": metrics(baseline_values)}
    scores["exposure_baseline"] = {"scores": baseline_values}
    torch.save(scores, output / "scores.pt")
    report["scores_sha256"] = sha256(output / "scores.pt")
    report["wall_seconds"] = time.perf_counter() - start
    report["within_cycle_budget"] = report["wall_seconds"] <= max_seconds
    report["limits"] = ["single familiar defect; reused development evidence",
                        "identical recipes across sessions, no unseen input claim",
                        "sampled benign calibration cannot certify FPR 1e-4",
                        "CPU training/inference on retained tensors; no collection rate claim"]
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def evaluate_preserved(data_root: Path, checkpoints: Path, baseline_path: Path,
                       output: Path, plan_hash: str, hashes: dict[str, str],
                       expected_seed: int) -> dict:
    """Score hash-pinned completed arms on Mac after an interrupted host cycle.

    This never resumes an optimizer or modifies the preserved checkpoint tree.
    Missing host training statistics stay missing, rather than reconstructed.
    """
    if platform.system() != "Darwin" or output.exists() or set(hashes) != set(ARMS):
        raise ValueError("use Mac, a fresh output, and all three archived checkpoint hashes")
    if any(output.resolve().is_relative_to(root.resolve()) for root in (data_root, checkpoints)):
        raise ValueError("scoring output must be outside preserved data/checkpoint trees")
    torch.set_num_threads(1)
    start = time.perf_counter()
    deadline = start + 270
    if sha256(data_root / "plan.json") != plan_hash or sha256(baseline_path) != BASELINE_SHA256:
        raise ValueError("preserved plan/baseline hash differs")
    plan = json.loads((data_root / "plan.json").read_text())
    for name in ("train", "evaluation"):
        if sha256(data_root / f"{name}.pt") != plan[f"{name}_sha256"]:
            raise ValueError("preserved tensor cache hash differs")
    configurations = []
    cpu_threads = []
    for arm in ARMS:
        if sha256(checkpoints / f"{arm}.pt") != hashes[arm]:
            raise ValueError("preserved checkpoint hash differs")
        payload = torch.load(checkpoints / f"{arm}.pt", map_location="cpu", weights_only=True)
        if payload["arm"] != arm or payload["seed"] != expected_seed:
            raise ValueError("preserved arm/seed metadata differs")
        configurations.append(payload["config"])
        cpu_threads.append(payload["cpu_threads"])
    if any(config != configurations[0] for config in configurations) or len(set(cpu_threads)) != 1:
        raise ValueError("preserved arm architectures/thread settings differ")
    training = torch.load(data_root / "train.pt", map_location="cpu", weights_only=True)
    evaluation = torch.load(data_root / "evaluation.pt", map_location="cpu", weights_only=True)
    for key in ("subject_identity_sha256", "event_identity_sha256", "schema", "manifest_sha256"):
        if training[key] != plan[key] or evaluation[key] != plan[key]:
            raise ValueError("preserved evaluation identity differs from plan")
    report = {"scope": plan["scope"], "operational_go": False,
              "training_host": "iseeyou (trail-x86)", "evaluation_host": platform.node(),
              "device": "cpu", "cpu_threads": 1, "plan_sha256": plan_hash,
              "training_seed": expected_seed, "training_cpu_threads": cpu_threads[0],
              "matched_schedule_archived": False, "qualified_paired_comparison": False,
              "evaluation_source_sha256": sha256(Path(__file__)),
              "checkpoint_hashes": hashes, "arms": {},
              "limits": ["interrupted host cycle; training statistics not reconstructed",
                         "row schedules not preserved; comparison remains preliminary",
                         "single familiar defect, reused development data",
                         "no operational FPR or new-defect claim"]}
    scores = {}
    for arm in ARMS:
        detector = load_detector(checkpoints / f"{arm}.pt")
        values = {key: score(detector, value, deadline, physical_host=False)
                  for key, value in {**evaluation["pairs"], **evaluation["benign"]}.items()}
        train_values = {key: score(detector, value, deadline, physical_host=False)
                        for key, value in training["pairs"].items()}
        comparisons = train_values["effect"][:, None] - train_values["neutral"][None, :]
        report["arms"][arm] = {"evaluation": metrics(values), "training_separation": {
            "auc": float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean()),
            "paired_wins": int((train_values["effect"] > train_values["neutral"]).sum()),
            "pairs": len(train_values["effect"])}}
        with torch.no_grad():
            example = HardwareMultimodalBatch(**evaluation["pairs"]["effect"])
            contributions = detector.contributions(example)
            if not torch.allclose(detector(example), contributions.sum((1, 2)) + detector.head.bias[0],
                                  atol=1e-5, rtol=1e-5):
                raise RuntimeError("preserved head attribution mismatch")
        scores[arm] = {"scores": values, "effect_token_contributions": contributions}
    baseline_values = exposure_scores(evaluation, baseline_path, plan["manifest_sha256"])
    report["baseline"] = {"checkpoint_sha256": sha256(baseline_path),
                          "evaluation": metrics(baseline_values)}
    scores["exposure_baseline"] = {"scores": baseline_values}
    output.mkdir(parents=True)
    torch.save(scores, output / "scores.pt")
    report["scores_sha256"] = sha256(output / "scores.pt")
    report["wall_seconds"] = time.perf_counter() - start
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("pretrained", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=270)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--score-checkpoints", type=Path)
    parser.add_argument("--checkpoint-sha256", nargs=3, metavar="ARM=HASH")
    args = parser.parse_args()
    if (args.score_checkpoints is None) != (args.checkpoint_sha256 is None):
        parser.error("--score-checkpoints and --checkpoint-sha256 must be provided together")
    try:
        if args.score_checkpoints is not None:
            hashes = dict(value.split("=", 1) for value in (args.checkpoint_sha256 or []))
            report = evaluate_preserved(args.data, args.score_checkpoints, args.baseline,
                                        args.output, args.plan_sha256, hashes, args.seed)
        else:
            report = run(args.data, args.pretrained, args.baseline, args.output,
                         args.seed, args.steps, args.max_seconds, args.plan_sha256, args.threads)
    except (RuntimeError, ValueError, TimeoutError) as error:
        if args.output.is_dir() and not (args.output / "report.json").exists():
            (args.output / "failure.json").write_text(json.dumps({
                "error_type": type(error).__name__, "message": str(error),
                "source_sha256": sha256(Path(__file__)), "operational_go": False,
            }, indent=2) + "\n")
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
