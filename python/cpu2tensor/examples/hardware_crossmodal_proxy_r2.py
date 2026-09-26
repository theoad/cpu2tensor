# SPDX-License-Identifier: AGPL-3.0-only
"""Two-stage, development-only cross-modal splice diagnostic (never a bug gate)."""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import torch

from cpu2tensor.examples.hardware_autoresearch_eval import (
    CHECKPOINT_SCHEMA, CHECKPOINT_SCHEMA_V2, EXPOSURE_FIELDS,
    SEALED_V2_EVAL_SHA256, SEALED_V2_TRAIN_SHA256, calibrate_threshold,
    sha256, validate_eval_cache,
)
from cpu2tensor.examples.hardware_autoresearch_train import load_frozen, score_features
from cpu2tensor.examples.hardware_review_budget_r1 import exact_binomial_upper_bound


SCHEMA = "cpu2tensor-crossmodal-proxy-pairs-r2"
REPORT_SCHEMA = "cpu2tensor-crossmodal-proxy-report-r2"
SEALED_V1_EVAL_SHA256 = "378bbdc43adae83fbc539e50a7d03e04818eb0bff88e8b8112c588809b54b3a8"
SEALED_MANIFEST_SHA256 = "0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d"
MAX_POSITION_GAP = 5_000
CALIPERS = (math.log(1.10), math.log(1.10), math.log(1.20))
FPRS = (1e-3, 1e-4)


def _load_eval(path: Path, expected: str) -> dict[str, object]:
    if sha256(path) != expected:
        raise ValueError("development cache hash differs from locked SHA-256")
    return validate_eval_cache(torch.load(path, map_location="cpu", weights_only=True))


def _hash_id(value: str) -> str:
    return hashlib.sha256(("crossmodal-proxy-r2:" + value).encode()).hexdigest()


def _distance(exposure: list[list[int]], a: int, b: int) -> float | None:
    first = exposure[a]
    second = exposure[b]
    differences = [
        abs(math.log(first[0] / second[0])),
        abs(math.log(first[1] / second[1])),
        abs(math.log((1 + first[2]) / (1 + second[2]))),
    ]
    if any(value > limit for value, limit in zip(differences, CALIPERS)):
        return None
    return sum((value / limit) ** 2 for value, limit in zip(differences, CALIPERS))


def make_pairs(eval_cache: Path, manifest_path: Path, output: Path, *,
               expected_manifest_entries: int = 102_000) -> dict[str, object]:
    """Commit a label- and score-blind pair list before loading any checkpoint."""
    if output.exists():
        raise FileExistsError(output)
    payload = _load_eval(eval_cache, SEALED_V2_EVAL_SHA256)
    if payload["schema"] != "cpu2tensor-autoresearch-eval-v2":
        raise ValueError("pairing requires the sealed v2 exposure cache")
    if sha256(manifest_path) != SEALED_MANIFEST_SHA256 or \
            payload["source_manifest_sha256"] != SEALED_MANIFEST_SHA256:
        raise ValueError("source manifest hash differs from the sealed manifest")
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_manifest_entries:
        raise ValueError("source manifest has incomplete entries")
    position = {entry["execution_id"]: index for index, entry in enumerate(entries)}
    ids = payload["execution_ids"]
    if len(position) != len(entries) or any(value not in position for value in ids):
        raise ValueError("manifest/evaluation execution IDs do not join uniquely")
    group: dict[tuple[int, str, int], list[int]] = {}
    part = payload["partition"].tolist()
    for row, execution_id in enumerate(ids):
        entry = entries[position[execution_id]]
        if (entry.get("partition") not in ("familiar_validation", "heldout_family")
                and part[row] in (2, 3)) or \
                entry.get("family") != payload["families"][row] or \
                entry.get("loops") != payload["loops"][row]:
            raise ValueError("manifest/evaluation row metadata disagree")
        if part[row] not in (2, 3):
            continue
        capture = entry.get("capture", {})
        lanes = entry.get("lanes", [])
        if (entry.get("admission", {}).get("attempt") != 1 or
                any(capture.get(key) != 0 for key in
                    ("lost_sources", "missing_sources", "multiplexed_sources")) or
                not lanes or not all(lane.get("migration_verified") for lane in lanes)):
            continue
        key = (part[row], payload["families"][row], payload["loops"][row])
        group.setdefault(key, []).append(row)
    exposure = payload["exposure_raw"].tolist()
    pairs: list[dict[str, object]] = []
    coverage: dict[str, dict[str, int]] = {}
    for key in sorted(group):
        rows = group[key]
        ordered = sorted(rows, key=lambda row: position[ids[row]])
        positions = [position[ids[row]] for row in ordered]
        unused = set(rows)
        for anchor in sorted(rows, key=lambda row: _hash_id(ids[row])):
            if anchor not in unused:
                continue
            center = position[ids[anchor]]
            low = bisect_left(positions, center - MAX_POSITION_GAP)
            high = bisect_right(positions, center + MAX_POSITION_GAP)
            options = []
            for donor in ordered[low:high]:
                if donor == anchor or donor not in unused:
                    continue
                distance = _distance(exposure, anchor, donor)
                if distance is not None:
                    options.append((distance, _hash_id(ids[donor]), donor))
            if not options:
                continue
            _, _, donor = min(options)
            unused.remove(anchor)
            unused.remove(donor)
            pairs.append({"anchor": ids[anchor], "donor": ids[donor],
                          "partition": key[0], "family": key[1], "loops": key[2]})
        coverage["/".join(map(str, key))] = {
            "admitted_rows": len(rows), "paired_rows": len(rows) - len(unused),
            "unpaired_rows": len(unused),
        }
    artifact = {
        "schema": SCHEMA, "development_eval_sha256": SEALED_V2_EVAL_SHA256,
        "source_manifest_sha256": SEALED_MANIFEST_SHA256,
        "pairing": {"max_manifest_position_gap": MAX_POSITION_GAP,
                    "log_exposure_calipers": CALIPERS,
                    "without_replacement": True,
                    "selection": "minimum exposure distance; SHA-256 execution-ID tie break"},
        "coverage": coverage, "pairs": pairs,
    }
    output.write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n")
    return artifact


def _summary(scores: dict[str, torch.Tensor], threshold: float,
             selected: torch.Tensor) -> dict[str, object]:
    count = int(selected.sum())
    if count == 0:
        return {"pairs": 0}
    result: dict[str, object] = {"pairs": count}
    for name, values in scores.items():
        subset = values[selected]
        result[name + "_alerts"] = int((subset > threshold).sum())
        result[name + "_alert_rate"] = result[name + "_alerts"] / count
    hybrid = scores["hybrid"][selected]
    original_max = torch.maximum(scores["anchor"][selected], scores["donor"][selected])
    delta = hybrid - original_max
    result["hybrid_vs_original_max_win_rate"] = float((delta > 0).float().mean())
    result["hybrid_vs_original_max_tie_rate"] = float((delta == 0).float().mean())
    result["hybrid_vs_original_max_delta_median"] = float(delta.median())
    result["hybrid_vs_original_max_delta_q25"] = float(torch.quantile(delta, 0.25))
    result["hybrid_vs_original_max_delta_q75"] = float(torch.quantile(delta, 0.75))
    result["only_hybrid_alerts"] = int(((hybrid > threshold) &
        (scores["anchor"][selected] <= threshold) &
        (scores["donor"][selected] <= threshold) &
        (scores["metadata_only"][selected] <= threshold)).sum())
    return result


def _benign_counts(originals: torch.Tensor, payload: dict[str, object],
                   threshold: float) -> dict[str, object]:
    """Full unmodified validation ledger, including rows omitted by pairing."""
    partition = payload["partition"].tolist()
    families = payload["families"]
    loops = payload["loops"]
    result: dict[str, object] = {}
    for code, name in ((2, "familiar_validation"), (3, "heldout_family")):
        selected = [index for index, part in enumerate(partition) if part == code]

        def counts(indices: list[int]) -> dict[str, float | int]:
            alerts = sum(float(originals[index]) > threshold for index in indices)
            return {"rows": len(indices), "alerts": alerts,
                    "alert_rate": alerts / len(indices),
                    "one_sided_95pct_upper": exact_binomial_upper_bound(
                        alerts, len(indices))}

        by_family = {}
        for family in sorted(set(families[index] for index in selected)):
            family_rows = [index for index in selected if families[index] == family]
            by_family[family] = {
                "all": counts(family_rows),
                "by_intensity_loops": {
                    str(intensity): counts([index for index in family_rows
                                            if loops[index] == intensity])
                    for intensity in sorted(set(loops[index] for index in family_rows))},
            }
        result[name] = {"all": counts(selected), "by_family": by_family}
    return result


def score_pairs(eval_v1: Path, eval_v2: Path, pairs_path: Path, pairs_sha256: str,
                checkpoints_v1: Sequence[Path], checkpoints_v2: Sequence[Path],
                output: Path) -> dict[str, object]:
    """Score immutable originals and prespecified splices on CPU only."""
    if output.exists():
        raise FileExistsError(output)
    if sha256(pairs_path) != pairs_sha256:
        raise ValueError("pair-list SHA-256 differs from frozen digest")
    pair_artifact = json.loads(pairs_path.read_text())
    if (pair_artifact.get("schema") != SCHEMA or
            pair_artifact.get("development_eval_sha256") != SEALED_V2_EVAL_SHA256 or
            pair_artifact.get("source_manifest_sha256") != SEALED_MANIFEST_SHA256):
        raise ValueError("pair-list identity is invalid")
    v1 = _load_eval(eval_v1, SEALED_V1_EVAL_SHA256)
    v2 = _load_eval(eval_v2, SEALED_V2_EVAL_SHA256)
    if (v1["schema"] != "cpu2tensor-autoresearch-eval-v1" or
            v2["schema"] != "cpu2tensor-autoresearch-eval-v2" or
            v1["execution_ids"] != v2["execution_ids"] or
            v1["families"] != v2["families"] or
            v1["loops"] != v2["loops"] or
            v1["source_manifest_sha256"] != v2["source_manifest_sha256"] or
            v1["source_cache_sha256"] != v2["source_cache_sha256"] or
            not torch.equal(v1["features"], v2["features"]) or
            not torch.equal(v1["partition"], v2["partition"])):
        raise ValueError("v1/v2 development rows do not align exactly")
    if len(checkpoints_v1) != 2 or len(checkpoints_v2) != 2:
        raise ValueError("require exactly two frozen checkpoints for each version")
    ids = v2["execution_ids"]
    row = {value: index for index, value in enumerate(ids)}
    pairs = pair_artifact["pairs"]
    if not pairs:
        raise ValueError("frozen pair list is empty")
    if any(pair["anchor"] not in row or pair["donor"] not in row for pair in pairs):
        raise ValueError("pair-list execution ID is absent from development rows")
    a = torch.tensor([row[pair["anchor"]] for pair in pairs], dtype=torch.int64)
    b = torch.tensor([row[pair["donor"]] for pair in pairs], dtype=torch.int64)
    if len(set(a.tolist() + b.tolist())) != 2 * len(pairs):
        raise ValueError("frozen pairs reuse an execution")
    for index, pair in enumerate(pairs):
        if (int(v2["partition"][a[index]]) != pair["partition"] or
                int(v2["partition"][b[index]]) != pair["partition"] or
                v2["families"][a[index]] != pair["family"] or
                v2["families"][b[index]] != pair["family"] or
                v2["loops"][a[index]] != pair["loops"] or
                v2["loops"][b[index]] != pair["loops"]):
            raise ValueError("pair-list stratum mismatch")
    features = v2["features"]
    exposure = v2["exposure_raw"]
    anchor = features[a]
    donor = features[b]
    hybrid = anchor.clone()
    hybrid[:, 512:789] = donor[:, 512:789]
    pebs_only = anchor.clone()
    pebs_only[:, 512:784] = donor[:, 512:784]
    pebs_only[:, 788] = donor[:, 788]
    pmu_only = anchor.clone()
    pmu_only[:, 784:788] = donor[:, 784:788]
    hybrid_exposure = exposure[a].clone()
    hybrid_exposure[:, 2] = exposure[b, 2]
    reports = []
    score_artifact: dict[str, object] = {"schema": "cpu2tensor-crossmodal-proxy-scores-r2",
                                         "pairs_sha256": pairs_sha256, "checkpoints": {}}
    for version, paths in ((1, checkpoints_v1), (2, checkpoints_v2)):
        expected_identity: dict[str, object] = {
            "source_manifest_sha256": v2["source_manifest_sha256"],
            "source_cache_sha256": v2["source_cache_sha256"],
        }
        if version == 2:
            expected_identity.update({"exposure_fields": EXPOSURE_FIELDS,
                                      "train_cache_sha256": SEALED_V2_TRAIN_SHA256})
        for path in paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if (checkpoint.get("schema") !=
                    (CHECKPOINT_SCHEMA_V2 if version == 2 else CHECKPOINT_SCHEMA) or
                    checkpoint.get("train_cache_identity") != expected_identity):
                raise ValueError("checkpoint provenance/version differs from sealed cache")
            scorer = load_frozen(path, device="cpu")
            def run(values: torch.Tensor, raw: torch.Tensor | None = None) -> torch.Tensor:
                result = (score_features(scorer, values, exposure_raw=raw)
                          if version == 2 else score_features(scorer, values))
                if (result.dtype != torch.float32 or result.shape != (len(values),) or
                        not bool(torch.isfinite(result).all())):
                    raise ValueError("nonfinite or incomplete checkpoint scores")
                return result
            originals = run(features, exposure if version == 2 else None)
            scores = {
                "anchor": originals[a], "donor": originals[b],
                "hybrid": run(hybrid, hybrid_exposure),
                "sham": run(anchor, exposure[a] if version == 2 else None),
                "metadata_only": run(anchor, hybrid_exposure) if version == 2 else originals[a].clone(),
                "pebs_only": run(pebs_only, hybrid_exposure),
                "pmu_only": run(pmu_only, exposure[a]),
            }
            seed = checkpoint["seed"]
            key = f"v{version}-seed-{seed}"
            if key in score_artifact["checkpoints"]:
                raise ValueError("duplicate checkpoint seed")
            score_artifact["checkpoints"][key] = {"originals": originals, **scores}
            levels: dict[str, object] = {}
            all_pairs = torch.ones(len(pairs), dtype=torch.bool)
            for fpr in FPRS:
                threshold, calibration_allowed = calibrate_threshold(
                    originals[v2["partition"] == 1], target_fpr=fpr)
                by_family: dict[str, object] = {}
                for family in sorted(set(pair["family"] for pair in pairs)):
                    family_mask = torch.tensor([pair["family"] == family for pair in pairs])
                    by_family[family] = {
                        "all": _summary(scores, threshold, family_mask),
                        "by_intensity_loops": {
                            str(loops): _summary(scores, threshold, family_mask & torch.tensor(
                                [pair["loops"] == loops for pair in pairs]))
                            for loops in sorted(set(pair["loops"] for pair in pairs
                                                    if pair["family"] == family))},
                    }
                pool = torch.cat((originals[v2["partition"] >= 2], scores["hybrid"]))
                top_count = min(100, len(pool))
                top_positions = torch.topk(pool, top_count).indices
                levels[str(fpr)] = {
                    "threshold": threshold, "calibration_allowed_alerts": calibration_allowed,
                    "calibration_alerts": int((originals[v2["partition"] == 1] > threshold).sum()),
                    "familiar_original_alerts": int((originals[v2["partition"] == 2] > threshold).sum()),
                    "heldout_original_alerts": int((originals[v2["partition"] == 3] > threshold).sum()),
                    "unmodified_benign": _benign_counts(originals, v2, threshold),
                    "overall": _summary(scores, threshold, all_pairs),
                    "by_family": by_family,
                    "retrospective_top100_hybrids": int((top_positions >=
                        int((v2["partition"] >= 2).sum())).sum()),
                }
            reports.append({"version": version, "seed": seed,
                            "checkpoint": str(path), "checkpoint_sha256": sha256(path),
                            "levels": levels})
    output.mkdir(parents=True)
    scores_path = output / "scores.pt"
    torch.save(score_artifact, scores_path)
    report = {
        "schema": REPORT_SCHEMA, "status": "development_synthetic_proxy_not_real_bug_gate",
        "keep_discard_authorized": False, "device": "cpu",
        "development_eval_v1_sha256": SEALED_V1_EVAL_SHA256,
        "development_eval_v2_sha256": SEALED_V2_EVAL_SHA256,
        "source_manifest_sha256": SEALED_MANIFEST_SHA256,
        "pairs_sha256": pairs_sha256, "pairs": len(pairs),
        "coverage": pair_artifact["coverage"],
        "scores_file": scores_path.name, "scores_sha256": sha256(scores_path),
        "checkpoints": reports,
        "limitations": ["synthetic feature splice only", "one collection/no session holdout",
                        "compact PT/PEBS order discarded", "not real-bug recall"],
    }
    (output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("pair")
    pair.add_argument("--eval-v2", required=True, type=Path)
    pair.add_argument("--manifest", required=True, type=Path)
    pair.add_argument("--output", required=True, type=Path)
    score = sub.add_parser("score")
    score.add_argument("--eval-v1", required=True, type=Path)
    score.add_argument("--eval-v2", required=True, type=Path)
    score.add_argument("--pairs", required=True, type=Path)
    score.add_argument("--pairs-sha256", required=True)
    score.add_argument("--checkpoint-v1", required=True, action="append", type=Path)
    score.add_argument("--checkpoint-v2", required=True, action="append", type=Path)
    score.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "pair":
        artifact = make_pairs(args.eval_v2, args.manifest, args.output)
        print(json.dumps({"pairs": len(artifact["pairs"]), "sha256": sha256(args.output)}))
    else:
        report = score_pairs(args.eval_v1, args.eval_v2, args.pairs,
                             args.pairs_sha256, args.checkpoint_v1,
                             args.checkpoint_v2, args.output)
        print(json.dumps({"status": report["status"], "pairs": report["pairs"],
                          "output": str(args.output)}))


if __name__ == "__main__":
    main()
