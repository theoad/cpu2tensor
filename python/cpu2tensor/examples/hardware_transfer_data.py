# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare separate training/evaluation caches from retained kernel captures.

This imports existing evidence only. It never executes an input or invokes perf.
The first experiment uses two already inspected sessions of the same defect;
its evaluation is a development replay, not an unseen-defect benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import random
import re

import torch

from cpu2tensor.examples.hardware_multimodal import HardwareMultimodalBatch
from cpu2tensor.examples.hardware_anomaly_model_r1 import compact_features


SCHEMA = "cpu2tensor-hardware-transfer-data-v1"
MANIFEST_HASH = "0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d"
CANARY_REPORTS = {
    "fca22177df0926f1a14a6caf34f3f2bbb865bc431b6acfec3fd050b47fdcfc79",
    "1c2a91d54f61c20c254efb456fbbe1d74c9fc3c51b0d616f365d1200a30b2941",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
        raise ValueError("retained artifact path/hash differs from its report")
    return path


def combine(payloads: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not payloads:
        raise ValueError("empty multimodal partition")
    result = {field.name: torch.cat([row[field.name] for row in payloads])
              for field in fields(HardwareMultimodalBatch)}
    HardwareMultimodalBatch(**result)
    return result


def balanced_rows(entries: list[dict], partition: str, count: int) -> list[dict]:
    """Sample using declared strata, before any model or score is loaded."""
    groups: dict[tuple, list] = defaultdict(list)
    generator = random.Random(20260926)
    for entry in entries:
        if entry["partition"] == partition:
            groups[(entry["family"], entry["loops"])].append(entry)
    for group in groups.values():
        generator.shuffle(group)
    selected = []
    while len(selected) < count:
        progress = False
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progress = True
        if not progress:
            raise ValueError("insufficient rows in requested partition")
    return selected


def baseline_row(batch: dict, counts: dict, elapsed_ns: int) -> tuple[torch.Tensor, torch.Tensor]:
    return compact_features({"batch": batch}), torch.log1p(torch.tensor(
        [counts["pt_bytes"], elapsed_ns / 1_000_000.0, counts["pebs_samples"]],
        dtype=torch.float64,
    ))


def load_pairs(root: Path, subject: str, event: str) -> tuple[dict, dict, dict]:
    report_hash = sha256(root / "report.json")
    if report_hash not in CANARY_REPORTS:
        raise ValueError("report is outside the two previously audited canary sessions")
    report = json.loads((root / "report.json").read_text())
    canonical = dict(report)
    content_hash = canonical.pop("content_sha256")
    if hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != content_hash:
        raise ValueError("canary report content hash mismatch")
    identity = report["subject"]
    if identity["subject_identity_sha256"] != subject or identity["event_identity_sha256"] != event:
        raise ValueError("canary/pretraining subject or capture events differ")
    loops = report["protocol"]["loops"]
    by_pair: dict[int, dict] = defaultdict(dict)
    ledger = []
    baseline_by_pair: dict[int, dict] = defaultdict(dict)
    seen_raw = set()
    time_intervals = []
    for row in report["rows"]:
        arm = row["arm"]
        if (arm not in ("effect", "neutral") or row["attempt"] != 1 or
                row["mutations"] != (loops if arm == "effect" else 0) or
                any(row["capture"][key] for key in
                    ("lost_sources", "missing_sources", "multiplexed_sources"))):
            raise ValueError("canary row does not meet the matched effect contract")
        if arm in by_pair[row["anonymous_pair"]] or row["raw_sha256"] in seen_raw:
            raise ValueError("duplicate canary arm/raw artifact")
        seen_raw.add(row["raw_sha256"])
        raw_path = checked_path(root, row["raw_path"], row["raw_sha256"])
        evidence_path = checked_path(root, row["evidence_path"], row["evidence_sha256"])
        raw = torch.load(raw_path, map_location="cpu", weights_only=True)
        evidence = torch.load(evidence_path, map_location="cpu", weights_only=True)
        expected = {"execution_id": f"{arm}-{row['anonymous_pair']:05d}",
                    "family": arm, "repetition": row["anonymous_pair"],
                    "partition": "known_cve_validation"}
        if (raw.get("schema") != "cpu2tensor-kernel-multimodal-raw-v3" or
                raw["execution"] != expected or evidence["execution"] != expected or
                evidence["raw_sha256"] != row["raw_sha256"] or raw["loops"] != loops or
                bytes(raw["stdout"].tolist()).decode("ascii").strip() != row["output"]):
            raise ValueError("canary raw/evidence identity or oracle mismatch")
        output_match = re.fullmatch(r"mode=(effect|neutral) loops=(\d+) mutations=(\d+)",
                                   bytes(raw["stdout"].tolist()).decode("ascii").strip())
        if output_match is None or output_match.groups() != (
                arm, str(loops), str(loops if arm == "effect" else 0)):
            raise ValueError("raw semantic oracle differs from training labels")
        for batch in raw["batches"]:
            envelope = batch["envelope"]
            if envelope["clock"] != "CLOCK_MONOTONIC_RAW":
                raise ValueError("session timestamps do not share a qualified clock")
            time_intervals.append((envelope["arm_before_ns"], envelope["stop_after_ns"]))
        by_pair[row["anonymous_pair"]][arm] = evidence["batch"]
        baseline_by_pair[row["anonymous_pair"]][arm] = baseline_row(
            evidence["batch"], row["capture"], raw["elapsed_ns"],
        )
        ledger.append({key: row[key] for key in
                       ("anonymous_pair", "arm", "raw_sha256", "evidence_sha256")})
    if len(by_pair) != 12 or any(set(pair) != {"effect", "neutral"} for pair in by_pair.values()):
        raise ValueError("expected twelve complete matched pairs")
    pairs = {arm: combine([by_pair[key][arm] for key in sorted(by_pair)])
             for arm in ("effect", "neutral")}
    baseline = {arm: {
        "features": torch.stack([baseline_by_pair[key][arm][0] for key in sorted(by_pair)]),
        "exposure": torch.stack([baseline_by_pair[key][arm][1] for key in sorted(by_pair)]),
    } for arm in ("effect", "neutral")}
    return pairs, {"report_sha256": report_hash, "rows": ledger,
                   "clock": "CLOCK_MONOTONIC_RAW",
                   "start_ns": min(interval[0] for interval in time_intervals),
                   "end_ns": max(interval[1] for interval in time_intervals)}, baseline


def prepare(corpus: Path, train_session: Path, test_session: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("use a fresh output directory")
    torch.set_num_threads(1)
    manifest_path = corpus / "capture-manifest.json"
    if sha256(manifest_path) != MANIFEST_HASH:
        raise ValueError("unexpected pretraining corpus")
    manifest = json.loads(manifest_path.read_text())
    subject, event = (manifest[key] for key in
                      ("subject_identity_sha256", "event_identity_sha256"))
    train_pairs, train_ledger, _ = load_pairs(train_session, subject, event)
    test_pairs, test_ledger, baseline_pairs = load_pairs(test_session, subject, event)
    if ({row["raw_sha256"] for row in train_ledger["rows"]} &
            {row["raw_sha256"] for row in test_ledger["rows"]}):
        raise ValueError("train/test sessions share raw executions")
    if not (train_ledger["end_ns"] < test_ledger["start_ns"] or
            test_ledger["end_ns"] < train_ledger["start_ns"]):
        raise ValueError("the audited capture session time ranges overlap")
    data, ledgers, baseline_benign = {}, {}, {}
    for partition, count in (("training", 256), ("calibration", 1024),
                             ("familiar_validation", 512), ("heldout_family", 512)):
        rows = balanced_rows(manifest["entries"], partition, count)
        payloads = []
        baseline_rows = []
        for row in rows:
            path = checked_path(corpus, row["derived_path"], row["derived_sha256"])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (payload["execution"]["execution_id"] != row["execution_id"] or
                    payload["raw_sha256"] != row["raw_sha256"]):
                raise ValueError("benign derived identity mismatch")
            payloads.append(payload["batch"])
            baseline_rows.append(baseline_row(payload["batch"], row["capture"], row["elapsed_ns"]))
        data[partition] = combine(payloads)
        baseline_benign[partition] = {
            "features": torch.stack([row[0] for row in baseline_rows]),
            "exposure": torch.stack([row[1] for row in baseline_rows]),
        }
        ledgers[partition] = [{key: row[key] for key in
                              ("execution_id", "family", "loops", "derived_sha256")}
                             for row in rows]
    output.mkdir(parents=True)
    common = {"schema": SCHEMA, "subject_identity_sha256": subject,
              "event_identity_sha256": event, "manifest_sha256": MANIFEST_HASH}
    torch.save({**common, "pairs": train_pairs, "benign": data["training"]}, output / "train.pt")
    torch.save({**common, "pairs": test_pairs,
                "baseline": {**baseline_pairs, **{key: value for key, value in
                                                  baseline_benign.items() if key != "training"}},
                "benign": {key: value for key, value in data.items() if key != "training"}},
               output / "evaluation.pt")
    plan = {**common, "scope": "previously inspected single-defect development replay",
            "unseen_defect_evaluation": False, "independent_input_evaluation": False,
            "session_holdout": True, "train_session": train_ledger,
            "evaluation_session": test_ledger, "benign_rows": ledgers,
            "train_sha256": sha256(output / "train.pt"),
            "evaluation_sha256": sha256(output / "evaluation.pt")}
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    plan["plan_file_sha256"] = sha256(output / "plan.json")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("train_session", type=Path)
    parser.add_argument("test_session", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = prepare(args.corpus, args.train_session, args.test_session, args.output)
    print(json.dumps({key: plan[key] for key in
                      ("scope", "train_sha256", "evaluation_sha256", "plan_file_sha256")}, indent=2))


if __name__ == "__main__":
    main()
