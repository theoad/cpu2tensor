# SPDX-License-Identifier: AGPL-3.0-only
"""Independently audit a remotely stored foundation-corpus release."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Sequence

from cpu2tensor.examples.hardware_foundation_corpus import load_plan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aws(region: str, arguments: Sequence[str]) -> dict[str, object]:
    completed = subprocess.run(
        ("aws", "--region", region, "s3api", *arguments, "--output", "json"),
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return json.loads(completed.stdout or "{}")


def _verify_object(
    *, bucket: str, region: str, key: str, digest: str, size: int,
) -> dict[str, object]:
    head = _aws(region, (
        "head-object", "--bucket", bucket, "--key", key,
        "--checksum-mode", "ENABLED",
    ))
    expected = base64.b64encode(bytes.fromhex(digest)).decode()
    if (head.get("ContentLength") != size or
            head.get("ChecksumSHA256") != expected or
            not isinstance(head.get("VersionId"), str)):
        raise ValueError(f"remote object failed integrity: {key}")
    return {"key": key, "version_id": head["VersionId"], "bytes": size}


def _range_probe(
    *, bucket: str, region: str, shard: dict[str, object],
) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        index_path = root / "index.jsonl"
        _aws(region, (
            "get-object", "--bucket", bucket, "--key", str(shard["index_key"]),
            str(index_path),
        ))
        first = json.loads(index_path.read_text().splitlines()[0])
        offset = int(first["offset"])
        size = int(first["size"])
        member_path = root / "member.pt"
        _aws(region, (
            "get-object", "--bucket", bucket, "--key", str(shard["raw_key"]),
            "--range", f"bytes={offset}-{offset + size - 1}", str(member_path),
        ))
        if _sha256(member_path) != first["sha256"]:
            raise ValueError("S3 byte-range member hash mismatch")
        return {
            "execution_id": first["execution_id"], "offset": offset,
            "bytes": size, "sha256": first["sha256"],
        }


def audit(
    plan_path: Path, release_paths: Sequence[Path], *, minimum_bytes: int,
    minimum_application_pt_bytes: int, bucket: str | None, region: str,
) -> dict[str, object]:
    plan, rows = load_plan(plan_path)
    expected = {row.execution_id: row for row in rows}
    releases = [json.loads(path.read_text()) for path in release_paths]
    if len(releases) < 2:
        raise ValueError("qualification requires at least two session releases")
    corpus_ids = {release.get("corpus_id") for release in releases}
    sessions = {release.get("session") for release in releases}
    if corpus_ids != {plan.get("corpus_id")} or sessions != {"session-a", "session-b"}:
        raise ValueError("release corpus/session identities do not match the plan")
    plan_digest = _sha256(plan_path)
    subjects = set()
    observed: dict[str, dict[str, object]] = {}
    shards: list[dict[str, object]] = []
    total_bytes = 0
    for release in releases:
        if release.get("schema") != "cpu2tensor-hardware-foundation-release-v1":
            raise ValueError("unknown release schema")
        manifest = release.get("capture_manifest")
        if not isinstance(manifest, dict):
            raise ValueError("release omitted its capture manifest")
        if manifest.get("plan_sha256") != plan_digest:
            raise ValueError("release plan hash mismatch")
        if manifest.get("rejections") != []:
            raise ValueError("release contains rejected executions")
        if manifest.get("stopped_at_byte_target"):
            raise ValueError("partial byte-target capture cannot qualify")
        if manifest.get("executions") != manifest.get("planned_executions"):
            raise ValueError("release did not execute its complete session plan")
        subject = manifest.get("subject", {}).get("identity_sha256")
        if not isinstance(subject, str):
            raise ValueError("release omitted exact subject identity")
        subjects.add(subject)
        entries = manifest.get("entries")
        if not isinstance(entries, list) or len(entries) != manifest.get("executions"):
            raise ValueError("release entry count is inconsistent")
        for entry in entries:
            execution_id = entry.get("execution_id")
            if not isinstance(execution_id, str) or execution_id in observed:
                raise ValueError(f"duplicate or invalid execution ID: {execution_id}")
            if execution_id not in expected:
                raise ValueError(f"execution is absent from the plan: {execution_id}")
            row = expected[execution_id]
            for name in ("family", "application", "partition", "session"):
                if entry.get(name) != getattr(row, name):
                    raise ValueError(f"entry metadata mismatch: {execution_id}/{name}")
            observed[execution_id] = entry
        release_shards = release.get("shards")
        if not isinstance(release_shards, list):
            raise ValueError("release shard list is invalid")
        manifest_digests = {item["sha256"] for item in manifest.get("shards", [])}
        release_digests = {item.get("sha256") for item in release_shards}
        if manifest_digests != release_digests:
            raise ValueError("release shards differ from capture manifest")
        shards.extend(release_shards)
        total_bytes += sum(int(item["bytes"]) for item in release_shards)
    if len(subjects) != 1:
        raise ValueError("sessions were captured from different subjects")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise ValueError(f"release omitted planned executions: {missing[:3]}")
    if total_bytes < minimum_bytes:
        raise ValueError(f"raw release is too small: {total_bytes} < {minimum_bytes}")
    pt_by_application: dict[str, int] = defaultdict(int)
    pebs_by_application: dict[str, int] = defaultdict(int)
    partition_counts: Counter[str] = Counter()
    for execution_id, entry in observed.items():
        application = expected[execution_id].application
        pt_by_application[application] += int(entry["pt_bytes"])
        pebs_by_application[application] += int(entry["pebs_samples"])
        partition_counts[expected[execution_id].partition] += 1
    underweight = {
        application: value for application, value in pt_by_application.items()
        if value < minimum_application_pt_bytes
    }
    if underweight:
        raise ValueError(f"applications missed the PT quota: {underweight}")
    remote = []
    probes = []
    if bucket is not None:
        for shard in shards:
            remote.append(_verify_object(
                bucket=bucket, region=region, key=str(shard["raw_key"]),
                digest=str(shard["sha256"]), size=int(shard["bytes"]),
            ))
            remote.append(_verify_object(
                bucket=bucket, region=region, key=str(shard["index_key"]),
                digest=str(shard["index_sha256"]),
                size=int(_aws(region, (
                    "head-object", "--bucket", bucket,
                    "--key", str(shard["index_key"]),
                ))["ContentLength"]),
            ))
        probes = [
            _range_probe(bucket=bucket, region=region, shard=release["shards"][0])
            for release in releases
        ]
    return {
        "schema": "cpu2tensor-hardware-foundation-audit-v1",
        "accepted": True,
        "corpus_id": next(iter(corpus_ids)),
        "plan_sha256": plan_digest,
        "subject_identity_sha256": next(iter(subjects)),
        "sessions": sorted(sessions),
        "executions": len(observed),
        "raw_bytes": total_bytes,
        "applications": len(pt_by_application),
        "partition_counts": dict(sorted(partition_counts.items())),
        "application_pt_bytes": dict(sorted(pt_by_application.items())),
        "application_pebs_samples": dict(sorted(pebs_by_application.items())),
        "remote_objects_verified": len(remote),
        "range_probes": probes,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("plan", type=Path)
    result.add_argument("releases", nargs="+", type=Path)
    result.add_argument("--minimum-bytes", type=int, default=8 * 1024 ** 3)
    result.add_argument(
        "--minimum-application-pt-bytes", type=int, default=180 * 1024 ** 2,
    )
    result.add_argument("--bucket")
    result.add_argument("--region", default="us-east-1")
    return result


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(audit(
        args.plan.resolve(), tuple(path.resolve() for path in args.releases),
        minimum_bytes=args.minimum_bytes,
        minimum_application_pt_bytes=args.minimum_application_pt_bytes,
        bucket=args.bucket, region=args.region,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
