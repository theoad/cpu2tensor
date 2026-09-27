# SPDX-License-Identifier: AGPL-3.0-only
"""Verify and publish foundation-corpus objects to an immutable S3 layout."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_base64(digest: str) -> str:
    return base64.b64encode(bytes.fromhex(digest)).decode()


class S3CorpusStore:
    """Small verified writer; trainers need only ordinary S3 range reads."""

    def __init__(self, bucket: str, *, region: str = "us-east-1") -> None:
        self.bucket = bucket
        self.region = region

    def _aws(self, arguments: Sequence[str]) -> dict[str, object]:
        command = (
            "aws", "--region", self.region, "s3api", *arguments,
            "--output", "json",
        )
        completed = subprocess.run(
            command, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        return json.loads(completed.stdout or "{}")

    def _head(self, key: str) -> dict[str, object] | None:
        try:
            return self._aws((
                "head-object", "--bucket", self.bucket, "--key", key,
                "--checksum-mode", "ENABLED",
            ))
        except subprocess.CalledProcessError as error:
            if "Not Found" in error.stderr or "404" in error.stderr:
                return None
            raise

    def put_verified(self, path: Path, key: str, digest: str | None = None) -> None:
        digest = sha256(path) if digest is None else digest
        if sha256(path) != digest:
            raise ValueError(f"local checksum changed: {path}")
        expected_checksum = checksum_base64(digest)
        head = self._head(key)
        if head is None:
            self._aws((
                "put-object", "--bucket", self.bucket, "--key", key,
                "--body", str(path), "--checksum-algorithm", "SHA256",
                "--metadata", f"sha256={digest}",
            ))
            head = self._head(key)
        if head is None:
            raise RuntimeError(f"S3 object disappeared after upload: {key}")
        if (head.get("ContentLength") != path.stat().st_size or
                head.get("ChecksumSHA256") != expected_checksum or
                not isinstance(head.get("VersionId"), str)):
            raise RuntimeError(f"S3 verification failed: {key}")

    def upload_shard(self, descriptor_path: Path) -> dict[str, object]:
        descriptor = json.loads(descriptor_path.read_text())
        digest = descriptor.get("sha256")
        index_digest = descriptor.get("index_sha256")
        if not isinstance(digest, str) or not isinstance(index_digest, str):
            raise ValueError("shard descriptor omitted checksums")
        root = descriptor_path.parent
        tar_path = root / str(descriptor["tar"])
        index_path = root / str(descriptor["index"])
        if (sha256(tar_path) != digest or sha256(index_path) != index_digest or
                tar_path.stat().st_size != descriptor.get("bytes")):
            raise ValueError("shard bundle failed local verification")
        self.put_verified(
            tar_path, f"objects/raw/sha256/{digest[:2]}/{digest}.tar", digest,
        )
        self.put_verified(
            index_path, f"objects/index/sha256/{index_digest[:2]}/{index_digest}.jsonl",
            index_digest,
        )
        descriptor_digest = sha256(descriptor_path)
        self.put_verified(
            descriptor_path,
            f"objects/descriptor/sha256/{descriptor_digest[:2]}/{descriptor_digest}.json",
            descriptor_digest,
        )
        return {
            **descriptor,
            "descriptor_sha256": descriptor_digest,
            "raw_key": f"objects/raw/sha256/{digest[:2]}/{digest}.tar",
            "index_key": (
                f"objects/index/sha256/{index_digest[:2]}/{index_digest}.jsonl"
            ),
        }

    def upload_decode_state(self, path: Path) -> dict[str, object]:
        digest = sha256(path)
        key = f"objects/decode/sha256/{digest[:2]}/{digest}.pt"
        self.put_verified(path, key, digest)
        return {"path": path.name, "sha256": digest, "bytes": path.stat().st_size,
                "key": key}

    def upload_custody(self, artifact: Path) -> dict[str, object]:
        root = artifact / "custody"
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        def upload_item(item: dict[str, object]) -> dict[str, object]:
            digest = item["sha256"]
            path = root / item["path"]
            key = f"objects/custody/sha256/{digest[:2]}/{digest}"
            self.put_verified(path, key, digest)
            return {**item, "key": key}

        items = manifest.get("unique_objects", [])
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError("custody manifest objects are invalid")
        with ThreadPoolExecutor(max_workers=8) as pool:
            uploaded = list(pool.map(upload_item, items))
        for name, kind in (("plan", "plan"), ("package_inventory", "packages")):
            item = manifest[name]
            digest = item["sha256"]
            path = root / item["path"]
            key = f"objects/{kind}/sha256/{digest[:2]}/{digest}"
            self.put_verified(path, key, digest)
            item["key"] = key
        digest = sha256(manifest_path)
        key = f"objects/custody-manifest/sha256/{digest[:2]}/{digest}.json"
        self.put_verified(manifest_path, key, digest)
        return {"manifest_sha256": digest, "manifest_key": key,
                "objects": uploaded, **manifest}

    def publish_release(
        self, artifact: Path, uploaded_shards: Sequence[dict[str, object]],
        uploaded_decode: Sequence[dict[str, object]], uploaded_custody: dict[str, object],
    ) -> dict[str, object]:
        manifest_path = artifact / "capture-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        corpus_id = manifest.get("corpus_id")
        session = manifest.get("session")
        if not isinstance(corpus_id, str) or not isinstance(session, str):
            raise ValueError("manifest has no corpus/session identity")
        expected = {item["sha256"] for item in manifest.get("shards", [])}
        observed = {item["sha256"] for item in uploaded_shards}
        if expected != observed:
            raise ValueError("uploaded shards do not match the capture manifest")
        release = {
            "schema": "cpu2tensor-hardware-foundation-release-v1",
            "corpus_id": corpus_id,
            "session": session,
            "capture_manifest_sha256": sha256(manifest_path),
            "capture_manifest": manifest,
            "shards": list(uploaded_shards),
            "decode_states": list(uploaded_decode),
            "custody": uploaded_custody,
        }
        release_path = artifact / "release.json"
        release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
        key = f"releases/{corpus_id}/sessions/{session}.json"
        self.put_verified(release_path, key)
        release["key"] = key
        release["sha256"] = sha256(release_path)
        return release


def upload_artifact(
    artifact: Path, *, bucket: str, region: str,
) -> dict[str, object]:
    store = S3CorpusStore(bucket, region=region)
    descriptors = sorted((artifact / "shards/ready").glob("*.json"))
    uploaded_shards = [store.upload_shard(path) for path in descriptors]
    uploaded_decode = [
        store.upload_decode_state(path)
        for path in sorted((artifact / "decode").glob("kernel-*.pt"))
    ]
    uploaded_custody = store.upload_custody(artifact)
    return store.publish_release(
        artifact, uploaded_shards, uploaded_decode, uploaded_custody
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("artifact", type=Path)
    result.add_argument("--bucket", required=True)
    result.add_argument("--region", default="us-east-1")
    return result


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(upload_artifact(
        args.artifact.resolve(), bucket=args.bucket, region=args.region,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
