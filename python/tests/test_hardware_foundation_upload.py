# SPDX-License-Identifier: AGPL-3.0-only

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from unittest import mock

from cpu2tensor.examples.hardware_foundation_upload import S3CorpusStore


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_verified_upload_uses_checksum_and_versioned_head() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "object"
        path.write_bytes(b"foundation")
        digest = _digest(b"foundation")
        checksum = base64.b64encode(bytes.fromhex(digest)).decode()
        missing = subprocess.CalledProcessError(
            254, ("aws",), stderr="Not Found (404)",
        )
        uploaded = {
            "ContentLength": len(b"foundation"),
            "ChecksumSHA256": checksum,
            "VersionId": "version-1",
        }
        store = S3CorpusStore("bucket")
        with mock.patch.object(
            store, "_aws", side_effect=[missing, {}, uploaded]
        ) as command:
            store.put_verified(path, "objects/item", digest)
        put = command.call_args_list[1].args[0]
        assert put[0] == "put-object"
        assert "--checksum-algorithm" in put
        assert f"sha256={digest}" in put


def test_upload_shard_rejects_tampered_bundle() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tar_path = root / "a.tar"
        index_path = root / "a.index.jsonl"
        descriptor_path = root / "a.json"
        tar_path.write_bytes(b"raw")
        index_path.write_bytes(b"index")
        descriptor_path.write_text(json.dumps({
            "sha256": _digest(b"other"),
            "index_sha256": _digest(b"index"),
            "bytes": 3,
            "tar": tar_path.name,
            "index": index_path.name,
        }))
        store = S3CorpusStore("bucket")
        try:
            store.upload_shard(descriptor_path)
        except ValueError as error:
            assert "verification" in str(error)
        else:
            raise AssertionError("tampered raw shard was uploaded")
