# SPDX-License-Identifier: AGPL-3.0-only
"""Relay closed corpus shards from a collector host into verified S3 objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import time

from cpu2tensor.examples.hardware_foundation_upload import S3CorpusStore, sha256


DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class CorpusRelay:
    """Keep at most one raw shard on the coordinator while collection runs."""

    def __init__(
        self, *, host: str, remote: PurePosixPath, staging: Path,
        store: S3CorpusStore, identity: Path | None = None,
    ) -> None:
        if not host or any(character.isspace() for character in host):
            raise ValueError("remote host must be one SSH token")
        if not remote.is_absolute():
            raise ValueError("remote artifact path must be absolute")
        self.host = host
        self.remote = remote
        self.staging = staging
        self.store = store
        self.identity = identity
        self.staging.mkdir(parents=True, exist_ok=True)
        self.state_path = staging / "relay-state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict[str, object]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"schema": "cpu2tensor-hardware-foundation-relay-v1", "shards": []}

    def _save_state(self) -> None:
        temporary = self.state_path.with_suffix(".partial")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.state_path)

    def _ssh(self, command: str, *, capture: bool = True) -> str:
        ssh = ["ssh"]
        if self.identity is not None:
            ssh.extend(("-i", str(self.identity)))
        result = subprocess.run(
            (*ssh, self.host, command), check=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE, text=True,
        )
        return result.stdout if capture else ""

    def _fetch(self, relative: PurePosixPath, destination: Path) -> None:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("relay paths must remain below the artifact")
        source = f"{self.host}:{self.remote / relative}"
        command = ["rsync", "-a"]
        if self.identity is not None:
            command.extend(("-e", f"ssh -i {shlex.quote(str(self.identity))}"))
        subprocess.run((*command, source, str(destination)), check=True)

    def ready_descriptors(self) -> tuple[str, ...]:
        ready = self.remote / "shards/ready"
        command = (
            f"if test -d {shlex.quote(str(ready))}; then "
            f"find {shlex.quote(str(ready))} -maxdepth 1 -type f "
            "-name '*.json' -printf '%f\\n'; fi"
        )
        return tuple(sorted(line for line in self._ssh(command).splitlines() if line))

    def _retire_remote(self, descriptor: dict[str, object]) -> None:
        digest = str(descriptor["sha256"])
        if DIGEST.fullmatch(digest) is None:
            raise ValueError("refusing to retire a non-content-addressed shard")
        tar_name = str(descriptor["tar"])
        index_name = str(descriptor["index"])
        descriptor_name = f"{digest}.json"
        if tar_name != f"{digest}.tar" or index_name != f"{digest}.index.jsonl":
            raise ValueError("descriptor names are not bound to the raw digest")
        ready = self.remote / "shards/ready"
        uploaded = self.remote / "shards/uploaded"
        command = " && ".join((
            f"mkdir -p {shlex.quote(str(uploaded))}",
            f"mv {shlex.quote(str(ready / index_name))} {shlex.quote(str(uploaded / index_name))}",
            f"mv {shlex.quote(str(ready / descriptor_name))} {shlex.quote(str(uploaded / descriptor_name))}",
            f"unlink {shlex.quote(str(ready / tar_name))}",
        ))
        self._ssh(command, capture=False)

    def relay_one(self, name: str) -> dict[str, object]:
        if not name.endswith(".json") or DIGEST.fullmatch(name[:-5]) is None:
            raise ValueError(f"unexpected ready descriptor: {name}")
        local = self.staging / "current"
        local.mkdir(parents=True, exist_ok=True)
        descriptor_path = local / name
        self._fetch(PurePosixPath("shards/ready") / name, descriptor_path)
        descriptor = json.loads(descriptor_path.read_text())
        digest = descriptor.get("sha256")
        if digest != name[:-5]:
            raise ValueError("descriptor filename and checksum disagree")
        for member in (descriptor["tar"], descriptor["index"]):
            self._fetch(PurePosixPath("shards/ready") / str(member), local / str(member))
        uploaded = self.store.upload_shard(descriptor_path)
        self._retire_remote(descriptor)
        self.state["shards"].append(uploaded)
        self._save_state()
        for member in (descriptor["tar"], descriptor["index"], name):
            (local / str(member)).unlink()
        return uploaded

    def capture_finished(self) -> bool:
        manifest = self.remote / "capture-manifest.json"
        result = subprocess.run(
            (("ssh", "-i", str(self.identity), self.host,
              f"test -f {shlex.quote(str(manifest))}")
             if self.identity is not None else
             ("ssh", self.host, f"test -f {shlex.quote(str(manifest))}")),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def capture_failure(self) -> str | None:
        failure = self.remote / "capture-failed.json"
        command = (
            f"if test -f {shlex.quote(str(failure))}; then "
            f"cat {shlex.quote(str(failure))}; fi"
        )
        value = self._ssh(command).strip()
        return value or None

    def finalize(self) -> dict[str, object]:
        artifact = self.staging / "final"
        artifact.mkdir(parents=True, exist_ok=True)
        self._fetch(PurePosixPath("capture-manifest.json"), artifact / "capture-manifest.json")
        for directory in ("decode", "custody"):
            destination = artifact / directory
            destination.mkdir(parents=True, exist_ok=True)
            command = ["rsync", "-a"]
            if self.identity is not None:
                command.extend(("-e", f"ssh -i {shlex.quote(str(self.identity))}"))
            subprocess.run((*command,
                f"{self.host}:{self.remote / directory}/", f"{destination}/",
            ), check=True)
        uploaded_decode = [
            self.store.upload_decode_state(path)
            for path in sorted((artifact / "decode").glob("kernel-*.pt"))
        ]
        uploaded_custody = self.store.upload_custody(artifact)
        release = self.store.publish_release(
            artifact, self.state["shards"], uploaded_decode, uploaded_custody,
        )
        self.state["release"] = {
            key: release[key] for key in ("key", "sha256")
        }
        self._save_state()
        shutil.rmtree(artifact)
        current = self.staging / "current"
        if current.exists():
            shutil.rmtree(current)
        return release

    def run(self, *, poll_seconds: float) -> dict[str, object]:
        while True:
            for name in self.ready_descriptors():
                self.relay_one(name)
            failure = self.capture_failure()
            if failure is not None:
                raise RuntimeError(f"remote capture failed: {failure}")
            if self.capture_finished() and not self.ready_descriptors():
                return self.finalize()
            time.sleep(poll_seconds)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--host", required=True)
    result.add_argument("--remote", type=PurePosixPath, required=True)
    result.add_argument("--staging", type=Path, required=True)
    result.add_argument("--bucket", required=True)
    result.add_argument("--region", default="us-east-1")
    result.add_argument("--identity", type=Path)
    result.add_argument("--poll-seconds", type=float, default=2.0)
    return result


def main() -> None:
    args = parser().parse_args()
    relay = CorpusRelay(
        host=args.host, remote=args.remote, staging=args.staging.resolve(),
        store=S3CorpusStore(args.bucket, region=args.region),
        identity=None if args.identity is None else args.identity.expanduser().resolve(),
    )
    print(json.dumps(relay.run(poll_seconds=args.poll_seconds), sort_keys=True))


if __name__ == "__main__":
    main()
