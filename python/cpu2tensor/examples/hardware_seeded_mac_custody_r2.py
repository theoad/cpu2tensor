# SPDX-License-Identifier: AGPL-3.0-only
"""Independently verify a copied 51-row smoke before issuing Mac custody ack.

The copied spool is read-only. Only the explicitly named, new acknowledgement
file outside that spool is written after every check passes.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from cpu2tensor.examples import hardware_seeded_capture_plan_r2 as plans
from cpu2tensor.examples import hardware_seeded_capture_supervisor_r2 as guard
from cpu2tensor.examples import hardware_multimodal_experiment as collector


ACK_SCHEMA = "cpu2tensor-seeded-mac-custody-ack-r2"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _bounded_files(spool: Path) -> dict[str, str]:
    if not spool.is_dir() or spool.is_symlink():
        raise ValueError("custody spool must be a real directory")
    hashes = {}
    total_bytes = 0
    for path in spool.rglob("*"):
        if path.is_symlink():
            raise ValueError("custody spool contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("custody spool contains a non-regular path")
        relative = path.relative_to(spool).as_posix()
        if path.name.endswith(".partial") or path.name == "failure.json":
            raise ValueError("custody spool contains incomplete capture evidence")
        total_bytes += path.stat().st_size
        if total_bytes > guard.SPOOL_CAP:
            raise ValueError("custody spool exceeds 896 MiB")
        hashes[relative] = guard.sha256(path)
    return hashes


def verify_copy(
    spool: Path, *, expected_manifest_sha256: str,
    expected_source_bundle_sha256: str, expected_binary_sha256: str,
) -> dict[str, object]:
    """Return a complete ack payload, without writing the copied evidence."""
    for value in (
        expected_manifest_sha256, expected_source_bundle_sha256,
        expected_binary_sha256,
    ):
        if not HEX_SHA256.fullmatch(value):
            raise ValueError("expected hashes must be lowercase SHA-256")
    if spool.is_symlink():
        raise ValueError("custody spool may not be a symlink")
    spool = spool.resolve(strict=True)
    file_hashes = _bounded_files(spool)
    required = {
        "source-bundle.json", "smoke-plan.json", "smoke/execution-plan.json",
        "smoke/capture-manifest.json", "collector.log",
    }
    if not required.issubset(file_hashes):
        raise ValueError("copied spool is missing a required sealed file")
    if file_hashes["smoke/capture-manifest.json"] != expected_manifest_sha256:
        raise ValueError("copied manifest differs from host SHA-256")
    bundle = json.loads((spool / "source-bundle.json").read_text())
    if (bundle.get("schema") != "cpu2tensor-seeded-source-bundle-r2" or
            type(bundle.get("files")) is not dict or
            bundle.get("digest_sha256") != _digest(bundle["files"]) or
            bundle["digest_sha256"] != expected_source_bundle_sha256):
        raise ValueError("source bundle does not match preapproved digest")
    for name, local in (
        ("python/cpu2tensor/examples/hardware_multimodal_experiment.py", collector.__file__),
        ("python/cpu2tensor/examples/hardware_seeded_capture_plan_r2.py", plans.__file__),
        ("python/cpu2tensor/examples/hardware_seeded_capture_supervisor_r2.py", guard.__file__),
    ):
        if bundle["files"].get(name) != guard.sha256(Path(local)):
            raise ValueError("Mac verifier source differs from host bundle")

    artifact = spool / "smoke"
    manifest = json.loads((artifact / "capture-manifest.json").read_text())
    subject = manifest.get("subject", {})
    event = manifest.get("event", {})
    build = manifest.get("build", {})
    if (subject.get("boot_id") != guard.EXPECTED_BOOT or
            subject.get("kernel_release") != guard.EXPECTED_KERNEL or
            subject.get("intel_pstate_no_turbo") != "1" or
            subject.get("target_scaling_max_freq_khz") != guard.CAPTURE_FREQ_KHZ or
            subject.get("target_cpu") != 2 or subject.get("controller_cpu") != 3 or
            event.get("pebs_event") != "memory_loads" or
            event.get("pebs_period") != 10_000 or
            build.get("workload_sha256") != expected_binary_sha256):
        raise ValueError("copied capture differs from approved physical subject")
    identity = manifest.get("identity_sha256")
    plan = plans.load_plan(artifact / "execution-plan.json", identity)
    if (plan["kind"] != "smoke" or
            json.loads((spool / "smoke-plan.json").read_text()) != plan):
        raise ValueError("copied smoke plan differs from sealed plan")
    guard.verify_smoke_artifact(artifact, identity, plan)

    expected_paths = set(required)
    for entry in manifest["entries"]:
        expected_paths.add("smoke/" + entry["raw_path"])
        expected_paths.add("smoke/" + entry["derived_path"])
    states = manifest.get("kernel_decode_states")
    if type(states) is not list or not states:
        raise ValueError("copied capture lacks decode sideband custody")
    for state in states:
        relative = Path(state["path"])
        if (relative.is_absolute() or not relative.parts or
                relative.parts[0] != "decode" or ".." in relative.parts):
            raise ValueError("unsafe decode sideband path")
        path = "smoke/" + relative.as_posix()
        expected_paths.add(path)
        if file_hashes.get(path) != state["sha256"]:
            raise ValueError("decode sideband hash mismatch")
    unexpected = set(file_hashes) - expected_paths
    if unexpected != {name for name in unexpected if name.endswith(".log") and "/" not in name}:
        raise ValueError("copied spool contains unaccounted files")
    if expected_paths - set(file_hashes):
        raise ValueError("copied spool is missing sealed evidence")
    if any(not (spool / relative).is_file() for relative in expected_paths):
        raise ValueError("copied spool contains non-regular sealed evidence")
    return {
        "schema": ACK_SCHEMA,
        "status": "verified",
        "copy_path": str(spool),
        "host_manifest_sha256": expected_manifest_sha256,
        "subject_identity_sha256": identity,
        "source_bundle_digest_sha256": expected_source_bundle_sha256,
        "source_bundle_file_sha256": file_hashes["source-bundle.json"],
        "binary_sha256": expected_binary_sha256,
        "plan_sha256": file_hashes["smoke/execution-plan.json"],
        "execution_count": 51,
        "retained_raw_count": 51,
        "files": dict(sorted(file_hashes.items())),
    }


def write_ack(output: Path, spool: Path, value: dict[str, object]) -> None:
    if output.resolve().is_relative_to(spool.resolve()):
        raise ValueError("acknowledgement must be outside the copied spool")
    if not output.parent.is_dir() or output.exists():
        raise ValueError("acknowledgement output must be new in an existing directory")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spool", type=Path)
    parser.add_argument("--expected-host-manifest-sha256", required=True)
    parser.add_argument("--expected-source-bundle-sha256", required=True)
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--ack-output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_copy(
        args.spool,
        expected_manifest_sha256=args.expected_host_manifest_sha256,
        expected_source_bundle_sha256=args.expected_source_bundle_sha256,
        expected_binary_sha256=args.expected_binary_sha256,
    )
    write_ack(args.ack_output, args.spool, result)
    print(f"verified 51/51 smoke rows; ack: {args.ack_output}")


if __name__ == "__main__":
    main()
