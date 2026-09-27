# SPDX-License-Identifier: AGPL-3.0-only
"""Build a deterministic broad-software plan for the x86 foundation corpus."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from typing import Callable, Sequence

from cpu2tensor.examples.hardware_foundation_corpus import PLAN_SCHEMA


@dataclass(frozen=True)
class Fixture:
    ordinal: int
    seed: int
    binary: Path
    text: Path
    sorted_text: Path
    csv: Path
    left: Path
    right: Path


@dataclass(frozen=True)
class Workload:
    family: str
    application: str
    command: Callable[[Fixture, Path], tuple[tuple[str, ...], tuple[Path, ...]]]
    expected_exit_code: int = 0


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required broad-software tool is missing: {name}")
    return str(Path(path).resolve())


def _write(path: Path, value: bytes) -> None:
    path.write_bytes(value)
    os.utime(path, (0, 0))


def make_fixtures(root: Path, *, count: int, seed: int, size: int) -> tuple[Fixture, ...]:
    root.mkdir(parents=True, exist_ok=True)
    fixtures = []
    vocabulary = (
        "alpha", "beta", "gamma", "delta", "epsilon", "lambda", "omega",
        "kernel", "cache", "branch", "memory", "packet", "tensor", "trace",
    )
    for ordinal in range(count):
        input_seed = seed + ordinal
        binary = root / f"binary-{ordinal:03d}.bin"
        text = root / f"text-{ordinal:03d}.txt"
        sorted_text = root / f"sorted-{ordinal:03d}.txt"
        csv = root / f"table-{ordinal:03d}.csv"
        left = root / f"left-{ordinal:03d}.txt"
        right = root / f"right-{ordinal:03d}.txt"
        raw = hashlib.shake_256(f"binary:{input_seed}".encode()).digest(size)
        words = []
        cursor = 0
        while sum(len(line) + 1 for line in words) < size:
            digest = hashlib.sha256(f"text:{input_seed}:{cursor}".encode()).digest()
            tokens = [vocabulary[value % len(vocabulary)] for value in digest[:8]]
            words.append(f"{cursor:06d} " + " ".join(tokens))
            cursor += 1
        lines = words
        sorted_lines = sorted(lines)
        csv_lines = [
            f"{index},{line.split()[1]},{hashlib.sha256(line.encode()).hexdigest()[:16]}"
            for index, line in enumerate(lines)
        ]
        _write(binary, raw)
        _write(text, ("\n".join(lines) + "\n").encode())
        _write(sorted_text, ("\n".join(sorted_lines) + "\n").encode())
        _write(csv, ("\n".join(csv_lines) + "\n").encode())
        _write(left, ("\n".join(sorted_lines[::2]) + "\n").encode())
        changed = list(sorted_lines[::2])
        if changed:
            changed[len(changed) // 2] += " changed"
        _write(right, ("\n".join(changed) + "\n").encode())
        fixtures.append(Fixture(
            ordinal, input_seed, binary, text, sorted_text, csv, left, right,
        ))
    return tuple(fixtures)


def _single(tool: str, *arguments: str, member: str = "binary"):
    executable = _tool(tool)

    def command(fixture: Fixture, _root: Path) -> tuple[tuple[str, ...], tuple[Path, ...]]:
        path = getattr(fixture, member)
        return (executable, *arguments, str(path)), (path,)
    return command


def _fixed(tool: str, *arguments: str, input_tool: str):
    executable = _tool(tool)
    input_path = Path(_tool(input_tool))

    def command(_fixture: Fixture, _root: Path) -> tuple[tuple[str, ...], tuple[Path, ...]]:
        return (executable, *arguments, str(input_path)), (input_path,)
    return command


def workloads() -> tuple[Workload, ...]:
    openssl = _tool("openssl")
    tar = _tool("tar")
    git = _tool("git")
    perl = _tool("perl")
    python = _tool("python3")
    cmake = _tool("cmake")
    join = _tool("join")
    comm = _tool("comm")
    paste = _tool("paste")
    diff = _tool("diff")
    cmp = _tool("cmp")

    def pair(executable: str, left_flag: Sequence[str] = ()):
        def command(fixture: Fixture, _root: Path):
            return (
                executable, *left_flag, str(fixture.left), str(fixture.right),
            ), (fixture.left, fixture.right)
        return command

    def archive(fixture: Fixture, root: Path):
        return (
            tar, "--sort=name", "--mtime=@0", "--owner=0", "--group=0",
            "--numeric-owner", "-cf", "-", "-C", str(root), fixture.binary.name,
            fixture.text.name,
        ), (fixture.binary, fixture.text)

    def interpreter(executable: str, script_name: str):
        def command(fixture: Fixture, root: Path):
            script = root / script_name
            return (executable, str(script), str(fixture.binary)), (script, fixture.binary)
        return command

    return (
        Workload("hash", "sha256sum", _single("sha256sum")),
        Workload("hash", "md5sum", _single("md5sum")),
        Workload("hash", "sha512sum", _single("sha512sum")),
        Workload("hash", "cksum", _single("cksum")),
        Workload("crypto", "openssl-sha3", lambda f, _r: (
            (openssl, "dgst", "-sha3-256", str(f.binary)), (f.binary,),
        )),
        Workload("crypto", "openssl-aes", lambda f, _r: ((
            openssl, "enc", "-aes-256-ctr", "-K", "11" * 32,
            "-iv", "22" * 16, "-in", str(f.binary),
        ), (f.binary,))),
        Workload("compression", "gzip", _single("gzip", "-n", "-c")),
        Workload("compression", "bzip2", _single("bzip2", "-c")),
        Workload("compression", "xz", _single("xz", "--threads=1", "-c")),
        Workload("text", "sort", _single("sort", member="text")),
        Workload("text", "grep", _single("grep", "-E", "alpha|omega", member="text")),
        Workload("text", "sed", _single("sed", "-E", "s/[0-9]+/<n>/g", member="text")),
        Workload("text", "awk", _single(
            "awk", "{s += length($0)} END {print s}", member="text"
        )),
        Workload("text", "wc", _single("wc", "-l", "-w", "-c", member="text")),
        Workload("text", "uniq", _single("uniq", "-c", member="sorted_text")),
        Workload("text", "cut", _single("cut", "-d,", "-f1,3", member="csv")),
        Workload("search", "ripgrep", _single(
            "rg", "--no-heading", "--no-line-number", "alpha|omega", member="text"
        )),
        Workload("archive", "tar", archive),
        Workload("object", "git-hash-object", lambda f, _r: (
            (git, "hash-object", str(f.binary)), (f.binary,),
        )),
        Workload("interpreter", "perl-byte-analysis", interpreter(perl, "analyze.pl")),
        Workload("interpreter", "python-byte-analysis", interpreter(python, "analyze.py")),
        Workload("build-tool", "cmake-sha256", lambda f, _r: (
            (cmake, "-E", "sha256sum", str(f.binary)), (f.binary,),
        )),
        Workload("binary-analysis", "objdump", _fixed(
            "objdump", "-d", input_tool="sha256sum"
        )),
        Workload("binary-analysis", "readelf", _fixed(
            "readelf", "-a", input_tool="sha256sum"
        )),
        Workload("binary-analysis", "strings", _single("strings")),
        Workload("encoding", "base64", _single("base64")),
        Workload("encoding", "base32", _single("base32")),
        Workload("encoding", "iconv", _single(
            "iconv", "-f", "UTF-8", "-t", "UTF-16LE", member="text"
        )),
        Workload("format", "pr", _single("pr", "-t", "-w", "80", member="text")),
        Workload("relational", "join", pair(join)),
        Workload("relational", "comm", pair(comm)),
        Workload("relational", "paste", pair(paste)),
        Workload("binary-analysis", "od", _single("od", "-An", "-tx1")),
        Workload("comparison", "diff", pair(diff, ("--label", "left", "--label", "right", "-u")), 1),
        Workload("comparison", "cmp", pair(cmp, ("-l",)), 1),
    )


def _scripts(root: Path) -> None:
    _write(root / "analyze.py", (
        b"import hashlib, pathlib, sys\n"
        b"data = pathlib.Path(sys.argv[1]).read_bytes()\n"
        b"hist = [data.count(i) for i in range(256)]\n"
        b"print(hashlib.sha256(data).hexdigest(), "
        b"sum((i + 1) * n for i, n in enumerate(hist)))\n"
    ))
    _write(root / "analyze.pl", (
        b"use strict; use warnings;\n"
        b"open my $fh, '<:raw', $ARGV[0] or die $!;\n"
        b"local $/; my $data = <$fh>; my @h = (0) x 256; "
        b"$h[$_]++ for unpack('C*', $data);\n"
        b"my $sum = 0; $sum += ($_ + 1) * $h[$_] for 0..255; "
        b"print length($data), ' ', $sum, \"\\n\";\n"
    ))


def _run_twice(
    argv: Sequence[str], *, cwd: Path, environment: dict[str, str],
    exit_code: int,
) -> tuple[str, str]:
    observed = []
    for _ in range(2):
        result = subprocess.run(
            argv, cwd=cwd, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30,
        )
        evidence = (
            result.returncode,
            hashlib.sha256(result.stdout).hexdigest(),
            hashlib.sha256(result.stderr).hexdigest(),
        )
        observed.append(evidence)
    if observed[0] != observed[1] or observed[0][0] != exit_code:
        raise RuntimeError(f"workload is not deterministic: {argv!r}: {observed!r}")
    return observed[0][1], observed[0][2]


def make_plan(
    root: Path, *, fixture_count: int, repetitions: int, seed: int,
    input_bytes: int,
) -> dict[str, object]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    _scripts(inputs)
    fixtures = make_fixtures(inputs, count=fixture_count, seed=seed, size=input_bytes)
    applications = workloads()
    generator = random.Random(seed)
    heldout_count = max(3, len(applications) // 5)
    heldout = {
        item.application for item in generator.sample(list(applications), heldout_count)
    }
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "SOURCE_DATE_EPOCH": "0",
        "TZ": "UTC",
    }
    expectations: dict[tuple[str, int], tuple[str, str]] = {}
    rows = []
    for session in ("session-a", "session-b"):
        for workload in applications:
            for repetition in range(repetitions):
                fixture = fixtures[repetition % len(fixtures)]
                argv, paths = workload.command(fixture, inputs)
                cache_key = (workload.application, fixture.ordinal)
                if cache_key not in expectations:
                    expectations[cache_key] = _run_twice(
                        argv, cwd=inputs, environment=environment,
                        exit_code=workload.expected_exit_code,
                    )
                stdout_hash, stderr_hash = expectations[cache_key]
                if workload.application in heldout:
                    partition = "heldout_application"
                elif session == "session-b":
                    partition = "heldout_session"
                else:
                    fraction = repetition / repetitions
                    partition = (
                        "training" if fraction < 0.70 else
                        "calibration" if fraction < 0.85 else
                        "familiar_validation"
                    )
                rows.append({
                    "execution_id": f"{workload.application}-{session[-1]}-{repetition:06d}",
                    "family": workload.family,
                    "application": workload.application,
                    "partition": partition,
                    "session": session,
                    "input_seed": fixture.seed,
                    "argv": list(argv),
                    "stdin_base64": base64.b64encode(b"").decode(),
                    "cwd": str(inputs),
                    "input_paths": [str(path) for path in paths],
                    "expected_exit_code": workload.expected_exit_code,
                    "expected_stdout_sha256": stdout_hash,
                    "expected_stderr_sha256": stderr_hash,
                    "timeout_seconds": 30,
                })
    generator.shuffle(rows)
    payload = {
        "schema": PLAN_SCHEMA,
        "corpus_id": (
            f"intel-x86-foundation-s{seed}-f{fixture_count}"
            f"-b{input_bytes}-r{repetitions}"
        ),
        "generator_seed": seed,
        "fixture_count": fixture_count,
        "input_bytes": input_bytes,
        "heldout_applications": sorted(heldout),
        "environment": environment,
        "rows": rows,
    }
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("root", type=Path)
    result.add_argument("--fixture-count", type=int, default=16)
    result.add_argument("--repetitions", type=int, default=400)
    result.add_argument("--seed", type=int, default=20260928)
    result.add_argument("--input-bytes", type=int, default=256 * 1024)
    return result


def main() -> None:
    args = parser().parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = make_plan(
        root, fixture_count=args.fixture_count, repetitions=args.repetitions,
        seed=args.seed, input_bytes=args.input_bytes,
    )
    path = root / "plan.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "plan": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(payload["rows"]),
        "applications": len({row["application"] for row in payload["rows"]}),
        "heldout_applications": payload["heldout_applications"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
