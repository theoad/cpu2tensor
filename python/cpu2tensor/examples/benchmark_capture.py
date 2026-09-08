# SPDX-License-Identifier: AGPL-3.0-only
"""Measure operator-supplied QEMU commands on one host, with bounded trace draining."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import selectors
import signal
import statistics
import struct
import subprocess
import threading
import time

from cpu2tensor import _native


HEADER = struct.Struct("<IHHIIQQ")
RUN_HEADER = struct.Struct("<HHI")
MIXED = 14
MIXED_FEATURE = 1 << 18
READ_BYTES = 64 * 1024
EVENT_NAMES = {2: "blocks", 7: "registers", 8: "memory", 12: "address_context"}


class TraceCounter:
    """Count frames without decoding rows or allocating tensors.

    The native parser owns header sizes and limits. Payload semantics belong to
    correctness tests; this sink checks framing, source progress and completion.
    """

    def __init__(self):
        self.pending = bytearray()
        self.frames = 0
        self.run_headers = 0
        self.mixed_allowed = False
        self.bytes = 0
        self.events = Counter()
        self.next_sequence = {}
        self.ended = set()
        self.started = False
        self.complete = False
        self.layout_required = False
        self.layout_seen = False
        self.data_seen = False

    def feed(self, data: bytes):
        self.bytes += len(data)
        self.pending.extend(data)
        offset = 0
        while len(self.pending) - offset >= HEADER.size:
            header = bytes(self.pending[offset:offset + HEADER.size])
            size = _native.payload_size(header)
            frame_size = HEADER.size + size
            if len(self.pending) - offset < frame_size:
                break
            _, _, kind, source, count, sequence, detail = HEADER.unpack(header)
            if kind == MIXED:
                self.accept(kind, source, count, sequence, detail,
                            memoryview(self.pending)[offset + HEADER.size:offset + frame_size])
            else:
                self.accept(kind, source, count, sequence, detail)
            self.frames += 1
            offset += frame_size
        del self.pending[:offset]

    def accept(self, kind, source, count, sequence, detail, payload=None):
        if self.complete:
            raise ValueError("Trace contains data after Complete")
        if not self.started:
            if kind != 1:
                raise ValueError("Trace must start with Hello")
            if detail & ((1 << 11) | (1 << 13)):
                raise ValueError("This benchmark requires observation-only capture")
            self.started = True
            self.layout_required = bool(detail & (1 << 17))
            self.mixed_allowed = bool(detail & MIXED_FEATURE)
            return
        if kind == 1:
            raise ValueError("Duplicate trace Hello")
        if kind == 5:
            raise ValueError(f"Plugin reported trace failure {detail}")
        if kind == 13:
            if not self.layout_required or self.layout_seen or self.data_seen:
                raise ValueError("Executable layout must precede source data once")
            self.layout_seen = True
            return  # Worker metadata contributes bytes/frames, not CPU events.
        if self.layout_required and not self.layout_seen and kind != 6:
            raise ValueError("Missing executable layout")
        if kind == 4:
            if detail != 0:
                raise ValueError(f"Trace reported exit code {detail}")
            if self.next_sequence.keys() != self.ended:
                raise ValueError("Complete arrived before every source ended")
            self.complete = True
            return
        if kind == MIXED:
            self.accept_mixed(source, count, sequence, detail, payload)
            return
        if kind not in (2, 3, 6, 7, 8, 12):
            raise ValueError(f"Unsupported observation frame kind {kind}")
        if source in self.ended:
            raise ValueError("Trace contains data after source end")
        expected = self.next_sequence.setdefault(source, 0)
        if kind == 6:
            return  # Schema rows do not advance the source's event sequence.
        self.data_seen = True
        if sequence != expected:
            raise ValueError(f"Source {source} has missing or repeated events")
        if sequence + count > (1 << 64) - 1:
            raise ValueError("Source sequence overflow")
        self.next_sequence[source] += count
        if kind == 3:
            self.ended.add(source)
        else:
            self.events[EVENT_NAMES[kind]] += count

    def accept_mixed(self, source, count, sequence, detail, payload):
        if not self.mixed_allowed or payload is None:
            raise ValueError("Mixed capture needs its advertised feature and payload")
        if not 1 <= count <= 256 or not 16 <= detail <= 4064 or len(payload) != detail:
            raise ValueError("Invalid mixed batch size")
        offset = 0
        events = 0
        while offset < detail:
            if detail - offset < RUN_HEADER.size:
                raise ValueError("Truncated mixed run header")
            kind, rows, size = RUN_HEADER.unpack_from(payload, offset)
            offset += RUN_HEADER.size
            if kind not in EVENT_NAMES or rows == 0 or rows > count - events or size > detail - offset:
                raise ValueError("Invalid mixed run kind, count or length")
            if sequence + events > (1 << 64) - 1:
                raise ValueError("Mixed sequence overflow")
            run_detail = 0 if kind == 2 else size
            inner = HEADER.pack(0x31543243, 2, kind, source, rows, sequence + events, run_detail)
            if _native.payload_size(inner) != size:
                raise ValueError("Mixed run payload does not match its header")
            # Reuse the ordinary progress checks. Only outer frames contribute
            # 32-byte frame headers; each run contributes another eight bytes.
            self.accept(kind, source, rows, sequence + events, run_detail)
            self.run_headers += 1
            events += rows
            offset += size
        if events != count:
            raise ValueError("Mixed event count does not match its runs")

    def finish(self):
        if self.pending:
            raise ValueError("Trace ended in a partial frame")
        if not self.complete:
            raise ValueError("Trace ended without Complete")
        if not sum(self.events.values()):
            raise ValueError("Capture completed without any observed events")

    def result(self):
        return {"bytes": self.bytes, "frames": self.frames,
                "run_headers": self.run_headers,
                "frame_header_bytes": self.frames * HEADER.size,
                "run_header_bytes": self.run_headers * RUN_HEADER.size,
                "framing_bytes": self.frames * HEADER.size + self.run_headers * RUN_HEADER.size,
                "events": dict(self.events), "sources": len(self.next_sequence),
                "complete": self.complete}


def file_contains(path: Path, needle: bytes) -> bool:
    """Search diagnostics without retaining an unbounded guest console log."""
    previous = b""
    with path.open("rb") as stream:
        while data := stream.read(READ_BYTES):
            combined = previous + data
            if needle in combined:
                return True
            previous = combined[-max(0, len(needle) - 1):] if len(needle) > 1 else b""
    return False


def stop_group(pid: int):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_case(case: dict, directory: Path, cwd: Path, timeout: float,
             stdin_path: Path | None = None) -> dict:
    """Run one process group; success requires clean exit and a sealed capture."""
    trace = TraceCounter() if any("{trace_fd}" in part for part in case["argv"]) else None
    trace_read, trace_write = os.pipe() if trace else (-1, -1)
    done_read, done_write = os.pipe()
    command = [part.replace("{trace_fd}", str(trace_write)) for part in case["argv"]]
    result = {"case": case["name"], "argv": command, "status": "failed"}
    child = None
    waiter = None
    child_result = {}
    parent_usage = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    finished = None
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"

    def wait_child():
        _, status, usage = os.wait4(child.pid, 0)
        child_result.update(exit_code=os.waitstatus_to_exitcode(status), usage=usage,
                            elapsed=time.perf_counter() - started)
        os.write(done_write, b"1")

    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr, \
                (stdin_path or Path(os.devnull)).open("rb") as stdin, \
                selectors.DefaultSelector() as selector:
            child = subprocess.Popen(command, cwd=cwd, stdin=stdin, stdout=stdout,
                                     stderr=stderr, pass_fds=(trace_write,) if trace else (),
                                     start_new_session=True)
            if trace:
                os.close(trace_write)
                trace_write = -1
                selector.register(trace_read, selectors.EVENT_READ, "trace")
            selector.register(done_read, selectors.EVENT_READ, "exit")
            waiter = threading.Thread(target=wait_child, name="benchmark-child")
            waiter.start()
            while selector.get_map():
                remaining = timeout - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError(f"Command or trace drain exceeded {timeout:g} seconds")
                for key, _ in selector.select(remaining):
                    if key.data == "exit":
                        os.read(done_read, 1)
                        selector.unregister(done_read)
                    else:
                        data = os.read(trace_read, READ_BYTES)
                        if data:
                            trace.feed(data)
                        else:
                            selector.unregister(trace_read)
            waiter.join()
            finished = time.perf_counter()
            child.returncode = child_result["exit_code"]
            if child.returncode != 0:
                raise ValueError(f"Command exited with status {child.returncode}")
            if trace:
                trace.finish()
            stdout.flush()
            if case.get("success_stdout") and not file_contains(
                    stdout_path, case["success_stdout"].encode()):
                raise ValueError("Expected workload success text was absent from stdout")
            result["status"] = "ok"
    except (OSError, ValueError, TimeoutError) as error:
        result["error"] = str(error)
        result["status"] = "timeout" if isinstance(error, TimeoutError) else "failed"
    finally:
        if child is not None and result["status"] != "ok":
            stop_group(child.pid)
        if waiter is not None:
            waiter.join()
            child.returncode = child_result["exit_code"]
        for descriptor in (trace_read, trace_write, done_read, done_write):
            if descriptor >= 0:
                os.close(descriptor)
    result["elapsed_seconds"] = (finished or time.perf_counter()) - started
    parent_after = resource.getrusage(resource.RUSAGE_SELF)
    result["drain_user_seconds"] = parent_after.ru_utime - parent_usage.ru_utime
    result["drain_system_seconds"] = parent_after.ru_stime - parent_usage.ru_stime
    if child_result:
        usage = child_result["usage"]
        result.update(exit_code=child_result["exit_code"],
                      child_elapsed_seconds=child_result["elapsed"],
                      user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                      peak_rss_bytes=usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
                      minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt,
                      voluntary_switches=usage.ru_nvcsw, involuntary_switches=usage.ru_nivcsw)
    if trace:
        result["trace"] = trace.result()
    result.update(stdout_bytes=stdout_path.stat().st_size,
                  stderr_bytes=stderr_path.stat().st_size)
    return result


def summarize(runs: list, baseline: str) -> dict:
    """Never compute a successful median by silently discarding failed repeats."""
    groups = {}
    for run in runs:
        groups.setdefault(run["case"], []).append(run)
    summaries = {}
    for name, rows in groups.items():
        summary = {"runs": len(rows), "successful_runs": sum(row["status"] == "ok" for row in rows)}
        if all(row["status"] == "ok" for row in rows):
            summary["median_seconds"] = statistics.median(row["elapsed_seconds"] for row in rows)
            if all("trace" in row for row in rows):
                summary["median_events_per_second"] = statistics.median(
                    sum(row["trace"]["events"].values()) / row["elapsed_seconds"] for row in rows)
                summary["median_trace_bytes_per_second"] = statistics.median(
                    row["trace"]["bytes"] / row["elapsed_seconds"] for row in rows)
                summary["median_framing_fraction"] = statistics.median(
                    (row["trace"]["frames"] * HEADER.size + row["trace"].get("run_header_bytes", 0))
                    / row["trace"]["bytes"] for row in rows)
        summaries[name] = summary
    baseline_time = summaries.get(baseline, {}).get("median_seconds")
    if baseline_time:
        for summary in summaries.values():
            if "median_seconds" in summary:
                summary["slowdown_vs_baseline"] = summary["median_seconds"] / baseline_time
    return summaries


def positive_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return value


def validate_manifest(manifest: dict):
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    for name in ("host_name", "host_isa", "guest_isa", "workload", "baseline"):
        if not isinstance(manifest.get(name), str) or not manifest[name].strip():
            raise ValueError(f"Manifest needs a nonempty {name}")
    if manifest["host_isa"] != platform.machine():
        raise ValueError("Manifest host_isa does not match this host; do not compare cross-ISA runs")
    repeats = manifest.get("repeats", 3)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    positive_number(manifest.get("timeout_seconds", 60), "timeout_seconds")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest needs a nonempty cases array")
    names = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str) or not case["name"]:
            raise ValueError("Each case needs a name")
        if case["name"] in names:
            raise ValueError("Case names must be unique")
        names.add(case["name"])
        argv = case.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and arg for arg in argv):
            raise ValueError("Each argv must be a nonempty array of nonempty strings")
        if not isinstance(case.get("signals"), str) or not case["signals"]:
            raise ValueError("Each case needs a signals description")
        if "success_stdout" in case and (not isinstance(case["success_stdout"], str) or
                                          not 0 < len(case["success_stdout"].encode()) <= 4096):
            raise ValueError("success_stdout must contain 1 to 4096 bytes")
    if manifest["baseline"] not in names:
        raise ValueError("baseline must name one case")
    baseline = next(case for case in cases if case["name"] == manifest["baseline"])
    if any("{trace_fd}" in arg for arg in baseline["argv"]):
        raise ValueError("The baseline must not produce a cpu2tensor trace")
    for name in ("cwd", "stdin"):
        if name in manifest and not isinstance(manifest[name], str):
            raise ValueError(f"{name} must be a path string")
    if not isinstance(manifest.get("files", []), list) or not all(
            isinstance(path, str) for path in manifest.get("files", [])):
        raise ValueError("files must be an array of provenance paths")


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def benchmark(manifest_path: Path, output: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    cwd = (manifest_path.parent / manifest.get("cwd", ".")).resolve()
    stdin_path = (cwd / manifest["stdin"]).resolve() if "stdin" in manifest else None
    files = [file_identity((cwd / name).resolve()) for name in manifest.get("files", [])]
    if stdin_path:
        files.append(file_identity(stdin_path))
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {"format": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "host": {"name": manifest["host_name"], "hostname": platform.node(),
                       "machine": platform.machine(), "platform": platform.platform(),
                       "python": platform.python_version(), "cpu_count": os.cpu_count(),
                       "load_average": os.getloadavg()},
              "runner": file_identity(Path(__file__).resolve()),
              "native_decoder": file_identity(Path(_native.__file__).resolve()),
              "cwd": str(cwd), "files": files, "runs": []}
    cases = manifest["cases"]
    for repeat in range(manifest.get("repeats", 3)):
        # Rotate the starting case to expose rather than hide warm-cache/order effects.
        for offset in range(len(cases)):
            case = cases[(offset + repeat) % len(cases)]
            directory = output / f"run-{len(report['runs']):03d}"
            directory.mkdir()
            run = run_case(case, directory, cwd, manifest.get("timeout_seconds", 60), stdin_path)
            run.update(repeat=repeat, artifacts=directory.name)
            report["runs"].append(run)
            report["summary"] = summarize(report["runs"], manifest["baseline"])
            (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON manifest of argv arrays; no shell is used")
    parser.add_argument("--output", required=True, type=Path, help="New host-local artifact directory")
    arguments = parser.parse_args()
    try:
        report = benchmark(arguments.manifest, arguments.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"benchmark_capture: {error}\n")
    print(json.dumps(report["summary"], indent=2))
    raise SystemExit(0 if all(run["status"] == "ok" for run in report["runs"]) else 1)


if __name__ == "__main__":
    main()
