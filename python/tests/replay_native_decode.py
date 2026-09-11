# SPDX-License-Identifier: AGPL-3.0-only
"""Compare threaded and process-isolated native decode of one recorded trace."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import argparse
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import platform
import statistics
import struct
import time


HEADER = struct.Struct("<IHHIIQQ")
_native = None


def load_native(path: str) -> None:
    global _native
    spec = importlib.util.spec_from_file_location("_native", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load native decoder at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _native = module


def read_frames(path: Path) -> tuple[bytes, ...]:
    data = path.read_bytes()
    frames = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < HEADER.size:
            raise ValueError("Recorded trace ends in a partial header")
        header = data[offset:offset + HEADER.size]
        size = _native.payload_size(header)
        end = offset + HEADER.size + size
        if end > len(data):
            raise ValueError("Recorded trace ends in a partial payload")
        frames.append(data[offset:end])
        offset = end
    return tuple(frames)


def decode(frames: tuple[bytes, ...], group_frames: int = 1):
    stream = _native.new_stream()
    decoded = []
    offset = 0
    while offset < len(frames):
        _, _, kind, _, _, _, _ = HEADER.unpack(frames[offset][:HEADER.size])
        if group_frames > 1 and kind == 14:
            end = offset + 1
            while end < min(len(frames), offset + group_frames):
                _, _, following, _, _, _, _ = HEADER.unpack(frames[end][:HEADER.size])
                if following != 14:
                    break
                end += 1
            ready, error = _native.decode_context_frames(stream, frames[offset:end])
            decoded.extend(ready)
            if error is not None:
                raise ValueError(error)
            offset = end
        else:
            decoded.append(_native.decode(stream, frames[offset]))
            offset += 1
    return tuple(decoded)


def trace_summary(decoded) -> dict:
    sources = {}
    ended = {}
    complete = False
    for kind, source, count, sequence, detail, payload in decoded:
        if kind == 14:
            state = sources.setdefault(source, {"sequences": [], "blocks": 0,
                                                "contexts": 0, "address_sum": 0})
            if "blocks" in payload:
                addresses = [value[0] for value in struct.iter_unpack("<Q", payload["blocks"])]
                sequences = [value[0] for value in struct.iter_unpack(
                    "<Q", payload["block_sequences"]
                )]
                if len(addresses) != len(sequences):
                    raise AssertionError("Block values and sequences have different row counts")
                state["sequences"].extend(sequences)
                state["blocks"] += len(addresses)
                state["address_sum"] += sum(addresses)
            if "context" in payload:
                sequences = [value[0] for value in struct.iter_unpack(
                    "<Q", payload["context"]["sequences"]
                )]
                state["sequences"].extend(sequences)
                state["contexts"] += len(sequences)
        elif kind in (2, 12):
            state = sources.setdefault(source, {"sequences": [], "blocks": 0,
                                                "contexts": 0, "address_sum": 0})
            state["sequences"].extend(range(sequence, sequence + count))
            state["blocks" if kind == 2 else "contexts"] += count
            if kind == 2:
                state["address_sum"] += sum(
                    value[0] for value in struct.iter_unpack("<Q", payload)
                )
        elif kind == 3:
            ended[source] = sequence
        elif kind == 4:
            complete = detail == 0
    for source, state in sources.items():
        tail = ended.get(source)
        if tail is None or sorted(state["sequences"]) != list(range(tail)):
            raise AssertionError(f"Source {source} rows are missing, repeated or unsealed")
        del state["sequences"]
        state["end_sequence"] = tail
    if not complete or set(sources) != set(ended):
        raise AssertionError("Trace does not complete every source")
    return {"complete": True, "sources": sources}


def context_trace_digest(decoded) -> str:
    """Hash every context-only row after restoring each source's sequence order."""
    rows = {}
    for kind, source, _, sequence, _, payload in decoded:
        if kind == 14:
            if "blocks" in payload:
                addresses = struct.iter_unpack("<Q", payload["blocks"])
                sequences = struct.iter_unpack("<Q", payload["block_sequences"])
                rows.setdefault(source, []).extend(
                    (position[0], 2, address[0])
                    for position, address in zip(sequences, addresses, strict=True)
                )
            if "context" in payload:
                context = payload["context"]
                columns = [struct.iter_unpack("<Q", context[name]) for name in (
                    "pc", "cr0", "cr3", "cr4", "efer", "cs_base", "mode", "known",
                )]
                sequences = struct.iter_unpack("<Q", context["sequences"])
                rows.setdefault(source, []).extend(
                    (position[0], 12, *(value[0] for value in values))
                    for position, values in zip(sequences, zip(*columns, strict=True), strict=True)
                )
        elif kind == 2:
            rows.setdefault(source, []).extend(
                (sequence + offset, 2, address[0])
                for offset, address in enumerate(struct.iter_unpack("<Q", payload))
            )
        elif kind == 12:
            columns = [struct.iter_unpack("<Q", payload[name]) for name in (
                "pc", "cr0", "cr3", "cr4", "efer", "cs_base", "mode", "known",
            )]
            rows.setdefault(source, []).extend(
                (sequence + offset, 12, *(value[0] for value in values))
                for offset, values in enumerate(zip(*columns, strict=True))
            )
    digest = hashlib.sha256()
    for source in sorted(rows):
        for row in sorted(rows[source]):
            digest.update(struct.pack(f"<{len(row) + 1}Q", source, *row))
    return digest.hexdigest()


def run_worker(frames: tuple[bytes, ...], expected, iterations: int, check_every: bool,
               group_frames: int):
    started = time.thread_time()
    result = None
    for _ in range(iterations):
        result = decode(frames, group_frames)
        if check_every and result != expected:
            raise AssertionError("Decoded rows differ from the canonical trace")
    if result != expected:
        raise AssertionError("Final decoded rows differ from the canonical trace")
    return time.thread_time() - started


def measure(kind: str, workers: int, repetitions: int, iterations: int,
            module: str, frames: tuple[bytes, ...], expected, rows: int, check_every: bool,
            group_frames: int):
    executor_type = ThreadPoolExecutor if kind == "threads" else ProcessPoolExecutor
    options = {"max_workers": workers, "initializer": load_native, "initargs": (module,)}
    if kind == "processes":
        options["mp_context"] = multiprocessing.get_context("spawn")
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        with executor_type(**options) as executor:
            futures = [executor.submit(
                run_worker, frames, expected, iterations, check_every, group_frames,
            )
                       for _ in range(workers)]
            cpu_seconds = sum(future.result() for future in futures)
        wall_seconds = time.perf_counter() - started
        samples.append({
            "wall_seconds": wall_seconds,
            "worker_cpu_seconds": cpu_seconds,
            "rows_per_second": workers * iterations * rows / wall_seconds,
        })
    return {
        "mode": kind,
        "workers": workers,
        "samples": samples,
        "median_wall_seconds": statistics.median(sample["wall_seconds"] for sample in samples),
        "median_rows_per_second": statistics.median(sample["rows_per_second"] for sample in samples),
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workers", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--modes", nargs="+", choices=("threads", "processes"),
                        default=("threads", "processes"))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--group-frames", type=int, choices=range(1, 33), default=1,
                        metavar="1..32", help="Collate this many context-only mixed frames")
    parser.add_argument("--check-last", action="store_true",
                        help="Compare only each worker's final replay for bottleneck isolation")
    arguments = parser.parse_args()
    if any(worker < 1 for worker in arguments.workers):
        parser.error("Worker counts must be positive")
    if arguments.iterations < 1 or arguments.repetitions < 1:
        parser.error("Iterations and repetitions must be positive")

    module = str(arguments.module.resolve())
    load_native(module)
    frames = read_frames(arguments.capture)
    legacy = decode(frames)
    expected = decode(frames, arguments.group_frames)
    if (trace_summary(expected) != trace_summary(legacy) or
            context_trace_digest(expected) != context_trace_digest(legacy)):
        raise AssertionError("Grouped decode changed exact context-only rows")
    summary = trace_summary(expected)
    # Stream validation establishes continuous per-source sequences and complete
    # termination. Equality checks every returned header and owned column byte.
    _, _, final_kind, _, _, _, final_detail = HEADER.unpack(frames[-1][:HEADER.size])
    if final_kind != 4 or final_detail != 0:
        raise ValueError("Recorded trace does not end in successful completion")
    rows = sum(HEADER.unpack(frame[:HEADER.size])[4] for frame in frames)
    results = [
        measure(mode, workers, arguments.repetitions, arguments.iterations,
                module, frames, expected, rows, not arguments.check_last,
                arguments.group_frames)
        for workers in arguments.workers
        for mode in arguments.modes
    ]
    print(json.dumps({
        "host": platform.node(),
        "platform": platform.platform(),
        "logical_cpus": multiprocessing.cpu_count(),
        "revision": arguments.revision,
        "native_module_sha256": file_hash(arguments.module),
        "capture": str(arguments.capture.resolve()),
        "capture_sha256": file_hash(arguments.capture),
        "capture_bytes": arguments.capture.stat().st_size,
        "frames_per_iteration": len(frames),
        "native_group_frames": arguments.group_frames,
        "rows_per_iteration": rows,
        "trace_summary": summary,
        "exact_row_sha256": context_trace_digest(expected),
        "iterations_per_worker": arguments.iterations,
        "repetitions": arguments.repetitions,
        "payload_check": "last" if arguments.check_last else "every replay",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
