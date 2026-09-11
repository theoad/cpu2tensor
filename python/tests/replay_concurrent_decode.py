# SPDX-License-Identifier: AGPL-3.0-only
"""Measure deterministic concurrent context-only Pool replay."""

import argparse
import json
import platform
import statistics

from concurrent_decode import expected_digest, replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--frames-per-source", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    arguments = parser.parse_args()
    measurements = []
    for worker_count in arguments.workers:
        wall_times = []
        for _ in range(arguments.repetitions):
            results, errors, elapsed = replay(worker_count, arguments.frames_per_source)
            expected = [expected_digest(identity, arguments.frames_per_source)
                        for identity in range(worker_count)]
            if errors or sorted(results, key=lambda result: result.identity) != expected:
                raise RuntimeError(f"incorrect replay at {worker_count} workers: {errors}")
            wall_times.append(elapsed)
        rows = worker_count * 2 * arguments.frames_per_source * _rows_per_frame()
        median = statistics.median(wall_times)
        measurements.append({
            "workers": worker_count,
            "wall_seconds": wall_times,
            "median_seconds": median,
            "rows_per_second": rows / median,
        })
    print(json.dumps({
        "host": platform.node(),
        "machine": platform.machine(),
        "frames_per_source": arguments.frames_per_source,
        "repetitions": arguments.repetitions,
        "measurements": measurements,
    }, indent=2))


def _rows_per_frame() -> int:
    return 256


if __name__ == "__main__":
    main()
