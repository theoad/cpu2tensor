# SPDX-License-Identifier: AGPL-3.0-only
"""Measure supplied kernel workers through the public tensor Pool.

Use the CPU2TENSOR_KERNEL_* variables documented by test_remote_kernel. The
operator supplies an archive and its matching start/stop symbols explicitly.
"""

import argparse
import json
import os
from pathlib import Path

import test_remote_kernel
from cpu2tensor.examples.benchmark_pipeline import _write_failure, benchmark


def _clean_up_worker(worker, index, output):
    """Reap first, then copy diagnostics without moving the writer's file offset."""
    errors = []
    descriptor = None
    try:
        descriptor = os.dup(worker.log.fileno())
    except BaseException as error:
        errors.append(error)
    try:
        worker.tearDown()
    except BaseException as error:
        errors.append(error)
    finally:
        try:
            output.mkdir(parents=True, exist_ok=True)
            if descriptor is not None:
                with (output / f'worker-{index}.log').open('wb') as destination:
                    offset = 0
                    while data := os.pread(descriptor, 65536, offset):
                        destination.write(data)
                        offset += len(data)
            if hasattr(worker, 'arguments'):
                (output / f'worker-{index}.command.json').write_text(json.dumps(worker.arguments, indent=2) + '\n')
        except BaseException as error:
            errors.append(error)
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as error:
                    errors.append(error)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bytes', type=int, default=64)
    parser.add_argument('--batch-bytes', type=int, default=0)
    parser.add_argument('--max-seconds', type=float, default=120)
    parser.add_argument('--mode', choices=('drain', 'train'), default='drain')
    parser.add_argument('--start-pc', type=lambda value: int(value, 0), required=True)
    parser.add_argument('--stop-pc', type=lambda value: int(value, 0), required=True)
    parser.add_argument('--publication', choices=('pipe', 'ring'), default='pipe')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('Use between 1 and 4 workers for this small host example')
    args.output = args.output.expanduser()
    if args.output.exists():
        parser.error('Output directory must not already exist')
    workers = []
    endpoints = []
    result = None
    failure = None
    cleanup_errors = []
    try:
        for _ in range(args.workers):
            worker = test_remote_kernel.RemoteKernelTests('runTest')
            worker.setUp()
            workers.append(worker)
            endpoints.append(worker.start(interactive=False, rich=True, timeout=300000,
                                          start=args.start_pc, stop=args.stop_pc,
                                          batching='mixed', publication=args.publication, workload_bytes=args.bytes))
        result = benchmark(endpoints, args.output, device=args.device, mode=args.mode,
                           timeout=30, capture_hosts=[worker.host for worker in workers],
                           batch_bytes=args.batch_bytes, max_seconds=args.max_seconds,
                           write_metrics=False)
        for worker in workers:
            log = worker.finish()
            if 'C2T {"event":"complete","steps":4,"ok":true}' not in log:
                raise RuntimeError('Guest did not report successful completion: ' + log[-4000:])
            if 'capture stop reached' not in log:
                raise RuntimeError('Guest did not reach the capture stop: ' + log[-4000:])
    except BaseException as error:
        failure = error
    finally:
        for index, worker in enumerate(workers):
            # A broken connection or artifact write must not skip later workers.
            cleanup_errors.extend(_clean_up_worker(worker, index, args.output))
    try:
        if failure is not None:
            raise failure
        if cleanup_errors:
            raise RuntimeError('Worker cleanup failed: ' + str(cleanup_errors[0])) from cleanup_errors[0]
        (args.output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    except BaseException as error:
        partial = dict(result or {})
        if cleanup_errors:
            partial['cleanup_errors'] = [f'{type(item).__name__}: {item}' for item in cleanup_errors]
        _write_failure(args.output, error, partial)
        raise


if __name__ == '__main__':
    main()
