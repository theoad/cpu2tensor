# SPDX-License-Identifier: AGPL-3.0-only
"""Replay recorded wire frames through loopback endpoints for learner isolation.

This is a transport/tensor benchmark, not a new QEMU or remote capture run.
The optional legacy conversion preserves every recorded event and source order.
"""

import argparse
import json
from pathlib import Path
import socket
import struct
import threading

from cpu2tensor import _native
from cpu2tensor.examples.benchmark_capture import TraceCounter
from cpu2tensor.examples.benchmark_pipeline import _write_failure, benchmark


HEADER = struct.Struct('<IHHIIQQ')
RUN = struct.Struct('<HHI')


def frames(path, legacy):
    with path.open('rb') as stream:
        while header := stream.read(HEADER.size):
            size = _native.payload_size(header)
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError('Truncated recorded frame')
            magic, version, kind, source, count, sequence, detail = HEADER.unpack(header)
            if kind == 1 and legacy:
                yield HEADER.pack(magic, version, kind, source, count, sequence, detail & ~(1 << 18))
            elif kind == 14 and legacy:
                offset = 0
                while offset < len(payload):
                    inner_kind, rows, size = RUN.unpack_from(payload, offset)
                    offset += RUN.size
                    yield HEADER.pack(magic, version, inner_kind, source, rows, sequence,
                                      0 if inner_kind == 2 else size) + payload[offset:offset + size]
                    offset += size
                    sequence += rows
            else:
                yield header + payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mode', choices=('drain', 'train'), default='drain')
    parser.add_argument('--batch-bytes', type=int, default=0)
    parser.add_argument('--max-seconds', type=float)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('Use between 1 and 16 replay workers')
    args.output = args.output.expanduser()
    if args.output.exists():
        parser.error('Output directory must not already exist')
    counter = TraceCounter()
    listeners = []
    threads = []
    errors = []
    endpoints = []
    result = None

    def serve(listener):
        try:
            with listener, listener.accept()[0] as connection:
                connection.settimeout(120)
                chunk = bytearray()
                for frame in frames(args.capture, args.legacy):
                    chunk.extend(frame)
                    if len(chunk) >= 65536:
                        connection.sendall(chunk)
                        chunk.clear()
                if chunk:
                    connection.sendall(chunk)
        except BaseException as error:
            errors.append(error)

    try:
        try:
            for frame in frames(args.capture, args.legacy):
                counter.feed(frame)
            counter.finish()
            for _ in range(args.workers):
                listener = socket.socket()
                listener.settimeout(120)
                listener.bind(('127.0.0.1', 0))
                listener.listen()
                listeners.append(listener)
                endpoints.append(f'tcp://127.0.0.1:{listener.getsockname()[1]}')
                thread = threading.Thread(target=serve, args=(listener,), daemon=True)
                thread.start()
                threads.append(thread)
            options = {}
            if args.batch_bytes:
                options['batch_bytes'] = args.batch_bytes
            if args.max_seconds is not None:
                options['max_seconds'] = args.max_seconds
            result = benchmark(endpoints, args.output, device=args.device, mode=args.mode, timeout=120,
                               capture_hosts=['loopback replay'] * args.workers, write_metrics=False, **options)
            for thread in threads:
                thread.join(5)
            if errors or any(thread.is_alive() for thread in threads):
                raise RuntimeError(f'Replay server failed: {errors}')
            result['replay'] = dict(source=str(args.capture), legacy_conversion=args.legacy,
                                    per_worker_wire=counter.result(),
                                    scope='Same-process loopback sender; no QEMU execution. Sender CPU is included in process usage.')
        finally:
            for listener in listeners:
                listener.close()
        (args.output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    except BaseException as error:
        _write_failure(args.output, error, result)
        raise


if __name__ == '__main__':
    main()
