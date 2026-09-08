# SPDX-License-Identifier: AGPL-3.0-only
"""Acceptance capture for AlphaFlow's fixed, benign getpid observation window."""

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

import torch
from cpu2tensor import Pool
from cpu2tensor.examples.benchmark_capture import TraceCounter


def ssh_arguments(host, command):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, command]


def supervised_command(argv, pid_file):
    """The remote shell owns wait(), including termination before client connect."""
    path = shlex.quote(pid_file)
    return f'''umask 077
capture_pid=
cleanup() {{
    trap '' HUP INT TERM
    if test -n "$capture_pid"; then
        kill -TERM "$capture_pid" 2>/dev/null || true
        wait "$capture_pid" 2>/dev/null || true
    fi
    rm -f -- {path}
}}
trap 'cleanup; exit 143' HUP INT TERM
{shlex.join(argv)} &
capture_pid=$!
if ! printf '%s\\n' "$capture_pid" > {path}; then cleanup; exit 125; fi
printf 'cpu2tensor-worker-pid:%s\\n' "$capture_pid" >&2
wait "$capture_pid"
capture_status=$?
printf 'cpu2tensor-worker-reaped:%s\\n' "$capture_pid" >&2
rm -f -- {path}
exit "$capture_status"
'''


def wait_for_endpoint(worker, diagnostics, timeout=15):
    """Poll regular-file bytes; an incomplete diagnostic line cannot block us."""
    deadline = time.monotonic() + timeout
    position = 0
    pending = b''
    pid = endpoint = None
    while time.monotonic() < deadline:
        data = os.pread(diagnostics.fileno(), 65536, position)
        position += len(data)
        pending += data
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            text = line.decode(errors='replace')
            found_pid = re.fullmatch(r'cpu2tensor-worker-pid:([1-9][0-9]*)', text)
            if found_pid:
                pid = int(found_pid[1])
            found_endpoint = re.search(r'listening on (tcp://\S+)', text)
            if found_endpoint:
                endpoint = found_endpoint[1]
        pending = pending[-65536:]  # The complete diagnostics remain on disk.
        if pid is not None and endpoint is not None:
            return pid, endpoint
        if worker.poll() is not None:
            raise RuntimeError('Worker stopped during startup; see worker.stderr')
        time.sleep(0.05)
    raise TimeoutError('Worker startup timed out; see worker.stderr')


def stop_remote_worker(host, pid_file, expected_pid, diagnostics):
    """Terminate the owned child; pid-file removal follows its supervisor's wait."""
    path = shlex.quote(pid_file)
    expected = 'true' if expected_pid is None else f'test "$capture_pid" = {expected_pid}'
    command = f'''if ! test -r {path}; then exit 0; fi
read -r capture_pid < {path}
case "$capture_pid" in ''|*[!0-9]*) exit 1;; esac
if ! test "$capture_pid" -gt 1 || ! {expected}; then exit 1; fi
kill -TERM "$capture_pid" 2>/dev/null || true
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if ! test -e {path}; then exit 0; fi
    sleep 0.1
done
kill -KILL "$capture_pid" 2>/dev/null || true
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if ! test -e {path}; then exit 0; fi
    sleep 0.1
done
printf 'Remote supervisor did not confirm worker reap\\n' >&2
exit 1
'''
    subprocess.run(ssh_arguments(host, command), check=True, timeout=15,
                   stdout=diagnostics, stderr=diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='trail-x86')
    parser.add_argument('--address', default='10.0.0.17')
    parser.add_argument('--build', required=True)
    parser.add_argument('--qemu-root', required=True)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--remote-output', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rich', action='store_true')
    parser.add_argument('--batching', choices=('legacy', 'mixed'), default='legacy')
    parser.add_argument('--publication', choices=('pipe', 'ring'), default='pipe')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    serial = args.remote_output + '/getpid.serial'
    pid_file = args.remote_output + '/getpid-' + uuid.uuid4().hex + '.pid'
    argv = [args.build + '/cpu2tensor-worker', '--qemu', args.qemu_root + '/build/qemu-system-x86_64',
            '--plugin', args.build + '/libcpu2tensor_plugin.so', '--input', '/dev/null', '--system', 'on',
            '--registers', 'general' if args.rich else 'none', '--memory', 'on' if args.rich else 'off',
            '--memory-values', 'on' if args.rich else 'off', '--start-pc', '0x404f40', '--stop-pc', '0x404f50',
            '--batching', args.batching, '--publication', args.publication, '--timeout-ms', '180000',
            '--host', args.address, '--port', '0', '--', '-accel', 'tcg,thread=multi', '-cpu', 'max',
            '-m', '512', '-smp', '2', '-L', args.qemu_root + '/install/share/qemu',
            '-kernel', args.subject + '/bzImage', '-initrd', args.subject + '/evaluation.cpio.gz',
            '-append', 'console=ttyS0 rdinit=/init nokaslr panic=-1 loglevel=3 afuid=1000 afgid=1000 aftimeoutms=2000 afprog=39',
            '-nic', 'none', '-display', 'none', '-monitor', 'none', '-serial', 'file:' + serial, '-no-reboot']
    (args.output / 'command.json').write_text(json.dumps(argv, indent=2) + '\n')
    stdout_log = (args.output / 'worker.stdout').open('wb')
    stderr_log = (args.output / 'worker.stderr').open('w+b')
    worker = None
    remote_pid = None
    try:
        subprocess.run(ssh_arguments(args.host, shlex.join(['mkdir', '-p', args.remote_output])),
                       check=True, timeout=15, stdout=stdout_log, stderr=stderr_log)
        worker = subprocess.Popen(ssh_arguments(args.host, supervised_command(argv, pid_file)),
                                  stdout=stdout_log, stderr=stderr_log)
        remote_pid, endpoint = wait_for_endpoint(worker, stderr_log)
        counter = TraceCounter()
        retained = {}
        counts = dict(blocks=0, registers=0, memory=0, context=0)
        starts = 0
        batches = 0
        with (args.output / 'capture.bin').open('wb') as wire:
            class RecordedPool(Pool):
                def _receive(self, connection, size):
                    data = super()._receive(connection, size)
                    wire.write(data)
                    counter.feed(data)
                    return data
            started = time.monotonic()
            with RecordedPool([endpoint], device=args.device, timeout=180) as pool:
                for batch in pool.read():
                    batches += 1
                    rows = [('blocks', batch.addresses, batch.block_sequences)]
                    for name in ('registers', 'memory', 'context'):
                        table = getattr(batch, name)
                        if table is not None:
                            rows.append((name, table.pc, table.sequences))
                            if name + '.pc' not in retained:
                                for field in fields(table):
                                    tensor = getattr(table, field.name)
                                    if isinstance(tensor, torch.Tensor):
                                        retained[name + '.' + field.name] = (tensor, tensor.cpu().clone())
                    positions = []
                    for name, pc, sequence in rows:
                        counts[name] += pc.numel()
                        if not pc.numel():
                            continue
                        raw = pc.cpu()
                        assert not bool(((raw == 0x404f50) | (raw == 0x405080)
                                         | ((raw >= 0x4259e0) & (raw < 0x425a12))).any()), 'Stop/shutdown PC captured'
                        if name == 'blocks':
                            starts += int((raw == 0x404f40).sum())
                            if name not in retained:
                                retained[name] = (pc, raw.clone())
                        positions.extend(sequence.cpu().tolist() if sequence is not None else
                                         range(batch.first_sequence, batch.first_sequence + pc.numel()))
                    assert sorted(positions) == list(range(batch.first_sequence, batch.first_sequence + len(positions)))
            elapsed = time.monotonic() - started
        worker.wait(timeout=30)
        stderr = (args.output / 'worker.stderr').read_bytes()
        console = subprocess.run(ssh_arguments(args.host, shlex.join(['cat', serial])),
                                 check=True, timeout=20, capture_output=True).stdout
        (args.output / 'guest.serial').write_bytes(console)
        assert worker.returncode == 0, stderr.decode()
        assert b'capture stop reached at 0x404f50' in stderr
        assert b'AGENT: accepted bytes=2 fnv1a64=07ff8707b4c014a1' in console
        assert b'AGENT: candidate-exit status=0' in console
        assert b'AGENT: verifier-boundary' in console
        counter.finish()
        assert counter.ended == {0, 1}, counter.ended
        assert starts == 1, starts
        assert all(torch.equal(tensor.cpu(), saved) for tensor, saved in retained.values())
        if args.rich:
            assert all(counts[name] > 0 for name in ('registers', 'memory', 'context')), counts
        parameter = torch.ones(1, device=args.device, requires_grad=True)
        (retained['blocks'][0].remainder(65536).float().mean() * parameter).sum().backward()
        assert bool(torch.isfinite(parameter.grad).all())
        result = dict(host=args.host, learner_device=args.device, batching=args.batching, publication=args.publication,
                      rich=args.rich, seconds=elapsed, batches=batches, tensor_rows=counts,
                      ended_sources=sorted(counter.ended), retained_columns=len(retained), capture=counter.result())
    finally:
        active_failure = sys.exc_info()[0] is not None
        cleanup_errors = []
        try:
            if worker is not None:
                with (args.output / 'cleanup.log').open('ab') as diagnostics:
                    try:
                        stop_remote_worker(args.host, pid_file, remote_pid, diagnostics)
                    except BaseException as error:
                        cleanup_errors.append(f'Remote cleanup: {error}')
                    if not (args.output / 'guest.serial').exists():
                        # A failed fetch must not overwrite earlier diagnostics.
                        partial = args.output / 'guest.serial.partial'
                        try:
                            with partial.open('wb') as destination:
                                subprocess.run(ssh_arguments(args.host, shlex.join(['cat', serial])),
                                               check=True, timeout=20, stdout=destination, stderr=diagnostics)
                            partial.replace(args.output / 'guest.serial')
                        except BaseException as error:
                            cleanup_errors.append(f'Guest diagnostics: {error}')
        except BaseException as error:
            cleanup_errors.append(f'Cleanup diagnostics: {error}')
            if worker is not None:
                # Artifact creation must not prevent the remote cleanup attempt.
                try:
                    stop_remote_worker(args.host, pid_file, remote_pid, stderr_log)
                except BaseException as cleanup_error:
                    cleanup_errors.append(f'Remote cleanup fallback: {cleanup_error}')
        finally:
            try:
                if worker is not None and worker.poll() is None:
                    worker.terminate()
                    try:
                        worker.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait(timeout=5)
            except BaseException as error:
                cleanup_errors.append(f'Local SSH reap: {error}')
            stdout_log.close()
            stderr_log.close()
        if cleanup_errors:
            (args.output / 'cleanup-errors.json').write_text(json.dumps(cleanup_errors, indent=2) + '\n')
            if not active_failure:
                raise RuntimeError('; '.join(cleanup_errors))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
