# SPDX-License-Identifier: AGPL-3.0-only
"""Capture the fixed paging boot fixture with operator-provided QEMU builds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading


def capture(qemu, plugin, guest, directory, name, registers, expected_exit, *, memory='on', values='on', context='auto'):
    read_fd, write_fd = os.pipe()
    trace = directory / f'{name}.trace'
    errors = []

    def drain():
        try:
            with os.fdopen(read_fd, 'rb') as source, trace.open('wb') as destination:
                while chunk := source.read(65536):
                    destination.write(chunk)
        except BaseException as error:
            errors.append(str(error))

    thread = threading.Thread(target=drain)
    thread.start()
    argv = [str(qemu), '-accel', 'tcg', '-smp', '1', '-m', '16M',
            '-drive', f'file={guest},format=raw,if=floppy', '-display', 'none',
            '-serial', 'none', '-monitor', 'none', '-no-reboot', '-device',
            'isa-debug-exit,iobase=0xf4,iosize=0x04', '-plugin',
            f'{plugin},fd={write_fd},start=0x7c00,registers={registers},memory={memory},values={values},context={context}']
    child = None
    try:
        with (directory / f'{name}.stdout').open('wb') as out, (directory / f'{name}.stderr').open('wb') as err:
            try:
                child = subprocess.Popen(argv, pass_fds=(write_fd,), stdout=out, stderr=err, start_new_session=True)
            finally:
                os.close(write_fd)
            code = child.wait(timeout=30)
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        thread.join(timeout=5)
    if thread.is_alive() or errors:
        raise RuntimeError(f'Trace drain did not complete: {errors}')
    digest = hashlib.sha256()
    with trace.open('rb') as source:
        while chunk := source.read(65536): digest.update(chunk)
    result = {'argv': argv, 'exit_code': code, 'expected_exit': expected_exit,
              'trace_bytes': trace.stat().st_size,
              'trace_sha256': digest.hexdigest()}
    (directory / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
    if code != expected_exit:
        raise RuntimeError(f'{name}: exit {code}, expected {expected_exit}; see saved stderr')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('qemu', 'plugin', 'guest', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--context-only', action='store_true', help='Capture paging checkpoints with blocks only')
    parser.add_argument('--upstream', type=Path, help='Optional unmodified build for fallback/diagnostic checks')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.context_only:
        capture(args.qemu, args.plugin, args.guest, args.output, 'context-only', 'none', 33,
                memory='off', values='off', context='on')
        if args.upstream:
            capture(args.upstream, args.plugin, args.guest, args.output, 'public-context-only', 'none', 33,
                    memory='off', values='off', context='on')
        return
    # isa-debug-exit uses exit status 33 deliberately; ordinary workers still
    # report their target's actual exit status. Never infer episode success here.
    for name, registers, code in [('all', 'all', 33), ('selected', 'rax:r8', 33),
                                  ('missing', 'not_a_register', 125)]:
        capture(args.qemu, args.plugin, args.guest, args.output, name, registers, code)
    if args.upstream:
        capture(args.upstream, args.plugin, args.guest, args.output, 'public-memory', 'none', 33)
        capture(args.upstream, args.plugin, args.guest, args.output, 'missing-hook', 'general', 1)


if __name__ == '__main__':
    main()
