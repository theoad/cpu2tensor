# SPDX-License-Identifier: AGPL-3.0-only
"""Real kernel integration; operators supply all paths and SSH connectivity."""
import os
import re
import select
import shlex
import socket
import struct
import subprocess
import tempfile
import time
import unittest

import torch
from cpu2tensor import KernelEnv, Pool


@unittest.skipUnless(os.environ.get('CPU2TENSOR_KERNEL_REMOTE'), 'Set CPU2TENSOR_KERNEL_REMOTE for real kernel checks')
class RemoteKernelTests(unittest.TestCase):
    def setUp(self):
        self.host = os.environ['CPU2TENSOR_KERNEL_REMOTE']
        self.address = os.environ['CPU2TENSOR_KERNEL_ADDRESS']
        self.build = os.environ['CPU2TENSOR_KERNEL_BUILD']
        self.qemu = os.environ['CPU2TENSOR_KERNEL_QEMU']
        self.image = os.environ['CPU2TENSOR_KERNEL_IMAGE']
        self.initramfs = os.environ['CPU2TENSOR_KERNEL_INITRAMFS']
        self.worker = None
        self.log = tempfile.TemporaryFile()
        self.init = os.environ.get('CPU2TENSOR_KERNEL_INIT', f'{self.build}/kernel_init')
        symbols = self.ssh(['nm', '-n', self.init]).decode()
        self.begin = int(re.search(r'^([0-9a-f]+) T cpu2tensor_capture_begin$', symbols, re.M)[1], 16)
        self.action_begin = int(re.search(r'^([0-9a-f]+) T cpu2tensor_action_begin$', symbols, re.M)[1], 16)
        self.action_end = int(re.search(r'^([0-9a-f]+) T cpu2tensor_action_end$', symbols, re.M)[1], 16)
        self.action_abort = int(re.search(r'^([0-9a-f]+) T cpu2tensor_action_abort$', symbols, re.M)[1], 16)
        self.parallel = int(re.search(r'^([0-9a-f]+) T cpu2tensor_parallel_memory$', symbols, re.M)[1], 16)
        sized_symbols = self.ssh(['nm', '-S', '-n', self.init]).decode()
        compute = re.search(r'^([0-9a-f]+) ([0-9a-f]+) T cpu2tensor_compute$',
                            sized_symbols, re.M)
        self.compute = int(compute[1], 16)
        self.compute_end = self.compute + int(compute[2], 16)

    def ssh(self, args):
        return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', self.host, shlex.join(args)], timeout=20)

    def start(self, *, interactive=True, episodes=1, rich=False, timeout=30000, start=None, full_boot=False, cpus='2', stop=None, batching='legacy', publication='pipe', workload_bytes=None, max_run_ms=None, context='auto', action_windows=False, observation_reduction=False):
        port = int(self.ssh(['python3', '-c', 'import socket; s=socket.socket(); s.bind(("0.0.0.0",0)); print(s.getsockname()[1])']))
        args = [f'{self.build}/cpu2tensor-worker', '--qemu', self.qemu, '--plugin', f'{self.build}/libcpu2tensor_plugin.so',
                '--system', 'on', '--host', self.address, '--port', str(port), '--episodes', str(episodes),
                '--timeout-ms', str(timeout), '--registers', 'general' if rich else 'none',
                '--memory', 'on' if rich else 'off', '--memory-values', 'on' if rich else 'off',
                '--batching', batching, '--publication', publication, '--context', context]
        if max_run_ms is not None:
            args += ['--max-run-ms', str(max_run_ms)]
        if not full_boot:
            args += ['--start-pc', hex(self.begin if start is None else start)]
        if stop is not None:
            args += ['--stop-pc', hex(stop)]
        if interactive:
            args += ['--kernel-adapter', 'on']
            if action_windows:
                args += ['--window-start-pc', hex(self.action_begin),
                         '--window-end-pc', hex(self.action_end),
                         '--window-abort-pc', hex(self.action_abort),
                         '--reducer', 'block-transitions',
                         '--transition-capacity', '4096']
        else:
            # This example guest always reports its finite workload over ttyS1.
            # The managed observation protocol validates those records without
            # exposing action frames to Pool.
            args += ['--kernel-protocol', 'on']
            if observation_reduction:
                args += ['--blocks', 'off', '--reducer', 'block-transitions',
                         '--transition-capacity', '4096']
        args += ['--', '-accel', 'tcg,thread=multi', '-smp', cpus, '-m', '256M', '-nic', 'none', '-no-reboot',
                 '-kernel', self.image, '-initrd', self.initramfs,
                 '-append', 'console=ttyS0 rdinit=/init panic=-1 cpu2tensor.mode=' + ('interactive' if interactive else 'observe')
                 + (' cpu2tensor.parallel_timeout=300' if rich else '')]
        if workload_bytes is not None:
            args[-1] += f' cpu2tensor.bytes={workload_bytes}'
        self.arguments = args
        command = 'echo cpu2tensor-pid:$$; exec ' + shlex.join(args)
        # A full boot can fill an unread stdout pipe. Keep both diagnostic
        # channels in a file, independently of tensor consumption.
        self.worker = subprocess.Popen(['ssh', self.host, command], stdout=self.log, stderr=self.log)
        self.pid = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            text = os.pread(self.log.fileno(), 4096, 0)
            found = re.search(rb'cpu2tensor-pid:(\d+)', text)
            if found:
                self.pid = int(found[1])
            if b'listening on' in text and self.pid is not None:
                return f'tcp://{self.address}:{port}'
            if self.worker.poll() is not None:
                self.fail('Worker exited during startup')
            time.sleep(.02)
        self.fail('Worker did not listen')

    def children(self):
        return [int(value) for value in self.ssh(['ps', '-o', 'pid=', '--ppid', str(self.pid)]).split()]

    def gone(self, pid):
        result = subprocess.run(['ssh', self.host, shlex.join(['test', '!', '-d', f'/proc/{pid}'])])
        self.assertEqual(result.returncode, 0)

    def finish(self, expected=0):
        self.worker.communicate(timeout=20)
        self.log.seek(0)
        text = self.log.read().decode(errors='replace')
        self.assertEqual(self.worker.returncode, expected, text[-3000:])
        return text

    def tearDown(self):
        if self.worker is not None and self.worker.poll() is None and self.pid is not None:
            subprocess.run(['ssh', self.host, shlex.join(['kill', str(self.pid)])], capture_output=True, timeout=20)
            self.worker.communicate(timeout=20)
        if self.worker is not None and self.worker.returncode:
            self.log.seek(0)
            print(self.log.read().decode(errors='replace')[-4000:])
        self.log.close()

    @staticmethod
    def drain(batches):
        return sum(batch.addresses.numel() for batch in batches)

    def test_context_only_associates_blocks_on_both_cpus(self):
        endpoint = self.start(interactive=False, context='on', batching='mixed',
                              workload_bytes=64, max_run_ms=120000, timeout=120000)
        previous = {}
        blocks = {}
        changes = {}
        with Pool([endpoint], timeout=120, batch_bytes=65536) as pool:
            for batch in pool.read():
                self.assertIsNone(batch.registers)
                self.assertIsNone(batch.memory)
                source = batch.source
                table = batch.context
                old = previous.get(source)
                sequences = [] if old is None else [old]
                if table is not None:
                    self.assertTrue(bool((table.known == 63).all()))
                    sequences += table.sequences.tolist()
                    changes[source] = changes.get(source, 0) + table.pc.numel()
                    previous[source] = sequences[-1]
                if batch.addresses.numel():
                    self.assertTrue(sequences, 'Blocks arrived without source context')
                    available = torch.tensor(sequences, dtype=torch.int64)
                    positions = torch.searchsorted(available, batch.block_sequences, right=True) - 1
                    self.assertTrue(bool((positions >= 0).all()))
                    blocks[source] = blocks.get(source, 0) + batch.addresses.numel()
        self.assertEqual(set(blocks), {0, 1})
        self.assertEqual(set(changes), {0, 1})
        self.assertIn('C2T {"event":"complete","steps":4,"ok":true}', self.finish())

    def test_observation_total_deadline_reaps_guest(self):
        endpoint = self.start(interactive=False, full_boot=True, max_run_ms=1000)
        started = time.monotonic()
        with Pool([endpoint], timeout=10) as pool:
            batches = pool.read()
            first = next(batches)
            self.assertGreater(first.addresses.numel(), 0)
            children = self.children()
            self.assertEqual(len(children), 1)
            with self.assertRaisesRegex(RuntimeError, 'Incomplete trace'):
                for _ in batches:
                    pass
        self.assertLess(time.monotonic() - started, 10)
        self.assertIn('Target exceeded --max-run-ms deadline', self.finish(expected=1))
        self.gone(children[0])

    def test_bounded_observation_reduction(self):
        endpoint = self.start(
            interactive=False, context='on', observation_reduction=True,
            workload_bytes=64, max_run_ms=120000, timeout=120000,
        )
        summaries = {}
        tensor_bytes = 0
        with Pool([endpoint], timeout=120, batch_bytes=65536) as pool:
            for batch in pool.read():
                self.assertEqual(batch.addresses.numel(), 0)
                if batch.observation_transitions is not None:
                    table = batch.observation_transitions
                    tensor_bytes += sum(column.numel() * column.element_size() for column in (
                        table.from_addresses, table.destinations, table.counts,
                    ))
                if batch.observation_context is not None:
                    table = batch.observation_context
                    tensor_bytes += sum(
                        getattr(table, name).numel() * getattr(table, name).element_size()
                        for name in table.__dataclass_fields__
                    )
                if batch.observation_summary is not None:
                    summaries[batch.source] = batch.observation_summary
        self.assertEqual(set(summaries), {0, 1})
        for summary in summaries.values():
            self.assertEqual(summary.transitions, max(0, summary.blocks - 1))
            self.assertEqual(
                summary.retained_contexts + summary.context_overflow,
                summary.context_changes,
            )
        self.assertLessEqual(tensor_bytes, 2 * 4096 * (24 + 72))
        self.assertIn('"ok":true', self.finish())

    def test_rich_parallel_values_and_retained_device_storage(self):
        endpoint = self.start(rich=True, timeout=300000)
        kinds = set()
        sources = set()
        contexts = {}
        mapped_sources = set()
        retained = None
        def check_context(batch):
            if batch.context is not None:
                self.assertEqual(batch.context.known.tolist(), [63])
                contexts[batch.source] = batch.first_sequence
            if batch.memory is not None:
                table = batch.memory
                self.assertTrue(bool((table.context_sequences == contexts[batch.source]).all()))
                self.assertTrue(bool((table.mapped_sizes <= table.sizes).all()))
                if bool((table.mapping_flags & 1).any()):
                    mapped_sources.add(batch.source)
        with KernelEnv(endpoint, timeout=300) as env:
            for batch in env.reset():
                check_context(batch)
                if batch.registers is not None:
                    kinds.add('registers')
                    self.assertNotIn('eflags', batch.registers.names.values())
                if batch.memory is not None:
                    kinds.add('memory')
                    self.assertIsNotNone(batch.memory.values)
                    if retained is None:
                        retained = (batch.memory.values, batch.memory.values.clone())
            for batch in env.step(b'parallel 17 4096\n'):
                check_context(batch)
                if batch.addresses.numel() and bool((batch.addresses == self.parallel).any()):
                    sources.add(batch.source)
            self.assertEqual(sources, {0, 1})
            self.assertEqual(mapped_sources, {0, 1})
            self.assertEqual(kinds, {'registers', 'memory'})
            self.assertEqual((env.result['checksum0'], env.result['checksum1']), (520419, 522544))
            self.assertTrue(torch.equal(*retained))
            if torch.backends.mps.is_available():
                copied = retained[0].to('mps')
                self.assertTrue(torch.equal(copied.cpu(), retained[1]))
                weight = torch.ones(16, device='mps', requires_grad=True)
                (copied.float() @ weight).sum().backward()
                self.assertTrue(torch.isfinite(weight.grad).all())
            self.drain(env.step(b'quit\n'))
            self.assertEqual(env.exit_code, 0)
        self.assertIn('"ok":true', self.finish())

    def test_action_windows_ignore_transport_size_and_distinguish_bodies(self):
        endpoint = self.start(action_windows=True, timeout=120000)
        commands = (b'compute 257 0\n', b'compute 0000000257 64\n',
                    b'compute 521 0\n')
        body_transitions = []
        window_row_counts = []
        report_sizes = []
        with KernelEnv(endpoint, timeout=120) as env:
            self.drain(env.reset())
            child = self.children()[0]
            for command in commands:
                batches = list(env.step(command))
                window_rows = tuple(sorted(
                    (batch.source, int(source), int(destination), int(count))
                    for batch in batches if batch.transitions is not None
                    for source, destination, count in zip(
                        batch.transitions.from_addresses,
                        batch.transitions.destinations,
                        batch.transitions.counts, strict=True)
                ))
                window_row_counts.append(len(window_rows))
                body_transitions.append(tuple(
                    row for row in window_rows
                    if self.compute <= row[1] < self.compute_end and
                    self.compute <= row[2] < self.compute_end
                ))
                report_sizes.append(len(str(env.result)))
                summary = next(batch.transition_window for batch in batches
                               if batch.transition_window is not None)
                self.assertTrue(summary.complete)
            rejected = list(env.step(b'compute 0 0\n'))
            aborted = next(batch.transition_window for batch in rejected
                           if batch.transition_window is not None)
            self.assertEqual(aborted.status, 'aborted')
            self.assertFalse(aborted.complete)
            self.assertEqual(env.event['event'], 'ready')
            self.assertIsNone(env.result)
            self.drain(env.step(b'quit\n'))
            self.assertEqual(env.exit_code, 0)
        self.assertNotEqual(len(commands[0]), len(commands[1]))
        self.assertNotEqual(report_sizes[0], report_sizes[1])
        self.assertEqual(body_transitions[0], body_transitions[1])
        self.assertNotEqual(body_transitions[1], body_transitions[2])
        self.assertTrue(body_transitions[0])
        self.assertIn('"ok":true', self.finish())
        self.gone(child)
        print(f'Action windows on {self.host}: commands {[len(value) for value in commands]}, '
              f'reports {report_sizes}, window_rows={window_row_counts}, '
              f'body_transitions={[sum(row[3] for row in value) for value in body_transitions]}, '
              'abort=aborted')

    def test_paused_and_midstream_reset_reap_old_guests(self):
        endpoint = self.start(episodes=3)
        with KernelEnv(endpoint) as env:
            self.drain(env.reset())
            first = self.children()[0]
            # At a fence no trace bytes may arrive while the model is deciding.
            self.assertFalse(select.select([env._connection], [], [], .2)[0])
            second_run = env.reset()
            next(second_run)
            second = self.children()[0]
            self.gone(first)
            # Cancel while native forwarding may be blocked in a socket write.
            time.sleep(.2)
            self.drain(env.reset())
            self.gone(second)
            self.drain(env.step(b'quit\n'))
            self.assertEqual(env.exit_code, 0)
        self.finish()

    def test_partial_action_deadline_reaps_guest(self):
        endpoint = self.start(timeout=5000)
        with KernelEnv(endpoint, timeout=10) as env:
            self.drain(env.reset())
            child = self.children()[0]
            env._connection.sendall(struct.pack('<I', 10) + b'm')
            time.sleep(3)
            env._connection.sendall(b'e')
            text = self.finish(expected=1)
            self.assertIn('action', text.lower())
            self.gone(child)

    def test_observation_backpressure_and_both_sources(self):
        endpoint = self.start(interactive=False, rich=True, timeout=120000)
        sources = set()
        with Pool([endpoint], timeout=120) as pool:
            batches = pool.read()
            next(batches)
            time.sleep(.5)
            child = self.children()[0]
            wait = self.ssh(['python3', '-c',
                'import glob; print([open(p).read() for p in glob.glob("/proc/' + str(child) + '/task/*/wchan")])']).decode()
            self.assertIn('pipe', wait)
            for batch in batches:
                if batch.addresses.numel() and bool((batch.addresses == self.parallel).any()): sources.add(batch.source)
        self.assertEqual(sources, {0, 1})
        self.assertIn('"ok":true', self.finish())

    def test_unreached_capture_start_is_incomplete(self):
        endpoint = self.start(interactive=False, start=0x123456789)
        with Pool([endpoint]) as pool:
            with self.assertRaises(RuntimeError):
                self.drain(pool.read())
        self.assertIn('before the requested capture start', self.finish(expected=1))

    def test_full_boot_observation(self):
        endpoint = self.start(interactive=False, full_boot=True, timeout=120000)
        first = None
        kernel = False
        sources = set()
        count = 0
        with Pool([endpoint], timeout=120) as pool:
            for batch in pool.read():
                count += batch.addresses.numel()
                if first is None and batch.addresses.numel(): first = int(batch.addresses[0])
                kernel |= bool((batch.addresses < 0).any())
                if bool((batch.addresses == self.parallel).any()): sources.add(batch.source)
        self.assertNotEqual(first, self.begin)
        self.assertTrue(kernel)
        self.assertEqual(sources, {0, 1})
        self.assertIn('"ok":true', self.finish())
        print(f'Full boot on {self.host}: {count} blocks, first PC {first:#x}')

    def test_kernel_adapter_rejects_hotplug_slots(self):
        endpoint = self.start(cpus='2,maxcpus=3')
        with KernelEnv(endpoint) as env:
            with self.assertRaises(RuntimeError):
                self.drain(env.reset())
        self.assertIn('fixed vCPU count', self.finish(expected=1))
