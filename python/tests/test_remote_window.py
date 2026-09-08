# SPDX-License-Identifier: AGPL-3.0-only
"""Real capture-window boundaries, checked against the target's ELF symbols."""

import os
import re
import shlex
import time
import unittest

import test_remote
import torch
from cpu2tensor import Pool


@unittest.skipUnless(os.environ.get("CPU2TENSOR_REMOTE"), "Set CPU2TENSOR_REMOTE for real window checks")
class RemoteWindowTests(unittest.TestCase):
    def setUp(self):
        self.remote = test_remote.RemoteTests("runTest")
        self.remote.setUp()
        self.addCleanup(self.remote.tearDown)
        self.target = f"{self.remote.build}/window_target"
        output = self.remote.ssh(shlex.join(["nm", "-S", "-n", self.target])).stdout.decode()
        self.symbols = {
            name: (int(address, 16), int(size, 16))
            for address, size, name in re.findall(
                r"^([0-9a-f]+) ([0-9a-f]+) [Tt] (window_\w+)$", output, re.MULTILINE
            )
        }
        for name in ("start_marker", "stop_marker", "workload", "shutdown_tail"):
            self.assertIn(f"window_{name}", self.symbols)

    def start(self, mode, *, rich=False, batching="mixed", publication="ring"):
        return self.remote.start(
            target=[self.target, mode],
            start_pc=self.symbols["window_start_marker"][0],
            stop_pc=self.symbols["window_stop_marker"][0],
            registers="general" if rich else "none",
            memory="on" if rich else "off",
            values="on" if rich else "off",
            batching=batching,
            publication=publication,
        )

    def check_window(self, mode="normal", *, rich=False, batching="mixed", publication="ring", device="cpu"):
        endpoint = self.start(mode, rich=rich, batching=batching, publication=publication)
        blocks = []
        observed_pc = []
        next_sequence = {}
        retained = {}
        signal_counts = {"registers": 0, "memory": 0}
        with Pool([endpoint], device=device, timeout=60) as pool:
            for batch in pool.read():
                self.assertIsNotNone(batch.source)
                sequences = []
                rows = [("blocks", batch.addresses, batch.block_sequences)]
                for name in ("registers", "memory", "context"):
                    table = getattr(batch, name)
                    if table is not None:
                        rows.append((name, table.pc, table.sequences))
                        if name in signal_counts:
                            signal_counts[name] += table.pc.numel()
                        if name not in retained:
                            retained[name] = (table.pc, table.pc.cpu().clone())
                            if name in ("registers", "memory"):
                                self.assertIsNotNone(table.values)
                                retained[name + "_values"] = (table.values, table.values.cpu().clone())
                for name, pcs, row_sequences in rows:
                    values = pcs.cpu().tolist()
                    observed_pc.extend(values)
                    if name == "blocks":
                        blocks.extend(values)
                    if not values:
                        continue
                    if batching == "mixed":
                        self.assertIsNotNone(row_sequences)
                        self.assertEqual(row_sequences.numel(), len(values))
                        sequences.extend(row_sequences.cpu().tolist())
                    else:
                        sequences.extend(range(batch.first_sequence, batch.first_sequence + len(values)))
                first = next_sequence.get(batch.source, 0)
                self.assertEqual(sorted(sequences), list(range(first, first + len(sequences))))
                self.assertEqual(batch.first_sequence, first)
                next_sequence[batch.source] = first + len(sequences)
                if "blocks" not in retained and batch.addresses.numel():
                    retained["blocks"] = (batch.addresses, batch.addresses.cpu().clone())

        expected_checksum = 669888 if mode == "repeat" else 663776
        code, output, errors = self.remote.finish()
        self.assertEqual((code, output), (0, f"window: checksum={expected_checksum}\n".encode()))
        stop = self.symbols["window_stop_marker"][0]
        diagnostics = rf"cpu2tensor: capture stop reached at 0x{stop:x}\n"
        if publication == "ring":
            diagnostics += r"cpu2tensor: ring frames=\d+ full_waits=\d+\n"
        self.assertRegex(errors.decode(), "\\A" + diagnostics + "\\Z")
        self.assertEqual(blocks.count(self.symbols["window_start_marker"][0]), 1)
        self.assertEqual(blocks.count(self.symbols["window_workload"][0]), 1)
        for name in ("window_stop_marker", "window_shutdown_tail"):
            start, size = self.symbols[name]
            self.assertFalse(any(start <= pc < start + size for pc in observed_pc), name)
        if rich:
            self.assertTrue(all(count > 0 for count in signal_counts.values()))
        for tensor, saved in retained.values():
            self.assertTrue(torch.equal(tensor.cpu(), saved))
        parameter = torch.ones(1, device=device, requires_grad=True)
        features = retained["blocks"][0].remainder(65536).float()
        (features.mean() * parameter).sum().backward()
        self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_block_window_legacy_pipe(self):
        self.check_window(batching="legacy", publication="pipe")

    def test_block_window_mixed_ring(self):
        self.check_window()

    def test_rich_window_mixed_ring(self):
        self.check_window(rich=True)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_rich_window_retained_on_mps(self):
        self.check_window(rich=True, device="mps")

    def test_stop_before_start_is_ignored(self):
        self.check_window("stop-before-start", rich=True)

    def test_later_start_does_not_reopen_capture(self):
        self.check_window("repeat", rich=True)

    def test_missing_stop_is_incomplete(self):
        endpoint = self.start("missing-stop", rich=True)
        with Pool([endpoint], timeout=60) as pool:
            with self.assertRaisesRegex(RuntimeError, "Incomplete trace"):
                for _ in pool.read():
                    pass
        code, _, errors = self.remote.finish()
        self.assertNotEqual(code, 0)
        self.assertIn(b"stop", errors)

    def test_missing_start_is_incomplete(self):
        endpoint = self.start("missing-start")
        with Pool([endpoint], timeout=60) as pool:
            with self.assertRaisesRegex(RuntimeError, "Incomplete trace"):
                for _ in pool.read():
                    pass
        code, _, errors = self.remote.finish()
        self.assertNotEqual(code, 0)
        self.assertIn(b"start", errors)

    def test_ring_backpressure_drains_without_loss(self):
        data = b"abcde\n" * 20000
        endpoint = self.remote.start(data, publication="ring")
        with Pool([endpoint], timeout=60) as pool:
            batches = pool.read()
            first = next(batches)
            saved = first.addresses.clone()
            time.sleep(0.5)
            count = first.addresses.numel()
            for batch in batches:
                self.assertEqual(batch.first_sequence, count)
                count += batch.addresses.numel()
            self.assertTrue(torch.equal(first.addresses, saved))
        code, output, errors = self.remote.finish()
        self.assertEqual((code, output), (0, f"bytes={len(data)} sum={sum(data)}\n".encode()))
        waits = re.search(rb"ring frames=\d+ full_waits=(\d+)", errors)
        self.assertIsNotNone(waits)
        self.assertGreater(int(waits[1]), 0)

    def test_ring_disconnect_reaps_target(self):
        endpoint = self.remote.start(b"abcde\n" * 100000, publication="ring")
        with Pool([endpoint]) as pool:
            batches = pool.read()
            next(batches)
            children = self.remote.ssh(f"ps -o pid= --ppid {self.remote.pid}").stdout.split()
            self.assertEqual(len(children), 1)
            child = int(children[0])
        code, _, errors = self.remote.finish()
        self.assertNotEqual(code, 0)
        self.assertIn(b"disconnected", errors)
        self.assertEqual(self.remote.ssh(f"test ! -d /proc/{child} && echo gone").stdout, b"gone\n")


if __name__ == "__main__":
    unittest.main()
