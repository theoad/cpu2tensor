# SPDX-License-Identifier: AGPL-3.0-only
"""Real input boundaries on the operator's ARM worker."""
import os
import re
import shlex
import unittest

import torch
import test_remote
from cpu2tensor import StdioEnv
from cpu2tensor.examples.learn_stdio import observe, finish


@unittest.skipUnless(os.environ.get("CPU2TENSOR_REMOTE"), "Set CPU2TENSOR_REMOTE for real stdin checks")
class RemoteStdioTests(unittest.TestCase):
    setUp = test_remote.RemoteTests.setUp
    ssh = test_remote.RemoteTests.ssh
    start = test_remote.RemoteTests.start
    finish_worker = test_remote.RemoteTests.finish
    tearDown = test_remote.RemoteTests.tearDown

    def entries(self):
        symbols = self.ssh(shlex.join(["nm", "-n", f"{self.build}/stdio_digits"])).stdout.decode()
        return torch.tensor([int(re.search(rf"^([0-9a-f]+) T digit_{i}$", symbols, re.MULTILINE)[1], 16)
                             for i in range(10)], dtype=torch.int64)

    def child(self):
        children = self.ssh(f"ps -o pid= --ppid {self.pid}").stdout.split()
        self.assertEqual(len(children), 1)
        return int(children[0])

    def test_paused_reset_and_retained_observation(self):
        # Repeated workers keep the listener open; fixed nonzero port is required.
        port = int(self.ssh("python3 -c 'import socket; s=socket.socket(); s.bind((\"0.0.0.0\",0)); print(s.getsockname()[1])'").stdout)
        endpoint = self.start(stdio=True, episodes=2, port=port, target=[f"{self.build}/stdio_digits"])
        with StdioEnv(endpoint) as env:
            entries = self.entries()
            first = observe(env.reset(), entries)
            saved = first.clone()
            old_child = self.child()
            state = self.ssh(f"cat /proc/{old_child}/status").stdout.decode()
            self.assertRegex(state, r"State:\s+T")
            second = observe(env.reset(), entries)
            self.assertTrue(torch.equal(first, saved))
            gone = self.ssh(f"test ! -d /proc/{old_child} && echo gone").stdout
            self.assertEqual(gone, b"gone\n")
            self.assertEqual(finish(env, (int(second.argmax()) + 1) % 10), 0.0)
            self.assertEqual(env.exit_code, 1)
        code, _, errors = self.finish_worker()
        self.assertEqual((code, errors), (0, b""))

    def test_two_reads_and_rich_signals(self):
        endpoint = self.start(stdio=True, target=[f"{self.build}/stdio_test_target", "twice"],
                              registers="general", memory="on", values="on")
        with StdioEnv(endpoint, device="mps") as env:
            initial = list(env.reset())
            self.assertTrue(env.needs_input)
            self.assertTrue(any(batch.registers is not None for batch in initial))
            self.assertTrue(any(batch.memory is not None for batch in initial))
            for _ in env.step(b"a\n"): pass
            self.assertTrue(env.needs_input)
            for _ in env.step(b"b\n"): pass
            self.assertEqual(env.exit_code, 0)
        self.assertEqual(self.finish_worker()[0], 0)

    def test_undelivered_input_is_rejected(self):
        endpoint = self.start(stdio=True, target=[f"{self.build}/stdio_test_target", "short"])
        with StdioEnv(endpoint) as env:
            for _ in env.reset(): pass
            with self.assertRaises(RuntimeError):
                for _ in env.step(b"a\n"): pass
        code, _, errors = self.finish_worker()
        self.assertNotEqual(code, 0)
        self.assertIn(b"previous action", errors)

    def test_stdin_remapping_is_rejected(self):
        endpoint = self.start(stdio=True, target=[f"{self.build}/stdio_test_target", "close"])
        with StdioEnv(endpoint) as env:
            with self.assertRaises(RuntimeError):
                for _ in env.reset(): pass
        code, _, errors = self.finish_worker()
        self.assertNotEqual(code, 0)
        self.assertIn(b"stdin remapping", errors)

    def test_sigcont_handler_is_rejected(self):
        endpoint = self.start(stdio=True, target=[f"{self.build}/stdio_test_target", "sigcont"])
        with StdioEnv(endpoint) as env:
            with self.assertRaises(RuntimeError):
                for _ in env.reset(): pass
        code, _, errors = self.finish_worker()
        self.assertNotEqual(code, 0)
        self.assertIn(b"SIGCONT handlers", errors)

    def test_input_deadline_reaps_paused_target(self):
        endpoint = self.start(stdio=True, target=[f"{self.build}/stdio_digits"], timeout_ms=300)
        with StdioEnv(endpoint) as env:
            for _ in env.reset(): pass
            child = self.child()
            code, _, errors = self.finish_worker()
            self.assertNotEqual(code, 0)
            self.assertIn(b"stdin action", errors)
            self.assertEqual(self.ssh(f"test ! -d /proc/{child} && echo gone").stdout, b"gone\n")
