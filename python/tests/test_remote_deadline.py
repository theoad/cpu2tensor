# SPDX-License-Identifier: AGPL-3.0-only
"""Operator-run wall-clock deadlines against the real Linux worker."""

import os
import shlex
import subprocess
import time
import unittest

import test_remote
from cpu2tensor import Pool, TerminalReason, TraceTerminalError


@unittest.skipUnless(os.environ.get("CPU2TENSOR_REMOTE"), "Set CPU2TENSOR_REMOTE for real deadline checks")
class RemoteDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.remote = test_remote.RemoteTests("runTest")
        self.remote.setUp()
        self.addCleanup(self.remote.tearDown)
        self.target = f"{self.remote.build}/window_target"

    def child_pid(self):
        children = self.remote.ssh(f"ps -o pid= --ppid {self.remote.pid}").stdout.split()
        self.assertEqual(len(children), 1, "Expected the running QEMU child")
        return int(children[0])

    def check_expired(self, child, started, result=None):
        code, _, errors = self.remote.finish() if result is None else result
        self.assertNotEqual(code, 0)
        self.assertIn(b"Target exceeded --max-run-ms deadline", errors)
        self.assertLess(time.monotonic() - started, 10, "The deadline did not bound child cleanup")
        exists = self.remote.ssh(f"if test -d /proc/{child}; then echo alive; else echo gone; fi").stdout
        self.assertEqual(exists, b"gone\n", "The worker must reap its target, not leave a zombie")

    def check_target_deadline(self, mode, *, blocked=False, qemu=None):
        endpoint = self.remote.start(
            target=[self.target, mode], qemu=qemu, max_run_ms=1000,
            # Backpressure is deliberately independent of the much longer I/O timeout.
            timeout_ms=30000,
        )
        started = time.monotonic()
        with Pool([endpoint], device="cpu", timeout=8) as pool:
            batches = pool.read()
            first = next(batches)
            self.assertGreater(first.addresses.numel(), 0)
            child = self.child_pid()
            result = None
            if blocked:
                # Keep the socket open but do not receive. Filling its bounded
                # buffers must not turn the run deadline into an I/O timeout.
                self.remote.worker.wait(timeout=8)
                result = self.remote.finish()
            count = first.addresses.numel()
            with self.assertRaises(TraceTerminalError) as raised:
                for batch in batches:
                    count += batch.addresses.numel()
            if not blocked:
                self.assertEqual(raised.exception.outcome.reason, TerminalReason.MAX_RUN_DEADLINE)
                self.assertTrue(raised.exception.outcome.hello_reported)
            if mode == "spin" and not blocked:
                self.assertGreater(count, first.addresses.numel(), "The active target must stream before expiry")
            self.check_expired(child, started, result)

    def test_continuous_trace_does_not_refresh_deadline(self):
        self.check_target_deadline("spin")

    def test_idle_target_expires_without_new_frames(self):
        self.check_target_deadline("idle")

    def test_blocked_consumer_does_not_extend_deadline(self):
        self.check_target_deadline("spin", blocked=True)

    def test_sealed_trace_does_not_complete_before_child_exit(self):
        # This existing lifecycle fixture emits source_end + complete, then
        # waits forever. Its worker must still report an incomplete run.
        self.check_target_deadline("idle", qemu=f"{self.remote.build}/worker_test_target")

    def test_normal_completion_with_generous_deadline(self):
        endpoint = self.remote.start(target=[self.target], max_run_ms=5000)
        with Pool([endpoint], device="cpu", timeout=8) as pool:
            count = sum(batch.addresses.numel() for batch in pool.read())
        self.assertGreater(count, 0)
        self.assertEqual(self.remote.finish(), (0, b"window: checksum=663776\n", b""))

    def test_waiting_for_connection_is_outside_run_deadline(self):
        endpoint = self.remote.start(target=[self.target], max_run_ms=1000)
        time.sleep(1.3)
        self.assertIsNone(self.remote.worker.poll(), "The run deadline started before the client connected")
        children = self.remote.ssh(f"ps -o pid= --ppid {self.remote.pid} || true").stdout.split()
        self.assertEqual(children, [], "QEMU should launch only after connection")
        with Pool([endpoint], device="cpu", timeout=8) as pool:
            count = sum(batch.addresses.numel() for batch in pool.read())
        self.assertGreater(count, 0)
        self.assertEqual(self.remote.finish(), (0, b"window: checksum=663776\n", b""))

    def test_invalid_or_interactive_deadlines_fail_before_listening(self):
        cases = [
            (["--max-run-ms", value], b"Invalid numeric option")
            for value in ("0", "-1", "2147483648")
        ]
        cases.extend([
            (["--max-run-ms", "1000", "--stdio", "on"], b"requires observation-only capture"),
            (["--max-run-ms", "1000", "--system", "on", "--kernel-adapter", "on"],
             b"requires observation-only capture"),
        ])
        for options, diagnostic in cases:
            with self.subTest(options=options):
                # The outer timeout also cleans up a worker if a regression
                # accidentally lets it reach listen. No QEMU should launch.
                command = shlex.join([
                    "timeout", "3s", f"{self.remote.build}/cpu2tensor-worker",
                    "--qemu", "/bin/true", "--plugin", "/dev/null", "--input", "/dev/null",
                    *options, "--", "/bin/true",
                ])
                result = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", self.remote.host, command],
                    capture_output=True, timeout=12,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(diagnostic, result.stderr)
                self.assertNotIn(b"listening", result.stderr)


if __name__ == "__main__":
    unittest.main()
