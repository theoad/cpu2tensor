# SPDX-License-Identifier: AGPL-3.0-only
"""Real worker and plugin capability rejection for explicit paging context."""
import os
import shlex
import subprocess
import unittest

import test_remote


@unittest.skipUnless(os.environ.get("CPU2TENSOR_REMOTE"), "Set CPU2TENSOR_REMOTE for backend checks")
class RemoteContextTests(unittest.TestCase):
    def setUp(self):
        self.remote = test_remote.RemoteTests("runTest")
        self.remote.setUp()
        self.addCleanup(self.remote.tearDown)
        self.target = f"{self.remote.build}/window_target"

    def test_context_option_rejects_unsupported_user_mode(self):
        command = shlex.join([
            f"{self.remote.build}/cpu2tensor-worker", '--context', 'on',
            '--qemu', self.remote.qemu, '--plugin', f"{self.remote.build}/libcpu2tensor_plugin.so",
            '--input', '/dev/null', '--', self.target,
        ])
        result = subprocess.run(['ssh', self.remote.host, command], capture_output=True, timeout=12)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b'--context on requires x86 system emulation', result.stderr)
        self.assertNotIn(b'listening', result.stderr)
        # Direct plugin use must enforce its backend capability too. This
        # unsupported user backend rejects the option before inspecting fd=3.
        command = shlex.join([
            self.remote.qemu, '-plugin',
            f"{self.remote.build}/libcpu2tensor_plugin.so,fd=3,registers=none,memory=off,context=on",
            self.target,
        ])
        result = subprocess.run(['ssh', self.remote.host, command], capture_output=True, timeout=12)
        self.assertEqual(result.returncode, 1)
        self.assertIn(b'context=on requires x86 system emulation', result.stderr)

