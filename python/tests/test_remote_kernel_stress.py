# SPDX-License-Identifier: AGPL-3.0-only
"""Opt-in sustained check using only the package's ordinary kernel adapter."""
import concurrent.futures
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from cpu2tensor import KernelEnv
import test_remote_kernel


@unittest.skipUnless(os.environ.get("CPU2TENSOR_KERNEL_STRESS"),
                     "Set CPU2TENSOR_KERNEL_STRESS for the four-worker kernel check")
class RemoteKernelStressTests(unittest.TestCase):
    def test_four_workers_complete_repeated_episodes(self):
        episodes = int(os.environ.get("CPU2TENSOR_KERNEL_STRESS_EPISODES", "32"))
        self.assertGreater(episodes, 0)
        output = Path(tempfile.mkdtemp(prefix="kernel-stress-",
                                      dir=os.environ["CPU2TENSOR_KERNEL_STRESS_OUTPUT"]))
        print(f"Kernel stress logs: {output}", flush=True)

        def lane(index):
            remote = test_remote_kernel.RemoteKernelTests("runTest")
            remote.setUp()
            report = {"lane": index, "episodes": 0, "blocks": 0, "actions": 0}
            started = time.monotonic()
            try:
                endpoint = remote.start(episodes=episodes, timeout=180000,
                                        context="on", batching="mixed")
                report["arguments"] = remote.arguments
                with KernelEnv(endpoint, timeout=180) as env:
                    for episode in range(episodes):
                        report["blocks"] += remote.drain(env.reset())
                        child = remote.children()[0]
                        commands = (b"getpid\n", b"memory 17 64\n", b"parallel 17 64\n")
                        for step, command in enumerate(commands):
                            report["blocks"] += remote.drain(env.step(command))
                            self.assertIsNotNone(env.result)
                            self.assertEqual(env.result["action"], command.split()[0].decode())
                            self.assertEqual(env.result["step"], step)
                            if step == 0:
                                self.assertEqual(env.result["value"], 1)
                            elif step == 2:
                                self.assertEqual({env.result["cpu0"], env.result["cpu1"]}, {0, 1})
                            report["actions"] += 1
                        report["blocks"] += remote.drain(env.step(b"quit\n"))
                        self.assertEqual(env.exit_code, 0)
                        self.assertTrue(env.event["ok"])
                        remote.gone(child)
                        report["episodes"] += 1
                        if report["episodes"] % 8 == 0:
                            print(f"Lane {index}: {report['episodes']}/{episodes} episodes", flush=True)
                remote.finish()
                report["ok"] = True
            finally:
                report["seconds"] = time.monotonic() - started
                # Preserve all diagnostics, including a failure before the first
                # episode. tearDown still owns cancellation and temporary files.
                saved_log = os.dup(remote.log.fileno())
                try:
                    remote.tearDown()
                finally:
                    with os.fdopen(saved_log, "rb") as log:
                        log.seek(0)
                        (output / f"lane-{index}.log").write_bytes(log.read())
                    (output / f"lane-{index}.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

        # Each lane owns its worker, endpoint and adapter. There is no shared
        # guest state, action stream or all-worker step barrier.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(lane, index) for index in range(4)]
            reports = [future.result() for future in futures]
        self.assertEqual(sum(report["episodes"] for report in reports), 4 * episodes)
        self.assertTrue(all(report["ok"] for report in reports))


if __name__ == "__main__":
    unittest.main()
