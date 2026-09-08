# SPDX-License-Identifier: AGPL-3.0-only
"""Several real worker processes feeding one synchronous CPU or MPS learner."""

import os
from pathlib import Path
import tempfile
import unittest

import torch
import test_remote_window
from cpu2tensor.examples.benchmark_pipeline import benchmark


@unittest.skipUnless(os.environ.get('CPU2TENSOR_REMOTE'), 'Set CPU2TENSOR_REMOTE for real multiworker checks')
class RemoteMultiworkerTests(unittest.TestCase):
    def check_workers(self, device):
        endpoints = []
        workers = []
        for _ in range(3):
            worker = test_remote_window.RemoteWindowTests('runTest')
            worker.setUp()
            self.addCleanup(worker.doCleanups)
            endpoints.append(worker.start('normal', rich=True, batching='mixed', publication='ring'))
            workers.append(worker)
        with tempfile.TemporaryDirectory() as directory:
            result = benchmark(endpoints, Path(directory) / 'run', device=device, mode='train',
                               timeout=120, capture_hosts=[workers[0].remote.host] * 3,
                               batch_bytes=65536 if device == 'mps' else 0)
            self.assertTrue((Path(directory) / 'run' / 'model.pt').exists())
        for worker in workers:
            code, output, errors = worker.remote.finish()
            self.assertEqual((code, output), (0, b'window: checksum=663776\n'))
            self.assertIn(b'capture stop reached', errors)
        self.assertEqual({row['worker'] for row in result['per_source']}, {0, 1, 2})
        for name in ('blocks', 'registers', 'memory'):
            self.assertGreater(result['rows'][name], 0)

    def test_three_workers_cpu_training(self):
        self.check_workers('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable')
    def test_three_workers_mps_training(self):
        self.check_workers('mps')
