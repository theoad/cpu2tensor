# SPDX-License-Identifier: AGPL-3.0-only
"""Pipeline measurements use real TCP decoding; summaries stay bounded and explicit."""

from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import replay_pipeline
import run_kernel_pipeline
from cpu2tensor import Batch, ExecutableLayout, Pool
from cpu2tensor.examples.benchmark_pipeline import (
    FEATURE_COUNT, FEATURE_GROUPS, HISTOGRAM_BINS, benchmark, row_counts, summarize, tensor_bytes,
)
from test_consumer import frame, worker
from test_mixed import capture


class PipelineBenchmarkTests(unittest.TestCase):
    def test_layout_metadata_has_its_own_summary_and_row_count(self):
        batch = Batch(None, None, torch.empty(0, dtype=torch.int64),
                      layout=ExecutableLayout(torch.tensor([0x400000, 0x401000, 0x400040])))
        self.assertEqual(row_counts(batch), {"blocks": 0, "registers": 0, "memory": 0, "context": 0, "layout": 1})
        features = summarize(batch).reshape(len(FEATURE_GROUPS), HISTOGRAM_BINS)
        torch.testing.assert_close(features.sum(1), torch.tensor([0., 0., 0., 0., 0., 0., 1.]))
        self.assertEqual(tensor_bytes(batch), 24)

    def mixed_batch(self):
        with worker(capture(), fragment=17) as endpoint, Pool([endpoint]) as pool:
            batches = list(pool.read())
        self.assertEqual(len(batches), 1)
        return batches[0]

    def test_every_mixed_table_contributes_and_memory_padding_is_excluded(self):
        batch = self.mixed_batch()
        self.assertEqual(row_counts(batch), {"blocks": 3, "registers": 2, "memory": 2, "context": 2, "layout": 0})
        summaries = summarize(batch).reshape(len(FEATURE_GROUPS), HISTOGRAM_BINS)
        self.assertEqual(summaries.shape, (7, 16))
        torch.testing.assert_close(summaries.sum(1), torch.tensor([1., 1., 1., 1., 1., 1., 0.]))
        expected_memory = torch.zeros(HISTOGRAM_BINS)
        expected_memory[1:5] = 0.25
        torch.testing.assert_close(summaries[FEATURE_GROUPS.index("memory_values")], expected_memory)
        changes = {
            "blocks": replace(batch, addresses=torch.full_like(batch.addresses, 240)),
            "register_bytes": replace(batch, registers=replace(batch.registers, values=torch.full_like(batch.registers.values, 240))),
            "memory_addresses": replace(batch, memory=replace(batch.memory, addresses=torch.full_like(batch.memory.addresses, 16))),
            "memory_sizes": replace(batch, memory=replace(batch.memory, sizes=torch.full_like(batch.memory.sizes, 16))),
            "memory_values": replace(batch, memory=replace(batch.memory, values=torch.full_like(batch.memory.values, 240))),
            "context": replace(batch, context=replace(batch.context, cr3=torch.full_like(batch.context.cr3, 240))),
        }
        for name, changed in changes.items():
            with self.subTest(name=name):
                row = FEATURE_GROUPS.index(name)
                self.assertFalse(torch.equal(summarize(changed).reshape(7, 16)[row], summaries[row]))
        without_values = replace(batch, memory=replace(batch.memory, values=None))
        self.assertEqual(summarize(without_values).reshape(7, 16)[4].sum().item(), 0)
        self.assertGreater(tensor_bytes(batch), tensor_bytes(without_values))

    def test_drain_counts_all_rows_from_two_public_endpoints(self):
        expected_bytes = tensor_bytes(self.mixed_batch())
        with tempfile.TemporaryDirectory() as temporary, worker(capture()) as first, worker(capture()) as second:
            output = Path(temporary) / "run"
            result = benchmark([first, second], output, capture_hosts=["fixture-a", "fixture-b"])
            self.assertEqual(json.loads((output / "metrics.json").read_text()), result)
            self.assertFalse((output / "model.pt").exists())
        self.assertEqual(result["rows"], {"blocks": 6, "registers": 4, "memory": 4, "context": 4, "layout": 0})
        self.assertEqual(result["events"], 18)
        self.assertEqual(result["workers_completed"], 2)
        self.assertEqual([(source["worker"], source["source"]) for source in result["per_source"]], [(0, 0), (1, 0)])
        self.assertEqual(result["max_batch_tensor_bytes"], expected_bytes)
        self.assertEqual(result["updates"], 0)
        self.assertGreater(result["elapsed_seconds"], 0)
        self.assertGreater(result["next_batch_seconds"], 0)
        self.assertGreater(result["host_peak_rss_bytes"], 0)

    def check_training(self, device):
        with tempfile.TemporaryDirectory() as temporary, worker(capture()) as first, worker(capture()) as second:
            output = Path(temporary) / "run"
            result = benchmark([first, second], output, device=device, mode="train", group_size=1)
            checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["feature_groups"], list(FEATURE_GROUPS))
        self.assertEqual(result["summary_features"], FEATURE_COUNT)
        self.assertEqual(result["updates"], 2)
        self.assertEqual(result["trained_summaries"], 2)
        self.assertTrue(result["weights_changed"])
        self.assertTrue(result["checkpoint_reloaded"])
        self.assertTrue(result["all_traces_complete"])
        self.assertTrue(torch.isfinite(torch.tensor(result["mean_loss"])).item())
        self.assertGreater(result["feature_seconds"], 0)
        self.assertGreater(result["model_seconds"], 0)

    def test_cpu_training_updates_and_reloads_checkpoint(self):
        self.check_training("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_training_updates_and_reloads_checkpoint(self):
        self.check_training("mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training_updates_and_reloads_checkpoint(self):
        self.check_training("cuda")

    def test_partial_summary_group_is_trained_at_completion(self):
        with tempfile.TemporaryDirectory() as temporary, worker(capture()) as endpoint:
            result = benchmark([endpoint], Path(temporary) / "run", mode="train", group_size=64)
        self.assertEqual((result["updates"], result["trained_summaries"]), (1, 1))

    def test_trace_failure_propagates_without_success_metrics(self):
        data = frame(1, detail=1) + frame(2, addresses=(8,)) + frame(5, detail=1)
        with tempfile.TemporaryDirectory() as temporary, worker(data) as endpoint:
            output = Path(temporary) / "failed"
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                benchmark([endpoint], output, mode="train", group_size=1)
            self.assertFalse((output / "metrics.json").exists())
            self.assertFalse((output / "model.pt").exists())
            failure = json.loads((output / "failure.json").read_text())
            self.assertFalse(failure["all_traces_complete"])
            self.assertEqual((failure["batches"], failure["updates"]), (1, 1))
            self.assertEqual(failure["rows"]["blocks"], 1)
            self.assertEqual(failure["error_type"], "RuntimeError")

    def test_empty_training_stream_is_not_a_successful_training_run(self):
        data = frame(1, detail=1) + frame(3) + frame(4)
        with tempfile.TemporaryDirectory() as temporary, worker(data) as endpoint:
            output = Path(temporary) / "empty"
            with self.assertRaisesRegex(ValueError, "at least one observation batch"):
                benchmark([endpoint], output, mode="train")
            failure = json.loads((output / "failure.json").read_text())
            self.assertFalse(failure["all_traces_complete"])
            self.assertEqual((failure["batches"], failure["updates"]), (0, 0))
            self.assertEqual(failure["error_type"], "ValueError")
            self.assertFalse((output / "metrics.json").exists())

    def test_checkpoint_save_and_reload_failures_keep_partial_counters(self):
        for operation in ("save", "load"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary, \
                    worker(capture()) as endpoint:
                output = Path(temporary) / "failed"
                with mock.patch(f"cpu2tensor.examples.benchmark_pipeline.torch.{operation}",
                                side_effect=OSError(f"checkpoint {operation} failed")):
                    with self.assertRaisesRegex(OSError, f"checkpoint {operation} failed"):
                        benchmark([endpoint], output, mode="train")
                failure = json.loads((output / "failure.json").read_text())
                self.assertFalse(failure["all_traces_complete"])
                self.assertEqual((failure["batches"], failure["updates"]), (1, 1))
                self.assertEqual(failure["rows"]["memory"], 2)
                self.assertEqual(failure["error_type"], "OSError")
                self.assertFalse((output / "metrics.json").exists())

    def test_final_device_fence_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary, worker(capture()) as endpoint:
            output = Path(temporary) / "failed"
            with mock.patch("cpu2tensor.examples.benchmark_pipeline.synchronize",
                            side_effect=[None, None, RuntimeError("final fence failed")]):
                with self.assertRaisesRegex(RuntimeError, "final fence failed"):
                    benchmark([endpoint], output)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["batches"], 1)
            self.assertFalse(failure["all_traces_complete"])
            self.assertFalse((output / "metrics.json").exists())

    def test_partial_metrics_write_is_removed_on_failure(self):
        original = Path.write_text

        def write(path, *args, **kwargs):
            if path.name == "metrics.json":
                original(path, '{"all_traces_complete": true')
                raise OSError("metrics write failed")
            return original(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, worker(capture()) as endpoint:
            output = Path(temporary) / "failed"
            with mock.patch.object(Path, "write_text", write):
                with self.assertRaisesRegex(OSError, "metrics write failed"):
                    benchmark([endpoint], output)
            self.assertEqual(json.loads((output / "failure.json").read_text())["batches"], 1)
            self.assertFalse((output / "metrics.json").exists())

    def test_soft_budget_failure_records_consumed_rows(self):
        ticks = itertools.count()
        with tempfile.TemporaryDirectory() as temporary, worker(capture()) as endpoint:
            output = Path(temporary) / "budget"
            with mock.patch("cpu2tensor.examples.benchmark_pipeline.time.perf_counter",
                            side_effect=lambda: next(ticks)):
                with self.assertRaisesRegex(TimeoutError, "consumption budget"):
                    benchmark([endpoint], output, max_seconds=2)
            failure = json.loads((output / "failure.json").read_text())
            self.assertFalse(failure["all_traces_complete"])
            self.assertEqual(failure["max_seconds"], 2)
            self.assertEqual(failure["batches"], 1)
            self.assertEqual(failure["rows"]["blocks"], 3)
            self.assertFalse((output / "metrics.json").exists())

    def test_runner_can_defer_metrics_until_its_extra_checks_pass(self):
        with tempfile.TemporaryDirectory() as temporary, worker(capture()) as endpoint:
            output = Path(temporary) / "pending"
            result = benchmark([endpoint], output, write_metrics=False)
            self.assertTrue(result["all_traces_complete"])
            self.assertEqual(result["events"], 9)
            self.assertFalse((output / "metrics.json").exists())
            self.assertFalse((output / "failure.json").exists())

    def test_replay_sender_validation_failure_has_no_success_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "replay"
            recording = Path(temporary) / "capture.bin"
            recording.write_bytes(capture())

            def measure(endpoints, directory, **options):
                self.assertFalse(options["write_metrics"])
                directory.mkdir()
                return {"all_traces_complete": True, "batches": 3, "events": 9}

            with mock.patch("sys.argv", ["replay", "--capture", str(recording), "--output", str(output)]), \
                    mock.patch.object(replay_pipeline, "benchmark", side_effect=measure), \
                    mock.patch.object(replay_pipeline.threading, "Thread") as thread:
                thread.return_value.is_alive.return_value = True
                with self.assertRaisesRegex(RuntimeError, "Replay server failed"):
                    replay_pipeline.main()
            failure = json.loads((output / "failure.json").read_text())
            self.assertFalse(failure["all_traces_complete"])
            self.assertEqual((failure["batches"], failure["events"]), (3, 9))
            self.assertFalse((output / "metrics.json").exists())

    def test_guest_validation_failure_reaps_every_worker_and_keeps_final_logs(self):
        class Worker:
            host = "fixture"
            arguments = ["benign-fixture"]

            def __init__(self, fails_cleanup):
                self.log = tempfile.TemporaryFile()
                self.log.write(b"started\n")
                self.fails_cleanup = fails_cleanup
                self.cleaned = False

            def setUp(self):
                pass

            def start(self, **options):
                return "tcp://fixture:1"

            def finish(self):
                return "guest oracle failed"

            def tearDown(self):
                self.log.write(b"final shutdown\n")
                self.log.close()
                self.cleaned = True
                if self.fails_cleanup:
                    raise OSError("first worker cleanup failed")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "kernel"
            workers = [Worker(True), Worker(False)]

            def measure(endpoints, directory, **options):
                self.assertFalse(options["write_metrics"])
                directory.mkdir()
                return {"all_traces_complete": True, "batches": 7, "events": 18}

            arguments = ["kernel", "--output", str(output), "--workers", "2",
                         "--start-pc", "0x1000", "--stop-pc", "0x2000"]
            with mock.patch("sys.argv", arguments), \
                    mock.patch.object(run_kernel_pipeline, "benchmark", side_effect=measure), \
                    mock.patch.object(run_kernel_pipeline.test_remote_kernel, "RemoteKernelTests", side_effect=workers):
                with self.assertRaisesRegex(RuntimeError, "Guest did not report successful completion"):
                    run_kernel_pipeline.main()
            self.assertTrue(all(worker.cleaned for worker in workers))
            for index in range(2):
                self.assertEqual((output / f"worker-{index}.log").read_bytes(), b"started\nfinal shutdown\n")
            failure = json.loads((output / "failure.json").read_text())
            self.assertFalse(failure["all_traces_complete"])
            self.assertEqual((failure["batches"], failure["events"]), (7, 18))
            self.assertIn("first worker cleanup failed", failure["cleanup_errors"][0])
            self.assertFalse((output / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
