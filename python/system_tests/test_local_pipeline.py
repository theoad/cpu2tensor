# SPDX-License-Identifier: AGPL-3.0-only
"""Short end-to-end checks for the main observation and action paths."""

from __future__ import annotations

import os
from pathlib import Path
import re
import selectors
import subprocess
import tempfile
import unittest

import torch

from cpu2tensor import Pool, StdioEnv, TerminalReason, TraceTerminalError


BUILD = Path(os.environ["CPU2TENSOR_CI_BUILD"])
PLUGIN = BUILD / "libcpu2tensor_plugin.so"


class LocalWorker:
    def __init__(
        self,
        qemu: str,
        target: Path,
        *,
        data: bytes = b"",
        stdio: bool = False,
        rich: bool = False,
        max_run_ms: int | None = None,
    ) -> None:
        input_file = tempfile.NamedTemporaryFile(prefix="cpu2tensor-ci-", delete=False)
        input_file.write(data)
        input_file.close()
        self._input_path = Path(input_file.name)
        arguments = [
            str(BUILD / "cpu2tensor-worker"),
            "--qemu", qemu,
            "--plugin", str(PLUGIN),
            *(["--stdio", "on"] if stdio else ["--input", str(self._input_path)]),
            "--host", "127.0.0.1",
            "--port", "0",
            "--timeout-ms", "30000",
            "--registers", "general" if rich else "none",
            "--memory", "on" if rich else "off",
            "--memory-values", "on" if rich else "off",
            "--batching", "mixed" if rich else "legacy",
            "--publication", "ring" if rich else "pipe",
            *([] if max_run_ms is None else ["--max-run-ms", str(max_run_ms)]),
            "--", str(target),
        ]
        self.process = subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.endpoint = self._read_endpoint()

    def _read_endpoint(self) -> str:
        assert self.process.stderr is not None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stderr, selectors.EVENT_READ)
            line = bytearray()
            for _ in range(150):
                if not selector.select(timeout=0.1):
                    continue
                byte = os.read(self.process.stderr.fileno(), 1)
                if not byte:
                    raise RuntimeError(f"Worker stopped during startup: {line!r}")
                line += byte
                if byte == b"\n":
                    match = re.search(rb"listening on (tcp://\S+)", line)
                    if match:
                        return match.group(1).decode()
                    line.clear()
        raise TimeoutError("Worker did not publish an endpoint in 15 seconds")

    def finish(self) -> tuple[int, bytes, bytes]:
        output, errors = self.process.communicate(timeout=20)
        self._input_path.unlink(missing_ok=True)
        return self.process.returncode, output, errors

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.communicate(timeout=10)
        self._input_path.unlink(missing_ok=True)


class PipelineSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workers: list[LocalWorker] = []

    def tearDown(self) -> None:
        for worker in self.workers:
            worker.close()

    def start(self, *args, **kwargs) -> LocalWorker:
        worker = LocalWorker(*args, **kwargs)
        self.workers.append(worker)
        return worker

    def test_observation_crosses_x86_and_aarch64_workers(self) -> None:
        data = b"cpu2tensor system test\n"
        expected = f"bytes={len(data)} sum={sum(data)}\n".encode()
        cases = (
            (os.environ["CPU2TENSOR_CI_QEMU_X86"], BUILD / "checksum"),
            (os.environ["CPU2TENSOR_CI_QEMU_ARM"], Path(os.environ["CPU2TENSOR_CI_ARM_TARGET"])),
        )
        for qemu, target in cases:
            with self.subTest(qemu=Path(qemu).name):
                worker = self.start(qemu, target, data=data)
                with Pool([worker.endpoint], timeout=20) as pool:
                    batches = list(pool.read())
                self.assertGreater(sum(batch.addresses.numel() for batch in batches), 0)
                code, output, errors = worker.finish()
                self.assertEqual((code, output), (0, expected), errors.decode())

    def test_rich_observation_from_three_workers_feeds_a_model(self) -> None:
        workers = [
            self.start(
                os.environ["CPU2TENSOR_CI_QEMU_X86"],
                BUILD / "signals_target",
                rich=True,
            )
            for _ in range(3)
        ]
        rows = torch.zeros((3, 3), dtype=torch.float32)
        with Pool([worker.endpoint for worker in workers], timeout=20) as pool:
            for batch in pool.read():
                rows[batch.worker, 0] += batch.addresses.numel()
                if batch.registers is not None:
                    rows[batch.worker, 1] += batch.registers.ids.numel()
                if batch.memory is not None:
                    rows[batch.worker, 2] += batch.memory.addresses.numel()
                    self.assertIsNotNone(batch.memory.values)
        self.assertTrue(bool((rows > 0).all()))
        model = torch.nn.Linear(3, 1)
        loss = model(torch.log1p(rows)).square().mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        for worker in workers:
            code, output, errors = worker.finish()
            self.assertEqual(code, 0, errors.decode())
            self.assertIn(b"signals: ok checksum=", output)

    def test_stdio_actions_resume_the_paused_target(self) -> None:
        worker = self.start(
            os.environ["CPU2TENSOR_CI_QEMU_X86"],
            BUILD / "system_stdio_target",
            stdio=True,
            rich=True,
        )
        with StdioEnv(worker.endpoint, timeout=20) as env:
            initial = list(env.reset())
            if not env.needs_input:
                code, output, errors = worker.finish()
                self.fail(
                    f"target ended before input: env={env.exit_code}, worker={code}, "
                    f"stdout={output!r}, stderr={errors!r}"
                )
            self.assertTrue(any(batch.memory is not None for batch in initial))
            list(env.step(b"a\n"))
            self.assertTrue(env.needs_input)
            list(env.step(b"b\n"))
            self.assertEqual(env.exit_code, 0)
        code, output, errors = worker.finish()
        self.assertEqual((code, output), (0, b""), errors.decode())

    def test_worker_deadline_has_a_structured_terminal_outcome(self) -> None:
        worker = self.start(
            str(BUILD / "worker_test_target"),
            BUILD / "checksum",
            max_run_ms=100,
        )
        with Pool([worker.endpoint], timeout=10) as pool:
            with self.assertRaises(TraceTerminalError) as raised:
                list(pool.read())
        outcome = raised.exception.outcome
        self.assertEqual(outcome.reason, TerminalReason.MAX_RUN_DEADLINE)
        self.assertTrue(outcome.hello_reported)
        self.assertTrue(outcome.data_reported)
        self.assertFalse(outcome.complete)
        code, _, errors = worker.finish()
        self.assertNotEqual(code, 0)
        self.assertIn(b"Target exceeded --max-run-ms deadline", errors)
