# SPDX-License-Identifier: AGPL-3.0-only
"""Operator-run checks against a real Linux worker. See docs/validation.md."""

import os
import re
import selectors
import shlex
import subprocess
import time
import unittest
import uuid

from cpu2tensor import Pool


@unittest.skipUnless(os.environ.get("CPU2TENSOR_REMOTE"), "Set CPU2TENSOR_REMOTE to run real worker checks")
class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.host = os.environ["CPU2TENSOR_REMOTE"]
        self.address = os.environ["CPU2TENSOR_REMOTE_ADDRESS"]
        self.build = os.environ["CPU2TENSOR_REMOTE_BUILD"]
        self.qemu = os.environ["CPU2TENSOR_REMOTE_QEMU"]
        self.worker = None
        self.pid = None
        self.input_path = f"/tmp/cpu2tensor-{uuid.uuid4().hex}.input"

    def ssh(self, command, **kwargs):
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", self.host, command],
            check=True, capture_output=True, timeout=20, **kwargs,
        )

    def start(self, data=b"cpu2tensor\n", target=None, closed_stdin=False, qemu=None, port=0, registers="none", memory="off", values="off", stdio=False, episodes=1, timeout_ms=30000, start_pc=None, stop_pc=None, batching="legacy", publication="pipe", max_run_ms=None):
        self.ssh(f"umask 077; cat > {shlex.quote(self.input_path)}", input=data)
        arguments = [
            f"{self.build}/cpu2tensor-worker", "--qemu", qemu or self.qemu,
            "--plugin", f"{self.build}/libcpu2tensor_plugin.so",
            *(["--stdio", "on"] if stdio else ["--input", self.input_path]),
            "--episodes", str(episodes), "--host", self.address, "--port", str(port),
            "--timeout-ms", str(timeout_ms), "--registers", registers, "--memory", memory, "--memory-values", values,
            "--batching", batching, "--publication", publication,
            *([] if max_run_ms is None else ["--max-run-ms", str(max_run_ms)]),
            *([] if start_pc is None else ['--start-pc', hex(start_pc)]),
            *([] if stop_pc is None else ['--stop-pc', hex(stop_pc)]),
            "--", *(target or [f"{self.build}/checksum"]),
        ]
        command = "ulimit -c 0; echo cpu2tensor-pid:$$ >&2; "
        if closed_stdin:
            command += "exec 0<&-; "
        command += "exec " + shlex.join(arguments)
        self.worker = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", self.host, command],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        # Read a byte at a time only for the short startup lines, not trace data.
        with selectors.DefaultSelector() as selector:
            selector.register(self.worker.stderr, selectors.EVENT_READ)
            line = bytearray()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if not selector.select(timeout=1):
                    continue
                byte = os.read(self.worker.stderr.fileno(), 1)
                if not byte:
                    self.fail(f"Worker stopped during startup: {line!r}")
                line += byte
                if byte != b"\n":
                    continue
                text = line.decode()
                line.clear()
                if text.startswith("cpu2tensor-pid:"):
                    self.pid = int(text.split(":")[1])
                match = re.search(r"listening on (tcp://\S+)", text)
                if match:
                    return match.group(1)
        self.fail("Worker did not report a listening endpoint")

    def finish(self):
        output, errors = self.worker.communicate(timeout=20)
        return self.worker.returncode, output, errors

    def tearDown(self):
        if self.worker is not None and self.worker.poll() is None:
            if self.pid is not None:
                subprocess.run(["ssh", self.host, f"kill {self.pid}"], capture_output=True, timeout=10)
            self.worker.communicate(timeout=20)
        self.ssh(f"rm -f -- {shlex.quote(self.input_path)}")

    def test_real_mps_and_known_main(self):
        endpoint = self.start(closed_stdin=True)
        symbols = self.ssh(shlex.join(["nm", "-n", f"{self.build}/checksum"])).stdout.decode()
        main = int(re.search(r"^([0-9a-f]+) T main$", symbols, re.MULTILINE)[1], 16)
        count = 0
        found_main = False
        partial = False
        retained = None
        with Pool([endpoint], device="mps") as pool:
            for batch in pool.read():
                count += batch.addresses.numel()
                found_main |= bool((batch.addresses == main).any().item())
                partial |= batch.addresses.numel() < 256
                if retained is None:
                    retained = (batch, batch.addresses.cpu().clone())
        self.assertGreater(count, 0)
        self.assertTrue(found_main)
        self.assertTrue(partial)
        self.assertTrue(retained[0].addresses.cpu().equal(retained[1]))
        code, output, errors = self.finish()
        self.assertEqual((code, output, errors), (0, b"bytes=11 sum=1055\n", b""))
        print(f"Real MPS: {count} entries; main={main:#x}; retained batch valid")

    def test_slow_consumer_blocks_without_losing_entries(self):
        data = b"abcde\n" * 100000
        endpoint = self.start(data)
        count = 0
        with Pool([endpoint], device="cpu") as pool:
            batches = pool.read()
            first = next(batches)
            count += first.addresses.numel()
            time.sleep(0.5)
            children = self.ssh(f"ps -o pid= --ppid {self.pid}").stdout.split()
            self.assertEqual(len(children), 1)
            child = int(children[0])
            wait_channel = self.ssh(f"cat /proc/{child}/wchan").stdout.decode()
            self.assertIn("pipe", wait_channel, "Target did not reach pipe backpressure")
            for batch in batches:
                count += batch.addresses.numel()
        code, output, errors = self.finish()
        expected = f"bytes={len(data)} sum={sum(data)}\n".encode()
        self.assertEqual((code, output, errors), (0, expected, b""))
        print(f"Slow consumer: {count} entries, contiguous sequences, target wait={wait_channel}")

    def test_disconnect_cancels_target(self):
        endpoint = self.start(b"abcde\n" * 100000)
        with Pool([endpoint]) as pool:
            batches = pool.read()
            next(batches)
            children = self.ssh(f"ps -o pid= --ppid {self.pid}").stdout.split()
            self.assertEqual(len(children), 1)
            child = int(children[0])
        code, _, errors = self.finish()
        self.assertNotEqual(code, 0)
        self.assertIn(b"disconnected", errors)
        exists = self.ssh(f"if test -d /proc/{child}; then echo alive; else echo gone; fi").stdout
        self.assertEqual(exists, b"gone\n")

    def test_disconnect_after_internal_seal(self):
        endpoint = self.start(qemu=f"{self.build}/worker_test_target")
        with Pool([endpoint]) as pool:
            batches = pool.read()
            next(batches)
            time.sleep(0.1)  # Let the worker reach its child-exit wait.
        started = time.monotonic()
        code, _, errors = self.finish()
        self.assertNotEqual(code, 0)
        self.assertIn(b"disconnected", errors)
        self.assertLess(time.monotonic() - started, 3)

    def test_real_threads_have_separate_sources(self):
        endpoint = self.start(target=[f"{self.build}/plugin_test_target", "threads"], registers="general", memory="on", values="on")
        counts = {}
        baselines = {}
        with Pool([endpoint]) as pool:
            for batch in pool.read():
                counts[batch.source] = counts.get(batch.source, 0) + batch.addresses.numel()
                if batch.registers is not None:
                    table = batch.registers
                    for register, flags in zip(table.ids.tolist(), table.flags.tolist()):
                        if flags & 256:
                            selected = baselines.setdefault(batch.source, set())
                            self.assertNotIn(register, selected)
                            selected.add(register)
        self.assertEqual(len(counts), 3)
        self.assertTrue(all(count > 0 for count in counts.values()))
        self.assertEqual({source: len(ids) for source, ids in baselines.items()}, {0: 34, 1: 34, 2: 34})
        self.assertEqual(self.finish(), (0, b"", b""))
        print(f"Real pthread sources: {counts}")

    def test_target_fault_is_not_successful_completion(self):
        endpoint = self.start(target=[f"{self.build}/plugin_test_target", "fault"], registers="general", memory="on", values="on")
        with Pool([endpoint]) as pool:
            with self.assertRaisesRegex(RuntimeError, "target was killed"):
                for _ in pool.read():
                    pass
        self.finish()

    def test_fork_and_exec_fail_explicitly(self):
        for behavior in ("fork", "exec"):
            with self.subTest(behavior=behavior):
                endpoint = self.start(target=[f"{self.build}/plugin_test_target", behavior])
                with Pool([endpoint]) as pool:
                    with self.assertRaisesRegex(RuntimeError, "not supported"):
                        for _ in pool.read():
                            pass
                code, _, errors = self.finish()
                self.assertNotEqual(code, 0)
                self.assertIn(b"failure", errors)

    def test_nonzero_target_exit(self):
        endpoint = self.start(target=["/bin/false"])
        with Pool([endpoint]) as pool:
            with self.assertRaisesRegex(RuntimeError, "code 1"):
                for _ in pool.read():
                    pass
        code, _, _ = self.finish()
        self.assertEqual(code, 0, "Transport can complete while reporting a failed target")

    def check_signals(self, device, values, architecture="arm", profile="general"):
        import torch
        target = f"{self.build}/signals_target"
        qemu = None
        if architecture == "x86":
            if not os.environ.get("CPU2TENSOR_REMOTE_X86_QEMU"):
                self.skipTest("Set CPU2TENSOR_REMOTE_X86_QEMU for x86 signal checks")
            qemu = os.environ["CPU2TENSOR_REMOTE_X86_QEMU"]
            target = os.environ["CPU2TENSOR_REMOTE_X86_TARGET"]
        symbols_text = self.ssh(shlex.join(["nm", "-n", target])).stdout.decode()
        symbols = {name: int(address, 16) for address, name in
                   re.findall(r"^([0-9a-f]+) [A-Za-z] (signals_\w+)$", symbols_text, re.MULTILINE)}
        expected = {8: 0xA5, 16: 0xB6C7, 32: 0xD8E9FA0B, 64: 0xFEDCBA9876543210,
                    128: 0xFEDCBA98765432100123456789ABCDEF}
        wanted_pc = {symbols[f"signals_{operation}_u{bits}"]
                     for bits in expected for operation in ("load", "store")}
        wanted_pc.add(symbols["signals_load_signed_u8"])
        checkpoint_names = ("zero", "high", "unchanged", "zero_again")
        checkpoints = {symbols[f"signals_register_{name}"]: name for name in checkpoint_names}
        transactions = {}
        changes = {}
        baseline = {}
        retained = None
        counts = {"blocks": 0, "registers": 0, "memory": 0}
        endpoint = self.start(target=[target], qemu=qemu, registers=profile, memory="on", values=values)
        with Pool([endpoint], device=device, timeout=60) as pool:
            for batch in pool.read():
                counts["blocks"] += batch.addresses.numel()
                if batch.memory is not None:
                    table = batch.memory
                    counts["memory"] += table.pc.numel()
                    pcs = table.pc.cpu().tolist()
                    selected = [i for i, pc in enumerate(pcs) if pc in wanted_pc]
                    if selected:
                        addresses = table.addresses.cpu().tolist()
                        sizes = table.sizes.cpu().tolist()
                        flags = table.flags.cpu().tolist()
                        raw = table.values.cpu().tolist() if table.values is not None else None
                        for i in selected:
                            transactions.setdefault(pcs[i], []).append((addresses[i], sizes[i], flags[i],
                                None if raw is None else bytes(raw[i])))
                        if retained is None:
                            retained = (table, table.addresses.cpu().clone())
                    if values == "off": self.assertIsNone(table.values)
                if batch.registers is not None:
                    table = batch.registers
                    counts["registers"] += table.pc.numel()
                    pcs = table.pc.cpu().tolist()
                    flags = table.flags.cpu().tolist()
                    ids = table.ids.cpu().tolist()
                    widths = table.widths.cpu().tolist()
                    raw = table.values.cpu().tolist()
                    for i, pc in enumerate(pcs):
                        name = table.names[ids[i]]
                        value = int.from_bytes(bytes(raw[i][:widths[i]]), "little")
                        if flags[i] & 256:
                            self.assertNotIn((batch.source, name), baseline)
                            baseline[batch.source, name] = value
                        if pc in checkpoints and (flags[i] & 255) == 1:
                            changes[checkpoints[pc], name] = value
        self.assertTrue(all(value > 0 for value in counts.values()))
        self.assertTrue(retained[0].addresses.cpu().equal(retained[1]))
        for bits, value in expected.items():
            for operation in ("load", "store"):
                pc = symbols[f"signals_{operation}_u{bits}"]
                rows = transactions[pc]
                self.assertEqual(sum(row[1] for row in rows), bits // 8)
                expected_bytes = value.to_bytes(bits // 8, "little")
                rows = sorted(rows)
                offset = 0
                for address, size, flags, raw in rows:
                    self.assertEqual(address, symbols[f"signals_u{bits}"] + offset)
                    self.assertEqual(flags, int(operation == "store"))
                    if values == "on":
                        self.assertEqual(raw[:size], expected_bytes[offset:offset+size])
                        self.assertEqual(raw[size:], bytes(16-size))
                    offset += size
        signed = transactions[symbols["signals_load_signed_u8"]][0]
        self.assertEqual(signed[:3], (symbols["signals_u8"], 1, 0))
        if values == "on": self.assertEqual(signed[3], bytes([0xA5]) + bytes(15))
        changed_name = "x0" if architecture == "arm" else "rax"
        self.assertIn((0, changed_name), baseline)
        self.assertEqual(changes["high", changed_name], expected[64])
        self.assertNotIn(("unchanged", changed_name), changes)
        self.assertEqual(changes["zero_again", changed_name], 0)
        code, output, errors = self.finish()
        expected_errors = b""
        if architecture == "x86" and profile == "all":
            omitted = ("ftag", "fiseg", "fioff", "foseg", "fooff", "fop")
            expected_errors = b"".join(f"cpu2tensor: omitting unavailable register {name}\n".encode()
                                       for name in omitted)
            self.assertFalse(set(omitted) & {name for source, name in baseline})
        self.assertEqual((code, output, errors),
                         (0, b"signals: ok checksum=0123456751428186\n", expected_errors))
        print(f"{architecture} {device} values={values}: {counts}, {len(baseline)} baselines")

    def test_arm_signals_values_off(self):
        self.check_signals("cpu", "off")

    def test_arm_signals_on_mps(self):
        self.check_signals("mps", "on")

    def test_x86_signals_all_registers(self):
        self.check_signals("cpu", "on", "x86", "all")

    def test_worker_can_restart_on_same_port(self):
        port = 0
        for _ in range(2):
            endpoint = self.start(port=port)
            port = int(endpoint.rsplit(":", 1)[1])
            with Pool([endpoint]) as pool:
                for _ in pool.read():
                    pass
            self.assertEqual(self.finish(), (0, b"bytes=11 sum=1055\n", b""))


if __name__ == "__main__":
    unittest.main()
