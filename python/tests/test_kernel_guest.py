# SPDX-License-Identifier: AGPL-3.0-only
"""Check an operator-built kernel init and optional ordinary guest boots."""

import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import time
import unittest


INIT = os.environ.get("CPU2TENSOR_KERNEL_INIT")
QEMU = os.environ.get("CPU2TENSOR_QEMU_SYSTEM")
KERNEL = os.environ.get("CPU2TENSOR_KERNEL")
INITRAMFS = os.environ.get("CPU2TENSOR_INITRAMFS")
RESULTS = os.environ.get("CPU2TENSOR_KERNEL_RESULTS")
MAXIMUM_GUEST_OUTPUT_BYTES = 1024 * 1024


def checksum(seed: int, count: int) -> int:
    total = 0
    for _ in range(count):
        seed = (1664525 * seed + 1013904223) % (1 << 32)
        total += seed // (1 << 24)
    return total


def events(output: str) -> list[dict]:
    return [json.loads(line.strip()[4:]) for line in output.splitlines()
            if line.strip().startswith("C2T ")]


def run_guest(
    mode: str, commands: tuple[str, ...] = ()
) -> tuple[int, list[dict], bytes, bytes]:
    console_host, console_guest = socket.socketpair()
    adapter_host, adapter_guest = socket.socketpair()
    command = [QEMU, "-accel", "tcg,thread=multi", "-smp", "2", "-m", "256M",
               "-display", "none", "-monitor", "none", "-nic", "none", "-no-reboot",
               "-chardev", f"socket,id=c2tconsole,fd={console_guest.fileno()}",
               "-serial", "chardev:c2tconsole",
               "-chardev", f"socket,id=c2tadapter,fd={adapter_guest.fileno()}",
               "-serial", "chardev:c2tadapter",
               "-kernel", KERNEL, "-initrd", INITRAMFS, "-append",
               f"console=ttyS0 rdinit=/init panic=-1 nokaslr cpu2tensor.mode={mode}"]
    process = None
    qemu_output = bytearray()
    console_output = bytearray()
    adapter_output = bytearray()
    pending = bytearray()
    rows = []
    sent = 0
    deadline = time.monotonic() + 60
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            pass_fds=(console_guest.fileno(), adapter_guest.fileno()),
            start_new_session=True,
        )
        console_guest.close()
        adapter_guest.close()
        with selectors.DefaultSelector() as selector:
            selector.register(console_host, selectors.EVENT_READ,
                              ("console", console_output))
            selector.register(adapter_host, selectors.EVENT_READ,
                              ("adapter", adapter_output))
            selector.register(process.stdout, selectors.EVENT_READ,
                              ("QEMU diagnostics", qemu_output))
            while selector.get_map():
                remaining = deadline - time.monotonic()
                ready = selector.select(max(0.0, remaining))
                if remaining <= 0 or not ready:
                    raise AssertionError("Guest did not complete within its 60-second deadline")
                for key, _ in ready:
                    name, output = key.data
                    data = os.read(key.fd, 4096)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    output.extend(data)
                    if len(output) > MAXIMUM_GUEST_OUTPUT_BYTES:
                        raise AssertionError(f"Guest {name} exceeded the bounded log size")
                    if name != "adapter":
                        continue
                    pending.extend(data)
                    while b"\n" in pending:
                        line, _, rest = pending.partition(b"\n")
                        pending = bytearray(rest)
                        line = line.strip()
                        if not line:
                            continue
                        if not line.startswith(b"C2T "):
                            raise AssertionError(
                                f"Protocol UART contained non-C2T bytes: {line[:128]!r}"
                            )
                        event = json.loads(line[4:])
                        rows.append(event)
                        if event["event"] == "ready":
                            if sent == len(commands):
                                raise AssertionError("Guest requested an unexpected extra command")
                            adapter_host.sendall((commands[sent] + "\n").encode("ascii"))
                            sent += 1
        status = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        if pending:
            raise AssertionError(f"Protocol UART ended inside a line: {pending[:128]!r}")
        if sent != len(commands):
            raise AssertionError("Guest exited before accepting every command")
        return status, rows, bytes(console_output), bytes(adapter_output)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if process is not None:
            process.stdout.close()
        console_host.close()
        console_guest.close()
        adapter_host.close()
        adapter_guest.close()
        if RESULTS:
            directory = Path(RESULTS)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"guest-{mode}.log").write_bytes(console_output)
            (directory / f"guest-{mode}-protocol.log").write_bytes(adapter_output)
            (directory / f"guest-{mode}-qemu.log").write_bytes(qemu_output)


class KernelResultChecks(unittest.TestCase):
    def check_results(self, rows: list[dict]) -> None:
        for row in rows:
            if row.get("action") in ("memory", "pipe"):
                self.assertEqual(row["checksum"], checksum(row["seed"], row["bytes"]))
            if row.get("action") == "parallel":
                self.assertNotEqual(row["cpu0"], row["cpu1"])
                self.assertEqual(row["checksum0"], checksum(row["seed"], row["bytes"]))
                self.assertEqual(row["checksum1"], checksum((row["seed"] + 1) % (1 << 32), row["bytes"]))


@unittest.skipUnless(INIT and hasattr(os, "sched_getaffinity"), "Set CPU2TENSOR_KERNEL_INIT on Linux")
class KernelHostTests(KernelResultChecks):
    def test_host_workload_checksums_and_distinct_cpus(self) -> None:
        available = os.sched_getaffinity(0)
        if len(available) < 2:
            self.skipTest("Parallel host fixture needs two available CPUs")
        result = subprocess.run([INIT, "--check"], capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = events(result.stdout)
        self.check_results(rows)
        self.assertEqual([row.get("action") for row in rows[:-1]], ["getpid", "memory", "pipe", "parallel"])
        self.assertIn(rows[3]["cpu0"], available)
        self.assertIn(rows[3]["cpu1"], available)
        self.assertEqual(rows[-1], {"event": "complete", "steps": 4, "ok": True})

    def test_one_available_cpu_fails_explicitly(self) -> None:
        cpu = min(os.sched_getaffinity(0))
        result = subprocess.run([INIT, "--check"], capture_output=True, text=True, timeout=40,
                                preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
        self.assertEqual(result.returncode, 1)
        rows = events(result.stdout)
        self.assertEqual(rows[-2]["message"], "parallel work needs two available CPUs")
        self.assertEqual(rows[-1], {"event": "complete", "steps": 3, "ok": False})

    def test_plain_host_launch_is_rejected(self) -> None:
        result = subprocess.run([INIT], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertIn("run as guest PID 1", result.stderr)
        self.assertEqual(result.stdout, "")


@unittest.skipUnless(QEMU and KERNEL and INITRAMFS, "Set CPU2TENSOR_QEMU_SYSTEM, KERNEL and INITRAMFS fixture paths")
class KernelGuestTests(KernelResultChecks):
    def check_uart_isolation(self, console: bytes, adapter: bytes) -> None:
        self.assertIn(b"Linux version", console)
        self.assertNotIn(b"C2T ", console)
        lines = [line.strip() for line in adapter.splitlines() if line.strip()]
        self.assertTrue(lines)
        self.assertTrue(all(line.startswith(b"C2T ") for line in lines))

    def test_observation_guest_runs_both_cpus(self) -> None:
        status, rows, console, adapter = run_guest("observe")
        self.assertEqual(status, 0)
        self.check_uart_isolation(console, adapter)
        self.check_results(rows)
        parallel = [row for row in rows if row.get("action") == "parallel"]
        self.assertEqual(len(parallel), 1)
        self.assertEqual({parallel[0]["cpu0"], parallel[0]["cpu1"]}, {0, 1})
        self.assertEqual(rows[-1], {"event": "complete", "steps": 4, "ok": True})

    def test_interactive_parallel_wraparound_and_rejected_size(self) -> None:
        commands = ("parallel 17 4096", "parallel 4294967295 17", "parallel 1 0", "getpid", "quit")
        status, rows, console, adapter = run_guest("interactive", commands)
        self.assertEqual(status, 0)
        self.check_uart_isolation(console, adapter)
        self.check_results(rows)
        self.assertEqual([row["step"] for row in rows if row["event"] == "ready"], [0, 1, 2, 2, 3])
        self.assertEqual(len([row for row in rows if row["event"] == "error"]), 1)
        self.assertEqual(rows[-1], {"event": "complete", "steps": 3, "ok": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
