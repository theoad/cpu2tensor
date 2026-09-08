# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded, independent observation readers for a synchronous endpoint pool."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
import threading

import torch

from cpu2tensor.batch import Batch
from cpu2tensor._device import to_device
from cpu2tensor.pool import Pool


class EndpointReaders:
    """Keep one waiting batch and at most one in-progress batch per worker."""

    def __init__(self, endpoints: Sequence[str], device: str, timeout: float, *, batch_bytes: int = 0) -> None:
        # Construct every pool before starting any connection. Invalid endpoint or
        # device settings therefore cannot leave a partly started target group.
        self._endpoints = tuple(endpoints)
        self._device = torch.device(device)
        if self._device.type not in ("cpu", "mps", "cuda"):
            raise ValueError("Pool supports CPU, MPS, and CUDA devices")
        if self._device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available in this Python environment")
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this Python environment")
        self._pools = [Pool([endpoint], device="cpu", timeout=timeout, batch_bytes=batch_bytes)
                       for endpoint in self._endpoints]
        self._ready = threading.Condition()
        self._batches: list[Batch | None] = [None] * len(self._pools)
        self._finished = [False] * len(self._pools)
        self._failure: tuple[int, BaseException] | None = None
        self._threads: list[threading.Thread] = []
        self._started = False
        self._closed = False

    def read(self) -> Iterator[Batch]:
        with self._ready:
            if self._closed:
                raise RuntimeError("Pool is closed; create a new pool for another run")
            if self._started:
                raise RuntimeError("Pool.read() can only be called once per run")
            self._started = True
        return self._read()

    def _receive(self, worker: int) -> None:
        try:
            for batch in self._pools[worker].read():
                batch = replace(batch, worker=worker)
                with self._ready:
                    self._ready.wait_for(lambda: self._closed or self._batches[worker] is None)
                    if self._closed:
                        return
                    self._batches[worker] = batch
                    self._ready.notify_all()
                # Do not retain an extra reference while the next read blocks.
                del batch
        except BaseException as error:
            with self._ready:
                if not self._closed and self._failure is None:
                    # A failure must not wait behind a full batch queue.
                    self._failure = (worker, error)
                    self._ready.notify_all()
        finally:
            self._pools[worker].close()
            with self._ready:
                self._finished[worker] = True
                self._ready.notify_all()

    def _read(self) -> Iterator[Batch]:
        try:
            with self._ready:
                if self._closed:
                    raise RuntimeError("Pool was closed before reading started")
                for worker in range(len(self._pools)):
                    thread = threading.Thread(
                        target=self._receive, args=(worker,),
                        name=f"cpu2tensor-worker-{worker}", daemon=True,
                    )
                    thread.start()
                    self._threads.append(thread)
            next_worker = 0
            while True:
                with self._ready:
                    self._ready.wait_for(lambda: (
                        self._closed or self._failure is not None
                        or any(batch is not None for batch in self._batches)
                        or all(self._finished)
                    ))
                    if self._failure is not None:
                        worker, error = self._failure
                        raise RuntimeError(
                            f"Worker {worker} ({self._endpoints[worker]}) failed: {error}"
                        ) from error
                    if self._closed:
                        return
                    # Rotating the first slot avoids starving a sparse worker
                    # when another worker always has a batch ready.
                    for offset in range(len(self._pools)):
                        worker = (next_worker + offset) % len(self._pools)
                        batch = self._batches[worker]
                        if batch is not None:
                            self._batches[worker] = None
                            next_worker = (worker + 1) % len(self._pools)
                            self._ready.notify_all()
                            break
                    else:
                        return  # Every worker completed and every slot is empty.
                yield self._on_device(batch)
                del batch
        finally:
            self.close()

    def _on_device(self, batch: Batch) -> Batch:
        # Device work belongs to the consumer thread. In particular, concurrent
        # MPS transfers from reader threads can conflict with synchronization.
        return to_device(batch, self._device)

    def close(self) -> None:
        """Cancel target connections, wake blocked publishers, and reap readers."""
        with self._ready:
            self._closed = True
            self._batches = [None] * len(self._pools)
            self._ready.notify_all()
            threads = tuple(self._threads)
        for pool in self._pools:
            pool.close()
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join()
