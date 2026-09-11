# Repeated action windows

Action windows let a kernel guest name the execution that belongs to one client
action. The first implementation also provides a fixed-size native reducer for
adjacent basic-block transitions. It keeps the synchronous `KernelEnv` API and
does not add trajectory IDs, replay policy, or Python work on each block.

## Guest boundary contract

The guest binary supplies three distinct functions or basic blocks:

- begin: the next blocks belong to a new window;
- end: the current window completed normally;
- abort: the current window ended without a normal result.

Pass their addresses from the exact guest binary to the worker:

```sh
cpu2tensor-worker \
  --qemu "$QEMU_SYSTEM" --plugin "$C2T_PLUGIN" \
  --system on --kernel-adapter on \
  --window-start-pc "$WINDOW_BEGIN" \
  --window-end-pc "$WINDOW_END" \
  --window-abort-pc "$WINDOW_ABORT" \
  --reducer block-transitions --transition-capacity 4096 \
  --blocks off --registers none --memory off \
  --host 127.0.0.1 --port 9400 -- TARGET_ARGS
```

The marker blocks themselves are excluded. Begin resets each vCPU predecessor.
End and abort stop admission immediately. The guest adapter must request its next
action after end or abort. The worker pauses all vCPUs with QMP and asks the plugin
to drain every source before the plugin publishes the reduced rows and final
window metadata. Every delivered action requires exactly one window before the
next request or successful exit. Missing, duplicate, or unexpected windows fail
the stream. An action request while a window is still open is a capture error.
Process exit can instead close an open window with status `incomplete`.

One admission token governs the reducer and all raw callbacks belonging to a
block. It is linearized before or after the end marker: a block admitted before
end finishes all of its selected observations even if its callback overlaps end.
These atomics do not schedule vCPUs or impose a total order between CPUs. The
stopped-world drain establishes the final published frontier.

## Fixed transition reducer

Each vCPU owns one fixed open-addressed table. Its key is
`(from_address, destination)` and its value is the observed count. Execution
callbacks allocate nothing and take no shared mutex. The configured capacity is
per vCPU, must be a power of two, and all tables together may contain at most
$2^{20}$ slots.

The first block on each vCPU has no predecessor. A new begin always resets that
predecessor, so a row never connects two action windows. When a new distinct key
cannot fit, `overflow` increases for each unrepresented occurrence. Existing keys
continue to count. The reducer never silently evicts a row.

`--blocks off` omits raw basic-block rows while retaining reduced rows. Other
selected signals remain independent. Leave blocks on while checking a new guest
so raw per-vCPU transitions can be compared with the reducer.

## Python objects

`KernelEnv.reset()` and `KernelEnv.step()` still return ordinary synchronous
iterators. Reduced rows arrive in batches with `batch.transitions`:

```python
for batch in env.step(action):
    if batch.transitions is not None:
        source = batch.source
        window = batch.transitions.window
        edges = torch.stack((batch.transitions.from_addresses,
                             batch.transitions.destinations), dim=1)
        counts = batch.transitions.counts
    if batch.transition_window is not None:
        result = batch.transition_window
        if not result.complete:
            raise RuntimeError(
                f"window {result.id}: {result.status}, overflow={result.overflow}"
            )
```

`TransitionWindow.status` is `ended`, `aborted`, or `incomplete`. Its `complete`
property is true only for `ended` with zero overflow. `sources`,
`capacity_per_source`, `distinct`, `observed`, and `overflow` make bounded loss
visible. Transition batches retain the ordinary worker and source attribution;
window metadata is worker-wide and has no source.

## Evidence and limits

Portable native tests cover repeated two-source windows, a forced end/admission
interleaving, predecessor reset, overflow, abort, incomplete close, invalid
lifecycle, bounded payloads, unique rows, fixed shape, source tails, and action
binding. Python socket fixtures compare raw and reduced rows and retain explicit
status. The existing bounded ring stress test forces a producer to wait for a
slow consumer and verifies one million frames without loss; reducer publication
uses the same blocking worker transport after the all-source drain.

On 2026-09-11 the portable preset passed all three CTest targets on macOS. The
Linux x86 build used `trail-x86` with the operator-provided QEMU 11.0.3 header:

```sh
export PKG_CONFIG_PATH=/home/user/.cache/cpu2tensor/json-c/root/usr/lib/x86_64-linux-gnu/pkgconfig
cmake -S /home/user/.cache/cpu2tensor/action-window-review-20260911-final/source/native \
  -B /home/user/.cache/cpu2tensor/action-window-review-20260911-final/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCPU2TENSOR_BUILD_WORKER=ON \
  -DCPU2TENSOR_QEMU_INCLUDE_DIR=/home/user/.cache/cpu2tensor/qemu-system-x86_64/11.0.3-20260907/install/include
cmake --build /home/user/.cache/cpu2tensor/action-window-review-20260911-final/build
ctest --test-dir /home/user/.cache/cpu2tensor/action-window-review-20260911-final/build \
  --output-on-failure
```

That build compiled all 34 targets, including the worker and plugin, and passed
all three CTest targets. The local Python 3.10 fixture run passed 165 tests with
71 environment-dependent tests skipped; CPU and available MPS paths ran.

This slice has compile and fixture evidence only. It has not run these markers in
a real QEMU guest and makes no performance claim. Fresh paused-world register
snapshots, replay selection, distributed policy updates, and snapshot migration
remain separate work.
