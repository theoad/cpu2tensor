# Observing exact wire frames

Pass `wire_observer` to `Pool` when a client needs a replay corpus or exact wire
accounting. The callback receives one `WireFrame` after its complete header and
payload have arrived and before cpu2tensor decodes it:

```python
from cpu2tensor import Pool

with open("capture.bin", "wb") as capture:
    def record(frame):
        capture.write(frame.data)

    with Pool(["tcp://127.0.0.1:9000"], wire_observer=record) as pool:
        for batch in pool.read():
            train(batch)
```

`frame.data` is a read-only `memoryview` over the receive buffer. Writing that
view to a binary stream adds no Python bytes copy. The view is released when the
callback returns. A callback that needs to retain a frame must copy it with
`bytes(frame.data)`; one copy is then bounded by the worker's maximum frame size.

The callback runs synchronously in receive order after a whole frame is read and
before that frame is decoded. It therefore sees the exact accepted header and
payload even when later semantic validation rejects the frame. A partial header
or payload is not a frame and is not published. Observer exceptions propagate
directly from a single-worker pool and cancel the read. A multiworker pool reports
the worker failure and chains the observer exception as its cause. Slow observers
add receive backpressure.

`WireFrame` also carries the endpoint index, endpoint URL, wire version, kind,
source, row count, sequence and detail header fields. A single worker invokes
callbacks serially. Multiworker pools preserve each worker's order, while
independent worker callbacks can overlap on reader threads. Use one sink per
worker or make a shared sink thread-safe; cpu2tensor does not invent a total order
between independent streams. Transport end has no wire bytes and does not invoke
the observer. Read `Pool.outcomes` after completion or failure for clean,
truncated, reset and timeout classification.

With no observer, `Pool` allocates no `WireFrame` or `memoryview` and follows the
normal receive and decode path.
