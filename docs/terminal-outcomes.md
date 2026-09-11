# Terminal capture outcomes

`Pool.outcomes` exposes one immutable `TerminalOutcome` per configured endpoint.
An entry is `None` while that endpoint is still running. A successful trace records
`reason=complete` and `complete=True`. A failed trace attaches the same outcome to
its exception:

```python
from cpu2tensor import Pool, TraceTerminalError

try:
    with Pool(["tcp://127.0.0.1:9000"]) as pool:
        for batch in pool.read():
            consume(batch)
except TraceTerminalError as error:
    print(error.outcome.reason, error.outcome.endpoint)
```

`TraceTerminalError` derives from `RuntimeError`. Its connection and timeout
variants also derive from `ConnectionError` and `TimeoutError`, respectively, so
existing catch clauses continue to work. In a multiple-endpoint pool, `worker` is
the endpoint-list index and `endpoint` is the original endpoint string. Terminal
events are worker-wide, so `source` is `None` rather than an invented vCPU number.

The client-received and worker-reported fields deliberately describe different
facts. `hello_received` and `data_received` mean this client validated those
frames. `hello_reported` and `data_reported` are present only in a valid worker
terminal report. `start` and `stop` are `observed`, `not_observed`,
`not_configured`, or `unknown`. Legacy peers do not report boundary progress, so
their values remain `unknown` even if nearby data suggests likely progress.

`transport` records clean EOF, reset, another socket error, client timeout, or
`not_observed`. It does not replace `reason`. For example, a complete max-run
deadline report retains `reason=max_run_deadline` whether the following close is
a FIN or reset. A FIN or reset at a frame boundary without that complete report
has `reason=unknown_disconnection`; EOF inside a frame has
`reason=truncated_stream`. Clients must not infer the worker's stderr reason.
A normal Complete frame is authoritative and does not wait for a later socket
close, so its transport remains `not_observed`.

## Wire compatibility

Wire version 3 identifies streams that can carry a terminal report. The new
decoder accepts complete legacy v2 streams as well as v3, and rejects a version
change within one stream. A legacy decoder rejects v3 at the first Hello rather
than consuming a long run and failing only at its terminal frame. A new worker
preserves v2 while forwarding a legacy plugin and omits the v3-only deadline
report for that stream.

Terminal report kind 17 has no payload; its `detail` word contains a report version, reason,
and six progress flags. Version 1 currently defines only `max_run_deadline`.
Decoders validate the reserved bits and flag dependencies. A report may be the
first frame when the target never produced Hello.

A malformed report, a report
that contradicts already received progress, extra bytes after a report, or a
partial report never becomes a named deadline or a complete training example.

The worker sends a deadline report only when no target frame is partly forwarded.
It makes one nonblocking 32-byte attempt after the absolute budget expires. A full
or broken socket can therefore lose or truncate the diagnostic, preserving the
run and cleanup bound. The client then reports the transport evidence it actually
received rather than guessing the missing reason.

The direct process worker and the managed full-system `--kernel-protocol` path use
the same version 3 report. The managed path tracks the Hello and observation data
that it has actually forwarded, and caps client and QMP writes by the remaining
absolute budget. A learner that stops reading therefore cannot turn
`--max-run-ms` into the longer transport inactivity timeout.
