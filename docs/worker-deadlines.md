# Bound an observation worker's runtime

`--max-run-ms 60000` sets one absolute 60-second budget for an observation run:

```sh
cpu2tensor-worker --qemu /path/to/qemu --plugin /path/to/plugin \
  --input /path/to/input --max-run-ms 60000 --timeout-ms 30000 \
  --batching mixed --publication pipe -- /path/to/target
```

The budget starts immediately before QEMU is launched, after the consumer has
connected. It includes startup, guest execution, trace forwarding, backpressure,
and waiting for the child to exit after its trace seal. Waiting for a consumer
and opening the prescribed input happen before it. Each new run gets a new budget.
The accepted value is 1 through 2,147,483,647 milliseconds; omitting the option
leaves total runtime unlimited. Observation-only user processes and full-system
guests support it. Stdin and kernel action adapters reject it explicitly for now.

Successful reads, writes and diagnostic activity never refresh this deadline.
The worker caps blocking polls by the remaining budget and checks it while data
is continuously ready, while sending, and during post-seal shutdown. This uses
monotonic wall time, not guest instructions or guest virtual time. It therefore
includes time spent paused by backpressure or descheduled by the host.

On expiry the worker reports `Target exceeded --max-run-ms deadline`, attempts a
versioned in-band [terminal outcome](terminal-outcomes.md), kills and
reaps its owned child/process group, closes the connection and exits nonzero.
It does not synthesize a successful capture seal. Pool consumers get an incomplete
trace error; already received tensors remain valid partial data, not a complete
training example. A client may still have buffered bytes after the worker stops,
so this is not a deadline on learner work or on when the client reads its error.
The report attempt is nonblocking and can be absent or truncated under transport
backpressure. Python names the deadline only after validating the complete report.
Operating-system scheduling and child reaping add latency; this is an absolute
worker budget, not a hard real-time scheduling guarantee.

The managed full-system path applies the same bound to trace forwarding and QMP
writes. Its terminal report records whether plugin negotiation and observation
data reached the client before expiry. This keeps deadline diagnosis identical
for ordinary process targets and kernel guests.

`--timeout-ms` remains the existing operation timeout. It can fail an inactive
operation before the total budget expires. The pipeline benchmark's
`--max-seconds` is a separate soft learner consumption budget checked between
batches. Its `run_kernel_pipeline.py` wrapper now also accepts `--max-run-ms` and
forwards that value to every observation worker. Keep all three meanings distinct.

No plugin callback, trace format, tensor API or QEMU build change is needed.
When disabled, the forwarding path makes no additional clock reads. When enabled,
it checks the clock at worker I/O boundaries, not per guest instruction or memory
callback. No throughput comparison of deadline-enabled runs has been measured.
The operator must build the new worker; old workers reject the unknown option.

## Checked behavior

On 2026-09-08, `trail-arm` ran the actual AArch64 QEMU worker with benign fixtures.
Seventeen checks passed across two commands: seven deadline tests (including one
five-case option validation test) and ten existing window/ring tests. Expiry tests
cover continuously produced blocks, an idle target, a non-reading consumer and a
lifecycle fixture that sends a seal but never exits. Each checks incomplete client
completion, exact deadline diagnostics, nonzero worker exit and disappearance of
the QEMU child. Generous-budget normal completion and waiting longer than the
budget before connecting also pass. Parser checks reject zero, negative and
overflow values, plus both interactive adapter modes.

On `trail-x86`, an ordinary two-vCPU Linux 6.9.0-dirty boot with the existing benign
example archive hit a 1,000 ms budget while its operation timeout stayed 30 seconds.
The test received real blocks, observed an incomplete trace, checked nonzero worker
exit and the deadline diagnostic, and confirmed the child was reaped. The single
acceptance test passed in 3.06 seconds including setup and SSH checks; that is not
measured deadline overshoot or a throughput result. The reserved host was released
with no QEMU process left running.

Builds are separate from the prior performance evidence: x86 source/build live
under `~/.cache/cpu2tensor/deadline/`; the previous performance binaries remain
unchanged. The ARM shared-source worker build was rebuilt. The plugin is unchanged.
Both restored ring tests were formerly placed below `unittest.main()` outside
the test class; this iteration fixes discovery and runs them as real tests.

The final default Python regression suite passed 158 tests, with 62 explicit
unavailable-backend or unconfigured-worker skips. The real tests above ran
separately with their operator environment. [Source hashes](deadline-source.sha256)
pin this implementation; earlier performance manifests retain their measured
historical sources.

The accepted x86 worker binary SHA-256 is
`d809fe02b6d205f671f37081b78ed6f3410a2f681ded22ae60f92ea7204d1ecb`.
Its plugin SHA-256 remains
`c0e8f8e184d713580c8f85c2c1f602797525141bc456373816ad395990c67bcd`.
