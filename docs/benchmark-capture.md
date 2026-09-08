# Measure capture overhead

`cpu2tensor.examples.benchmark_capture` runs a small operator-written list of
commands on one machine. It compares each complete instrumented run with the
same workload under vanilla QEMU. It does not start cloud instances, connect to
remote hosts, install QEMU, or choose workload inputs.

Use an otherwise idle Linux x86-64 host for performance evidence. Reserve that
host before running; the benchmark does not coordinate competing jobs. macOS and
ARM fixture runs establish runner portability only. Results from different host
or guest ISAs are not comparable slowdown measurements.

## A manifest and one command

Save this JSON on the benchmark host. Replace the absolute paths with the
operator-provided QEMU, plugin and benign target. This example uses the existing
native checksum fixture and the same input file in every case. `checksum` is a
placeholder target filename; use the actual built target path.

```json
{
  "host_name": "trail-x86",
  "host_isa": "x86_64",
  "guest_isa": "x86_64",
  "workload": "Checksum fixture, fixed input; record the source revision here",
  "cwd": "/home/user/benchmarks",
  "stdin": "input.bin",
  "baseline": "vanilla",
  "repeats": 5,
  "timeout_seconds": 120,
  "files": [
    "/operator/bin/qemu-x86_64",
    "/operator/lib/cpu2tensor-plugin.so",
    "/operator/targets/checksum"
  ],
  "cases": [
    {
      "name": "vanilla",
      "signals": "No plugin",
      "argv": ["/operator/bin/qemu-x86_64", "/operator/targets/checksum"]
    },
    {
      "name": "blocks",
      "signals": "Basic-block entries only",
      "argv": ["/operator/bin/qemu-x86_64", "-plugin", "/operator/lib/cpu2tensor-plugin.so,fd={trace_fd},registers=none,memory=off,values=off", "/operator/targets/checksum"]
    },
    {
      "name": "rich",
      "signals": "General registers at block entry; memory addresses and sizes",
      "argv": ["/operator/bin/qemu-x86_64", "-plugin", "/operator/lib/cpu2tensor-plugin.so,fd={trace_fd},registers=general,memory=on,values=off", "/operator/targets/checksum"]
    },
    {
      "name": "rich-values",
      "signals": "Same coverage as rich, plus transaction values",
      "argv": ["/operator/bin/qemu-x86_64", "-plugin", "/operator/lib/cpu2tensor-plugin.so,fd={trace_fd},registers=general,memory=on,values=on", "/operator/targets/checksum"]
    }
  ]
}
```

```bash
python -m cpu2tensor.examples.benchmark_capture manifest.json --output /home/user/benchmarks/results-001
```

Run this command on the intended benchmark host with its installed cpu2tensor.
The Python package and native extension must match the tested plugin's protocol.
The output directory must be new and should be on a local filesystem, outside a
shared source mount. The manifest's `cwd` defaults to its own directory. Relative
`stdin` and `files` paths resolve against `cwd`; arguments retain their literal
spelling. No shell expansion, globbing or command substitution occurs.

`{trace_fd}` marks an instrumented case. The runner creates a pipe and substitutes
its inherited write descriptor into that argument. QEMU's plugin option is
`-plugin /path/plugin.so,fd=NUMBER,...`; a filename is not a replacement for this
pipe. Baseline commands must have no `{trace_fd}`. Commands must remain in the
foreground and must not escape their process group or leave background children.
Use observation-only plugin options, with `stdio=off` and `kernel=off` if specified.
The runner rejects interactive trace features.

The process receives the same `stdin` file on every run, or `/dev/null` when it
is omitted. Standard output and diagnostics go directly to each run's files;
they cannot fill an undrained subprocess pipe. Traces are discarded after bounded
framing checks and counts. Output files can be large: provide enough local disk,
and keep logging identical across cases. A console-heavy target also measures
console/filesystem overhead.

For system QEMU, use the same kernel, initramfs, vCPU count, TCG accelerator and
kernel arguments in every case. Supply `-nographic` or equivalent serial routing
in the operator arguments. Set each case's optional `success_stdout` string to
an unambiguous workload-completion marker. This matters because a powered-off or
panicked guest can leave QEMU with exit status zero. The benign kernel fixture's
`C2T` JSON completion line is one possible oracle; choose the exact successful
line emitted by the configured workload. Add the kernel, initramfs and any target
input artifacts to `files` so their SHA-256 identities enter the result. A marker
is an extra workload check, not proof of every guest result; use an independently
checked workload when making performance claims.

## What a successful run establishes

The child must exit with status zero. Instrumented runs must also produce Hello,
continuous event sequences within each source, every source's end frame, and a
final successful Complete frame followed by EOF. A capture with no observed
events is rejected. A trace error, partial frame, missing seal, timeout or
nonzero child exit makes the run unsuccessful. A plugin's callback seal alone
cannot establish the actual process exit status.

The native header parser validates frame kinds, sizes and supported features.
The sink counts rows from headers without interpreting individual events or
building tensors. It does **not** check register baselines, architectural values,
physical translations or payload contents; run the corresponding correctness
checks before comparing performance. Source progress is independent: no global
ordering between vCPUs is invented. New event kinds need an explicit counting
rule in `TraceCounter`, so an unfamiliar signal fails visibly.

The timeout covers execution **and** complete pipe draining. Failure kills the
child's process group and reaps the direct child. A descendant holding the pipe
open cannot turn a partial capture into a successful timing. Failed runs retain
their diagnostics and are excluded from successful medians; if any repetition
of a case fails, that case receives no timing median or slowdown ratio. The
command exits nonzero if any run fails. Output artifacts from earlier runs remain
available when a later run fails.

## Reading the results

`manifest.json` records the configuration. `results.json` is updated after every
run and contains the host name, OS, ISA, Python version, CPU count, initial load,
input/provenance file hashes, runner/native-decoder hashes, resolved working
directory, executed arguments and
raw repeated measurements. `run-000/`, `run-001/`, and subsequent directories hold
stdout and stderr. Cases run sequentially; each repetition rotates the first
case to expose cache and run-order effects. There is no automatic warmup or
outlier removal.

The reported wall time starts immediately before process launch and ends once
both the child has exited and the trace has reached EOF. It includes QEMU startup,
guest execution, plugin callbacks, local pipe publication and drain/validation.
The workload-success text search occurs afterward. Very short targets mainly
measure startup; choose workloads lasting seconds and inspect all raw times.
`child_elapsed_seconds` separately records the direct child's wait completion.

Per-child resource usage comes from POSIX `wait4`: user/system CPU time, peak RSS
in bytes, page faults and context switches. Reaped descendants can contribute to
these OS-reported child totals. `drain_user_seconds` and `drain_system_seconds`
measure the Python runner's CPU usage during the run, including framing and
bookkeeping. They help identify a slow sink. These metrics do not attribute time
to individual QEMU callbacks or measure ring occupancy.

For a case with only successful repetitions:

- `median_seconds` is the median complete-run wall time.
- `slowdown_vs_baseline` divides that median by the vanilla median on this host.
- Instrumented cases also report median events/s and trace bytes/s, using each
  complete run's total events/bytes divided by its own wall time.

Slowdown is meaningful only if the target, inputs, guest configuration and exit
conditions match. Register changes and address layouts can vary with scheduling
and ASLR; retain raw event counts instead of demanding identical counts between
independent runs. This CLI does not automatically prove workload equivalence or
host isolation.

The Python drain performs work per frame and can become the bottleneck for many
tiny rich frames. This is part of the measured pipe path, not an isolated register
getter benchmark. High drain CPU usage, target blocking or unstable timings call
for native profiling or a native sink before attributing the cost to a signal.
The benchmark performs no worker socket forwarding, tensor materialization,
accelerator transfer or learning. CUDA/AWS and multiworker throughput require
separate end-to-end measurements; these timings cannot establish GPU-bound work.

## Add a signal without hiding its cost

Start with the four profiles above. Keep block/checkpoint coverage and target
inputs fixed. For register comparisons, use the plugin's exact-name selection:
`registers=rip`, `registers=rax`, or `registers=rax:cr3` on a supported x86 build.
The plugin automatically includes the program counter. Compare the same
PC-only selection with one additional register at a time, then compare the
combined intended profile. Inspect the emitted schema and diagnostics; `all`
means supported fields, not proof that every QEMU register is meaningful.

Changing a register selection changes both getter work and the number of emitted
deltas. These runs measure their combined practical overhead. Register values
that remain constant can still be expensive to sample. If one profile is very
slow or times out, keep that result visible, investigate its QEMU reader and
sampling cost, and document the measured host/workload before making it a default.
A timeout is a failed measurement, not a measured slowdown ratio.

Apply the same method to memory values, translation metadata or a new native
signal: first validate its semantics, then measure it individually and in the
intended rich profile. Add its framing/counting rule and tensor tests alongside
the producer/consumer code. See [the capture audit](capture-efficiency.md) for the
ring, collector and accelerator boundaries that this simple baseline cannot
separate.
