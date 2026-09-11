# Paging-context filtered rich capture

Full-system traces often contain a small target workload after a large boot and
among background kernel and process activity. `--rich-context-start-pc ADDRESS`
starts one observation window, keeps the block and paging-context stream across
all vCPUs in that window, and limits register checkpoints and memory transactions
to one address space.

The address is an exact guest basic-block entry. When any vCPU reaches it, the
plugin latches that source's raw CR3 once and publishes that gate as the first
block of the window. Rich events are admitted when the current CR3 has the same
page-table root, `CR3 & ~0xfff`. Ignoring the low PCID and cache-control bits lets
one address space migrate between vCPUs. Both the raw gate value and compared root
are present in the final metadata. They identify a relation within one run;
neither value is a PID or a stable identity across boots.

The filter deliberately has no process or syscall attribution semantics. Linux
may use a distinct, paired kernel page-table root while Kernel Page Table
Isolation (KPTI) handles a syscall or interrupt. That root is `foreign` because
cpu2tensor has no generic evidence that it belongs to the process whose user root
was latched. With the default `drop` policy, those kernel-side transactions are
therefore omitted. The current relation guarantees capture under the exact
normalized root only; it does not guarantee complete syscall activity for a
process. A workload or benchmark must state its KPTI configuration and keep its
acceptance values under the nominated root unless it separately validates a
kernel-root relation.

```text
cpu2tensor-worker \
  --qemu /operator/qemu-system-x86_64 \
  --plugin /operator/libcpu2tensor_plugin.so \
  --system on --kernel-protocol on \
  --registers general --memory on --memory-values on --context on \
  --rich-context-start-pc 0xffffffff81001000 \
  --rich-context-policy drop \
  --max-run-ms 300000 \
  -- image-and-machine-arguments...
```

No public block, context, register or memory row precedes the gate. Inline memory
totals accumulated before a source enters the window are discarded rather than
reported as unknown. `drop` is the default policy and drops in-window memory from
foreign or unknown roots. `keep` retains matching, foreign and unknown rich events
inside the same window; it does not retain boot activity. A requested gate that
never executes fails the capture. The mode is observation-only and is incompatible
with the older whole-stream start/stop gates and action windows.

The gate source starts at the gate block. Each other vCPU starts at its first block
boundary after it observes the published latch; the block it was already executing
is outside that source's window. Blocks and sparse `AddressContext` changes remain
per-vCPU events with their ordinary source sequences. Dropping a rich event does
not create a synthetic event or a sequence hole. Publication remains per source;
arrival across sources does not imply guest memory order.

Registers are sampled only at admitted basic-block and vCPU-exit checkpoints.
The first admitted checkpoint on each source is a complete baseline, including
when the latched address space first migrates there. A source that never runs the
target context can end without register data. These remain boundary samples, not
every register write or a paused-world snapshot.

Filtered memory callbacks use the relation selected from CR3 at their owning
block entry. x86 paging-control writes terminate translated execution before a
later memory instruction, so the next access follows a new block entry and a
refreshed relation. The optional [QEMU conditional-memory extension](qemu-memory-condition.md)
tests a live per-vCPU admission scoreboard before entering the rich callback.
This avoids both a register read and a rejected C++ callback on every background
transaction, while already translated blocks respond to later vCPU migration.
Unfiltered capture retains its stronger per-memory callback context refresh.

Each source retains its latest in-window block relation. The gate root and gate
rows are published before the global started state is released. A concurrent vCPU
therefore discards the memory delta of the block it was already executing, then
observes the latch and enters at its next block boundary. The gate source also
discards its preceding block and classifies the gate block as matching. Later
context changes take effect at that source's next block boundary. These per-source
boundaries do not claim a total order across vCPUs.

The last yielded worker-wide batch contains `batch.context_filter`:

- `policy`, `latch_known`, `gate_source`, `gate_pc`, `cr3`, and `paging_root`
  describe the one-shot latch and comparison.
- `matching`, `foreign`, and `unknown` partition every in-window candidate memory
  callback.
- `kept` and `dropped` report the result of the selected policy.

An unconditional inline per-vCPU total counts candidate QEMU plugin callbacks.
Before a source enters the window, block entry advances that source's baseline and
discards the delta. From its first in-window block onward, each block entry and
vCPU exit assigns the delta to the preceding block's cached relation. The
conditional rich callback emits only admitted rows. All counters are therefore
exact for callbacks in the observation window, including rejected callbacks that
never enter cpu2tensor C++. They do not count DMA, implicit MMU writes, or memory
activity outside QEMU's plugin callbacks. The summary follows every source end
and consumes no vCPU sequence. The decoder rejects a missing, repeated, internally
inconsistent, or early summary. It also counts emitted memory rows, including
memory subruns in a mixed frame, and requires that count to equal `kept`. The named
gate source must have appeared and ended. These checks expose a broken admission
hook or producer counter instead of silently accepting its metadata.

`--max-run-ms` remains one absolute observation budget. Rich filtering does not
renew it: pipe reads, socket sends and lossless backpressure all consume the same
deadline. A timeout still kills and reaps only that worker's QEMU process and
reports an incomplete trace.

## Current acceptance status

The portable two-source fixture validates exact kept values, source-local order,
target migration, background exclusion, and final accounting. The Linux CI build
also compiles the opt-in full-system fixture. This is contract evidence, not a
kernel performance result.

On 2026-09-11, the first bounded `trail-x86` run with the conditional-memory QEMU
extension used cpu2tensor `64ca11f`, the operator's x86-64 QEMU 11.0.3 MTTCG
build, two vCPUs, and Linux `6.9.0-dirty`. A faulty measurement relay disconnected
inside a frame, so this run is not completion or correctness evidence. Its
761,633,664-byte partial trace nevertheless contained 93,369,970 complete block
rows and 993 context rows, all before the gate, and no rich rows. This exposed a
separate contract bug: the filter rejected rich boot activity but still published
the entire low-cost boot stream. The worker and QEMU guest exited and the host was
audited clean. The retained issue baseline of 39.24 seconds and 16,689,464 bytes
is a different context-only workload, so it is not a matched comparison.

The relay now keeps both forwarding sockets blocking, applies downstream
backpressure through `sendall`, and reports forwarding errors separately from
decoder errors. A local 128 MiB fixture with 4 KiB socket buffers and 2,048
deliberate consumer pauses forwarded the exact byte count and SHA-256 in 5.327
seconds without retaining the stream in memory. This validates the measurement
path; it does not validate the guest capture.

The tested kernel configuration has
`CONFIG_MITIGATION_PAGE_TABLE_ISOLATION=y`, and its command line contains no PTI
override. Its boot log does not independently prove that PTI became active for
this virtual CPU. The fixture therefore keeps its asserted target values in user
space and makes no kernel-root attribution claim.

The repository now keeps every public signal outside the pre-gate interval and
contains the QEMU extension needed to keep that interval and later background
memory out of the rich C++ callback. QEMU 11.0.3 and ordinary system tests compile
the revised plugin locally. This mode has not yet passed the real two-vCPU
acceptance or demonstrated the requested size/runtime reduction. It remains
incomplete until the patched QEMU runs the exact target/background fixture within
the bounded deadline.
