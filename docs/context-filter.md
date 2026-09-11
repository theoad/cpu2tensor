# Paging-context filtered rich capture

Full-system traces often contain a small target workload among large amounts of
background kernel and process activity. `--rich-context-start-pc ADDRESS` keeps
the inexpensive block and paging-context stream across all vCPUs, while limiting
register checkpoints and memory transactions to one address space.

The address is an exact guest basic-block entry. When any vCPU reaches it, the
plugin latches that source's raw CR3 once. Rich events are admitted when the
current CR3 has the same page-table root, `CR3 & ~0xfff`. Ignoring the low PCID
and cache-control bits lets one address space migrate between vCPUs. Both the raw
gate value and compared root are present in the final metadata. They identify a
relation within one run; neither value is a PID or a stable identity across boots.

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

`drop` is the default policy. It drops memory transactions before the latch and
from foreign roots. `keep` retains matching, foreign and not-yet-related rich
events; it is a diagnostic escape hatch with the same accounting. A requested
gate that never executes fails the capture. The mode is observation-only and is
incompatible with the older whole-stream start/stop gates and action windows.

Blocks and sparse `AddressContext` changes remain per-vCPU events with their
ordinary source sequences. Dropping a rich event does not create a synthetic
event or a sequence hole. Publication remains per source; arrival across sources
does not imply guest memory order.

Registers are sampled only at admitted basic-block and vCPU-exit checkpoints.
The first admitted checkpoint on each source is a complete baseline, including
when the latched address space first migrates there. A source that never runs the
target context can end without register data. These remain boundary samples, not
every register write or a paused-world snapshot.

Filtered memory callbacks use the CR3 cached at their owning block entry. x86
paging-control writes terminate translated execution before a later memory
instruction, so the next callback follows a new block entry and refreshed cache.
This avoids a register read on every memory transaction. Events before the latch
are explicitly `unknown`; the plugin does not guess their relation. Unfiltered
capture retains its stronger per-memory callback context refresh.

The last yielded worker-wide batch contains `batch.context_filter`:

- `policy`, `latch_known`, `gate_source`, `gate_pc`, `cr3`, and `paging_root`
  describe the one-shot latch and comparison.
- `matching`, `foreign`, and `unknown` partition every candidate memory callback.
- `kept` and `dropped` report the result of the selected policy.

All counters are exact for callbacks admitted by the surrounding observation
run. They do not count DMA, implicit MMU writes, or memory activity outside QEMU's
plugin callbacks. The summary follows every source end and consumes no vCPU
sequence. The decoder rejects a missing, repeated, internally inconsistent, or
early summary.

`--max-run-ms` remains one absolute observation budget. Rich filtering does not
renew it: pipe reads, socket sends and lossless backpressure all consume the same
deadline. A timeout still kills and reaps only that worker's QEMU process and
reports an incomplete trace.
