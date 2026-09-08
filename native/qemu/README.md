# QEMU capture

`plugin.cpp` records per-vCPU blocks, selected register changes, and successful
memory transactions from operator-provided user/system QEMU. See the
[signal contract](../../docs/instrumentation.md) and
[kernel API review](../../docs/kernel-qemu-build.md) for ordering and coverage.

The optional [developer QEMU patch](../../docs/qemu-state-hook.md) supplies exact
x86 raw state. For new signals, follow [the contributor recipe](../../docs/adding-a-signal.md).
