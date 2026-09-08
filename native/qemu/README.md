# QEMU capture

`plugin.cpp` records per-vCPU blocks, selected register changes, and successful
memory transactions from operator-provided user/system QEMU. See the
[signal contract](../../docs/instrumentation.md) and
[kernel API review](../../docs/kernel-qemu-build.md) for ordering and coverage.
