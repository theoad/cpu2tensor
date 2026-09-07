# Kernel pretraining

The benign guest workload boots and completes. Full-system trace capture and a measured
multiworker training run are pending a compatible operator-provided QEMU and
backend implementation.

See [build, workload, and acceptance instructions](../../docs/kernel-examples.md#observation-only-pretraining).
The shared target lives in [native/](../../native/examples/kernel_init.c).

The [bounded pretraining learner](../../docs/kernel-pretraining.md) accepts
operator-started observation endpoints. Its user-process check is separate from
pending kernel trace capture.
