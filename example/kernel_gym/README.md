# Kernel Gym example

The guest side adapter passes real getpid, memory-checksum, and pipe-roundtrip
checks. A runnable kernel Gym wrapper, whole-machine pause, and trace-tail barrier
remain pending.

See [adapter and boot instructions](../../docs/kernel-examples.md#named-syscall-adapter).
The shared target lives in [native/](../../native/examples/kernel_init.c).
