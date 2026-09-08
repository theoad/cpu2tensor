# Worker

Run one prescribed target and forward its trace to one consumer. `main.cpp`
owns the listener and Linux-user path; `kernel.cpp` owns QMP/serial system actions.
See [quickstart](../../docs/quickstart.md), [kernel setup](../../docs/kernel-examples.md),
and [validation](../../docs/validation.md).
