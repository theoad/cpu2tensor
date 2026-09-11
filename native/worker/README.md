# Worker

Run one prescribed target and forward its trace to one consumer. `main.cpp`
owns the listener and Linux-user path; `kernel.cpp` owns QMP and the two serial
channels for validated system-guest protocols.
See [quickstart](../../docs/quickstart.md), [kernel setup](../../docs/kernel-examples.md),
[repeated action windows](../../docs/action-windows.md), and
[validation](../../docs/validation.md).
