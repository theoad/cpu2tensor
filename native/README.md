# Native implementation

Capture, framing, worker control, Python bindings, tests, and example targets use
one CMake build. See [architecture](../docs/architecture.md),
[build and validation](../docs/validation.md), and [the current slice](../docs/backlog.md).

Language: C++20, with C17 guest-facing headers. The standalone CMake project opens
directly from this directory; see [IDE setup](../docs/ide.md).
