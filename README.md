# cpu2tensor

[![CI](https://github.com/theoad/cpu2tensor/actions/workflows/ci.yml/badge.svg)](https://github.com/theoad/cpu2tensor/actions/workflows/ci.yml) [![coverage](https://codecov.io/gh/theoad/cpu2tensor/branch/main/graph/badge.svg)](https://codecov.io/gh/theoad/cpu2tensor)

CPU execution traces as tensors and interactive learning environments.

cpu2tensor is a C++20 library with a small Python interface. It streams CPU
execution observations into tensors for learning.

**Status:** early development. ARM/x86 user-process tracing and CPU/MPS learning
are checked. Single-vCPU stdin interaction is available with a Gymnasium wrapper.
Rich x86 kernel capture, multiworker pretraining, and kernel Gym actions are
checked with two active vCPUs. CUDA execution and capture throughput optimization remain open.

Start with [the examples](example/README.md):

- [Trace digits](example/trace_digits/README.md): generate 300 inputs, capture one
  run, and train a 110-parameter classifier. CPU and MPS reached 100% on 60 held-out
  inputs; shuffled controls are recorded.
- [Stdin Gym](example/stdio_gym/README.md): observe a cue, choose a legal digit,
  and learn from success rewards. Real MPS REINFORCE passed 40/40 fresh decisions.
- [Kernel pretraining](example/kernel_pretraining/README.md) and
  [kernel actions](example/kernel_gym/README.md): real kernel observation-only training and paused-world Gym actions.

With an [operator-started worker](docs/quickstart.md):

```python
import torch
from cpu2tensor import Pool

accessed_bytes = torch.zeros((), dtype=torch.int64, device="mps")
with Pool(["tcp://127.0.0.1:9000"], device="mps") as pool:
    for batch in pool.read():
        if batch.memory is not None:
            accessed_bytes += batch.memory.sizes.sum()
print(accessed_bytes.item())
```

`batch.registers` contains named register changes and exact value bytes;
`batch.memory` contains instruction PCs, addresses, sizes, flags, and optional
value bytes. `batch.addresses` retains block-entry addresses. Each batch holds one
signal's consecutive events. See [the instrumentation contract](docs/instrumentation.md).

General registers and memory addresses/sizes are enabled by default. Memory values
are opt-in with worker option `--memory-values on`. The target runs on prescribed
input. No policy or action calls are required.
Slow consumers apply backpressure; failures raise errors instead of silently
dropping trace data. See the [complete quickstart](docs/quickstart.md).

- [Product and architecture](docs/architecture.md)
- [Development workflow](docs/development.md)
- [Current work and milestones](docs/backlog.md)
- [Development environments](docs/environments.md)

QEMU is supplied by the operator and is never bundled or installed by the package.
The license is [AGPL-3.0-only](LICENSE).
