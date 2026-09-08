# Observe an AArch64 program

Use an AArch64 Linux worker and a Mac learner. The same client also works on CPU.
The worker runs one prescribed target per connection and reports block entries,
sampled general-register changes and successful memory transactions. For system guests, see [kernel capture](kernel-examples.md). For real model training, continue with
[trace digits](learn-trace.md); [stdin learning](stdio-example.md) uses the
separate interactive worker mode.

## Build the worker on Linux

The operator provides QEMU with Linux-user AArch64 plugin support, its matching
`qemu-plugin.h`, CMake 3.24+, Ninja, a C/C++ compiler, pkg-config, GLib headers, and json-c 0.15+ development headers.
The package does not install or bundle QEMU. The tested development build is
described in [the probe](qemu-probe.md); a version string alone is not proof of a
compatible build. A mismatched plugin API must be resolved using the matching
operator build, not by changing the plugin's reported version.

From the repository on the Linux worker:

```sh
export QEMU=/path/to/qemu-aarch64
export QEMU_INCLUDE=/path/to/matching/include
export C2T_BUILD="$HOME/.cache/cpu2tensor/quickstart-worker"
cmake -S native -B "$C2T_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCPU2TENSOR_BUILD_WORKER=ON \
  -DCPU2TENSOR_QEMU_INCLUDE_DIR="$QEMU_INCLUDE"
cmake --build "$C2T_BUILD" -j2
printf 'cpu2tensor\n' > "$C2T_BUILD/input.txt"
"$C2T_BUILD/cpu2tensor-worker" \
  --qemu "$QEMU" --plugin "$C2T_BUILD/libcpu2tensor_plugin.so" \
  --input "$C2T_BUILD/input.txt" --port 9000 -- "$C2T_BUILD/checksum"
```

The worker prints its endpoint and waits for one client. Connection starts the
target; its stdout remains on the worker terminal. This input must produce
`bytes=11 sum=1055`. The fixture is a normal C17 executable with no trace adapter.
It is linked without PIE so a test can independently match its ELF `main` address
to a captured block address. Other binaries may use ASLR; addresses stay raw.

General-register sampling and memory addresses/sizes/direction are enabled by
default. Add `--memory-values on` to include actual transaction values. Use
`--registers none --memory off` for the minimal block-only path. `--registers all`
requests all QEMU registers within the supported limits; the default ARM SME ZA
register exceeds those limits and produces a clear failure. See
[register and memory semantics](instrumentation.md) before treating samples as
completed-block effects or final process state.

The default listener is loopback. To use the VM from a Mac, either add
`--host WORKER_PRIVATE_IP` or provide your own tunnel, for example run
`ssh -N -L 9000:127.0.0.1:9000 trail-arm` in another terminal. The transport has no
authentication or encryption; use an operator-controlled connection or tunnel.
Worker and client timeouts default to 30 seconds. Increase `--timeout-ms` and the
client's `timeout` for longer pauses. The worker serves one run and then exits.

## Install and read on the learner

From the repository on macOS, using a Python with MPS-capable PyTorch support:

```sh
python3 -m venv "$HOME/.cache/cpu2tensor/quickstart-python"
. "$HOME/.cache/cpu2tensor/quickstart-python/bin/activate"
python -m pip install -e .
```

Use the tunnel endpoint, or replace the address with the worker's private endpoint:

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

For a complete command-line example:

```sh
python python/examples/observe.py tcp://127.0.0.1:9000 --device mps
```

Use `device="cpu"` (and a CPU accumulator) or `--device cpu` on a CPU learner.
Batch sizes can vary. `batch.source` identifies a vCPU and `batch.first_sequence`
is its first entry number. Retained batches keep their own storage. There are no
policy calls or action requests; the consumer chooses when to request another batch.

Slow consumption applies bounded backpressure. Leaving the `with` block early
cancels the run. Disconnects, sequence gaps, and failed targets raise errors; do
not treat the last yielded batch as evidence of successful completion. See the
[instrumentation contract](instrumentation.md) and [validation](validation.md) for exact limits.
