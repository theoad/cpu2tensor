# Kernel Gym example

The [installed example](../python/cpu2tensor/examples/learn_kernel.py) uses
`KernelEnv` to observe an ordinary Linux guest and choose among four fixed,
benign workloads. It checks their results, performs a few policy-gradient
updates, and sends `quit`. Clients define the action space, trace reducer,
reward, and episode outcome. The package owns neither that learning objective
nor the operator's QEMU installation or remote connection setup.

## Start a worker

Build the static guest and initramfs using [the guest setup](kernel-examples.md).
The operator needs a compatible plugin-enabled `qemu-system-x86_64`, its matching
header, an ordinary x86-64 Linux kernel, and a built worker/plugin. Detailed
capture uses TCG. KVM does not produce these TCG plugin events.

On the worker host, set paths to the supplied artifacts and start:

```sh
export C2T_BUILD=/path/to/native/build
export QEMU_SYSTEM=/path/to/qemu-system-x86_64
export KERNEL=/path/to/x86_64/bzImage
export INITRAMFS=/path/to/initramfs.cpio.gz

# The marker belongs to this exact guest binary. Do not reuse another build's PC.
export C2T_START_PC="$(nm "$C2T_BUILD/kernel_init" | awk '$3 == "cpu2tensor_capture_begin" {print "0x" $1}')"

"$C2T_BUILD/cpu2tensor-worker" \
  --qemu "$QEMU_SYSTEM" --plugin "$C2T_BUILD/libcpu2tensor_plugin.so" \
  --system on --kernel-adapter on --registers none --memory off \
  --start-pc "$C2T_START_PC" --host 127.0.0.1 --port 9400 \
  --episodes 1 --timeout-ms 120000 -- \
  -accel tcg,thread=multi -smp 2 -m 256M -nic none -no-reboot \
  -kernel "$KERNEL" -initrd "$INITRAMFS" \
  -append 'console=ttyS0 rdinit=/init panic=-1 cpu2tensor.mode=interactive'
```

The marker starts a post-boot capture window; boot events are deliberately
excluded. Omit `--start-pc` when the full boot trace is required. The example
reduces only basic blocks, so this command disables register and memory capture.
Those signals can remain enabled for clients that use their columns. The worker
owns QMP, ttyS0 diagnostics and the private ttyS1 protocol channel; do not supply
QEMU serial, monitor, or initial pause options. An operator can expose this
endpoint through an existing tunnel or trusted network. The Python example does
not configure that connection.

## Run the learning client

Install the optional Gym dependency with `pip install 'cpu2tensor[gym]'`. On the
learner, where `tcp://127.0.0.1:9400` reaches the worker:

```sh
python -m cpu2tensor.examples.learn_kernel \
  --endpoint tcp://127.0.0.1:9400 --device cpu --sources 2 \
  --policy-steps 4 --output "$HOME/.cache/cpu2tensor/kernel-gym-check"
```

Use `--device mps` on Apple Silicon or `--device cuda` on a configured NVIDIA
learner. Trace reduction stays on CPU; only the small fixed feature vector and
model live on the selected accelerator. `--policy-steps 0` checks all commands
and completion without training. More steps require a suitably sized worker
deadline. This example allows at most 1000 updates and retains only their scalar
metrics, four calibration results, and a model checkpoint.

The stream API itself is small:

```python
from cpu2tensor import KernelEnv
from cpu2tensor.examples.learn_kernel import SourceBlockFeatures, verify_result

observe = SourceBlockFeatures(sources=2)
with KernelEnv("tcp://127.0.0.1:9400", timeout=120) as env:
    initial = observe(env.reset())
    observation = observe(env.step(b"memory 17 256\n"))
    verify_result(1, env.result)
    final = observe(env.step(b"quit\n"))
    assert env.exit_code == 0 and env.event["ok"] is True
```

Each reducer drains the iterator as chunks arrive. A reset or step returns its
last chunk before the worker-certified action request. At that boundary all
vCPUs are paused and their captured tails have been drained. The guest's serial
`ready` line alone is insufficient to establish this boundary. The source ID and
sequence stay attached to every trace batch; stream arrival order does not make
a total memory order across CPUs. Reset cancels the current run and starts a new
guest through a worker configured for another episode.

## What the learning check means

The five client actions are `getpid`, `memory 17 256`, `pipe 17 64`,
`parallel 17 256`, and `quit`. The policy samples only the first four. These have
fixed guest implementations and accept no arbitrary syscall, address, filename,
or code. The memory and pipe byte sum follows the documented unsigned 32-bit
generator; the parallel action checks two independent sums and distinct guest
CPU IDs. A wrong result raises an error before a model update. Normal completion
requires both the guest's successful completion event and a complete trace with
worker exit code zero.

Each CPU gets a separate 64-bin block histogram and a log block-count feature.
The histogram hashes exact 64-bit block addresses; collisions and loss of order
are intentional model-feature choices. The original trace tensors keep their
bits. The reducer keeps bounded counts while streaming and does no Python loop
per block. It rejects an unexpected CPU instead of merging that CPU into an
existing row. `--sources` must cover the guest's configured vCPU indices.

After four calibration actions, the client starts a fresh coverage bitmap for
one guest trajectory. Reward is newly encountered CPU/bin pairs divided by all
bins, minus a small penalty for executed blocks. This is a coarse illustration
of client reward: it saturates, can count hash collisions, and counts kernel
background work. The nominal block budget scales that penalty; it does not
pause execution at a hard budget. Every sampled action uses the preceding trace
features. Its verified result determines the reward, never the policy inputs.
One REINFORCE update uses that immediate reward and a small entropy bonus. There
is no replay buffer, value model, or cross-worker recurrent state.

The check establishes a functioning Gym/action/observation/model path and finite,
changed parameters. Four updates cannot establish convergence, improved kernel
coverage, or an efficient learning algorithm. For a held-out prediction task and
negative controls, see [trace classification](learn-trace.md). For observation
without actions, use [kernel pretraining](kernel-pretraining.md).

## Validation

Nine socket fixture tests exercise the native decoder, multiple CPU sequences,
retained block and memory-value tensors, malformed control frames, missing
source tails, cancellation, timeouts, exact guest-result checks, and a complete
Gym policy update. Run them with:

```sh
python -m unittest discover -s python/tests -p test_kernel.py -v
```

On 2026-09-07 all nine fixtures passed on the macOS Apple M2 learner. Two separate
real guest runs then passed through the worker on Linux x86-64 `trail-x86`, using
QEMU 11.0.3, TCG with two vCPUs, 256 MiB RAM, and the operator's
`/boot/vmlinuz-6.9.0-dirty`. The Mac learner used PyTorch 2.13.0, first with a CPU
model and then with MPS. Both runs used post-boot block capture, 64 bins per CPU,
seed 7, four calibration commands, four sampled policy updates, and explicit
`quit`.

Both runs returned guest PID 1; memory checksum 31717; pipe checksum 6863; and
parallel checksums 31717/31310 on guest CPUs 0 and 1. Every update had finite loss
and gradients. Both models changed parameters, and both traces completed with
successful guest completion and exit 0. Their checkpoints reloaded on CPU and
produced finite logits. The command choices differed between CPU and MPS;
the seed does not make different device RNGs or kernel scheduling identical.

Artifacts are under the learner's `$HOME/.cache/cpu2tensor/kernel-gym-check/`,
with `cpu/` and `mps/` containing `metrics.json` and `policy.pt`, plus a
`reload.json` check. Checkpoint SHA-256 values are:

| Model device | Checkpoint SHA-256 |
| --- | --- |
| CPU | `bb041fcfc5b5f080aff666859e1a48f03a7fb99e947ab2142b22545300ba6733` |
| MPS | `af0561e7b08b3092eace8a1d5ceed48fdb8c64c0fbf0d44f86d86d4059a21b3e` |

The checked worker had SHA-256
`7067fe87cc9aeb001cb34772f17faa87b4a9980ba31456e1779ab7d6636916c6`,
and plugin
`287452b554f56c0410fbfa7d12e8e304303b8c223feef26705229e8410ee0e8d`.
The static init binary had SHA-256
`cab2a45a87f49ac9f7f43b4ce19f8bf08d96e02571f0c6068eb3954681222c84`
and marker PC `0x402670`. These values identify this check, not stable addresses
for another build. The repeated worker exited after its two episodes.

This block-only learning run supplements the worker's separate pause and rich
capture checks; fixture success or guest CPU IDs alone cannot prove captured
activity on two CPUs or a drained trace boundary. No timing comparison or
convergence claim was made. CUDA remains unverified until a configured CUDA
learner runs the example.
