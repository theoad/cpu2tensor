# Observation-only next-block training

The trainer has run on three real x86 Linux kernel workers with learning on
Apple MPS, as well as earlier AArch64 user-process checks on CPU/MPS. The system
backend and guest setup are described in [the kernel examples](kernel-examples.md).

The example predicts the next executed block from the previous block. It is a
small self-supervised starting point: an embedding and linear classifier, ordinary
PyTorch optimization, independent test runs and saved model weights. Target inputs
come from the operator. There are no model actions, rewards or Gym calls.

## Run the trainer

Start compatible observation workers through the operator's normal runner setup.
Use different prescribed workloads or inputs for training and test runs. Use the kernel worker command below on each operator-managed runner.

With endpoints available on a private connection or operator-provided tunnels:

```sh
python -m cpu2tensor.examples.pretrain_kernel \
  --train tcp://127.0.0.1:9000 --train tcp://127.0.0.1:9001 \
  --test tcp://127.0.0.1:9010 \
  --device mps --max-updates 100 \
  --output "$HOME/.cache/cpu2tensor/pretraining" \
  --capture-host "operator's worker and guest description"
```

Repeat `--train` or `--test` for more workers. The endpoint lists must be disjoint.
Use `--device cpu` for CPU execution; CUDA is accepted when the operator supplies
a CUDA learner, but no CUDA run is claimed here. Workers should finish their
prescribed workload so that the trainer can validate completion.

The learner opens no SSH connections and does not create cloud instances or
install dependencies. The operator-started native worker launches the supplied QEMU. The operator supplies endpoints for local, remote or
AWS runners. One learner process owns one device in this example; DDP, multiple
learner devices and fault-tolerant restart are future integrations.

## Trace and batching rules

The implementation is in
[`pretrain_kernel.py`](../python/cpu2tensor/examples/pretrain_kernel.py).
It opens one ordinary CPU `Pool` per endpoint in a reader thread. This is a small
example-level multiplexer; the core `Pool` still accepts one endpoint. A bounded
queue carries variable-sized block batches to the learner.

Each worker/vCPU has its own previous token. A pair is formed only between
consecutive observed blocks from that same source. Interleaving two workers or
two vCPUs never creates a training transition between them. A completed worker's
state is discarded. The model has no recurrent state, attention cache or public
trajectory registry.

The example maps addresses into a small vocabulary:

```python
token = ((address >> 2) ^ (address >> 12)) & (vocabulary - 1)
```

This is a deliberately lossy feature. Distinct addresses can collide. The raw
`Batch.addresses` tensor remains unchanged and retains its exact address bits.
ASLR and different code layouts can change tokens; address-space normalization
is a separate modeling choice. Memory transactions and register values are not
model inputs in this first trainer. Rich kernel capture is checked separately; this model intentionally uses block
transitions only.

The learner combines independent pairs into fixed-size minibatches. The final
partial minibatch uses its actual size, so there are no padding tokens or masked
loss entries. Default bounds are eight queued trace batches, 1,024 pairs per model
minibatch, and sixteen retained test minibatches. Each reader can also hold one
batch while waiting for queue space. Per-source state is one previous token.
Increasing the worker count therefore adds bounded reader and source state rather
than retaining complete traces.

Test readers run concurrently with training readers. A uniform reservoir of
completed test minibatches is kept on CPU for final evaluation. Sampling is by
minibatch, not by individual event; the last minibatch may be smaller. Loss and
accuracy weight the retained minibatches by their actual number of pairs. No test
data is used for gradient updates or feature fitting. Initial and trained models
are evaluated on the same retained test pairs.

The update budget limits optimization, not capture validation. After
`--max-updates`, the trainer keeps draining every stream through its completion
record and reports how many observed training pairs were unused. A late capture
failure still raises an error and does not produce successful metrics. Leaving
the reader context early cancels its unfinished runs; cancellation is not called
complete capture.

There is one synchronous minibatch transfer to the selected learner device.
This example has no pinned-memory pool, copy/compute overlap or throughput claim.
Reader queues and the worker transport apply bounded backpressure when learning
falls behind. Increase the worker and learner timeouts together for long pauses.

## Reproduce the checked user-process run

First build the worker and `branch_digits` target using the
[ordinary quickstart](quickstart.md). On the learner, create distinct inputs:

```sh
export C2T_DATA="$HOME/.cache/cpu2tensor/pretraining"
python -m cpu2tensor.examples.learn_trace inputs \
  --seed 7 --output "$C2T_DATA/train-inputs.json"
python -m cpu2tensor.examples.learn_trace inputs \
  --seed 17 --output "$C2T_DATA/test-inputs.json"
```

Transfer the adjacent `train-inputs.txt` and `test-inputs.txt` files using the
operator's share or file-transfer tool. In separate worker terminals, run:

```sh
"$C2T_BUILD/cpu2tensor-worker" \
  --qemu "$QEMU" --plugin "$C2T_BUILD/libcpu2tensor_plugin.so" \
  --registers none --memory off --input /path/to/train-inputs.txt \
  --port 9000 -- "$C2T_BUILD/branch_digits"
```

```sh
"$C2T_BUILD/cpu2tensor-worker" \
  --qemu "$QEMU" --plugin "$C2T_BUILD/libcpu2tensor_plugin.so" \
  --registers none --memory off --input /path/to/test-inputs.txt \
  --port 9010 -- "$C2T_BUILD/branch_digits"
```

Expose both endpoints through the operator's connection or tunnels, then use one
`--train` and one `--test` endpoint in the trainer command. Each worker reports
`samples=300` and exits after its run. No symbol extraction or input-label manifest
is passed to this model; its prediction targets come from subsequent trace blocks.

`model.pt` contains CPU weights, vocabulary size and run metrics.
`test-pairs.pt` preserves the bounded held-out sample for checkpoint verification.
`metrics.json` records observed/trained counts, stream completion and initial/final
test metrics; `loss.json` stores losses from optimization steps. `--seed` controls
model initialization and reservoir sampling. Guest behavior and the arrival order
of multiple workers are not made deterministic.

## Recorded evidence

On 2026-09-07, two independent `branch_digits` runs streamed concurrently from
`trail-arm`, Ubuntu AArch64 with the operator's QEMU 11.0.3 AArch64 Linux-user
build. Training and test used the seed-7 and seed-17 input corpora. Their 300-line
input sets had no lines in common. The learner was macOS 15.7.3 arm64 with
Python 3.10 and PyTorch 2.13.0. CPU and MPS used separate fresh capture runs.

| Result | CPU run | MPS run |
| --- | ---: | ---: |
| Completed worker streams | 2 | 2 |
| Observed training pairs | 123,167 | 123,172 |
| Observed test pairs | 123,105 | 123,105 |
| Optimizer updates | 100 | 100 |
| Pairs used for optimization | 102,400 | 102,400 |
| Retained held-out pairs | 16,384 | 16,384 |
| Initial held-out loss | 5.8133 | 5.8897 |
| Final held-out loss | 2.2419 | 2.2922 |
| Final held-out accuracy | 68.86% | 70.53% |

Both runs consumed their complete traces after the update budget, and both saved
checkpoints reproduced their held-out accuracy when restored on CPU. Saved MPS
loss agreed within $10^{-5}$. All four worker processes exited. These results show
the observation-to-learning path on different inputs of the same benign program.
They are not kernel results, unseen-program generalization, or a device-performance
comparison. The recorded CPU and MPS runs had different captured observations.

Seven tests pass: worker/source isolation, final partial batches and retained
storage, bounded reservoir selection, early cancellation, incomplete capture,
full draining after the update budget, checkpoint reload, and rejection of a late
capture failure. Some tests cover more than one invariant.

Local evidence is under `~/.cache/cpu2tensor/kernel-pretrain-check/`; worker logs
are under the same directory on `trail-arm`. Input SHA-256 values:

- Training: `fe6a630f7337bb38a52e92354e456fccffeb134f6365000d6dfeeace71002729`.
- Test: `2e935ab46e771cfa0b2bdb735203c09cb3638c286d482f05644ba41f1ea2b196`.

## Real kernel worker

Follow [guest setup](kernel-examples.md) to build the static target, initramfs,
and worker with matching QEMU headers. In each worker terminal:

```sh
export C2T_START_PC="$(nm "$C2T_BUILD/kernel_init" | awk '$3 == "cpu2tensor_capture_begin" {print "0x" $1}')"
"$C2T_BUILD/cpu2tensor-worker" \
  --qemu "$QEMU_SYSTEM" --plugin "$C2T_BUILD/libcpu2tensor_plugin.so" \
  --system on --registers none --memory off --start-pc "$C2T_START_PC" \
  --port 9000 --timeout-ms 120000 -- \
  -accel tcg,thread=multi -smp 2 -m 256M -nographic -monitor none \
  -nic none -no-reboot -kernel "$KERNEL" -initrd "$INITRAMFS" \
  -append 'console=ttyS0 rdinit=/init panic=-1 cpu2tensor.mode=observe cpu2tensor.seed=7'
```

Use separate ports on the same host, or the same port on distinct hosts. Supply
seed 11 for another training run and seed 17 for the held-out run. No interactive
adapter, action request, reward or policy is involved. `--start-pc` selects an
explicit postboot window; omit it to request capture from QEMU startup. Use
`--registers general --memory on --memory-values on` to collect rich signals for
a model that consumes them; the next-block model ignores those extra columns.

The same commands apply to operator-provisioned AWS runners: copy the chosen
guest artifacts and start one worker per target with the operator's existing
process manager and secure connection. The package handles neither provisioning
nor credentials. This release has no AWS execution evidence or multi-device DDP.

## Real kernel evidence

On 2026-09-07, three workers on `trail-x86` (Ubuntu 24.04.2 x86-64) ran upstream
QEMU 11.0.3 TCG with two vCPUs and Linux 6.9.0-dirty. Two complete seeded runs
(7, 11) trained the model; a separate seed-17 run supplied the held-out reservoir.
All used the explicit postboot marker and block-only capture. The learner was
Apple M2, macOS 15.7.3, PyTorch 2.13.0, MPS.

| Result | Value |
| --- | ---: |
| Completed kernel workers | 3 |
| Observed training pairs | 808,398 |
| Observed test pairs | 305,167 |
| Updates / optimized pairs | 100 / 102,400 |
| Retained test pairs | 16,384 |
| Initial / final held-out loss | 5.71269 / 4.67176 |
| Final held-out accuracy | 16.949% |

All streams were drained after the update budget. Each guest logged successful
completion. The saved weights reproduce held-out metrics on CPU. Artifacts are
under `~/.cache/cpu2tensor/kernel-integration/pretraining/` on the Mac; worker logs
are under `~/.cache/cpu2tensor/kernel-integration/` on `trail-x86`.
This is an integration/learning check, not a throughput result, a comparison
between host architectures, or evidence of generalization to a different kernel.
