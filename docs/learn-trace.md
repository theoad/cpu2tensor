# Learn from execution traces

This is a ten-class learning exercise with the scale and shape of an MNIST
quickstart. It uses program execution instead of handwritten images. A normal C
target reads sixteen digits per line, calls a different function for each digit,
and marks the end of each sample with another ordinary function call. The learner
predicts the most frequent executed digit from observed function-entry counts.

The example is deliberately easy enough to diagnose. Its purpose is to prove
sample boundaries, capture attribution, tensor feature preparation, gradients,
held-out evaluation and saved model weights together. It is not a foundation-model
benchmark or a claim that a neural network is needed to compute a histogram.

## Prepare the target and inputs

Follow the [worker build and learner installation](quickstart.md), using a QEMU
binary and matching plugin header supplied by the operator. Build the
`branch_digits` target with the same native CMake build. Its source lives at
`native/examples/branch_digits.c`; shared digit functions live in
`native/examples/digit_paths.c`.

On the learner:

```sh
export C2T_DATA="$HOME/.cache/cpu2tensor/ml-demo"
python -m cpu2tensor.examples.learn_trace inputs \
  --output "$C2T_DATA/inputs.json"
```

This writes 300 unique inputs, 30 in each class, and the adjacent `inputs.txt`
file for target stdin. All lines have sixteen digits. The most frequent digit
occurs six to ten times, with the same sampling rule for every class. Input order
and digit positions are shuffled using a reproducible seed. The label is stored
in the learner's manifest; the target does not print it.

Move `inputs.txt` to the worker using an operator-provided shared folder or normal
file transfer. Run the target through the worker, setting `C2T_INPUT` to that file:

```sh
"$C2T_BUILD/cpu2tensor-worker" \
  --qemu "$QEMU" --plugin "$C2T_BUILD/libcpu2tensor_plugin.so" \
  --registers none --memory off --input "$C2T_INPUT" \
  --port 9000 -- "$C2T_BUILD/branch_digits"
```

The observation-only path needs no environment, action or policy. The target runs
the whole corpus in one process, then reports only the number of processed samples.
The usual operator tunnel can expose this endpoint to a Mac learner.

## Collect and train

Export the target's `digit_0` through `digit_9` and `sample_end` symbol addresses
into `symbols.json`. Addresses must come from the exact worker executable. The
file is a JSON object from symbol name to integer address. The target is non-PIE,
so these ELF addresses match the captured addresses without ASLR relocation.
On the worker, save the symbol listing:

```sh
nm -n "$C2T_BUILD/branch_digits" > /path/to/shared/branch_digits.nm
```

Move that file to the learner using the same operator-provided transfer or share:

```sh
python -m cpu2tensor.examples.learn_trace symbols \
  --input /path/to/shared/branch_digits.nm --output "$C2T_DATA/symbols.json"
python -m cpu2tensor.examples.learn_trace collect tcp://127.0.0.1:9000 \
  --manifest "$C2T_DATA/inputs.json" --symbols "$C2T_DATA/symbols.json" \
  --output "$C2T_DATA/dataset.pt" --host "my ARM Linux worker; AArch64 guest"
python -m cpu2tensor.examples.learn_trace train "$C2T_DATA/dataset.pt" \
  --device mps --output "$C2T_DATA/mps"
```

Use `--device cpu` for a CPU learner. CUDA uses the same training command with
`--device cuda` when the operator provides a working CUDA environment. The example
does not provision workers, open SSH connections, install QEMU, or download data.

Collection filters block addresses to the ten function entries, counts calls
between sample markers, and checks that each sample contains exactly sixteen
calls. Each count vector is independently checked against the target input before
the dataset can be saved. This oracle detects wrong addresses, missing calls,
incorrect markers and attribution errors before a model can hide them.

The essential training loop is ordinary PyTorch:

```python
model = torch.nn.Linear(10, 10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for _ in range(120):
    optimizer.zero_grad(set_to_none=True)
    logits = model(training_counts / 16)
    loss = torch.nn.functional.cross_entropy(logits, training_labels)
    loss.backward()
    optimizer.step()
```

The [implementation](../python/cpu2tensor/examples/learn_trace.py) has input
generation, a vectorized streaming reducer and the complete training command.
Python does no per-instruction processing. Completed feature rows are stored on
CPU, then transferred once for this small full-batch training exercise. This makes
the example easy to rerun on a different learner; it does not measure live training
throughput or overlap capture with device computation.

## What the checks establish

- The model sees ten observed execution counts. It never receives stdin bytes,
  stdout, raw symbol addresses, sample indices or the manifest as input.
- A stratified split holds out 20% of each class before fitting. Duplicate target
  inputs and duplicate count vectors are rejected, so a test histogram cannot
  already appear in training. The only feature scaling is the known fixed input length;
  no statistics are fitted on test data.
- The 110-parameter classifier must achieve at least 90% held-out accuracy and
  reduce training loss to less than one fifth of its first value.
- A training-majority baseline should score 10% on the balanced test set. A second
  model trained on shuffled labels and the trained model evaluated on shuffled
  test traces must both score below 30%. These are diagnostics for this fixed
  example, not statistical significance tests.
- Gradient finiteness is checked during training. `model.pt` stores CPU weights,
  feature order, scaling, train/test indices and metrics. `metrics.json` and
  `loss.json` preserve evaluation and the learning curve.

The deterministic rule `argmax(observed_counts)` is also an exact oracle for this
task. Training demonstrates that real captured observations support learning and
that the label association matters; it does not show generalization to new
programs, relocated builds or a different representation of execution.

Unit tests exercise fragmented sample boundaries, extra runtime blocks, missing
markers, independent count validation, duplicate rejection and a complete learning
run on mathematical fixtures. Real QEMU collection must be checked separately;
synthetic test success alone is not instrumentation evidence.

## Recorded run

On 2026-09-07, the corpus ran through the real `cpu2tensor-worker` and QEMU plugin
on `trail-arm`, Ubuntu AArch64, using the operator's QEMU 11.0.3 AArch64 Linux-user
build. Capture enabled block entries and disabled register and memory observation.
The target reported `samples=300`. The learner was macOS 15.7.3 arm64, with Python
3.10 and PyTorch 2.13.0. Both CPU and Apple MPS training completed on the same
captured dataset; no CUDA execution is claimed.

Every captured vector exactly matched the independently computed input histogram.
All 300 input lines and all 300 captured count vectors were distinct. The fixed
split used 240 training and 60 test examples, with six test examples in every class.

| Check | CPU | MPS |
| --- | ---: | ---: |
| Training accuracy | 100% | 100% |
| Held-out accuracy | 100% | 100% |
| First training loss | 2.298911 | 2.298911 |
| Final training loss | 0.093722 | 0.093722 |
| Majority baseline | 10% | 10% |
| Shuffled training labels | 0% | 0% |
| Shuffled test traces | 13.33% | 13.33% |

Both saved checkpoints were reloaded on CPU and reproduced their held-out
accuracy. These are correctness results for this constructed task; no throughput
or relative device-performance measurement was made.

The local run artifacts are under `~/.cache/cpu2tensor/ml-demo/`: `dataset.pt`,
`inputs.json`, `inputs.txt`, `symbols.json`, and the `cpu/` and `mps/` model/metric
directories. The dataset SHA-256 is
`51d889a10fa3cddb5da0f0b2ee2ed8c659cdf6932d34897fec87825426d20c09`.
The six learning tests also pass, covering symbol extraction, fragmented
boundaries, incorrect counts, incomplete samples, distinct splits and a
saved/reloaded classifier.

The current runtime still uses callback pipe publication. This example does not
establish that the proposed per-vCPU ring optimization has been implemented.
