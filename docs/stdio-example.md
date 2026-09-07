# Learn a stdin action from a trace

The small `stdio_digits` target chooses a random digit, calls its `digit_0` through
`digit_9` cue function, and waits for a two-byte digit/newline response on stdin.
It exits with code 0 for a correct response or 1 for an incorrect response. This
is a legal input task: the target validates its input and has no memory corruption
or control-flow exploitation behavior.

The client observes execution, reduces it to ten function-entry counts, and sends
an ordinary response. Every decision uses a fresh target run. The target has one
vCPU; the worker confirms that QEMU has stopped before forwarding the input request.
The pause occurs before a real stdin `read` syscall, independent of printed prompts.

## Start an interactive worker

Build the worker and fixture with the operator-provided QEMU and matching header,
following [the worker quickstart](quickstart.md). Use a fixed port and enough runs
for training and evaluation:

```sh
"$C2T_BUILD/cpu2tensor-worker" \
  --qemu "$QEMU" --plugin "$C2T_BUILD/libcpu2tensor_plugin.so" \
  --stdio on --episodes 200 --port 9000 \
  --registers none --memory off -- "$C2T_BUILD/stdio_digits"
```

Keep the worker terminal visible. As with observation-only collection, the operator
provides a private connection or tunnel, and installs all host dependencies. The
package does not provision machines or QEMU. Increase both the worker timeout and
client timeout if a policy needs longer than the default timeout to choose input.
The worker's episode count is a maximum; terminate it after an early-ending client
if it is still waiting for another connection.

Record the ten `digit_N` ELF symbol addresses in a JSON object with decimal integer
values, using the same `nm` symbol extraction as the [trace-learning example](learn-trace.md).
Extract these symbols from `stdio_digits`, since another binary has different addresses.
The fixtures are linked without PIE for this small example. General applications
may use ASLR; a client must account for load addresses when matching symbols.

## Train and evaluate

On the learner, with the repository installed:

```sh
python -m cpu2tensor.examples.learn_stdio tcp://127.0.0.1:9000 \
  --symbols /path/to/stdio-symbols.json --device mps \
  --output /path/to/stdio-imitation.json
```

The default method collects one real trace of each of the ten cues, fits a linear
classifier, and checks that its cross-entropy loss falls. It then restarts the
target for 40 fresh random cues, chooses model actions, and measures rewards from
the target's exit codes. Evaluation must reach at least 90% success; random guessing
has expected success 10%. This is a tiny supervised sanity check with online action
evaluation. It validates trace-to-feature-to-model-to-action wiring; the feature
definition already makes the cue identity easy to recognize.

The same command supports a contextual bandit trained only from sampled actions
and their rewards. Start a worker with at least 440 episodes, then run:

```sh
python -m cpu2tensor.examples.learn_stdio tcp://127.0.0.1:9000 \
  --symbols /path/to/stdio-symbols.json --device mps \
  --method reinforce --episodes 400 --evaluate 40 \
  --output /path/to/stdio-reinforce.json
```

REINFORCE uses a categorical policy, a moving reward baseline, and a small entropy
term. It receives no cue labels during training. Training can fail the evaluation
threshold; that is reported as a failed run, not hidden. The output records the
method, learner device, number of target runs, evaluation cue counts, and success
rate. `--seed` seeds PyTorch only. The target uses operating-system randomness and
has no seed adapter, so runs are not exactly reproducible.

## A small streaming API

`StdioEnv` keeps the same bounded batch delivery as `Pool`. It returns iterators
instead of saving all observations up to a decision:

```python
from cpu2tensor import StdioEnv

with StdioEnv("tcp://127.0.0.1:9000", device="mps") as env:
    for batch in env.reset():
        model.observe(batch)
    while env.needs_input:
        action = model.choose_bytes()  # For this target, e.g. b"3\n".
        for batch in env.step(action):
            model.observe(batch)
    success = env.exit_code == 0
```

Fully consume a reset or step iterator before sending an action. Bytes are sent
unchanged; include a newline when the target expects one. `max_action_bytes` gives
the current request limit. Empty actions and actions larger than the limit are
rejected before transmission. Reset cancels the old target and starts another run.
Leaving the context or closing the environment cancels a paused run as well.

A normal nonzero exit is available as `exit_code` for client-defined reward and
success rules. A capture error, missing completion, or connection failure raises
an exception. It must not silently become a negative task reward.

## Optional Gymnasium conventions

Install the optional `cpu2tensor[gym]` dependency to use `env.as_gym`. The client
provides an `observe(batches)` reducer, an `encode(action)` function, `reward(env)`,
and Gymnasium observation/action spaces. The reducer must fully consume the batch
iterator and return its bounded observation. Target exit terminates an episode by
default; `episode_end(env)` can return the client's `(terminated, truncated)` pair.

The wrapper returns `(observation, info)` from reset and
`(observation, reward, terminated, truncated, info)` from step, following the
[Gymnasium API](https://gymnasium.farama.org/api/env/). There is no target seed or
reset-options adapter yet, so nonempty reset options and an explicit reset seed
are rejected rather than implying that a remote target was seeded. Rendering,
vector environments, recurrent state ownership, and automatic replay buffers are
outside this initial wrapper.

The initial adapter covers a single emulated user-process vCPU and scalar stdin
`read` calls. It does not define multi-thread world-pause semantics, `readv`, buffered
input readiness, kernel interactions, or delayed execution actions. The
observation-only path continues to use `Pool` without action protocol work.

Here is a complete reducer and wrapper for this target. It keeps ten counts and
returns a NumPy observation. The terminal observation may contain all zeros:

```python
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from cpu2tensor import StdioEnv

symbols = json.loads(Path("stdio-symbols.json").read_text())
entries = torch.tensor([symbols[f"digit_{digit}"] for digit in range(10)])

def observe(batches):
    counts = torch.zeros(10, dtype=torch.float32)
    for batch in batches:
        counts += (batch.addresses[:, None] == entries[None, :]).sum(0)
    return counts.numpy()

def reward(stream):
    if stream.exit_code not in (None, 0, 1):
        raise RuntimeError("The digit target reported an unexpected failure")
    return float(stream.exit_code == 0)

with StdioEnv("tcp://127.0.0.1:9000") as stream:
    env = stream.as_gym(
        observe=observe,
        encode=lambda action: f"{action}\n".encode("ascii"),
        reward=reward,
        observation_space=gym.spaces.Box(0, np.inf, shape=(10,), dtype=np.float32),
        action_space=gym.spaces.Discrete(10),
    )
    observation, info = env.reset()
    action = int(observation.argmax())  # Replace the known cue rule with a policy.
    observation, reward_value, terminated, truncated, info = env.step(action)
```

The NumPy wrapper uses CPU observations because ordinary Gymnasium spaces describe
NumPy arrays. The training command above consumes Torch observations directly on
its selected device.

## Recorded real runs

On 2026-09-07, the REINFORCE command completed 400 training episodes and 40 new
evaluation episodes through the AArch64 Linux worker at `trail-arm`, consuming
exactly 440 target runs. The learner was an Apple M2 Mac running Darwin arm64,
PyTorch 2.13.0, and MPS. The target used block-only capture and the ten symbols
from its exact non-PIE ELF. No target stdout or input labels entered policy training.

Mean training reward was 0.655. The first 50 sampled actions averaged 0.20 reward;
the last 50 averaged 1.00. Greedy evaluation succeeded in all 40 runs, with every
cue represented: `[3, 4, 3, 3, 4, 3, 5, 2, 8, 5]` appearances for digits 0 through 9.
The PyTorch seed was 19; the target was not seeded. No failed real run or parameter
retry preceded this result. This is correctness and learning evidence for this
small fixed task, not a throughput measurement or a cross-program benchmark.

The default imitation command passed on the same host pair and MPS learner. It
needed 25 random target runs to collect all ten cue classes. Cross-entropy fell
from 2.302585 to 0.016129. It then succeeded in all 40 new action evaluations;
evaluation cue counts were `[3, 7, 3, 1, 7, 3, 3, 2, 3, 8]`. This used 65 runs.

The concrete Gymnasium example above also passed against a real target, using
Gymnasium 1.3.0 and a CPU NumPy reducer. Its initial observation contained one cue;
the legal digit action received reward 1, `terminated=True`, `truncated=False`,
and exit code 0. Its terminal observation contained ten zeros. Both observations
were accepted by the declared space. This separate API smoke check used one run
and the known cue rule, not a learned policy.

Operator-local results are `~/.cache/cpu2tensor/ml-demo/stdio-reinforce.json`,
`stdio-imitation.json`, and `stdio-gym.json` in that directory. The commands above
let another contributor record an independent run without relying on that machine's
artifacts. The learner seeds do not make any of these target runs reproducible.

## Supported target behavior

This initial pause mechanism uses host SIGSTOP/SIGCONT. Installing a SIGCONT handler,
closing or replacing stdin, and creating another vCPU are rejected explicitly.
The target must consume the delivered bytes through scalar stdin reads. If a
failed/short read leaves action bytes pending at the next request, the worker
reports unsupported input progress instead of blocking on a full pipe. Action
reception and nonblocking delivery share a deadline. No stdout capture or prompt
parser is part of this adapter.

A cancelled stdin run is a normal worker episode outcome: the old child is killed
and reaped before the next connection starts a target. Known capture errors still
stop the worker and raise errors. Observation-only cancellation keeps its existing
incomplete-stream behavior. The repeated worker holds one listening socket across
runs, so immediate reset does not race a socket rebind.
