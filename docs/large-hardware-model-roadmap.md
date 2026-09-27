# Large Hardware Model research program

The product hypothesis is not a supervised anomaly classifier. It is a
foundation model that learns a reusable grammar of hardware execution from raw
PT, PEBS, PMU, timing, topology, and workload context. Vulnerability detection
is one downstream task.

## Four stages

1. **Self-supervised scaling laws.** Measure held-out reconstruction,
   cross-modal prediction, next-window prediction, and representation transfer
   while independently varying parameter count, training tokens/executions,
   workload diversity, modality diversity, and compute. Report loss against
   each axis and compute-optimal frontiers; do not select scale using a known
   vulnerability.
2. **Large Hardware Model pretraining.** Train the largest justified model on
   diverse release hardware/software workloads. Inputs retain modality identity,
   timing, CPU/topology, address-layout context, and missingness. Evaluation
   includes unseen workloads, sessions, boots, intensity, and lawful effects.
3. **Task post-training.** Compare prompting/probing, few-shot heads, encoder
   fine-tuning, preference/ranking objectives, and reinforcement learning on
   safe effect proxies and later vulnerability evidence. Every method is paired
   with an identical from-scratch baseline so pretraining must demonstrate
   sample or compute efficiency.
4. **Frozen fuzzing harness.** Freeze a qualified model and use it to rank
   perturbations, score every execution, and package suspicious raw windows for
   deep LLM review. Harness feedback cannot silently update the deployed model.

## Scaling-law matrix

The first matrix should cross at least four model sizes, four data sizes, and
three workload-diversity levels under matched training compute. Primary curves
are validation loss versus parameters, executions, and FLOPs. Secondary curves
measure zero-shot lawful-effect ranking, linear-probe/few-shot sample efficiency,
cross-session false-positive tails, and inference executions per second. A larger
model is justified only when loss and transfer improve predictably rather than
on one hand-picked effect.

## Immediate transfer probe

The next short experiment is diagnostic, not a detector promotion. A task head
trains on few labeled `fstat` and `openat` lawful error/control examples from
session A, then evaluates the unseen `read_efault` family in session B. Identical
label selections, head initialization, update counts, and normalization are used
for three arms:

- random encoder plus trained head;
- frozen pretrained encoder plus trained head;
- pretrained encoder and head fine-tuned together.

Few-shot counts are 1, 2, 4, 8, and 16 examples per class per training pair,
with three paired seeds. Pilot cohort A sets a maximum-benign threshold; cohorts
B and C measure false positives. Pretraining demonstrates meaningful few-shot
value only if a pretrained arm beats scratch by at least 0.10 median held-out
`read` AUROC at one of 1, 2, or 4 shots, wins in at least two of three paired
seeds, and does not increase the median benign alert rate. Otherwise the result
is evidence that the current pretraining objective or scale has not produced
useful task transfer.
