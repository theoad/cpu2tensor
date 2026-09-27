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

The [R1 result](hardware-fewshot-transfer-r1.md) passes this rule: the frozen
pretrained encoder reaches median cross-family AUROC 1.000 at 1--4 shots, while
scratch reaches 0.496 at one shot and 0 at two and four shots. Absolute tail
calibration still fails, so this promotes the scaling-law program rather than a
detector.

The [first capacity × data slice](hardware-scaling-pilot-r1.md) shows clean
capacity scaling from d64 to d256 with a local loss exponent near −0.16, while
data scaling is flat under a fixed 2,560-example optimizer budget. This makes a
data × optimizer-compute grid the next necessary measurement. Its one-seed
transfer result is unstable, so later matrices must aggregate several fixed
few-shot selections rather than treating one head as a scale signal.

The preregistered [R2 compute grid](hardware-compute-scaling-r2.md) crosses d128
and d256 with 255, 1,020, and 2,040 rows at 80, 160, and 320 optimizer steps.
Reconstruction loss remains primary; every point also receives three fixed
two-shot frozen-head probes solely as a representation-transfer diagnostic.

R2 passes its self-supervised scaling gate: all six compute trajectories improve,
and at 320 steps the largest data budget reduces held-out loss by 2.09% at d128
and 5.47% at d256. Its replicated transfer probe fails, however, with median
AUROC at or near zero. The next experiment should therefore vary a generic
representation objective on the same retained tensors and locked transfer
protocol—without tuning to the effect labels—before paying for more parameters
or collection. A successful objective must preserve reconstruction scaling and
recover stable transfer across fixed heads; otherwise the project should report
grammar compression and task transfer as separate capabilities.

The preregistered [R3 objective ablation](hardware-objective-ablation-r3.md)
holds scale and data fixed while comparing two-view reconstruction against
instance contrast and VICReg-style invariance. Three independent pretraining
seeds and three locked heads per model prevent another single-head conclusion.

R3 does not promote an objective. Contrastive pretraining raises median transfer
AUROC from 0 to 0.180, while VICReg has one striking 1.000 seed and two near-zero
seeds. Both preserve reconstruction quality, so the failure is reproducibility
of representation geometry rather than simple underfitting. The effect gate is
now sealed while a benign-only diagnostic measures cross-session family and
intensity retrieval, view agreement, and effective rank for all nine retained
encoders. Those generic metrics—not the successful effect seed—must determine
the next objective change.

The [R4 latent-geometry audit](hardware-latent-geometry-r4.md) therefore freezes
all nine R3 encoders and uses only benign sessions to measure family/intensity
retrieval, masked-view identity, centroid drift, and effective rank. It has no
promotion threshold and cannot select the exceptional VICReg seed.

R4 shows stable, nontrivial benign geometry. VICReg has median effective rank
14.95 versus 4.11 for reconstruction, raises cross-session family retrieval from
about 91.9% to 96.5%, intensity retrieval from 46.8% to 61.2%, and masked-view
identity from 54.1% to 63.5%. All three VICReg seeds are tightly grouped; its
single perfect effect-transfer seed is not generically exceptional. The correct
program boundary is now clearer: confirm the generic result on a fresh benign
session, then evaluate task post-training sample efficiency over scratch across
multiple effect-family splits. Zero-shot effect orientation is not a reliable
proxy for the quality of foundation pretraining.
