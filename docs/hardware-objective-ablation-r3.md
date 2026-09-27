# Hardware objective ablation R3

Preregistered 2026-09-27. R2 established that masked reconstruction scales with
parameters, data, and optimizer compute, but its replicated cross-family probe
degraded as reconstruction improved. R3 tests whether a generic representation
objective retains reusable execution effects without consuming effect labels.

## Fixed comparison

All arms use d128, 2,040 balanced benign rows, 320 AdamW steps, batch size 32,
and pretraining seeds 3901, 3902, and 3903. Each step receives two independently
masked views. The three objectives are:

- the mean masked-reconstruction loss of both views;
- reconstruction plus symmetric instance-contrastive loss at temperature 0.2
  and weight 0.03;
- reconstruction plus VICReg invariance/variance/covariance loss with canonical
  component weights 25/25/1 and aggregate weight 0.01.

The effect corpus supplies no gradients or hyperparameter choices. Evaluation
uses the locked R2 protocol: three fixed two-shot heads train on lawful `fstat`
and `openat` pairs from session A, then rank unseen `read_efault` against
`read_copy` in session B.

An alternative objective advances only if its median transfer AUROC is at least
0.75, exceeds reconstruction by at least 0.10, wins at least two of three paired
pretraining seeds, and keeps median held-out masked loss within 10% of the
reconstruction baseline. These are development data and lawful effects, so even
a pass would qualify an objective—not a vulnerability detector.

## Result

The nine-model matrix completed on Mac MPS in approximately four minutes. No
objective passed the preregistered gate:

| Objective | Median held-out masked loss | Per-seed median `read` AUROC | Overall median AUROC | Paired wins |
| --- | ---: | --- | ---: | ---: |
| Reconstruction | 0.0787 | 0.000, 0.000, 0.000 | 0.000 | — |
| Contrastive | 0.0750 | 0.000, 0.180, 0.402 | 0.180 | 2/3 |
| VICReg | 0.0756 | 0.000, 0.004, 1.000 | 0.004 | 2/3 |

Both alternatives preserve—and slightly improve—the reconstruction metric.
Contrastive learning produces a modest but insufficient transfer improvement.
VICReg seed 3903 reaches AUROC 0.996--1.000 for all three locked heads, proving
that the architecture can encode a transferable direction, but seeds 3901 and
3902 remain near zero. Selecting the successful seed would violate the protocol;
VICReg is therefore not promoted.

The result sharpens the blocker from “no latent signal” to “latent geometry is
not reproducible across initializations.” The next diagnostic must use generic
benign-only geometry measures—cross-session family retrieval, intensity
retrieval, representation effective rank, and view agreement—to explain the
seed split without consulting the effect labels. Objective tuning may resume
only against those generic measures; the held-out effect gate remains sealed.

Artifact directory:
`/Users/theoad/.cache/cpu2tensor/hardware-objective-ablation-r3`. All nine
checkpoint hashes verify. Report SHA-256:
`b4b6517c2948b7c33e34227c5bc9d1df1feb0e60e87f15b38daa2252752969cb`.
