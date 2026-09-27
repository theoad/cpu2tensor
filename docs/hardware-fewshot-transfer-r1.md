# Hardware few-shot transfer R1

Preregistered and run 2026-09-27. This experiment asks whether benign-only
self-supervised pretraining creates a representation that learns a new lawful
hardware-effect task more efficiently than the same architecture initialized at
random. It is a representation probe, not the proposed deployed detector.

## Design

Three pretrained d128 checkpoints are paired with random-encoder controls.
Every scratch, frozen-pretrained, and fine-tuned-pretrained arm receives the
same head initialization, normalization, labeled rows, 80 optimizer updates,
and evaluation data. Few-shot counts are 1, 2, 4, 8, and 16 examples per class
per training pair.

Training labels come only from lawful `fstat` and `openat` error/control pairs in
effect session A. The cross-family test is `read_efault` versus `read_copy` in
independent effect session B; no `read` row is used for head training. Benign
pilot cohort A calibrates a maximum-score threshold, while cohorts B and C are
the benign evaluation set.

The preregistered success rule requires a pretrained arm at 1, 2, or 4 shots to
beat scratch by at least 0.10 median held-out `read` AUROC, win at least two of
three paired seeds, and not increase median benign alerts.

## Result

The representation-transfer criterion passed for both pretrained arms at every
low-shot count.

| Shots per class/pair | Scratch median `read` AUROC | Frozen pretrained | Fine-tuned pretrained |
| ---: | ---: | ---: | ---: |
| 1 | 0.4961 | 1.0000 | 0.9961 |
| 2 | 0.0000 | 1.0000 | 1.0000 |
| 4 | 0.0000 | 1.0000 | 1.0000 |
| 8 | 0.0000 | 1.0000 | 1.0000 |
| 16 | 0.0000 | 1.0000 | 1.0000 |

At one shot, the frozen arm's median advantage is 0.5039; at two and four shots
it is 1.0000. Frozen and fine-tuned arms win all three paired seeds at each low
shot count. Median benign-alert deltas versus scratch are non-positive, satisfying
the full preregistered rule. Same-family `fstat` and `openat` AUROC is 1.000 for
all arms, so the differentiator is cross-family transfer rather than memorizing
the supervised task.

This is evidence that the small pretrained model contains a generic lawful
error-path geometry that a few labels can orient, whereas the random encoder
fits the labeled families but usually reverses on unseen `read`. It is the first
clear sample-efficiency advantage over scratch in this hardware-trace program.

## Boundary

Perfect ranking did not yet produce an operational alert: the maximum score on
benign calibration cohort A exceeded all 16 `read_efault` head scores. Thus the
pretrained representation transfers, but absolute tail calibration remains
unsolved. Benign evaluation alerts range from zero to several among 2,040 rows,
also far above evidence for a $10^{-4}$ tail. These are already-inspected
development datasets and lawful effects, not vulnerability evidence.

The 45 training arms consumed 58.8 aggregate training seconds on the Mac MPS;
individual arms took 0.77--1.90 seconds. The artifact directory is
`/Users/theoad/.cache/cpu2tensor/hardware-fewshot-transfer-r1`. Report SHA-256 is
`bf60a3761137c001346dd5287fb37396128957463e22c46802276be7bc827123`;
it records every checkpoint and source hash.

The result justifies measuring neural scaling laws rather than promoting an
anomaly head. Next, vary model capacity, benign pretraining data volume, and
workload diversity independently, measuring both held-out self-supervised loss
and this paired few-shot transfer curve. A new physical session remains required
for any prospective claim.
