# Hardware scaling pilot R1

Preregistered 2026-09-27. This is the first local scaling-law slice for the
Large Hardware Model program. It measures whether held-out self-supervised loss
and cross-family transfer respond coherently to model width and benign data
volume before spending compute on d512, diversity, or long-training axes.

## Fixed matrix

- Model widths: 64, 128, and 256, with feed-forward width twice the model width,
  two local layers, and two cross-CPU layers.
- Benign pretraining rows: 255, 1,020, and 2,040, deterministically balanced over
  17 families and three intensities from pilot cohorts A and B.
- Compute: one seed (`3901`), 80 AdamW steps, batch size 32, and identical mask
  policy and optimizer hyperparameters for all nine points.
- Validation: all 1,020 rows of untouched pilot cohort C, with deterministic
  masks. Cohort C is a development holdout already seen by earlier diagnostics,
  so this run cannot make a prospective claim.
- Transfer: train only a frozen linear head from two labeled examples per class
  for `fstat` and `openat` in effect session A; report AUROC on the unseen
  `read_efault`/`read_copy` pair in effect session B.
- Efficiency: parameter count, parameter-example product, training time, and
  frozen encoder throughput on validation rows.

This pilot reports local curves rather than fitting a universal power law. A
coherent result requires held-out loss to improve with more data at at least two
of three widths and to improve from d64 to d256 at the 2,040-row budget. Transfer
is secondary and must not be used to select a known-effect-specific model.

## Result

The nine-point matrix completed on the Mac MPS in under one minute. Capacity
scaling is coherent at every data budget:

| Width | Parameters | Loss at 255 rows | Loss at 1,020 rows | Loss at 2,040 rows |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 189,206 | 0.1459 | 0.1475 | 0.1452 |
| 128 | 640,150 | 0.1154 | 0.1131 | 0.1143 |
| 256 | 2,328,470 | 0.0971 | 0.0970 | 0.0984 |

The local log-loss slopes against parameter count are −0.162, −0.167, and
−0.155 at 255, 1,020, and 2,040 rows. Encoder throughput remains high but falls
as expected with width: approximately 4,058 rows/s for d64, 3,787 rows/s for
d128, and 3,131 rows/s for d256 at the largest data budget.

Data-volume slopes are effectively flat and mixed (−0.0006, −0.0061, and
+0.0053 for d64/d128/d256). This run fixed training at 80 × 32 = 2,560 sampled
examples, so the 2,040-row corpus receives far fewer exposures per row than the
255-row corpus. The result identifies a compute-limited regime; it does not show
that additional data lacks value. The next slice must cross data size with
optimizer compute.

The single-seed two-shot transfer probe is not coherent: d64 ranges from AUROC
0.879 to 0.523 as data grows, while d128 and d256 produce 0. This conflicts with
the earlier three-seed d128 result and shows that one labeled-row/head seed is
too noisy for a scale metric. Future scaling points will aggregate several fixed
few-shot selections and keep reconstruction loss primary.

Artifact directory:
`/Users/theoad/.cache/cpu2tensor/hardware-scaling-pilot-r1`. Report SHA-256:
`d001ffe18e4e87f68a6c73af07ca3372c827c693bba53cbd243d21260e33d8ab`.
