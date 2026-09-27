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
