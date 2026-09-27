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

Pending.
