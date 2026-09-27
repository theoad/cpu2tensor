# Hardware compute scaling R2

Preregistered 2026-09-27. R1 showed a stable capacity exponent near −0.16,
but its fixed 2,560-example optimizer budget could not expose a data-volume
benefit. R2 crosses data volume with optimizer compute before increasing model
size, workload diversity, or collection volume.

## Fixed matrix

- Model widths: 128 and 256, preserving the R1 architecture.
- Benign pretraining rows: 255, 1,020, and 2,040, deterministically balanced
  over 17 families and three intensities from pilot cohorts A and B.
- Optimizer steps: 80, 160, and 320 at batch size 32. Every one of the 18
  checkpoints trains from the same initialization seed (`3901`); the 80-step
  points are rerun rather than copied from R1.
- Validation: all 1,020 rows from untouched pilot cohort C with deterministic
  masks. It is a development holdout, so this experiment cannot support a
  prospective claim.
- Transfer diagnostic: freeze each encoder, train a two-shot linear head on
  lawful `fstat` and `openat` effects from session A, and evaluate unseen
  `read_efault` versus `read_copy` in session B. Three fixed labeled-row/head
  seeds (`2801`, `2802`, `2803`) replace R1's unstable single probe. Transfer
  cannot select or rescue a checkpoint.

The primary compute gate passes if held-out masked loss at 320 steps is below
loss at 80 steps for at least five of six width/data trajectories. Data scaling
emerges only if, at 320 steps, both widths have a negative log-loss slope with
data and at least one improves by 2% or more from 255 to 2,040 rows. R2 passes
only when both conditions hold.

## Result

Pending.
