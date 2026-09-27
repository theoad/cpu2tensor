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

The full 18-checkpoint matrix completed on Mac MPS in about three minutes and
passed the preregistered gate. Every compute trajectory improved:

| Width | Rows | Loss at 80 steps | Loss at 160 steps | Loss at 320 steps |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 255 | 0.1154 | 0.0955 | 0.0818 |
| 128 | 1,020 | 0.1131 | 0.0965 | 0.0809 |
| 128 | 2,040 | 0.1143 | 0.0935 | 0.0801 |
| 256 | 255 | 0.0971 | 0.0834 | 0.0727 |
| 256 | 1,020 | 0.0970 | 0.0836 | 0.0710 |
| 256 | 2,040 | 0.0984 | 0.0824 | 0.0688 |

The six log-loss slopes against optimizer steps range from −0.209 to −0.259.
At 320 steps, data-volume slopes are −0.0099 for d128 and −0.0257 for d256;
increasing the corpus from 255 to 2,040 rows improves loss by 2.09% and 5.47%,
respectively. R1's flat data curve was therefore a compute-budget artifact, not
evidence of data saturation. Capacity and data are complementary at this scale.

The transfer diagnostic remains negative. Median unseen-`read` AUROC is 0 for
15 of 18 checkpoints, 0.074 for d256/n1,020/80, and 0.023 for
d256/n2,040/80. Longer self-supervised training improves reconstruction while
driving this particular cross-family linear probe in the wrong direction. This
does not invalidate the grammar-learning result, but it rules out treating
masked reconstruction loss as a sufficient proxy for downstream effect
semantics.

Artifact directory:
`/Users/theoad/.cache/cpu2tensor/hardware-compute-scaling-r2`. All 18 checkpoint
hashes verify. Report SHA-256:
`fb405c45ae7efaaf3233ee45377852699cb1c42924b30b01b04472b64163512f`.
