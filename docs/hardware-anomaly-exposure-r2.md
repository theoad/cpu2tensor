# Hardware anomaly exposure R2: bounded benign conditioning ablation

This experiment tests whether the whole-family benign alert flood in
[model R1](hardware-anomaly-model-r1.md) came from measured workload exposure
rather than a new execution pattern. It uses the same sealed 102,000-execution
physical-host corpus and 789-feature cache. It does **not** train on, score,
or select against the corrected Dirty Pipe canary. It is a nuisance-control
ablation, not a detector acceptance test.

## Frozen design and split

The manifest SHA-256 is
`0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d`;
the previously hash-verified compact cache SHA-256 is
`5c689576cfd8d1576aa6f063e598a02cfee21d0fa629141e4c9913a8fda1fcdc`.
The exact unchanged split is 56,000 training, 21,000 calibration, 7,000
familiar validation, and 18,000 whole-family holdout. This run rechecked the
manifest and compact-cache hashes; it did not reopen or rehash all original
derived/raw files. Family and loop labels were used **only** for retrospective
report grouping, never as features or fitting targets.

Three variants were fixed before reading validation results:

1. `volume`: training-only ridge residualization of 512 PT byte-moment
   features against log PT bytes, and two PEBS log-sample-count moments against
   log PEBS samples;
2. `volume_time`: the same with log execution elapsed time in both regressions;
3. `volume_time_quadratic`: the same with fixed squares and interaction terms.

The ridge penalty is 0.001 per training row. All exposure and feature centers,
scales, polynomial centers, regression coefficients, and the 789-feature
diagonal-density normalizer are fit on training only and saved in each frozen
checkpoint. All other PEBS features and PMU features remain unchanged. The
score is the mean squared standardized residual, with the R1 clipping bound
of 20. No threshold, coefficient, or feature choice was adjusted after viewing
the benign holdout. PT bytes, PEBS sample count, and elapsed time are same-run
hardware/capture observables, not workload family or vulnerability labels.

## Matched benign ranking

Thresholds are independent calibration order statistics. At $10^{-3}$, 21
calibration executions exceed each threshold; at $10^{-4}$, only two do, so
the latter numbers are exploratory and cannot certify an operational FPR.
Counts below are alerts over 7,000 familiar and 18,000 whole-family benign
executions, on the same physical-host split as R1.

| Frozen score | Familiar $10^{-3}$ | Whole-family $10^{-3}$ | Familiar $10^{-4}$ | Whole-family $10^{-4}$ |
| --- | ---: | ---: | ---: | ---: |
| R2 volume | 7 | 11 | 0 | 4 |
| R2 volume + time | 3 | 12 | 0 | 1 |
| R2 volume + time + quadratic | 5 | 22 | 0 | 5 |
| R1 unconditioned diagonal | 9 | 2,068 | 0 | 2,068 |
| Archived d128 span-only | 4 | 40 | 0 | 1 |
| Archived marginal | 7 | 3 | 0 | 2 |

Exposure conditioning materially reduces the R1 diagonal's unseen-family
flood by 98.9--99.5% at $10^{-3}$. The simple volume and linear volume/time
variants are close to, but do not dominate, the archived controls: marginal
still has the fewest whole-family alerts, and the 7,000-row familiar view
cannot resolve small differences reliably. The quadratic variant is not an
improvement. This table is a benign-readout diagnosis, not grounds to select
one scorer, claim recall, or launch another collection.

The remaining $10^{-3}$ held-out alerts still cluster by intensity. `volume`
has 11/2,005 alerts in `dup` at 5,000 loops, and zero in all other held-out
family/intensity cells. `volume_time` has 12/2,005 in that same cell and zero
elsewhere. The quadratic variant has 20 in `dup`-5000 and one each in
`memfd`-5000 and `pipe`-5000. Thus exposure removes the mass `memfd` alert
flood but leaves a narrow high-intensity `dup` tail. No individual alert has
been adjudicated as a bug or false positive. There is no established effect
recall for any R2 score because the canary was deliberately not read.

## Named-host rate and reproducibility

The run used `mac.local`, Apple ARM64, Torch 2.13, four CPU threads, no GPU and
no paid resource. Exposure scalar extraction from the already-loaded manifest
took 0.045 s. Fitting plus applying each residualizer to all 102,000 cached
rows took 0.537 s (`volume`), 0.599 s (`volume_time`), and 0.617 s
(`volume_time_quadratic`). Dividing the 56,000 fitting rows by those inclusive
wall times gives conservative 104,258, 93,488, and 90,816 training rows/s.
Ten repeated scorer-only in-RAM passes over the prestandardized 102,000 rows
measured 2.36M, 2.30M, and 2.37M executions/s respectively. These scorer
rates exclude feature-cache creation, manifest/file reads, conditioning,
capture, and custody; they are not an online pipeline rate. R1's actual
hash-verified compact extraction was 1,244 executions/s on this Mac.

The R2 report is
`/Users/theoad/.cache/cpu2tensor/hardware-anomaly-exposure-r2/report.json`,
SHA-256
`a8604defcaf2ba3d0a068ca8ef56a6de6dc14cf935631122422cf2c122f7990e`.
It contains both thresholds, family-by-intensity counts at both budgets,
rates, and checkpoint hashes. The checkpoints are `volume.pt` SHA-256
`d6be69c649793249a7aad6c165d6f5162920348c38393460e91cbaf9f8d65777`,
`volume_time.pt` SHA-256
`63d241e035c2ea969332b4ba5001868150d622713560a62dad37c3c7733bdab9`,
and `volume_time_quadratic.pt` SHA-256
`d71e58b0ecff33191480b407f39e2fd10c7d2022c616671de10d5c47940bb8a2`.
Reloading each checkpoint reproduced every score bit-for-bit on CPU. The
source was Git `c730eb9` plus the new R2 script (SHA-256
`56a1952e539cabee10b4b9291a53763e3ca74c2f8d09fb6d08e7f2c522b5f6bc`)
and test (SHA-256
`fc40ea76d59d7afa24bb58078f1e6ca2314ea541e6cdabdf06fc9a8e742e548f`).
The run command was:

```bash
/Users/theoad/.cache/cpu2tensor/dev-python/bin/python \
  python/cpu2tensor/examples/hardware_anomaly_exposure_r2.py \
  /Users/theoad/.cache/cpu2tensor/hardware-pretraining-go/store-v3-scale100k-r2 \
  /Users/theoad/.cache/cpu2tensor/hardware-anomaly-model-r1/compact-features.pt \
  /Users/theoad/.cache/cpu2tensor/hardware-anomaly-exposure-r2
```

Three focused tests pass, including rejection of validation-target leakage
into fitted coefficients and replay of frozen quadratic design centering.
This result identifies exposure as a major nuisance in the compact PT byte
score, but it does not add PT packet grammar, fine PT↔PEBS timing, cross-boot
robustness, or sensitivity to a semantic kernel fault. No further separate
model-variant experiment is launched from this readout; subsequent model work
belongs to the coordinator's locked single-file autoresearch process.
