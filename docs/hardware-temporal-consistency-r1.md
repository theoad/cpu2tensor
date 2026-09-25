# Temporal consistency R1

This isolated experiment asks whether a self-supervised temporal-ranking head can
repair the first multimodal model's insensitivity to PEBS timestamp alignment.
It is not part of the accepted pretraining path and does not change the frozen
capture or input tensor schemas.

## Intervention and objective

Each negative starts from a retained raw custody shard. PEBS timestamps move by
a permutation of the sixteen relative time bins inside that lane's existing
capture envelope. Every PEBS column is sorted by the resulting timestamps and
the complete capture passes through the unchanged production featurizer. PT
bytes remain raw and undecoded. The intervention does not move a sample between
lanes or manufacture an order between CPUs.

A shared scalar head observes the contextual PEBS tokens of each lane. Training
adds a pairwise soft-margin term requiring the clean execution energy to be lower
than the energy of one- and four-bin shifts. The existing masked reconstruction
loss remains present. Execution energy averages lane energies symmetrically, so
CPU-lane permutations leave it unchanged. Existing
`multimodal_anomaly_score` behavior is unchanged; the experiment exposes a
separate score which adds the calibrated nonnegative temporal component.

## Retained-corpus result

The input was the frozen 204-execution corpus with identity SHA-256
`70bb8a60486285788773843d06c04b308ef1fe6abdc65367dd28900cb6323f74`:
84 training, 42 calibration, 42 familiar validation, and 36 whole-family-held-out
executions. Training used shifts one and four. Shifts two and eight plus a swap
of bins 0--3 with 8--11 were unseen tests. Three 300-step seeds ran on
`mac.local` (Darwin arm64, Torch 2.14.0, one CPU thread).

Median results across seeds 17, 23, and 31 were:

| Partition / intervention | Original-score AUROC | Temporal AUROC | Temporal mean ratio | Combined AUROC |
| --- | ---: | ---: | ---: | ---: |
| Familiar / shift 2 | 0.716 | 0.819 | 2.25 | 0.829 |
| Familiar / shift 8 | 0.646 | 0.747 | 1.90 | 0.739 |
| Familiar / block swap | 0.626 | 0.717 | 1.81 | 0.717 |
| Held-out / shift 2 | 0.433 | 0.742 | 2.04 | 0.668 |
| Held-out / shift 8 | 0.526 | 0.704 | 2.21 | 0.705 |
| Held-out / block swap | 0.527 | 0.678 | 2.23 | 0.704 |

The held-out shift-two result was not uniform. `dup` was perfectly separated in
all seeds, while `memfd` ranged from 0.604 to 0.660 AUROC and `pipe` from 0.417
to 0.611. The block-swap localization overlap was 0.521--0.542 on held-out rows
against a random expectation of 0.5, so the head did not localize the affected
bins.

One extra unmasked encoder pass reduced measured scoring throughput from
1,012--1,060 to 893--935 executions/s, a 12.7--16.1% overhead. The pilot-only
42-row calibration was also unstable: the combined clean held-out alert rate was
0, 194, or 333 per thousand depending on seed. These rates are descriptive only;
the calibration set is too small for an operational false-alert claim.

The retained reports and SHA-256 hashes are:

- seed 17: `9e8cd6a0aa04de1fdcd036b8c21d417fc6811ea79252dd65211eb674d34edcf0`
- seed 23: `21f48e8f7923dbc9976b25a6c2bdf999b6346db2ab709dac01a61a8953018a0c`
- seed 31: `bd281b21493048331ddb71562f3081dd7604287f63eb8c188be63ff971141e16`

They live beside the retained corpus as
`temporal-consistency-report-seed<seed>.json`.

## Decision

R1 proves that coherent raw-timestamp interventions and a per-lane objective add
real held-out sensitivity beyond the original reconstruction score. It does not
pass an integration gate: family behavior and calibration are unstable,
localization is random, and inference overhead exceeds 10%. Keep the branch
experimental. The next bounded test should use phase-rich benign workloads and
reuse one reconstruction encoder pass for temporal energy before considering a
larger model or 24-hour run.
