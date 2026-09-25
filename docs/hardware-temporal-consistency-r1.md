# Temporal consistency R1

This isolated experiment asks whether a self-supervised temporal-ranking head
can repair the first multimodal model's insensitivity to PEBS timing. The code
is preserved by tag `hardware-temporal-consistency-r1`; it is not part of the
accepted pretraining path and changes neither the capture nor input schema.

## Intervention

Each negative starts from a retained raw shard. PEBS timestamps move between
the 16 relative time bins inside the same lane and capture envelope. Every PEBS
column is reordered consistently, then the complete capture passes through the
unchanged featurizer. PT remains raw and undecoded. Samples never move between
lanes and no cross-CPU total order is manufactured.

A shared scalar head observes contextual PEBS tokens. Training combines masked
reconstruction with a pairwise soft-margin objective that puts clean energy
below one- and four-bin shifts. Execution energy is the symmetric mean of lane
energies. The existing anomaly scorer is unchanged; the experiment uses a
separate score with a calibrated nonnegative temporal component.

## Retained-corpus result

The 204-execution corpus supplied 84 training, 42 calibration, 42 familiar
validation, and 36 whole-family-held-out executions. Training used shifts one
and four; unseen tests used shifts two and eight plus a swap of bins 0--3 with
8--11. Three 300-step seeds ran on `mac.local` using Torch 2.14 and one CPU
thread.

| Partition / intervention | Original AUROC | Temporal AUROC | Temporal mean ratio | Combined AUROC |
| --- | ---: | ---: | ---: | ---: |
| Familiar / shift 2 | 0.716 | 0.819 | 2.25 | 0.829 |
| Familiar / shift 8 | 0.646 | 0.747 | 1.90 | 0.739 |
| Familiar / block swap | 0.626 | 0.717 | 1.81 | 0.717 |
| Held-out / shift 2 | 0.433 | 0.742 | 2.04 | 0.668 |
| Held-out / shift 8 | 0.526 | 0.704 | 2.21 | 0.705 |
| Held-out / block swap | 0.527 | 0.678 | 2.23 | 0.704 |

The held-out result was uneven: `dup` reached 1.0 shift-two AUROC in every seed,
`memfd` ranged from 0.604 to 0.660, and `pipe` ranged from 0.417 to 0.611.
Block-swap top-$k$ localization overlap was 0.521--0.542 against a random 0.5.
One extra unmasked encoder pass reduced scoring from 1,012--1,060 to 893--935
executions/s, a 12.7--16.1% overhead. The 42-row pilot calibration was unstable:
clean held-out alerts were 0, 194, or 333 per thousand across seeds.

Retained report hashes are:

- seed 17: `9e8cd6a0aa04de1fdcd036b8c21d417fc6811ea79252dd65211eb674d34edcf0`;
- seed 23: `21f48e8f7923dbc9976b25a6c2bdf999b6346db2ab709dac01a61a8953018a0c`;
- seed 31: `bd281b21493048331ddb71562f3081dd7604287f63eb8c188be63ff971141e16`.

## Decision

R1 proves that a coherent raw-timestamp intervention and explicit within-lane
objective add held-out timing sensitivity. It does not pass integration:
family behavior and calibration are unstable, localization is random, and
overhead exceeds 10%. Collect phase-rich benign workloads and reuse an existing
encoder forward before retesting; do not scale this head for 24 hours.
