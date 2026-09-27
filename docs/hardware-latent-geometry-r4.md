# Hardware latent geometry R4

Preregistered 2026-09-27. R3 left one VICReg seed with perfect lawful-effect
transfer and two with none. R4 does not retrain models or read effect data. It
audits all nine frozen encoders using only the three independent 1,020-row benign
cohorts.

For each encoder, cohort A supplies nearest-centroid references and cohorts B
and C measure family (17 classes), intensity (3 classes), and family/intensity
stratum (51 classes) retrieval. Additional metrics are representation effective
rank, same-family centroid cosine across sessions, and same-execution retrieval
between two deterministic masked views of 255 balanced cohort-C rows.

These metrics diagnose collapse, nuisance memorization, and session stability.
There is deliberately no promotion threshold: the held-out effect labels remain
sealed, and R4 cannot select an objective or model seed.

## Result

Pending.
