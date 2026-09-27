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

The frozen audit completed on Mac MPS in 17 seconds. Median results across the
three pretraining seeds are:

| Objective | Effective rank | Family B/C | Intensity B/C | Stratum B/C | Masked-view identity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reconstruction | 4.11 | 91.6% / 91.9% | 46.8% / 46.8% | 99.0% / 99.2% | 54.1% |
| Contrastive | 10.68 | 95.6% / 95.3% | 60.3% / 58.8% | 96.9% / 97.2% | 55.7% |
| VICReg | 14.95 | 96.6% / 96.5% | 61.4% / 61.2% | 97.3% / 97.0% | 63.5% |

Chance rates are 5.88% family, 33.3% intensity, 1.96% stratum, and 0.39%
masked-view identity. All objectives learn stable cross-session structure.
Reconstruction retains the exact family/intensity stratum best but concentrates
the representation in about four effective dimensions. Contrastive and VICReg
trade roughly two points of stratum accuracy for substantially higher rank,
family/intensity generalization, and masked-view identity.

Most importantly, the three VICReg seeds are tightly grouped on every generic
metric. Seed 3903—the sole R3 seed with perfect effect transfer—is not a benign-
geometry outlier. Its success therefore cannot be predicted or justified by a
generic metric selected here. R3's split is an unstable semantic relationship
between lawful effect families, not representation collapse or gross session
memorization.

This supports separating foundation pretraining from task post-training. VICReg
is the strongest *generic latent-space candidate* in this development audit,
but R4 does not promote it or a particular seed. The next confirmation should
freeze the metric and repeat it prospectively on a fresh benign session; task
post-training must then demonstrate sample-efficiency over scratch across
several effect-family splits.

Artifact directory:
`/Users/theoad/.cache/cpu2tensor/hardware-latent-geometry-r4`. Report SHA-256:
`b47cbd5dca927ec1f11745599972c15b63cbeba0aa15a11a155f64bd4a8a29d2`.
