# Hardware effect calibration pilot R1

Preregistered 2026-09-27 before pilot collection. The confirmed fused-plus-PMU
detector is evaluated unchanged on 3,060 new ordinary-benign executions. This is
a short engineering and false-positive pilot before the 30,000-row calibration
gate; it cannot itself establish an operational $10^{-4}$ false-positive rate.

## Frozen experiment

- Models, thresholds, and gate are exactly those in the confirmation report with
  SHA-256 `5142e6ec6c96c4861b16e0845c7d7bd6385cd96e02bbde0ac9eb8e78c26f07a8`.
- Three independent seeded cohorts contain 1,020 rows each: 17 benign families,
  three intensities, and 20 repetitions per family/intensity.
- Cohort seeds are `2026092710`, `2026092711`, and `2026092712`.
- Plan SHA-256 values are, respectively,
  `03f674f56bb86fa8d6f3e7d92916a3db2a3eb0e25759bd4dc865bdb62e4757fd`,
  `1016866829c322a349a394c96082d070e6b651e4de62464b314a89214ed28ba5`,
  and `1cec0d7ee65edbd5e80870255a00e2a714b5c7dd22e8115651d8e5c4e3750264`.
- Each plan retains 204 raw captures selected before execution and retains all
  1,020 derived tensors. Retries, loss, missing sources, and multiplexing fail
  the cohort.
- The pilot passes only if each of the three frozen models produces zero alerts
  among all 3,060 rows. No threshold, model, modality, plan, or acceptance rule
  may change after collection.

With zero alerts, the one-sided 95% binomial upper bound is about $9.79\times
10^{-4}$ per model. That only justifies proceeding to the 30,000-row gate. Any
alert fails the pilot and is preserved for diagnosis; confirmation data will not
be used to repair or retune the frozen detector.
