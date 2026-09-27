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

## Result

The pilot failed on 2026-09-27. Collection itself was clean: all 3,060 physical
executions completed first-attempt in 2 minutes 36 seconds, with zero rejected,
lost, missing, or multiplexed sources. All 3,060 derived tensors and all 612
preregistered raw captures were copied off-host and hash-verified. CPU policy was
restored after collection.

| Frozen model seed | Alerts | Empirical rate | Concentration |
| ---: | ---: | ---: | --- |
| 2801 | 1/3,060 | $3.27\times10^{-4}$ | one `yield` row |
| 2802 | 0/3,060 | 0 | none |
| 2803 | 23/3,060 | $7.52\times10^{-3}$ | 22 `uname`, one `yield` |

The one seed-2801 alert is the same `yield` execution also flagged by seed 2803.
The 22 `uname` alerts from seed 2803 span all three new cohorts, so they are not
one transient collection failure. A post-hoc two-of-three vote would still flag
the shared `yield` row and was not the preregistered detector in any case.

Evidence custody:

- cohort-A manifest SHA-256:
  `f7b0145dd0b895e482e8e0793b8e83cd59a354e0fa58be9e7a83fd84f5dc7d01`;
- cohort-B manifest SHA-256:
  `02396a3a2726442fe874aba0247e0d848500b6e77db05325488bda61f6f7fbc4`;
- cohort-C manifest SHA-256:
  `09adc6c4d15aec533e77ff31cf3f0ac05499023ace495031ab74b83a05e1e454`;
- pilot report SHA-256:
  `176546fc014182ec1c95cebeb4cbaadb4d24a52070fc628f00ee4c649711bdde`;
- off-host captures:
  `/Users/theoad/.cache/cpu2tensor/effect-calibration-pilot-r1-92618e9`;
- frozen evaluation:
  `/Users/theoad/.cache/cpu2tensor/effect-calibration-pilot-result-r1`.

This invalidates the raw fused-plus-PMU rule as an operational detector and
blocks the 30,000-row calibration. The confirmed `read_efault` sensitivity still
shows that the pretrained representation contains useful signal, but a learned
and independently calibrated anomaly head must suppress stable family/intensity
nuisance. These pilot rows may be used for development only; any replacement
head needs another untouched physical session for prospective evaluation.
