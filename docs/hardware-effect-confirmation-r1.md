# Hardware effect confirmation R1

Preregistered 2026-09-27 before collecting confirmation data. The prior
development session suggested a cross-modal rule that retained the lawful
read-`EFAULT` signal while rejecting its stable benign `memfd` false positive.
This experiment freezes that rule and evaluates it once on newly captured data.

## Frozen inputs and rule

- Development report SHA-256:
  `2d0efd99db73c03562b4a196c4054c8f03b5edfab473acbd3123a1ac437da04c`.
- Models: the three d128 checkpoints for seeds 2801, 2802, and 2803 declared by
  that report. No retraining or fine-tuning is permitted.
- Threshold source: the original 51 benign anchor-A executions used for model
  training. For each checkpoint, separately retain the maximum fused residual
  and maximum PMU residual. These are training/resubstitution maxima.
- Gate: flag only when **both** the fused residual and PMU residual strictly
  exceed their respective frozen maxima.
- Fresh effect cohort: 16 `read_efault` and 16 `read_copy` executions.
- Fresh benign cohort: the same 17 ordinary families as the anchor protocol,
  three intensities each, collected in a new session using the frozen R2 smoke
  plan (cohort seed `2026092706`, plan SHA-256
  `d76c443b7221a429c4dd63e41b862c56a554f822e7b392d135bf6827205bd679`).
- Acquisition parity: every benign row retains raw evidence, has an explicit
  input seed, and uses the anchor's $1$, $1/2$, and $1/5$ loop schedule. The
  effect cohort uses seed `2026092705`, 16 varied-input repetitions per family,
  full intensity, and retains every raw capture.

The confirmation passes only if every one of the three frozen models produces
0/51 benign alerts, 16/16 `read_efault` alerts, and 0/16 `read_copy` alerts.
No model, threshold, modality, family, or acceptance rule may change after the
new traces are inspected.

## Interpretation boundary

Passing confirms this one hardware-visible memory-fault proxy on a fresh session.
It does not establish vulnerability sensitivity or an operational false-positive
rate: 51 benign executions cannot measure $10^{-4}$. Failure invalidates this
gate without launching a compensating search on the confirmation data. Either
outcome leaves the million-execution and 24-hour campaigns **NO-GO**.

## Result

The frozen rule passed on 2026-09-27. Physical collection on `trail-x86` used
the exact boot and Ubuntu kernel already bound by the capture wrapper. All 83
executions completed on their first attempt in 11.2 seconds. Capture admitted
every row and reported zero lost, missing, or multiplexed sources; every raw and
derived file was retained and hash-verified after copying off-host.

| Frozen model seed | Fresh ordinary benign | Fresh `read_efault` | Fresh `read_copy` |
| ---: | ---: | ---: | ---: |
| 2801 | 0/51 alerts | 16/16 alerts | 0/16 alerts |
| 2802 | 0/51 alerts | 16/16 alerts | 0/16 alerts |
| 2803 | 0/51 alerts | 16/16 alerts | 0/16 alerts |

The confirmation therefore produced 48/48 effect detections and 0/201 combined
benign/control alerts across the three frozen models. This is stronger evidence
than the development result because the rule, thresholds, models, collection
plans, and acceptance criterion were committed before these traces existed.

Evidence custody:

- fresh benign manifest SHA-256:
  `1aff1544a09f83a050716a15ca479771bcee9cbbf4c6dec1a0cf8c0f5f97add3`;
- fresh effect manifest SHA-256:
  `3b27c871557adcb31dc7b77d5b1d03322f2382f0cd9181f4baf586d0bb82b6d0`;
- confirmation report SHA-256:
  `5142e6ec6c96c4861b16e0845c7d7bd6385cd96e02bbde0ac9eb8e78c26f07a8`;
- off-host captures:
  `/Users/theoad/.cache/cpu2tensor/effect-confirmation-r1-92618e9`;
- frozen evaluation:
  `/Users/theoad/.cache/cpu2tensor/effect-confirmation-result-r1`.

This promotes the conjunction from a post-hoc observation to a confirmed proxy
detector. It does not promote the system to operational fuzzing: the next gate
needs at least 30,000 independent benign executions to make zero observed alerts
compatible with a one-sided 95% upper bound near $10^{-4}$, followed by a frozen
ranking test on distinct held-out effects. The 24-hour and million-execution
campaigns remain **NO-GO** until those gates pass.
