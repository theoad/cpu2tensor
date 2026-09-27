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
  `11ea870dfd7303e73c0dcc0d5ffc414bf69d602316a725810d257a4b678b3f2d`).
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
