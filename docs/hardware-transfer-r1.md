# Hardware transfer R1

Accepted 2026-09-26: test whether offline pretraining helps a small learned
triage head before spending on further pretraining or a large fuzzing campaign.
The first slice uses retained evidence only, with short cycles on the physical
`trail-x86` i7-10510U. Collection, CPU policy, and the running kernel are unchanged.

## Frozen first comparison

The three arms share the archived d128 transformer architecture, pretraining-only
normalization, and a linear head on the three modality-pooled contextual token
embeddings. Scratch starts with random encoder weights. Frozen starts with the
archived encoder and trains only the head. Fine-tuned starts with the archived
encoder and trains the head and encoder. Each arm uses the same head seed,
paired-row schedule, optimizer steps, and training examples. Trainable parameter
counts and timing are reported; a frozen encoder intentionally costs less.

Training uses twelve matched Dirty Pipe pairs from `canary-102k-d128` plus 256
benign examples selected by a fixed balanced rule from the 102k training partition.
Testing uses twelve matched pairs from the separately captured `canary-102k-d512`
session. No raw execution is shared across sessions. These are the same known
input recipe and previously inspected data, so this is a single-defect development
replay. It cannot establish new-defect or new-input generalization.

The model receives only the original 16-segment PT/PEBS/PMU observations,
availability, timing bounds, and timing quality. Arm labels, semantic outputs,
pair IDs, previous scores, and record paths never become model features. Input
and oracle metadata verify the importer before tensor-only caches are written.
The training CLI opens evaluation only after all three checkpoints are saved.

The loss combines paired logistic ranking with weak positive/negative anchoring
and benign negative examples. Scratch and fine-tuned arms also retain masked
reconstruction on background benign examples with weight 0.05. Head learning rate
is 0.001; encoder rate is 0.0001. The initial cycle has 40 optimizer steps,
eight pairs and eight background examples per step, seed 1729, and two Torch CPU
threads. A second predetermined seed 1730 checks stability with the same protocol.
Each cycle has a 270-second internal budget and a 300-second external timeout.
No hyperparameter is selected from the replay scores.

The comparator is the already frozen `volume_time` exposure-conditioned scorer.
Its checkpoint was fit on benign data before either canary was used here.
All scores are archived, with exact token contributions from the linear head
for the held-out effect examples. The contribution sum plus bias reconstructs
the head score; it is an attribution, not evidence of causal localization.

Calibration uses a fixed balanced subset of 1,024 benign calibration executions.
Familiar and whole-family validation each contain 512 separate executions.
Report AUROC, paired wins, and effect/control/benign alerts at a development 1%
tail and a conservative threshold at the calibration maximum. This sample cannot
certify the operational $10^{-4}$ FPR, and no result authorizes a large campaign.
Checkpoints, scores, input hashes, source hashes, timing, and failure artifacts
remain in a fresh host-local directory and are copied off-host after completion.

## Next discriminating step

If pretraining helps this development replay without flooding benign traces,
freeze its design and evaluate new input variants and a different effect family
in later independent sessions. If scratch performs equally well, pretraining has
not earned its compute here. If all arms fail, test the observations before
increasing capacity. A supervised detector is permitted at deployment without
an oracle, but its training is explicitly semi-supervised; an unsupervised
generalization claim requires separate evidence.

The initial two 40-step cycles showed low transfer AUROC despite large training
loss reductions. The next diagnostic fixes 500 steps for both seeds, with no other
loss/architecture change, and reports training separation alongside held-out
session separation. This is a development follow-up to check undertraining versus
overfitting; it is not another independent final test or a model promotion.

Review found the initial shared RNG was consumed differently by reconstruction
masking in scratch/fine-tuned arms and the cached frozen arm. Those early results
are exploratory and do not qualify as a matched-schedule comparison. The corrected
runner uses independent row/mask RNGs, verifies identical row-schedule hashes in
all arms, pins archived checkpoint/report hashes, requires an externally supplied
plan hash, and verifies nonoverlapping capture-time ranges on the same boot.

The initial 500-step two-thread diagnostic was manually stopped when the physical
package reached 97 C with turbo enabled. It has partial checkpoints and no result
report. No CPU policy was modified. Corrected runs use one thread, require a
starting package temperature at most 70 C, stop at 80 C, and wait for cooling from
72 to 68 C within their fixed wall-time budget. Tensor import also uses one thread.

## Measured outcome

The corrected seed-1729 40-step job trained all three arms on physical `trail-x86`
(`iseeyou`, i7-10510U, Linux x86-64). It used one Torch/BLAS thread, CPU-2 affinity,
and a transient user service with CPUQuota=15% and RuntimeMaxSec=300s. This reduced
average compute load without changing the target's frequency policy. It still
hit the 80 C cutoff during scoring; service exit status was 1. The job lasted
48.16 seconds including throttling, consumed 7.105 CPU seconds, peaked at
455,232 KiB RSS, and used no swap. These are offline tensor-training measurements,
not capture or operational inference throughput. The service is inactive and no
training remains running.

Earlier unthrottled startup attempts were refused above 70 C or stopped at
95–97 C before creating this output tree. The two-thread 500-step diagnostic was
also stopped at 97 C. We did not repeat the long diagnostic or start the corrected
second seed after the quota-limited run tripped its guard. Further x86 training
requires safe cooling; no inference benchmark or new capture has been launched.

All three completed checkpoints and the tensor caches were copied to the Mac and
their SHA-256 values matched the physical host. A score-only path, restricted to
the Mac coordinator, ran without an optimizer or checkpoint modification. It
checks checkpoint arm/seed/config/thread metadata, input/checkpoint hashes, and
refuses partial score flags or output beneath preserved input trees. Scoring was
on `mac.local` CPU with one Torch thread, 2.44 seconds for this retained-tensor
evaluation. This is not a cross-host speed comparison.

| Arm | Training AUROC | Second-session AUROC | Second-session paired wins |
| --- | --- | --- | --- |
| Scratch | 1.000 | 0.424 | 5/12 |
| Frozen pretrained encoder | 0.715 | 0.417 | 4/12 |
| Fine-tuned pretrained encoder | 1.000 | 0.438 | 5/12 |
| Frozen exposure baseline | Not applicable | 0.549 | 6/12 |

At the calibration-maximum threshold, all learned arms flag zero effect and zero
neutral examples. Frozen also flags 1/512 held-out-family benign examples; the
other arms flag zero. The exposure baseline flags all 12 effects **and** all 12
neutral examples: family novelty, not effect separation. At the development 1%
tail, scratch flags 3 effects/4 controls; frozen and fine-tuned flag neither arm.

The interrupted runner did not persist its in-memory training statistics or row
schedule hashes. It reached scoring only after its equal-schedule gate, but the
recovered report deliberately marks `qualified_paired_comparison=false` and
`matched_schedule_archived=false`. New runs now preserve `training.json` after
each completed arm. This result is useful as a preliminary train-versus-session
diagnostic, not a qualified comparison or proof that pretraining has no value.

Interpretation: the trainable models can fit these labels; the learned distinction
does not survive a second session of the same recipe. Session memorization,
unstable observation, insufficient examples, and missing invariant-bearing signal
remain competing explanations. More steps/parameters alone have not earned their
cost. First improve stable observations and independently varied matched examples;
then rerun a fully archived comparison and evaluate a different effect family.
The detector and million-execution campaign remain **NO-GO**.

## Qualified completion and observation diagnosis

The preliminary section above records the thermal/debugging path. It is
superseded for the matched comparison by two complete one-thread physical-host
runs. Seed 1729 ran in a transient user service with a 5% CPU quota and required
240.36 seconds of service wall time (214.32 seconds inside the report), 12.02 CPU
seconds, 459,920 KiB peak RSS, and no swap. All three resulting checkpoint hashes
are byte-identical to the earlier completed training checkpoint files, while this
run additionally preserves every training statistic and the common row-schedule
hash. Seed 1730 ran with CPU 2 capped at 1.8 GHz and turbo disabled under an
independent restore timer. It required 27.12 seconds of service wall time (24.46
seconds inside the report), 27.09 CPU seconds, 460,144 KiB peak RSS, and no swap.
The host policy was restored to `no_turbo=0` and a 4.9 GHz maximum immediately
afterward. These timings use different duty/frequency controls and are not a
performance comparison.

| Seed | Arm | Training result | Second-session AUROC | Paired wins | Alerts at calibration maximum (effect/control) |
| --- | --- | --- | ---: | ---: | ---: |
| 1729 | Scratch | loss 1.292 $\rightarrow$ 0.059 | 0.424 | 5/12 | 0/0 |
| 1729 | Frozen pretrained | loss 1.202 $\rightarrow$ 1.028 | 0.417 | 4/12 | 0/0 |
| 1729 | Fine-tuned pretrained | loss 1.207 $\rightarrow$ 0.011 | 0.438 | 5/12 | 0/0 |
| 1730 | Scratch | loss 1.544 $\rightarrow$ 0.033 | 0.715 | 8/12 | 2/0 |
| 1730 | Frozen pretrained | loss 1.237 $\rightarrow$ 1.038 | 0.424 | 4/12 | 0/0 |
| 1730 | Fine-tuned pretrained | loss 1.241 $\rightarrow$ 0.090 | 0.465 | 5/12 | 0/0 |

The seed-1730 scratch result is not a promotion: it is one reused 12-pair
development replay, unstable across predetermined seeds, and its conservative
threshold catches only 2/12 effects. Most importantly, neither pretrained arm
beats scratch. Offline pretraining has not demonstrated reusable bug signal.

A separate 1.20-second x86 diagnostic used 256 benign training rows only to fit
per-coordinate scale. For each modality, the first canary session then defined a
single mean effect-minus-control direction; the second session was projected
without refitting. This is deliberately a linear representation diagnostic, not
a claim that every nonlinear classifier must fail. Coordinates constant in the
256 benign scaling rows are omitted, so a rare site confined to the canary is
also outside this diagnostic:

| Modality | Train AUROC | Second-session AUROC | Effect-direction cosine across sessions | Train effect norm / session-shift norm |
| --- | ---: | ---: | ---: | ---: |
| Raw PT v3 tensor | 1.000 | 0.326 | -0.071 | 1.182 |
| PEBS v3 tensor | 1.000 | 0.354 | -0.047 | 1.369 |
| PMU | 0.618 | 0.361 | -0.987 | 0.106 |
| Timing | 0.569 | 0.493 | 0.176 | 0.716 |
| Fused | 1.000 | 0.354 | -0.049 | 1.320 |

The raw, fixed-boot grammar audit verified all 24 raw hashes per session against
the content-hashed canary report, then used all 132 within-arm and 144 cross-arm
pairs per session. Those pairwise comparisons reuse 24 executions and are
descriptive similarities, not independent trials. PT ordinal-bigram cosine was
0.9986 within an arm and 0.9985 across
arms in the first session (0.9987 for both in the second): stationary, but not a
Dirty Pipe separator. PEBS time/site/address cosine was 0.2123 within an arm and
0.2117 across arms (0.2133 and 0.2140 in the second): sampling noise overwhelms
this effect. All 52,119 combined PEBS records had exact IP and nonzero address,
and more than 99.8% mapped to core text, so missing/censored samples do not
explain the result.

The v3 feature code also chooses `min(ip)` and `min(address)` independently per
execution. Each 24-row session had 16 distinct IP minima and 24 distinct address
minima. Because those minima determine IP hash bins and offset moments, one rare
lower sample moves otherwise unchanged features. This is a generic coordinate
bug, now tracked as cpu2tensor issue
[#28](https://github.com/theoad/cpu2tensor/issues/28). The existing 102k derived
corpus cannot be repaired because only 525 raw audit traces were retained. The
next experiment is therefore a small two-session seeded collection under a
schema using exact-boot `_text` anchoring and relocation-safe address summaries,
followed by a distinct safe hardware-visible effect gate. Dirty Pipe stays a
secondary logic-flaw challenge. Scale remains **NO-GO**.

## Artifact custody

Physical artifacts remain under
`/home/user/.cache/cpu2tensor/hardware-transfer-20260926`; the verified off-host copy
is `/Users/theoad/.cache/cpu2tensor/hardware-transfer-20260926`. Neither the original
102k corpus nor its retained canaries was modified or deleted.

- Plan: `data-reviewed/plan.json`, SHA-256
  `8ff3c4dbbb244ef8ea6364a3fb2538c74717d38828a6cc0477a429f76136e4b0`.
- Training cache: `8529903cbb9f48fd94849d4274b040c90deeea6824694d2b1dc78812e87c28bb`.
- Evaluation cache: `476c5328c669b2fc1140a8a7ce81d035ded496b9d64eeae6473281499dd93ff1`.
- Corrected host checkpoints: `reviewed-seed1729-step40/`:
  scratch `76ce57a1827c1497f438d26e7d6b337d4ba3b33123b67e7e12e595d9b01fc957`,
  frozen `70ea42223a66e72e8e4bff495765593b037903daea7491698c04688535928947`,
  fine-tuned `719fd5d1bcc9c177cc30cffeffcd9814109f307951a2e1979b59866cc3636333`.
- Host failure pins executed source
  `8821039f3c5931a5a88ed653081e093506fef95160bde9bb12ea702f0411bbed`.
- Recovered score report: `preserved-evaluation-seed1729-step40-v3/report.json`,
  SHA-256 `46b6363c1283ca581c74d6085535262d86cfcf33f5223221ea1866898f7b1e37`.
- Recovered scores: `6cdc551e0b05bb034ee902ca1c86285074c96f616b54741f7567f12f454aba75`.
- Qualified seed-1729 report: `qualified-seed1729-step40/report.json`,
  `e7387c2d234582e86ee92e1aff672717b1cb559517e27d5873c09755dbbf4e41`;
  training ledger `5133ffe20156200d4f31c45c2f25de6d1ea3acc9e0127f78bf5d90a05676316b`.
- Qualified seed-1730 report: `capped-seed1730-step40/report.json`,
  `8efcbc18f7807f3df80ce66ff9b18bfb97a32592a2c47b82bc8ba379d2c93890`;
  training ledger `cbf8230c71fcaa1befedc7348b8b1619b73b856bf96d0d9d59bd3e5b21310e27`.
- Cross-session modality diagnostic: `stability-r2-v2.json`,
  `af949ba064016cdf4f662732470bfdab2f07e2e75f41d80031e6c5fc3864f166`.
- Corrected, raw-hash-verified grammar reports: `raw-grammar-d128-v5.json`,
  `2b9f5db93520661878d954c95247055f79263f58584ea6371a5abd7ddad70c78`,
  and `raw-grammar-d512-v5.json`,
  `89741ed7560660a13671d4da2b8427b3ab9bb0b4f5d22ed74983786a33d22798`.

The old `cycle-seed*-step40` outputs have unequal sampling schedules and are
excluded from the table. `cycle-seed1729-step500` contains partial evidence only.
The off-host v1/v2 score reports are superseded by v3, not independent trials.

## Development checks

On the Mac coordinator, 33 focused tests pass across `test_hardware_transfer.py`,
`test_hardware_multimodal.py`, and `test_hardware_multimodal_experiment.py`.
They cover unchanged reconstruction, frozen weights, identical sampling schedules,
faithful token readout, label-independent calibration, fail-closed thermal guards,
and score-only CLI/output isolation. `git diff --check` is clean. No native capture
code changed; this is not a claim of new physical capture or full-backend coverage.
