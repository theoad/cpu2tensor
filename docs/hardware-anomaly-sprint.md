# Six-hour hardware-anomaly sprint

Started 2026-09-26 00:20 UTC; review deadline 06:20 UTC. The subject is the
same physical `trail-x86` laptop and exact release-kernel boot for collection,
pretraining, calibration, and prospective validation. The 102,000-execution
PT/PEBS/PMU corpus is immutable. No cloud collector or debug kernel is in scope.

## Decision to test

Dirty Pipe remains a **secondary logic-flaw challenge**, not the sole scale
gate. A write through a lawful-looking kernel path may have little distinctive
raw PT/PEBS/PMU signature even when the resulting page-cache state is wrong.
The primary gate needs a matched, safely reproducible known corruption or
pathological-execution effect, with nontriggering and benign high-load siblings
on the same kernel. A crash, spin, or latency alone must not count as learned
bug sensitivity: compare against lawful events with similar duration and load.
The detector may use generic hardware traces and must not receive a CVE-specific
feature, trigger label, or hand-coded oracle. Ground truth is used only after
scores are frozen.

## What the model can actually observe

The current v3 model gets 16 PT *byte histograms*, compressed PEBS time-bin
moments and 32-bin site/address sketches, plus whole-interval PMU summaries.
It does not see PT packet adjacency or decoded control flow, a timestamp for
each PT segment, a sequence of precise-store sites, or a trustworthy total
order across CPUs. The compact model baseline discards even the 16-segment
order. These facts limit what either model could infer about a causal write.

The next ablation should preserve raw PT order and PEBS site/address/time
relationships while passing nuisance context separately: exact kernel-build
identity and runtime kernel-text base, perf event identity/period and running-time
coverage, CPU/lane identity, frequency policy, session, and action/intensity
metadata. Kernel instruction addresses can be represented both raw and relative
to that exact boot's relocated base; keep raw values and provenance so an
unusual address is not normalized away. Memory addresses need similarly
explicit region/provenance plus page offset, not just a global absolute value.
The retained sideband has runtime `_text` but no link-time `vmlinux` or
`System.map` anchor, so this corpus supports same-boot text-relative offsets,
not an empirically known numeric KASLR slide or cross-boot invariance. The
saved PT configuration also has TSC/MTC/CYC timing packets disabled; PT-to-PEBS
fine timing cannot be reconstructed from byte positions. Qualifying a future
timing-enabled capture requires a separate overhead and loss comparison.
Record each execution's actual input hash/seed and action parameters in the
future manifest: the current corpus records one global split seed and repeat
indices, which are not evidence of per-execution seed invariance. These
conditioners are for this exact executable/build/hardware, not a cross-build
foundation model or universal symbol vocabulary.
No model may invent a precise PT-to-PEBS timestamp or cross-CPU order where
the capture supplies only intervals. Train conditional masked/cross-modal
predictions so it can learn relationships between local flow, sampled memory
behavior, and counter response rather than merely classify which benchmark
ran. Test residual stability across *benign* input seeds/intensities,
then on a new session/boot if safe. A synthetic relocation test checks code
behavior but does not establish real KASLR generalization from this one-boot
corpus. Full PT decode remains a sampled audit/LLM-evidence tool unless its
production scoring cost is separately measured and justified.

## Prospective review-budget gate

At one million executions, one review per ten thousand permits at most **100**
LLM bundles. Therefore the operational quantity is `recall@top100` (and
`TPR@FPR<=1e-4` on benign runs), not aggregate AUROC. Reserve headroom for
candidate clusters and uncertain captures: target benign FPR at most `5e-5`
(at most 50 benign reviews per million), while reporting exact alert volume at
the full `1e-4` cap. A coarse AUROC of 0.99 is useful only as a sanity check;
no AUROC value by itself guarantees useful recall in the extreme tail. For
example, an ROC curve that detects no effect until benign FPR reaches `1e-4`
and then detects all effects can have AUROC about `0.9999` yet miss every effect
under the operational cap. Conversely, a useful low-FPR detector need not have
near-perfect AUROC if ranking degrades later in the curve.

Calibrate the threshold without exploit labels on separate session blocks;
evaluate on untouched sessions and benchmark families. Report effect recall,
family-macro recall, false positives per million, top-100 composition,
capture validity/loss, confidence bounds, and deduplicated review workload.
If one checkpoint passes a pilot, repeat the frozen procedure across at least
three model initialization seeds and independently randomized collection
sessions before declaring it reliable; no best-seed selection on a bug canary.
Aim for at least 50 independent, safe effect/control pairs after a small
feasibility pilot, then 100 or more across sessions for a strong recall bound
if the trigger remains safe; report the exact interval rather than calling a
12-pair canary reliable. No dangerous repeat is justified merely to meet a
sample-size target.
With zero observed false alerts, a one-sided 95% binomial upper bound is about
`3/N`; bounding FPR below `5e-5` thus needs at least 60,000 effectively
independent benign evaluations. Correlated executions make this optimistic:
report block-level estimates as well. A 100,000-row tail calibration contains
only about ten observations at `1e-4`, so a million-row independent validation
is desirable before a high-confidence full-scale claim.

## Parallel tracks and coordination

| Track | Owned output | Question |
| --- | --- | --- |
| Validation | `hardware-anomaly-validation-r1.md` | Which same-subject effect/control families establish sensitivity without confounding, and how stable is recall under the review cap? |
| Capture | `hardware-anomaly-capture-r1.md` | Which generic PT/PEBS/PMU timing and address information was lost in the current raw representation, and what can be retained without slowing collection? |
| Model | `hardware-anomaly-model-r1.md` | Do lightweight density and denoising baselines improve tail ranking while GPU training and frozen inference stay ahead of collection? |
| Pretraining objective | `hardware-anomaly-pretraining-r1.md` | Can a benign-only conditional objective suppress family novelty and preserve local deviations without exploiting known-bug labels? |
| Candidate search | `hardware-anomaly-candidate-r1.md` | Is there a safely bounded, actually vulnerable defect on the exact signed laptop kernel that could qualify as the primary gate? |
| Nuisance metadata follow-up | `hardware-anomaly-nuisance-r1.md` | Which exact-session relocation and timing facts are present in retained raw/sideband, before proposing an embedding the data cannot support? |
| Coordinator | this document and `backlog.md` | Resolve shared contracts, resource contention, independent review, and final GO/NO-GO. |

Only the capture track may run perf on `trail-x86` during the sprint. Other
tracks work from immutable off-host artifacts. Paid GPU use needs a bounded
spend, a hard stop, and off-instance checkpoint custody. Record every
experiment's host, kernel/boot, model/data hashes, split, runtime, and cost.

## Exit decision

### Initial ablation readout (first 30 minutes)

The immutable 102k corpus has so far rejected the tested compact alternatives:
at an exploratory `1e-3` calibration tail, the denoising MLP alerts on
10,379/18,000 held-out benign executions, low-rank density on 9,809, and a
PT+PMU-to-PEBS conditional residual on 126. The latter is worse than its
unconditional PEBS control (7/18,000). The old d128 span-only and marginal
baselines produce fewer held-out alerts but still lack sufficient tail
calibration and any established real-bug sensitivity. Model-only Mac MPS
training/scoring outruns this laptop's 13.07-execution/s finite collector;
quality and observation, not accelerator speed, are the immediate blockers.

The 525 preregistered raw audits show stable within-family PT byte adjacency,
but a frozen Dirty Pipe descriptor readout found no compelling effect-specific
separation. The PT capture has no TSC/MTC/CYC timing packets and only one
sampled CPU lane. A distinct safe real-bug primary gate is still not admitted
on the exact signed kernel. These are early results, not the six-hour verdict.

The first single-file autoresearch smoke is likewise **report-only**, not a
model-selection result. On the Mac MPS GPU, two fixed-seed 438,101-parameter
denoising models trained in 36.13 s total and scored an in-memory 8,192-row
batch at roughly 339k--396k rows/s; those numbers are Mac model throughput,
not physical-host capture speed. Independent CPU evaluation at an exploratory
`1e-4` calibration tail flagged 10,463/18,000 and 10,100/18,000 lawful
held-out-family executions. In both seeds, all 6,000 held-out `memfd` rows
were flagged; most `dup` rows were flagged as load increased. The sealed
evaluation report hashes are
`e13863f51e2db5d66d0dd650f21397b8172fab27226cb5f8d3a51c6fb6c9bffc`
(`1e-4`) and
`bb4383fa75d7fb2bcd476008074d94c14f281ba5ee24ffbdee4b85fc6d37a59e`
(`1e-3`). This is lawful-family novelty flooding, not demonstrated bug
sensitivity. The evaluator refuses keep/discard, and no five-minute automatic
model loop or 24-hour campaign has begun.

A second report-only trial added training-only PT-byte/PEBS-count exposure
correction using the same verified corpus and two seeds. It also failed the
lawful-family-transfer diagnostic: at `1e-3`, it flagged 12,042/18,000 and
12,036/18,000 held-out rows versus 12,015 and 12,018 for the first compact
autoencoder. At exploratory `1e-4`, it flagged 11,500 and 12,000 held-out
rows. The simpler R2 exposure-corrected diagonal score had only 11--12
held-out alerts at `1e-3`; adding this autoencoder did not preserve that
specificity. No effect recall or independent-session false-positive bound is
available, so there is no justified model promotion or recurring five-minute
search on this compact one-session dataset.

R2 follow-ups now test two measured gaps in parallel: benign-only exposure
conditioning on the full derived 102k corpus, and lightweight raw order plus
PEBS sequence grammar on the 525 verified audit shards. The latter must first
produce a safety-gated two-session capture plan; it is not authorization to
run a new physical collection or to infer timing absent from old PT packets.

**GO** for a one-million-execution prospective campaign only if at least one
predeclared effect family has repeat-stable detection at the review cap,
matched benign siblings stay below the measured tail budget, artifact custody
and LLM bundles are complete, and the physical collection/training/scoring
pipeline can keep up safely. This is a pilot GO, not a zero-day claim.
Otherwise **NO-GO**, with the measured failure mode and next discriminating
experiment. The prior 102k scale comparison and corrected Dirty Pipe canary
were NO-GO: d128 AUROC 0.5208, d512 AUROC 0.4722; greater capacity did not
create sensitivity. Do not launch a 24-hour campaign merely because training
loss improved.
