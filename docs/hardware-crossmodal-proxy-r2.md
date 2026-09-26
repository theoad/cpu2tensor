# Cross-modal discordance proxy R2: review-only design

Status: **proposed development diagnostic, not run or trained** (2026-09-26).
This protocol deliberately makes an internally discordant *synthetic feature
record* from two lawful executions. It can test whether a frozen scorer uses
relationships among hardware modalities. It cannot establish sensitivity to a
real kernel defect, corruption, a pathological effect, or its affected interval.
It is separate from the immutable checkpoint evaluator pending review.

## Available evidence and representation limit

The sealed v2 train-only cache SHA-256 is
`f949c7dd6df006a5cf84e2da59950e3ddad8fd0a31f9e78573814bd798b04292`;
the development-eval cache SHA-256 is
`9b875bab80db03afcc6a612bba6216c0e42af2fa6c8063cc51ae63ade4b40d26`.
Its 46,000 evaluation rows retain 789 compact features, family, loop count,
execution ID, partition, and raw exposure
`(pt_bytes, elapsed_ns, pebs_samples)`. The source manifest SHA-256 is
`0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d`.
Join it to the eval rows by execution ID and verify the hash and unique join
before choosing pairs. The manifest list order is only a collection-order
proxy: it has no per-execution start timestamp or session ID. The 102,000 rows
are one long collection on one boot, not independent held-out sessions.

The compact feature layout is PT histogram mean/std `[0:512]`, PEBS feature
mean/std `[512:784]`, four PMU rates/IPC `[784:788]`, and PEBS availability
fraction `[788]`. The compacting step discards exact PT byte order and PEBS
segment order. Thus a splice below tests **execution-level sensor
co-occurrence**, not temporal alignment, kernel control-flow correctness, or
localization. Any stronger time-resolved proxy would need separately retained
raw or full segmented derived traces and a new reviewed design.

## Proposed locked intervention

1. Use only familiar-validation and held-out-family rows as anchors; do not
   perturb or select on training/calibration rows. Within each partition,
   family, and *exact* loop count, choose two distinct admitted, complete rows
   $A$ and $B$ no more than 5,000 manifest positions apart. Require matching
   capture/subject/event identity and no loss, multiplexing, migration, or
   missing modality. Predeclare exposure calipers before opening scores:
   $|\log(\mathrm{PT}_A/\mathrm{PT}_B)|\leq\log(1.10)$,
   $|\log(t_A/t_B)|\leq\log(1.10)$, and
   $|\log((1+\mathrm{PEBS}_A)/(1+\mathrm{PEBS}_B))|\leq\log(1.20)$.
   Use minimum exposure distance within these calipers, then a fixed hash of
   execution IDs for ties; pair without replacement. Freeze the complete pair
   list and its hash before scoring. Report eligible anchors, unpaired rows,
   caliper distances, and coverage per family/intensity; a sparse cell is a
   failed diagnostic, not permission to relax matching after results.
2. The main synthetic record takes $A$'s PT `[0:512]` and $B$'s PEBS
   `[512:784]`, PMU `[784:788]`, and PEBS-availability `[788]`. Supply the
   v2 scorer with raw exposure `(PT_A, elapsed_A, PEBS_B)` so the exposure
   correction sees the source of each sensor block. This is a whole-block
   splice, not arbitrary random noise or a fabricated overflow. Predeclare
   PEBS-only and PMU-only swaps as ablations, not opportunities to select the
   best-looking perturbation.
3. Score, with identical frozen inference, the unmodified $A$, unmodified $B$,
   hybrid, an $A\to A$ sham splice, and a **metadata-only** control: $A$'s
   unchanged features with the hybrid exposure tuple. The unmodified $B$ is
   the whole-record replacement control. A score increase that is also present
   in the metadata-only control is an exposure mismatch artifact, not evidence
   of cross-modal modeling. A high score on $B$ alone is benign novelty, not
   hybrid-specific discordance. Keep every row and failed pair in the ledger;
   never choose donors using modality values, anomaly scores, or review labels.

Matching by family, exact intensity, collection-order neighborhood, and
exposure removes easy runtime/trace-size explanations. It does **not** guarantee
that the two lawful executions have different relevant behavior; a near-zero
effect is a legitimate result. Current target input was not independently
seed-varied per execution, so this corpus cannot establish input-seed
invariance. It also lacks the separately labeled slow-success, matched-error,
and real-effect siblings needed for a physical bug gate.

## Effect size, nuisance checks, and low-FPR reporting

The intervention dose is one complete PEBS+PMU block from one matched lawful
donor. Do not tune a mixing fraction or donor distance after observing scores.
Report per family/intensity the paired score difference
$s(H)-\max(s(A),s(B))$, its median and interquartile range, and the win rate
$P[s(H)>\max(s(A),s(B))]$ with ties shown separately. Report the same quantities
for sham and metadata-only controls. Positive controls are synthetic hybrids;
there is no semantic wrong-byte or kernel-effect oracle.

With a threshold frozen on **unmodified calibration** rows, report hybrid
admission, both original-row admissions, sham/metadata-only admissions, and
the fraction of pairs where only the hybrid alerts. Report per-family and
per-intensity cells, worst-cell rate, missing/censored count, and raw top-100
rank among *all* unmodified benign development rows plus prespecified hybrids.
The latter is a retrospective queue diagnostic, not a prospective review-budget
claim. Show both $10^{-3}$ and exploratory $10^{-4}$ operating points; the
21,000-row calibration has only about two tail examples at $10^{-4}$, and the
7,000/18,000 validation splits are too small to certify a 100-per-million
population FPR. Use exact one-sided benign binomial upper bounds and a
collection-block bootstrap, acknowledging that blocks from one session are
not independent sessions. A pooled rate must not hide a concentrated family.

At least three *new, later, separately identified physical collection sessions*
with disjoint per-execution input seeds, shuffled run order, matched high-load
and error controls, unchanged kernel/boot/capture policy, and a separate large
calibration set are required before even a cross-session **proxy** claim.
Pre-register pair construction and a single frozen checkpoint before opening
those scores. The existing one-session sealed cache can support a development
diagnostic only. Neither a proxy pass nor an artificially improved low-FPR
number can replace the [real-bug physical effect/control gate](hardware-anomaly-validation-r1.md).
