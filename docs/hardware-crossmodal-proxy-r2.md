# Cross-modal discordance proxy R2: review-only design

Status: **development diagnostic executed on frozen cache/checkpoints, not a
bug gate** (2026-09-26). No model was trained or selected by this diagnostic.
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

## Frozen CPU-only execution and result

The two-stage implementation is
[`hardware_crossmodal_proxy_r2.py`](../python/cpu2tensor/examples/hardware_crossmodal_proxy_r2.py).
Its pair stage did not load a checkpoint. It froze 12,252 disjoint pairs from
24,504 of 25,000 validation rows (98.016% coverage), leaving 496 unpaired
across 51 family-by-intensity cells. The immutable
[`pairs.json`](/Users/theoad/.cache/cpu2tensor/hardware-autoresearch-v2-exposure/crossmodal-proxy-r2/pairs.json)
has SHA-256
`7071a3e50c6c811a6ebe54565d5b59a2efd42c4a7a0b3d77a2fad7358c845730`.
All four checkpoints were then scored on **Darwin arm64 CPU**, not the physical
Linux capture host or Mac GPU. The final
[`report.json`](/Users/theoad/.cache/cpu2tensor/hardware-autoresearch-v2-exposure/crossmodal-proxy-r2/report-verified/report.json)
SHA-256 is
`f68977263a3b05d5c619400abf70d38f703daf18d9e402eff185cec2216ea6ba`;
its seed-wise [`scores.pt`](/Users/theoad/.cache/cpu2tensor/hardware-autoresearch-v2-exposure/crossmodal-proxy-r2/report-verified/scores.pt)
SHA-256 is
`a3477f0abdb8cdbdcf28ebadd74b029907609f6bd6bdc1ab157cd2b62431d396`.
The report retains every family/intensity coverage and score/control metric.
The runner-smoke checkpoint hashes are v1 seeds 1729/1730
`b44cd086c1c7f1b5fafb54c0968026d6785638f70c836be069367ef17d128f0c`/
`aaed68240b843d2d437af898ddae4400b04bef18627752fa061aef758a8495c8`,
and v2 seeds 1729/1730
`abf3d8af55927292d57a1416f0a4e289ba1d1eae2411a889d9ad736f93a6ffe0`/
`543ff4f97a3b0ebc30ee20ef451fda7d8a01c1dc5d4fa4b63f0293fffe46fc60`.

| Frozen scorer | Hybrid wins versus both originals | Hybrid-only alerts at exploratory $10^{-4}$ | Unmodified held-out benign alerts at $10^{-4}$ |
| --- | ---: | ---: | ---: |
| v1 seed 1729 | 27.16% | 25/12,252 | 10,463/18,000 |
| v1 seed 1730 | 26.73% | 25/12,252 | 10,100/18,000 |
| v2 seed 1729 | 26.84% | 24/12,252 | 11,500/18,000 |
| v2 seed 1730 | 26.67% | 0/12,252 | 12,000/18,000 |

Every scorer's median $s(H)-\max(s(A),s(B))$ was negative. The hybrid alert
counts were close to the anchor, donor, sham, and metadata-only counts; PEBS-only
and PMU-only ablations are in the report. At $10^{-4}$, all 6,000 unmodified
`memfd` rows alerted for every checkpoint; `dup` contributed 4,099–6,000 more
of its 6,000 rows, while `pipe` contributed 0–3. Most hybrid-only alerts were
in `dup` (one v1 seed-1730 `pipe` exception). At $10^{-3}$, held-out benign
alerts remained 12,015–12,042/18,000 and hybrid-only alerts were only 5–10.
This is dominated by a known lawful-family novelty failure, not a selective
cross-modal discordance response. The retrospective mixed-pool top-100 held
32–37 hybrids, but that queue is neither a real-bug recall estimate nor an
operational 100-per-million result.

Familiar validation had 0/7,000 original alerts at exploratory $10^{-4}$ for
all four checkpoints, yet its exact one-sided 95% upper bound is
$4.28\times 10^{-4}$ even under an independence assumption. The 21,000-row
calibration admitted two rows at that threshold. This one-session cache
cannot support a session-block confidence interval or certify a low-FPR
production threshold. The proxy verdict is **NO-GO for cross-modal sensitivity
evidence from these frozen models**; it is not a model-selection decision and
says nothing affirmative or negative about real-bug recall.

Focused tests for pairing, controls, the v1/v2 external evaluator, and the
real frozen-v2 scorer interface passed: `12 passed in 0.64s` with source-first
`PYTHONPATH` and `python3.12 -S -m pytest`.
