# Hardware anomaly validation R1: exact-host decision gate

Status: **research protocol, not an authorized host run** (2026-09-26). This
document asks whether a frozen, non-CVE-tailored kernel model can rank a real
pathological effect against matched lawful and high-load activity on the *same*
production kernel and physical host. It does not equate a synthetic perturbation,
a training loss, or a discovered label with bug detection. The six-hour deadline
is for a research/launch decision; one million physical executions need not fit
inside it. No run described here has been launched by this validation slice.

## Decision from retained evidence

The only validated vulnerable subject is the unchanged, signed Ubuntu Impish
`5.13.0-30.33` kernel on the physical `trail-x86` i7-10510U. The no-turbo
102,000-execution corpus contains 56,000 training, 21,000 calibration, 7,000
familiar validation, and 18,000 whole-family holdout rows, all admitted on the
first attempt without source loss. Its finite runner sustained 13.07 sealed
executions/s over 7,805.9 seconds. The small fused model admitted 65/7,000
familiar and 2,073/18,000 unseen-family benign executions; the larger model
admitted 67/7,000 and 2,240/18,000. Thus familiar admission was **9,286** and
**9,571 per million**, respectively, versus the desired 100 reviews/million;
unseen-family admission was 115,167 and 124,444 per million. These are
execution-alert rates, not independent population FPR estimates, because rows
share sessions and workloads. In 12 blinded Dirty Pipe effect/near-miss pairs
per checkpoint, every effect manifested 20/20 disposable-file page-cache
mutations, yet small and large model AUROCs were 0.5208 and 0.4722, with zero
alerts in both cells and one alert in both cells, respectively. See the
[retained result log](hardware-pretraining-results.md#bounded-102000-execution-scale-collection).

The resulting decision is **NO-GO for a bug-detection or million-execution
launch using either current checkpoint/representation**. Scaling model width
did not fix the observed miss or nuisance rate. This does not establish that
raw PT, precise stores, or another model cannot work; it identifies the current
measurement/learning boundary. Dirty Pipe stays a secondary logical challenge,
not the gate used to select a replacement model.

## Primary real-bug gate: admission before execution

The primary must be a *different mechanism* from Dirty Pipe page-cache mutation:
a bounded wrong-byte write to a caller-owned object, a recoverable fault, or a
bounded pathological execution. It must run unmodified on this exact Ubuntu
package, boot, microcode, frequency policy, CPU assignment, perf events, and
capture revision. A public upstream version range alone does not establish that
Ubuntu's specific backport is vulnerable. Require an upstream fix and Ubuntu
package-diff check, followed by a **small, oracle-only rehearsal on disposable
infrastructure**, before considering the physical laptop. A fixed-build
comparison is informative but cannot replace the within-build matched sibling.

One promising-looking bounded candidate was **ruled out by exact-package
provenance**. The upstream idmapped-mount circular-mapping `setattr` defect
could wrongly permit or deny an ownership update on a disposable inode; the
[upstream fix](https://lkml.rescloud.iu.edu/hypermail/linux/kernel/2111.3/01773.html)
and [xfstests oracle](https://kernel.googlesource.com/pub/scm/linux/kernel/git/djwong/xfstests-dev/+/refs/tags/metadir-quotas_2024-08-22/tests/generic/656)
make it superficially attractive. But Canonical's
[Impish package chronology](https://lists.ubuntu.com/archives/impish-changes/2022-February/008704.html)
places `fs: handle circular mappings correctly` in `5.13.0-29.32`, **before**
the selected `5.13.0-30.33`. Its appearance again in later inherited
changelogs is not evidence that the earlier production package was vulnerable.
It is excluded unless an exact signed-package source/binary audit contradicts
that chronology; no physical trigger is justified. This is why an upstream
version range or a later changelog excerpt cannot establish the validation
subject.

No second candidate is **yet admitted on the exact physical subject**. The
existing documentation names `fs_context` overflow (CVE-2022-0185) and an
`io_uring` timeout race (CVE-2022-29582), but these can corrupt kernel memory
or crash the machine and have no demonstrated bounded, non-weaponized oracle on
this Ubuntu boot. They are inappropriate for a million repeats on a user's
physical laptop. The apparently simpler ICMPv6 memory-leak bug
([CVE-2022-0742](https://ubuntu.com/security/CVE-2022-0742), fixed in Ubuntu
Impish `5.13.0-37.42`) is also rejected for this path: a leak is cumulative and
could exhaust host memory, and network receive/softirq activity is not necessarily
attributed to the gated target process by the current `process_kernel` capture.
The exact archived `5.13.0-30.33` fix state and a safe per-execution oracle would
still need verification. These are **screening leads, not authorized triggers**.
If no candidate clears the risk and attribution checks, preserve the real-bug
gate as blocked; do not relabel a surrogate positive as a known bug.

### Read-only exact-package re-audit, 2026-09-26

The physical `trail-x86` host reports `5.13.0-30-generic` and
`/proc/version_signature` says `Ubuntu 5.13.0-30.33-generic 5.13.19`.
`dpkg-query` reports `5.13.0-30.33` for its image, modules, and extra-modules
packages. The boot config enables `CONFIG_WATCH_QUEUE=y`, `CONFIG_USER_NS=y`,
and `CONFIG_NF_TABLES=m`. These are read-only identity/prerequisite checks, not
evidence that a defect has manifested. The installed image's `linux-signed`
changelog records the `5.13.0-30.33` master version. The running kernel was not
replaced or modified.

| Lead distinct from Dirty Pipe | Exact-package evidence | Admission decision |
| --- | --- | --- |
| Filesystem-context out-of-bounds write, CVE-2022-0185 | Canonical's [USN-5240-1](https://ubuntu.com/security/notices/USN-5240-1) lists the Impish fix at `5.13.0-27.29`, before this host's `-30.33`. | **Ruled out** on the selected package; an upstream affected-version range must not override the Ubuntu fix. |
| `nf_tables` invalid-register out-of-bounds write, CVE-2022-1015 | Canonical's [CVE record](https://ubuntu.com/security/CVE-2022-1015) and [Impish `5.13.0-40.45` history](https://lists.ubuntu.com/archives/impish-changes/2022-April/009232.html) place the fix after `-30.33`. | Plausibly present but **not admitted**: kernel-memory overwrite may crash or persistently corrupt the host; no bounded caller-owned effect oracle or lawful same-load sibling has been shown. A userspace timeout cannot bound kernel damage. |
| Netfilter offload out-of-bounds write, CVE-2022-25636 | Canonical's [USN-5317-1](https://ubuntu.com/security/notices/USN-5317-1) first lists the Impish generic fix in `5.13.0-35.40`, after `-30.33`. | Plausibly present but **not admitted** for the same physical-host memory-corruption and oracle/control reasons. |
| Watch-queue out-of-bounds write, CVE-2022-0995 | The subsystem is configured in this boot and [Canonical describes](https://ubuntu.com/security/CVE-2022-0995) possible crash and privilege escalation. Its current Impish tracker entry is not a clear exact `-30.33` fix statement. | **Not admitted**: exact-source status is unresolved and a kernel-memory write is not a safely bounded oracle. `panic_on_oops=0` does not make memory corruption recoverable. |
| TCP page-fragment wrong-byte stream | The [upstream fix](https://github.com/torvalds/linux/commit/dacb5d8875cc6cd3a553363b4d6f06760fcbe70c) and later [Impish history](https://lists.ubuntu.com/archives/impish-changes/2022-March/009011.html) support a bounded receiver-byte oracle and a fix after `-30.33`; see the separate [candidate audit](hardware-anomaly-candidate-r1.md). | Best *research lead*, still **not admitted**: the required nested CIFS/SMB page-fault traffic has no demonstrated matched high-load sibling or repeatable untraced oracle, and the exact signed-tag source diff has not been completed. A read-only Launchpad tag-file fetch returned HTTP 403; chronology is not a binary audit. |

The decision remains **NO-GO for any second physical bug trigger**. The next
reversible step is an offline exact-tag source/patch comparison and a separately
reviewed, disposable VM rehearsal using the exact signed package, an isolated
network, a revertible snapshot, no writable host-shared mount, and small
memory/disk/runtime bounds. A receiver-side byte oracle for the TCP lead would
need the same CIFS mount, traffic, payload, size, affinity, and fault pressure
in both arms; only the hypothesized page-fragment recursion may differ. Any
kernel warning, hang, taint, wrong unrelated bytes, or non-repeatable oracle
stops that lead. A VM can establish mechanism and harness safety, but not
physical-host PT/PEBS/PMU sensitivity. No such rehearsal, physical trigger,
perf capture, or new collection was run in this audit.

An admissible candidate must pass all of the following before score inspection:

1. Source/fix provenance proves the defect exists in the exact signed package;
   a second person reviews the proposed harness and its maximum side effects.
2. Each invocation has a bounded runtime and resource limit, touches only
   caller-owned disposable state, needs no privilege escalation or payload, and
   cannot leave a persistent kernel or host change. Any kernel warning, taint,
   hung task, system memory trend, or thermal threshold is an immediate stop.
3. An independent semantic oracle distinguishes the defect from a legitimate
   result in repeated *untraced* rehearsals. The oracle does not inspect the
   anomaly score. Capture covers the affected target's execution, not only the
   caller or a softirq elsewhere.
4. The matched lawful sibling uses the same subsystem, call count, buffer sizes,
   data volume, affinity, and loop intensity while removing only the bug-enabling
   condition. Additional near misses include the boundary value, a rejected
   parameter, equivalent high-load success, and an equally slow lawful path.
   These controls prevent runtime, trace length, error return, or workload-family
   novelty alone from winning.
5. A small randomized physical pilot, separately authorized and coordinated with
   the capture owner, shows repeatable oracle-positive effects and lossless
   observation without host-health change. Failure at any step ends that
   candidate. Do not continue to increase iterations to force manifestation.

## Safe interim control, deliberately not a vulnerability claim

Use the existing unprivileged `hardware_multimodal_canary` process-scope flow,
memory, contention, and phase variants to check acquisition and model response.
For a kernel-only representation, use bounded benign syscall-family, intensity,
parameter-boundary, and modality/time-swap controls from retained raw traces.
These are sensor and nuisance controls. Neither an injected tensor corruption
nor a lawful slow/error path is a substitute for a real defect on the production
kernel. The existing process-scope canary also cannot be mixed into
`process_kernel` training or the kernel-only operating threshold; see the
[validation contract](hardware-pretraining-validation.md).

## Frozen evaluation contract

Freeze source revision, subject manifest, representation, architecture,
checkpoint, score definition, deterministic inference-thread count, seed list,
and one strict `score > threshold` rule **before** primary or Dirty Pipe labels
are revealed. Train on broad, benign, subject-matched families and intensity
variants only. Calibration uses separate entire executions and randomized
sessions from the same exact boot; final tests use later sessions and disjoint
seeds/inputs. A reboot, changed frequency policy, package, microcode, perf
configuration, or boot decode state creates a new subject and requires a new
model/calibration. Never split windows of one execution across partitions.

The final physical ledger targets at least **1,000,000 independently scored
benign executions** *after* training and calibration. This is an eventual
campaign gate, not a six-hour promise. It must include all 17 familiar families,
whole-family `dup`/`memfd`/`pipe`-style holdouts, boundary and intensity variants,
and the matched high-load/error near misses; report each family and session
separately and the worst family explicitly. Before final freeze, run a separate
17-fold leave-one-family-out *development diagnostic*, fitting a fresh benign
model/calibration in each fold and revealing every hidden-family rate. These
folds are not 17 independent final confirmations and their held-out labels
cannot tune the one production checkpoint. A pooled pass with a catastrophic
family fails. Pre-register the family mix and the minimum count per cell so the
aggregate cannot be improved by shifting weight toward easy `getpid` windows.
The 21,000-row inherited calibration has only about two expected examples beyond
a 1e-4 tail and cannot locate a stable operational threshold. Collect a much
larger independent calibration partition or use a distribution-free upper-tail
rule with an explicitly conservative threshold; either choice must be frozen
before the final million. Existing benign shards may be used for *development*
but not reused as independent final-test evidence.

For each admitted execution retain the score, threshold, decision, family,
input/seed, oracle, session and subject hashes, all loss/multiplexing and CPU
attribution fields, and a compact feature record. Score before eviction;
preserve original PT AUX, PEBS/PMU records, maps, and exact-boot decode state
for **every** alert, integrity failure, and a preselected content-independent
benign audit sample. A dropped/lost/migrated execution is censored and still
counts against capture yield; it is never treated as a non-alert. Every alert
enters the review ledger, even if an explained-benign cluster later saves an
LLM call. Any unresolved alert remains in conservative FPR accounting. The
[evidence bundle](hardware-anomaly-evidence.md) is the LLM review input; no
trigger label, oracle result, or CVE name is exposed to it.

An initial, separately authorized physical **pilot may stop at 50 randomized
pairs** to assess manifestation, attribution, and host safety; it cannot prove
an 80% population recall lower bound. The stronger final gate targets at least
100 independently randomized oracle-positive executions and 100 paired lawful
siblings across at least three sessions, plus at least 100 each of slow-success
and matched-error near misses. The oracle check precedes scoring and excludes
non-manifestations from the recall denominator while reporting their count.
The review budget has two separate
gates: **at most 100 raw admissions and at most 100 actual LLM escalations per
million scored benign executions**. Cluster
deduplication may reduce escalations, never the raw-admission rate. An alert is
not exonerated by an LLM narrative alone. At the frozen threshold, also report
recall on the oracle-positive primary bug, paired effect-versus-sibling win rate,
matched sibling/near-miss admissions, precision under the declared test mix,
time-to-first-alert, localization overlap with the independently known affected
interval, and PT-only/PEBS-only/PMU-only/fused ablations. AUROC is descriptive;
it cannot replace recall at the operational threshold. Dirty Pipe is scored
once as a separate label-hidden challenge after the primary/model decision.

As a **retrospective diagnostic only**, a predeclared unsupervised evidence
cluster ID may cap the number of rows per cluster in a score-descending top-100
queue. The total queue budget stays 100, the cluster assignment and cap freeze
without labels, and no threshold, model weight, or family-specific admission
rule changes. Report uncapped and capped raw top-100 recall together. A capped
queue can diversify reviews of familiar nuisance clusters, but retrospective
improvement on known labels is not prospective bug-recall evidence and does not
reduce the raw-alert/FPR count.

The standard-library-only
[`hardware_review_budget_r1.py`](../python/cpu2tensor/examples/hardware_review_budget_r1.py)
calculates these frozen-score diagnostics offline. Its input requires an
explicit complete-capture bit per row and refuses censored or non-finite
scores. It does not collect traces, choose a threshold, assign clusters, or
update a model.

## Uncertainty and scale arithmetic

With 100 alerts among one million independent benign trials, the point estimate
is 100/million but the exact one-sided 95% binomial upper bound is about
**118.08/million**. At most **83** alerts are needed for that upper bound to be
below 100/million (upper bound about 99.62); 84 gives about 100.71. These exact
bounds assume independent Bernoulli trials. Workload/session
dependence makes them optimistic, so report a session-block bootstrap and a
worst-family bound too; a single correlated episode must not be counted as
independent proof. If 100 oracle-positive primary executions all alert, the
one-sided 95% Wilson lower recall bound is about 97.4%; at 87/100 it is about
80.5%, while 80/100 gives only about 72.7%. Use at least 100 independently
randomized positive executions across multiple sessions for an 80%-recall
lower-bound gate, with paired lawful siblings; repeated loops inside one capture
are not 100 independent trials. The currently available 12 paired Dirty Pipe
tests are too small for a strong recall claim even if they all had alerted.

One million executions in six hours would require **46.30 complete executions/s**
including capture, tensorization, scoring, and the retention decision. The
existing 102,000-run finite all-derived/all-audit collector sustained **13.07/s**
on `trail-x86`; at that measured rate, one million takes about **21.25 hours**
before pilot, calibration, positive runs, or interruptions. A steady 8,119/s
`getpid` window smoke excludes durable alert custody and model inference and
must not be substituted for end-to-end throughput. A no-loss score-before-evict
pipeline and host-health stop policy must pass before scheduling the eventual
million; the capture owner reserves the physical host. The six-hour research
deliverable can only be a GO/NO-GO decision, a frozen protocol, and bounded
offline tests. Current evidence warrants **NO-GO** for the full campaign.

## Required next evidence

- Read-only source/package provenance and a reviewed bounded, independently
  oracled primary candidate of a different mechanism; no physical trigger
  until the host owner and user authorize that specific risk.
- Offline replay on the retained same-subject corpus to show a prospective
  representation reduces family/intensity nuisance without selecting on the
  primary or Dirty Pipe labels. Preserve both rejected model baselines.
- A new, separate large calibration set and independent final benign sessions,
  then a small oracle-positive/near-miss pilot. Only a pilot that passes safety,
  capture integrity, and matched sensitivity can unlock a million-execution
  schedule. A failure is a scientific result, not a reason to loosen the review
  budget or unblind another checkpoint.
