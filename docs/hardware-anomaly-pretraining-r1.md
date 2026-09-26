# Hardware-anomaly pretraining objective R1: conditional residual

Status: **NO-GO for this objective as an operational scorer** (2026-09-26). This
is one benign-only, CPU-only experiment on the immutable 102,000-execution
physical-host corpus. No vulnerability or canary labels were loaded, no capture
or model contract was changed, and no Mac MPS or `trail-x86` resources were used.
The question was whether a cross-modal residual suppresses benign family
novelty while retaining a discrepancy between PT/PMU and precise-store PEBS.

## Frozen experiment

The source was `cpu2tensor.examples.hardware_anomaly_pretraining_r1` at base
revision `4f34880880ec17519e6f771b224556dc6e4ee5c7`; the new script SHA-256
is `f9cddd5a54824c119daa47805102ed7ee5f1f03402261398095e098ccd7169dc`.
It read the completed compact-feature cache **read-only**, checking the complete
322,016,045-byte cache SHA-256
`5c689576cfd8d1576aa6f063e598a02cfee21d0fa629141e4c9913a8fda1fcdc`,
schema, finite `float32 [102000,789]` tensor, partition alignment, and sealed
manifest SHA-256
`0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d`.
The cache producer verified every derived-shard hash during extraction. The
captured subject was the physical `trail-x86` i7-10510U, Ubuntu
`5.13.0-30-generic` boot `31179aea-9a43-4c5d-8bf6-1205735e42c4`, with
no-turbo policy, process-kernel PT, period-1,000 precise stores, and boundary
instructions/cycles/reference cycles. The 102k rows are 56k training, 21k
calibration, 7k familiar validation, and 18k whole-family holdout (`dup`,
`memfd`, `pipe`; 6k each). These are one collection session, not independent
boot/session replications.

The input is the existing 789-D compact vector: PT byte-histogram mean/std
(512), PEBS sketch mean/std (272), PMU counters (4), and PEBS availability
(1). Training-only mean/std standardization, a 0.05 scale floor, and clipping
at eight standardized units are frozen. A ridge predictor takes PT and
PMU/availability (517 context features) to predict PEBS (272 target features)
from the **same benign execution**:

$$
W = \underset{W}{\arg\min}\;\|XW-Y\|_F^2 + 0.1N\|W_{\mathrm{nonbias}}\|_F^2.
$$

The anomaly score is the mean squared PEBS residual after dividing each target
feature by its *training-only* conditional residual scale (floor 0.05). Its
unconditional control scores standardized PEBS magnitude with a training-only
scale. This is one fixed objective and one fixed ridge strength; neither
validation partition nor a bug label selected a hyperparameter. An explicit
nuisance ablation adds operator-requested `log2(loops)` as a 518th context
feature, normalized on training rows only; its 141,440 weights/scales were
fit with the **same** objective and ridge fraction. This is a diagnostic, not
a score-selection search. Each scorer
gets its own threshold from the same 21k benign calibration executions using
strict `score > threshold`. The thresholds at `1e-4` are exploratory because
calibration supplies only about two tail order statistics there.

## Observed result

On `mac.local` Apple M2/arm64, Torch 2.14 CPU with four threads, cache hash/load
took 1.334 s, ridge fitting took 0.147 s, and conditional scoring of the 46k
nontraining rows took 0.066 s (692,424 rows/s for the **score kernel only**).
The loop-conditioned ablation fit plus score took 0.221 s.
This is Mac development throughput, not x86 production throughput or an
end-to-end capture-to-score result. The checkpoint has 141,168 weights/scales,
reloaded bit-exactly on CPU, and has SHA-256
`000b5f93849feb9a74485f7ac47aa06e04d406189811dbd216e5414e8d2ddcd8`.
The report SHA-256 is
`7e9f973eb89a89a284af3cfd708866e662b6784e88e5d19d13a9f60651427b9d`;
the report and checkpoint are under
`/Users/theoad/.cache/cpu2tensor/hardware-anomaly-pretraining-r1/`.

| Benign calibration tail | Score | Familiar alerts / 7k | Whole-family alerts / 18k | `dup` / `memfd` / `pipe` alerts |
| --- | --- | ---: | ---: | ---: |
| `1e-3` | conditional PT+PMU→PEBS residual | 4 | **126** | 123 / 2 / 1 |
| `1e-3` | same objective + `log2(loops)` context | 4 | **128** | 125 / 2 / 1 |
| `1e-3` | unconditional PEBS control | 3 | **7** | 6 / 1 / 0 |
| `1e-4` exploratory | conditional residual | 0 | 1 | 1 / 0 / 0 |
| `1e-4` exploratory | conditional + `log2(loops)` | 0 | 1 | 1 / 0 / 0 |
| `1e-4` exploratory | unconditional control | 0 | 0 | 0 / 0 / 0 |

The cross-modal objective learned a strong training relation (mean raw
conditional target MSE 0.132), but the relation does **not** transfer safely to
unseen `dup`. Conditioning explicitly on loop intensity did not repair it.
The `1e-3` hidden-family alarm count is 18 times the simple
unconditional control's: 7,000 versus 389 benign alerts per million on this
particular held-out mixture, both above the 100-per-million review cap.
The difference between one and zero alerts at
`1e-4` is below the available calibration resolution and is not an operational
success. No reusable low-false-alarm score improvement is demonstrated. The
conditional residual may still be a sensor-consistency diagnostic, but no
corruption sensitivity, bug recall, or execution-grammar learning is established.
In particular, this compact PT histogram discards packet adjacency and branch
targets, while PEBS sketching discards precise site transitions. The result
cannot falsify a later *raw-token* objective.

## Nuisance evidence and missing channels

The sealed manifest supplies exact boot ID, kernel release/microcode, CPU model,
target CPU 2 and controller CPU 3, no-turbo policy, perf event identities and
period, loop divisor/intensity, family, repetition, and one global randomized
split seed. It supplies per-execution elapsed time, loss/admission state, and
per-lane TID, observed CPU, and migration verification. Retained raw audit
shards additionally contain PEBS exact IP/address/data-source/timestamp/CPU,
PT AUX bytes and perf records, a `CLOCK_MONOTONIC_RAW` arm/stop envelope, PMU
`time_enabled_ns`/`time_running_ns`, process maps, kernel modules, and
hash-verified exact-boot symbol/decode snapshots. One inspected retained raw
lane had equal enabled/running time; that observation is not a corpus-wide
schedule estimate. The compact 102k cache omits most of these nuisance fields
and all raw PEBS addresses. Only 525 content-independent audit rows retain raw
shards, so they cannot replace the 102k training set.

There is **no explicit KASLR slide scalar** in the manifest or compact features.
The exact-boot symbol and module snapshots make some address normalization
possible offline, but they do not create observed slide variation. All 102k
executions share one boot and nominal target CPU; the global split seed is not
an execution seed. Repetition is an index, not a documented random input seed.
The PT byte offsets have order but no qualified clock alignment to PEBS;
assigning a PEBS timestamp to a PT byte window would manufacture precision.
Thus this run tests neither cross-boot relocation invariance nor session drift.

For a later representation, preserve **both** raw and normalized address
channels: raw exact-boot IP/address bits for within-boot evidence and
`(build-id or module identity, section-relative offset)` for resolvable
instruction addresses; raw data address alongside mapped-region identity plus
region-relative/page offset where a valid process map exists. Include explicit
unknown/ambiguous-map masks rather than coercing unresolved addresses to zero.
Condition on CPU/lane identity, capture duration/envelope uncertainty, perf
enabled/running ratio, and loop intensity without letting repetition or an
oracle label become a shortcut. Synthetic address shifts can unit-check the
normalizer only; one boot cannot support an empirical KASLR robustness claim.

## Small next falsification test, not a scale authorization

If the coordinator authorizes new raw retention, collect two randomized benign
sessions on the **same exact subject/boot**, including all 17 families and loop
intensities, with a predeclared family holdout and complete raw custody. In
session 1, fit a small masked conditional predictor on short within-lane PEBS
site/address *transition* tokens. Condition it on nearby PEBS tokens, an
execution-level PT byte-bigram sketch that retains order **without claiming
clock alignment**, and interval PMU plus the observed nuisance fields above.
Use the dual raw/normalized address channels and masks. A linear or one-hidden-
layer model with a predeclared parameter cap below 200k is sufficient for this
diagnostic; do not launch a full transformer. Fit and calibrate on disjoint
session-1 execution blocks, then freeze and evaluate untouched session 2,
reporting familiar and each whole-family benign tail separately. Compare
against an unconditional transition control at identical thresholds and time
capture-to-score on the x86 host. This two-session result can falsify nuisance
resistance, but cannot certify `1e-4` FPR without a much larger independent
calibration set, cannot establish cross-boot KASLR transfer, and cannot stand in
for label-hidden real-effect sensitivity.

Focused verification: `4 passed` in
`python/tests/test_hardware_anomaly_pretraining_r1.py` on `mac.local`; the tests
cover conditional response, training-only normalization, and strict
calibration ordering. The CPU script completed and reloaded its checkpoint.
The run used `python3.12 -S -m
cpu2tensor.examples.hardware_anomaly_pretraining_r1` with the corpus, the
sealed compact cache, the separate output directory above, and the cache hash
shown above. `PYTHONPATH` explicitly pointed at this worktree's `python/` plus
the installed Python 3.12 package directories; `-S` avoided a pre-existing
editable-install redirect to another checkout. The focused pytest command used
the same interpreter settings. Neither command modified the corpus or cache.
