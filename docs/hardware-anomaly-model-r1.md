# Hardware anomaly model R1: offline objective and density check

This is a bounded model-only investigation on the **immutable 102,000-execution
physical-host corpus**, not a detector release. The collection and corrected
Dirty Pipe failures are documented in [hardware pretraining results](hardware-pretraining-results.md).
The question here is whether a generic denoising objective or a simple density
score improves transfer to benign syscall families withheld in their entirety.
No vulnerability labels, corrected canary scores, or held-out-family examples
were used to fit weights, covariance, normalization, prototypes, or thresholds.

## Fixed data and representation

The source is the Mac copy at
`/Users/theoad/.cache/cpu2tensor/hardware-pretraining-go/store-v3-scale100k-r2`.
Its manifest SHA-256 is
`0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d`;
the dataset identity is
`c96e0376df8db87c8249066e17633c2c521888c296f0987e4e1ebe2eb581faa5`.
The exact existing split is 56,000 training, 21,000 calibration, 7,000 familiar
validation, and 18,000 whole-family holdout. All 102,000 derived-file hashes
and execution identities were checked while extracting this experiment's
compact feature cache. The prior extraction check also verified all retained
raw hashes; this run did not rehash the raw archive.

Each complete execution becomes 789 float32 features: 256 PT byte-histogram
means and 256 segment standard deviations, 136 PEBS feature means and 136
segment standard deviations over available samples, four log-transformed PMU
features, and one PEBS availability fraction. Unavailable PEBS tokens are
ignored, not treated as zeros. Training-only mean and standard deviation
normalize the features, with a fixed clipping bound of 20. This compression
deliberately discards exact segment order, packet semantics, and precise-store
causal context; it is a baseline, not a new invariant-bearing representation.

Three preregistered generic scores were fit to training only: a 438,101-parameter
denoising MLP (789→256→64→256→789, 15% feature masking plus Gaussian noise,
500 AdamW steps), a 64-component low-rank Gaussian/residual density score
(50,560 stored coefficients), and a 789-scale diagonal density control. A
diagnostic nearest-prototype score uses 42 benign training family-by-loop
centroids (33,138 stored coefficients), but does not receive validation family
labels at inference. The low-rank score was the prespecified primary new
candidate; the others expose objective and conditioning failure modes.

## Benign ranking results

Thresholds come only from the 21,000-row calibration partition. At the
exploratory $10^{-3}$ budget, 21 calibration rows exceed the score threshold;
at $10^{-4}$, only two do. An independent 7,000-row familiar test cannot
certify a $10^{-4}$ false-positive rate even if it has zero alerts. Scores
strictly greater than the selected calibration order statistic alert.

| Frozen score | Familiar at $10^{-3}$ | Whole-family at $10^{-3}$ | Familiar at $10^{-4}$ | Whole-family at $10^{-4}$ |
| --- | ---: | ---: | ---: | ---: |
| Denoising MLP | 6/7,000 | 10,379/18,000 | 0/7,000 | 9,044/18,000 |
| Low-rank density | 6/7,000 | 9,809/18,000 | 0/7,000 | 6,209/18,000 |
| Diagonal density | 9/7,000 | 2,068/18,000 | 0/7,000 | 2,068/18,000 |
| Nearest training prototype | 9/7,000 | 11,265/18,000 | 0/7,000 | 10,037/18,000 |

These are prospective-looking rankings on *benign* withheld families, not
false-positive adjudications of individual unexplained traces. The whole-family
failure is much larger than calibration uncertainty. None of the new scores
qualifies for a corrected Dirty Pipe replay; that canary remains untouched by
this model iteration.

The alerts are structured. At $10^{-3}$, diagonal density flags exactly the
2,068 high-intensity `memfd` executions (`loops=5000`) and no `dup` or `pipe`.
Low-rank density flags all 6,000 `memfd`, 3,809/6,000 `dup`, and no `pipe`.
The nearest training prototype still flags all 6,000 `memfd` and 5,265/6,000
`dup`. Among `dup`, low-rank alerts rise from 158/2,023 at 1,250 loops to
1,646/1,972 at 2,500 and 2,005/2,005 at 5,000. Thus the tail is concentrated
in unseen family and loop-intensity regimes, not spread across all benign
executions. For diagonal density's `memfd`-5000 group, 2.472 of its 2.521 mean
score comes from PT features, versus 0.042 from PEBS and 0.002 from PMU; byte
composition dominates the failure. Training-family conditional prototypes do
not fix this transfer problem. A future unsupervised cluster-balanced top-$k$
review cap may reduce duplicate review load, but must be reported as retrospective
budgeting rather than prospective sensitivity.

### Exact matched-tail replay of the archived models

The prior R1 capacity report used a **1%** calibration budget, so its alert
counts cannot be compared directly with the table above. I rescored the
already-frozen d128/d512 fused and span-only checkpoints, plus their frozen
marginal and PT-only PCA baselines, on exactly the same benign calibration and
validation rows. The weights did not change; only the order-statistic threshold
was recomputed at each benign review budget. The corrected canary was not read.

| Archived frozen score | Familiar at $10^{-3}$ | Whole-family at $10^{-3}$ | Familiar at $10^{-4}$ | Whole-family at $10^{-4}$ |
| --- | ---: | ---: | ---: | ---: |
| d128 fused | 3/7,000 | 296/18,000 | 0/7,000 | 3/18,000 |
| d512 fused | 4/7,000 | 203/18,000 | 0/7,000 | 8/18,000 |
| d128 span-only | 4/7,000 | 40/18,000 | 0/7,000 | 1/18,000 |
| d512 span-only | 8/7,000 | 318/18,000 | 0/7,000 | 6/18,000 |
| Marginal | 7/7,000 | 3/18,000 | 0/7,000 | 2/18,000 |
| PT-only PCA | 4/7,000 | 7,806/18,000 | 0/7,000 | 7,595/18,000 |

The old marginal score remains strongest on this particular benign transfer
test; it did **not** demonstrate vulnerability sensitivity in the earlier
corrected canary. Lowering a threshold budget can make an unseen-family alert
count look small, but two calibration-tail observations cannot certify the
$10^{-4}$ regime, and benign specificity alone is not anomaly usefulness.
The denoising, low-rank, diagonal, and nearest-prototype R1 scores fail even
relative to the archived d128/d512 fused controls. More generic density
capacity is not the next justified spend.

## Named-host throughput and scope

On `mac.local` (Apple ARM64, 24 GiB RAM, Torch 2.13 MPS), SHA-verified compact
extraction of all 102,000 derived rows took 81.99 s, or 1,244 executions/s.
The MLP's 500 steps at batch 512 took 3.30 s in the first run (77,520 sampled
training examples/s); a warmed final repeat took 2.57 s (99,628/s). The final
pre-materialized MPS scorer rates were 712,116/s for the MLP, 1,837,345/s for
low-rank density, 8,497,734/s for diagonal density, and 1,547,918/s for nearest
prototype over 46,000 calibration/validation rows. These rates exclude file
loads, hash checks, tensor extraction, CPU-to-MPS transfer, and capture. The
first cold MLP scorer pass was 137,098/s, so warmed component rates must not be
treated as cold-start guarantees. Neither rate is end-to-end capture-to-score
throughput. The verified extraction stage is ahead of this physical host's
13.07/s finite collector but remains far below the aspirational aggregate
100,000/s target. No paid GPU was launched: this evidence points to
generalization and extraction, not MPS optimization, as the current bottlenecks.

The matched-tail replay itself took 1,036.15 s on `mac.local` and rescored
46,000 benign calibration/validation executions with all six frozen models.
The d128 fused and d512 fused component rates were 1,205 and 103 executions/s
respectively; both exclude capture/feature extraction. The result file is
`prior-rescore.json` (SHA-256
`37e093b2df040a6eccec0f2b7644c5813d25600dc0fbad822e1cc4dd35dca412`).
The final new-model report SHA-256 is
`74dbadc70e50f60876e00d9ce93a47e7d7297c991928d577da2054de227ab3cd`;
the checkpoint SHA-256 is
`2e2851dd6b87a0d79f7c41c0b52188730e37c95eae15775727cdc2f6bd564e2f`.
Its CPU reload reproduced 128 calibration scores bit-for-bit. The compact-cache
SHA-256 is
`5c689576cfd8d1576aa6f063e598a02cfee21d0fa629141e4c9913a8fda1fcdc`.

The source is based on Git `4f34880` plus new R1 scripts and test; all artifacts
are under `/Users/theoad/.cache/cpu2tensor/hardware-anomaly-model-r1`.
`report.json` contains exact thresholds, scores, and family-by-loop counts;
`offline-model-r1.pt` contains the frozen compact model and density parameters;
`compact-features.pt` is a reproducible derived cache. The experiment commands
are:

```bash
/Users/theoad/.cache/cpu2tensor/dev-python/bin/python \
  python/cpu2tensor/examples/hardware_anomaly_model_r1.py \
  /Users/theoad/.cache/cpu2tensor/hardware-pretraining-go/store-v3-scale100k-r2 \
  /Users/theoad/.cache/cpu2tensor/hardware-anomaly-model-r1 \
  --steps 500 --batch-size 512
```

The prior replay command used the same corpus, then the archived
`control-d128-step500` and `large-d512-step500` checkpoint directories and
`python -m cpu2tensor.examples.hardware_anomaly_prior_r1` to write
`prior-rescore.json`. It used a host-local Python environment with Torch 2.13;
that environment has a stale editable package finder, so execution explicitly
put this checkout's `python/` source root first without changing repository
imports or runtime code.

## What this compact model cannot condition on

The 789 features retain PT byte-frequency moments, PEBS feature-channel
moments, aggregate PMU values, and one PEBS availability fraction. They drop
PT packet order and branch transitions; original AUX byte offsets; 16-segment
sequence positions; per-sample PEBS IP/address/weight/data-source identities;
the precise-store event identity; per-lane CPU/TID; token availability masks;
per-token timing bounds and timing-quality estimates; and cross-CPU timing
relations. The source PEBS featurizer has relative-IP/address sketches that
survive uniform relocation, but averaging them again loses rare sites and
does not establish KASLR robustness. The compact score also does not see
manifest boot/session identity, KASLR slide, workload family, seed, repetition,
loop count, PT byte count, elapsed time, or the exact perf enable/running
ratio. These are retained in the manifest/raw custody where available, but
not passed as model inputs. The single exact boot and fixed CPU policy in this
corpus cannot test cross-boot, cross-seed, CPU-topology, or KASLR invariance.

The highest-value **next conditioning ablation on this immutable corpus** is
training-only exposure conditioning: normalize or residualize the PT byte
moments against observed log PT bytes and elapsed time (and PEBS count against
its sampling exposure), freeze that map and benign threshold, then repeat the
same whole-family and loop-intensity holdouts. These hardware-observable
conditioning variables do not reveal held-out family or canary labels. It
directly tests the identified high-intensity PT failure without retroactively
changing this frozen score. It would still be a compressed-byte control, not
execution grammar. A separate grammar path should start from raw PT bytes,
decoding only as far as measured throughput permits: first test lightweight raw
packet/boundary tokens with exact-boot relative IP where available, while
preserving PEBS sample identities/timestamps and per-lane sequence. Full branch
decode belongs in sampled audit/LLM evidence unless a production-rate decode
path is demonstrated. The current PT configuration did not retain timing
packets, so this archive cannot establish fine PT↔PEBS alignment. New boots,
seeds, and KASLR epochs require independent held-out tests. Only 0.5% of
benign raw PT was retained here; that custody constraint limits what grammar
pretraining can claim from this archive alone.

Only a *model-component* speed claim is supported on this Mac. The
physical-host capture rate is separately named and is not compared as an
ISA-matched performance benchmark. The 24-hour pretraining campaign remains
**NO-GO** pending a better representation, independent calibration, and a
validated prospective review budget.
