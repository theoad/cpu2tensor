# One-GPU hardware-anomaly autoresearch

Status: design and integration in progress. This adopts the single-editable-file,
fixed-five-minute experiment and keep/discard discipline of
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch). It does not
make the current detector operational or replace a real-effect validation gate.

## One producer, one learner

The physical `trail-x86` laptop is the only perf collector. It produces
immutable, hash-verified PT/PEBS/PMU shards in separate sessions; the Mac MPS
GPU is the only active training device. Collection is **not** repeated for every
model trial. "Bugless collection" means a trustworthy measurement channel,
not a corpus restricted to benign executions: lawful and effect-bearing inputs
are both needed. Before reuse, a dataset must record the exact boot, build, CPU and
frequency policy, perf event/period and running time, actual per-execution
input bytes/hash/seed, session, raw/derived hashes, and explicit loss or
censorship. Reject rather than silently train on incomplete rows. The collection
GO gate is measured complete-execution yield, raw/derived integrity, PT/PEBS/PMU
loss and multiplex rates, matched-repeat score/feature variability, family and
intensity coverage, ambient/thermal drift, and completed executions per second
including custody. These figures must be reported per session; a zero-error
aggregate cannot hide a noisy subgroup. The old
102k corpus has zero failed executions but only one boot, no true per-execution
input seed, and 13.07 complete executions/s; it is a development substrate,
not proof of seed/KASLR robustness. With 6.0 GiB free on the collector, any
raw-retention expansion needs off-host custody and a hard disk stop first.

## One mutable experiment file

`python/cpu2tensor/examples/hardware_autoresearch_train.py` alone owns the
architecture, objective, optimizer, batching, frozen scorer, and all model
hyperparameters. Data preparation, split identities, evaluation, budget and
results ledger live outside it and are immutable during a run. One Mac GPU
process has an exclusive lock. Each trial gets at most 300 seconds **total**:
target 240 seconds for training and no more than 60 for reload/evaluation,
including any compilation and data-transfer overhead. A timeout or incomplete
score is `crash`, never a candidate win. The runner records source, dataset,
split, checkpoint and report hashes, wall time, peak memory, and host.

After every trial the coordinator compares the locked score with the current
champion. Keep only a reproducible improvement; otherwise restore the champion
file. Compare seeds on the *same* immutable data before accepting a small
delta. Record `keep`, `discard`, or `crash` in an append-only ledger. An agent
may propose a new edit to the single file between trials, but no second GPU
trial may overlap it. The existing 30-minute heartbeat reviews the ledger and
resource health; it does not spend Codex usage every five minutes merely to
poll an unchanged run.

## Validation must be locked before automatic keep/discard

The model cannot win by giving every execution the same score or by flooding
an unfamiliar lawful family. Development evaluation must combine: (1) a
separate-session, seed-varied benign calibration and test, reported per family
and intensity; (2) a predeclared, matched physical effect challenge whose
trace difference is observable in kernel-only PT/PEBS/PMU; and (3) capture
completeness and score throughput. If the effect is a lawful intervention or
offline cross-modal mismatch, it is a **sensor/representation proxy**, not bug
detection. Neither Dirty Pipe nor a synthetic proxy may be optimized and then
reused as an independent vulnerability claim.

Until a safe exact-package real-effect gate exists, the five-minute loop may
be used only for development. Its provisional keep/discard metric and hard
constraints must be frozen on separate data before the first autonomous trial;
the independent real-bug set remains sealed. The operational gate is still
recall in the top 100 of one million executions at benign FPR at most `1e-4`,
with independent calibration and sessions. The current 21k calibration tail
has about two observations at that rate and cannot certify it. A model that
improves training loss or development proxy but fails that gate is NO-GO for
the large campaign.

## Two report-only trials on the old one-session corpus

Two fixed-seed, report-only trials used the same 56k/21k/7k/18k split and
shared compact source SHA-256
`5c689576cfd8d1576aa6f063e598a02cfee21d0fa629141e4c9913a8fda1fcdc`
and manifest SHA-256
`0cf77ff5c64106598e20873cede98401fd7293ab6a59b295b15633389616b37d`.
The v1 compact autoencoder and the v2 train-only PT-byte/elapsed-time/PEBS
exposure-conditioned autoencoder each trained seeds 1729 and 1730 once on
`mac.local` ARM64 MPS. The 300-second-capped runner finished in 38.19 and
39.00 seconds, respectively, with `report_only` outcomes and no champion
mutation. The v2 model retained 438,101 parameters; its two 12k-step training
passes took 17.16 and 17.02 seconds. CPU evaluator scoring, including v2
conditioning, was 203k and 202k rows/s; this is not capture or end-to-end
pipeline throughput.

| Calibration rule and benign alerts | v1 seeds 1729 / 1730 | v2 seeds 1729 / 1730 |
| --- | ---: | ---: |
| Strict $10^{-3}$: calibration / 21k | 21 / 21 | 21 / 21 |
| Strict $10^{-3}$: familiar / 7k | 5 / 4 | 7 / 5 |
| Strict $10^{-3}$: whole-family / 18k | 12,015 / 12,018 | 12,042 / 12,036 |
| Exploratory $10^{-4}$: calibration / 21k | 2 / 2 | 2 / 2 |
| Exploratory $10^{-4}$: familiar / 7k | 0 / 0 | 0 / 0 |
| Exploratory $10^{-4}$: whole-family / 18k | 10,463 / 10,100 | 11,500 / 12,000 |

The v2 autoencoder is a negative result: exposure correction did not control
the unseen-family flood, despite the separate diagonal-density R2 exposure
ablation doing so. At exploratory $10^{-4}$, v2 alerts include all 6,000
`memfd` rows for both seeds and 5,500/6,000 `dup` rows. Only about two
calibration tail points exist at that rate. These are benign-development
readouts, not effect recall, an operational FPR certificate, or a keep/discard
decision. No canary was used; further GPU trials on this old corpus stop here.

The exact custody hashes for those trial artifacts are:

| SHA-256 artifact | v1 | v2 |
| --- | --- | --- |
| Trainer source snapshot | `49f0dd20557fd4f2fedd6e8d0ce28b7737e6d012729c0f0ff0436275e9bcc657` | `0bfc8a25c9aee09117abe3fd6dff11681a8eb3d548d822772f1e8e629efa6232` |
| Train-only cache | `b212a68a879a665075b07d1a1b3fe9e6cc56daa7b552a1064248936d057a9643` | `f949c7dd6df006a5cf84e2da59950e3ddad8fd0a31f9e78573814bd798b04292` |
| Development-eval cache | `378bbdc43adae83fbc539e50a7d03e04818eb0bff88e8b8112c588809b54b3a8` | `9b875bab80db03afcc6a612bba6216c0e42af2fa6c8063cc51ae63ade4b40d26` |
| Seed 1729 checkpoint | `b44cd086c1c7f1b5fafb54c0968026d6785638f70c836be069367ef17d128f0c` | `abf3d8af55927292d57a1416f0a4e289ba1d1eae2411a889d9ad736f93a6ffe0` |
| Seed 1730 checkpoint | `aaed68240b843d2d437af898ddae4400b04bef18627752fa061aef758a8495c8` | `543ff4f97a3b0ebc30ee20ef451fda7d8a01c1dc5d4fa4b63f0293fffe46fc60` |
| Evaluator source | `4fa8ac4d24f131282f7ff93838794156973206c18fae92cb0322aadf490ab65d` | `f01b96fb652462b5069761af93df270941d4e113b75a539479b18303e0793600` |
| Evaluation report | `47b3f419b158f8dec11a6fb02c66a9114489cf6938cd3b7e7d5b8268c1ea2334` | `eea86b28d26c599657dc85dc8a3469e6e2d742992592ce3c220b1059d6d9fc6d` |
| Row scores | `0d4e3db29c9c60dde9c5973d95dfabf9b50a0b8db7c8ef2ac7709283f4ef9f3c` | `1079a870b75ccf045796d4a8f6a58345e360e4a8764de4abbe578d5e7a59401a` |

The preserved trial directories are
`/Users/theoad/.cache/cpu2tensor/hardware-autoresearch-r1/runner-smoke-20260926/20260926T011225Z-30a510ae`
and
`/Users/theoad/.cache/cpu2tensor/hardware-autoresearch-v2-exposure/runner-smoke-20260926/20260926T012239Z-ec8ad270`.
Each contains the source snapshot, both checkpoints, append-only ledger,
scores, evaluation report, and `outcome.json`. Model work resumes only after
independently qualified capture and a locked separate-session effect/control
metric.

## Immediate sequence

1. The opt-in deterministic input-seed contract and one bounded, exclusive
   runner trial are implemented and tested; neither has authorized a live
   laptop capture or automatic model promotion.
2. Finish and independently test the physical capture supervisor, exact
   input/raw custody, policy restoration, and off-host acknowledgement. Then
   review a 51-row smoke before the two planned seeded sessions.
3. Freeze a separate-session development metric with matched nuisance
   controls. The read-only exact-package audit found no currently safe known
   bug trigger for the laptop; a proxy can guide development but cannot open
   the real-bug gate.
4. Start a five-minute exclusive GPU loop only if that locked metric and data
   pass. Stop at the sprint deadline or any resource/integrity fault; promote
   no model merely because training loss or proxy specificity improves.
